#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <vector>

#include "core/config.h"
#include "core/parameters.h"
#include "interpolation/interpolation_point.h"
#include "serial/response.h"
#include "signal/radar_signal.h"

using Catch::Matchers::WithinAbs;

namespace
{
	struct ParamGuard
	{
		params::Parameters saved;
		ParamGuard() : saved(params::params) {}
		ParamGuard(const ParamGuard&) = delete;
		ParamGuard& operator=(const ParamGuard&) = delete;
		ParamGuard(ParamGuard&&) = delete;
		ParamGuard& operator=(ParamGuard&&) = delete;
		~ParamGuard() { params::params = saved; }
	};

	fers_signal::RadarSignal makeSampledRadarSignal(const std::string& name, const RealType power,
													std::vector<ComplexType> samples, const RealType rate)
	{
		fers_signal::SampledSignal sampled;
		sampled.load(samples, static_cast<unsigned>(samples.size()), rate);
		return {name, power, 1.0e9, std::move(sampled)};
	}
}

TEST_CASE("Response start/end time always pad a single real point by controlPointEdgePeriod on both sides",
		  "[serial][response]")
{
	ParamGuard const guard;
	params::setOversampleRatio(1);
	params::setSimSamplingRate(1000.0);

	const auto wave = makeSampledRadarSignal("wave", 1.0, {ComplexType{0.0, 0.0}}, 1000.0);
	const interp::InterpPoint first{.gain = 1.0, .rx_time = 0.5, .delay = 0.0, .phase_delay = 0.0};
	serial::Response const response(&wave, first);

	// Response no longer materializes a synthetic lead point at construction time (or
	// requires a finalise() call to add a trailing one) -- startTime()/endTime() simply
	// report the real point's range padded by one controlPointEdgePeriod() on each side,
	// computed fresh every call. This holds even for a single-sample response, which is
	// the common case once responses are built from ephemeral ray-traced paths rather
	// than always spanning a whole transmitted pulse.
	REQUIRE_THAT(response.startTime(), WithinAbs(0.5 - 1.0e-3, 1e-12));
	REQUIRE_THAT(response.endTime(), WithinAbs(0.5 + 1.0e-3, 1e-12));
	REQUIRE_THAT(response.getRxDuration(), WithinAbs(2.0e-3, 1e-12));
}

TEST_CASE("Response taper width tracks the actual local rx_time spacing between real points", "[serial][response]")
{
	ParamGuard const guard;
	params::setOversampleRatio(1);
	params::setSimSamplingRate(1000.0);

	const auto wave = makeSampledRadarSignal("wave", 1.0, {ComplexType{0.0, 0.0}}, 1000.0);
	serial::Response response(&wave, {.gain = 1.0, .rx_time = 0.0, .delay = 0.0, .phase_delay = 0.0});
	response.addInterpPoint({.gain = 1.0, .rx_time = 1.0, .delay = 0.0, .phase_delay = 0.0});

	// These two real points are spaced 1.0s apart, far more than controlPointEdgePeriod()
	// (1e-3 here) -- e.g. because they're the endpoints of a response that was split from
	// an adjacent one under heavy Doppler compression/stretching (a path_id change mid
	// ray-trace, say). The taper must extrapolate using that same 1.0s step, not the
	// nominal control-point width, or an adjacent response's own taper wouldn't meet this
	// one's real data where it should -- causing rendered energy to overlap or gap right
	// at the seam between them. No finalise() call is needed (or exists) any more.
	REQUIRE_THAT(response.startTime(), WithinAbs(-1.0, 1e-9));
	REQUIRE_THAT(response.endTime(), WithinAbs(2.0, 1e-9));
}

TEST_CASE("Response renderSlice reproduces the source waveform at zero delay", "[serial][response]")
{
	ParamGuard const guard;
	params::setOversampleRatio(1);
	params::setSimSamplingRate(1000.0);

	const std::vector<ComplexType> samples = {
		ComplexType{1.0, -1.0},
		ComplexType{0.5, 0.25},
		ComplexType{-0.25, 0.75},
		ComplexType{0.0, -0.5},
	};
	constexpr RealType rate = 1000.0;
	const auto wave = makeSampledRadarSignal("wave", 1.0, samples, rate);

	const interp::InterpPoint first{.gain = 1.0, .rx_time = 0.0, .delay = 0.0, .phase_delay = 0.0};
	serial::Response response(&wave, first);
	// One real control point per native sample (matching how the point-scatter model
	// covers a whole pulse) keeps gain == 1 across the entire buffer under test; a
	// single real point would only hold full gain exactly at that one point; the very
	// next native sample already falls inside the response's own trailing taper.
	for (std::size_t i = 1; i < samples.size(); ++i)
	{
		response.addInterpPoint(
			{.gain = 1.0, .rx_time = static_cast<RealType>(i) / rate, .delay = 0.0, .phase_delay = 0.0});
	}

	// At exactly zero fractional delay the render filter is an exact identity (its
	// center tap is a bit-exact 1.0; every other tap is sinc(integer) ~ 0), so with
	// delay == 0.0 throughout, rendered samples must reproduce the loaded ones
	// verbatim. Native sample 0 lines up with `first` itself (the true first real
	// point), not with the synthetic taper -- the taper is a padded, separate concern
	// (see the startTime()/endTime() tests) that doesn't shift the native buffer.
	const auto data = response.renderSlice(rate, first.rx_time, samples.size(), 0.0);

	REQUIRE(data.size() == samples.size());
	for (std::size_t i = 0; i < samples.size(); ++i)
	{
		REQUIRE_THAT(data[i].real(), WithinAbs(samples[i].real(), 1e-9));
		REQUIRE_THAT(data[i].imag(), WithinAbs(samples[i].imag(), 1e-9));
	}
}
