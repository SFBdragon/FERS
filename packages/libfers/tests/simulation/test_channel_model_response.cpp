// Tests for simulation::calculateResponses. Verifies co-location skipping,
// and that direct/reflected-path responses are generated and routed to the
// correct receiver queue (pulsed inbox vs. streaming-mode interference log).

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <memory>
#include <vector>

#include "antenna/antenna_factory.h"
#include "core/config.h"
#include "core/parameters.h"
#include "core/world.h"
#include "math/coord.h"
#include "math/geometry_ops.h"
#include "propagation/pointscatter/pointscatter.h"
#include "radar/platform.h"
#include "radar/receiver.h"
#include "radar/target.h"
#include "radar/transmitter.h"
#include "serial/response.h"
#include "signal/radar_signal.h"
#include "simulation/channel_model.h"
#include "timing/timing.h"

using Catch::Matchers::WithinAbs;
using Catch::Matchers::WithinRel;

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

	void setupPlatform(radar::Platform& plat, const math::Vec3& pos)
	{
		plat.getMotionPath()->addCoord(math::Coord{pos, 0.0});
		plat.getMotionPath()->finalize();
		plat.getRotationPath()->addCoord(math::RotationCoord{0.0, 0.0, 0.0});
		plat.getRotationPath()->finalize();
	}

	// calculateResponses requires a Response's underlying RadarSignal to be
	// isSampled() (it throws otherwise), so every transmitter here carries a
	// real SampledSignal rather than a CwSignal or a mock.
	fers_signal::RadarSignal makeSampledWave(const std::string& name, const RealType power, const RealType carrier,
											 std::vector<ComplexType> samples, const RealType rate)
	{
		fers_signal::SampledSignal sampled;
		sampled.load(samples, static_cast<unsigned>(samples.size()), rate);
		return {name, power, carrier, std::move(sampled)};
	}
}

// =============================================================================
// calculateResponses: co-location skipping produces no responses
// =============================================================================

TEST_CASE("calculateResponses produces no responses when Tx and Rx share the same platform (direct path)",
		  "[simulation][channel_model][response]")
{
	ParamGuard const guard;
	params::params.reset();
	params::setSimSamplingRate(1000.0);

	core::World world;

	auto shared_plat = std::make_unique<radar::Platform>("shared");
	setupPlatform(*shared_plat, math::Vec3{0.0, 0.0, 0.0});
	auto* shared_plat_ptr = shared_plat.get();

	antenna::Isotropic iso_ant("iso");
	auto timing = std::make_shared<timing::Timing>("clk", 42);

	auto tx = std::make_unique<radar::Transmitter>(shared_plat_ptr, "tx", radar::OperationMode::PULSED_MODE);
	tx->setAntenna(&iso_ant);
	tx->setTiming(timing);

	auto wave = makeSampledWave("sig", 1.0, 1e9, std::vector<ComplexType>{ComplexType{1.0, 0.0}, ComplexType{1.0, 0.0}},
								1000.0);
	tx->setSignal(&wave);

	auto rx = std::make_unique<radar::Receiver>(shared_plat_ptr, "rx", 42, radar::OperationMode::PULSED_MODE);
	rx->setAntenna(&iso_ant);
	rx->setTiming(timing);
	auto* rx_ptr = rx.get();
	auto* tx_ptr = tx.get();

	world.add(std::move(shared_plat));
	world.add(std::move(tx));
	world.add(std::move(rx));

	const propagation::pointscatter::PointScatterModel prop(&world);
	simulation::calculateResponses(*tx_ptr, prop, 0.0);

	REQUIRE(rx_ptr->drainInbox().empty());
}

TEST_CASE("calculateResponses produces no direct-path response for a co-located monostatic pair",
		  "[simulation][channel_model][response]")
{
	// Real monostatic Tx/Rx pairs (see parseMonostatic) are always parsed onto the
	// same platform, so co-location alone (not attachment) is what pointscatter's
	// directPath() uses to suppress self-interference -- exercise that realistic case.
	ParamGuard const guard;
	params::params.reset();
	params::setSimSamplingRate(1000.0);

	core::World world;

	auto plat = std::make_unique<radar::Platform>("mono_plat");
	setupPlatform(*plat, math::Vec3{0.0, 0.0, 0.0});
	auto* plat_ptr = plat.get();

	antenna::Isotropic iso_ant("iso");
	auto timing = std::make_shared<timing::Timing>("clk", 42);

	auto tx = std::make_unique<radar::Transmitter>(plat_ptr, "tx", radar::OperationMode::PULSED_MODE);
	tx->setAntenna(&iso_ant);
	tx->setTiming(timing);

	auto wave = makeSampledWave("sig", 1.0, 1e9, std::vector<ComplexType>{ComplexType{1.0, 0.0}, ComplexType{1.0, 0.0}},
								1000.0);
	tx->setSignal(&wave);

	auto rx = std::make_unique<radar::Receiver>(plat_ptr, "rx", 42, radar::OperationMode::PULSED_MODE);
	rx->setAntenna(&iso_ant);
	rx->setTiming(timing);
	auto* rx_ptr = rx.get();
	auto* tx_ptr = tx.get();

	tx_ptr->setAttached(rx_ptr);
	rx_ptr->setAttached(tx_ptr);

	world.add(std::move(plat));
	world.add(std::move(tx));
	world.add(std::move(rx));

	const propagation::pointscatter::PointScatterModel prop(&world);
	simulation::calculateResponses(*tx_ptr, prop, 0.0);

	REQUIRE(rx_ptr->drainInbox().empty());
}

TEST_CASE("calculateResponses produces no reflected-path response when target co-located with Tx",
		  "[simulation][channel_model][response]")
{
	ParamGuard const guard;
	params::params.reset();
	params::setSimSamplingRate(1000.0);

	core::World world;

	auto tx_plat = std::make_unique<radar::Platform>("tx_plat");
	setupPlatform(*tx_plat, math::Vec3{0.0, 0.0, 0.0});
	auto* tx_plat_ptr = tx_plat.get();

	auto rx_plat = std::make_unique<radar::Platform>("rx_plat");
	setupPlatform(*rx_plat, math::Vec3{1000.0, 0.0, 0.0});
	auto* rx_plat_ptr = rx_plat.get();

	antenna::Isotropic iso_ant("iso");
	auto timing = std::make_shared<timing::Timing>("clk", 42);

	auto tx = std::make_unique<radar::Transmitter>(tx_plat_ptr, "tx", radar::OperationMode::PULSED_MODE);
	tx->setAntenna(&iso_ant);
	tx->setTiming(timing);

	auto wave = makeSampledWave("sig", 1.0, 1e9, std::vector<ComplexType>{ComplexType{1.0, 0.0}, ComplexType{1.0, 0.0}},
								1000.0);
	tx->setSignal(&wave);

	auto rx = std::make_unique<radar::Receiver>(rx_plat_ptr, "rx", 42, radar::OperationMode::PULSED_MODE);
	rx->setAntenna(&iso_ant);
	rx->setTiming(timing);
	// Suppress the (co-location-independent) direct path so only the co-located
	// target's reflected-path skip is under test.
	rx->setFlag(radar::Receiver::RecvFlag::FLAG_NODIRECT);
	auto* rx_ptr = rx.get();
	auto* tx_ptr = tx.get();

	// Target on same platform as Tx
	auto tgt = radar::createIsoTarget(tx_plat_ptr, "tgt", 10.0, 42);

	world.add(std::move(tx_plat));
	world.add(std::move(rx_plat));
	world.add(std::move(tx));
	world.add(std::move(rx));
	world.add(std::move(tgt));

	const propagation::pointscatter::PointScatterModel prop(&world);
	simulation::calculateResponses(*tx_ptr, prop, 0.0);

	REQUIRE(rx_ptr->drainInbox().empty());
}

TEST_CASE("calculateResponses produces no reflected-path response when target co-located with Rx",
		  "[simulation][channel_model][response]")
{
	ParamGuard const guard;
	params::params.reset();
	params::setSimSamplingRate(1000.0);

	core::World world;

	auto tx_plat = std::make_unique<radar::Platform>("tx_plat");
	setupPlatform(*tx_plat, math::Vec3{0.0, 0.0, 0.0});
	auto* tx_plat_ptr = tx_plat.get();

	auto rx_plat = std::make_unique<radar::Platform>("rx_plat");
	setupPlatform(*rx_plat, math::Vec3{1000.0, 0.0, 0.0});
	auto* rx_plat_ptr = rx_plat.get();

	antenna::Isotropic iso_ant("iso");
	auto timing = std::make_shared<timing::Timing>("clk", 42);

	auto tx = std::make_unique<radar::Transmitter>(tx_plat_ptr, "tx", radar::OperationMode::PULSED_MODE);
	tx->setAntenna(&iso_ant);
	tx->setTiming(timing);

	auto wave = makeSampledWave("sig", 1.0, 1e9, std::vector<ComplexType>{ComplexType{1.0, 0.0}, ComplexType{1.0, 0.0}},
								1000.0);
	tx->setSignal(&wave);

	auto rx = std::make_unique<radar::Receiver>(rx_plat_ptr, "rx", 42, radar::OperationMode::PULSED_MODE);
	rx->setAntenna(&iso_ant);
	rx->setTiming(timing);
	rx->setFlag(radar::Receiver::RecvFlag::FLAG_NODIRECT);
	auto* rx_ptr = rx.get();
	auto* tx_ptr = tx.get();

	// Target on same platform as Rx
	auto tgt = radar::createIsoTarget(rx_plat_ptr, "tgt", 10.0, 42);

	world.add(std::move(tx_plat));
	world.add(std::move(rx_plat));
	world.add(std::move(tx));
	world.add(std::move(rx));
	world.add(std::move(tgt));

	const propagation::pointscatter::PointScatterModel prop(&world);
	simulation::calculateResponses(*tx_ptr, prop, 0.0);

	REQUIRE(rx_ptr->drainInbox().empty());
}

// =============================================================================
// calculateResponses: valid paths produce routed responses with correct timing
// =============================================================================

TEST_CASE("calculateResponses direct path produces a routed response with interp points",
		  "[simulation][channel_model][response]")
{
	ParamGuard const guard;
	params::params.reset();
	params::setSimSamplingRate(1000.0); // 1000 samples/sec

	core::World world;

	auto tx_plat = std::make_unique<radar::Platform>("tx_plat");
	setupPlatform(*tx_plat, math::Vec3{0.0, 0.0, 0.0});
	auto* tx_plat_ptr = tx_plat.get();

	auto rx_plat = std::make_unique<radar::Platform>("rx_plat");
	setupPlatform(*rx_plat, math::Vec3{1000.0, 0.0, 0.0});
	auto* rx_plat_ptr = rx_plat.get();

	antenna::Isotropic iso_ant("iso");
	auto timing = std::make_shared<timing::Timing>("clk", 42);

	auto tx = std::make_unique<radar::Transmitter>(tx_plat_ptr, "tx", radar::OperationMode::PULSED_MODE);
	tx->setAntenna(&iso_ant);
	tx->setTiming(timing);

	auto wave = makeSampledWave("sig", 1.0, 1e9, std::vector<ComplexType>{ComplexType{1.0, 0.0}, ComplexType{1.0, 0.0}},
								1000.0);
	tx->setSignal(&wave);

	auto rx = std::make_unique<radar::Receiver>(rx_plat_ptr, "rx", 42, radar::OperationMode::PULSED_MODE);
	rx->setAntenna(&iso_ant);
	rx->setTiming(timing);
	auto* rx_ptr = rx.get();
	auto* tx_ptr = tx.get();

	world.add(std::move(tx_plat));
	world.add(std::move(rx_plat));
	world.add(std::move(tx));
	world.add(std::move(rx));

	const propagation::pointscatter::PointScatterModel prop(&world);
	simulation::calculateResponses(*tx_ptr, prop, 0.0);

	auto inbox = rx_ptr->drainInbox();
	REQUIRE(inbox.size() == 1);

	// startTime() sits one control-point period before the true propagation delay
	// (Response always pads its real points by controlPointEdgePeriod() on each side),
	// so compare with a tolerance covering a couple of control-point periods rather
	// than an exact match.
	const RealType expected_delay = 1000.0 / params::c();
	const RealType edge = 1.0 / params::simSamplingRate();
	REQUIRE_THAT(inbox[0]->startTime(), WithinAbs(expected_delay, 3.0 * edge));
}

TEST_CASE("calculateResponses reflected path produces a routed response", "[simulation][channel_model][response]")
{
	ParamGuard const guard;
	params::params.reset();
	params::setSimSamplingRate(1000.0);

	core::World world;

	auto tx_plat = std::make_unique<radar::Platform>("tx_plat");
	setupPlatform(*tx_plat, math::Vec3{0.0, 0.0, 0.0});

	auto tgt_plat = std::make_unique<radar::Platform>("tgt_plat");
	setupPlatform(*tgt_plat, math::Vec3{500.0, 0.0, 0.0});
	auto* tgt_plat_ptr = tgt_plat.get();

	auto rx_plat = std::make_unique<radar::Platform>("rx_plat");
	setupPlatform(*rx_plat, math::Vec3{500.0, 500.0, 0.0});
	auto* rx_plat_ptr = rx_plat.get();

	antenna::Isotropic iso_ant("iso");
	auto timing = std::make_shared<timing::Timing>("clk", 42);

	auto tx = std::make_unique<radar::Transmitter>(tx_plat.get(), "tx", radar::OperationMode::PULSED_MODE);
	tx->setAntenna(&iso_ant);
	tx->setTiming(timing);

	auto wave = makeSampledWave("sig", 1.0, 1e9, std::vector<ComplexType>{ComplexType{1.0, 0.0}, ComplexType{1.0, 0.0}},
								1000.0);
	tx->setSignal(&wave);

	auto rx = std::make_unique<radar::Receiver>(rx_plat_ptr, "rx", 42, radar::OperationMode::PULSED_MODE);
	rx->setAntenna(&iso_ant);
	rx->setTiming(timing);
	// Isolate the reflected path from this geometry's (perfectly valid) direct path.
	rx->setFlag(radar::Receiver::RecvFlag::FLAG_NODIRECT);
	auto* rx_ptr = rx.get();
	auto* tx_ptr = tx.get();

	auto tgt = radar::createIsoTarget(tgt_plat_ptr, "tgt", 10.0, 42);

	world.add(std::move(tx_plat));
	world.add(std::move(rx_plat));
	world.add(std::move(tgt_plat));
	world.add(std::move(tx));
	world.add(std::move(rx));
	world.add(std::move(tgt));

	const propagation::pointscatter::PointScatterModel prop(&world);
	simulation::calculateResponses(*tx_ptr, prop, 0.0);

	auto inbox = rx_ptr->drainInbox();
	REQUIRE(inbox.size() == 1);

	const RealType r1 = 500.0;
	const RealType r2 = 500.0;
	const RealType expected_delay = (r1 + r2) / params::c();
	const RealType edge = 1.0 / params::simSamplingRate();
	REQUIRE_THAT(inbox[0]->startTime(), WithinAbs(expected_delay, 3.0 * edge));
}

TEST_CASE("calculateResponses direct path response spans approximately the signal duration",
		  "[simulation][channel_model][response]")
{
	ParamGuard const guard;
	params::params.reset();
	params::setSimSamplingRate(10000.0); // High sample rate for many points

	const RealType dist = 2000.0;
	const RealType carrier = 1.0e9;

	core::World world;

	auto tx_plat = std::make_unique<radar::Platform>("tx_plat");
	setupPlatform(*tx_plat, math::Vec3{0.0, 0.0, 0.0});

	auto rx_plat = std::make_unique<radar::Platform>("rx_plat");
	setupPlatform(*rx_plat, math::Vec3{dist, 0.0, 0.0});

	antenna::Isotropic iso_ant("iso");
	auto timing = std::make_shared<timing::Timing>("clk", 42);

	auto tx = std::make_unique<radar::Transmitter>(tx_plat.get(), "tx", radar::OperationMode::PULSED_MODE);
	tx->setAntenna(&iso_ant);
	tx->setTiming(timing);

	// 5e-3 s worth of samples at 10000 Hz -- matches the old TestSignal's 5ms duration.
	std::vector<ComplexType> samples(50, ComplexType{1.0, 0.0});
	auto wave = makeSampledWave("sig", 1.0, carrier, samples, 10000.0);
	tx->setSignal(&wave);

	auto rx = std::make_unique<radar::Receiver>(rx_plat.get(), "rx", 42, radar::OperationMode::PULSED_MODE);
	rx->setAntenna(&iso_ant);
	rx->setTiming(timing);
	auto* rx_ptr = rx.get();
	auto* tx_ptr = tx.get();

	world.add(std::move(tx_plat));
	world.add(std::move(rx_plat));
	world.add(std::move(tx));
	world.add(std::move(rx));

	const propagation::pointscatter::PointScatterModel prop(&world);
	simulation::calculateResponses(*tx_ptr, prop, 0.0);

	auto inbox = rx_ptr->drainInbox();
	REQUIRE(inbox.size() == 1);

	// Response length should be approximately the signal length (5ms), plus the lead/trail
	// control-point padding Response always adds at each end (~1 edge period apiece).
	const RealType edge = 1.0 / params::simSamplingRate();
	REQUIRE_THAT(inbox[0]->getRxDuration(), WithinAbs(5e-3, 4.0 * edge));
}
