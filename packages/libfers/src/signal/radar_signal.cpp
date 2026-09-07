// SPDX-License-Identifier: GPL-2.0-only
//
// Copyright (c) 2006-2008 Marc Brooker and Michael Inggs
// Copyright (c) 2008-present FERS Contributors (see AUTHORS.md).
//
// See the GNU GPLv2 LICENSE file in the FERS project root for more information.

/**
 * @file radar_signal.cpp
 * @brief Classes for handling radar waveforms and signals.
 */

#include "radar_signal.h"

#include <algorithm>
#include <cmath>
#include <complex>
#include <limits>
#include <stdexcept>
#include <utility>
#include <variant>

#include "core/config.h"
#include "core/parameters.h"
#include "dsp_filters.h"
#include "interpolation/interpolation_point.h"

namespace fers_signal
{
	std::string_view fmcwChirpDirectionToken(const FmcwChirpDirection direction) noexcept
	{
		return direction == FmcwChirpDirection::Down ? "down" : "up";
	}

	FmcwChirpDirection parseFmcwChirpDirection(const std::string_view direction)
	{
		if (direction == "up")
		{
			return FmcwChirpDirection::Up;
		}
		if (direction == "down")
		{
			return FmcwChirpDirection::Down;
		}
		throw std::runtime_error("Unsupported FMCW chirp direction '" + std::string(direction) + "'.");
	}

	void SampledSignal::clear() noexcept
	{
		_data.clear();
		_rate = 0;
	}

	void SampledSignal::load(std::span<const ComplexType> inData, const unsigned samples, const RealType sampleRate)
	{
		clear();
		const unsigned ratio = params::oversampleRatio();
		const auto oversampled_samples = static_cast<std::size_t>(samples) * static_cast<std::size_t>(ratio);
		if (oversampled_samples > std::numeric_limits<unsigned>::max())
		{
			throw std::overflow_error("Oversampled signal sample count exceeds unsigned range");
		}
		_data.resize(oversampled_samples);
		_rate = sampleRate * static_cast<RealType>(ratio);

		if (ratio == 1)
		{
			std::ranges::copy(inData, _data.begin());
		}
		else
		{
			upsample(inData, samples, _data);
		}
	}


	std::tuple<RealType, RealType, RealType, long>
	SampledSignal::calculateWeightsAndDelays(const interp::InterpPoint& iter, const interp::InterpPoint& next,
											 const RealType sampleTime, const RealType idelay,
											 const RealType fracWinDelay, const RealType amplitudeScale) const noexcept
	{
		const RealType bw =
			iter.rx_time < next.rx_time ? (sampleTime - iter.rx_time) / (next.rx_time - iter.rx_time) : 0.0;

		const RealType amplitude = amplitudeScale * std::lerp(std::sqrt(iter.gain), std::sqrt(next.gain), bw);
		const RealType phase = std::lerp(iter.phase_delay, next.phase_delay, bw);
		RealType fdelay = -(std::lerp(iter.delay, next.delay, bw) * _rate - idelay + fracWinDelay);
		const RealType int_part = std::floor(fdelay);
		fdelay -= int_part;

		const long i_sample_unwrap = static_cast<long>(int_part);

		return {amplitude, phase, fdelay, i_sample_unwrap};
	}

	ComplexType SampledSignal::performConvolution(const long i, const RealType* filt, const long filtLength,
												  const RealType amplitude, const long iSampleUnwrap) const noexcept
	{
		const long start = std::max(-filtLength / 2, -i);
		const long end = std::min(filtLength / 2, static_cast<int>(getSampleCount()) - i);

		ComplexType accum(0.0, 0.0);

		for (long j = start; j < end; ++j)
		{
			const long sample_idx = i + j + iSampleUnwrap;
			const long filt_idx = j + filtLength / 2;
			if (sample_idx >= 0 && sample_idx < static_cast<long>(getSampleCount()) && filt_idx >= 0 &&
				filt_idx < filtLength)
			{
				accum += amplitude * _data[static_cast<std::size_t>(sample_idx)] * filt[filt_idx];
			}
		}

		return accum;
	}

	SteppedFrequencySignal::SteppedFrequencySignal(const RealType start_frequency_offset, const RealType step_size,
												   const std::size_t step_count, const RealType dwell_time,
												   const RealType step_period, std::optional<std::size_t> sweep_count) :
		_start_frequency_offset(start_frequency_offset), _step_size(step_size), _step_count(step_count),
		_dwell_time(dwell_time), _step_period(step_period), _sweep_count(sweep_count)
	{
	}

	RealType SteppedFrequencySignal::firstFrequency(const RealType carrier_frequency) const noexcept
	{
		return carrier_frequency + _start_frequency_offset;
	}

	RealType SteppedFrequencySignal::lastFrequency(const RealType carrier_frequency) const noexcept
	{
		if (_step_count == 0)
		{
			return firstFrequency(carrier_frequency);
		}
		return firstFrequency(carrier_frequency) + static_cast<RealType>(_step_count - 1U) * _step_size;
	}

	RealType SteppedFrequencySignal::frequencySpan() const noexcept
	{
		if (_step_count == 0)
		{
			return 0.0;
		}
		return std::abs(static_cast<RealType>(_step_count - 1U) * _step_size);
	}

	RealType SteppedFrequencySignal::effectiveBandwidth() const noexcept
	{
		return static_cast<RealType>(_step_count) * std::abs(_step_size);
	}

	std::optional<RealType> SteppedFrequencySignal::totalDuration() const noexcept
	{
		if (!_sweep_count.has_value())
		{
			return std::nullopt;
		}
		return static_cast<RealType>(*_sweep_count) * getSweepPeriod();
	}

	std::optional<SteppedFrequencySignal::StepState>
	SteppedFrequencySignal::activeStepAt(const RealType time_since_segment_start,
										 const RealType carrier_frequency) const noexcept
	{
		if (time_since_segment_start < 0.0 || _step_count == 0 || _step_period <= 0.0 || _dwell_time <= 0.0)
		{
			return std::nullopt;
		}

		const RealType sweep_period = getSweepPeriod();
		const auto sweep_index = static_cast<std::size_t>(std::floor(time_since_segment_start / sweep_period));
		if (_sweep_count.has_value() && sweep_index >= *_sweep_count)
		{
			return std::nullopt;
		}

		const RealType local_sweep_time = time_since_segment_start - static_cast<RealType>(sweep_index) * sweep_period;
		const auto step_index = static_cast<std::size_t>(std::floor(local_sweep_time / _step_period));
		if (step_index >= _step_count)
		{
			return std::nullopt;
		}

		const RealType step_start_time =
			static_cast<RealType>(sweep_index) * sweep_period + static_cast<RealType>(step_index) * _step_period;
		const RealType local_step_time = time_since_segment_start - step_start_time;
		if (local_step_time < 0.0 || local_step_time >= _dwell_time)
		{
			return std::nullopt;
		}

		return StepState{.step_index = step_index,
						 .sweep_index = sweep_index,
						 .step_start_time = step_start_time,
						 .dwell_end_time = step_start_time + _dwell_time,
						 .step_end_time = step_start_time + _step_period,
						 .rf_frequency =
							 firstFrequency(carrier_frequency) + static_cast<RealType>(step_index) * _step_size};
	}

	FmcwChirpSignal::FmcwChirpSignal(const RealType chirp_bandwidth, const RealType chirp_duration,
									 const RealType chirp_period, const RealType start_frequency_offset,
									 std::optional<std::size_t> chirp_count, const FmcwChirpDirection direction) :
		_chirp_bandwidth(chirp_bandwidth), _chirp_duration(chirp_duration), _chirp_period(chirp_period),
		_start_frequency_offset(start_frequency_offset), _chirp_count(chirp_count),
		_chirp_rate(chirp_bandwidth / chirp_duration), _direction(direction)
	{
	}

	std::optional<std::size_t>
	FmcwChirpSignal::activeChirpIndexAt(const RealType time_since_segment_start) const noexcept
	{
		if (time_since_segment_start < 0.0)
		{
			return std::nullopt;
		}

		const auto chirp_index = static_cast<std::size_t>(std::floor(time_since_segment_start / _chirp_period));
		if (_chirp_count.has_value() && chirp_index >= *_chirp_count)
		{
			return std::nullopt;
		}

		const RealType chirp_time = time_since_segment_start - static_cast<RealType>(chirp_index) * _chirp_period;
		// Exact arithmetic cannot make this negative, but floating-point boundary rounding can.
		if (chirp_time < 0.0 || chirp_time >= _chirp_duration)
		{
			return std::nullopt;
		}

		return chirp_index;
	}

	std::optional<RealType>
	FmcwChirpSignal::instantaneousBasebandPhase(const RealType time_since_segment_start) const noexcept
	{
		const auto chirp_index = activeChirpIndexAt(time_since_segment_start);
		if (!chirp_index.has_value())
		{
			return std::nullopt;
		}

		const RealType chirp_time = time_since_segment_start - static_cast<RealType>(*chirp_index) * _chirp_period;
		return basebandPhaseForChirpTime(chirp_time);
	}

	RealType FmcwChirpSignal::basebandPhaseForChirpTime(const RealType chirp_time) const noexcept
	{
		return 2.0 * PI * _start_frequency_offset * chirp_time + PI * getSignedChirpRate() * chirp_time * chirp_time;
	}

	FmcwTriangleSignal::FmcwTriangleSignal(const RealType chirp_bandwidth, const RealType chirp_duration,
										   const RealType start_frequency_offset,
										   std::optional<std::size_t> triangle_count) :
		_chirp_bandwidth(chirp_bandwidth), _chirp_duration(chirp_duration),
		_start_frequency_offset(start_frequency_offset), _triangle_count(triangle_count),
		_chirp_rate(chirp_bandwidth / chirp_duration), _triangle_period(2.0 * chirp_duration),
		_delta_phi_up(2.0 * PI * start_frequency_offset * chirp_duration +
					  PI * _chirp_rate * chirp_duration * chirp_duration)
	{
	}

	RealType FmcwTriangleSignal::basebandPhaseForTriangleTime(const RealType triangle_time) const noexcept
	{
		if (triangle_time <= 0.0)
		{
			return 0.0;
		}

		const auto triangle_index = static_cast<std::size_t>(std::floor(triangle_time / _triangle_period));
		const RealType local_triangle_time = triangle_time - static_cast<RealType>(triangle_index) * _triangle_period;
		const bool down_leg = local_triangle_time >= _chirp_duration;
		const RealType u = down_leg ? local_triangle_time - _chirp_duration : local_triangle_time;
		const RealType phi_base =
			static_cast<RealType>(triangle_index) * 2.0 * _delta_phi_up + (down_leg ? _delta_phi_up : 0.0);
		if (!down_leg)
		{
			return phi_base + 2.0 * PI * _start_frequency_offset * u + PI * _chirp_rate * u * u;
		}
		return phi_base + 2.0 * PI * (_start_frequency_offset + _chirp_bandwidth) * u - PI * _chirp_rate * u * u;
	}

	std::optional<RealType>
	FmcwTriangleSignal::instantaneousBasebandPhase(const RealType time_since_segment_start) const noexcept
	{
		if (time_since_segment_start < 0.0)
		{
			return std::nullopt;
		}

		const auto triangle_index = static_cast<std::size_t>(std::floor(time_since_segment_start / _triangle_period));
		if (_triangle_count.has_value() && triangle_index >= *_triangle_count)
		{
			return std::nullopt;
		}

		const RealType local_triangle_time =
			time_since_segment_start - static_cast<RealType>(triangle_index) * _triangle_period;
		if (local_triangle_time < 0.0 || local_triangle_time >= _triangle_period)
		{
			return std::nullopt;
		}
		return basebandPhaseForTriangleTime(time_since_segment_start);
	}

	RadarSignal::RadarSignal(std::string name, const RealType power, const RealType carrierfreq, Waveform wave,
							 const SimId id) :
		_name(std::move(name)), _id(id == 0 ? SimIdGenerator::instance().generateId(ObjectType::Waveform) : id),
		_power(power), _carrierfreq(carrierfreq), _wave(std::move(wave))
	{
	}

	bool RadarSignal::isSampled() const noexcept { return std::holds_alternative<SampledSignal>(_wave); }

	bool RadarSignal::isCw() const noexcept { return std::holds_alternative<CwSignal>(_wave); }

	bool RadarSignal::isFmcwChirp() const noexcept { return std::holds_alternative<FmcwChirpSignal>(_wave); }

	bool RadarSignal::isFmcwTriangle() const noexcept { return std::holds_alternative<FmcwTriangleSignal>(_wave); }

	bool RadarSignal::isFmcwFamily() const noexcept { return isFmcwChirp() || isFmcwTriangle(); }

	bool RadarSignal::isSteppedFrequency() const noexcept
	{
		return std::holds_alternative<SteppedFrequencySignal>(_wave);
	}

	const SampledSignal* RadarSignal::getSampledSignal() const noexcept { return std::get_if<SampledSignal>(&_wave); }

	const FmcwChirpSignal* RadarSignal::getFmcwChirpSignal() const noexcept
	{
		return std::get_if<FmcwChirpSignal>(&_wave);
	}

	const FmcwTriangleSignal* RadarSignal::getFmcwTriangleSignal() const noexcept
	{
		return std::get_if<FmcwTriangleSignal>(&_wave);
	}

	const SteppedFrequencySignal* RadarSignal::getSteppedFrequencySignal() const noexcept
	{
		return std::get_if<SteppedFrequencySignal>(&_wave);
	}

	const Waveform& RadarSignal::getWaveform() const noexcept { return _wave; }
}
