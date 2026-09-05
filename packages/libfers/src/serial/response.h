// SPDX-License-Identifier: GPL-2.0-only
//
// Copyright (c) 2006-2008 Marc Brooker and Michael Inggs
// Copyright (c) 2008-present FERS Contributors (see AUTHORS.md).
//
// See the GNU GPLv2 LICENSE file in the FERS project root for more information.

/**
 * @file response.h
 * @brief Classes for managing radar signal responses.
 */

#pragma once

#include <cstddef>
#include <stdexcept>
#include <vector>

#include "core/config.h"
#include "core/parameters.h"
#include "interpolation/interpolation_point.h"
#include "signal/radar_signal.h"

class XmlElement;

namespace fers_signal
{
	class RadarSignal;
}

namespace radar
{
	class Transmitter;
}

namespace serial
{
	RealType controlPointEdgePeriod() noexcept { return 1.0 / params::simSamplingRate(); }

	/**
	 * @class Response
	 * @brief Manages radar signal responses from a transmitter.
	 */
	class Response
	{
	public:
		/**
		 * @brief Constructor for the `Response` class.
		 *
		 * @param wave Pointer to the radar signal object.
		 * @param transmitter Pointer to the transmitter object.
		 */
		Response(const fers_signal::RadarSignal* signal, const interp::InterpPoint first_point) : _signal(signal)
		{
			if (!signal->isSampled())
			{
				throw std::logic_error(
					"Attempted to create Reponse with RadarSignal that didn't contain a SampledSignal");
			}

			_wave = signal->getSampledSignal();

			interp::InterpPoint lead = first_point;
			lead.rx_time -= controlPointEdgePeriod();
			lead.gain = 0.0;
			// TODO_SHAUN This currently assumes no Doppler at first.
			// We need another point to linearly project, but there may not even be one.
			// It's better than assuming a hard cutoff though.
			// Especially for 1-sample responses.
			addInterpPoint(lead);
			addInterpPoint(first_point);
		}

		~Response() = default;
		Response(const Response&) = delete;
		Response& operator=(const Response&) = delete;
		Response(Response&&) = delete;
		Response& operator=(Response&&) = delete;

		/**
		 * @brief Retrieves the start time of the response.
		 *
		 * @return Start time as a `RealType`. Returns 0.0 if no points are present.
		 */
		[[nodiscard]] RealType startTime() const noexcept { return _points.empty() ? 0.0 : _points.front().rx_time; }

		/**
		 * @brief Retrieves the end time of the response.
		 *
		 * @return End time as a `RealType`. Returns 0.0 if no points are present.
		 */
		[[nodiscard]] RealType endTime() const noexcept { return _points.empty() ? 0.0 : _points.back().rx_time; }

		/**
		 * @brief Adds an interpolation point to the response.
		 *
		 * @param point The interpolation point to be added.
		 * @throws std::logic_error If the new point has a time earlier than the last point.
		 */
		void addInterpPoint(const interp::InterpPoint& point);

		/**
		 * @brief Adds a final zero point for interpolation.
		 */
		void finishResponse()
		{
			// We know there are at least two samples.
			// The final point can linearly project most of the last two samples' values.
			auto prev_prev_point = _points.at(_points.size() - 2);
			auto prev_point = _points.back();
			prev_point.rx_time += prev_point.rx_time - prev_prev_point.rx_time;
			prev_point.delay += prev_point.delay - prev_prev_point.delay;
			prev_point.phase_delay += prev_prev_point.phase_delay;
			prev_point.gain = 0.0; // fade out
			addInterpPoint(prev_point);
		}

		/**
		 * @brief Renders the response in binary format.
		 *
		 * @param rate Output parameter for the signal rate.
		 * @param size Output parameter for the size of the binary data.
		 * @param fracWinDelay Delay factor applied during windowing.
		 * @return A vector of `ComplexType` representing the rendered signal with a sample rate of `sampleRate`.
		 */
		[[nodiscard]] std::vector<ComplexType> render(RealType fracWinDelay) const;

		/// Renders a bounded absolute-time response slice on the requested output grid.
		[[nodiscard]] std::vector<ComplexType> renderSlice(RealType outputRate, RealType outputStartTime,
														   std::size_t sampleCount, RealType fracWinDelay) const;

		/// Returns the waveform native sample rate.
		[[nodiscard]] RealType sampleRate() const noexcept;

		/// Returns the waveform native sample count.
		[[nodiscard]] size_t sampleCount() const noexcept;

		/**
		 * @brief Retrieves the receiving duration of the response.
		 *
		 * @return The receiving of the response, in seconds.
		 */
		[[nodiscard]] RealType getRxDuration() const noexcept { return endTime() - startTime(); }


	private:
		const fers_signal::RadarSignal* _signal; ///< Pointer to the radar signal object.
		const fers_signal::SampledSignal* _wave; ///< Pointer to the radar signal object's SampledSignal.
		std::vector<interp::InterpPoint> _points; ///< Vector of interpolation points.
	};
}
