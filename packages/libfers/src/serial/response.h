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
#include <iterator>
#include <stdexcept>
#include <vector>

#include "core/config.h"
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
		 * @param signal Pointer to the radar signal object.
		 * @param first_point The response's first sample.
		 * @param left_interval The tx-time sampling gap immediately before `first_point`
		 * (0 if `first_point` sits at the transmission's start).
		 * @param right_interval This response's tx-time sampling gap after `first_point`,
		 * used only if no further point is ever added.
		 */
		Response(const fers_signal::RadarSignal* signal, const interp::InterpPoint first_point,
				 const RealType left_interval, const RealType right_interval) :
			_signal(signal), _left_interval(left_interval), _right_interval(right_interval)
		{
			if (!signal->isPulsed())
			{
				throw std::logic_error(
					"Attempted to create Reponse with RadarSignal that didn't contain a PulseWaveform");
			}

			_wave = signal->getPulseWaveform();

			// This ensures Response never has zero points by construction.
			// A Response should never have no points.
			_points.push_back(first_point);
		}

		~Response() = default;
		Response(const Response&) = delete;
		Response& operator=(const Response&) = delete;
		Response(Response&&) = delete;
		Response& operator=(Response&&) = delete;

		/**
		 * @brief Retrieves the start time of the response, including its leading fade-in taper.
		 * Coincides with the first simulation sample's tx time when `left_interval` is 0 (the
		 * response's first point is at the transmission's start).
		 */
		[[nodiscard]] RealType startTime() const noexcept { return taperEdges().lead.rx_time; }

		/**
		 * @brief Retrieves the end time of the response, including its trailing fade-out taper.
		 * Coincides with the last simulation sample's tx time when `right_interval` is 0 (the
		 * response's last point is at the transmission's end).
		 */
		[[nodiscard]] RealType endTime() const noexcept { return taperEdges().tail.rx_time; }

		/**
		 * @brief Adds an interpolation point to the response.
		 *
		 * @param point The interpolation point to be added.
		 * @param right_interval This response's tx-time gap after `point`, used when no
		 * further point is detected and added.
		 * @throws std::logic_error If the new point has a time earlier than the last point.
		 */
		void addInterpPoint(const interp::InterpPoint& point, RealType right_interval);

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

		/**
		 * @brief Forward iterator over a Response's tapered point sequence:
		 * a synthetic zero-gain lead point, the real points, and a synthetic zero-gain tail point.
		 */
		class TaperedPointIterator
		{
		public:
			using iterator_category = std::forward_iterator_tag;
			using value_type = interp::InterpPoint;
			using difference_type = std::ptrdiff_t;
			using pointer = const interp::InterpPoint*;
			using reference = const interp::InterpPoint&;

			TaperedPointIterator() noexcept = default;

			[[nodiscard]] reference operator*() const noexcept
			{
				if (_index == 0)
				{
					return _lead;
				}
				if (_index == _real_count + 1)
				{
					return _tail;
				}
				return _real[_index - 1];
			}

			[[nodiscard]] pointer operator->() const noexcept { return &**this; }

			TaperedPointIterator& operator++() noexcept
			{
				++_index;
				return *this;
			}

			TaperedPointIterator operator++(int) noexcept
			{
				auto copy = *this;
				++*this;
				return copy;
			}

			[[nodiscard]] friend bool operator==(const TaperedPointIterator& lhs,
												 const TaperedPointIterator& rhs) noexcept
			{
				return lhs._index == rhs._index;
			}

		private:
			friend class Response;

			TaperedPointIterator(const interp::InterpPoint* real, const std::size_t real_count,
								 const interp::InterpPoint& lead, const interp::InterpPoint& tail,
								 const std::size_t index) noexcept :
				_real(real), _lead(lead), _tail(tail), _real_count(real_count), _index(index)
			{
			}

			const interp::InterpPoint* _real = nullptr;
			interp::InterpPoint _lead{};
			interp::InterpPoint _tail{};
			std::size_t _real_count = 0;
			std::size_t _index = 0;
		};

		/// A view of a Response's tapered point sequence (see TaperedPointIterator).
		class TaperedPoints
		{
		public:
			[[nodiscard]] TaperedPointIterator begin() const noexcept { return {_real, _real_count, _lead, _tail, 0}; }

			[[nodiscard]] TaperedPointIterator end() const noexcept
			{
				return {_real, _real_count, _lead, _tail, _real_count + 2};
			}

		private:
			friend class Response;

			TaperedPoints(const interp::InterpPoint* real, const std::size_t real_count,
						  const interp::InterpPoint& lead, const interp::InterpPoint& tail) noexcept :
				_real(real), _lead(lead), _tail(tail), _real_count(real_count)
			{
			}

			const interp::InterpPoint* _real;
			interp::InterpPoint _lead;
			interp::InterpPoint _tail;
			std::size_t _real_count;
		};

		/// Returns a view over this response's samples bracketed by synthetic taper points.
		[[nodiscard]] TaperedPoints taperedPoints() const
		{
			const auto [lead, tail] = taperEdges();
			return {_points.data(), _points.size(), lead, tail};
		}

	private:
		struct TaperEdges
		{
			interp::InterpPoint lead;
			interp::InterpPoint tail;
		};

		/**
		 * @brief Extrapolates a taper point by a Doppler-adjusted tx-time `interval` in seconds
		 * beyond `edge`, using the Doppler rate observed between `edge` and `neighbor`
		 * (the next interp point toward the response's interior).
		 */
		[[nodiscard]] static inline interp::InterpPoint extrapolateTaperPoint(const interp::InterpPoint& edge,
																			  const interp::InterpPoint& neighbor,
																			  const RealType interval,
																			  const bool forward)
		{
			interp::InterpPoint point = edge;

			// Because the response was not detected at the previous/next sample, we know the taper
			// point should have zero gain here.
			point.gain = 0.0;


			// `interval` is a tx-time duration.
			// The corresponding rx-time interval is affected by Doppler rate.
			// Doppler rate is estimated by the edge and neighbour's delays over their own
			// transmission gap. This rate is linearly extrapolated to determine the taper's
			// own delay and phase delay.
			const RealType tx_gap = (neighbor.rx_time - neighbor.delay) - (edge.rx_time - edge.delay);
			const RealType scale = interval / tx_gap;
			const RealType delay_delta = (neighbor.delay - edge.delay) * scale;
			const RealType phase_delta = (neighbor.phase_delay - edge.phase_delay) * scale;
			const RealType rx_width = interval + delay_delta;

			const RealType sign = forward ? 1.0 : -1.0;
			point.rx_time += sign * rx_width;
			point.delay += sign * delay_delta;
			point.phase_delay += sign * phase_delta;
			return point;
		}

		/**
		 * @brief Computes this response's synthetic zero-gain lead and tail points. (Triangular tapering.)
		 *
		 * `_left_interval`/`_right_interval` are the tx-time sampling intervals
		 * on either side at which no response was detected (response is thus goes to zero there).
		 * If the first or last point was the first or last sample of the transmission,
		 * then no extrapolation should occur and the intervals have been set to zero.
		 * Extrapolation methods is defined by `extrapolateTaperPoint()`.
		 */
		[[nodiscard]] TaperEdges taperEdges() const
		{
			const auto& front = _points.front();
			const auto& back = _points.back();

			interp::InterpPoint lead, tail;
			if (_points.size() > 1)
			{
				lead = extrapolateTaperPoint(front, _points[1], _left_interval, /*forward=*/false);
				tail = extrapolateTaperPoint(back, _points[_points.size() - 2], _right_interval, /*forward=*/true);
			}
			else
			{
				lead = front;
				lead.gain = 0.0;
				lead.rx_time -= _left_interval;

				tail = back;
				tail.gain = 0.0;
				tail.rx_time += _right_interval;
			}

			return {lead, tail};
		}

		const fers_signal::RadarSignal* _signal; ///< Pointer to the radar signal object.
		const fers_signal::PulseWaveform* _wave; ///< Pointer to the radar signal object's PulseWaveform.
		std::vector<interp::InterpPoint> _points; ///< Vector of simulated InterpPoints.
		RealType _left_interval; ///< True tx-time gap before the first real point (0 if untapered - see taperEdges()).
		RealType _right_interval; ///< Assumed tx-time gap after the last real point (0 if untapered - see taperEdges()).
	};
}
