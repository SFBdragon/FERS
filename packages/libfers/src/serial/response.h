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
	inline RealType controlPointEdgePeriod() noexcept { return 1.0 / params::simSamplingRate(); }

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
		 * @param first_point The response's first real point.
		 * @param left_interval The true tx-time gap immediately before `first_point` (0 if
		 * `first_point` sits at the transmission's actual start - see `taperEdges()`).
		 * Defaults to `controlPointEdgePeriod()` for callers that don't track a sample grid.
		 * @param right_interval This response's assumed tx-time gap after `first_point`,
		 * used only if no further point is ever added (see `addInterpPoint()`).
		 */
		Response(const fers_signal::RadarSignal* signal, const interp::InterpPoint first_point,
				const RealType left_interval = controlPointEdgePeriod(),
				const RealType right_interval = controlPointEdgePeriod()) :
			_signal(signal), _left_interval(left_interval), _right_interval(right_interval)
		{
			if (!signal->isPulsed())
			{
				throw std::logic_error(
					"Attempted to create Reponse with RadarSignal that didn't contain a PulseWaveform");
			}

			_wave = signal->getPulseWaveform();
			_points.push_back(first_point);
		}

		~Response() = default;
		Response(const Response&) = delete;
		Response& operator=(const Response&) = delete;
		Response(Response&&) = delete;
		Response& operator=(Response&&) = delete;

		/**
		 * @brief Retrieves the start time of the response, including its leading fade-in taper.
		 * Coincides with the first real point's own time when `left_interval` is 0 (the
		 * response's first point sits at the transmission's actual start - see `taperEdges()`).
		 *
		 * @return Start time as a `RealType`. Returns 0.0 if no points are present.
		 */
		[[nodiscard]] RealType startTime() const noexcept { return _points.empty() ? 0.0 : taperEdges().lead.rx_time; }

		/**
		 * @brief Retrieves the end time of the response, including its trailing fade-out taper.
		 * Coincides with the last real point's own time when `right_interval` is 0 (the
		 * response's last point sits at the transmission's actual end - see `taperEdges()`).
		 *
		 * @return End time as a `RealType`. Returns 0.0 if no points are present.
		 */
		[[nodiscard]] RealType endTime() const noexcept { return _points.empty() ? 0.0 : taperEdges().tail.rx_time; }

		/**
		 * @brief Adds an interpolation point to the response.
		 *
		 * @param point The interpolation point to be added.
		 * @param right_interval This response's assumed tx-time gap after `point`, used only
		 * if no further point is ever added (see `taperEdges()`). Defaults to
		 * `controlPointEdgePeriod()` for callers that don't track a sample grid.
		 * @throws std::logic_error If the new point has a time earlier than the last point.
		 */
		void addInterpPoint(const interp::InterpPoint& point, RealType right_interval = controlPointEdgePeriod());

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

		/// Returns a view over this response's real points bracketed by its synthetic
		/// lead/tail taper points (see taperEdges()).
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
		 * @brief Extrapolates one taper point (lead or tail) `interval` tx-seconds beyond `edge`,
		 * using the Doppler rate observed between `edge` and `neighbor` (the next real point in
		 * toward the response's interior).
		 *
		 * `interval` is a tx-time duration (see `_left_interval`/`_right_interval`). The
		 * corresponding rx-time width isn't `interval` itself unless delay is constant: if delay
		 * is changing at the locally-observed rate `(neighbor.delay - edge.delay) / tx_gap`
		 * (`tx_gap` being `neighbor`/`edge`'s own tx-time separation, recovered as
		 * `rx_time - delay`), extrapolating `interval` more tx-seconds carries that same rate's
		 * worth of extra delay with it. So the rx-time width is `interval + extrapolated_delay`,
		 * not `interval` - equivalently `interval * (rx_gap/tx_gap)`.
		 */
		[[nodiscard]] static interp::InterpPoint extrapolateTaperPoint(const interp::InterpPoint& edge,
																	   const interp::InterpPoint& neighbor,
																	   const RealType interval, const bool forward)
		{
			interp::InterpPoint point = edge;
			point.gain = 0.0;

			const RealType tx_gap = (neighbor.rx_time - neighbor.delay) - (edge.rx_time - edge.delay);
			const RealType scale = interval / tx_gap;
			const RealType delay_delta = (neighbor.delay - edge.delay) * scale;
			const RealType phase_delta = (neighbor.phase_delay - edge.phase_delay) * scale;
			const RealType rx_width = interval + delay_delta;

			const RealType sign = forward ? RealType(1.0) : RealType(-1.0);
			point.rx_time += sign * rx_width;
			point.delay += sign * delay_delta;
			point.phase_delay += sign * phase_delta;
			return point;
		}

		/**
		 * @brief Computes this response's synthetic zero-gain lead and tail points.
		 *
		 * `_left_interval`/`_right_interval` (supplied by the caller building this response -
		 * see the constructor and `addInterpPoint()`) are the true tx-time gaps to whatever lies
		 * immediately outside the response's first/last real point: 0 if that side is the
		 * transmission's actual start/end (untapered - nothing renders out there), otherwise the
		 * real local sample-grid gap. With at least two real points, that interval is
		 * Doppler-extrapolated per `extrapolateTaperPoint()` so two responses covering the same
		 * continuous physical path but split by a path_id change (as ray tracing can do) still
		 * meet exactly where their real data does, rather than overlapping (double-counting) or
		 * gapping. With only one real point there's no rate to extrapolate, so the interval is
		 * used as a plain rx-time offset with delay/phase_delay unchanged.
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
