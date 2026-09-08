//
// Created by sfbea on 2026/08/18.
//

#pragma once

#include <cstdint>
#include <vector>

#include "core/config.h"
#include "core/simulation_state.h"

namespace radar
{
	class Receiver;
	class Transmitter;
}
namespace propagation
{
	enum class PropagationModelType : uint8_t
	{
		/// Applies the radar range equation to point-target approximations
		/// using Radar Cross-Section (RCS). Optimized for very-long-range,
		/// low-clutter scenarios. Low compute intensity.
		RcsPointScatter = 1,

		/// Applies Geometric Optics (GO) ray tracing against target meshes.
		/// Supports multi-bounce, occlusion, transmission, and dense clutter.
		/// High compute intensity.
		GoRayTracing = 2,
	};

	// struct PropagationBounce
	// {
	// 	math::Vec3 position;
	// 	const radar::Target* target;
	// };

	/**
	 * @struct PropagationPath
	 * @brief Data type containing a valid radar path identified by the path finder.
	 */
	class PropagationPath
	{
	public:
		/// Total path delay, in seconds.
		///
		/// This is never zero. No path is created if this would be zero.
		RealType delay;

		/// The power
		///
		/// Figure out whether this includes:
		/// - the signal strength (no?)
		/// - the rx gain (maybe?)
		/// - the tx gain (?)
		/// - many other fun aspects of the radar equation.
		///
		/// This will definitely include:
		/// - RCS-based factors
		/// - ray power division, dispersion, whatever.
		/// - ray reflection/transmission coeffs
		RealType gain;

		/// A unique, stable ID for a continuously-varying channel over time.
		uint64_t path_id;

		/// The streaaming source index for the path's transmitter, if `findRxFromTxPaths` is used. Otherwise undefined.
		size_t source_index;
		/// The receiver of the path.
		radar::Receiver* receiver;
		// ^ It's mutable so that calculateResponses can stuff the reponse into the receiver.

		// /// The list of bounces from transmitter to receiver.
		// ///
		// /// Zero-length for direct paths.
		// std::vector<PropagationBounce> vertices;
	};


	struct PathsAtTime
	{
		/// All propagation channel found from the transmitter to any receiver at tx_time.
		std::vector<PropagationPath> paths;
		/// The transmission time of each path in `paths`.
		RealType tx_time;
	};

	class PropagationModel
	{
	public:
		virtual ~PropagationModel();

		[[nodiscard]] virtual std::vector<PathsAtTime> findTxToRxPaths(const radar::Transmitter& transmitter,
																	   RealType start_tx_time) const = 0;

		[[nodiscard]] virtual std::vector<PropagationPath>
		findRxFromTxPaths(radar::Receiver* receiver, const std::vector<core::ActiveStreamingSource>& sources,
						  RealType rx_time) const = 0;
	};

	class TimePointRange
	{
		RealType start_, end_, step_;
		int count_;

	public:
		TimePointRange(RealType start_time, RealType end_time, RealType sampling_rate) :
			start_(start_time), end_(end_time), step_(RealType(1) / sampling_rate),
			count_(static_cast<int>(std::ceil((end_time - start_time) / step_)))
		{
		}

		class iterator
		{
			const TimePointRange* r_;
			int i_;

		public:
			using iterator_category = std::input_iterator_tag;
			using value_type = RealType;
			using difference_type = std::ptrdiff_t;
			using pointer = const RealType*;
			using reference = RealType;

			iterator(const TimePointRange* r, int i) : r_(r), i_(i) {}

			RealType operator*() const { return i_ < r_->count_ ? r_->start_ + i_ * r_->step_ : r_->end_; }

			iterator& operator++()
			{
				++i_;
				return *this;
			}
			iterator operator++(int)
			{
				auto t = *this;
				++i_;
				return t;
			}

			bool operator==(const iterator& o) const { return i_ == o.i_; }
			bool operator!=(const iterator& o) const { return i_ != o.i_; }
		};

		[[nodiscard]] iterator begin() const { return {this, 0}; }
		[[nodiscard]] iterator end() const { return {this, count_ + 1}; }
	};

}
