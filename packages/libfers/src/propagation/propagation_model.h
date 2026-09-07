//
// Created by sfbea on 2026/08/18.
//

#pragma once

#include <cstdint>
#include <generator>
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
		/// Total path length, in meters.
		///
		/// This is never zero. No path is created if this would be zero.
		RealType length;
		// /// d(range)/dt at eval_time, m/s
		// RealType range_rate;

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

		/// The source of the path, if `findRxFromTxPaths` is used. Otherwise null.
		const core::ActiveStreamingSource* source;
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

	std::generator<RealType> timePointGenerator(RealType start_time, RealType end_time, RealType sampling_rate);

	/**
	 * @class RangeError
	 * @brief Exception thrown when a range calculation fails, typically due to objects being too close.
	 */
	class RangeError final : public std::exception
	{
	public:
		/**
		 * @brief Provides the error message for the exception.
		 * @return A C-style string describing the error.
		 */
		[[nodiscard]] const char* what() const noexcept override
		{
			return "Range error in radar equation calculations";
		}
	};

}
