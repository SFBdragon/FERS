// SPDX-License-Identifier: GPL-2.0-only
//
// Copyright (c) 2026-present FERS Contributors (see AUTHORS.md).
//
// See the GNU GPLv2 LICENSE file in the FERS project root for more information.

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

		/// The ratio of received to transmitted voltage (Vr / Vt) at unit impedance.
		/// This is phase-coherent: path-dependent phase modulation is included,
		/// however bulk phase is not (e^{-jkR}).
		///
		/// The magnitude is equivalent to the square root of the linear power gain.
		ComplexType gain;

		/// A unique, stable ID for a continuously-varying channel over time.
		uint64_t path_id;

		/// The streaaming source index for the path's transmitter, if `findRxFromTxPaths` is used. Otherwise undefined.
		size_t source_index;
		/// The receiver of the path.
		radar::Receiver* receiver;
		// ^ It's mutable so that calculateResponses can stuff the reponse into the receiver.
	};

	struct ThreadContext
	{
		virtual ~ThreadContext() = default;
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

		// TODO_SHAUN document

		/// Called once per worker thread that will call the methods below.
		/// Default: no context needed.
		[[nodiscard]] virtual std::unique_ptr<ThreadContext> makeThreadContext() const { return nullptr; }

		[[nodiscard]] virtual std::vector<PathsAtTime> findTxToRxPaths(ThreadContext* ctx,
																	   const radar::Transmitter& transmitter,
																	   const std::vector<RealType>& times) const = 0;

		[[nodiscard]] virtual std::vector<PropagationPath>
		findRxFromTxPaths(ThreadContext* ctx, radar::Receiver* receiver,
						  const std::vector<core::ActiveStreamingSource>& sources, RealType rx_time) const = 0;
	};
}
