// SPDX-License-Identifier: GPL-2.0-only
//
// Copyright (c) 2026-present FERS Contributors (see AUTHORS.md).
//
// See the GNU GPLv2 LICENSE file in the FERS project root for more information.

#pragma once

#include <cstdint>
#include <functional>
#include <memory>
#include <unordered_map>
#include <vector>

#include "core/config.h"
#include "core/simulation_state.h"
#include "core/world.h"
#include "propagation/math.h"
#include "propagation/propagation_model.h"
#include "propagation/raytracing/sbr_shared.h"
#include "radar/receiver.h"

namespace propagation::raytracing
{
	enum class RayTracingBackend : uint8_t
	{
		/// Use the NVIDIA CUDA-based OptiX ray tracing engine.
		///
		/// Requires a compatible CUDA driver and CUDA device.
		OptiX = 1,
	};

	enum class TraceJobType : uint8_t
	{
		FindTxToRx = 0,
		FindRxFromTx = 1,
	};

	struct SceneData
	{
		std::unordered_map<SimId, uint32_t> mesh_ids;
		std::vector<Float3> vertices;
		std::vector<Uint3> indices;
		std::vector<TriangleMeshView> meshes;

		std::unordered_map<SimId, uint32_t> mat_ids;
		std::vector<Material> materials;

		std::vector<float> antenna_gains;
		std::vector<AntennaModel> antenna_models;
	};

	struct TraceJob
	{
		core::World* world;
		std::span<const RealType> times;

		std::vector<CarrierModel<float>> carriers;

		std::vector<ActiveAntenna> source_antennas;
		std::vector<ActiveAntenna> dest_antennas;

		std::vector<RxFlags> rx_flags;

		TraceJobType type;
	};

	/**
	 * @brief A `(source_times_index, dest_index, path_id)` key. Contributions sharing a key are
	 * ray-tube samples along the same TX->facet-sequence->RX path, at the same timestep.
	 */
	struct ContributionGroupKey
	{
		/// `(source_times_index << 32) | dest_index`.
		uint64_t coarse;
		uint64_t path_id;

		bool operator==(const ContributionGroupKey&) const = default;
	};

	struct ContributionGroupKeyHash
	{
		size_t operator()(const ContributionGroupKey& k) const
		{
			return std::hash<uint64_t>{}(k.coarse) ^ (std::hash<uint64_t>{}(k.path_id) << 1);
		}
	};

	using ContributionGroups =
		std::unordered_map<ContributionGroupKey, std::vector<const Contribution*>, ContributionGroupKeyHash>;

	/**
	 * @brief Buckets `contributions` by `(source_times_index, dest_index, path_id)`: ray-tube samples
	 * along the same TX->facet-sequence->RX path, at the same timestep.
	 */
	[[nodiscard]] ContributionGroups groupContributions(const std::vector<Contribution>& contributions);

	/**
	 * @brief The reference delay for a group: the minimum delay across its contributions, i.e. the
	 * most direct of the ray-tube samples making up the path.
	 *
	 * @param group Non-empty.
	 */
	[[nodiscard]] RealType contributionGroupDelay(const std::vector<const Contribution*>& group);

	/**
	 * @brief Coherently sums a group of `Contribution`s that share a `path_id` key
	 * (same source, facet sequence, and destination) into a single `PropagationPath`.
	 *
	 * @param group Non-empty. All contributions must share the same `source_times_index`/`dest_index`/`path_id`.
	 * @param group_delay The reference delay (typically `contributionGroupDelay(group)`) used to align phases.
	 * @param carrier_frequency The carrier frequency (Hz) shared by every contribution in `group`.
	 */
	[[nodiscard]] PropagationPath aggregateContributionGroup(const std::vector<const Contribution*>& group,
															 RealType group_delay, RealType carrier_frequency);


	class RayTracingEngine
	{
	public:
		virtual ~RayTracingEngine() = default;

		/**
		 * @brief Create a worker thread's mutable state object.
		 */
		[[nodiscard]] virtual std::unique_ptr<ThreadContext> makeThreadContext() = 0;

		/**
		 * @brief Execute the ray tracing batch job.
		 */
		[[nodiscard]] virtual std::vector<Contribution> trace(ThreadContext*, TraceJob&) = 0;
	};


	class RayTracingModel : public PropagationModel
	{
	public:
		RayTracingModel(core::World* world, RayTracingBackend backend);

		[[nodiscard]] std::unique_ptr<ThreadContext> makeThreadContext() const override;

		[[nodiscard]] std::vector<PathsAtTime> findTxToRxPaths(ThreadContext* ctx,
															   const radar::Transmitter& transmitter,
															   const std::vector<RealType>& times) const override;

		[[nodiscard]] std::vector<PropagationPath>
		findRxFromTxPaths(ThreadContext* ctx, radar::Receiver* receiver,
						  const std::vector<core::ActiveStreamingSource>& sources, RealType rx_time) const override;

	private:
		core::World* _world;

		std::unordered_map<SimId, uint32_t> _antenna_id_to_index;

		std::unique_ptr<RayTracingEngine> _engine;
	};
}
