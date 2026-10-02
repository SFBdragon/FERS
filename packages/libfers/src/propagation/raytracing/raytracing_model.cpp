// SPDX-License-Identifier: GPL-2.0-only
//
// Copyright (c) 2026-present FERS Contributors (see AUTHORS.md).
//
// See the GNU GPLv2 LICENSE file in the FERS project root for more information.

#include "raytracing_model.h"

#include <algorithm>
#include <cassert>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <ranges>
#include <span>
#include <unordered_map>
#include <utility>
#include <vector>

#include "core/assets.h"
#include "core/config.h"
#include "core/parameters.h"
#include "core/sim_id.h"
#include "core/simulation_state.h"
#include "core/world.h"
#include "propagation/common.h"
#include "propagation/common_defs.h"
#include "propagation/math.h"
#include "propagation/propagation_model.h"
#include "propagation/raytracing/antenna_model.h"
#include "propagation/raytracing/mesh_importer.h"
#include "propagation/raytracing/sbr_shared.h"
#include "propagation/utils.h"
#include "radar/radar_obj.h"
#include "radar/receiver.h"
#include "radar/target.h"
#include "radar/transmitter.h"
#include "signal/radar_signal.h"

#if defined(FERS_ENABLE_OPTIX)
#include "optix/optix.h"
#endif


using namespace math;
using namespace radar;

namespace propagation::raytracing
{
	template <std::ranges::forward_range I, typename Proj = std::identity>
	static void buildRadarBuffer(const std::span<const RealType> times,
								 const std::unordered_map<SimId, uint32_t>& antenna_id_to_index, const I& items,
								 std::vector<ActiveAntenna>& antennas_out, Proj proj = {}) noexcept
	{
		antennas_out.clear();

		for (const auto t : times)
		{
			for (auto& item : items)
			{
				const radar::Radar& radar = std::invoke(proj, item);

				const auto antenna_index = antenna_id_to_index.at(radar.getAntenna()->getId());
				const auto posititon = radar.getPosition(t);
				const auto orient = radar.getRotation(t);

				const auto antenna = ActiveAntenna{
					.position = Double3{posititon.x, posititon.y, posititon.z},
					.direction = AzEl{float(orient.azimuth), float(orient.elevation)},
					.antenna_model_index = antenna_index,
				};

				antennas_out.push_back(antenna);
			}
		}
	}

	static void buildRxFlags(core::World* world, TraceJob& job_out) noexcept
	{
		job_out.rx_flags.reserve(world->getReceivers().size());
		for (const auto& rx : world->getReceivers())
		{
			uint8_t flags = 0;
			if (rx->checkFlag(radar::Receiver::RecvFlag::FLAG_NOPROPLOSS))
				flags |= RxFlags::FLAG_NOSPREADING;
			if (rx->checkFlag(radar::Receiver::RecvFlag::FLAG_NODIRECT))
				flags |= RxFlags::FLAG_NODIRECT;

			job_out.rx_flags.push_back(static_cast<RxFlags>(flags));
		}
	}

	ContributionGroups groupContributions(const std::vector<Contribution>& contributions)
	{
		ContributionGroups groups;
		for (const auto& c : contributions)
		{
			const auto coarse = (static_cast<uint64_t>(c.source_times_index) << 32) | c.dest_index;
			groups[ContributionGroupKey{coarse, c.path_id}].push_back(&c);
		}
		return groups;
	}

	[[nodiscard]] RealType contributionGroupDelay(const std::vector<const Contribution*>& group)
	{
		assert(!group.empty());

		RealType reference_delay = group.front()->delay;
		for (const auto* c : group)
			reference_delay = std::min(reference_delay, c->delay);

		return reference_delay;
	}

	[[nodiscard]] PropagationPath aggregateContributionGroup(const std::vector<const Contribution*>& group,
															 RealType group_delay, RealType carrier_frequency)
	{
		assert(!group.empty());

		// FERS adds the effect of bulk phase delay later (post down-mixing, -2pi*f_c*tau phase shift).
		// This works in part because PointScatterModel doesn't aggregate at all.
		// We want to cut down on the number of paths before handing them back to the rest of FERS.
		// Paths that took the exact same facet sequence are essentially the same path but we're
		// integrating over different ray tubes' contributions. Coherent summation is important.
		//
		// Up-mixing and down-mixing the full paths is poorly conditioned, as the carrier
		// is incredibly high-frequency and high magntidude (accumulative phase).
		// Instead, we can coherently sum physically accounting for relative phase offset.
		// The bulk phase adjustment will be added later.
		//
		// The approach:
		// - The most direct path has the least delay. Use that as the reference path delay.
		// - Coherently sum the other contributions, adjusting for the relative phase delay for coherence.

		ComplexType summed_voltage{};
		for (const auto* c : group)
		{
			const double dt = c->delay - group_delay;
			const RealType angle = -2.0 * PI * carrier_frequency * dt;
			const ComplexType v{double(c->voltage.re), double(c->voltage.im)};
			summed_voltage += v * std::exp(ComplexType{0.0, angle});
		}

		PropagationPath path{};
		path.delay = group_delay;
		path.gain = summed_voltage;
		path.path_id = group.front()->path_id;
		return path;
	}


	RayTracingModel::RayTracingModel(core::World* world, RayTracingBackend backend) : _world(world)
	{
		SceneData scene{};

		for (const auto& target : world->getTargets())
		{
			const auto& geometry = target->getGeometry();
			if (!geometry.has_value())
			{
				continue;
			}

			if (const SimId mesh_id = geometry->mesh->id; !scene.mesh_ids.contains(mesh_id))
			{
				scene.mesh_ids.insert({mesh_id, static_cast<uint32_t>(scene.meshes.size())});
				scene.meshes.emplace_back(loadMesh(geometry->mesh->path.string(), scene.vertices, scene.indices));
			}

			if (const SimId material_id = geometry->material->id; !scene.mat_ids.contains(material_id))
			{
				scene.mat_ids.insert({material_id, static_cast<uint32_t>(scene.materials.size())});
				const auto* mat = geometry->material;
				scene.materials.emplace_back(mat->relative_permittivity, mat->conductivity);
			}
		}

		for (const auto& [id, antenna] : world->getAntennas())
		{
			_antenna_id_to_index.insert({id, static_cast<uint32_t>(scene.antenna_models.size())});
			scene.antenna_models.emplace_back(buildAntennaModel(*antenna, scene.antenna_gains, 180, 90));
		}

		switch (backend)
		{
		case RayTracingBackend::OptiX:
#ifdef FERS_ENABLE_OPTIX
			_engine = optix::OptixEngine::Create(std::move(scene));
			break;
#else
			throw std::runtime_error("FERS was not built with OptiX support enabled.");
#endif
		}
	}

	std::unique_ptr<ThreadContext> RayTracingModel::makeThreadContext() const { return _engine->makeThreadContext(); }

	std::vector<PathsAtTime> RayTracingModel::findTxToRxPaths(ThreadContext* ctx, const Transmitter& transmitter,
															  const std::vector<RealType>& times) const
	{
		TraceJob job{};
		job.type = TraceJobType::FindTxToRx;
		job.world = _world;
		job.times = times;

		CarrierModel<float> carrier;
		carrier.kind = CarrierModelKind::Constant;
		carrier.params.constant.frequency = float(transmitter.getSignal()->getCarrier());
		job.carriers.push_back(carrier);

		buildRadarBuffer(job.times, _antenna_id_to_index, std::span<const radar::Transmitter, 1>(&transmitter, 1),
						 job.source_antennas);

		buildRadarBuffer(job.times, _antenna_id_to_index, _world->getReceivers(), job.dest_antennas,
						 [&](auto& rx) -> const radar::Radar& { return *rx; });

		buildRxFlags(_world, job);

		auto contributions = _engine->trace(ctx, job);

		std::vector<PathsAtTime> paths_at_time(job.times.size());
		for (size_t i = 0; i < job.times.size(); ++i)
		{
			paths_at_time[i].tx_time = job.times[i];
		}

		const auto& receivers = _world->getReceivers();
		for (const auto& [key, group] : groupContributions(contributions))
		{
			const auto source_times_index = static_cast<uint32_t>(key.coarse >> 32);
			const auto dest_index = static_cast<uint32_t>(key.coarse);
			const auto time_idx = source_times_index; // source_count == 1
			assert(time_idx < paths_at_time.size());
			assert(dest_index < receivers.size());

			const RealType carrier_freq = transmitter.getSignal()->getCarrier();

			auto group_delay = contributionGroupDelay(group);
			auto path = aggregateContributionGroup(group, group_delay, carrier_freq);
			path.source_index = 0; // Undefined for findTxToRxPaths, matching PointScatterModel
			path.receiver = receivers[dest_index].get();
			paths_at_time[time_idx].paths.push_back(path);
		}

		return paths_at_time;
	};

	std::vector<PropagationPath>
	RayTracingModel::findRxFromTxPaths(ThreadContext* ctx, radar::Receiver* receiver,
									   const std::vector<core::ActiveStreamingSource>& sources, RealType rx_time) const
	{
		// TODO_SHAUN time batching? receiver batching?
		const RealType times = rx_time;

		TraceJob job{};
		job.type = TraceJobType::FindRxFromTx;
		job.world = _world;
		job.times = std::span(&times, 1);

		for (const auto& source : sources)
		{
			job.carriers.push_back(buildPropagationCarrier<float>(source));
		}

		buildRadarBuffer(job.times, _antenna_id_to_index, std::span<radar::Receiver, 1>(receiver, 1),
						 job.source_antennas);

		buildRadarBuffer(job.times, _antenna_id_to_index, sources, job.dest_antennas,
						 [&](auto& src) -> const radar::Radar& { return *src.transmitter; });

		buildRxFlags(_world, job);

		auto contributions = _engine->trace(ctx, job);

		std::vector<PropagationPath> paths;
		const auto groups = groupContributions(contributions);
		paths.reserve(groups.size());
		for (const auto& [key, group] : groups)
		{
			const auto tx_index = static_cast<uint32_t>(key.coarse);
			assert(tx_index < sources.size());

			auto group_delay = contributionGroupDelay(group);
			auto carrier_frequency = RealType(sampleCarrierFrequency(job.carriers[tx_index], rx_time - group_delay));
			auto path = aggregateContributionGroup(group, group_delay, carrier_frequency);
			path.source_index = tx_index;
			path.receiver = receiver;
			paths.push_back(path);
		}

		return paths;
	}

	unsigned int RayTracingModel::maxPropagationLegs() const noexcept
	{
		// For bouncing off of N scatterers, there are N+1 propagation legs.
		// e.g. 1 scatterer = bistatic path = 2 legs
		return params::rtModelParams().scatter_limit + 1;
	}
}
