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
#include <stdexcept>
#include <unordered_map>
#include <utility>
#include <vector>

#include "core/assets.h"
#include "core/config.h"
#include "core/logging.h"
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

#ifdef FERS_ENABLE_OPTIX
#include "optix/optix.h"
#endif

#ifdef FERS_ENABLE_EMBREE
#include "embree/embree.h"
#endif

using namespace math;
using namespace radar;

namespace propagation::raytracing
{
	Float3x4 computeTargetTransform(const radar::Target& target, const RealType t)
	{
		const auto pos = target.getPosition(t);
		const auto rot = target.getRotation(t);

		const auto forward = normalize(Double3{
			std::cos(rot.azimuth) * std::cos(rot.elevation),
			std::sin(rot.azimuth) * std::cos(rot.elevation),
			std::sin(rot.elevation),
		});

		Double3 world_up{0.0, 0.0, 1.0};
		if (std::abs(dot(forward, world_up)) > 0.999)
			world_up = Double3{1.0, 0.0, 0.0}; // forward ~parallel to world up; avoid a degenerate cross product

		const auto right = normalize(cross(forward, world_up));
		const auto up = cross(right, forward);

		// `pos` is `math::Vec3` (the legacy general-purpose vector type `radar::Object::getPosition`
		// returns), distinct from `right`/`forward`/`up`'s `propagation::Double3` - generic on `.x/.y/.z`
		// so it accepts both.
		const auto toFloat3 = [](const auto& v) {
			return Float3{static_cast<float>(v.x), static_cast<float>(v.y), static_cast<float>(v.z)};
		};
		return Float3x4{toFloat3(right), toFloat3(forward), toFloat3(up), toFloat3(pos)};
	}

	Float3x4 affineInverse(const Float3x4& m) noexcept
	{
		// m's columns (x, y, z) form an orthonormal rotation basis (see computeTargetTransform), so
		// the inverse rotation is just the transpose, and the inverse translation falls out of
		// distributing R^T across `world = R * local + T` solved for `local`.
		const Float3 row0{m.x.x, m.y.x, m.z.x};
		const Float3 row1{m.x.y, m.y.y, m.z.y};
		const Float3 row2{m.x.z, m.y.z, m.z.z};
		return Float3x4{row0, row1, row2, Float3{-dot(row0, m.w), -dot(row1, m.w), -dot(row2, m.w)}};
	}

	uint32_t maxScatterLimit() noexcept
	{
		return std::min<uint32_t>(MAX_SCATTER_LIMIT, params::rayTracingParams().scatter_limit);
	}

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
			const ComplexType v{double(c->path_gain.re), double(c->path_gain.im)};
			summed_voltage += v * std::exp(ComplexType{0.0, angle});
		}

		PropagationPath path{};
		path.delay = group_delay;
		path.gain = summed_voltage;
		path.path_id = group.front()->path_id;
		return path;
	}


	RayTracingModel::RayTracingModel(core::World* world, params::RayTracingEnginePreference engine,
									 pool::ThreadPool& pool) :
		_world(world)
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

		if (params::rayTracingParams().scatter_limit != maxScatterLimit())
		{
			LOG(logging::Level::WARNING,
				"Ray-tracing propagation model kernel was not built with support for a scatter limit of {}, this "
				"build's maximum is {}. It's possible to edit the kernel's `MAX_SCATTER_LIMIT` and rebuild. This run "
				"will use a scatter limit of {}.",
				params::rayTracingParams().scatter_limit, maxScatterLimit(), maxScatterLimit());
		}

		switch (engine)
		{
		case params::RayTracingEnginePreference::Auto:
			LOG(logging::Level::INFO, "Attempting to initialise supported engines.");
#ifdef FERS_ENABLE_OPTIX
			try
			{
				LOG(logging::Level::INFO, "Attempting to initialise OptiX back-end...");
				_engine = std::unique_ptr<RayTracingBackend>(new optix::OptixBackend(std::move(scene)));
				LOG(logging::Level::INFO, "OptiX back-end successfully initialised.");
				break;
			}
			catch (std::runtime_error& error)
			{
				LOG(logging::Level::INFO, "OptiX back-end failed to initialise.\n  {}", error.what());
			}
#endif
#ifdef FERS_ENABLE_EMBREE
			try
			{
				LOG(logging::Level::INFO, "Attempting to initialise Embree back-end...");
				_engine = std::unique_ptr<RayTracingBackend>(new embree::EmbreeBackend(std::move(scene), pool));
				LOG(logging::Level::INFO, "Embree back-end successfully initialised.");
				break;
			}
			catch (std::runtime_error& error)
			{
				LOG(logging::Level::INFO, "Embree back-end failed to initialise.\n  {}", error.what());
			}
#endif
			throw std::runtime_error("No ray tracing engine back-ends could be initialised.");
		case params::RayTracingEnginePreference::OptiX:
#ifdef FERS_ENABLE_OPTIX
			_engine = std::unique_ptr<RayTracingBackend>(new optix::OptixBackend(std::move(scene)));
			LOG(logging::Level::INFO, "OptiX back-end initialised.");
			break;
#else
			throw std::runtime_error("FERS was not built with OptiX support enabled.");
#endif
		case params::RayTracingEnginePreference::Embree:
#ifdef FERS_ENABLE_EMBREE
			_engine = std::unique_ptr<RayTracingBackend>(new embree::EmbreeBackend(std::move(scene), pool));
			break;
#else
			throw std::runtime_error("FERS was not built with Embree support enabled.");
#endif
		}
	}

	std::unique_ptr<ThreadContext> RayTracingModel::makeThreadContext() const
	{
		return _engine->makeThreadContext(_world);
	}

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

		// double incoherent_sum = 0;
		// for (auto& contrib : contributions)
		// 	incoherent_sum += std::sqrt(
		// 		double(contrib.path_gain.re * contrib.path_gain.re + contrib.path_gain.im * contrib.path_gain.im));
		// LOG(logging::Level::DEBUG, "Incoherent sum: {}", incoherent_sum);

		// ComplexType coherent_sum{};
		// for (auto& contrib : contributions)
		// 	coherent_sum += ComplexType{contrib.path_gain.re, contrib.path_gain.im};
		// LOG(logging::Level::DEBUG, "Coherent sum arg: {}", std::arg(coherent_sum));

		// double total_area = 0;
		// for (auto& contrib : contributions)
		// 	total_area += double(contrib.area);
		// LOG(logging::Level::DEBUG, "Total patch area: {}", total_area);

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
		return params::rayTracingParams().scatter_limit + 1;
	}
}
