// SPDX-License-Identifier: GPL-2.0-only
//
// Copyright (c) 2026-present FERS Contributors (see AUTHORS.md).
//
// See the GNU GPLv2 LICENSE file in the FERS project root for more information.

#include "optix.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cuda.h>
#include <cuda_device_runtime_api.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <iostream>
#include <memory>
#include <optix.h>
#include <optix_function_table_definition.h>
#include <optix_stubs.h>
#include <optix_types.h>
#include <stdexcept>
#include <unordered_map>
#include <vector>

#include "core/config.h"
#include "core/parameters.h"
#include "core/world.h"
#include "propagation/math.h"
#include "propagation/propagation_model.h"
#include "propagation/raytracing/optix/kernel_defs.h"
#include "propagation/raytracing/optix/utils.h"
#include "propagation/raytracing/raytracing_model.h"
#include "propagation/raytracing/sbr_shared.h"
#include "radar/target.h"

// kernel.cu compiled to PTX with nvcc. See `cmake/FersOptiX.cmake`.
// Embedded as `devicePrograms_ptx` and `devicePrograms_ptx_len`.
#include "kernel_ptx.h"

namespace propagation::raytracing::optix
{
	static int pickCudaDevice()
	{
		// Intitialize CUDA with a no-op (for CUDA versions before v12)
		cudaFree(nullptr);

		// Check for available CUDA-capable devices.
		int numDevices{};
		cudaGetDeviceCount(&numDevices);
		if (numDevices == 0)
			throw std::runtime_error("No CUDA capable devices found");

		int bestDevice = -1;
		double bestScore = -1.0;

		for (int i = 0; i < numDevices; ++i)
		{
			cudaDeviceProp prop{};
			if (cudaGetDeviceProperties(&prop, i) != cudaSuccess)
				continue;

			// Reject very old architectures. (Below SM 5.0)
			if (prop.major < 5)
				continue;

			// Base score: number of SMs (throughput proxy)
			double score = static_cast<double>(prop.multiProcessorCount);

			// Bonus for RT-core-capable architectures (Turing = SM 7.5 and up),
			// since that's what accelerates OptiX.
			if (prop.major > 7 || (prop.major == 7 && prop.minor >= 5))
				score *= 4.0;

			// Mild penalty for integrated/low-power parts.
			if (prop.integrated != 0)
				score *= 0.5;

			if (score > bestScore)
			{
				bestScore = score;
				bestDevice = i;
			}
		}

		if (bestDevice < 0)
			throw std::runtime_error("No suitable CUDA capable device found");

		return bestDevice;
	}

	CudaContext::CudaContext(int device_id)
	{
		// Intitialize CUDA with a no-op (for CUDA versions before v12)
		cudaFree(nullptr);

		// Initialize OptiX
		OPTIX_CHECK(optixInit());

		// Set the device to use.
		// This sets a CUDA context for this thread as well.
		CUDA_CHECK(cudaSetDevice(device_id));

		// Get the current CUDA context.
		CUDA_RES_CHECK(cuCtxGetCurrent(&_context));
	}


	static void contextLogCallback(unsigned int level, const char* tag, const char* message, void*)
	{
		std::cerr << "OptiX Device Context Log [level " << level << "] " << tag << ": " << message << '\n';
	}
	OptixContext::OptixContext(CUcontext context) : _context(nullptr)
	{
		OPTIX_CHECK(optixDeviceContextCreate(context, nullptr, &_context));
		OPTIX_CHECK(optixDeviceContextSetLogCallback(_context, contextLogCallback, nullptr, 4));
	}

	// By initialising the CUDA and OptiX RAII wrappers first, they get automatically
	// cleaned up if part of the initialization throws. Desctruction of the OptiX device
	// context will automatically clean up all the modules, program groups, pipelines, etc.
	// associated with it. CudaBuffers handle cleaning up CUDA memory.
	OptixEngine::OptixEngine(SceneData scene) :
		_cuda_ctx(pickCudaDevice()), _optix_ctx(_cuda_ctx.get()), _stream(), _module_comp_options(),
		_pipeline_comp_options(), _pipeline_link_options(), _module(), _raygen_program(), _raygen_record(_stream.get()),
		_miss_records(_stream.get()), _hitgroup_records(_stream.get()), _sbt(), _materials(_stream.get()),
		_vertices(_stream.get()), _indeces(_stream.get()), _antenna_models(_stream.get()), _antenna_gains(_stream.get())
	{
		// Set up the compile and link options used by engine going forward.
		// These are persisted as OptixPipelines need them, which are created later
		// in ThreadContexts as pipelines aren't thread-safe, and the options must be consistent.

		// TODO_SHAUN fix all these

		_module_comp_options.maxRegisterCount = 50;
		_module_comp_options.optLevel = OPTIX_COMPILE_OPTIMIZATION_DEFAULT;
		_module_comp_options.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_DEFAULT;

		// ALLOW_SINGLE_GAS would forbid using an IAS as the top-level traversable at all, which
		// buildOrRefitIAS below needs — one level of instancing (IAS -> GAS, no nested IASes).
		_pipeline_comp_options.traversableGraphFlags = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_LEVEL_INSTANCING;
		_pipeline_comp_options.usesMotionBlur = 0;
		_pipeline_comp_options.numPayloadValues = 2;
		_pipeline_comp_options.numAttributeValues = 2;
		_pipeline_comp_options.exceptionFlags = OPTIX_EXCEPTION_FLAG_NONE;
		_pipeline_comp_options.pipelineLaunchParamsVariableName = "shader_params";

		_pipeline_link_options.maxTraceDepth = MAX_RAY_DEPTH;

		createDevicePrograms();
		processAssets(std::move(scene));
		createSbt();
	}

	void OptixEngine::createDevicePrograms()
	{
		char log[2048];
		size_t log_size = sizeof(log);

		// NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast): unsigned char -> char is memory safe
		const auto* ptx = reinterpret_cast<const char*>(devicePrograms_ptx);

		OPTIX_CHECK(optixModuleCreate(_optix_ctx.get(), &_module_comp_options, &_pipeline_comp_options, ptx,
									  devicePrograms_ptx_len, &log[0], &log_size, &_module));
		OPTIX_LOG(log, log_size);

		// ----------------------------------- Raygen programs ----------------------------------- //

		OptixProgramGroupOptions pg_opts = {};
		OptixProgramGroupDesc pg_desc = {};
		pg_desc.kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
		pg_desc.raygen.module = _module;
		pg_desc.raygen.entryFunctionName = "__raygen__paths";

		OPTIX_CHECK(
			optixProgramGroupCreate(_optix_ctx.get(), &pg_desc, 1, &pg_opts, &log[0], &log_size, &_raygen_program));
		OPTIX_LOG(log, log_size);

		// ----------------------------------- Hit-group programs ----------------------------------- //

		_hitgroup_programs.resize(RAY_TYPE_COUNT);

		pg_desc.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
		pg_desc.hitgroup.moduleCH = _module;
		pg_desc.hitgroup.moduleAH = _module;

		// Indirect path rays
		pg_desc.hitgroup.entryFunctionNameCH = "__closesthit__indirect";
		pg_desc.hitgroup.entryFunctionNameAH = "__anyhit__indirect";
		OPTIX_CHECK(optixProgramGroupCreate(_optix_ctx.get(), &pg_desc, 1, &pg_opts, &log[0], &log_size,
											&_hitgroup_programs[RAY_TYPE_INDIRECT]));
		OPTIX_LOG(log, log_size);

		// Shadow rays
		pg_desc.hitgroup.entryFunctionNameCH = "__closesthit__shadow";
		pg_desc.hitgroup.entryFunctionNameAH = "__anyhit__shadow";
		OPTIX_CHECK(optixProgramGroupCreate(_optix_ctx.get(), &pg_desc, 1, &pg_opts, &log[0], &log_size,
											&_hitgroup_programs[RAY_TYPE_SHADOW]));
		OPTIX_LOG(log, log_size);

		// ----------------------------------- Miss programs ----------------------------------- //

		_miss_programs.resize(RAY_TYPE_COUNT);

		pg_desc.kind = OPTIX_PROGRAM_GROUP_KIND_MISS;
		pg_desc.miss.module = _module;

		// Indirect path rays
		pg_desc.miss.entryFunctionName = "__miss__indirect";
		OPTIX_CHECK(optixProgramGroupCreate(_optix_ctx.get(), &pg_desc, 1, &pg_opts, &log[0], &log_size,
											&_miss_programs[RAY_TYPE_INDIRECT]));
		OPTIX_LOG(log, log_size);

		// Shadow rays
		pg_desc.miss.entryFunctionName = "__miss__shadow";
		OPTIX_CHECK(optixProgramGroupCreate(_optix_ctx.get(), &pg_desc, 1, &pg_opts, &log[0], &log_size,
											&_miss_programs[RAY_TYPE_SHADOW]));
		OPTIX_LOG(log, log_size);
	}

	void OptixEngine::processAssets(SceneData scene)
	{
		// ------------------------ Materials ------------------------ //

		_material_indices = std::move(scene.mat_ids);
		_materials.upload(scene.materials);

		// ------------------------- Antennas ------------------------- //

		_antenna_models.upload(scene.antenna_models);
		_antenna_gains.upload(scene.antenna_gains);

		// ------------------------- Geometry ------------------------- //

		_mesh_indices = std::move(scene.mesh_ids);
		_meshes = std::move(scene.meshes);
		_vertices.upload(scene.vertices);
		_indeces.upload(scene.indices);

		// ------------- Geometry Acceleration Structures ------------- //

		// One GAS per mesh (not one combined GAS for the whole scene) — each mesh's geometry
		// stays in its own local space, so it can be instanced independently, at its own
		// transform, per target via the IAS built in buildOrRefitIAS.

		// Enable OPTIX_GEOMETRY_FLAG_DISABLE_TRIANGLE_FACE_CULLING if we implement transmission.
		const uint32_t tri_flags[1] = {OPTIX_GEOMETRY_FLAG_DISABLE_ANYHIT}; // opaque => faster

		const auto mesh_count = static_cast<uint32_t>(_meshes.size());
		_gas_handles.resize(mesh_count);
		_gas_data.reserve(mesh_count);
		for (uint32_t i = 0; i < mesh_count; ++i)
			_gas_data.emplace_back(_stream.get());

		for (uint32_t i = 0; i < mesh_count; ++i)
		{
			OptixBuildInput bi = {};
			bi.type = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;

			auto& tri = bi.triangleArray;
			tri.vertexFormat = OPTIX_VERTEX_FORMAT_FLOAT3;
			tri.vertexStrideInBytes = sizeof(float3);
			tri.numVertices = static_cast<uint32_t>(_vertices.size());
			const auto vertex_buffer = _vertices.ptr(); // vertex indices are 'global'
			tri.vertexBuffers = &vertex_buffer; // array of 1 ptr (1 motion key)

			tri.indexFormat = OPTIX_INDICES_FORMAT_UNSIGNED_INT3;
			tri.indexStrideInBytes = sizeof(uint3);
			tri.numIndexTriplets = _meshes[i].triangle_count;
			tri.indexBuffer = _indeces.offset(_meshes[i].index_buffer_index);

			tri.flags = &tri_flags[0];
			tri.numSbtRecords = 1; // one hit record per mesh
			tri.primitiveIndexOffset = _meshes[i].index_buffer_index;

			OptixAccelBuildOptions opts = {};
			opts.buildFlags = OPTIX_BUILD_FLAG_ALLOW_COMPACTION | OPTIX_BUILD_FLAG_PREFER_FAST_TRACE;
			opts.operation = OPTIX_BUILD_OPERATION_BUILD;

			OptixAccelBufferSizes sizes;
			OPTIX_CHECK(optixAccelComputeMemoryUsage(_optix_ctx.get(), &opts, &bi, 1, &sizes));

			DeviceBuffer<uint8_t> temp(_stream.get()), output_temp(_stream.get());
			DeviceBuffer<uint64_t> compact_size_buf(_stream.get());
			temp.reserve(sizes.tempSizeInBytes);
			output_temp.reserve(sizes.outputSizeInBytes);
			compact_size_buf.reserve(1);

			OptixAccelEmitDesc emit = {};
			emit.type = OPTIX_PROPERTY_TYPE_COMPACTED_SIZE;
			emit.result = compact_size_buf.ptr();

			OPTIX_CHECK(optixAccelBuild(_optix_ctx.get(), _stream.get(), &opts, &bi, 1, temp.ptr(),
										sizes.tempSizeInBytes, output_temp.ptr(), sizes.outputSizeInBytes,
										&_gas_handles[i], &emit, 1));

			uint64_t compact_size = 0;
			compact_size_buf.download(compact_size);

			_gas_data[i].reserve(compact_size);
			OPTIX_CHECK(optixAccelCompact(_optix_ctx.get(), _stream.get(), _gas_handles[i], _gas_data[i].ptr(),
										  compact_size, &_gas_handles[i]));

			temp.free();
			output_temp.free();
			compact_size_buf.free();
		}

		CUDA_CHECK(cudaStreamSynchronize(_stream.get()));
	}

	void OptixEngine::createSbt()
	{
		// ----------------------------------- Build the SBT ----------------------------------- //

		// Build raygen records
		{
			RaygenRecord raygen_record{};
			OPTIX_CHECK(optixSbtRecordPackHeader(_raygen_program, &raygen_record.header));
			_raygen_record.upload(raygen_record);
			_sbt.raygenRecord = _raygen_record.ptr();
		}

		// Build miss records
		std::vector<MissRecord> miss_records;
		for (auto& _miss_program : _miss_programs)
		{
			MissRecord rec{};
			OPTIX_CHECK(optixSbtRecordPackHeader(_miss_program, &rec.header));
			miss_records.push_back(rec);
		}
		_miss_records.upload(miss_records);
		_sbt.missRecordBase = _miss_records.ptr();
		_sbt.missRecordCount = static_cast<unsigned int>(_miss_records.size());
		_sbt.missRecordStrideInBytes = sizeof(MissRecord);

		// Build hitgroup records
		std::vector<HitgroupRecord> hitgroup_records;
		for (const auto& _ : _meshes)
		{
			for (size_t rayID = 0; rayID < RAY_TYPE_COUNT; rayID++)
			{
				HitgroupRecord rec{};
				OPTIX_CHECK(optixSbtRecordPackHeader(_hitgroup_programs[rayID], &rec.header));
				// We currently don't upload any per-mesh data as OptiX provides the necessary data in the CH program.
				hitgroup_records.push_back(rec);
			}
		}
		_hitgroup_records.upload(hitgroup_records);
		_sbt.hitgroupRecordBase = _hitgroup_records.ptr();
		_sbt.hitgroupRecordCount = static_cast<unsigned int>(_hitgroup_records.size());
		_sbt.hitgroupRecordStrideInBytes = sizeof(HitgroupRecord);

		// Ensure all records have been uploaded successfully.
		CUDA_CHECK(cudaStreamSynchronize(_stream.get()));
	}

	std::unique_ptr<ThreadContext> OptixEngine::makeThreadContext()
	{
		// Set the CUDA context for this thread.
		//
		// This assumes that the thread context is made on the thread using
		// the context. This'll need to be done slightly differently if that's
		// not the case in the future.
		CUDA_RES_CHECK(cuCtxSetCurrent(_cuda_ctx.get()));


		// Create the thread-local context.
		// This internally creates the thread-local CUDA stream.
		auto ctx = std::make_unique<OptixThreadContext>();

		// ----------------------------------- Build the pipeline ----------------------------------- //

		std::vector<OptixProgramGroup> program_groups;
		program_groups.push_back(_raygen_program);
		for (auto* pg : _hitgroup_programs)
			program_groups.push_back(pg);
		for (auto* pg : _miss_programs)
			program_groups.push_back(pg);

		char log[2048];
		size_t log_size = sizeof(log);
		OPTIX_CHECK(optixPipelineCreate(_optix_ctx.get(), &_pipeline_comp_options, &_pipeline_link_options,
										program_groups.data(), static_cast<unsigned int>(program_groups.size()),
										&log[0], &log_size, &ctx->pipeline));
		OPTIX_LOG(log, log_size);

		OPTIX_CHECK(optixPipelineSetStackSize(/* [in] The pipeline to configure the stack size for */
											  ctx->pipeline,
											  /* [in] The direct stack size requirement for direct
												 callables invoked from IS or AH. */
											  2 * 1024,
											  /* [in] The direct stack size requirement for direct
												 callables invoked from RG, MS, or CH.  */
											  2 * 1024,
											  /* [in] The continuation stack requirement. */
											  2 * 1024,
											  /* [in] The maximum depth of a traversable graph
												 passed to trace. */
											  8));
		OPTIX_LOG(log, log_size);

		return ctx;
	}

	/// Builds a target's OptixInstance::transform (row-major 3x4, world = R * local + T) from its
	/// position (translation) and az/el rotation. SVec3 only carries azimuth+elevation (a pointing
	/// direction, like antenna boresight) — there's no roll/bank component, so this assumes roll
	/// = 0. That's the best this data can give until the planned transform-hierarchy rework lands.
	///
	/// Mesh local-axis convention: +Y is "forward" (nose direction), +Z is "up", +X is "right" —
	/// there's no other convention already established in this codebase to match against, so
	/// meshes must be authored this way or they'll appear rotated.
	static void computeInstanceTransform(const radar::Target* target, const RealType t, float transform[12])
	{
		const auto pos = target->getPosition(t);
		const auto rot = target->getRotation(t);

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

		transform[0] = static_cast<float>(right.x);
		transform[1] = static_cast<float>(forward.x);
		transform[2] = static_cast<float>(up.x);
		transform[3] = static_cast<float>(pos.x);

		transform[4] = static_cast<float>(right.y);
		transform[5] = static_cast<float>(forward.y);
		transform[6] = static_cast<float>(up.y);
		transform[7] = static_cast<float>(pos.y);

		transform[8] = static_cast<float>(right.z);
		transform[9] = static_cast<float>(forward.z);
		transform[10] = static_cast<float>(up.z);
		transform[11] = static_cast<float>(pos.z);
	}

	OptixTraversableHandle OptixEngine::buildOrRefitIAS(OptixThreadContext& ctx, core::World* world, RealType t)
	{
		// Host-side, every call: cheap (no device work), so topology changes are caught rather
		// than assumed away.
		std::vector<OptixInstance> instances;
		std::vector<uint32_t> topology;

		for (const auto& target : world->getTargets())
		{
			const auto& geom = target->getGeometry();
			if (!geom.has_value())
				continue;

			const auto mesh_idx = _mesh_indices.at(geom->mesh->id);
			const auto mat_idx = _material_indices.at(geom->material->id);

			OptixInstance inst{};
			// instanceId currently corresponds to the material index of the object.
			// Once there's more per-instance data to track, this should be converted to an index
			// into an instance data buffer (containing the material).
			inst.instanceId = mat_idx;
			inst.sbtOffset = mesh_idx * RAY_TYPE_COUNT;
			inst.visibilityMask = 0xFF;
			inst.flags = OPTIX_INSTANCE_FLAG_NONE;
			inst.traversableHandle = _gas_handles[mesh_idx];
			// NOLINTNEXTLINE(cppcoreguidelines-pro-bounds-array-to-pointer-decay): OptiX C API uses arrays.
			computeInstanceTransform(target.get(), t, inst.transform);

			instances.push_back(inst);
			topology.push_back(mesh_idx);
		}

		const bool topology_changed = topology != ctx.ias_topology;

		OptixBuildInput ias_input{};
		ias_input.type = OPTIX_BUILD_INPUT_TYPE_INSTANCES;

		OptixAccelBuildOptions opts{};
		// No compaction: unlike the static per-mesh GASes, this is rebuilt/refit every call, so
		// compaction overhead isn't worth it.
		opts.buildFlags = OPTIX_BUILD_FLAG_ALLOW_UPDATE;

		if (topology_changed)
		{
			ctx.ias_instances.upload(instances);

			ias_input.instanceArray.instances = ctx.ias_instances.ptr();
			ias_input.instanceArray.numInstances = static_cast<unsigned int>(instances.size());

			opts.operation = OPTIX_BUILD_OPERATION_BUILD;

			OPTIX_CHECK(optixAccelComputeMemoryUsage(_optix_ctx.get(), &opts, &ias_input, 1, &ctx.ias_sizes));

			ctx.ias_build_temp.reserve(std::max(ctx.ias_sizes.tempSizeInBytes, ctx.ias_sizes.tempUpdateSizeInBytes));
			ctx.ias_output.reserve(ctx.ias_sizes.outputSizeInBytes);

			OPTIX_CHECK(optixAccelBuild(_optix_ctx.get(), ctx.stream.get(), &opts, &ias_input, 1,
										ctx.ias_build_temp.ptr(), ctx.ias_sizes.tempSizeInBytes, ctx.ias_output.ptr(),
										ctx.ias_sizes.outputSizeInBytes, &ctx.ias_handle, nullptr, 0));

			ctx.ias_topology = std::move(topology);
		}
		else
		{
			ctx.ias_instances.upload(instances);

			ias_input.instanceArray.instances = ctx.ias_instances.ptr();
			ias_input.instanceArray.numInstances = static_cast<unsigned int>(instances.size());

			opts.operation = OPTIX_BUILD_OPERATION_UPDATE;

			// TODO_SHAUN a large time jump between refits leaves the IAS's tree structure fit to
			// the old transforms, degrading trace quality (still correct, just slower) — consider
			// forcing a full rebuild above some time-delta threshold rather than always refitting.
			OPTIX_CHECK(optixAccelBuild(_optix_ctx.get(), ctx.stream.get(), &opts, &ias_input, 1,
										ctx.ias_build_temp.ptr(), ctx.ias_sizes.tempUpdateSizeInBytes,
										ctx.ias_output.ptr(), ctx.ias_sizes.outputSizeInBytes, &ctx.ias_handle, nullptr,
										0));
		}

		return ctx.ias_handle;
	}

	std::vector<Contribution> OptixEngine::trace(ThreadContext* thread_context, TraceJob& job)
	{
		auto* ctx = dynamic_cast<OptixThreadContext*>(thread_context);
		if (ctx == nullptr)
			throw std::runtime_error("OptixEngine::trace did not receive a valid OptixThreadContext.");

		const auto rays_per_source = params::rtModelParams().raysPerSource();
		const auto x_rays_per_source = rays_per_source;
		const auto y_source_antennas = static_cast<uint32_t>(job.source_antennas.size());
		const auto z_times = static_cast<uint32_t>(job.times.size());

		DeviceBuffer<ActiveAntenna> source_antennas(_stream.get());
		source_antennas.upload(job.source_antennas);
		DeviceBuffer<ActiveAntenna> dest_antennas(_stream.get());
		dest_antennas.upload(job.dest_antennas);
		DeviceBuffer<double> times(_stream.get());
		times.upload(job.times);
		DeviceBuffer<CarrierModel<float>> carriers(_stream.get());
		carriers.upload(job.carriers);
		DeviceBuffer<RxFlags> rx_flags(_stream.get());
		rx_flags.upload(job.rx_flags);

		std::vector<OptixTraversableHandle> iass;
		iass.reserve(job.times.size());
		for (auto t : job.times)
			iass.push_back(buildOrRefitIAS(*ctx, job.world, t));
		ctx->iass.upload(iass);

		// Reserve lots of space for return contributions.
		//
		// The number of contribtions is going to be equal to (where E[.] is the expectation function):
		// |sources| * |rays_per_source| * E[facet->dest is unshadowed] * |dests| * E[steps before hitting nothing]
		// where
		// - `E[facet->dest is unshadowed]` is likely to be under half
		//      (half the triangles on average will face away from a random point, some face the point but are occluded)
		// - `E[steps before hitting nothing]` is likely to be some tiny nonlinearly scaling proportion of MAX_GO_STEPS
		//
		// These `E[facet->dest is unshadowed] * E[steps before hitting nothing]` are estimated to be bound by
		// `MAX_GO_STEPS/depreciation_factor`.
		const size_t depreciation_factor = 64;
		ctx->contribs.reserve(source_antennas.size() * dest_antennas.size() * rays_per_source * MAX_RAY_DEPTH /
							  depreciation_factor);

		ShaderParams params{.iass = ctx->iass.ptr_t(),
							.contribution_count = 0,
							.contribution_capacity = static_cast<uint32_t>(ctx->contribs.capacity()),
							.sbr{
								.scatter_limit = params::rtModelParams().scatter_limit,
								.rays_per_source = rays_per_source,
								.boresight_rays = params::rtModelParams().raysAtBoresightCap(),
								.boresight_fraction = float(params::rtModelParams().boresightSolidAngle() / (4.0 * PI)),
								.vertices = _vertices.ptr_t(),
								.indices = _indeces.ptr_t(),
								.materials = _materials.ptr_t(),
								.times = times.ptr_t(),
								.carrier_models = carriers.ptr_t(),
								.antenna_models = _antenna_models.ptr_t(),
								.antenna_gains = _antenna_gains.ptr_t(),
								.source_antennas = source_antennas.ptr_t(),
								.source_antenna_count = static_cast<uint32_t>(source_antennas.size()),
								.dest_antennas = dest_antennas.ptr_t(),
								.dest_antenna_count = static_cast<uint32_t>(dest_antennas.size()),
								.rx_flags = rx_flags.ptr_t(),
								.rx_to_tx = job.type == TraceJobType::FindRxFromTx,
								.contributions = ctx->contribs.ptr_t(),
							}};

		ctx->params.upload(params);

		OPTIX_CHECK(optixLaunch(ctx->pipeline, ctx->stream.get(),
								/*! parameters and SBT */
								ctx->params.ptr(), ctx->params.size_bytes(), &_sbt,
								/*! dimensions of the launch: */
								x_rays_per_source, y_source_antennas, z_times));


		// Read back the device-side contribution count, so we only download what was actually
		// written. `ctx->params` was uploaded with `contribution_count = 0`; the kernel increments
		// its own device-side copy via `fetch_add`, so we must download that copy, not reuse `params`.
		ShaderParams readback{};
		ctx->params.download(readback);

		CUDA_CHECK(cudaStreamSynchronize(ctx->stream.get()));

		// Clamp defensively: if the device wrote more contributions than `ctx->contribs` was
		// reserved for (see the `depreciation_factor` sizing estimate above), then the kernel
		// began overwriting contributions, not writing more.
		// Only download at most the full buffer, not the write count.
		const auto contribution_count =
			std::min<size_t>(readback.contribution_count.load(cuda::memory_order_relaxed), ctx->contribs.capacity());
		// TODO_SHAUN grow/reallocate `ctx->contribs` and re-launch instead of dropping contributions past capacity?

		std::vector<Contribution> contribs;
		ctx->contribs.set_size(contribution_count);
		ctx->contribs.download(contribs);
		CUDA_CHECK(cudaStreamSynchronize(ctx->stream.get()));

		return contribs;
	}
}
