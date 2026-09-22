#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cuda_runtime_api.h>
#include <optix_device.h>
#include <optix_types.h>
#include <vector_types.h>

#include "propagation/math.h"
#include "propagation/raytracing/optix/common.h"
#include "propagation/raytracing/sbr_impl.h"

// Get rid of the warnings about the names starting with two underscores.
// NOLINTBEGIN(bugprone-reserved-identifier)

namespace propagation::raytracing::optix
{
	// -------------------------------------------------------------------------------- //
	// Utilities
	// -------------------------------------------------------------------------------- //

	template <typename T>
	static constexpr inline __device__ void packPointer(T* ptr, unsigned int* hi, unsigned int* lo)
	{
		auto bits = reinterpret_cast<uintptr_t>(ptr);
		*lo = static_cast<unsigned int>(bits);
		*hi = static_cast<unsigned int>(bits >> sizeof(unsigned int) * 8);
	}

	template <typename T>
	static constexpr inline __device__ T* unpackPointer(unsigned int hi, unsigned int lo)
	{
		auto bits = static_cast<uintptr_t>(lo) + (static_cast<uintptr_t>(hi) << (sizeof(unsigned int) * 8));
		return reinterpret_cast<T*>(bits);
	}

	static constexpr inline __device__ float3 asF32(Double3& v)
	{
		return float3{static_cast<float>(v.x), static_cast<float>(v.y), static_cast<float>(v.z)};
	}

	/**
	 * @brief Build a linalg.h affine 3x4 transform from an OptiX transform float array.
	 */
	static constexpr inline __device__ Float3x4 fromOptixTransform(const float t[12])
	{
		// OptiX dumps it's matrix as row-major list.
		// Float3x4 is currently column-major.
		return Float3x4{
			Float3{t[0], t[4], t[8]},
			Float3{t[1], t[5], t[9]},
			Float3{t[2], t[6], t[10]},
			Float3{t[3], t[7], t[11]},
		};
	}


	// -------------------------------------------------------------------------------- //
	// Shadow/Occlusion rays
	// -------------------------------------------------------------------------------- //

	/// Launch parameters in constant memory, filled in by optix upon optixLaunch.
	extern "C" __constant__ ShaderParams shader_params;


	extern "C" __global__ void __closesthit__shadow()
	{
		unsigned int hi = optixGetPayload_0(), lo = optixGetPayload_1();
		auto* shadow_t = unpackPointer<float>(hi, lo);
		*shadow_t = optixGetRayTmax();
	}

	extern "C" __global__ void __anyhit__shadow() {}

	extern "C" __global__ void __miss__shadow() {}


	static __device__ inline float shadowTest(Float3 origin, Float3 segment, OptixTraversableHandle ias)
	{
		// Default to unoccluded. Occluded results in this value ending up <1f.
		float shadow_t = INFINITY;

		unsigned int hi{}, lo{};
		packPointer(&shadow_t, &hi, &lo);
		optixTrace(ias, // AS handle
				   float3{origin.x, origin.y, origin.z}, // rayOrigin
				   float3{segment.x, segment.y, segment.z}, // rayDirection
				   1e-3f, // tmin
				   INFINITY, // tmax
				   0.0f, // rayTime
				   OptixVisibilityMask(0xFF), // visibilityMask TODO?
				   OPTIX_RAY_FLAG_DISABLE_ANYHIT, // rayFlags
				   RAY_TYPE_SHADOW, // SBT offset
				   RAY_TYPE_COUNT, // SBT stride
				   RAY_TYPE_SHADOW, // missSBTIndex
				   hi, lo); // payload; up to 32 unsigned ints

		return shadow_t;
	}

	static __device__ inline uint32_t contributionIndex()
	{
		return min(shader_params.contribution_count.fetch_add(1), shader_params.contribution_capacity - 1);
	}

	// -------------------------------------------------------------------------------- //
	// Geometric Optics (GO) rays
	// -------------------------------------------------------------------------------- //

	/// This is going to be the Geometric Optics (GO) closest hit program.
	///
	/// This is where the fun stuff is going to happen.
	extern "C" __global__ void __closesthit__go()
	{
		const uint32_t material_index = optixGetInstanceId();
		const uint32_t triangle_index = optixGetPrimitiveIndex();
		const float2 o_bary = optixGetTriangleBarycentrics();

		HitInfo hit{};
		hit.triangle_index = triangle_index;
		hit.material_index = material_index;
		hit.bary_u = o_bary.x;
		hit.bary_v = o_bary.y;

		float world_to_obj_raw[12];
		float obj_to_world_raw[12];
		optixGetWorldToObjectTransformMatrix(world_to_obj_raw);
		optixGetObjectToWorldTransformMatrix(obj_to_world_raw);

		hit.world_to_obj = fromOptixTransform(world_to_obj_raw);
		hit.obj_to_world = fromOptixTransform(obj_to_world_raw);

		unsigned int hi = optixGetPayload_0(), lo = optixGetPayload_1();
		auto* path = unpackPointer<PathState>(hi, lo);

		auto ias = shader_params.iass[path->time_index];

		auto should_reflect = facetIncidentRay(
			shader_params.shared, path, hit, [ias](Float3 origin, Float3 seg) { return shadowTest(origin, seg, ias); },
			contributionIndex);

		if (!should_reflect)
			return; // we're done with this path

		optixTrace(ias, // AS handle
				   asF32(path->origin), // rayOrigin
				   asF32(path->direction), // rayDirection
				   1e-3f, // tmin TODO?
				   INFINITY, // tmax
				   0.0f, // rayTime
				   OptixVisibilityMask(0xFF), // visibilityMask
				   OPTIX_RAY_FLAG_DISABLE_ANYHIT, // rayFlags
				   RAY_TYPE_GO, // SBT offset
				   RAY_TYPE_COUNT, // SBT stride
				   RAY_TYPE_GO, // missSBTIndex
				   hi, lo // payload; up to 32 unsigned ints
		);
	}

	/// This is the Geometric Optics (GO) any-hit program.
	/// This does nothing because we only care about the closest hit.
	/// Any-hit checks get disabled in a number of places.
	extern "C" __global__ void __anyhit__go() {}

	/// This is going to be the Geometric Optics (GO) miss program.
	/// The radar energy has zoomed off into space. Goodbye energy. Stop tracing.
	extern "C" __global__ void __miss__go() {}


	/// This function is where we're going to for each radar "source"
	/// (transmitter or receiver antenna, depending on findTxToRxPaths or findRxFromTxPaths)
	/// and each time step and generate lots of rays to trace through the scene and
	/// determine contributions to each of the "sinks" (receivers or transmitters, respectively).
	extern "C" __global__ void __raygen__antennaeOverTimes()
	{
		uint3 idx = optixGetLaunchIndex();
		/// The ray/path index to launch.
		unsigned int path_index = idx.x;
		/// The "source" antenna to launch from. Could be a transmitter or receiver.
		unsigned int antenna_index = idx.y;
		/// The time-step index. Determines which IAS to use.
		unsigned int t_index = idx.z;

		auto ias = shader_params.iass[t_index];

		// Zero-bounce direct-path check, piggybacked on spare X-lanes (dest_antenna_count is tiny).
		// Looped, not `if`, so it stays correct even if rays_per_source < dest_antenna_count.
		for (uint32_t d = path_index; d < shader_params.shared.dest_antenna_count;
			 d += shader_params.shared.rays_per_source)
		{
			directPath(
				shader_params.shared, antenna_index, d, t_index,
				[ias](Float3 origin, Float3 seg) { return shadowTest(origin, seg, ias); }, contributionIndex);
		}

		PathState path{};
		initPath(shader_params.shared, path_index, antenna_index, t_index, path);

		unsigned int hi{}, lo{};
		packPointer(&path, &hi, &lo);
		optixTrace(ias, // AS handle
				   asF32(path.origin), // rayOrigin
				   asF32(path.direction), // rayDirection
				   1e-3f, // tmin
				   INFINITY, // tmax
				   0.0f, // rayTime
				   OptixVisibilityMask(0xFF), // visibilityMask
				   OPTIX_RAY_FLAG_DISABLE_ANYHIT, // rayFlags
				   RAY_TYPE_GO, // SBT offset
				   RAY_TYPE_COUNT, // SBT stride
				   RAY_TYPE_GO, // missSBTIndex
				   hi, lo // payload; up to 32 unsigned ints
		);
	}
}

// NOLINTEND(bugprone-reserved-identifier)
