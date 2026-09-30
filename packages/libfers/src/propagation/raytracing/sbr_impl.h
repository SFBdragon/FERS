// SPDX-License-Identifier: GPL-2.0-only
//
// Copyright (c) 2026-present FERS Contributors (see AUTHORS.md).
//
// See the GNU GPLv2 LICENSE file in the FERS project root for more information.

#pragma once

#include <array>
#include <cassert>
#include <cmath>
#include <cstdint>

#include "antenna/antenna_gain.h"
#include "propagation/common.h"
#include "propagation/common_defs.h"
#include "propagation/math.h"
#include "propagation/raytracing/sbr_shared.h"

#ifndef HC_FN
#if defined(__CUDACC__) || defined(__HIPCC__)
#define HC_FN __host__ __device__ inline
#else
#define HC_FN inline
#endif
#endif

// This file defines most of the propagation logic for the RayTracingModel propagation model.
//
// ## On Precision
//
// Floating point precision is worth careful consideration in FERS.
// For example, the difference of a facet 1cm wide verses 2cm is important, even 200km away.
//
// A number of bounding facts are considered in the approach used:
// - Double-precision is valuable for cases where a very big number and a very small number are added together,
//      and the small number must retain a high degree of precision.
// - OptiX and similar engines trace against 32-bit precision geometry (local mesh space).
// - This geometry remains in local space (centered at its local origin), so this precision issue is minimal.
// - FERS may be used for very long-distance radar, e.g. ground-to-space, i.e. 100s of kilometers between objects.
// - This means that global spacial data and geometry must be in double-precision.
// - However, normals often don't need to be in double-precision - they're unit-magnitude.
// - Complex math operations (e.g. sqrt, trig) on GPUs have significant performance gains at 32-bit precision.
// - Similarly, most physical constants are only known up to roughly single-precision (~11d.p.)
//      e.g. Z0, E0, material properties, and C by definition.
// - The carrier frequencies in use are often narrowband approximations to far worse accuracy than 32-bit precision.
// - Antenna gains are linear, often between 0-1, and are often approximated by interpolation anyway.
//
// As such, the following approach is taken to choices of precision:
// - Local geometry is in single-precision.
// - Global spatial positions and lengths is kept in double-precision.
// - Unit vectors' precision is generally chosen to match the precision of the other data in operations.
// - AzEl orientations are in 32-bit precision.
// - Carrier frequencies are in 32-bit precision.
// - Antenna gains are in 32-bit precision.
// - Material property calculations are in 32-bit precision, and hence so are electric fields.

namespace propagation::raytracing
{
	struct RadiationCache
	{
		/// Electric field-oriented unitless radiation vector. Carrier-specific.
		/// Perpendicular to direction of propagation.
		///
		/// Note that the units here are a bit interesting.
		/// far-field E(r-hat, R) = sqrt(Z0/4/pi * power) * sqrt(gain) * unit-polarisation(r-hat) * e^(-jkR) / R
		/// The radiation vector consist only of this   :   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
		/// `sqrt(power)`: Signal power is applied later by FERS.
		/// `e^(-jkR)`: The bulk phase term (after down-converting) is computed later by FERS.
		/// `1/R`: The spherical spreading term is applied at the end of the path. See `facetIncidentRay`
		/// at conversion to voltage at end of path.
		/// `sqrt(Z0/4/pi)` cancels out? TODO_SHAUN CHECK THIS
		///
		/// The units of this quantity is therefore dimensionless.
		/// The point of it is to pick up all the voltage attenuation and phase effects
		/// of the path while being proportional to the electric field.
		/// All geometric optics and physical optics physics performed is linear with
		/// respect to the electric field, and thus the results remain physical. std::array<CFloat3, MAX_CARRIERS>
		CFloat3 radiation{};
		/// Similar to the above, this is actually unit-less.
		/// TODO_SHAUN describe
		CFloat3 surface_current{};
		/// The carrier frequency for which the cached radiation and surface current are valid.
		float carrier_frequency = -1.0f;
		/// The depth for which the cached radiation and surface current were computed for.
		uint32_t depth = 0;
	};

	/**
	 * @brief The ray state. This is updated at each step.
	 *
	 * All spacial properties are in FERS's global simulation space, cartesian, in meters.
	 */
	struct PathState
	{
		// Launch information.

		/// Which radar source antenna this ray originates from, by index.
		uint32_t source_index{};
		/// Which time this ray originates from, by index.
		uint32_t time_index{};
		/// The "weight" of the ray tube, in steradians. (solid
		/// angle/pdf)
		float weight{};
		/// The antenna-local far-field radiation direction. Used to
		/// sample the antenna gain/pol.
		Float3 local_emmission_direction;


		// Path information.

		/// The number of path segments found along the GO path thus far.
		uint32_t ray_count = 0;
		/// Path vertices or hit points, positions, in meters.
		///
		/// Contains `ray_count + 1` vertices.
		std::array<Double3, MAX_RAY_DEPTH> path_vertices;
		/// Unit directions from path vertex i to i+1 in trace direction.
		///
		/// Contains `ray_count + 1` unit directions.
		std::array<Float3, MAX_RAY_DEPTH> trace_directions;
		/// Unit normals at hit surfaces.
		///
		/// There are `ray_count` unit normals.
		/// Normal 0 corresponds to Path Vertex 1.
		std::array<Float3, MAX_RAY_DEPTH - 1> surface_normals;
		/// Materials of hit surfaces.
		///
		/// There are `ray_count` materials.
		/// Material 0 corresponds to Path Vertex 1.
		std::array<const Material*, MAX_RAY_DEPTH - 1> surface_materials{};
		/// The accumulative path length.
		double length = 0.0;
		/// The hash of the path so far, unique per facet sequence.
		uint64_t hash = HASH_SEED;

		///
		RadiationCache cache;
	};

	struct HitInfo
	{
		/// The triangle index that was hit.
		uint32_t triangle_index{};
		/// The hit objects surface material.
		uint32_t material_index{};
		/// The barycentric hit posititon u coordinate.
		float bary_u{};
		/// The barycentric hit posititon v coordinate.
		float bary_v{};
		/// The hit object's world to local transformation (affine 3D matrix with translation).
		Float3x4 world_to_obj;
		/// The hit object's local to world transformation (affine 3D matrix with translation).
		Float3x4 obj_to_world;
	};

	/// Bilinearly samples an `AntennaPatternView` at a local (azimuth, elevation), clamping
	/// both axes to the grid's covered range.
	///
	/// Clamping (rather than wrapping) azimuth is correct here because the grid already samples the
	/// full circle inclusive at both ends (az=-pi and az=+pi are both present, redundantly), so
	/// clamping at the edges reproduces the same behavior wrapping would.
	HC_FN float sampleAntennaPattern(const SbrParams& params, const AntennaPatternView& pattern, float azimuth,
									 float elevation)
	{
		assert(pattern.az_count >= 2);
		assert(pattern.el_count >= 2);

		const auto az_max = static_cast<float>(pattern.az_count - 1);
		const auto el_max = static_cast<float>(pattern.el_count - 1);

		float az_frac = (azimuth + PI_V<float>) / (2.0f * PI_V<float>)*az_max;
		float el_frac = (elevation + PI_V<float> / 2.0f) / PI_V<float> * el_max;
		az_frac = az_frac < 0.0f ? 0.0f : (az_frac > az_max ? az_max : az_frac);
		el_frac = el_frac < 0.0f ? 0.0f : (el_frac > el_max ? el_max : el_frac);

		const auto az0 = static_cast<uint32_t>(az_frac);
		const auto el0 = static_cast<uint32_t>(el_frac);
		const uint32_t az1 = az0 + 1 < pattern.az_count ? az0 + 1 : az0;
		const uint32_t el1 = el0 + 1 < pattern.el_count ? el0 + 1 : el0;

		const auto taz = az_frac - static_cast<float>(az0);
		const auto tel = el_frac - static_cast<float>(el0);

		const float g00 = params.antenna_gains[pattern.gains_buffer_offset + el0 * pattern.az_count + az0];
		const float g01 = params.antenna_gains[pattern.gains_buffer_offset + el0 * pattern.az_count + az1];
		const float g10 = params.antenna_gains[pattern.gains_buffer_offset + el1 * pattern.az_count + az0];
		const float g11 = params.antenna_gains[pattern.gains_buffer_offset + el1 * pattern.az_count + az1];

		const float g0 = g00 + (g01 - g00) * taz;
		const float g1 = g10 + (g11 - g10) * taz;
		return g0 + (g1 - g0) * tel;
	}

	/**
	 * @brief Evaluates an `AntennaModel` at a local direction, returning linear gain (efficiency
	 * already applied).
	 *
	 * @param local_dir The local-frame direction (unit vector; see `AzEl`), used directly for the
	 * rotationally-symmetric analytic kinds (`Sinc`/`SquareHorn`/`Parabolic`: angle off boresight is
	 * `acos(local_dir.x)`, since boresight is local +X by construction - cheaper than round-tripping
	 * through `local_az`/`local_el`).
	 * @param local_az, local_el As used by `Gaussian` (directly, as the az/el deltas from boresight)
	 * and by the tabulated kinds (`Grid2D`).
	 * @param wavelength Only used by the wavelength-dependent kinds (`SquareHorn`/`Parabolic`).
	 */
	HC_FN float sampleAntennaModel(const SbrParams& params, const AntennaModel& model, const Float3& local_dir,
								   float local_az, float local_el, float wavelength)
	{
		const float boresight_cos = local_dir.x < -1.0f ? -1.0f : (local_dir.x > 1.0f ? 1.0f : local_dir.x);

		float pattern_gain = 0.0;
		switch (model.kind)
		{
		case AntennaKind::Isotropic:
			pattern_gain = 1.0;
			break;
		case AntennaKind::Sinc:
			pattern_gain = antenna::gain::sincGain<float>(std::acos(boresight_cos), model.params.sinc.alpha,
														  model.params.sinc.beta, model.params.sinc.gamma);
			break;
		case AntennaKind::Gaussian:
			pattern_gain = antenna::gain::gaussianGain<float>(local_az, local_el, model.params.gaussian.azimuth_scale,
															  model.params.gaussian.elevation_scale);
			break;
		case AntennaKind::SquareHorn:
			pattern_gain = antenna::gain::squareHornGain<float>(std::acos(boresight_cos),
																model.params.square_horn.dimension, wavelength);
			break;
		case AntennaKind::Parabolic:
			pattern_gain = antenna::gain::parabolicGain<float>(std::acos(boresight_cos),
															   model.params.parabolic.diameter, wavelength);
			break;
		case AntennaKind::Grid2D:
			pattern_gain = sampleAntennaPattern(params, model.params.grid_2d.grid, local_az, local_el);
			break;
		}

		return float(pattern_gain * model.efficiency);
	}

	/**
	 * @brief Isotropic direction sampling using the Fibonacci/Vogel spiral point set.
	 * In local coordinates; `path_idx=0` lands near local (1,0,0), sweeping to near (-1,0,0)
	 * as `path_idx` approaches `num_rays-1`.
	 *
	 * This is the isotropic MVP sampler, to be replaced by gain-weighted importance sampling.
	 */
	[[nodiscard]] HC_FN Float3 sampleIsotropicDirection(uint32_t path_idx, uint32_t num_rays)
	{
		// Golden angle = pi * (3 - sqrt(5)), the standard Fibonacci/Vogel-spiral spacing constant.
		const float phi = PI_V<float> * (3.0f - std::sqrt(5.0f));

		const float n = static_cast<float>(num_rays);
		const float i = static_cast<float>(path_idx);
		const float x = 1.0f - (2.0f * i) / n; // x from 1 to -1
		const float r = std::sqrt(1.0f - x * x); // radius at x
		const float theta = phi * i;
		return Float3{x, r * std::cos(theta), r * std::sin(theta)};
	}


	HC_FN void initPath(const SbrParams& params, uint32_t path_idx, uint32_t source_idx, uint32_t time_idx,
						PathState& path_out)
	{
		const ActiveAntenna& antenna = params.source_antennas[source_idx + time_idx * params.source_antenna_count];

		const Float3 local_dir = sampleIsotropicDirection(path_idx, params.rays_per_source);
		const Float3 world_dir = rotateLocalToWorld(antenna.direction, local_dir);

		path_out.ray_count = 0;
		path_out.source_index = source_idx;
		path_out.time_index = time_idx;
		path_out.weight = 4.0f * PI_V<float> / float(params.rays_per_source);
		path_out.local_emmission_direction = local_dir;
		path_out.path_vertices[0] = antenna.position;
		path_out.trace_directions[0] = world_dir;
		path_out.length = 0.0;
		path_out.hash = hash_mix(HASH_SEED, source_idx);
	}

	namespace po
	{
		struct TubePatch
		{
			/// The vertices defining the hit ray tube patch triangle.
			/// The origin of the vertices is the ray hit point, thus these are at local scale.
			/// CCW winding order (right-hand-rule)
			std::array<Float3, 3> verts;
			/// The pre-computed area of the patch defined by `verts`.
			float area;
		};

		/**
		 * @brief Computes the ray tube's PO integration patch at a facet hit: the tube's footprint,
		 * clamped to the triangle's own area, expressed as a triangle scaled toward the hit point.
		 *
		 * Relies on the flat-facet assumption: a flat-mirror reflection of a spherical wave is exactly
		 * another spherical wave from the mirrored virtual source, so the wavefront keeps diverging as if
		 * from the *original* source over the full accumulated path length `dist`, not just the current
		 * segment. This breaks once curved facets are introduced (planned future work) - the
		 * virtual-source distance would then need tracking explicitly instead of being read off `dist`.
		 *
		 * Footprint area: for a tube of solid angle `solid_angle` steradians hitting the facet at
		 * incidence angle theta_i off its normal, the footprint projected onto the facet plane is
		 * `dist^2 * solid_angle / cos(theta_i)` (standard oblique-projection geometry). The returned
		 * patch is then clamped to `min(footprint, tri_area)` by scaling the triangle toward `hit_pos` -
		 * which, since it's a scale toward an interior point, stays inside the parent triangle
		 * automatically, without needing separate edge-clipping.
		 *
		 * @param tri_verts World-space triangle vertices.
		 * @param hit_pos World-space hit point (assumed inside the triangle).
		 * @param norm Outward triangle normal (unit length).
		 * @param incoming_dir Unit direction the ray is travelling in (towards `hit_pos`).
		 * @param tri_area The triangle's own area.
		 * @param dist Total accumulated path length from the ray's original source to this hit.
		 * @param solid_angle The ray tube's solid angle (steradians).
		 */
		HC_FN TubePatch computeTubePatch(const std::array<const Double3, 3>& tri_verts, const Double3& hit_pos,
										 const Float3& u_norm, const Float3& u_inc, float facet_area, float dist,
										 float solid_angle)
		{
			const float cos_theta_i = std::max(dot(-u_inc, u_norm), 1e-3f);
			const float tube_area = dist * dist * solid_angle / cos_theta_i;
			const float scale = std::sqrt(std::min(tube_area, facet_area) / facet_area);

			TubePatch patch{};
			for (size_t vi = 0; vi < 3; vi++)
				patch.verts.at(vi) = scale * Float3(tri_verts.at(vi) - hit_pos);
			patch.area = scale * scale * facet_area;
			return patch;
		}

		/// Numerically stable unnormalised sinc = sin(x)/x with the limit at sin(0) = 1.
		HC_FN float sinc(float x, float tol = 1e-5f) { return fabsf(x) < tol ? 1.0f : sinf(x) / x; }

		/// Gordon's formula for computing the Physical Optics phase integral.
		HC_FN CFloat scaledPhaseIntegral(const TubePatch& patch, const Float3& norm, const Float3& u_inc,
										 const Float3& u_scat, float wavenumber)
		{
			/// Scattering vector, representing phase difference across patch.
			Float3 scatter = wavenumber * (u_inc - u_scat);
			/// In-plane component of the scattering vector.
			Float3 scatter_inplane = scatter - dot(scatter, norm) * norm;
			float scatter_inplane2 = length2(scatter_inplane);

			// Handle the near-specular case.
			// The normalized phase integral is I = A
			if (scatter_inplane2 < 1e-8f)
				return CFloat{patch.area};

			std::array<Float3, 3> delta_vs = {patch.verts[1] - patch.verts[0], patch.verts[2] - patch.verts[1],
											  patch.verts[0] - patch.verts[2]};
			std::array<Float3, 3> mid_vs = {0.5f * (patch.verts[1] + patch.verts[0]),
											0.5f * (patch.verts[2] + patch.verts[1]),
											0.5f * (patch.verts[0] + patch.verts[2])};

			CFloat acc{0.0f};
			for (size_t i = 0; i < 3; i++)
			{
				float re = dot(cross(scatter, norm), delta_vs.at(i)) * sinc(0.5f * dot(scatter, delta_vs.at(i)));
				acc = acc + re * cexp_i(dot(scatter, mid_vs.at(i)));
			}

			// [j k0 / 4 / pi] * [j k0 / |w_t|^2] = -k0^2 / |w_t|^2 / 4 / pi
			float scalars = -wavenumber * wavenumber / scatter_inplane2 * (1.0f / 4.0f / PI_V<float>);
			return acc * scalars;
		}

		HC_FN CFloat3 facetScatteredRadiation(const TubePatch& patch, const Float3& u_norm, const Float3& u_inc,
											  const Float3& u_scat, const CFloat3& surface_current,
											  const float wavenumber)
		{
			const CFloat scaled_phase_integral = scaledPhaseIntegral(patch, u_norm, u_inc, u_scat, wavenumber);
			return cross(u_scat, cross(u_scat, surface_current)) * scaled_phase_integral;
		}
	} // namespace po

	namespace go
	{
		HC_FN void complexSnellsLaw(const Float3& norm, const Float3& k_in, CFloat n2, CFloat& gamma_tm_out,
									CFloat& gamma_te_out)
		{
			// Index of refraction of free space.
			constexpr CFloat N1 = CFloat{1.0, 0.0};

			const float cos_theta_i = -dot(k_in, norm);
			const float sin2_theta_i = 1 - cos_theta_i * cos_theta_i;
			const CFloat cos_theta_t = csqrt(1.0f - (N1 / n2) * sin2_theta_i);

			gamma_tm_out = (n2 * cos_theta_t - N1 * cos_theta_i) / (n2 * cos_theta_t + N1 * cos_theta_i);
			gamma_te_out = (n2 * cos_theta_i - N1 * cos_theta_t) / (n2 * cos_theta_i + N1 * cos_theta_t);
		}

		/// Determine the complex index of refraction of a Material at a given carrier frequency.
		HC_FN CFloat indexOfRefraction(float relative_permittivity, float conductivity, float frequency)
		{
			// 1/sqrt(2). Enough digits to saturate double-precision.
			constexpr float ISQRT2 = 0.70710678118654752440f;

			// Special-case the calculation for dielectrics to avoid a complex sqrt.
			if (conductivity == 0)
				return CFloat{std::sqrt(relative_permittivity), 0.0};

			const float im = conductivity / (E0 * 2 * PI_V<float> * frequency);

			// Special-case the calculation for PECs.
			if (conductivity == INFINITY)
				return CFloat{ISQRT2, -ISQRT2} * std::sqrt(im);

			// Handle the partially-conducting case.
			const CFloat permittivity{relative_permittivity, -im};
			return csqrt(permittivity);
		}


		/// Reflect Cartesian radiation vector off a facet specularly
		/// using geometric optics and Fresnel equations to enforce
		/// interface conditions.
		HC_FN void fresnelGoReflection(CFloat3& radiation, const Float3& norm, const Float3& k_inc,
									   const Float3& k_refl, const Material* mat, float frequency)
		{
			// This implements Fresnel equations for solving the reflected
			// field given an incident electric plane wave on a planar
			// surface.

			// Compute the indident basis.
			const auto basis = incidencePlaneBasis(norm, k_inc);
			const auto e_tm_out = cross(basis.e_te, k_refl);

			const auto relative_permittivity = mat->relative_permittivity;
			const auto conductivity = mat->conductivity;

			const Complex n2 = indexOfRefraction(relative_permittivity, conductivity, frequency);

			CFloat gamma_tm, gamma_te;
			complexSnellsLaw(norm, k_inc, n2, gamma_tm, gamma_te);

			const Complex elec_te = dot(radiation, basis.e_te);
			const Complex elec_tm = dot(radiation, basis.e_tm_in);
			radiation = basis.e_te * (gamma_te * elec_te) + e_tm_out * (gamma_tm * elec_tm);
		}


		/// Computes the reflected GO radiation vector and surface current at the hit point.
		///
		HC_FN void computeGoRadiation(const SbrParams& params, PathState* path, float frequency)
		{
			// This computes the radiation vector along the ray path.
			//
			// This function is called on-demand when a GO path ray hit occurs (many rays miss, so pre-computing is
			// likely not worthwhile). It uses a cache to avoid recomputing the full GO traced path radiation where
			// possible. Paths are found in monotonically increasing length, so the cache is always the full path or a
			// prefix. The main condition for the radiation vector being re-usable is whether the carrier wavenumber
			// matches. If it doesn't, then the antenna gain and Fresnel coefficients may differ as a function of the
			// wavenumber, and so re-computing is necessary.
			//
			// TODO: Consider whether a tolerance on the wavenumber matching is acceptable, rather than exact-match.


			CFloat3 radiation = path->cache.radiation;
			CFloat3 surface_current = path->cache.surface_current;
			uint32_t ray_index = path->cache.depth;

			if (path->cache.carrier_frequency != frequency)
			{
				// The radiation cache isn't for the correct carrier.
				// Re-compute the full radiation path, starting with sampling the source antenna properties.

				const ActiveAntenna& antenna =
					params.source_antennas[path->source_index + path->time_index * params.source_antenna_count];
				const AntennaModel& model = params.antenna_models[antenna.antenna_model_index];

				const Float3 local_dir = path->local_emmission_direction;
				const CFloat3 pol = propagation::antennaPolarizationVector(model.horizontal_pol, model.vertical_pol,
																		   antenna.direction, local_dir);
				// Gain from the source antenna.
				const float src_local_az = atan2f(local_dir.y, local_dir.x);
				const float src_local_el = asinf(local_dir.z);
				const float wavelength = C<float> / frequency;
				const float gain = sampleAntennaModel(params, model, local_dir, src_local_az, src_local_el, wavelength);
				const float amplitude = std::sqrt(gain);

				radiation = pol * amplitude;
				ray_index = 0;
			}

			for (; ray_index < path->ray_count; ray_index++)
			{
				const auto k_inc = path->trace_directions.at(ray_index);

				if (ray_index + 1 == path->ray_count)
				{
					// Js ~ normal x (h_inc + h_refl)
					// h_inc = k_inc x e_inc
					// The incident magnetic field isn't the surface current. The correction is applied afterwards.
					const CFloat3 magnetic_inc = cross(k_inc, radiation);
					surface_current = magnetic_inc;
				}

				fresnelGoReflection(radiation, path->surface_normals.at(ray_index),
									path->trace_directions.at(ray_index), path->trace_directions.at(ray_index + 1),
									path->surface_materials.at(ray_index), frequency);
			}

			// If the radiation vector was updated for any reason, we need to recompute the surface current.
			if (path->cache.carrier_frequency != frequency || path->cache.depth != path->ray_count)
			{
				// Js ~ normal x (h_inc + h_refl)
				// h_refl = k_refl x e_refl
				const CFloat3 magnetic_refl = cross(path->trace_directions.at(ray_index), radiation);
				// surface_current holds the incident magnetic radiation vector at the moment, as overwritten above.
				surface_current = surface_current + magnetic_refl;
				surface_current = cross(path->surface_normals.at(ray_index - 1), surface_current);

				path->cache.surface_current = surface_current;
				path->cache.radiation = radiation;
				path->cache.depth = path->ray_count;
				path->cache.carrier_frequency = frequency;
			}
		}
	} // namespace go


	/**
	 * @brief TODO
	 *
	 * @param shadow_test Engine-supplied occlusion test: `float(Float3 origin, Float3 segment)`,
	 * returning the hit distance as a fraction of `|segment|`.
	 * @param contrib_index Engine-supplied next contribution index. `uint32_t()`
	 */
	template <typename ShadowTest, typename ContribIndex>
	[[nodiscard]] HC_FN bool facetIncidentRay(const SbrParams& params, PathState* path, const HitInfo& hit,
											  ShadowTest shadow_test, ContribIndex get_contrib_index)
	{
		// Local geometry computations in 32-bit precision in object space.

		auto indices = params.indices[hit.triangle_index];
		auto l_v0 = params.vertices[indices.x];
		auto l_v1 = params.vertices[indices.y];
		auto l_v2 = params.vertices[indices.z];

		auto bary_w = 1.0f - hit.bary_u - hit.bary_v;
		auto l_pos = l_v0 * bary_w + l_v1 * hit.bary_u + l_v2 * hit.bary_v;

		// Simulation computations in 64-bit precision world space.

		const std::array<const Double3, 3> verts{
			amuld(hit.obj_to_world, l_v0),
			amuld(hit.obj_to_world, l_v1),
			amuld(hit.obj_to_world, l_v2),
		};

		const Double3 pos = amuld(hit.obj_to_world, l_pos);
		const Float3 u_inc = path->trace_directions.at(path->ray_count);

		// Outward-facing normal, per the CCW winding convention documented on `SbrParams.indices`.
		const Double3 dnormal = cross(verts[1] - verts[0], verts[2] - verts[0]);
		const double two_times_facet_area = length(dnormal);
		const Float3 u_norm = Float3{dnormal / two_times_facet_area};
		const float facet_area = float(two_times_facet_area) / 2;

		// Compute the reflected ray direction.
		const Float3 back = -u_inc;
		const Float3 cent = dot(u_norm, back) * u_norm;
		const Float3 k_refl = 2.0f * cent - back;

		// Compute the additional propagation distance.
		const double ray_len = dot(pos - path->path_vertices.at(path->ray_count), Double3{u_inc});
		const double go_path_len = path->length + ray_len;

		// Update path
		path->ray_count += 1;
		path->path_vertices.at(path->ray_count) = pos;
		path->trace_directions.at(path->ray_count) = k_refl;
		path->surface_normals.at(path->ray_count - 1) = u_norm;
		path->surface_materials.at(path->ray_count - 1) = &params.materials[hit.material_index];
		path->hash = hash_mix(path->hash, hit.triangle_index); // Add the current facet to the path hash
		path->length = go_path_len;

		// Determine PO contributions from this path and this ray's tube patch on this facet.
		for (uint32_t dst_idx = 0; dst_idx < params.dest_antenna_count; dst_idx++)
		{
			const ActiveAntenna& dst = params.dest_antennas[dst_idx + params.dest_antenna_count * path->time_index];

			// While this is in world-space, this segment is never used where double-precision is useful.
			const Float3 po_seg = Float3{dst.position - pos};

			// Does the triangle face the destination?
			if (dot(po_seg, u_norm) <= 0)
				continue;

			// Cast shadow ray from pos in direction po_seg.
			// OptiX/Embree use floating-point precision here. Conversion is required.
			const float t_shadow = shadow_test(Float3{pos}, po_seg);

			// t_shadow is scaled by |po_seg|, so if it's greater than one, the
			// nearest object is over |po_seg| away, in which case the
			// destination isn't occuded.
			if (t_shadow >= 1.0f)
			{
				// Perform the PO computation for the scattered field from the facet, over the ray
				// tube's patch (clipped to the triangle - see above), not the whole triangle.

				const double path_len = go_path_len + length(dst.position - pos);
				const double prop_time = path_len / C<double>;
				const Float3 u_scat = normalize(po_seg);

				// We have a full path now.
				// Therefore we know the transmitter, TX time, and thus also the carrier.
				const uint32_t tx_index = params.rx_to_tx ? dst_idx : path->source_index;
				const double tx_time = params.times[path->time_index] - (params.rx_to_tx ? prop_time : 0);
				const CarrierModel<float>& carrier = params.carrier_models[tx_index];
				const float frequency = sampleCarrierFrequency(carrier, tx_time);
				const float wavenumber = 2 * PI_V<float> / C<float> * frequency;
				const float wavelength = C<float> / frequency;

				// Updates the path's cached surface current to the current hit point.
				go::computeGoRadiation(params, path, frequency);

				// Ray-tube PO integration patch.
				// This is the part of the PO calculation that isn't symmetric under propagation direction.
				const Float3& prop_po_inc = params.rx_to_tx ? -u_scat : u_inc;
				const auto patch =
					po::computeTubePatch(verts, pos, u_norm, prop_po_inc, facet_area, float(go_path_len), path->weight);
				// The scattering vector `w = k (u_inc - u_scat)` computation is identical between
				// (u_inc, u_scat) and (-u_scat, -u_inc) so we just pass in the trace directions here.
				const CFloat3 radiation =
					po::facetScatteredRadiation(patch, u_norm, u_inc, u_scat, path->cache.surface_current, wavenumber);

				// Weight by the destination antenna's gain towards this facet.
				const AntennaModel& dst_model = params.antenna_models[dst.antenna_model_index];
				// Note that this points from the antenna outward. Negative of the scatter direction to the destination.
				const Float3 dst_local_dir = rotateWorldToLocal(dst.direction, -u_scat);
				const float dst_local_az = atan2f(dst_local_dir.y, dst_local_dir.x);
				const float dst_local_el = asinf(std::clamp(dst_local_dir.z, -1.0f, 1.0f));
				const float dst_gain =
					sampleAntennaModel(params, dst_model, dst_local_dir, dst_local_az, dst_local_el, wavelength);

				const CFloat3 dst_pol = antennaPolarizationVector(dst_model.horizontal_pol, dst_model.vertical_pol,
																  dst.direction, dst_local_dir);

				const float effective_length = wavelength * std::sqrt(dst_gain);

				// Standard antenna-reciprocity polarisation-loss-factor product.
				// Valid because `pol` was evaluated at `-u_scat` above
				// (the antenna's own as-if-transmitting-back-toward-the-source direction).
				// Don't "fix" this by conjugating.
				CFloat voltage = dot_no_conj(radiation, dst_pol) * effective_length;

				// Optionally apply sphere spreading gain loss.
				if ((params.rx_flags[params.rx_to_tx ? path->source_index : dst_idx] & RxFlags::FLAG_NOSPREADING) == 0)
				{
					voltage = voltage / float(path_len);
				}

				// Add a contribution to the sink.
				uint32_t contrib_index = get_contrib_index();
				auto& contrib = params.contributions[contrib_index];
				contrib.voltage = voltage;
				contrib.delay = prop_time;
				contrib.dest_index = dst_idx;
				contrib.path_id = hash_mix(path->hash, dst_idx);
				contrib.source_times_index = path->source_index + params.source_antenna_count * path->time_index;
			}
		}

		return path->ray_count < params.go_step_limit;
	} // facetIncidentRay

	template <typename ShadowTest, typename ContribIndex>
	HC_FN void directPath(const SbrParams& params, uint32_t src_idx, uint32_t dst_idx, uint32_t time_idx,
						  ShadowTest shadow_test, ContribIndex get_contrib_index)
	{
		const auto rx_idx = params.rx_to_tx ? dst_idx : src_idx;
		const auto rx_flags = params.rx_flags[rx_idx];

		if ((rx_flags & RxFlags::FLAG_NODIRECT) != 0)
		{
			return;
		}

		const auto& src_antenna = params.source_antennas[src_idx];
		const auto& dst_antenna = params.dest_antennas[dst_idx];
		const Float3 path = Float3{dst_antenna.position - src_antenna.position};
		const double path_len = length(dst_antenna.position - src_antenna.position);
		const double prop_time = path_len / C<double>;

		if (path_len < 1e-2)
		{
			// The antennas are co-located to within 1cm.
			// FERS does not handle such near-field calculations within its propagation model. Skipping.
			return;
		}

		const float hitlen = shadow_test(Float3{src_antenna.position}, path);

		if (hitlen < 1.0f)
		{
			// Direct path is occluded. Skip.
			return;
		}

		// Get the carrier.
		const auto tx_index = params.rx_to_tx ? dst_idx : src_idx;
		const double tx_time = params.times[time_idx] - (params.rx_to_tx ? double(prop_time) : 0);
		const CarrierModel<float>& carrier = params.carrier_models[tx_index];
		const float frequency = sampleCarrierFrequency(carrier, tx_time);
		const float wavelength = C<float> / frequency;

		// Compute the antenna gains.
		const Float3 u_dir = path / float(path_len);
		const AntennaModel& src_model = params.antenna_models[src_antenna.antenna_model_index];
		const AntennaModel& dst_model = params.antenna_models[dst_antenna.antenna_model_index];

		const Float3 src_local_dir = rotateWorldToLocal(src_antenna.direction, u_dir);
		const float src_local_az = atan2f(src_local_dir.y, src_local_dir.x);
		const float src_local_el = asinf(src_local_dir.z);
		const float src_gain =
			sampleAntennaModel(params, src_model, src_local_dir, src_local_az, src_local_el, wavelength);

		const Float3 dst_local_dir = rotateWorldToLocal(dst_antenna.direction, -u_dir);
		const float dst_local_az = atan2f(dst_local_dir.y, dst_local_dir.x);
		const float dst_local_el = asinf(dst_local_dir.z);
		const float dst_gain =
			sampleAntennaModel(params, dst_model, dst_local_dir, dst_local_az, dst_local_el, wavelength);

		// Compute antenna polarisation vectors.
		const CFloat3 src_pol = propagation::antennaPolarizationVector(src_model.horizontal_pol, src_model.vertical_pol,
																	   src_antenna.direction, src_local_dir);
		const CFloat3 dst_pol = propagation::antennaPolarizationVector(dst_model.horizontal_pol, dst_model.vertical_pol,
																	   dst_antenna.direction, dst_local_dir);

		const bool no_prop_loss = (rx_flags & RxFlags::FLAG_NOSPREADING) != 0;
		const CFloat voltage =
			computeDirectPathGain(src_gain, dst_gain, src_pol, dst_pol, wavelength, float(path_len), no_prop_loss);

		const uint32_t contrib_index = get_contrib_index();
		auto* contrib = &params.contributions[contrib_index];
		contrib->voltage = voltage;
		contrib->delay = prop_time;
		contrib->source_times_index = src_idx + time_idx * params.source_antenna_count;
		contrib->dest_index = dst_idx;
		contrib->path_id = hash_mix(hash_mix(HASH_SEED, src_idx), dst_idx);
	}

} // namespace propagation::raytracing
