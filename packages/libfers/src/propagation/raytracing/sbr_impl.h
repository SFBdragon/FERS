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
	/**
	 * @brief The ray state. This is updated at each step.
	 *
	 * All spacial properties are in FERS's global simulation space, cartesian, in meters.
	 */
	struct PathState
	{
		// Updating path information.

		/// Ray origin, a position, in meters.
		Double3 origin{};
		/// Ray direction from the origin, a unit vector.
		Double3 direction{};
		/// The length of the ray path to `origin`, in meters.
		double length = 0.0;
		/// The hash of the path, unique per facet sequence.
		uint64_t hash = HASH_SEED;
		/// The number of bounces along the path thus far.
		uint32_t ray_index = 0;

		// Launch information.

		/// The "weight" of the ray tube, in steradians. (solid angle / pdf)
		double weight{};
		/// Which radar source antenna this ray originates from, by index.
		uint32_t source_index{};
		/// Which time this ray originates from, by index.
		uint32_t time_index{};

		// Carriers and fields.

		/// Number of active carrier frequencies.
		uint32_t carrier_count{};
		/// Transmitter, by index, to carrier index.
		uint32_t* tx_to_carrier_indices{};
		/// Carrier angular wavenumbers active during this path trace.
		float* carrier_ks{};
		/// Electric field cartesian vectors. One per active carrier.
		/// Perpendicular to direction of propagation.
		///
		/// Note that the units here are a bit interesting.
		/// far-field E(r-hat, R) = sqrt(Z0/4/pi * power) * sqrt(gain) * unit-polarisation(r-hat) * e^(-jkR) / R
		/// These "electric fields" consist only of this:   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
		/// `sqrt(power)`: Signal power is applied later by FERS.
		/// `e^(-jkR)`: The bulk phase term (after down-converting) is computed later by FERS.
		/// `1/R`: The spherical spreading term is applied at the end of the path. See `facetIncidentRay`.
		/// `Z0/4/pi`: These constants are left out. Mostly cancelled at conversion to voltage at end of path.
		///
		/// The units of this quantity is therefore dimensionless.
		/// The point of it is to pick up all the voltage attenuation and phase effects
		/// of the path while being proportional to the electric field.
		/// All geometric optics and physical optics physics performed is linear with
		/// respect to the electric field, and thus the results remain physical.
		std::array<CFloat3, MAX_CARRIERS> elec_fields;
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

		float az_frac = (azimuth + PIf) / (2.0f * PIf) * az_max;
		float el_frac = (elevation + PIf / 2.0f) / PIf * el_max;
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

		double pattern_gain = 0.0;
		switch (model.kind)
		{
		case AntennaKind::Isotropic:
			pattern_gain = 1.0;
			break;
		case AntennaKind::Sinc:
			pattern_gain = antenna::gain::sincGain(double(acosf(boresight_cos)), model.params.sinc.alpha,
												   model.params.sinc.beta, model.params.sinc.gamma);
			break;
		case AntennaKind::Gaussian:
			pattern_gain =
				antenna::gain::gaussianGain(double(local_az), double(local_el), model.params.gaussian.azimuth_scale,
											model.params.gaussian.elevation_scale);
			break;
		case AntennaKind::SquareHorn:
			pattern_gain = antenna::gain::squareHornGain(double(acosf(boresight_cos)),
														 model.params.square_horn.dimension, double(wavelength));
			break;
		case AntennaKind::Parabolic:
			pattern_gain = antenna::gain::parabolicGain(double(acosf(boresight_cos)), model.params.parabolic.diameter,
														double(wavelength));
			break;
		case AntennaKind::Grid2D:
			pattern_gain = double(sampleAntennaPattern(params, model.params.grid_2d.grid, local_az, local_el));
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
		const float phi = PIf * (3.0f - sqrtf(5.0f));

		const float n = static_cast<float>(num_rays);
		const float i = static_cast<float>(path_idx);
		const float x = 1.0f - (2.0f * i) / n; // x from 1 to -1
		const float r = sqrtf(1.0f - x * x); // radius at x
		const float theta = phi * i;
		return Float3{x, r * cosf(theta), r * sinf(theta)};
	}


	HC_FN void initPath(const SbrParams& params, uint32_t path_idx, uint32_t source_idx, uint32_t time_idx,
						PathState& path_out)
	{
		const ActiveAntenna& antenna = params.source_antennae[source_idx + time_idx * params.source_antenna_count];
		const AntennaModel& model = params.antenna_models[antenna.antenna_model_index];

		const Float3 local_dir = sampleIsotropicDirection(path_idx, params.rays_per_source);
		const Float3 world_dir = rotateLocalToWorld(antenna.direction, local_dir);

		const CFloat3 pol = propagation::antennaPolarizationVector(model.horizontal_pol, model.vertical_pol,
																   antenna.direction, local_dir);

		// Get relevant carriers.
		auto carriers = params.carriers[time_idx];
		path_out.carrier_count = carriers.carrier_count;
		path_out.tx_to_carrier_indices = params.tx_to_carrier_indices + carriers.tx_to_carrier_buffer_offset;
		path_out.carrier_ks = params.carrier_ks + carriers.carrier_ks_buffer_offset;

		// Gain from the source antenna.
		const float src_local_az = atan2f(local_dir.y, local_dir.x);
		const float src_local_el = asinf(local_dir.z);
		for (uint32_t i = 0; i < path_out.carrier_count; i++)
		{
			const float wavelength = 2.0f * PIf / path_out.carrier_ks[i];
			const float gain = sampleAntennaModel(params, model, local_dir, src_local_az, src_local_el, wavelength);
			path_out.elec_fields.at(i) = pol * CFloat(std::sqrt(gain), 0.0f);
		}

		path_out.ray_index = 0;
		path_out.length = 0.0;
		path_out.hash = hash_mix(HASH_SEED, source_idx);
		path_out.source_index = source_idx;
		path_out.time_index = time_idx;
		path_out.origin = antenna.position;
		path_out.direction = Double3{world_dir};
		path_out.weight = 4.0 * PI / static_cast<double>(params.rays_per_source);
	}

	namespace po
	{
		struct TubePatch
		{
			/// The vertices defining the hit ray tube patch triangle, in local space.
			/// The vertices are centered on the ray hit point.
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
										 const Double3& norm, const Double3& k_in, double tri_area, double dist,
										 double solid_angle)
		{
			const double cos_theta_i = fmax(dot(-k_in, norm), 1e-3);
			const double a_tube = (dist * dist * solid_angle) / cos_theta_i;
			const double scale = sqrt(fmin(a_tube, tri_area) / tri_area);

			TubePatch patch{};
			for (size_t vi = 0; vi < 3; vi++)
				patch.verts.at(vi) = Float3{scale * (tri_verts.at(vi) - hit_pos)};
			patch.area = float(scale * scale * tri_area);
			return patch;
		}

		/// Numerically stable unnormalised sinc = sin(x)/x with the limit at sin(0) = 1.
		HC_FN float sinc(float x, float tol = 1e-5f) { return fabsf(x) < tol ? 1.0f : sinf(x) / x; }

		/// Gordon's formula for computing the Physical Optics phase integral.
		HC_FN CFloat scaledPhaseIntegral(const TubePatch& patch, const Float3& norm, const Float3& k_inc,
										 const Float3& k_scat, float k0)
		{
			/// Scattering vector, representing phase difference across patch.
			Float3 scatter = k0 * (k_inc - k_scat);
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
			float scalars = -k0 * k0 / scatter_inplane2 * (1.0f / 4.0f / PIf);
			return acc * scalars;
		}

		HC_FN CFloat3 facetScatteredField(const TubePatch& patch, const Float3& norm, const Float3& k_inc,
										  const Float3& k_scat, const CFloat3& mag_Z0_field, const float k0)
		{
			const CFloat3 Js_Z0 = cross(norm, mag_Z0_field); ///< Surface current on the lit patch
			const CFloat scaled_phase_integral = scaledPhaseIntegral(patch, norm, k_inc, k_scat, k0);
			const CFloat3 Es = cross(k_scat, cross(k_scat, Js_Z0)) * scaled_phase_integral;
			return Es;
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
		HC_FN CFloat indexOfRefraction(float relative_permittivity, float conductivity, float carrier_k)
		{
			// 1/sqrt(2). Enough digits to saturate double-precision.
			constexpr float ISQRT2 = 0.70710678118654752440f;

			// Special-case the calculation for dielectrics to avoid a complex sqrt.
			if (conductivity == 0)
				return CFloat{std::sqrt(relative_permittivity), 0.0};

			const float angular_frequency = carrier_k * float(C);
			const float im = conductivity / (angular_frequency * E0);

			// Special-case the calculation for PECs.
			if (conductivity == INFINITY)
				return CFloat{ISQRT2, -ISQRT2} * std::sqrt(im);

			// Handle the partially-conducting case.
			const CFloat permittivity{relative_permittivity, -im};
			return csqrt(permittivity);
		}

		/// Reflect Cartesian electric fields off a facet specularly using geometric optics.
		///
		/// Additionally writes out the combined surface magnetic fields H.Z0 = (H_inc + H_refl).Z0 to `mag_z0_fields`.
		HC_FN void reflectFields(PathState* path, const Float3& norm, const Float3& k_inc, const Float3& k_refl,
								 const Material* mat, std::array<CFloat3, MAX_CARRIERS>& mag_z0_fields)
		{
			// This implements Fresnel equations for solving the reflected field
			// given an incident electric plane wave on a planar surface.

			// Compute the indident basis.
			const auto basis = incidencePlaneBasis(norm, k_inc);
			const auto e_tm_out = cross(basis.e_te, k_refl);

			const auto relative_permittivity = mat->relative_permittivity;
			const auto conductivity = mat->conductivity;

			// For every carrier frequency...
			for (uint32_t i = 0; i < path->carrier_count; i++)
			{
				const float carrier_freq = path->carrier_ks[i];
				const Complex n2 = indexOfRefraction(relative_permittivity, conductivity, carrier_freq);

				CFloat gamma_tm, gamma_te;
				complexSnellsLaw(norm, k_inc, n2, gamma_tm, gamma_te);

				const auto elec_inc = path->elec_fields.at(i);
				const Complex elec_te = dot(elec_inc, basis.e_te);
				const Complex elec_tm = dot(elec_inc, basis.e_tm_in);
				const CFloat3 elec_refl = basis.e_te * (gamma_te * elec_te) + e_tm_out * (gamma_tm * elec_tm);

				mag_z0_fields.at(i) = cross(k_inc, elec_inc) + cross(k_refl, elec_refl);

				path->elec_fields.at(i) = elec_refl;
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
		const Double3 k_inc = path->direction;

		// Outward-facing normal, per the CCW winding convention documented on `SbrParams.indices`.
		Double3 norm = cross(verts[1] - verts[0], verts[2] - verts[0]);
		const double tri_area_2 = length(norm);
		norm /= tri_area_2;

		// Compute the reflected ray direction.
		Double3 back = -k_inc;
		Double3 cent = dot(norm, back) * norm;
		Double3 k_refl = 2.0 * cent - back;

		// Compute the additional propagation distance.
		const double dist = dot(pos - path->origin, k_inc);
		const double go_dist = path->length + dist;

		// Ray-tube PO integration patch.
		const auto patch = po::computeTubePatch(verts, pos, norm, k_inc, tri_area_2 / 2.0, go_dist, path->weight);

		// Compute the reflected eletric field and surface magnetic fields.
		const Material* material = &params.materials[hit.material_index];
		std::array<CFloat3, MAX_CARRIERS> mag_z0_fields;
		go::reflectFields(path, Float3{norm}, Float3{k_inc}, Float3{k_refl}, material, mag_z0_fields);

		// Add the current facet to the path hash
		path->hash = hash_mix(path->hash, hit.triangle_index);

		// Determine PO contributions from this path and this ray's tube patch on this facet.
		for (uint32_t dst_idx = 0; dst_idx < params.dest_antenna_count; dst_idx++)
		{
			const ActiveAntenna& dst = params.dest_antennae[dst_idx + params.dest_antenna_count * path->time_index];
			const Double3 po_seg = dst.position - pos;

			// Does the triangle face the destination?
			if (dot(po_seg, norm) <= 0)
				continue;

			// Cast shadow ray from pos in direction po_seg.
			// OptiX/Embree use floating-point precision here.
			const float t_shadow = shadow_test(Float3{pos}, Float3{po_seg});

			// t_shadow is scaled by |po_seg|, so if it's greater than one, the
			// nearest object is over |po_seg| away, in which case the
			// destination isn't occuded.
			if (t_shadow >= 1.0f)
			{
				// Perform the PO computation for the scattered field from the facet, over the ray
				// tube's patch (clipped to the triangle - see above), not the whole triangle.

				const double po_dist = length(po_seg);
				const Float3 k_scat = normalize(Float3{po_seg});

				// We finally know the frequency for sure. We can do all the field calculations now.
				uint32_t tx_index = params.rx_to_tx ? dst_idx : path->source_index;
				uint32_t carrier_index = path->tx_to_carrier_indices[tx_index];
				float k0 = path->carrier_ks[carrier_index];
				float wavelength = 2.0f * PIf / k0;

				CFloat3 e_po = po::facetScatteredField(patch, Float3{norm}, Float3{k_inc}, k_scat,
													   mag_z0_fields.at(carrier_index), k0);

				// Weight by the destination antenna's gain towards this facet.
				const AntennaModel& model = params.antenna_models[dst.antenna_model_index];
				// Note that this points from the antenna outward. Negative of the scatter direction to the destination.
				const Float3 dst_local_dir = rotateWorldToLocal(dst.direction, -k_scat);
				const float dst_local_az = atan2f(dst_local_dir.y, dst_local_dir.x);
				const float dst_local_el = asinf(std::clamp(dst_local_dir.z, -1.0f, 1.0f));
				const float dst_gain =
					sampleAntennaModel(params, model, dst_local_dir, dst_local_az, dst_local_el, wavelength);

				const CFloat3 pol =
					antennaPolarizationVector(model.horizontal_pol, model.vertical_pol, dst.direction, dst_local_dir);

				const double path_dist = go_dist + po_dist;
				const float effective_length = wavelength * sqrtf(dst_gain);

				// Standard antenna-reciprocity polarisation-loss-factor product.
				// Valid because `pol` was evaluated at `-k_scat` above
				// (the antenna's own as-if-transmitting-back-toward-the-source direction).
				// Don't "fix" this by conjugating.
				CFloat voltage = dot_no_conj(e_po, pol) * effective_length;
				// Optionally apply sphere spreading gain loss.
				if ((params.rx_flags[params.rx_to_tx ? dst_idx : path->source_index] & RxFlags::FLAG_NOSPREADING) == 0)
				{
					voltage = voltage / float(path_dist);
				}


				// Add a contribution to the sink.
				uint32_t contrib_index = get_contrib_index();
				auto& contrib = params.contributions[contrib_index];
				contrib.voltage = voltage;
				contrib.delay = path_dist / C;
				contrib.dest_index = dst_idx;
				contrib.path_id = hash_mix(path->hash, dst_idx);
				contrib.source_times_index = path->source_index + params.source_antenna_count * path->time_index;
			}
		}

		if (path->ray_index >= params.go_step_limit)
			return false;

		path->ray_index += 1;
		path->origin = pos;
		path->direction = k_refl;
		path->length = go_dist;

		return true;
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

		auto src_antenna = params.source_antennae[src_idx];
		auto dst_antenna = params.dest_antennae[dst_idx];
		auto src_to_dst = dst_antenna.position - src_antenna.position;

		if (length2(src_to_dst) < 1e-4)
		{
			// The antennas are co-located to within 1cm.
			// FERS does not handle such near-field calculations within its propagation model. Skipping.
			return;
		}

		auto hitlen = shadow_test(Float3{src_antenna.position}, Float3{src_to_dst});

		if (hitlen < 1.0f)
		{
			// Direct path is occluded. Skip.
			return;
		}

		// Get the carrier.
		const auto carriers = params.carriers[time_idx];
		const auto tx_idx = params.rx_to_tx ? dst_idx : src_idx;
		const auto k0 = params.carrier_ks[carriers.carrier_ks_buffer_offset +
										  params.tx_to_carrier_indices[carriers.tx_to_carrier_buffer_offset + tx_idx]];
		const auto wavelength = 2.0f * PIf / k0;

		// Compute the antenna gains.
		const auto distance = length(src_to_dst);
		const auto k_direct = Float3{normalize(src_to_dst)};
		const AntennaModel& src_model = params.antenna_models[src_antenna.antenna_model_index];
		const AntennaModel& dst_model = params.antenna_models[dst_antenna.antenna_model_index];

		const Float3 src_local_dir = rotateWorldToLocal(src_antenna.direction, k_direct);
		const float src_local_az = atan2f(src_local_dir.y, src_local_dir.x);
		const float src_local_el = asinf(src_local_dir.z);
		const float src_gain =
			sampleAntennaModel(params, src_model, src_local_dir, src_local_az, src_local_el, wavelength);

		const Float3 dst_local_dir = rotateWorldToLocal(dst_antenna.direction, -k_direct);
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
			computeDirectPathGain(src_gain, dst_gain, src_pol, dst_pol, wavelength, distance, no_prop_loss);

		const uint32_t contrib_index = get_contrib_index();
		auto* contrib = &params.contributions[contrib_index];
		contrib->voltage = voltage;
		contrib->delay = length(dst_antenna.position - src_antenna.position) / C;
		contrib->source_times_index = src_idx + time_idx * params.source_antenna_count;
		contrib->dest_index = dst_idx;
		contrib->path_id = hash_mix(hash_mix(HASH_SEED, src_idx), dst_idx);
	}

} // namespace propagation::raytracing
