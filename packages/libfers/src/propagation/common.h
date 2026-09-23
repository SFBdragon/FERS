// SPDX-License-Identifier: GPL-2.0-only
//
// Copyright (c) 2026-present FERS Contributors (see AUTHORS.md).
//
// See the GNU GPLv2 LICENSE file in the FERS project root for more information.

#pragma once

#include <cmath>
#include <cstdint>

#include "core/sim_id.h"
#include "propagation/math.h"

namespace propagation
{

	// --- Polarisation Calculations ----------------------------------------------------------------------------- //

	/// This uses the Ludwig-3 definition of antenna polarisation.
	///
	/// This has been specialised to suit FERS's coordinate frame:
	/// * +X is the boresight/forward (azimuth=0, elevation=0)
	/// * +Z is "up" (increasing elevation increases Z)
	///
	/// Ludwig-3 is usually oriented with +Z being the boresight.
	/// Instead, this version essentially rotates that definition through z->x->y->z
	/// Because Z is up, we can label the polarisation basis vectors H (+Y) and V (+Z)
	/// for horizontal and vertical polarisation with respect to the antenna boresight.
	template <typename Real>
	HC_FN void ludwig3Basis(const Real3<Real>& local_dir, Real3<Real>& h_out, Real3<Real>& v_out)
	{
		constexpr Real R2_EPS = Real(1e-12); // guards both poles, (+1,0,0) and (-1,0,0)

		const auto r2 = local_dir.y * local_dir.y + local_dir.z * local_dir.z;

		// Phi here is the angle that rotates from +Y through +Z
		// Theta is the angle away from the boresight (+X).
		Real cosPhi{}, sinPhi{};
		if (r2 < R2_EPS)
		{
			// Phi undefined at either pole.
			// At +x (boresight) it doesn't matter, the formula gives (0,1,0)/(0,0,1) for any phi.
			// At -x it's a discontinuity; phi=0 is as arbitrary as anything else.
			cosPhi = Real(1.0);
			sinPhi = Real(0.0);
		}
		else
		{
			const Real invR = 1 / std::sqrt(r2);
			cosPhi = local_dir.y * invR;
			sinPhi = local_dir.z * invR;
		}
		const Real cosTheta = local_dir.x;
		const Real sinTheta = std::sqrt(Real(1.0) - cosTheta * cosTheta);

		const Real3<Real> theta_hat{-sinTheta, cosTheta * cosPhi, cosTheta * sinPhi};
		const Real3<Real> phi_hat{Real(0.0), -sinPhi, cosPhi};

		h_out = theta_hat * cosPhi - phi_hat * sinPhi;
		v_out = theta_hat * sinPhi + phi_hat * cosPhi;
	}

	template <typename Real>
	[[nodiscard]] HC_FN Complex3<Real>
	antennaPolarizationVector(const Complex<Real>& jones_horizontal, const Complex<Real>& jones_vertical,
							  const AzEl<Real>& orientation, const Real3<Real>& local_dir)
	{
		Real3<Real> h_basis, v_basis;
		ludwig3Basis(local_dir, h_basis, v_basis);
		const Real3 world_h = rotateLocalToWorld(orientation, h_basis);
		const Real3 world_v = rotateLocalToWorld(orientation, v_basis);
		return jones_horizontal * world_h + jones_vertical * world_v;
	}

	// --- Specular Basis Decomposition ------------------------------------------------------------------ //

	/**
	 * @brief Local (perpendicular/TE, parallel/TM) basis for the plane of
	 * incidence at a facet hit.
	 *
	 * Degenerate at normal incidence (k_in parallel to n): e_perp is 0/0, so
	 * an arbitrary tangent is substituted. Safe because gamma_te ==
	 * gamma_tm at normal incidence for any isotropic material.
	 */
	template <typename Real>
	struct IncidencePlaneBasis
	{
		/// Transverse Electric (TE) basis. Perpendicular to facet normal and incoming direction.
		Real3<Real> e_te;
		/// Transverse Magnetic (TM) basis of the incoming ray. In-plane with incoming direction and facet normal.
		Real3<Real> e_tm_in;
	};

	/// Returns a unit vector perpendicular to `dir`. `dir` need not be normalised.
	template <typename Real>
	HC_FN Real3<Real> arbitraryPerpendicular(const Real3<Real>& dir)
	{
		const auto t = std::abs(dir.x) < Real(0.9) ? Real3<Real>{1, 0, 0} : Real3<Real>{0, 1, 0};
		return normalize(cross(t, dir));
	}

	template <typename Real>
	HC_FN IncidencePlaneBasis<Real> incidencePlaneBasis(const Real3<Real>& normal, const Real3<Real>& k_in)
	{
		auto raw = cross(k_in, normal);
		auto len = length(raw);
		Real3<Real> e_te;
		if (len < Real(1e-9))
		{
			e_te = arbitraryPerpendicular(normal);
		}
		else
		{
			e_te = raw / len;
		}
		return {e_te, cross(e_te, k_in)};
	}

	// --- Gain Calculations ----------------------------------------------------------------------------- //

	/**
	 * @brief Computes the power scaling factor for a direct path (Friis Transmission Equation).
	 * @param tx_gain Transmitter gain (linear).
	 * @param rx_gain Receiver gain (linear).
	 * @param lambda Wavelength (meters).
	 * @param dist Distance (meters).
	 * @param no_prop_loss If true, distance-based attenuation is ignored.
	 * @return The voltage scaling factor (Vr / Vt) or sqrt(Pr / Pt).
	 */
	template <typename Real>
	[[nodiscard]] HC_FN Complex<Real> computeDirectPathGain(Real tx_gain, Real rx_gain, const Complex3<Real>& tx_pol,
															const Complex3<Real>& rx_pol, Real lambda, double dist,
															bool no_prop_loss)
	{
		const Real numerator = std::sqrt(tx_gain * rx_gain) * lambda;

		double denominator = 4.0 * PI; // 4 * PI
		if (!no_prop_loss)
		{
			denominator *= dist;
		}

		return Real(numerator / denominator) * dot_no_conj(tx_pol, rx_pol);
	}

	// --- Path IDs ----------------------------------------------------------------------------- //

	constexpr uint64_t HASH_SEED = 0xcbf29ce484222325ULL;

	// Fold a single low-biased uint32_t into a running 64-bit hash.
	[[nodiscard]] HC_FN uint64_t hash_mix(uint64_t hash, uint32_t idx)
	{
		constexpr uint64_t P1 = 0x9E3779B97F4A7C15ULL; // odd, golden-ratio
		constexpr uint64_t P2 = 0xC2B2AE3D27D4EB4FULL; // odd, independent

		uint64_t v = uint64_t(idx) * P2; // spreads idx's low bits across all 64 output bits
		hash ^= v;
		hash *= P1;
		hash ^= hash >> 32; // fold keeps high+low bits interacting
		return hash;
	}

	// Fold a single sim ID into a running 64-bit hash.
	[[nodiscard]] HC_FN uint64_t hash_mix(uint64_t hash, SimId id)
	{
		constexpr uint64_t P1 = 0x9E3779B97F4A7C15ULL; // odd, golden-ratio
		constexpr uint64_t P2 = 0xC2B2AE3D27D4EB4FULL; // odd, independent

		uint64_t v = id * P2;
		hash ^= v;
		hash *= P1;
		hash ^= hash >> 32; // fold keeps high+low bits interacting
		return hash;
	}
}
