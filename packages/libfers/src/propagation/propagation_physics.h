
#include "propagation/math.h"

namespace propagation
{
	// This file includes physics code shared by the propagation models.

	struct TransverseBasis
	{
		/// Horizontal at the boresight. +Y for the +X direction.
		Float3 horizontal;
		/// Vertical at the boresight. +Z for the +X direction.
		Float3 vertical;
	};

	/// This uses the Ludwig-3 definition of antenna polarisation.
	///
	/// This has been specialised to suit FERS's coordinate frame:
	/// * +X is the boresight/forward (azimuth=0, elevation=0)
	/// * +Z is "up" (increasing elevation increases Z)
	///
	/// Ludwig-3 is usually oriented with +Z being the boresight.
	/// Instead, this version essentially rotates that definition through z->x->y->z
	/// Because Z is up, we can label the polarisation basis vectors H (+Y) and V (+Z)
	[[nodiscard]] HC_FN TransverseBasis ludwig3Basis(const Float3& local_dir)
	{
		constexpr float R2_EPS = 1e-12f; // guards both poles, (+1,0,0) and (-1,0,0)

		const float r2 = local_dir.y * local_dir.y + local_dir.z * local_dir.z;

		// Phi here is the angle that rotates from +Y through +Z
		// Theta is the angle away from the boresight (+X).
		float cosPhi{}, sinPhi{};
		if (r2 < R2_EPS)
		{
			// Phi undefined at either pole.
			// At +x (boresight) it doesn't matter, the formula gives (0,1,0)/(0,0,1) for any phi.
			// At -x it's a discontinuity; phi=0 is as arbitrary as anything else.
			cosPhi = 1.0f;
			sinPhi = 0.0f;
		}
		else
		{
			const float invR = 1.0f / sqrtf(r2);
			cosPhi = local_dir.y * invR;
			sinPhi = local_dir.z * invR;
		}
		const float cosTheta = local_dir.x;
		const float sinTheta = sqrtf(1.0f - cosTheta * cosTheta);

		const Float3 theta_hat{-sinTheta, cosTheta * cosPhi, cosTheta * sinPhi};
		const Float3 phi_hat{0.0f, -sinPhi, cosPhi};

		return {theta_hat * cosPhi - phi_hat * sinPhi, theta_hat * sinPhi + phi_hat * cosPhi};
	}

	[[nodiscard]] HC_FN CFloat3 antennaPolarizationVector(const CFloat& jones_horizontal, const CFloat& jones_vertical,
														  const AzEl& orientation, const Float3& local_dir)
	{
		const TransverseBasis basis = ludwig3Basis(local_dir);
		const Float3 world_h = rotateLocalToWorld(orientation, basis.horizontal);
		const Float3 world_v = rotateLocalToWorld(orientation, basis.vertical);
		return jones_horizontal * world_h + jones_vertical * world_v;
	}


	/**
	 * @brief Computes the power scaling factor for a direct path (Friis Transmission Equation).
	 * @param tx_gain Transmitter gain (linear).
	 * @param rx_gain Receiver gain (linear).
	 * @param lambda Wavelength (meters).
	 * @param dist Distance (meters).
	 * @param no_prop_loss If true, distance-based attenuation is ignored.
	 * @return The voltage scaling factor (Vr / Vt) or sqrt(Pr / Pt).
	 */
	[[nodiscard]] HC_FN CFloat computeDirectPathGain(float tx_gain, float rx_gain, const CFloat3& tx_pol,
													 const CFloat3& rx_pol, float lambda, double dist,
													 bool no_prop_loss)
	{
		const float numerator = std::sqrt(tx_gain * rx_gain) * lambda;

		double denominator = 4.0 * PI; // 4 * PI
		if (!no_prop_loss)
		{
			denominator *= dist;
		}

		return float(numerator / denominator) * dot_no_conj(tx_pol, rx_pol);
	}
}
