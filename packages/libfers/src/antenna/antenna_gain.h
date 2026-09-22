// SPDX-License-Identifier: GPL-2.0-only
//
// Copyright (c) 2026-present FERS Contributors (see AUTHORS.md).
//
// See the GNU GPLv2 LICENSE file in the FERS project root for more information.

/**
 * @file antenna_gain.h
 *
 * @brief Antenna gain-pattern formulas, shared between the CPU antenna model
 * (`antenna::Antenna` subclasses) and the GPU ray-tracing propagation model.
 *
 * Each function here is a plain `constexpr` free function operating only on scalar angles and
 * antenna parameters to be portable to CUDA/HIP device code.
 * This works because plain, unannotated `constexpr` functions are callable from both host and
 * device code by `nvcc` with `--expt-relaxed-constexpr` (see `cmake/FersOptiX.cmake`).
 */

#pragma once

#include <cmath>

#include "core/config.h"

namespace antenna::gain
{
	/// sinc(x) = sin(x)/x, with the x=0 singularity resolved to its limit, 1.
	constexpr RealType sinc(const RealType x) noexcept
	{
		if (std::abs(x) < EPSILON)
		{
			return RealType(1);
		}
		return std::sin(x) / x;
	}

	/// `antenna::Sinc`'s gain pattern (excluding the efficiency factor, applied by the caller).
	constexpr RealType sincGain(const RealType theta, const RealType alpha, const RealType beta,
								const RealType gamma) noexcept
	{
		return alpha * pow(fabs(sinc(beta * theta)), gamma);
	}

	/// `antenna::Gaussian`'s gain pattern (excluding the efficiency factor, applied by the caller).
	constexpr RealType gaussianGain(const RealType delta_azimuth, const RealType delta_elevation,
									const RealType azimuth_scale, const RealType elevation_scale) noexcept
	{
		return exp(-delta_azimuth * delta_azimuth * azimuth_scale) *
			exp(-delta_elevation * delta_elevation * elevation_scale);
	}

	/// `antenna::SquareHorn`'s gain pattern (excluding the efficiency factor, applied by the caller).
	constexpr RealType squareHornGain(const RealType theta, const RealType dimension,
									  const RealType wavelength) noexcept
	{
		const RealType directivity = RealType(4) * PI * dimension * dimension / (wavelength * wavelength);
		const RealType x = PI * dimension * sin(theta) / wavelength;
		const RealType s = sinc(x);
		return directivity * s * s;
	}

	/**
	 * @brief A portable rational approximation of the Bessel function of the first kind, order 1,
	 * J1(x), accurate to roughly 1e-8 relative error. Used in place of `core::besselJ1` (which just
	 * wraps libm's `j1()`, host-only, no CUDA device equivalent) so `parabolicGain` below is
	 * evaluable from both host and device code.
	 *
	 * Standard split-domain rational-polynomial form (Abramowitz & Stegun 9.4 coefficients); this is
	 * an independent implementation of that classical, public-domain numerical result, not a copy of
	 * any particular library's source.
	 */
	constexpr RealType besselJ1Approx(const RealType x) noexcept
	{
		const RealType ax = fabs(x);
		if (ax < RealType(8))
		{
			const RealType y = x * x;
			const RealType p = x *
				(RealType(72362614232.0) +
				 y *
					 (RealType(-7895059235.0) +
					  y *
						  (RealType(242396853.1) +
						   y * (RealType(-2972611.439) + y * (RealType(15704.48260) + y * RealType(-30.16036606))))));
			const RealType q = RealType(144725228442.0) +
				y *
					(RealType(2300535178.0) +
					 y *
						 (RealType(18583304.74) +
						  y * (RealType(99447.43394) + y * (RealType(376.9991397) + y * RealType(1.0)))));
			return p / q;
		}

		const RealType z = RealType(8) / ax;
		const RealType y = z * z;
		const RealType envelope_phase = ax - RealType(2.356194491);
		const RealType p = RealType(1.0) +
			y *
				(RealType(0.183105e-2) +
				 y * (RealType(-0.3516396496e-4) + y * (RealType(0.2457520174e-5) + y * RealType(-0.240337019e-6))));
		const RealType q = RealType(0.04687499995) +
			y *
				(RealType(-0.2002690873e-3) +
				 y * (RealType(0.8449199096e-5) + y * (RealType(-0.88228987e-6) + y * RealType(0.105787412e-6))));
		RealType result =
			sqrt(RealType(0.636619772) / ax) * (cos(envelope_phase) * p - z * std::sin(envelope_phase) * q);
		if (x < RealType(0))
		{
			result = -result;
		}
		return result;
	}

	/// J1(x)/x, with the x=0 singularity resolved to its limit, 1/2 (NOT 1 - see antenna_factory.cpp
	/// history: the pre-refactor `j1C` helper returned 1 at x=0, a bug causing a 4x peak-gain error
	/// for `Parabolic` right at boresight).
	constexpr RealType besselJ1OverX(const RealType x) noexcept
	{
		if (fabs(x) < EPSILON)
		{
			return 0.5;
		}
		return besselJ1Approx(x) / x;
	}

	/// `antenna::Parabolic`'s gain pattern (excluding the efficiency factor, applied by the caller).
	constexpr RealType parabolicGain(const RealType theta, const RealType diameter, const RealType wavelength) noexcept
	{
		const RealType k = PI * diameter / wavelength;
		const RealType directivity = k * k;
		const RealType x = k * sin(theta);
		const RealType j1_over_x = besselJ1OverX(x);
		return directivity * (RealType(2) * j1_over_x) * (RealType(2) * j1_over_x);
	}
}
