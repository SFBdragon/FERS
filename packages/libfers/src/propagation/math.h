//
// TODO_SHAUN
//

#pragma once

// TODO_SHAUN potentially consider dropping this dependency?
#include "linalg.h"

#ifndef HC_FN
#if defined(__CUDACC__) || defined(__HIPCC__)
#define HC_FN __host__ __device__ inline
#else
#define HC_FN inline
#endif
#endif

namespace propagation
{
	/// The ratio of a circle's circumference to its diameter.
	// Far more than enough digits to saturate double-precision.
	constexpr double PI = 3.1415926535897932384626433832795028841971693993751;
	constexpr float PIf = 3.1415926535897932384626433832795028841971693993751f;

	/// Intrinsic impedance of free space, Ohms.
	/// https://physics.nist.gov/cgi-bin/cuu/Value?z0
	/// Accessed 2026-09-12
	constexpr float Z0 = 376.730313412f;

	/// Vacuum electric permittivity, in Farads/meter.
	/// https://physics.nist.gov/cgi-bin/cuu/Value?ep0
	/// Accessed 2026-09-15
	constexpr float E0 = 8.8541878188e-12f;

	/// Speed of light in a vacuum. Meters per second.
	/// https://physics.nist.gov/cgi-bin/cuu/Value?c
	/// Accessed 2026-09-15
	constexpr float C = 299792458.0f;

	// Don't use HLSL/CUDA/HIP naming convention to mitigate naming clashes and confusion.
	// They use double3/float2/uint4 etc.
	//
	// We'll use `vecnt` where `n` is the number of elements and
	// `t` is f/d/c/u/i (float, double, complex, uint, int).
	// Complex numbers are always double-precsion for our purposes.

	using Vec2f = linalg::aliases::float2;
	using Uint3 = linalg::aliases::uint3;
	using Float3 = linalg::aliases::float3;
	using Double3 = linalg::aliases::double3;

	using Float3x4 = linalg::aliases::float3x4;

	/**
	 * @brief A pure azimuth/elevation orientation (no roll).
	 *
	 * Follows the same convention as `Vec3(SVec3)`: azimuth=0, elevation=0 means
	 * local +X; `rotateLocalToWorld(AzEl{0,0}, ...)`'s boresight direction is
	 * `(cos(az)cos(el), sin(az)cos(el), sin(el))`, i.e. `Vec3(SVec3(1, az, el))`.
	 */
	struct AzEl
	{
		float azimuth{};
		float elevation{};
	};

	/**
	 * @brief Rotates a local direction (in the same az/el spherical convention as `SVec3` - see
	 * `AzEl`) into world space, given the frame's own world-space orientation `rot`.
	 *
	 * Equivalent to constructing the rotation matrix R = Rz(rot.azimuth) * Ry(-rot.elevation) (the
	 * "yaw then pitch, no roll" rotation that carries local +X to
	 * `Vec3(SVec3(1, rot.azimuth, rot.elevation))`) and applying it to `local_dir`.
	 */
	HC_FN Float3 rotateLocalToWorld(const AzEl& rot, const Float3& local_dir)
	{
		const float caz = cosf(rot.azimuth), saz = sinf(rot.azimuth);
		const float cel = cosf(rot.elevation), sel = sinf(rot.elevation);

		return Float3{caz * cel * local_dir.x - saz * local_dir.y - caz * sel * local_dir.z,
					  saz * cel * local_dir.x + caz * local_dir.y - saz * sel * local_dir.z,
					  sel * local_dir.x + cel * local_dir.z};
	}

	/**
	 * @brief Inverse of `rotateLocalToWorld`: expresses a world-space direction in the frame's local
	 * coordinates. Since the rotation is orthonormal, this is just its transpose applied to
	 * `world_dir`.
	 */
	HC_FN Float3 rotateWorldToLocal(const AzEl& rot, const Float3& world_dir)
	{
		const float caz = cosf(rot.azimuth), saz = sinf(rot.azimuth);
		const float cel = cosf(rot.elevation), sel = sinf(rot.elevation);

		return Float3{caz * cel * world_dir.x + saz * cel * world_dir.y + sel * world_dir.z,
					  -saz * world_dir.x + caz * world_dir.y,
					  -caz * sel * world_dir.x - saz * sel * world_dir.y + cel * world_dir.z};
	}

	template <typename Real>
	struct Complex
	{
		Real re, im;
		HC_FN constexpr Complex() : re(0.), im(0.) {}
		HC_FN constexpr Complex(Real r, Real i = 0.) : re(r), im(i) {}
	};

	template <typename Real>
	HC_FN constexpr Complex<Real> operator+(Complex<Real> a, Complex<Real> b)
	{
		return Complex(a.re + b.re, a.im + b.im);
	}
	template <typename Real>
	HC_FN constexpr Complex<Real> operator-(Complex<Real> a, Complex<Real> b)
	{
		return Complex(a.re - b.re, a.im - b.im);
	}
	template <typename Real>
	HC_FN constexpr Complex<Real> operator-(Real a, Complex<Real> b)
	{
		return Complex(a - b.re, -b.im);
	}
	template <typename Real>
	HC_FN constexpr Complex<Real> operator-(Complex<Real> a)
	{
		return Complex(-a.re, -a.im);
	}
	template <typename Real>
	HC_FN constexpr Complex<Real> operator*(Complex<Real> a, Complex<Real> b)
	{
		return Complex(a.re * b.re - a.im * b.im, a.re * b.im + a.im * b.re);
	}
	template <typename Real>
	HC_FN constexpr Complex<Real> operator*(Complex<Real> a, Real s)
	{
		return Complex(a.re * s, a.im * s);
	}
	template <typename Real>
	HC_FN constexpr Complex<Real> operator*(Real s, Complex<Real> a)
	{
		return a * s;
	}
	template <typename Real>
	HC_FN constexpr Complex<Real> operator/(Complex<Real> a, Complex<Real> b)
	{
		Real d = b.re * b.re + b.im * b.im;
		return Complex((a.re * b.re + a.im * b.im) / d, (a.im * b.re - a.re * b.im) / d);
	}
	template <typename Real>
	HC_FN constexpr Complex<Real> operator/(Complex<Real> a, Real s)
	{
		return Complex(a.re / s, a.im / s);
	}
	template <typename Real>
	HC_FN constexpr Real cabs(Complex<Real> a)
	{
		return std::sqrt(a.re * a.re + a.im * a.im);
	}
	template <typename Real>
	HC_FN constexpr Complex<Real> conj(Complex<Real> a)
	{
		return Complex(a.re, -a.im);
	}
	template <typename Real>
	HC_FN constexpr Complex<Real> mul_i(Complex<Real> a)
	{
		return Complex(-a.im, a.re);
	} // a * j
	template <typename Real>
	HC_FN constexpr Complex<Real> cexp_i(Real x)
	{
		return Complex<Real>(std::cos(x), std::sin(x));
	} // e^{jx}
	template <typename Real>
	HC_FN constexpr Complex<Real> csqrt(Complex<Real> a)
	{
		if (a.re == Real(0.0) && a.im == Real(0.0))
			return Complex<Real>(0.0, 0.0);

		Real r = cabs(a);

		if (a.re >= Real(0.0))
		{
			Real w = std::sqrt(Real(0.5) * (r + a.re)); // safety: sum of positive terms
			return Complex<Real>(w, a.im / (Real(2.0) * w));
		}

		Real w = std::sqrt(Real(0.5) * (r - a.re)); // safety: positive minus negative
		Real im = (a.im >= Real(0.0)) ? w : -w;
		return Complex<Real>(a.im / (Real(2.0) * im), im);
	}

	using CFloat = Complex<float>;
	using CDouble = Complex<double>;

	struct CFloat3
	{
		CFloat x, y, z;
	};

	HC_FN constexpr CFloat3 operator+(CFloat3 a, CFloat3 b) { return CFloat3{a.x + b.x, a.y + b.y, a.z + b.z}; }
	HC_FN constexpr CFloat3 operator-(const CFloat3& v) { return CFloat3{-v.x, -v.y, -v.z}; }
	HC_FN constexpr CFloat3 operator*(const Float3& r, CFloat c) { return CFloat3{r.x * c, r.y * c, r.z * c}; }
	HC_FN constexpr CFloat3 operator*(CFloat c, const Float3& r) { return r * c; }
	HC_FN constexpr CFloat3 operator*(const CFloat3& v, CFloat c) { return CFloat3{v.x * c, v.y * c, v.z * c}; }
	HC_FN constexpr CFloat3 operator*(CFloat c, const CFloat3& v) { return v * c; }
	HC_FN constexpr CFloat3 operator*(const CFloat3& v, float s) { return CFloat3{v.x * s, v.y * s, v.z * s}; }
	HC_FN constexpr CFloat3 operator/(const CFloat3& v, float s) { return CFloat3{v.x / s, v.y / s, v.z / s}; }

	HC_FN constexpr CFloat3 cross(const Float3& a, const CFloat3& b)
	{
		return CFloat3{a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x};
	}
	HC_FN constexpr CFloat dot(const CFloat3& a, const Float3& b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
	HC_FN constexpr CFloat dot_no_conj(const CFloat3& a, const CFloat3& b) { return a.x * b.x + a.y * b.y + a.z * b.z; }

	/// Applies a 3x4 affine matrix to a 3-vector.
	/// Result = A * v + t
	template <class T>
	HC_FN Float3 amul(const Float3x4& m, const Float3& v) noexcept
	{
		using std::fma;
		T x = fma(m.x.x, v.x, fma(m.y.x, v.y, fma(m.z.x, v.z, m.w.x)));
		T y = fma(m.x.y, v.x, fma(m.y.y, v.y, fma(m.z.y, v.z, m.w.y)));
		T z = fma(m.x.z, v.x, fma(m.y.z, v.y, fma(m.z.z, v.z, m.w.z)));
		return {x, y, z};
	}

	/// Applies a 3x4 affine matrix to a 3-vector.
	/// Result = A * v + t
	///
	/// This takes floats, converts to doubles, and outputs doubles.
	/// This is designed for converting from 32-bit precision local space to
	/// 64-bit precision world space.
	HC_FN Double3 amuld(const Float3x4& m, const Float3& v) noexcept
	{
		const double vx = static_cast<double>(v.x), vy = static_cast<double>(v.y), vz = static_cast<double>(v.z);

		double x = static_cast<double>(m.w.x);
		x = fma(static_cast<double>(m.x.x), vx, x);
		x = fma(static_cast<double>(m.y.x), vy, x);
		x = fma(static_cast<double>(m.z.x), vz, x);

		double y = static_cast<double>(m.w.y);
		y = fma(static_cast<double>(m.x.y), vx, y);
		y = fma(static_cast<double>(m.y.y), vy, y);
		y = fma(static_cast<double>(m.z.y), vz, y);

		double z = static_cast<double>(m.w.z);
		z = fma(static_cast<double>(m.x.z), vx, z);
		z = fma(static_cast<double>(m.y.z), vy, z);
		z = fma(static_cast<double>(m.z.z), vz, z);


		return {x, y, z};
	}
}
