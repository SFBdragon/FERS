#include <algorithm>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <cmath>
#include <vector>

#include "math/geometry_ops.h"
#include "propagation/raytracing/sbr.h"

using Catch::Matchers::WithinAbs;
namespace rt = propagation::raytracing;

TEST_CASE("rotateLocalToWorld's boresight matches Vec3(SVec3(...))", "[raytracing][math]")
{
	const rt::AzEl rot{0.3, -0.2};
	const rt::Double3 local_boresight{1.0, 0.0, 0.0};
	const auto world = rt::rotateLocalToWorld(rot, local_boresight);

	const math::Vec3 expected(math::SVec3(1.0, rot.azimuth, rot.elevation));
	REQUIRE_THAT(world.x, WithinAbs(expected.x, 1e-12));
	REQUIRE_THAT(world.y, WithinAbs(expected.y, 1e-12));
	REQUIRE_THAT(world.z, WithinAbs(expected.z, 1e-12));
}

TEST_CASE("rotateLocalToWorld is the identity at zero rotation", "[raytracing][math]")
{
	const rt::AzEl rot{0.0, 0.0};
	const rt::Double3 local{std::cos(0.6) * std::cos(-0.4), std::sin(0.6) * std::cos(-0.4), std::sin(-0.4)};

	const auto world = rt::rotateLocalToWorld(rot, local);
	REQUIRE_THAT(world.x, WithinAbs(local.x, 1e-12));
	REQUIRE_THAT(world.y, WithinAbs(local.y, 1e-12));
	REQUIRE_THAT(world.z, WithinAbs(local.z, 1e-12));
}

TEST_CASE("rotateLocalToWorld/rotateWorldToLocal round-trip", "[raytracing][math]")
{
	const rt::AzEl rot{0.9, 0.4};
	constexpr double az_l = -0.6, el_l = 0.25;
	const rt::Double3 local{std::cos(az_l) * std::cos(el_l), std::sin(az_l) * std::cos(el_l), std::sin(el_l)};

	const auto world = rt::rotateLocalToWorld(rot, local);
	const auto back = rt::rotateWorldToLocal(rot, world);

	REQUIRE_THAT(back.x, WithinAbs(local.x, 1e-12));
	REQUIRE_THAT(back.y, WithinAbs(local.y, 1e-12));
	REQUIRE_THAT(back.z, WithinAbs(local.z, 1e-12));
}

TEST_CASE("sampleAntennaPattern reproduces exact grid values at grid points", "[raytracing][math]")
{
	// 3x3 grid: az in {-pi, 0, pi}, el in {-pi/2, 0, pi/2}, row-major by elevation.
	std::vector<float> gains{
		1.0f, 2.0f, 3.0f, // el = -pi/2
		4.0f, 5.0f, 6.0f, // el = 0
		7.0f, 8.0f, 9.0f, // el = pi/2
	};
	const rt::AntennaPatternView view{gains.data(), 3, 3};

	REQUIRE_THAT(rt::sampleAntennaPattern(view, -rt::PI, -rt::PI / 2.0), WithinAbs(1.0, 1e-6));
	REQUIRE_THAT(rt::sampleAntennaPattern(view, 0.0, -rt::PI / 2.0), WithinAbs(2.0, 1e-6));
	REQUIRE_THAT(rt::sampleAntennaPattern(view, 0.0, 0.0), WithinAbs(5.0, 1e-6));
	REQUIRE_THAT(rt::sampleAntennaPattern(view, rt::PI, rt::PI / 2.0), WithinAbs(9.0, 1e-6));
}

TEST_CASE("sampleAntennaPattern bilinearly interpolates between grid points", "[raytracing][math]")
{
	// 2x2 grid: (az,el) corners -> gain: (-pi,-pi/2)=0, (pi,-pi/2)=10, (-pi,pi/2)=20, (pi,pi/2)=30.
	std::vector<float> gains{0.0f, 10.0f, 20.0f, 30.0f};
	const rt::AntennaPatternView view{gains.data(), 2, 2};

	REQUIRE_THAT(rt::sampleAntennaPattern(view, 0.0, -rt::PI / 2.0), WithinAbs(5.0, 1e-6)); // az midpoint, el0 row
	REQUIRE_THAT(rt::sampleAntennaPattern(view, -rt::PI, 0.0), WithinAbs(10.0, 1e-6)); // az0, el midpoint
	REQUIRE_THAT(rt::sampleAntennaPattern(view, 0.0, 0.0), WithinAbs(15.0, 1e-6)); // center of all 4 corners
}

TEST_CASE("sampleAntennaPattern clamps queries outside the grid's range", "[raytracing][math]")
{
	std::vector<float> gains{0.0f, 10.0f, 20.0f, 30.0f};
	const rt::AntennaPatternView view{gains.data(), 2, 2};

	REQUIRE_THAT(rt::sampleAntennaPattern(view, -100.0, -100.0), WithinAbs(0.0, 1e-6));
	REQUIRE_THAT(rt::sampleAntennaPattern(view, 100.0, 100.0), WithinAbs(30.0, 1e-6));
}

TEST_CASE("sampleAntennaPattern returns 0 for a degenerate view", "[raytracing][math]")
{
	const rt::AntennaPatternView view{nullptr, 0, 0};
	REQUIRE_THAT(rt::sampleAntennaPattern(view, 0.0, 0.0), WithinAbs(0.0, 1e-6));
}

TEST_CASE("sampleIsotropicDirection produces unit vectors spanning the sphere", "[raytracing][math]")
{
	constexpr uint32_t num_rays = 2000;
	rt::Double3 sum{0.0, 0.0, 0.0};
	double min_z = 1.0, max_z = -1.0;

	for (uint32_t i = 0; i < num_rays; ++i)
	{
		const auto dir = rt::sampleIsotropicDirection(i, num_rays);
		const double len = std::sqrt(dir.x * dir.x + dir.y * dir.y + dir.z * dir.z);
		REQUIRE_THAT(len, WithinAbs(1.0, 1e-9));

		sum.x += dir.x;
		sum.y += dir.y;
		sum.z += dir.z;
		min_z = std::min(min_z, dir.z);
		max_z = std::max(max_z, dir.z);
	}

	// Points should be roughly symmetric about the origin for a large, evenly-spread sample.
	REQUIRE_THAT(sum.x / num_rays, WithinAbs(0.0, 1e-2));
	REQUIRE_THAT(sum.y / num_rays, WithinAbs(0.0, 1e-2));
	REQUIRE_THAT(sum.z / num_rays, WithinAbs(0.0, 1e-2));

	// And should actually span from near one pole to the other.
	REQUIRE(min_z < -0.99);
	REQUIRE(max_z > 0.99);
}

TEST_CASE("sampleIsotropicDirection is deterministic", "[raytracing][math]")
{
	const auto a = rt::sampleIsotropicDirection(17, 500);
	const auto b = rt::sampleIsotropicDirection(17, 500);
	REQUIRE_THAT(a.x, WithinAbs(b.x, 1e-15));
	REQUIRE_THAT(a.y, WithinAbs(b.y, 1e-15));
	REQUIRE_THAT(a.z, WithinAbs(b.z, 1e-15));
}

namespace
{
	// v0=(0,0,0), v1=(1,0,0), v2=(0,1,0): a right triangle in the z=0 plane, area 0.5, outward
	// normal +Z (CCW as seen from +Z, matching TriangleMeshView's winding convention).
	const rt::Double3 test_tri[3]{{0.0, 0.0, 0.0}, {1.0, 0.0, 0.0}, {0.0, 1.0, 0.0}};
	constexpr double test_tri_area = 0.5;
	const rt::Double3 test_norm{0.0, 0.0, 1.0};
	const rt::Double3 test_hit_pos{1.0 / 3.0, 1.0 / 3.0, 0.0}; // centroid
	const rt::Double3 test_incoming_dir{0.0, 0.0, -1.0}; // normal incidence, travelling in -Z
}

TEST_CASE("computeTubePatch shrinks to the tube footprint when it's smaller than the triangle",
		  "[raytracing][math]")
{
	constexpr double dist = 1.0;
	constexpr double solid_angle = 0.01; // a_tube = dist^2 * solid_angle / cos(0) = 0.01 < 0.5
	const auto patch =
		rt::computeTubePatch(test_tri, test_hit_pos, test_norm, test_incoming_dir, test_tri_area, dist, solid_angle);

	REQUIRE_THAT(patch.area, WithinAbs(0.01, 1e-9));
	// Scaled toward the hit point, so each vertex should lie on the segment from the hit point to
	// the original vertex.
	const double scale = std::sqrt(0.01 / 0.5);
	for (int i = 0; i < 3; ++i)
	{
		const auto expected = test_hit_pos + scale * (test_tri[i] - test_hit_pos);
		REQUIRE_THAT(patch.verts[i].x, WithinAbs(expected.x, 1e-9));
		REQUIRE_THAT(patch.verts[i].y, WithinAbs(expected.y, 1e-9));
		REQUIRE_THAT(patch.verts[i].z, WithinAbs(expected.z, 1e-9));
	}
}

TEST_CASE("computeTubePatch clamps to the full triangle when the tube footprint is larger",
		  "[raytracing][math]")
{
	constexpr double dist = 100.0;
	constexpr double solid_angle = 1.0; // a_tube = 10000, way over the 0.5 triangle area
	const auto patch =
		rt::computeTubePatch(test_tri, test_hit_pos, test_norm, test_incoming_dir, test_tri_area, dist, solid_angle);

	REQUIRE_THAT(patch.area, WithinAbs(test_tri_area, 1e-9));
	for (int i = 0; i < 3; ++i)
	{
		REQUIRE_THAT(patch.verts[i].x, WithinAbs(test_tri[i].x, 1e-9));
		REQUIRE_THAT(patch.verts[i].y, WithinAbs(test_tri[i].y, 1e-9));
		REQUIRE_THAT(patch.verts[i].z, WithinAbs(test_tri[i].z, 1e-9));
	}
}

TEST_CASE("computeTubePatch stays finite at grazing incidence", "[raytracing][math]")
{
	// Travelling almost parallel to the triangle plane - cos(theta_i) would be near 0 unclamped.
	const rt::Double3 grazing_dir = normalize(rt::Double3{1.0, 0.0, -1e-6});
	const auto patch =
		rt::computeTubePatch(test_tri, test_hit_pos, test_norm, grazing_dir, test_tri_area, 1.0, 0.001);

	REQUIRE(std::isfinite(patch.area));
	REQUIRE(patch.area >= 0.0);
	REQUIRE(patch.area <= test_tri_area + 1e-9);
}

TEST_CASE("arbitraryPerpendicular returns a unit vector perpendicular to its input", "[raytracing][math]")
{
	const std::vector<rt::Double3> dirs{
		{1.0, 0.0, 0.0},  {0.0, 1.0, 0.0},
		{0.0, 0.0, 1.0}, // near/at the "up" pole - exercises the fallback branch
		{0.0, 0.0, -1.0}, {0.5773502691896258, 0.5773502691896258, 0.5773502691896258}, // arbitrary diagonal
	};

	for (const auto& dir : dirs)
	{
		const auto perp = rt::arbitraryPerpendicular(dir);
		const double len = std::sqrt(perp.x * perp.x + perp.y * perp.y + perp.z * perp.z);
		const double dot = perp.x * dir.x + perp.y * dir.y + perp.z * dir.z;

		REQUIRE_THAT(len, WithinAbs(1.0, 1e-9));
		REQUIRE_THAT(dot, WithinAbs(0.0, 1e-9));
	}
}
