#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <cmath>
#include <vector>

#include "propagation/raytracing/sbr.h"

using Catch::Matchers::WithinAbs;
namespace rt = propagation::raytracing;

namespace
{
	// A single right triangle in the z=0 plane: v0=(0,0,0), v1=(1,0,0), v2=(0,1,0), area 0.5,
	// outward normal +Z (CCW winding as seen from +Z).
	struct TestScene
	{
		std::vector<rt::Float3> vertices{{0.0f, 0.0f, 0.0f}, {1.0f, 0.0f, 0.0f}, {0.0f, 1.0f, 0.0f}};
		std::vector<rt::Vec3u> indices{{0, 1, 2}};
		std::vector<rt::Float3> normals{{0.0f, 0.0f, 1.0f}};
		rt::TriangleMeshView mesh{vertices.data(), indices.data(), normals.data(), 1};
		rt::Material material{rt::Complex(1.0, 0.0), rt::Complex(1.0, 0.0)};

		// Identity object-to-world / world-to-object transforms.
		rt::Float3x4 identity{rt::Float3{1, 0, 0}, rt::Float3{0, 1, 0}, rt::Float3{0, 0, 1}, rt::Float3{0, 0, 0}};

		[[nodiscard]] rt::HitData hitAtCentroid() const
		{
			rt::HitData hit{};
			hit.mesh = &mesh;
			hit.triangle_index = 0;
			hit.bary_u = 1.0f / 3.0f;
			hit.bary_v = 1.0f / 3.0f;
			hit.mat = &material;
			hit.world_to_obj = identity;
			hit.obj_to_world = identity;
			return hit;
		}
	};

	// Ray originating 5 units above the triangle's centroid, travelling straight down (normal
	// incidence), with a unit reference field and no prior accumulated path length.
	rt::GOPathState makeIncidentRay(double weight)
	{
		rt::GOPathState ray{};
		ray.origin = rt::Double3{1.0 / 3.0, 1.0 / 3.0, 5.0};
		ray.direction = rt::Double3{0.0, 0.0, -1.0};
		ray.elec = rt::CFloat3{rt::Complex(1.0, 0.0), rt::Complex(0.0, 0.0), rt::Complex(0.0, 0.0)};
		ray.length = 0.0;
		ray.ray_index = 0;
		ray.hash = 0;
		ray.weight = weight;
		ray.source_index = 0;
		ray.time_index = 0;
		return ray;
	}

	// A constant-unit-gain isotropic model, so the destination's receive-gain weighting in
	// facetIncidentRay is a no-op (multiplies by sqrt(1) = 1) for these geometry/delay-focused tests.
	rt::AntennaModel unitGainModel{.kind = rt::AntennaKind::Isotropic, .efficiency = 1.0, .params = {}};

	double magnitude(const rt::CFloat3& v)
	{
		return std::sqrt(rt::cabs(v.x) * rt::cabs(v.x) + rt::cabs(v.y) * rt::cabs(v.y) +
						 rt::cabs(v.z) * rt::cabs(v.z));
	}

	// Always-unoccluded shadow test.
	auto unoccluded = [](rt::Float3, rt::Float3) { return INFINITY; };
}

TEST_CASE("facetIncidentRay computes delay as the full source-to-hit-to-destination distance",
		  "[raytracing][facet_incident_ray]")
{
	const TestScene scene;
	const auto hit = scene.hitAtCentroid();
	auto ray = makeIncidentRay(0.001);

	std::vector<rt::AntennaView> dst_antennas{
		{rt::Double3{1.0 / 3.0, 1.0 / 3.0, 100.0}, rt::AzEl{0.0, 0.0}, unitGainModel}};
	std::vector<double> k0{2.0 * rt::PI / 0.03};
	std::vector<rt::Contribution> contribs(1);

	rt::TraceParams params{};
	params.go_step_limit = 10;
	params.rays_per_source = 1000;
	params.rx_to_tx = false;
	params.dir.tx_to_rx.k0 = k0[0];
	params.dst_antenna_transforms = dst_antennas.data();
	params.dst_antenna_count = 1;
	params.contributions = contribs.data();

	uint32_t next_idx = 0;
	auto get_idx = [&next_idx] { return next_idx++; };

	rt::facetIncidentRay(params, &ray, hit, unoccluded, get_idx);

	REQUIRE(next_idx == 1);
	// go_dist (source to hit) = 5, po_dist (hit to destination) = 100.
	REQUIRE_THAT(contribs[0].delay, WithinAbs(105.0, 1e-9));
}

TEST_CASE("facetIncidentRay's contribution scales linearly with ray weight in the sub-triangle regime",
		  "[raytracing][facet_incident_ray]")
{
	const TestScene scene;
	const auto hit = scene.hitAtCentroid();

	std::vector<rt::AntennaView> dst_antennas{
		{rt::Double3{1.0 / 3.0, 1.0 / 3.0, 100.0}, rt::AzEl{0.0, 0.0}, unitGainModel}};
	std::vector<double> k0{2.0 * rt::PI / 0.03};
	std::vector<rt::Contribution> contribs(1);

	rt::TraceParams params{};
	params.go_step_limit = 10;
	params.rays_per_source = 1000;
	params.rx_to_tx = false;
	params.dir.tx_to_rx.k0 = k0[0];
	params.dst_antenna_transforms = dst_antennas.data();
	params.dst_antenna_count = 1;
	params.contributions = contribs.data();

	// go_dist=5, so a_tube = 25*weight; both weights below keep a_tube well under the 0.5 triangle
	// area (0.025 and 0.05), i.e. in the linear (uncapped) regime.
	auto run = [&](double weight) -> double
	{
		auto ray = makeIncidentRay(weight);
		uint32_t idx = 0;
		auto get_idx = [&idx] { return idx++; };
		rt::facetIncidentRay(params, &ray, hit, unoccluded, get_idx);
		return magnitude(contribs[0].gain);
	};

	const double mag1 = run(0.001);
	const double mag2 = run(0.002);

	REQUIRE(mag1 > 0.0);
	REQUIRE_THAT(mag2 / mag1, WithinAbs(2.0, 1e-6));
}

TEST_CASE("facetIncidentRay's contribution saturates once the tube footprint exceeds the triangle",
		  "[raytracing][facet_incident_ray]")
{
	const TestScene scene;
	const auto hit = scene.hitAtCentroid();

	std::vector<rt::AntennaView> dst_antennas{
		{rt::Double3{1.0 / 3.0, 1.0 / 3.0, 100.0}, rt::AzEl{0.0, 0.0}, unitGainModel}};
	std::vector<double> k0{2.0 * rt::PI / 0.03};
	std::vector<rt::Contribution> contribs(1);

	rt::TraceParams params{};
	params.go_step_limit = 10;
	params.rays_per_source = 1000;
	params.rx_to_tx = false;
	params.dir.tx_to_rx.k0 = k0[0];
	params.dst_antenna_transforms = dst_antennas.data();
	params.dst_antenna_count = 1;
	params.contributions = contribs.data();

	// go_dist=5, so a_tube = 25*weight; both weights well exceed the 0.5 triangle area (25 and 50),
	// i.e. both should clamp to the same full-triangle patch and produce identical contributions.
	auto run = [&](double weight) -> rt::CFloat3
	{
		auto ray = makeIncidentRay(weight);
		uint32_t idx = 0;
		auto get_idx = [&idx] { return idx++; };
		rt::facetIncidentRay(params, &ray, hit, unoccluded, get_idx);
		return contribs[0].gain;
	};

	const auto gain1 = run(1.0);
	const auto gain2 = run(2.0);

	REQUIRE_THAT(magnitude(gain1), WithinAbs(magnitude(gain2), 1e-9));
}
