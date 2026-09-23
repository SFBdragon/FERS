#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <cmath>
#include <vector>

#include "propagation/raytracing/sbr_impl.h"

using Catch::Matchers::WithinAbs;
namespace rt = propagation::raytracing;
namespace prop = propagation;

namespace
{
	// One source antenna's carrier: k0 for a 30mm wavelength.
	std::vector<float> carrier_ks{float(2.0 * prop::PI / 0.03)};
	std::vector<uint32_t> tx_to_carrier_indices{0};

	// A single right triangle in the z=0 plane: v0=(0,0,0), v1=(1,0,0), v2=(0,1,0), area 0.5,
	// outward normal +Z (CCW winding as seen from +Z). A lossless dielectric (relative_permittivity
	// != 1) so the facet has a nonzero reflection coefficient.
	struct TestScene
	{
		std::vector<prop::Float3> vertices{{0.0f, 0.0f, 0.0f}, {1.0f, 0.0f, 0.0f}, {0.0f, 1.0f, 0.0f}};
		std::vector<prop::Uint3> indices{{0, 1, 2}};
		std::vector<rt::Material> materials{{4.0f, 0.0f}};
		std::vector<rt::AntennaModel> antenna_models{{.kind = rt::AntennaKind::Isotropic,
													   .efficiency = 1.0,
													   .params = {},
													   .horizontal_pol = prop::CFloat{1.0f, 0.0f},
													   .vertical_pol = prop::CFloat{0.0f, 0.0f}}};

		// Identity object-to-world / world-to-object transforms.
		prop::Float3x4 identity{prop::Float3{1, 0, 0}, prop::Float3{0, 1, 0}, prop::Float3{0, 0, 1}, prop::Float3{0, 0, 0}};

		[[nodiscard]] rt::HitInfo hitAtCentroid() const
		{
			rt::HitInfo hit{};
			hit.triangle_index = 0;
			hit.material_index = 0;
			hit.bary_u = 1.0f / 3.0f;
			hit.bary_v = 1.0f / 3.0f;
			hit.world_to_obj = identity;
			hit.obj_to_world = identity;
			return hit;
		}
	};

	// Ray originating 5 units above the triangle's centroid, travelling straight down (normal
	// incidence), with a unit reference field and no prior accumulated path length.
	rt::PathState makeIncidentRay(double weight)
	{
		rt::PathState ray{};
		ray.origin = prop::Double3{1.0 / 3.0, 1.0 / 3.0, 5.0};
		ray.direction = prop::Double3{0.0, 0.0, -1.0};
		ray.length = 0.0;
		ray.ray_index = 0;
		ray.hash = 0;
		ray.weight = weight;
		ray.source_index = 0;
		ray.time_index = 0;
		ray.carrier_count = 1;
		ray.carrier_ks = carrier_ks.data();
		ray.tx_to_carrier_indices = tx_to_carrier_indices.data();
		ray.elec_fields[0] = prop::CFloat3{prop::CFloat(1.0f, 0.0f), prop::CFloat(0.0f, 0.0f), prop::CFloat(0.0f, 0.0f)};
		return ray;
	}

	std::vector<rt::ActiveAntenna> makeDestAntennas()
	{
		return {{.position = prop::Double3{1.0 / 3.0, 1.0 / 3.0, 100.0},
				.direction = prop::FloatAzEl{0.0f, 0.0f},
				.antenna_model_index = 0}};
	}

	rt::SbrParams makeParams(const TestScene& scene, const std::vector<rt::ActiveAntenna>& dst_antennas,
							 std::vector<rt::RxFlags>& rx_flags, std::vector<rt::Contribution>& contribs)
	{
		rt::SbrParams params{};
		params.go_step_limit = 10;
		params.rays_per_source = 1000;
		params.vertices = scene.vertices.data();
		params.indices = scene.indices.data();
		params.materials = scene.materials.data();
		params.antenna_models = scene.antenna_models.data();
		params.dest_antennae = dst_antennas.data();
		params.dest_antenna_count = static_cast<uint32_t>(dst_antennas.size());
		params.rx_flags = rx_flags.data();
		params.rx_to_tx = false;
		params.contributions = contribs.data();
		return params;
	}

	// Always-unoccluded shadow test.
	auto unoccluded = [](prop::Float3, prop::Float3) { return INFINITY; };
}

TEST_CASE("facetIncidentRay computes delay as the full source-to-hit-to-destination distance",
		  "[raytracing][facet_incident_ray]")
{
	const TestScene scene;
	const auto hit = scene.hitAtCentroid();
	auto ray = makeIncidentRay(0.001);

	const auto dst_antennas = makeDestAntennas();
	std::vector<rt::RxFlags> rx_flags{static_cast<rt::RxFlags>(0)};
	std::vector<rt::Contribution> contribs(1);
	const auto params = makeParams(scene, dst_antennas, rx_flags, contribs);

	uint32_t next_idx = 0;
	auto get_idx = [&next_idx] { return next_idx++; };

	(void)rt::facetIncidentRay(params, &ray, hit, unoccluded, get_idx);

	REQUIRE(next_idx == 1);
	// go_dist (source to hit) = 5, po_dist (hit to destination) = 100, converted to a time delay.
	REQUIRE_THAT(contribs[0].delay, WithinAbs(105.0 / prop::C, 1e-15));
}

TEST_CASE("facetIncidentRay's contribution scales linearly with ray weight in the sub-triangle regime",
		  "[raytracing][facet_incident_ray]")
{
	const TestScene scene;
	const auto hit = scene.hitAtCentroid();
	const auto dst_antennas = makeDestAntennas();

	// go_dist=5, so a_tube = 25*weight; both weights below keep a_tube well under the 0.5 triangle
	// area (0.025 and 0.05), i.e. in the linear (uncapped) regime.
	auto run = [&](double weight) -> double
	{
		std::vector<rt::RxFlags> rx_flags{static_cast<rt::RxFlags>(0)};
		std::vector<rt::Contribution> contribs(1);
		const auto params = makeParams(scene, dst_antennas, rx_flags, contribs);

		auto ray = makeIncidentRay(weight);
		uint32_t idx = 0;
		auto get_idx = [&idx] { return idx++; };
		(void)rt::facetIncidentRay(params, &ray, hit, unoccluded, get_idx);
		return prop::cabs(contribs[0].voltage);
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
	const auto dst_antennas = makeDestAntennas();

	// go_dist=5, so a_tube = 25*weight; both weights well exceed the 0.5 triangle area (25 and 50),
	// i.e. both should clamp to the same full-triangle patch and produce identical contributions.
	auto run = [&](double weight) -> double
	{
		std::vector<rt::RxFlags> rx_flags{static_cast<rt::RxFlags>(0)};
		std::vector<rt::Contribution> contribs(1);
		const auto params = makeParams(scene, dst_antennas, rx_flags, contribs);

		auto ray = makeIncidentRay(weight);
		uint32_t idx = 0;
		auto get_idx = [&idx] { return idx++; };
		(void)rt::facetIncidentRay(params, &ray, hit, unoccluded, get_idx);
		return prop::cabs(contribs[0].voltage);
	};

	const double mag1 = run(1.0);
	const double mag2 = run(2.0);

	REQUIRE_THAT(mag1, WithinAbs(mag2, 1e-9));
}
