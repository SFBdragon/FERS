// SPDX-License-Identifier: GPL-2.0-only
//
// Copyright (c) 2026-present FERS Contributors (see AUTHORS.md).
//
// See the GNU GPLv2 LICENSE file in the FERS project root for more information.

#pragma once

#include <cassert>
#include <cstdint>

#include "propagation/math.h"

// This file defines shared definitions for types used by the SBR-like engine
// for the RayTracingModel propagation model.
//
// ## On Precision
//
// See sbr_impl.h's section of the same name.

namespace propagation::raytracing
{
	constexpr uint32_t MAX_GO_STEPS = 8;
	constexpr uint32_t MAX_CARRIERS = 32;

	/// SBR engine triangle mesh view.
	struct TriangleMeshView
	{
		/// The offset of this mesh's vertex buffer within the SbrParams::vertices buffer.
		uint32_t vertex_buffer_offset;
		/// The number of vertices in this mesh's buffer.
		uint32_t vertex_count;
		/// The offset of this mesh's index buffer within the SbrParams::indices buffer.
		///
		/// Note that the indices index globally into the vertex buffer, not from `vertex_buffer_offset`.
		uint32_t index_buffer_index;
		/// The number of triangles in this mesh's buffer. Equal to the number of Uint3 index sets.
		uint32_t triangle_count;
	};

	/// SBR engine material.
	struct Material
	{
		/// The relative permittivity of the material.
		///
		/// This is used in conjuction with the `conductivity` to compute the
		/// complex index of refraction: n = sqrt(ε_r' - j·σ/(ω·ε0))
		float relative_permittivity;

		/// The conductivity of the material, Siemens per meter.
		///
		/// Zero for dielectrics, infinity for perfect electric conductors.
		float conductivity;
	};

	struct ActiveCarriers
	{
		uint32_t tx_to_carrier_buffer_offset;
		uint32_t carrier_ks_buffer_offset;
		uint32_t carrier_count;
	};

	/// SBR engine view of baked antenna gain pattern.
	struct AntennaPatternView
	{
		/// Dense az/el grid of raw linear gain, in the antenna's local frame.
		/// Boresight = local +X = local azimuth=0, elevation=0.
		///
		/// Index as `gains[el_idx * az_count + az_idx]`.
		/// There are `az_count` samples per elevation spanning `[-pi, pi]`.
		/// There are `el_count` samples per azimuth spanning `[-pi/2, pi/2]`.
		/// Inclusive at both ends, so index 0 and index count-1 land exactly on the boundary.
		uint32_t gains_buffer_offset;
		uint32_t az_count;
		uint32_t el_count;
	};

	/// Discriminates `AntennaModel`'s payload.
	enum class AntennaKind : uint8_t
	{
		Isotropic,
		Sinc,
		Gaussian,
		SquareHorn,
		Parabolic,
		Grid2D,
	};

	/// Device-side antenna gain model.
	///
	/// A tagged union over the parameterised and tabulated representations.
	struct AntennaModel
	{
		AntennaKind kind = AntennaKind::Isotropic;
		double efficiency = 1.0;
		union
		{
			struct
			{
				double alpha, beta, gamma;
			} sinc;

			struct
			{
				double azimuth_scale, elevation_scale;
			} gaussian;

			struct
			{
				double dimension;
			} square_horn;

			struct
			{
				double diameter;
			} parabolic;

			struct
			{
				AntennaPatternView grid;
			} grid_2d;
		} params{};

		CFloat horizontal_pol;
		CFloat vertical_pol;
	};

	struct ActiveAntenna
	{
		Double3 position;
		FloatAzEl direction;
		uint32_t antenna_model_index{};
	};


	struct Contribution
	{
		CFloat voltage;
		double delay{};
		uint64_t path_id{};
		/// source_index + time_index * source_count
		uint32_t source_times_index{};
		/// Which destination antenna was reached.
		///
		/// forward tracing (FindTxToRx) traces from one transmitter to all receivers, so this is an
		/// index into receivers; backward tracing (FindRxFromTx) traces from one receiver to all
		/// active streaming sources, so this is an index into those sources.
		uint32_t dest_index{};
	};

	enum RxFlags : uint8_t
	{
		/// Disable 1/R sphere spreading.
		FLAG_NOSPREADING = 1,
		/// Disable direct paths to the receiver.
		FLAG_NODIRECT = 2,
	};

	struct SbrParams
	{
		/// The number of GO steps allowed.
		/// (Reflections/transmissions/diffractions.)
		///
		/// Note that PO is performed at each hit point, so zero GO bounces
		/// still means that radar effectively bounces once.
		uint32_t go_step_limit;

		/// Number of ray tubes launched per source antenna (the launch's X dimension). `genRay` needs
		/// this to compute each tube's solid angle (dOmega = 4*pi / rays_per_source for the isotropic
		/// MVP sampler).
		uint32_t rays_per_source;

		/// Triangle mesh vertex buffer.
		const Float3* vertices;
		/// Triangle mesh index buffer.
		///
		/// Winding order convention: vertices v0=vertices[x], v1=vertices[y], v2=vertices[z]
		/// are wound counter-clockwise from the outside of the mesh, inward (front face).
		/// Thus the outward facing normal thus obeys the right-hand rule.
		/// OptiX and Embree use the same convention.
		const Uint3* indices;
		/// Materials buffer.
		const Material* materials;
		/// Antenna model buffer.
		const AntennaModel* antenna_models;
		/// Antenna gains buffer.
		const float* antenna_gains;

		/// Destination antennae positions and orientations.
		/// This is `src_antenna_count * times` in length.
		/// Indexed by `src_antennas[src_antenna_idx + src_antenna_count * time_idx]`.
		const ActiveAntenna* source_antennae;
		/// The number of source antennas to trace from.
		uint32_t source_antenna_count;
		/// Destination antennae positions and orientations.
		/// This is `src_antenna_count * times` in length.
		/// Indexed by `src_antennas[src_antenna_idx + src_antenna_count * time_idx]`.
		const ActiveAntenna* dest_antennae;
		/// The number of destination antennas to trace to.
		uint32_t dest_antenna_count;

		/// Indexed by `time_index`.
		const ActiveCarriers* carriers;
		///
		/// Indexed by [transmitter_antenna_index + time_index * transmitter_antenna_count]
		uint32_t* tx_to_carrier_indices;
		///
		/// Indexed by `tx_to_carrier_indices`.
		float* carrier_ks;

		/// Receiver flags.
		RxFlags* rx_flags;


		/// Whether we're tracing backwards from receivers to transmitters.
		///
		/// forward/findTxToRxPaths
		/// - one transmitter
		/// - at a sequence of times
		/// - to all receivers
		///
		/// backward/findRxFromTxPaths
		/// - currently one receiver, but that should probably change
		/// - currently one time, but that should probably change
		/// - to all active streaming sources
		bool rx_to_tx;

		// Currently the per-instance data is limited to a material index.
		// If there's a need to go beyond that, turn that into an instance
		// index which indexes into per-instance data here.
		// Instance* instances;

		// Output buffer of contributions. Indexing this is engine-specific.
		Contribution* contributions;
	};

} // namespace propagation::raytracing
