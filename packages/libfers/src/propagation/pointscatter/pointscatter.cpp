// SPDX-License-Identifier: GPL-2.0-only
//
// Copyright (c) 2026-present FERS Contributors (see AUTHORS.md).
//
// See the GNU GPLv2 LICENSE file in the FERS project root for more information.

#include "pointscatter.h"

#include <cassert>
#include <cmath>
#include <vector>

#include "core/config.h"
#include "core/logging.h"
#include "core/parameters.h"
#include "core/simulation_state.h"
#include "core/world.h"
#include "math/geometry_ops.h"
#include "propagation/common.h"
#include "propagation/math.h"
#include "propagation/propagation_model.h"
#include "radar/radar_obj.h"
#include "radar/receiver.h"
#include "radar/transmitter.h"

namespace propagation::pointscatter
{
	/**
	 * @brief Calculates the antenna gain for a specific direction and time.
	 * @param radar The radar object (Transmitter or Receiver).
	 * @param direction_vec The unit vector pointing AWAY from the antenna towards the target/receiver.
	 * @param time The simulation time for rotation lookup.
	 * @param lambda The signal wavelength.
	 * @return The linear gain value.
	 */
	static RealType computeAntennaGain(const radar::Radar* radar, const math::Vec3& direction_vec, RealType time,
									   RealType lambda)
	{
		return radar->getGain(math::SVec3(direction_vec), radar->getRotation(time), lambda);
	}

	Complex3<RealType> computeAntennaPolarisation(const radar::Radar* radar, const math::Vec3& direction, RealType time)
	{
		const auto pol = radar->getAntenna()->getPolarisation();
		const auto h = Complex{pol.horizontal.real(), pol.horizontal.imag()};
		const auto v = Complex{pol.veritcal.real(), pol.veritcal.imag()};
		const auto fers_orient = radar->getRotation(time);
		const AzEl orientation{fers_orient.azimuth, fers_orient.elevation};
		const RealType inv_len = 1.0 / direction.length();
		const Real3<RealType> world_dir{direction.x * inv_len, direction.y * inv_len, direction.z * inv_len};
		const Real3<RealType> local_dir = rotateWorldToLocal(orientation, world_dir);
		return antennaPolarizationVector(h, v, orientation, local_dir);
	}

	ComplexType computeReflectedPathGain(RealType tx_gain, RealType rx_gain, const Complex3<RealType>& tx_pol,
										 const Complex3<RealType>& rx_pol, const math::Vec3& tx_to_tgt,
										 const math::Vec3& tgt_to_rx, RealType r_tx, RealType r_rx, RealType rcs,
										 RealType lambda, bool no_prop_loss)
	{
		const auto k_in = Real3<RealType>{tx_to_tgt.x, tx_to_tgt.y, tx_to_tgt.z} / r_tx;
		const auto k_scat = Real3<RealType>{tgt_to_rx.x, tgt_to_rx.y, tgt_to_rx.z} / r_rx;

		// The facet normal bisects the incidence and scatter directions.
		const auto bisector_sum = -k_in + k_scat;
		const auto bisector_len = length(bisector_sum);
		// Assert against the k_in ~= k_scat case, aka forward scatter.
		// This function contractually doesn't handle this case. No reflection occurs.
		assert(bisector_len > RealType(1e-9));
		const auto bisection = bisector_sum / bisector_len;

		const auto basis = incidencePlaneBasis(bisection, k_in);

		const auto e_te = dot(tx_pol, basis.e_te);
		const auto e_tm = dot(tx_pol, basis.e_tm_in);

		const auto sqrt_rcs = std::sqrt(rcs);
		// Assume the Sinclair matrix for a PEC for now. TODO_SHAUN add polarimetric scattering matrices.
		const auto sinclair = Real2x2<RealType>{{sqrt_rcs, 0}, {0, -sqrt_rcs}};
		const auto e_scat = sinclair * Complex2{e_tm, e_te};

		const auto e_tm_scat = cross(basis.e_te, k_scat);
		const auto refl = e_tm_scat * e_scat.x + basis.e_te * e_scat.y;
		const auto pol = dot_no_conj(refl, rx_pol);

		RealType scalar = std::sqrt(tx_gain * rx_gain * (1 / (64.0 * PI * PI * PI))) * lambda;
		if (!no_prop_loss)
		{
			scalar /= r_tx * r_rx;
		}

		const auto ratio = scalar * pol;
		return ComplexType{ratio.re, ratio.im};
	}


	/**
	 * @brief Determine the PropagationPath along a direct tx->rx path
	 * @param time The time at which to find propagation paths. May be at reception or transmission time.
	 * @param tx_time True if time is a transmission time, otherwise it's considered to be the receiving time.
	 */
	static void directPath(const radar::Transmitter& tx, radar::Receiver* rx, const RealType time,
						   const bool is_tx_time, const RealType segment_start, size_t source_index,
						   std::vector<PropagationPath>& found_paths)
	{

		// Calculate the direct path contribution.
		//
		// Handle the co-location cases:
		// 1. If explicitly attached (monostatic), skip (internal leakage handled elsewhere).
		// 2. If independent but on the same platform, distance is 0. Far-field logic (1/R^2)
		//    diverges. We skip calculation, assuming no direct coupling/interference for
		//    co-located far-field antennas.
		if (tx.getPlatform() == rx->getPlatform())
		{
			return;
		}

		const auto p_tx = tx.getPosition(time);
		const auto p_rx = rx->getPosition(time); // Assumption: v_rx << c
		const auto tx_to_rx = p_rx - p_tx;
		const auto tx_to_rx_dist = tx_to_rx.length();

		// Handle co-location of independent TX/RX on distinct platforms.
		// Similar to case 2 above, far-field logic diverges and assume no direct interference
		// for co-located antennas.
		if (tx_to_rx_dist <= EPSILON)
		{
			return;
		}

		const auto delay = tx_to_rx_dist / params::c();
		const RealType tx_time = is_tx_time ? time : time - delay;
		const RealType rx_time = is_tx_time ? time + delay : time;

		const auto carrier_opt = tx.getSignal()->getModulatedCarrier(tx_time - segment_start);
		if (!carrier_opt.has_value())
		{
			return;
		}
		const RealType carrier = carrier_opt.value();
		const RealType lambda = params::c() / carrier;

		// Tx Gain and Polarisation: Direction Tx -> Rx
		const auto tx_gain = computeAntennaGain(&tx, tx_to_rx, tx_time, lambda);
		const auto tx_pol = computeAntennaPolarisation(&tx, tx_to_rx, tx_time);
		// Rx Gain and Polarisation: Direction Rx -> Tx
		const auto rx_gain = computeAntennaGain(rx, -tx_to_rx, rx_time, lambda);
		const auto rx_pol = computeAntennaPolarisation(rx, -tx_to_rx, tx_time);

		const bool no_loss = rx->checkFlag(radar::Receiver::RecvFlag::FLAG_NOPROPLOSS);
		const auto gain = computeDirectPathGain(tx_gain, rx_gain, tx_pol, rx_pol, lambda, tx_to_rx_dist, no_loss);

		found_paths.emplace_back(PropagationPath{
			.delay = delay,
			.gain = ComplexType{gain.re, gain.im},
			.path_id = hash_mix(hash_mix(HASH_SEED, tx.getId()), rx->getId()),
			.source_index = source_index,
			.receiver = rx,
		});
	}

	static void bistaticPath(const radar::Transmitter& tx, radar::Receiver* rx, const radar::Target& target,
							 const RealType time, const bool is_tx_time, const RealType segment_start,
							 size_t source_index, std::vector<PropagationPath>& found_paths)
	{

		// If calculating reflected path and target is co-located with either Tx or Rx:
		// Skip to avoid singularity. Simulating a radar tracking its own platform
		// requires near-field clutter models, not point-target RCS models.
		if (target.getPlatform() == tx.getPlatform() || target.getPlatform() == rx->getPlatform())
		{
			return;
		}

		const auto p_tx = tx.getPosition(time);
		const auto p_tgt = target.getPosition(time);
		const auto p_rx = rx->getPosition(time);
		const auto tx_to_tgt = p_tgt - p_tx;
		const auto tx_to_tgt_dist = tx_to_tgt.length();
		const auto tgt_to_rx = p_rx - p_tgt;
		const auto tgt_to_rx_dist = tgt_to_rx.length();

		// Handle co-location of independent TX/TGT/RX on distinct platforms.
		// Similar to the above case, skip to avoid a signularity.
		if (tx_to_tgt_dist <= EPSILON || tgt_to_rx_dist <= EPSILON)
		{
			LOG(logging::Level::TRACE,
				"Skipping reflected path calculation for Target {} co-located with Transmitter {} or Receiver {}",
				target.getName(), tx.getName(), rx->getName());

			return;
		}

		// Skip near-exact forward scatter (Rx directly behind the target as seen from Tx).
		// This isn't handled accurately by specular reflection physics, it's dominated by diffraction
		// and other effects. The direct path is already computed separately, this skips "reflecting"
		// off of a in-between target for now.
		if (math::dotProduct(tx_to_tgt / tx_to_tgt_dist, tgt_to_rx / tgt_to_rx_dist) > 1.0 - EPSILON)
		{
			LOG(logging::Level::TRACE,
				"Skipping reflected path calculation for Target {}: Rx is in the forward-scatter direction from Tx "
				"through the target, which is not handled as a bistatic reflection path for RCS point targets",
				target.getName());

			return;
		}

		const auto delay = (tx_to_tgt_dist + tgt_to_rx_dist) / params::c();
		RealType tx_time = is_tx_time ? time : time - delay;
		RealType rx_time = is_tx_time ? time + delay : time;
		const auto target_delay = tx_to_tgt_dist / params::c();
		RealType tgt_time = tx_time + target_delay;

		const auto carrier_opt = tx.getSignal()->getModulatedCarrier(tx_time - segment_start);
		if (!carrier_opt.has_value())
			return;
		const RealType carrier = carrier_opt.value();
		const RealType lambda = params::c() / carrier;

		// Calculate RCS
		// InAngle: Tx -> Tgt (link_tx_tgt.u_vec)
		// OutAngle: Rx -> Tgt (Opposite of Tgt->Rx, so -link_tgt_rx.u_vec)
		math::SVec3 in_angle(tx_to_tgt);
		math::SVec3 out_angle(-tgt_to_rx);
		const auto rcs = target.getRcs(in_angle, out_angle, tgt_time);

		// Tx Gain: Direction Tx -> Tgt
		const auto tx_gain = computeAntennaGain(&tx, tx_to_tgt, tx_time, lambda);
		const auto tx_pol = computeAntennaPolarisation(&tx, tx_to_tgt, tx_time);
		// Rx Gain: Direction Rx -> Tgt
		const auto rx_gain = computeAntennaGain(rx, -tgt_to_rx, rx_time, lambda);
		const auto rx_pol = computeAntennaPolarisation(rx, -tgt_to_rx, tx_time);

		const bool no_loss = rx->checkFlag(radar::Receiver::RecvFlag::FLAG_NOPROPLOSS);
		const auto gain = computeReflectedPathGain(tx_gain, rx_gain, tx_pol, rx_pol, tx_to_tgt, tgt_to_rx,
												   tx_to_tgt_dist, tgt_to_rx_dist, rcs, lambda, no_loss);

		found_paths.emplace_back(PropagationPath{
			.delay = delay,
			.gain = gain,
			.path_id = hash_mix(hash_mix(hash_mix(HASH_SEED, tx.getId()), target.getId()), rx->getId()),
			.source_index = source_index,
			.receiver = rx,
		});
	}

	std::vector<PathsAtTime> PointScatterModel::findTxToRxPaths(ThreadContext* /* ctx */,
																const radar::Transmitter& transmitter,
																const std::vector<RealType>& times) const
	{
		std::vector<PathsAtTime> timesteps;

		for (const auto current_time : times)
		{
			std::vector<PropagationPath> paths;

			for (const auto& receiver : _world->getReceivers())
			{
				// Note, we're passing zero as the source as the caller of findTxToRxPaths
				// only gives us a Transmitter, not a ActiveStreamingSource.
				// findTxToRxPaths is used for pulsed radar instead of continuous, so this is fine.

				if (!receiver->checkFlag(radar::Receiver::RecvFlag::FLAG_NODIRECT))
				{
					// Calculate the direct path contribution, if any.
					directPath(transmitter, receiver.get(), current_time, true, current_time, 0, paths);
				}

				for (const auto& target : _world->getTargets())
				{
					// Calculate bistatic reflection contribution, if any.
					bistaticPath(transmitter, receiver.get(), *target, current_time, true, current_time, 0, paths);
				}
			}

			timesteps.push_back({paths, current_time});
		}

		return timesteps;
	};

	std::vector<PropagationPath>
	PointScatterModel::findRxFromTxPaths(ThreadContext* /* ctx */, radar::Receiver* receiver,
										 const std::vector<core::ActiveStreamingSource>& sources,
										 RealType rx_time) const
	{
		std::vector<PropagationPath> paths;

		for (size_t s = 0; s < sources.size(); s++)
		{
			const auto& source = sources[s];

			if (!receiver->checkFlag(radar::Receiver::RecvFlag::FLAG_NODIRECT))
			{
				// Calculate the direct path contribution, if any.
				directPath(*source.transmitter, receiver, rx_time, false, source.segment_start, s, paths);
			}

			for (const auto& target : _world->getTargets())
			{
				// Calculate bistatic reflection contribution, if any.
				bistaticPath(*source.transmitter, receiver, *target, rx_time, false, source.segment_start, s, paths);
			}
		}

		return paths;
	}

	PointScatterModel::PointScatterModel(core::World* world) : _world(world) {}
}
