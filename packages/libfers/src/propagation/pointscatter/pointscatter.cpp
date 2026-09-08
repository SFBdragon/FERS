//
// TODO
//

#include "pointscatter.h"

#include <vector>

#include "core/config.h"
#include "core/logging.h"
#include "core/parameters.h"
#include "core/simulation_state.h"
#include "core/world.h"
#include "math/geometry_ops.h"
#include "propagation/propagation_model.h"
#include "radar/radar_obj.h"
#include "radar/receiver.h"
#include "radar/transmitter.h"

using namespace math;
using namespace radar;

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
	static RealType computeAntennaGain(const Radar* radar, const Vec3& direction_vec, RealType time, RealType lambda)
	{
		return radar->getGain(SVec3(direction_vec), radar->getRotation(time), lambda);
	}

	/**
	 * @brief Computes the power scaling factor for a direct path (Friis Transmission Equation).
	 * @param tx_gain Transmitter gain (linear).
	 * @param rx_gain Receiver gain (linear).
	 * @param lambda Wavelength (meters).
	 * @param dist Distance (meters).
	 * @param no_prop_loss If true, distance-based attenuation is ignored.
	 * @return The power scaling factor (Pr / Pt).
	 */
	static RealType computeDirectPathPower(RealType tx_gain, RealType rx_gain, RealType lambda, RealType dist,
										   bool no_prop_loss)
	{
		const RealType numerator = tx_gain * rx_gain * lambda * lambda;
		RealType denominator = 16.0 * PI * PI; // (4 * PI)^2

		if (!no_prop_loss)
		{
			denominator *= dist * dist;
		}

		return numerator / denominator;
	}

	/**
	 * @brief Computes the power scaling factor for a reflected path (Bistatic Radar Range Equation).
	 * @param tx_gain Transmitter gain (linear).
	 * @param rx_gain Receiver gain (linear).
	 * @param rcs Target Radar Cross Section (m^2).
	 * @param lambda Wavelength (meters).
	 * @param r_tx Distance from Transmitter to Target.
	 * @param r_rx Distance from Target to Receiver.
	 * @param no_prop_loss If true, distance-based attenuation is ignored.
	 * @return The power scaling factor (Pr / Pt).
	 */
	static RealType computeReflectedPathPower(RealType tx_gain, RealType rx_gain, RealType rcs, RealType lambda,
											  RealType r_tx, RealType r_rx, bool no_prop_loss)
	{
		const RealType numerator = tx_gain * rx_gain * rcs * lambda * lambda;
		RealType denominator = 64.0 * PI * PI * PI; // (4 * PI)^3

		if (!no_prop_loss)
		{
			denominator *= r_tx * r_tx * r_rx * r_rx;
		}

		return numerator / denominator;
	}


	/**
	 * @brief Determine the PropagationPath along a direct tx->rx path
	 * @param time The time at which to find propagation paths. May be at reception or transmission time.
	 * @param tx_time True if time is a transmission time, otherwise it's considered to be the receiving time.
	 */
	static void directPath(const Transmitter& tx, Receiver* rx, const RealType carrierWavelength, const RealType time,
						   const bool is_tx_time, size_t source_index, std::vector<PropagationPath>& found_paths)
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
		RealType txTime = is_tx_time ? time : time - delay;
		RealType rxTime = is_tx_time ? time + delay : time;

		// Tx Gain: Direction Tx -> Rx
		const auto tx_gain = computeAntennaGain(&tx, tx_to_rx, txTime, carrierWavelength);
		// Rx Gain: Direction Rx -> Tx
		const auto rx_gain = computeAntennaGain(rx, -tx_to_rx, rxTime, carrierWavelength);

		const bool no_loss = rx->checkFlag(radar::Receiver::RecvFlag::FLAG_NOPROPLOSS);
		const auto gain = computeDirectPathPower(tx_gain, rx_gain, carrierWavelength, tx_to_rx_dist, no_loss);

		found_paths.emplace_back(PropagationPath{
			.delay = delay,
			.gain = gain,
			// Direct paths are unique for a given receiver and transmitter.
			// We just need to try avoid colliding with a target SimID (bistatic path). 0 achieves this.
			.path_id = 0,
			.source_index = source_index,
			.receiver = rx,
		});
	}

	static void bistaticPath(const Transmitter& tx, Receiver* rx, const Target& target,
							 const RealType carrierWavelength, const RealType time, const bool is_tx_time,
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

		const auto delay = (tx_to_tgt_dist + tgt_to_rx_dist) / params::c();
		RealType tx_time = is_tx_time ? time : time - delay;
		RealType rx_time = is_tx_time ? time + delay : time;
		const auto target_delay = tx_to_tgt_dist / params::c();
		RealType tgt_time = tx_time + target_delay;

		// Calculate RCS
		// InAngle: Tx -> Tgt (link_tx_tgt.u_vec)
		// OutAngle: Rx -> Tgt (Opposite of Tgt->Rx, so -link_tgt_rx.u_vec)
		SVec3 in_angle(tx_to_tgt);
		SVec3 out_angle(-tgt_to_rx);
		const auto rcs = target.getRcs(in_angle, out_angle, tgt_time);

		// Tx Gain: Direction Tx -> Tgt
		const auto tx_gain = computeAntennaGain(&tx, tx_to_tgt, tx_time, carrierWavelength);
		// Rx Gain: Direction Rx -> Tgt
		const auto rx_gain = computeAntennaGain(rx, -tgt_to_rx, rx_time, carrierWavelength);

		const bool no_loss = rx->checkFlag(radar::Receiver::RecvFlag::FLAG_NOPROPLOSS);
		const auto gain = computeReflectedPathPower(tx_gain, rx_gain, rcs, carrierWavelength, tx_to_tgt_dist,
													tgt_to_rx_dist, no_loss);

		found_paths.emplace_back(PropagationPath{
			.delay = delay,
			.gain = gain,
			// Bistatic paths for a given TX and RX are uniquely identified by the target.
			.path_id = target.getId(),
			.source_index = source_index,
			.receiver = rx,
		});
	}

	std::vector<PathsAtTime> PointScatterModel::findTxToRxPaths(const Transmitter& transmitter,
																const RealType start_tx_time) const
	{
		std::vector<PathsAtTime> timesteps;

		const auto* signal = transmitter.getSignal()->getPulseWaveform();
		const auto end_tx_time = start_tx_time + signal->getDuration();

		for (const auto current_time : TimePointRange(start_tx_time, end_tx_time, params::simSamplingRate()))
		{
			std::vector<PropagationPath> paths;

			const auto carrierWavelength = params::c() / transmitter.getSignal()->getCarrier();

			for (const auto& receiver : _world->getReceivers())
			{
				// Note, we're passing null as the source as the caller of findTxToRxPaths
				// only gives us a Transmitter, not a ActiveStreamingSource.
				// findTxToRxPaths is used for pulsed radar instead of continuous, so this is fine.

				if (!receiver->checkFlag(Receiver::RecvFlag::FLAG_NODIRECT))
				{
					// Calculate the direct path contribution, if any.
					directPath(transmitter, receiver.get(), carrierWavelength, current_time, true, 0, paths);
				}

				for (const auto& target : _world->getTargets())
				{
					// Calculate bistatic reflection contribution, if any.
					bistaticPath(transmitter, receiver.get(), *target, carrierWavelength, current_time, true, 0, paths);
				}
			}

			timesteps.push_back({paths, current_time});
		}

		return timesteps;
	};

	std::vector<PropagationPath> PointScatterModel::findRxFromTxPaths(
		radar::Receiver* receiver, const std::vector<core::ActiveStreamingSource>& sources, RealType rx_time) const
	{
		std::vector<PropagationPath> paths;

		for (size_t s = 0; s < sources.size(); s++)
		{
			const auto& source = sources[s];
			const auto carrierWavelength = params::c() / source.transmitter->getSignal()->getCarrier();

			if (!receiver->checkFlag(Receiver::RecvFlag::FLAG_NODIRECT))
			{
				// Calculate the direct path contribution, if any.
				directPath(*source.transmitter, receiver, carrierWavelength, rx_time, false, s, paths);
			}

			for (const auto& target : _world->getTargets())
			{
				// Calculate bistatic reflection contribution, if any.
				bistaticPath(*source.transmitter, receiver, *target, carrierWavelength, rx_time, false, s, paths);
			}
		}

		return paths;
	}

	PointScatterModel::PointScatterModel(core::World* world) : _world(world) {}
}
