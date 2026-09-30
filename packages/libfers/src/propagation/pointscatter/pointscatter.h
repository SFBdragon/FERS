// SPDX-License-Identifier: GPL-2.0-only
//
// Copyright (c) 2026-present FERS Contributors (see AUTHORS.md).
//
// See the GNU GPLv2 LICENSE file in the FERS project root for more information.

#pragma once

#include <vector>

#include "core/config.h"
#include "core/simulation_state.h"
#include "core/world.h"
#include "math/geometry_ops.h"
#include "propagation/math.h"
#include "propagation/propagation_model.h"
#include "radar/receiver.h"

namespace radar
{
	class Radar;
}

namespace propagation::pointscatter
{
	/**
	 * @brief Computes an antenna's polarisation vector, in world space, for a given direction and time.
	 * @param radar The radar object (Transmitter or Receiver).
	 * @param direction The vector pointing away from the antenna towards the target/receiver (need not be unit).
	 * @param time The simulation time for rotation lookup.
	 * @return The antenna's polarisation vector, in world space.
	 */
	[[nodiscard]] Complex3<RealType> computeAntennaPolarisation(const radar::Radar* radar, const math::Vec3& direction,
																RealType time);

	/**
	 * @brief Computes the voltage scaling factor for a reflected path (Bistatic Radar Range Equation).
	 * @param tx_gain Transmitter gain (linear).
	 * @param rx_gain Receiver gain (linear).
	 * @param tx_pol Transmitter antenna polarisation vector, in world space, at the Tx->Tgt direction.
	 * @param rx_pol Receiver antenna polarisation vector, in world space, at the Rx->Tgt direction.
	 * @param tx_to_tgt Vector from the transmitter to the target.
	 * @param tgt_to_rx Vector from the target to the receiver.
	 * @param r_tx Distance from Transmitter to Target.
	 * @param r_rx Distance from Target to Receiver.
	 * @param rcs Target Radar Cross Section (m^2).
	 * @param lambda Wavelength (meters).
	 * @param no_prop_loss If true, distance-based attenuation is ignored.
	 * @return The amplitude/voltage scaling factor (Vr / Vt). Equivalent to sqrt(Pr / Pt).
	 * @pre `tx_to_tgt` and `tgt_to_rx` must not point in exactly the same direction (forward scatter).
	 */
	[[nodiscard]] ComplexType computeReflectedPathGain(RealType tx_gain, RealType rx_gain,
													   const Complex3<RealType>& tx_pol,
													   const Complex3<RealType>& rx_pol, const math::Vec3& tx_to_tgt,
													   const math::Vec3& tgt_to_rx, RealType r_tx, RealType r_rx,
													   RealType rcs, RealType lambda, bool no_prop_loss);

	class PointScatterModel : public PropagationModel
	{
	public:
		PointScatterModel(core::World* world);

		[[nodiscard]] std::vector<PathsAtTime> findTxToRxPaths(ThreadContext* ctx,
															   const radar::Transmitter& transmitter,
															   const std::vector<RealType>& times) const override;

		[[nodiscard]] std::vector<PropagationPath>
		findRxFromTxPaths(ThreadContext* ctx, radar::Receiver* receiver,
						  const std::vector<core::ActiveStreamingSource>& sources, RealType rx_time) const override;

		[[nodiscard]] unsigned int maxPropagationLegs() const noexcept override
		{
			// directPath (1 leg) and bistaticPath (2 legs)
			return 2;
		}

	private:
		core::World* _world;
	};
}
