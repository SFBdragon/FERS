// SPDX-License-Identifier: GPL-2.0-only
//
// Copyright (c) 2026-present FERS Contributors (see AUTHORS.md).
//
// See the GNU GPLv2 LICENSE file in the FERS project root for more information.

#pragma once

#include <vector>

#include "core/simulation_state.h"
#include "core/world.h"
#include "propagation/propagation_model.h"
#include "radar/receiver.h"

namespace propagation::pointscatter
{
	/**
	 * @brief Computes the power scaling factor for a direct path (Friis Transmission Equation).
	 * @param tx_gain Transmitter gain (linear).
	 * @param rx_gain Receiver gain (linear).
	 * @param lambda Wavelength (meters).
	 * @param dist Distance (meters).
	 * @param no_prop_loss If true, distance-based attenuation is ignored.
	 * @return The power scaling factor (Pr / Pt).
	 */
	RealType computeDirectPathPower(RealType tx_gain, RealType rx_gain, RealType lambda, RealType dist,
									bool no_prop_loss);

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
	RealType computeReflectedPathPower(RealType tx_gain, RealType rx_gain, RealType rcs, RealType lambda, RealType r_tx,
									   RealType r_rx, bool no_prop_loss);

	class PointScatterModel : public PropagationModel
	{
	public:
		PointScatterModel(core::World* world);

		[[nodiscard]] std::vector<PathsAtTime> findTxToRxPaths(const radar::Transmitter& transmitter,
															   RealType start_tx_time) const override;

		[[nodiscard]] std::vector<PropagationPath>
		findRxFromTxPaths(radar::Receiver* receiver, const std::vector<core::ActiveStreamingSource>& sources,
						  RealType rx_time) const override;

	private:
		core::World* _world;
	};
}
