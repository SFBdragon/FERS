// SPDX-License-Identifier: GPL-2.0-only
//
// Copyright (c) 2026-present FERS Contributors (see AUTHORS.md).
//
// See the GNU GPLv2 LICENSE file in the FERS project root for more information.


#pragma once

#include "core/simulation_state.h"
#include "propagation/common_defs.h"

namespace propagation
{
	template <typename Real>
	[[nodiscard]] CarrierModel<Real> buildPropagationCarrier(const core::ActiveStreamingSource& source)
	{
		CarrierModel<Real> model;

		switch (source.kind)
		{
		case core::StreamingWaveformKind::Cw:
		case core::StreamingWaveformKind::FileCw:
		case core::StreamingWaveformKind::FileFmcw:
			{
				model.kind = CarrierModelKind::Constant;
				model.params.constant.frequency = Real(source.carrier_freq);
				break;
			}
		case core::StreamingWaveformKind::FmcwLinear:
			{
				model.kind = CarrierModelKind::Chirps;
				model.params.chirps.start_time = source.segment_start;
				model.params.chirps.chirp_duration = Real(source.chirp_period);
				model.params.chirps.start_frequency = Real(source.carrier_freq + source.start_freq_off);
				model.params.chirps.slope_frequency_per_time = Real(source.chirp_rate);
				break;
			}
		case core::StreamingWaveformKind::FmcwTriangle:
			{
				model.kind = CarrierModelKind::Triangles;
				model.params.triangles.start_time = source.segment_start;
				model.params.triangles.chirp_duration = Real(source.triangle_period) * Real(0.5);
				model.params.triangles.start_frequency = Real(source.carrier_freq + source.start_freq_off);
				model.params.triangles.slope_frequency_per_time = Real(source.chirp_rate);
				break;
			}
		case core::StreamingWaveformKind::Sfcw:
			{
				model.kind = CarrierModelKind::Stairs;
				model.params.stairs.start_time = source.segment_start;
				model.params.stairs.step_duration = Real(source.sfcw_step_period);
				model.params.stairs.step_count = Real(source.sfcw_step_count);
				model.params.stairs.start_frequency = Real(source.carrier_freq + source.start_freq_off);
				model.params.stairs.step_frequency = Real(source.sfcw_step_size);
				break;
			}
		}

		return model;
	}
}
