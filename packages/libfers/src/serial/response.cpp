// SPDX-License-Identifier: GPL-2.0-only
//
// Copyright (c) 2006-2008 Marc Brooker and Michael Inggs
// Copyright (c) 2008-present FERS Contributors (see AUTHORS.md).
//
// See the GNU GPLv2 LICENSE file in the FERS project root for more information.

/**
 * @file response.cpp
 * @brief Implementation of the Response class
 */

#include "response.h"

#include <cmath>
#include <cstddef>

#include "radar/transmitter.h"
#include "signal/radar_signal.h"

using interp::InterpPoint;

namespace serial
{
	void Response::addInterpPoint(const InterpPoint& point, const RealType right_interval)
	{
		if (point.rx_time < _points.back().rx_time)
		{
			throw std::logic_error("Attempted to add an InterpPoint earlier than the Response's last point");
		}
		_points.push_back(point);
		_right_interval = right_interval;
	}

	std::vector<ComplexType> Response::render(const RealType fracWinDelay) const
	{
		auto amplitude = std::sqrt(_signal->getPower());
		return _wave->render(taperedPoints(), _points.front().rx_time, fracWinDelay, amplitude);
	}

	std::vector<ComplexType> Response::renderSlice(const RealType outputRate, const RealType outputStartTime,
												   const std::size_t sampleCount, const RealType fracWinDelay) const
	{
		auto amplitude = std::sqrt(_signal->getPower());
		return _wave->renderSlice(taperedPoints(), _points.front().rx_time, outputStartTime, outputRate, sampleCount,
								  fracWinDelay, amplitude);
	}

	RealType Response::sampleRate() const noexcept { return _wave->getRate(); }

	size_t Response::sampleCount() const noexcept { return _wave->getSampleCount(); }
}
