//
// TODO
//

#include "raytracing.h"

#include <vector>

#include "core/config.h"
#include "core/parameters.h"
#include "core/simulation_state.h"
#include "core/world.h"
#include "propagation/propagation_model.h"
#include "radar/receiver.h"
#include "radar/transmitter.h"

using namespace math;
using namespace radar;

namespace propagation::raytracing
{

	std::vector<PathsAtTime> RayTracingModel::findTxToRxPaths(const Transmitter& transmitter,
															  const RealType start_tx_time) const
	{
		std::vector<PathsAtTime> timesteps;

		const auto* signal = transmitter.getSignal()->getPulseWaveform();
		const auto end_tx_time = start_tx_time + signal->getDuration();

		for (const auto current_time : timePointGenerator(start_tx_time, end_tx_time, params::simSamplingRate()))
		{
		}

		std::vector<PropagationPath> paths;

		std::scoped_lock lock(_mutex);
		// TODO_SHAUN

		return timesteps;
	};

	std::vector<PropagationPath>
	RayTracingModel::findRxFromTxPaths(radar::Receiver* receiver,
									   const std::vector<core::ActiveStreamingSource>& sources, RealType rx_time) const
	{
		std::vector<PropagationPath> paths;

		std::scoped_lock lock(_mutex);
		// TODO_SHAUN

		return paths;
	}

	RayTracingModel::RayTracingModel(core::World* world) {}
}
