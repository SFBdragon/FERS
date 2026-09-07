//
// TODO_SHAUN
//

#pragma once

#include <mutex>
#include <vector>

#include "core/simulation_state.h"
#include "core/world.h"
#include "propagation/propagation_model.h"
#include "radar/receiver.h"

namespace propagation::raytracing
{
	class RayTracingModel : public PropagationModel
	{
	public:
		RayTracingModel(core::World* world);

		[[nodiscard]] std::vector<PathsAtTime> findTxToRxPaths(const radar::Transmitter& transmitter,
															   RealType start_tx_time) const override;

		[[nodiscard]] std::vector<PropagationPath>
		findRxFromTxPaths(radar::Receiver* receiver, const std::vector<core::ActiveStreamingSource>& sources,
						  RealType rx_time) const override;

	private:
		mutable std::mutex _mutex;
		// mutable /* GPU context, BVH handles, scratch buffers, etc. */ _state;

		// core::World* _world; ???
	};
}
