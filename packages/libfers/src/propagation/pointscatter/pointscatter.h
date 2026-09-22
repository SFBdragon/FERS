//
// TODO_SHAUN
//

#pragma once

#include <vector>

#include "core/config.h"
#include "core/simulation_state.h"
#include "core/world.h"
#include "propagation/propagation_model.h"
#include "radar/receiver.h"

namespace propagation::pointscatter
{
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

	private:
		core::World* _world;
	};
}
