#include "propagation/propagation_model.h"

#include <cmath>

namespace propagation
{
	PropagationModel::~PropagationModel() = default;


	std::generator<RealType> timePointGenerator(RealType start_time, RealType end_time, RealType sampling_rate)
	{
		const auto step = 1.0 / sampling_rate;
		const auto duration = end_time - start_time;
		const int point_count = static_cast<int>(std::ceil(duration / step));

		for (int i = 0; i <= point_count; ++i)
		{
			co_yield i < point_count ? start_time + i* step : end_time;
		}
	}
}
