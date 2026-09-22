//
// TODO_SHAUN
//

#pragma once

#include <string>
#include <vector>

#include "propagation/math.h"
#include "propagation/raytracing/sbr_shared.h"

namespace propagation::raytracing
{
	TriangleMeshView loadMesh(const std::string& file_path, std::vector<Float3>& vertices, std::vector<Uint3>& indices);
}
