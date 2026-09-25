/*
 * Copyright (C) 2023, Inria
 * GRAPHDECO research group, https://team.inria.fr/graphdeco
 * All rights reserved.
 */

#ifndef CUDA_RASTERIZER_FORWARD_H_INCLUDED
#define CUDA_RASTERIZER_FORWARD_H_INCLUDED

#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

namespace FORWARD
{
    void preprocess(
        const int total_P,
        const int P_per_batch,
        const int* points_xyz,
        const int* radii,
        const dim3 grid,
        uint32_t* tiles_touched);

    void render(
        const int total_N,
        const int N_per_batch,
        const float* pts,
        const int* points_int,
        const dim3 grid,
        const uint2* ranges,
        const uint32_t* point_list,
        const float* means3D,
        const float* cov3D,
        const float* opacity,
        const float* semantic,
        float* out);
}

#endif
