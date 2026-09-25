/*
 * Copyright (C) 2023, Inria
 * GRAPHDECO research group, https://team.inria.fr/graphdeco
 * All rights reserved.
 */

#ifndef CUDA_RASTERIZER_BACKWARD_H_INCLUDED
#define CUDA_RASTERIZER_BACKWARD_H_INCLUDED

#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

namespace BACKWARD
{
    void render(
        const int total_P,
        const uint32_t* offsets,
        const uint32_t* point_list_keys_unsorted,
        const int* voxel2pts,
        const float* pts,
        const float* means3D,
        const float* cov3D,
        const float* opacity,
        const float* semantic,
        const float* out_grad,
        float* means3D_grad,
        float* opacity_grad,
        float* semantics_grad,
        float* cov3D_grad);

    void preprocess(
        const int total_N,
        const int N_per_batch,
        const int* points_xyz,
        const dim3 grid,
        int* voxel2pts);
}

#endif
