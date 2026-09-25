/*
 * Copyright (C) 2023, Inria
 * GRAPHDECO research group, https://team.inria.fr/graphdeco
 * All rights reserved.
 */

#pragma once
#include <torch/extension.h>
#include <tuple>

std::tuple<int, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
LocalAggregateCUDA(
    const torch::Tensor& pts,            // B, N, 3
    const torch::Tensor& points_int,     // B, N, 3
    const torch::Tensor& means3D,        // B, G, 3
    const torch::Tensor& means3D_int,    // B, G, 3
    const torch::Tensor& opacity,        // B, G
    const torch::Tensor& semantics,      // B, G, C
    const torch::Tensor& radii,          // B, G
    const torch::Tensor& cov3D,          // B, G, 6
    const int H, int W, int D);

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
LocalAggregateBackwardCUDA(
    const torch::Tensor& geomBuffer,
    const torch::Tensor& binningBuffer,
    const torch::Tensor& imageBuffer,
    const int H, int W, int D,
    const int R,
    const torch::Tensor& means3D,        // B, G, 3
    const torch::Tensor& pts,            // B, N, 3
    const torch::Tensor& points_int,     // B, N, 3
    const torch::Tensor& cov3D,          // B, G, 6
    const torch::Tensor& opacities,      // B, G
    const torch::Tensor& semantics,      // B, G, C
    const torch::Tensor& out_grad);      // B, N, C
