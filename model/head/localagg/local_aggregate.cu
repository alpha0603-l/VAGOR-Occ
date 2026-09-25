/*
 * Copyright (C) 2023, Inria
 * GRAPHDECO research group, https://team.inria.fr/graphdeco
 * All rights reserved.
 */

#include <math.h>
#include <torch/extension.h>
#include <cstdio>
#include <tuple>
#include <cuda_runtime_api.h>
#include <functional>
#include "src/config.h"
#include "src/aggregator.h"

std::function<char*(size_t N)> resizeFunctional(torch::Tensor& t) {
    auto lambda = [&t](size_t N) {
        t.resize_({(long long)N});
        return reinterpret_cast<char*>(t.contiguous().data_ptr());
    };
    return lambda;
}

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
    const int H, int W, int D)
{
    TORCH_CHECK(pts.is_cuda(), "pts must be CUDA");
    TORCH_CHECK(points_int.is_cuda(), "points_int must be CUDA");
    TORCH_CHECK(means3D.is_cuda(), "means3D must be CUDA");
    TORCH_CHECK(means3D_int.is_cuda(), "means3D_int must be CUDA");
    TORCH_CHECK(opacity.is_cuda(), "opacity must be CUDA");
    TORCH_CHECK(semantics.is_cuda(), "semantics must be CUDA");
    TORCH_CHECK(radii.is_cuda(), "radii must be CUDA");
    TORCH_CHECK(cov3D.is_cuda(), "cov3D must be CUDA");
    TORCH_CHECK(pts.dim() == 3 && pts.size(2) == 3, "pts must be [B,N,3]");
    TORCH_CHECK(points_int.dim() == 3 && points_int.size(2) == 3, "points_int must be [B,N,3]");
    TORCH_CHECK(means3D.dim() == 3 && means3D.size(2) == 3, "means3D must be [B,G,3]");
    TORCH_CHECK(means3D_int.dim() == 3 && means3D_int.size(2) == 3, "means3D_int must be [B,G,3]");
    TORCH_CHECK(opacity.dim() == 2, "opacity must be [B,G]");
    TORCH_CHECK(semantics.dim() == 3 && semantics.size(2) == NUM_CHANNELS, "semantics must be [B,G,NUM_CHANNELS]");
    TORCH_CHECK(radii.dim() == 2, "radii must be [B,G]");
    TORCH_CHECK(cov3D.dim() == 3 && cov3D.size(2) == 6, "cov3D must be [B,G,6]");

    const int B = pts.size(0);
    const int N = pts.size(1);
    const int P = means3D.size(1);
    TORCH_CHECK(points_int.size(0) == B && points_int.size(1) == N, "points_int shape mismatch");
    TORCH_CHECK(means3D.size(0) == B, "means3D batch mismatch");
    TORCH_CHECK(means3D_int.size(0) == B && means3D_int.size(1) == P, "means3D_int shape mismatch");
    TORCH_CHECK(opacity.size(0) == B && opacity.size(1) == P, "opacity shape mismatch");
    TORCH_CHECK(semantics.size(0) == B && semantics.size(1) == P, "semantics shape mismatch");
    TORCH_CHECK(radii.size(0) == B && radii.size(1) == P, "radii shape mismatch");
    TORCH_CHECK(cov3D.size(0) == B && cov3D.size(1) == P, "cov3D shape mismatch");

    auto float_opts = means3D.options().dtype(torch::kFloat32);
    torch::Tensor out_logits = torch::full({B, N, NUM_CHANNELS}, 0.0, float_opts);

    torch::Device device(torch::kCUDA, means3D.get_device());
    torch::TensorOptions options(torch::kByte);
    torch::Tensor geomBuffer = torch::empty({0}, options.device(device));
    torch::Tensor binningBuffer = torch::empty({0}, options.device(device));
    torch::Tensor imgBuffer = torch::empty({0}, options.device(device));
    std::function<char*(size_t)> geomFunc = resizeFunctional(geomBuffer);
    std::function<char*(size_t)> binningFunc = resizeFunctional(binningBuffer);
    std::function<char*(size_t)> imgFunc = resizeFunctional(imgBuffer);

    int rendered = LocalAggregator::Aggregator::forward(
        geomFunc,
        binningFunc,
        imgFunc,
        B, P, N,
        pts.contiguous().data_ptr<float>(),
        points_int.contiguous().data_ptr<int>(),
        means3D.contiguous().data_ptr<float>(),
        means3D_int.contiguous().data_ptr<int>(),
        opacity.contiguous().data_ptr<float>(),
        semantics.contiguous().data_ptr<float>(),
        cov3D.contiguous().data_ptr<float>(),
        radii.contiguous().data_ptr<int>(),
        H, W, D,
        out_logits.contiguous().data_ptr<float>());

    return std::make_tuple(rendered, out_logits, geomBuffer, binningBuffer, imgBuffer);
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
LocalAggregateBackwardCUDA(
    const torch::Tensor& geomBuffer,
    const torch::Tensor& binningBuffer,
    const torch::Tensor& imageBuffer,
    const int H, int W, int D,
    const int R,
    const torch::Tensor& means3D,
    const torch::Tensor& pts,
    const torch::Tensor& points_int,
    const torch::Tensor& cov3D,
    const torch::Tensor& opacities,
    const torch::Tensor& semantics,
    const torch::Tensor& out_grad)
{
    TORCH_CHECK(means3D.is_cuda(), "means3D must be CUDA");
    TORCH_CHECK(pts.is_cuda(), "pts must be CUDA");
    TORCH_CHECK(points_int.is_cuda(), "points_int must be CUDA");
    TORCH_CHECK(cov3D.is_cuda(), "cov3D must be CUDA");
    TORCH_CHECK(opacities.is_cuda(), "opacities must be CUDA");
    TORCH_CHECK(semantics.is_cuda(), "semantics must be CUDA");
    TORCH_CHECK(out_grad.is_cuda(), "out_grad must be CUDA");

    const int B = means3D.size(0);
    const int P = means3D.size(1);
    const int N = pts.size(1);

    torch::Tensor means3D_grad = torch::zeros({B, P, 3}, means3D.options());
    torch::Tensor opacity_grad = torch::zeros({B, P}, means3D.options());
    torch::Tensor semantics_grad = torch::zeros({B, P, NUM_CHANNELS}, means3D.options());
    torch::Tensor cov3D_grad = torch::zeros({B, P, 6}, means3D.options());
    torch::Tensor voxel2pts = torch::full({B * H * W * D}, -1, means3D.options().dtype(torch::kInt32));

    LocalAggregator::Aggregator::backward(
        B, P, R, N,
        H, W, D,
        reinterpret_cast<char*>(geomBuffer.contiguous().data_ptr()),
        reinterpret_cast<char*>(binningBuffer.contiguous().data_ptr()),
        reinterpret_cast<char*>(imageBuffer.contiguous().data_ptr()),
        points_int.contiguous().data_ptr<int>(),
        voxel2pts.contiguous().data_ptr<int>(),
        pts.contiguous().data_ptr<float>(),
        means3D.contiguous().data_ptr<float>(),
        cov3D.contiguous().data_ptr<float>(),
        opacities.contiguous().data_ptr<float>(),
        semantics.contiguous().data_ptr<float>(),
        out_grad.contiguous().data_ptr<float>(),
        means3D_grad.contiguous().data_ptr<float>(),
        opacity_grad.contiguous().data_ptr<float>(),
        semantics_grad.contiguous().data_ptr<float>(),
        cov3D_grad.contiguous().data_ptr<float>());

    return std::make_tuple(means3D_grad, opacity_grad, semantics_grad, cov3D_grad);
}
