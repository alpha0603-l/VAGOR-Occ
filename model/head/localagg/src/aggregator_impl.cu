/*
 * Copyright (C) 2023, Inria
 * GRAPHDECO research group, https://team.inria.fr/graphdeco
 * All rights reserved.
 */

#include <torch/extension.h>
#include "aggregator_impl.h"
#include <iostream>
#include <algorithm>
#include <numeric>
#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cub/cub.cuh>
#include <cub/device/device_radix_sort.cuh>
#include <cooperative_groups.h>
namespace cg = cooperative_groups;

#include "auxiliary.h"
#include "forward.h"
#include "backward.h"


uint32_t getHigherMsb(uint32_t n)
{
    uint32_t msb = sizeof(n) * 4;
    uint32_t step = msb;
    while (step > 1)
    {
        step /= 2;
        if (n >> msb)
            msb += step;
        else
            msb -= step;
    }
    if (n >> msb)
        msb++;
    return msb;
}


__global__ void duplicateWithKeysBatch(
    const int total_P,
    const int P_per_batch,
    const int* points_xyz,
    const uint32_t* offsets,
    uint32_t* gaussian_keys_unsorted,
    uint32_t* gaussian_values_unsorted,
    const int* radii,
    const dim3 grid)
{
    auto idx = cg::this_grid().thread_rank();
    if (idx >= total_P)
        return;

    const int b = idx / P_per_batch;
    const uint32_t batch_offset = (uint32_t)b * grid.x * grid.y * grid.z;
    uint32_t off = (idx == 0) ? 0 : offsets[idx - 1];

    uint3 rect_min, rect_max;
    getRect(points_xyz + 3 * idx, radii[idx], rect_min, rect_max, grid);

    for (int x = rect_min.x; x < rect_max.x; x++)
    {
        for (int y = rect_min.y; y < rect_max.y; y++)
        {
            for (int z = rect_min.z; z < rect_max.z; z++)
            {
                uint32_t local_key = x * grid.y * grid.z + y * grid.z + z;
                gaussian_keys_unsorted[off] = batch_offset + local_key;
                gaussian_values_unsorted[off] = idx;
                off++;
            }
        }
    }
}


__global__ void identifyTileRanges(
    int L,
    uint32_t* point_list_keys,
    uint2* ranges)
{
    auto idx = cg::this_grid().thread_rank();
    if (idx >= L)
        return;

    uint32_t currtile = point_list_keys[idx];
    if (idx == 0)
        ranges[currtile].x = 0;
    else
    {
        uint32_t prevtile = point_list_keys[idx - 1];
        if (currtile != prevtile)
        {
            ranges[prevtile].y = idx;
            ranges[currtile].x = idx;
        }
    }
    if (idx == L - 1)
        ranges[currtile].y = L;
}


LocalAggregator::GeometryState LocalAggregator::GeometryState::fromChunk(char*& chunk, size_t P)
{
    GeometryState geom;
    obtain(chunk, geom.tiles_touched, P, 128);
    cub::DeviceScan::InclusiveSum(nullptr, geom.scan_size, geom.tiles_touched, geom.tiles_touched, P);
    obtain(chunk, geom.scanning_space, geom.scan_size, 128);
    obtain(chunk, geom.point_offsets, P, 128);
    return geom;
}


LocalAggregator::ImageState LocalAggregator::ImageState::fromChunk(char*& chunk, size_t N)
{
    ImageState img;
    obtain(chunk, img.ranges, N, 128);
    return img;
}


LocalAggregator::BinningState LocalAggregator::BinningState::fromChunk(char*& chunk, size_t P)
{
    BinningState binning;
    obtain(chunk, binning.point_list, P, 128);
    obtain(chunk, binning.point_list_unsorted, P, 128);
    obtain(chunk, binning.point_list_keys, P, 128);
    obtain(chunk, binning.point_list_keys_unsorted, P, 128);
    cub::DeviceRadixSort::SortPairs(
        nullptr, binning.sorting_size,
        binning.point_list_keys_unsorted, binning.point_list_keys,
        binning.point_list_unsorted, binning.point_list, P);
    obtain(chunk, binning.list_sorting_space, binning.sorting_size, 128);
    return binning;
}


int LocalAggregator::Aggregator::forward(
    std::function<char* (size_t)> geometryBuffer,
    std::function<char* (size_t)> binningBuffer,
    std::function<char* (size_t)> imageBuffer,
    const int B,
    const int P,
    const int N,
    const float* pts,
    const int* points_int,
    const float* means3D,
    const int* means3D_int,
    const float* opacities,
    const float* semantics,
    const float* cov3D,
    const int* radii,
    const int H,
    const int W,
    const int D,
    float* out,
    bool debug)
{
    const int total_P = B * P;
    const int total_N = B * N;
    const int total_voxels = B * H * W * D;

    size_t chunk_size = required<GeometryState>(total_P);
    char* chunkptr = geometryBuffer(chunk_size);
    GeometryState geomState = GeometryState::fromChunk(chunkptr, total_P);

    size_t img_chunk_size = required<ImageState>(total_voxels);
    char* img_chunkptr = imageBuffer(img_chunk_size);
    ImageState imgState = ImageState::fromChunk(img_chunkptr, total_voxels);

    dim3 grid(H, W, D);

    CHECK_CUDA(FORWARD::preprocess(
        total_P,
        P,
        means3D_int,
        radii,
        grid,
        geomState.tiles_touched), debug)

    CHECK_CUDA(cub::DeviceScan::InclusiveSum(
        geomState.scanning_space,
        geomState.scan_size,
        geomState.tiles_touched,
        geomState.point_offsets,
        total_P), debug);

    int num_rendered = 0;
    if (total_P > 0)
    {
        CHECK_CUDA(cudaMemcpy(&num_rendered, geomState.point_offsets + total_P - 1, sizeof(int), cudaMemcpyDeviceToHost), debug);
    }

    size_t binning_chunk_size = required<BinningState>(num_rendered);
    char* binning_chunkptr = binningBuffer(binning_chunk_size);
    BinningState binningState = BinningState::fromChunk(binning_chunkptr, num_rendered);

    duplicateWithKeysBatch << <(total_P + 255) / 256, 256 >> > (
        total_P,
        P,
        means3D_int,
        geomState.point_offsets,
        binningState.point_list_keys_unsorted,
        binningState.point_list_unsorted,
        radii,
        grid);
    CHECK_CUDA(, debug);

    CHECK_CUDA(cub::DeviceRadixSort::SortPairs(
        binningState.list_sorting_space,
        binningState.sorting_size,
        binningState.point_list_keys_unsorted, binningState.point_list_keys,
        binningState.point_list_unsorted, binningState.point_list,
        num_rendered, 0, 32), debug)

    CHECK_CUDA(cudaMemset(imgState.ranges, 0, total_voxels * sizeof(uint2)), debug);

    if (num_rendered > 0)
    {
        identifyTileRanges << <(num_rendered + 255) / 256, 256 >> > (
            num_rendered,
            binningState.point_list_keys,
            imgState.ranges);
    }
    CHECK_CUDA(, debug)

    CHECK_CUDA(FORWARD::render(
        total_N,
        N,
        pts,
        points_int,
        grid,
        imgState.ranges,
        binningState.point_list,
        means3D,
        cov3D,
        opacities,
        semantics,
        out), debug);

    return num_rendered;
}


void LocalAggregator::Aggregator::backward(
    const int B,
    const int P,
    const int R,
    const int N,
    const int H,
    const int W,
    const int D,
    char* geom_buffer,
    char* binning_buffer,
    char* img_buffer,
    const int* points_int,
    int* voxel2pts,
    const float* pts,
    const float* means3D,
    const float* cov3D,
    const float* opacities,
    const float* semantics,
    const float* out_grad,
    float* means3D_grad,
    float* opacity_grad,
    float* semantics_grad,
    float* cov3D_grad,
    bool debug)
{
    const int total_P = B * P;
    const int total_N = B * N;
    const int total_voxels = B * H * W * D;

    GeometryState geomState = GeometryState::fromChunk(geom_buffer, total_P);
    BinningState binningState = BinningState::fromChunk(binning_buffer, R);
    ImageState imgState = ImageState::fromChunk(img_buffer, total_voxels);

    const dim3 grid(H, W, D);

    CHECK_CUDA(BACKWARD::preprocess(
        total_N,
        N,
        points_int,
        grid,
        voxel2pts), debug)

    CHECK_CUDA(BACKWARD::render(
        total_P,
        geomState.point_offsets,
        binningState.point_list_keys_unsorted,
        voxel2pts,
        pts,
        means3D,
        cov3D,
        opacities,
        semantics,
        out_grad,
        means3D_grad,
        opacity_grad,
        semantics_grad,
        cov3D_grad), debug)
}
