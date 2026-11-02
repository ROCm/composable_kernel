// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

// Include these 2 headers instead of torch/extension.h since we don't need all of the torch
// headers.
#include <torch/python.h>
#include <torch/nn/functional.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <optional>
#include <string>

#ifdef USE_ROCM
#include <hip/hip_runtime.h>
#endif

#ifdef OLD_GENERATOR_PATH
#include <ATen/CUDAGeneratorImpl.h>
#else
#include <ATen/cuda/CUDAGeneratorImpl.h>
#endif

#define CK_TILE_FMHA_CHECK_DEVICE(x) TORCH_CHECK(x.is_cuda(), #x " must be on CUDA")
#define CK_TILE_FMHA_CHECK_SHAPE(x, ...)                        \
    TORCH_CHECK(x.sizes() == torch::IntArrayRef({__VA_ARGS__}), \
                #x " must have shape (" #__VA_ARGS__ ")")
#define CK_TILE_FMHA_CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")

namespace ck_tile::fmha::torch_api::detail {
// Elementwise bias is shared by all batches and heads.
inline void check_bias(const std::optional<at::Tensor>& bias,
                       const std::optional<at::Tensor>& alibi_slopes,
                       const at::Tensor& q,
                       int seqlen_q,
                       int seqlen_k)
{
    TORCH_CHECK(!(bias.has_value() && alibi_slopes.has_value()),
                "cannot apply bias and alibi at the same time");
    if(bias.has_value())
    {
        const auto& tensor = bias.value();
        TORCH_CHECK(tensor.device() == q.device(), "bias must be on the same device as query");
        TORCH_CHECK(tensor.dtype() == q.dtype(), "bias must have the same dtype as query");
        TORCH_CHECK(tensor.sizes() == torch::IntArrayRef({seqlen_q, seqlen_k}),
                    "bias shape must be [seqlen_q, seqlen_k]");
        TORCH_CHECK(tensor.stride(-1) == 1, "bias must have contiguous last dimension");
    }
}

inline __global__ void ParsePhiloxCudaState(at::PhiloxCudaState arg, uint64_t* rng_state)
{
    // Imitate from PyTorch
    // https://github.com/pytorch/pytorch/blob/8b61daaf7349e9102117e1aeefaa51666d887547/aten/src/ATen/cuda/detail/UnpackRaw.cuh#L17
    if(arg.captured_)
    {
        rng_state[0] = static_cast<uint64_t>(*arg.seed_.ptr);
        rng_state[1] = static_cast<uint64_t>(*(arg.offset_.ptr) + arg.offset_intragraph_);
    }
    else
    {
        rng_state[0] = arg.seed_.val;
        rng_state[1] = arg.offset_.val;
    }
}

inline int
num_splits_heuristic_ck(int batch_nheads_mblocks, int num_SMs, int num_n_blocks, int max_splits)
{
    // If we have enough to almost fill the SMs, then just use 1 split
    if(batch_nheads_mblocks >= 0.8f * num_SMs)
    {
        return 1;
    }
    max_splits           = std::min({max_splits, num_SMs, num_n_blocks});
    float max_efficiency = 0.f;
    std::vector<float> efficiency;
    efficiency.reserve(max_splits);
    auto ceildiv = [](int a, int b) { return (a + b - 1) / b; };
    // Some splits are not eligible. For example, if we have 64 blocks and choose 11 splits,
    // we'll have 6 * 10 + 4 blocks. If we choose 12 splits, we'll have 6 * 11 + (-2) blocks
    // (i.e. it's 11 splits anyway).
    // So we check if the number of blocks per split is the same as the previous num_splits.
    auto is_split_eligible = [&ceildiv, &num_n_blocks](int num_splits) {
        return num_splits == 1 ||
               ceildiv(num_n_blocks, num_splits) != ceildiv(num_n_blocks, num_splits - 1);
    };
    for(int num_splits = 1; num_splits <= max_splits; num_splits++)
    {
        if(!is_split_eligible(num_splits))
        {
            efficiency.push_back(0.f);
        }
        else
        {
            float n_waves = float(batch_nheads_mblocks * num_splits) / num_SMs;
            float eff     = n_waves / ceil(n_waves);
            // printf("num_splits = %d, eff = %f\n", num_splits, eff);
            if(eff > max_efficiency)
            {
                max_efficiency = eff;
            }
            efficiency.push_back(eff);
        }
    }
    for(int num_splits = 1; num_splits <= max_splits; num_splits++)
    {
        if(!is_split_eligible(num_splits))
        {
            continue;
        }
        if(efficiency[num_splits - 1] >= 0.85 * max_efficiency)
        {
            // printf("num_splits chosen = %d\n", num_splits);
            return num_splits;
        }
    }
    return 1;
}

int override_num_splits_if_necessary(
    int batch, int nhead, int max_seqlen_q, int hdim_v, float p_drop, int num_splits);

} // namespace ck_tile::fmha::torch_api::detail
