// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2025, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include <iostream>
#include <string>

#include "ck_tile/core.hpp"
#include "ck_tile/ops/common.hpp"
#include "ck_tile/ops/flatmm/kernel/flatmm_kernel.hpp"

namespace ck_tile {

struct GroupedFlatmmHostArgs
{
    CK_TILE_HOST GroupedFlatmmHostArgs() = default;
    CK_TILE_HOST GroupedFlatmmHostArgs(index_t group_count_,
                                       index_t* M_,
                                       index_t* N_,
                                       index_t* K_,
                                       const void** a_ptr_,
                                       index_t* stride_A_,
                                       const void** b_shuffle_ptr_,
                                       index_t* stride_B_,
                                       void** c_ptr_,
                                       index_t* stride_C_,
                                       index_t k_batch_)
        : group_count(group_count_),
          M(M_),
          N(N_),
          K(K_),
          a_ptr(a_ptr_),
          stride_A(stride_A_),
          b_shuffle_ptr(b_shuffle_ptr_),
          stride_B(stride_B_),
          c_ptr(c_ptr_),
          stride_C(stride_C_),
          k_batch(k_batch_)
    {
    }

    index_t group_count;
    index_t* M;
    index_t* N;
    index_t* K;
    const void** a_ptr;
    index_t* stride_A;
    const void** b_shuffle_ptr;
    index_t* stride_B;
    void** c_ptr;
    index_t* stride_C;
    index_t k_batch;
};

template <typename TilePartitioner_, typename FlatmmPipeline_, typename EpiloguePipeline_>
struct GroupedFlatmmKernel : FlatmmKernel<TilePartitioner_, FlatmmPipeline_, EpiloguePipeline_>
{
    using UnderlyingGemmKernel = FlatmmKernel<TilePartitioner_, FlatmmPipeline_, EpiloguePipeline_>;

    using TilePartitioner = remove_cvref_t<TilePartitioner_>;
    using FlatmmPipeline  = remove_cvref_t<FlatmmPipeline_>;

    using EpiloguePipeline = remove_cvref_t<EpiloguePipeline_>;

    using ADataType = remove_cvref_t<typename FlatmmPipeline::ADataType>;
    using BDataType = remove_cvref_t<typename FlatmmPipeline::BDataType>;
    // Below type is actually accumulation data type - the output of block GEMM.
    using CDataType = remove_cvref_t<typename EpiloguePipeline::ODataType>;

    using GroupedFlatmmKernelArgs = GroupedFlatmmHostArgs;

    CK_TILE_HOST static const std::string GetName()
    {
        return concat(
            '_', "grouped_flatmm", gemm_prec_str<ADataType, BDataType>, FlatmmPipeline::GetName());
    }

    CK_TILE_HOST_DEVICE static constexpr auto GridSize(const GroupedFlatmmKernelArgs& kernelArgs)
    {
        hipDeviceProp_t prop;
        int deviceId = 0; // default device

        auto e = hipGetDeviceProperties(&prop, deviceId);

        const int persistent_block_size = prop.multiProcessorCount;
        return dim3(persistent_block_size, 1, kernelArgs.k_batch);
    }

    CK_TILE_HOST static constexpr GroupedFlatmmKernelArgs
    MakeKernelArgs(const GroupedFlatmmHostArgs& hostArgs)
    {
        return hostArgs;
    }

    CK_TILE_DEVICE void operator()(GroupedFlatmmKernelArgs kargs) const
    {
        int group_idx        = 0;
        int block_linear_idx = blockIdx.x;
        int total_block_cnt  = gridDim.x;

        UnderlyingGemmKernel underlying_kernel{};
        for(; group_idx < kargs.group_count; ++group_idx)
        {
            const index_t M               = kargs.M[group_idx];
            const index_t N               = kargs.N[group_idx];
            const index_t group_block_cnt = TilePartitioner::GridSize(M, N);

            while(block_linear_idx < group_block_cnt)
            {
                // Found the group this block belongs to
                // create the kernel args for the underlying flatmm kernel
                typename UnderlyingGemmKernel::FlatmmKernelArgs impl_kargs{
                    kargs.a_ptr[group_idx],
                    kargs.b_shuffle_ptr[group_idx],
                    kargs.c_ptr[group_idx],
                    kargs.M[group_idx],
                    kargs.N[group_idx],
                    kargs.K[group_idx],
                    kargs.stride_A[group_idx],
                    kargs.stride_B[group_idx],
                    kargs.stride_C[group_idx],
                    kargs.k_batch,
                };
                // call the underlying flatmm kernel
                underlying_kernel(impl_kargs, block_linear_idx);
                block_linear_idx += total_block_cnt;
            }
            block_linear_idx -= group_block_cnt;
        }
    }
};

} // namespace ck_tile
