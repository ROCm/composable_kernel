// SPDX-License-Identifier: MIT
// Copyright (c) 2024, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck_tile/ops/gemm/kernel/gemm_kernel.hpp"

namespace ck_tile {

struct BatchedGemmHostArgs
{
    const void* a_ptr;
    const void* b_ptr;
    void* c_ptr;
    index_t M;
    index_t N;
    index_t K;
    index_t stride_A;
    index_t stride_B;
    index_t stride_C;
    index_t batch_stride_A;
    index_t batch_stride_B;
    index_t batch_stride_C;
    index_t batch_count;
};

template <typename TilePartitioner_, typename GemmPipeline_, typename EpiloguePipeline_>
struct BatchedGemmKernel : public GemmKernel<TilePartitioner_, GemmPipeline_, EpiloguePipeline_>
{
    using Base = GemmKernel<TilePartitioner_, GemmPipeline_, EpiloguePipeline_>;

    using GemmCommonKargs = typename Base::GemmCommonKargs;

    using ADataType = typename Base::ADataType;
    using BDataType = typename Base::BDataType;
    using CDataType = typename Base::CDataType;

    using TilePartitioner  = typename Base::TilePartitioner;
    using GemmPipeline     = typename Base::GemmPipeline;
    using EpiloguePipeline = typename Base::EpiloguePipeline;
    using ALayout          = typename Base::ALayout;
    using BLayout          = typename Base::BLayout;
    using CLayout          = typename Base::CLayout;

    struct BatchedGemmKargs : GemmCommonKargs
    {
        index_t batch_stride_A;
        index_t batch_stride_B;
        index_t batch_stride_C;
        index_t batch_count;
    };

    using Kargs = BatchedGemmKargs;
    using Hargs = BatchedGemmHostArgs;

    __host__ static constexpr auto GridSize(const Hargs& k)
    {
        return TilePartitioner::GridSize(k.M, k.N, k.batch_count);
    }

    __host__ static constexpr auto BlockSize() { return dim3(Base::KernelBlockSize); }

    CK_TILE_HOST static constexpr BatchedGemmKargs MakeKargs(const Hargs& h)
    {
        Kargs k;
        k.a_ptr          = h.a_ptr;
        k.b_ptr          = h.b_ptr;
        k.c_ptr          = h.c_ptr;
        k.M              = h.M;
        k.N              = h.N;
        k.K              = h.K;
        k.stride_A       = h.stride_A;
        k.stride_B       = h.stride_B;
        k.stride_C       = h.stride_C;
        k.batch_stride_A = h.batch_stride_A;
        k.batch_stride_B = h.batch_stride_B;
        k.batch_stride_C = h.batch_stride_C;
        k.batch_count    = h.batch_count;
        return k;
    }

    CK_TILE_HOST_DEVICE static constexpr index_t GetSmemSize()
    {
        return max(GemmPipeline::GetSmemSize(), EpiloguePipeline::GetSmemSize());
    }

    CK_TILE_DEVICE void operator()(Kargs kargs) const
    {
        const auto [i_m, i_n] = TilePartitioner{}();
        const auto i_batch    = __builtin_amdgcn_readfirstlane(blockIdx.z);

        //  options
        const auto batch_stride_A = __builtin_amdgcn_readfirstlane(kargs.batch_stride_A);
        const auto batch_offset_A = __builtin_amdgcn_readfirstlane(i_batch * batch_stride_A);
        const ADataType* a_start  = static_cast<const ADataType*>(kargs.a_ptr) + batch_offset_A;

        const auto batch_stride_B = __builtin_amdgcn_readfirstlane(kargs.batch_stride_B);
        const auto batch_offset_B = __builtin_amdgcn_readfirstlane(i_batch * batch_stride_B);
        const BDataType* b_start  = static_cast<const BDataType*>(kargs.b_ptr) + batch_offset_B;

        const auto batch_stride_C = __builtin_amdgcn_readfirstlane(kargs.batch_stride_C);
        const auto batch_offset_C = __builtin_amdgcn_readfirstlane(i_batch * batch_stride_C);
        CDataType* c_start        = static_cast<CDataType*>(kargs.c_ptr) + batch_offset_C;

        this->run_common_gemm_pipeline(a_start, b_start, c_start, kargs, i_m, i_n);
    }
};

} // namespace ck_tile
