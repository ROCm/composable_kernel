// SPDX-License-Identifier: MIT
// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/ops/batched_contraction/pipeline/batched_contraction_problem.hpp"
#include "ck_tile/ops/gemm/kernel/universal_gemm_kernel.hpp"

namespace ck_tile {

template <ck_tile::index_t NumDTensor = 0>
struct BatchedContractionHostArgs
{
    CK_TILE_HOST
    BatchedContractionHostArgs(const void* a_ptr_,
                               const void* b_ptr_,
                               const std::array<const void*, NumDTensor>& ds_ptr_,
                               void* e_ptr_,
                               ck_tile::index_t k_batch_,
                               ck_tile::index_t M_,
                               ck_tile::index_t N_,
                               ck_tile::index_t K_,
                               ck_tile::index_t G_,
                               ck_tile::index_t stride_A_,
                               ck_tile::index_t stride_B_,
                               const std::array<ck_tile::index_t, NumDTensor>& stride_Ds_,
                               ck_tile::index_t stride_E_,
                               ck_tile::index_t batch_stride_A_,
                               ck_tile::index_t batch_stride_B_,
                               const std::array<ck_tile::index_t, NumDTensor>& batch_stride_Ds_,
                               ck_tile::index_t batch_stride_E_)
        : a_ptr(a_ptr_),
          b_ptr(b_ptr_),
          ds_ptr(ds_ptr_),
          e_ptr(e_ptr_),
          k_batch(k_batch_),
          M(M_),
          N(N_),
          K(K_),
          G(G_),
          stride_A(stride_A_),
          stride_B(stride_B_),
          stride_Ds(stride_Ds_),
          stride_E(stride_E_),
          batch_stride_A(batch_stride_A_),
          batch_stride_B(batch_stride_B_),
          batch_stride_Ds(batch_stride_Ds_),
          batch_stride_E(batch_stride_E_)
    {
    }

    const void* a_ptr;
    const void* b_ptr;
    std::array<const void*, NumDTensor> ds_ptr;
    void* e_ptr;
    ck_tile::index_t k_batch;
    ck_tile::index_t M;
    ck_tile::index_t N;
    ck_tile::index_t K;
    ck_tile::index_t G;
    ck_tile::index_t stride_A;
    ck_tile::index_t stride_B;
    std::array<ck_tile::index_t, NumDTensor> stride_Ds;
    ck_tile::index_t stride_E;
    ck_tile::index_t batch_stride_A;
    ck_tile::index_t batch_stride_B;
    std::array<ck_tile::index_t, NumDTensor> batch_stride_Ds;
    ck_tile::index_t batch_stride_E;
};

template <ck_tile::index_t NumDTensor = 0>
struct BatchedContractionKernelArgs
{
    const void* a_ptr;
    const void* b_ptr;
    std::array<const void*, NumDTensor> ds_ptr;
    void* e_ptr;
    ck_tile::index_t k_batch;
    ck_tile::index_t M;
    ck_tile::index_t N;
    ck_tile::index_t K;
    ck_tile::index_t G;
    ck_tile::index_t stride_A;
    ck_tile::index_t stride_B;
    std::array<ck_tile::index_t, NumDTensor> stride_Ds;
    ck_tile::index_t stride_E;

    ck_tile::index_t batch_stride_A;
    ck_tile::index_t batch_stride_B;
    std::array<ck_tile::index_t, NumDTensor> batch_stride_Ds;
    ck_tile::index_t batch_stride_E;
};

template <typename Problem_,
          typename TilePartitioner_,
          typename GemmPipeline_,
          typename EpiloguePipeline_>
struct BatchedContractionKernel
{
    using Problem   = ck_tile::remove_cvref_t<Problem_>;
    using ADataType = ck_tile::remove_cvref_t<typename Problem::ADataType>;
    using BDataType = ck_tile::remove_cvref_t<typename Problem::BDataType>;
    using EDataType = ck_tile::remove_cvref_t<typename Problem::EDataType>;

    static constexpr ck_tile::index_t NumDimG    = Problem::NumDimG;
    static constexpr ck_tile::index_t NumDimM    = Problem::NumDimM;
    static constexpr ck_tile::index_t NumDimN    = Problem::NumDimN;
    static constexpr ck_tile::index_t NumDimK    = Problem::NumDimK;
    static constexpr ck_tile::index_t NumDTensor = Problem::NumDTensor;

    using TilePartitioner  = ck_tile::remove_cvref_t<TilePartitioner_>;
    using GemmPipeline     = ck_tile::remove_cvref_t<GemmPipeline_>;
    using EpiloguePipeline = ck_tile::remove_cvref_t<EpiloguePipeline_>;

    using UniversalGemmKernel =
        ck_tile::UniversalGemmKernel<TilePartitioner_, GemmPipeline_, EpiloguePipeline_>;
    static constexpr ck_tile::index_t kBlockSize = UniversalGemmKernel::kBlockSize;

    using KernelArgs = BatchedContractionKernelArgs<NumDTensor>;

    CK_TILE_HOST static constexpr auto GetKernelName() { return "batched_contraction_kernel"; }

    CK_TILE_HOST static constexpr bool IsSupportedArguments(const KernelArgs& kargs)
    {
        typename UniversalGemmKernel::KernelArgs gemm_kargs{{kargs.a_ptr},
                                                            {kargs.b_ptr},
                                                            kargs.ds_ptr,
                                                            kargs.e_ptr,
                                                            kargs.M,
                                                            kargs.N,
                                                            kargs.K,
                                                            {kargs.stride_A},
                                                            {kargs.stride_B},
                                                            kargs.stride_Ds,
                                                            kargs.stride_E,
                                                            kargs.k_batch};

        return UniversalGemmKernel::IsSupportedArguments(gemm_kargs) && kargs.G > 0;
    }

    CK_TILE_HOST static constexpr ck_tile::index_t GetSmemSize()
    {
        return UniversalGemmKernel::GetSmemSize();
    }

    CK_TILE_HOST static constexpr auto GetBlockSize()
    {
        return UniversalGemmKernel::GetBlockSize();
    }

    CK_TILE_HOST static constexpr auto
    GridSize(ck_tile::index_t M, ck_tile::index_t N, ck_tile::index_t KBatch, ck_tile::index_t G)
    {
        return dim3(TilePartitioner::GridSize(M, N), G, KBatch);
    }

    CK_TILE_HOST static constexpr KernelArgs
    MakeKernelArgs(const BatchedContractionHostArgs<NumDTensor>& host_args)
    {
        return KernelArgs{host_args.a_ptr,
                          host_args.b_ptr,
                          host_args.ds_ptr,
                          host_args.e_ptr,
                          host_args.k_batch,
                          host_args.M,
                          host_args.N,
                          host_args.K,
                          host_args.G,
                          host_args.stride_A,
                          host_args.stride_B,
                          host_args.stride_Ds,
                          host_args.stride_E,
                          host_args.batch_stride_A,
                          host_args.batch_stride_B,
                          host_args.batch_stride_Ds,
                          host_args.batch_stride_E};
    }

    CK_TILE_DEVICE void operator()(const KernelArgs& kargs) const
    {
        const auto tile_coord      = TilePartitioner::GetTileIndex(blockIdx.x, kargs.M, kargs.N);
        const ck_tile::index_t i_m = tile_coord.m_tile_idx * TilePartitioner::kMPerBlock;
        const ck_tile::index_t i_n = tile_coord.n_tile_idx * TilePartitioner::kNPerBlock;

        const auto i_batch  = __builtin_amdgcn_readfirstlane(blockIdx.y);
        const auto i_splitk = __builtin_amdgcn_readfirstlane(blockIdx.z);

        const auto batch_stride_A = __builtin_amdgcn_readfirstlane(kargs.batch_stride_A);
        const auto batch_offset_A = __builtin_amdgcn_readfirstlane(i_batch * batch_stride_A);
        const ADataType* a_ptr    = static_cast<const ADataType*>(kargs.a_ptr) + batch_offset_A;

        const auto batch_stride_B = __builtin_amdgcn_readfirstlane(kargs.batch_stride_B);
        const auto batch_offset_B = __builtin_amdgcn_readfirstlane(i_batch * batch_stride_B);
        const BDataType* b_ptr    = static_cast<const BDataType*>(kargs.b_ptr) + batch_offset_B;

        const auto batch_stride_E = __builtin_amdgcn_readfirstlane(kargs.batch_stride_E);
        const auto batch_offset_E = __builtin_amdgcn_readfirstlane(i_batch * batch_stride_E);
        EDataType* e_ptr          = static_cast<EDataType*>(kargs.e_ptr) + batch_offset_E;

        std::array<const void*, NumDTensor> ds_batch_ptr;
        static_for<0, NumDTensor, 1>{}([&](auto i) {
            const auto ds_batch_offset =
                __builtin_amdgcn_readfirstlane(i_batch * kargs.batch_stride_Ds[i]);
            ds_batch_ptr[i] = static_cast<const void*>(static_cast<const char*>(kargs.ds_ptr[i]) +
                                                       ds_batch_offset);
        });

        typename UniversalGemmKernel::KernelArgs gemm_kargs{{a_ptr},
                                                            {b_ptr},
                                                            ds_batch_ptr,
                                                            e_ptr,
                                                            kargs.M,
                                                            kargs.N,
                                                            kargs.K,
                                                            {kargs.stride_A},
                                                            {kargs.stride_B},
                                                            kargs.stride_Ds,
                                                            kargs.stride_E,
                                                            kargs.k_batch};

        const typename UniversalGemmKernel::SplitKBatchOffset splitk_batch_offset(gemm_kargs,
                                                                                  i_splitk);

        const ADataType* a_ptr_final = a_ptr + splitk_batch_offset.as_k_split_offset[0];
        const BDataType* b_ptr_final = b_ptr + splitk_batch_offset.bs_k_split_offset[0];

        __shared__ char smem_ptr[GetSmemSize()];

        UniversalGemmKernel::RunGemm({a_ptr_final},
                                     {b_ptr_final},
                                     ds_batch_ptr,
                                     e_ptr,
                                     smem_ptr,
                                     gemm_kargs,
                                     splitk_batch_offset,
                                     i_m,
                                     i_n);
    }
};

} // namespace ck_tile
