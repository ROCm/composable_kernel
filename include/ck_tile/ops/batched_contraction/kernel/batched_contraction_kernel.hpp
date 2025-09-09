// SPDX-License-Identifier: MIT
// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/ops/batched_contraction/pipeline/batched_contraction_problem.hpp"
#include "ck_tile/ops/gemm/kernel/universal_gemm_kernel.hpp"
#include "ck_tile/ops/batched_contraction/kernel/batched_conratction_utils.hpp"

namespace ck_tile {

template <ck_tile::index_t NumDTensor = 0>
struct BatchedContractionHostArgs
{
    CK_TILE_HOST
    BatchedContractionHostArgs(
        const void* a_ptr_,
        const void* b_ptr_,
        const std::array<const void*, NumDTensor>& ds_ptr_,
        void* e_ptr_,
        ck_tile::index_t k_batch_,
        ck_tile::index_t M_,
        ck_tile::index_t N_,
        ck_tile::index_t K_,
        const std::vector<ck_tile::index_t>& G_Lengths_, // [G0, G1, G2, ... , G_{NumDimG-1}]
        ck_tile::index_t stride_A_,
        ck_tile::index_t stride_B_,
        const std::array<ck_tile::index_t, NumDTensor>& stride_Ds_,
        ck_tile::index_t stride_E_,
        const std::vector<ck_tile::index_t>&
            G_strides_A_, // [G0_stride_A, G1_stride_A, ... , G_{NumDimG-1}_stride_A]
        const std::vector<ck_tile::index_t>&
            G_strides_B_, // [G0_stride_B, G1_stride_B, ... , G_{NumDimG-1}_stride_B]
        const std::array<ck_tile::index_t, NumDTensor>& G_strides_Ds_,
        const std::vector<ck_tile::index_t>&
            G_strides_E_) // [G0_stride_E, G1_stride_E, ... , G_{NumDimG-1}_stride_E]
        : a_ptr(a_ptr_),
          b_ptr(b_ptr_),
          ds_ptr(ds_ptr_),
          e_ptr(e_ptr_),
          k_batch(k_batch_),
          M(M_),
          N(N_),
          K(K_),
          G_Lengths(G_Lengths_),
          stride_A(stride_A_),
          stride_B(stride_B_),
          stride_Ds(stride_Ds_),
          stride_E(stride_E_),
          G_strides_A(G_strides_A_),
          G_strides_B(G_strides_B_),
          G_strides_Ds(G_strides_Ds_),
          G_strides_E(G_strides_E_)
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
    const std::vector<ck_tile::index_t> G_Lengths;
    ck_tile::index_t stride_A;
    ck_tile::index_t stride_B;
    std::array<ck_tile::index_t, NumDTensor> stride_Ds;
    ck_tile::index_t stride_E;
    const std::vector<ck_tile::index_t> G_strides_A;
    const std::vector<ck_tile::index_t> G_strides_B;
    std::array<std::vector<ck_tile::index_t>, NumDTensor> G_strides_Ds;
    const std::vector<ck_tile::index_t> G_strides_E;
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
    std::vector<ck_tile::index_t> G_Lengths; // [G0, G1, G2, ... , G_{NumDimG-1}]
    ck_tile::index_t stride_A;
    ck_tile::index_t stride_B;
    std::array<ck_tile::index_t, NumDTensor> stride_Ds;
    ck_tile::index_t stride_E;

    std::vector<ck_tile::index_t>
        G_strides_A; // [G0_stride_A, G1_stride_A, ... , G_{NumDimG-1}_stride_A]
    std::vector<ck_tile::index_t>
        G_strides_B; // [G0_stride_B, G1_stride_B, ... , G_{NumDimG-1}_stride_B]
    std::array<std::vector<ck_tile::index_t>, NumDTensor> G_strides_Ds;
    ck_tile::index_t G_strides_E;
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

        return UniversalGemmKernel::IsSupportedArgument(gemm_kargs) && kargs.G > 0;
    }

    CK_TILE_HOST static constexpr ck_tile::index_t GetSmemSize()
    {
        return UniversalGemmKernel::GetSmemSize();
    }

    CK_TILE_HOST static constexpr auto GetBlockSize()
    {
        return dim3(UniversalGemmKernel::kBlockSize);
    }

    CK_TILE_HOST static constexpr auto GridSize(ck_tile::index_t M,
                                                ck_tile::index_t N,
                                                ck_tile::index_t KBatch,
                                                const std::vector<ck_tile::index_t>& G_lengths)
    {
        ck_tile::index_t total_G = 1;
        for(auto g_len : G_lengths)
        {
            total_G *= g_len;
        }
        return dim3(TilePartitioner::GridSize(M, N), total_G, KBatch);
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
                          host_args.G_Lengths,
                          host_args.stride_A,
                          host_args.stride_B,
                          host_args.stride_Ds,
                          host_args.stride_E,
                          host_args.G_strides_A,
                          host_args.G_strides_B,
                          host_args.G_strides_Ds,
                          host_args.G_strides_E};
    }

    CK_TILE_DEVICE void operator()(const KernelArgs& kargs) const
    {
        const auto [iM, iN] = TilePartitioner{kargs.M, kargs.N}.GetOutputTileIndex(blockIdx.x);
        const ck_tile::index_t i_m =
            __builtin_amdgcn_readfirstlane(iM * TilePartitioner::MPerBlock);
        const ck_tile::index_t i_n =
            __builtin_amdgcn_readfirstlane(iN * TilePartitioner::NPerBlock);

        const auto i_batch_flat = __builtin_amdgcn_readfirstlane(blockIdx.y);
        const auto i_splitk     = __builtin_amdgcn_readfirstlane(blockIdx.z);

        const auto g_indices = DecomposeGIndex<NumDimG>(i_batch_flat, kargs.G_lengths);

        const auto G_offset_A =
            __builtin_amdgcn_readfirstlane(CalculateGOffset<NumDimG>(g_indices, kargs.G_strides_A));
        const ADataType* a_ptr = static_cast<const ADataType*>(kargs.a_ptr) + G_offset_A;

        const auto G_offset_B =
            __builtin_amdgcn_readfirstlane(CalculateGOffset<NumDimG>(g_indices, kargs.G_strides_B));
        const BDataType* b_ptr = static_cast<const BDataType*>(kargs.b_ptr) + G_offset_B;

        const auto G_offset_E =
            __builtin_amdgcn_readfirstlane(CalculateGOffset<NumDimG>(g_indices, kargs.G_strides_E));
        EDataType* e_ptr = static_cast<EDataType*>(kargs.e_ptr) + G_offset_E;

        std::array<const void*, NumDTensor> ds_batch_ptr;
        static_for<0, NumDTensor, 1>{}([&](auto i) {
            const auto G_offset_D = __builtin_amdgcn_readfirstlane(
                CalculateGOffset<NumDimG>(g_indices, kargs.G_strides_Ds[i]));
            ds_batch_ptr[i] =
                static_cast<const void*>(static_cast<const char*>(kargs.ds_ptr[i]) + G_offset_D);
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
