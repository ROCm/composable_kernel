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
        const std::vector<ck_tile::index_t>& G_lengths_, // Keep as vector for compatibility
        ck_tile::index_t stride_A_,
        ck_tile::index_t stride_B_,
        const std::array<ck_tile::index_t, NumDTensor>& stride_Ds_,
        ck_tile::index_t stride_E_,
        const std::vector<ck_tile::index_t>& G_strides_A_,
        const std::vector<ck_tile::index_t>& G_strides_B_,
        const std::array<std::vector<ck_tile::index_t>, NumDTensor>& G_strides_Ds_,
        const std::vector<ck_tile::index_t>& G_strides_E_)
        : a_ptr(a_ptr_),
          b_ptr(b_ptr_),
          ds_ptr(ds_ptr_),
          e_ptr(e_ptr_),
          k_batch(k_batch_),
          M(M_),
          N(N_),
          K(K_),
          G_lengths(G_lengths_),
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
    const std::vector<ck_tile::index_t> G_lengths;
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

    ck_tile::index_t G_lengths;
    ck_tile::index_t G_strides_A;
    ck_tile::index_t G_strides_B;
    ck_tile::index_t G_strides_E;
    // ck_tile::index_t G_strides_Ds[NumDTensor][2];     // Fixed size array for Ds
    static constexpr ck_tile::index_t ActualDTensor = (NumDTensor == 0) ? 1 : NumDTensor;
    ck_tile::index_t G_strides_Ds[ActualDTensor]; // Always at least [1][2]

    ck_tile::index_t stride_A;
    ck_tile::index_t stride_B;
    std::array<ck_tile::index_t, NumDTensor> stride_Ds;
    ck_tile::index_t stride_E;
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

        ck_tile::index_t total_G = kargs.G_lengths;
        // ✅ Use fixed bounds for hardcoded case
        // for(ck_tile::index_t i = 0; i < NumDimG && i < 2; ++i)
        // {
        //     total_G *= kargs.G_lengths[i];
        // }

        return UniversalGemmKernel::IsSupportedArgument(gemm_kargs) && total_G > 0;
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
        std::cout << "MakeKernelArgs called" << std::endl;

        // Test if we can access the vectors
        std::cout << "Testing vector access..." << std::endl;
        std::cout << "G_lengths size: " << host_args.G_lengths.size() << std::endl;
        std::cout << "G_strides_A size: " << host_args.G_strides_A.size() << std::endl;

        std::cout << "Creating KernelArgs..." << std::endl;

        KernelArgs kargs;
        kargs.a_ptr   = host_args.a_ptr;
        kargs.b_ptr   = host_args.b_ptr;
        kargs.ds_ptr  = host_args.ds_ptr;
        kargs.e_ptr   = host_args.e_ptr;
        kargs.k_batch = host_args.k_batch;
        kargs.M       = host_args.M;
        kargs.N       = host_args.N;
        kargs.K       = host_args.K;

        // Initialize with safe defaults
        kargs.G_lengths = host_args.G_lengths[0];

        kargs.G_strides_A = host_args.G_strides_A[0];
        kargs.G_strides_B = host_args.G_strides_B[0];
        kargs.G_strides_E = host_args.G_strides_E[0];

        // for(ck_tile::index_t i = 0; i < NumDTensor; ++i)
        // {
        //     kargs.G_strides_Ds[i][0] = host_dims > 0 ? host_args.G_strides_Ds[i][0] : 0;
        //     kargs.G_strides_Ds[i][1] = host_dims > 1 ? host_args.G_strides_Ds[i][1] : 0;
        // }

        if constexpr(NumDTensor > 0)
        {
            for(ck_tile::index_t i = 0; i < NumDTensor; ++i)
            {
                kargs.G_strides_Ds[i] = host_args.G_strides_Ds[i][0];
            }
        }
        else
        {

            kargs.G_strides_Ds[0] = 0;
        }

        kargs.stride_A  = host_args.stride_A;
        kargs.stride_B  = host_args.stride_B;
        kargs.stride_Ds = host_args.stride_Ds;
        kargs.stride_E  = host_args.stride_E;

        std::cout << "KernelArgs created successfully" << std::endl;
        std::cout << "Debug: G_lengths=" << kargs.G_lengths << std::endl;
        std::cout << "Debug: G_strides_A=" << kargs.G_strides_A << std::endl;
        std::cout << "Debug: G_strides_B=" << kargs.G_strides_B << std::endl;
        std::cout << "Debug: G_strides_E=" << kargs.G_strides_E << std::endl;

        return kargs;
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

        // const auto g_indices = DecomposeGIndex<NumDimG>(i_batch_flat, kargs.G_lengths);

        // const auto G_offset_A =
        //     __builtin_amdgcn_readfirstlane(CalculateGOffset<NumDimG>(g_indices,
        //     kargs.G_strides_A));

        // const auto G_offset_B =
        //     __builtin_amdgcn_readfirstlane(CalculateGOffset<NumDimG>(g_indices,
        //     kargs.G_strides_B));

        // const auto G_offset_E =
        //     __builtin_amdgcn_readfirstlane(CalculateGOffset<NumDimG>(g_indices,
        //     kargs.G_strides_E));

        const auto G_stride_A = __builtin_amdgcn_readfirstlane(kargs.G_strides_A);
        const auto G_offset_A = __builtin_amdgcn_readfirstlane(i_batch_flat * G_stride_A);

        const auto G_stride_B = __builtin_amdgcn_readfirstlane(kargs.G_strides_B);
        const auto G_offset_B = __builtin_amdgcn_readfirstlane(i_batch_flat * G_stride_B);

        const auto G_stride_E = __builtin_amdgcn_readfirstlane(kargs.G_strides_E);
        const auto G_offset_E = __builtin_amdgcn_readfirstlane(i_batch_flat * G_stride_E);

        // const auto G_offset_A = i_batch_flat * kargs.G_strides_A;  // Simple multiplication
        // const auto G_offset_B = i_batch_flat * kargs.G_strides_B;  // Simple multiplication
        // const auto G_offset_E = i_batch_flat * kargs.G_strides_E;  // Simple multiplication

        const ADataType* a_ptr = static_cast<const ADataType*>(kargs.a_ptr) + G_offset_A;
        const BDataType* b_ptr = static_cast<const BDataType*>(kargs.b_ptr) + G_offset_B;
        EDataType* e_ptr       = static_cast<EDataType*>(kargs.e_ptr) + G_offset_E;

        std::array<const void*, NumDTensor> ds_batch_ptr;
        static_for<0, NumDTensor, 1>{}([&](auto i) {
            const auto ds_G_offset =
                __builtin_amdgcn_readfirstlane(i_batch_flat * kargs.G_strides_Ds[i]);
            ds_batch_ptr[i] =
                static_cast<const void*>(static_cast<const char*>(kargs.ds_ptr[i]) + ds_G_offset);
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
