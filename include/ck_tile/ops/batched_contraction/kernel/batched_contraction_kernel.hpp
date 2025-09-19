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
        ck_tile::index_t G_total_,
        ck_tile::index_t M_total_,
        ck_tile::index_t N_total_,
        ck_tile::index_t K_total_,
        const std::vector<ck_tile::index_t>& A_dims_, // [G0, G1, ..., M0, M1, ... , K0, K1, ...]
        const std::vector<ck_tile::index_t>& B_dims_, // [G0, G1, ..., N0, N1, ... , K0, K1, ...]
        const std::array<std::vector<ck_tile::index_t>, NumDTensor>&
            Ds_dims_, // [G0, G1, ..., M0, M1, ... , N0, N1, ...][NumDTensor]
        const std::vector<ck_tile::index_t>& E_dims_, // [G0, G1, ..., M0, M1, ... , N0, N1, ...]

        const std::vector<ck_tile::index_t>& A_strides_, // [G0, G1, ..., M0, M1, ...,K0, K1, ...]
        const std::vector<ck_tile::index_t>& B_strides_, // [G0, G1, ..., N0, N1, ...,K0, K1, ...]
        const std::array<std::vector<ck_tile::index_t>, NumDTensor>&
            Ds_strides_, // [G0, G1, ..., M0, M1, ...,N0, N1, ...]
        const std::vector<ck_tile::index_t>&
            E_strides_) // [G0, G1, ..., M0, M1, ...,N0, N1, ...][NumDTensor]

        : a_ptr(a_ptr_),
          b_ptr(b_ptr_),
          ds_ptr(ds_ptr_),
          e_ptr(e_ptr_),
          k_batch(k_batch_),
          G_total(G_total_),
          M_total(M_total_),
          N_total(N_total_),
          K_total(K_total_),
          A_dims(A_dims_),
          B_dims(B_dims_),
          Ds_dims(Ds_dims_),
          E_dims(E_dims_),
          A_strides(A_strides_),
          B_strides(B_strides_),
          Ds_strides(Ds_strides_),
          E_strides(E_strides_)
    {
    }

    const void* a_ptr;
    const void* b_ptr;
    std::array<const void*, NumDTensor> ds_ptr;
    void* e_ptr;
    ck_tile::index_t k_batch;
    ck_tile::index_t G_total;
    ck_tile::index_t M_total;
    ck_tile::index_t N_total;
    ck_tile::index_t K_total;
    const std::vector<ck_tile::index_t> A_dims; // [G0, G1, ..., M0, M1, ... , K0, K1, ...]
    const std::vector<ck_tile::index_t> B_dims; // [G0, G1, ..., N0, N1, ... , K0, K1, ...]
    const std::array<std::vector<ck_tile::index_t>, NumDTensor>
        Ds_dims; // [G0, G1, ..., M0, M1, ... , N0, N1, ...][NumDTensor]
    const std::vector<ck_tile::index_t> E_dims;    // [G0, G1, ..., M0, M1, ... , N0, N1, ...]
    const std::vector<ck_tile::index_t> A_strides; // [G0, G1, ..., M0, M1, ...,K0, K1, ...]
    const std::vector<ck_tile::index_t> B_strides; // [G0, G1, ..., N0, N1, ...,K0, K1, ...]
    const std::array<std::vector<ck_tile::index_t>, NumDTensor>
        Ds_strides; // [G0, G1, ..., M0, M1, ...,N0, N1, ...]
    const std::vector<ck_tile::index_t>
        E_strides; // [G0, G1, ..., M0, M1, ...,N0, N1, ...][NumDTensor]
};

template <ck_tile::index_t NumDimG,
          ck_tile::index_t NumDimM,
          ck_tile::index_t NumDimN,
          ck_tile::index_t NumDimK,
          ck_tile::index_t NumDTensor = 0>
struct BatchedContractionKernelArgs
{
    const void* a_ptr;
    const void* b_ptr;
    std::array<const void*, NumDTensor> ds_ptr;
    void* e_ptr;
    ck_tile::index_t k_batch;

    ck_tile::index_t M_dims[NumDimM]; // [M0, M1, M2, ... , M_{NumDimM-1}]
    ck_tile::index_t N_dims[NumDimN]; // [N0, N1, N2, ... , N_{NumDimN-1}]
    ck_tile::index_t K_dims[NumDimK]; // [K0, K1, K2, ... , K_{NumDimK-1}]
    ck_tile::index_t G_dims[NumDimG]; // [G0, G1, G2, ... , G_{NumDimG-1}]

    // G_batch strides
    ck_tile::index_t
        G_strides_A[NumDimG]; // [G0_stride_A, G1_stride_A, ... , G_{NumDimG-1}_stride_A]
    ck_tile::index_t
        G_strides_B[NumDimG]; // [G0_stride_B, G1_stride_B, ... , G_{NumDimG-1}_stride_B]
    ck_tile::index_t
        G_strides_E[NumDimG]; // [G0_stride_E, G1_stride_E, ... , G_{NumDimG-1}_stride_E]
    std::array<std::array<ck_tile::index_t, NumDimG>, NumDTensor> G_strides_Ds;

    ck_tile::index_t G_total; // total G length
    ck_tile::index_t M_total; // total M length
    ck_tile::index_t N_total; // total N length
    ck_tile::index_t K_total; // total K length

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

    using KernelArgs = BatchedContractionKernelArgs<NumDimG, NumDimM, NumDimN, NumDimK, NumDTensor>;

    CK_TILE_HOST static constexpr auto GetKernelName() { return "batched_contraction_kernel"; }

    CK_TILE_HOST static constexpr bool IsSupportedArguments(const KernelArgs& kargs)
    {
        typename UniversalGemmKernel::KernelArgs gemm_kargs{{kargs.a_ptr},
                                                            {kargs.b_ptr},
                                                            {kargs.ds_ptr},
                                                            kargs.e_ptr,
                                                            kargs.M_total,
                                                            kargs.N_total,
                                                            kargs.K_total,
                                                            {kargs.stride_A},
                                                            {kargs.stride_B},
                                                            {kargs.stride_Ds},
                                                            kargs.stride_E,
                                                            kargs.k_batch};

        return UniversalGemmKernel::IsSupportedArgument(gemm_kargs) && kargs.G_total > 0;
    }

    CK_TILE_HOST static constexpr ck_tile::index_t GetSmemSize()
    {
        return UniversalGemmKernel::GetSmemSize();
    }

    CK_TILE_HOST static constexpr auto GetBlockSize()
    {
        return dim3(UniversalGemmKernel::kBlockSize);
    }

    CK_TILE_HOST static constexpr auto GridSize(const KernelArgs& kargs)
    {
        return dim3(
            TilePartitioner::GridSize(kargs.M_total, kargs.N_total), kargs.G_total, kargs.k_batch);
    }

    // A[G0, G1, ..., M0, M1, M2, ..., K0, K1, K2, ...]
    CK_TILE_HOST static constexpr auto
    Make_A_GridDescriptor_M_K(const std::vector<ck_tile::index_t>& A_dims    = {},
                              const std::vector<ck_tile::index_t>& A_strides = {})
    {
        const auto to_tuple = [&](auto& vec, auto start, auto end) {
            return generate_tuple([&](auto i) { return vec[start + i]; }, number<end - start>{});
        };

        // Remove G Dimensions
        const auto A_dims_M_K =
            to_tuple(A_dims, number<NumDimG>{}, number<NumDimG + NumDimM + NumDimK>{});
        const auto A_strides_M_K =
            to_tuple(A_strides, number<NumDimG>{}, number<NumDimG + NumDimM + NumDimK>{});

        // dimension Ids for M and K
        constexpr auto A_dims_M_ids = typename arithmetic_sequence_gen<0, NumDimM, 1>::type{};
        constexpr auto A_dims_K_ids =
            typename arithmetic_sequence_gen<NumDimM, NumDimM + NumDimK, 1>::type{};

        // Dimensions for M [M0, M1, ...] and K [K0, K1, ...]
        const auto dims_M = get_container_subset(A_dims_M_K, A_dims_M_ids);
        const auto dims_K = get_container_subset(A_dims_M_K, A_dims_K_ids);

        // naive tensor A[M0, M1, M2, ..., K0, K1, K2...] Discriptor
        const auto A_grid_desc_Ms_Ks =
            ck_tile::make_naive_tensor_descriptor(A_dims_M_K, A_strides_M_K);

        // transformed tensor to flatten M and K dimensions  [M_total = M0 * M1 * M2 * ... , K_total
        // = K0 * K1 * K2 * ...]
        const auto A_grid_desc_Mflat_Kflat = ck_tile::transform_tensor_descriptor(
            A_grid_desc_Ms_Ks,
            make_tuple(make_merge_transform(dims_M), make_merge_transform(dims_K)),
            make_tuple(A_dims_M_ids, A_dims_K_ids),
            make_tuple(sequence<0>{}, sequence<1>{}));

        return A_grid_desc_Mflat_Kflat;
    }

    // B[G0, G1, ..., N0, N1, N2, ..., K0, K1, K2, ...]
    CK_TILE_HOST static constexpr auto
    Make_B_GridDescriptor_N_K(const std::vector<ck_tile::index_t>& B_dims    = {},
                              const std::vector<ck_tile::index_t>& B_strides = {})
    {
        const auto to_tuple = [&](auto& vec, auto start, auto end) {
            return generate_tuple([&](auto i) { return vec[start + i]; }, number<end - start>{});
        };

        // Remove G Dimensions
        const auto B_dims_N_K =
            to_tuple(B_dims, number<NumDimG>{}, number<NumDimG + NumDimN + NumDimK>{});
        const auto B_strides_N_K =
            to_tuple(B_strides, number<NumDimG>{}, number<NumDimG + NumDimN + NumDimK>{});

        // dimension Ids for N and K
        constexpr auto B_dims_N_ids = typename arithmetic_sequence_gen<0, NumDimN, 1>::type{};
        constexpr auto B_dims_K_ids =
            typename arithmetic_sequence_gen<NumDimN, NumDimN + NumDimK, 1>::type{};

        // Dimensions for N [N0, N1, ...] and K [K0, K1, ...]
        const auto dims_N = get_container_subset(B_dims_N_K, B_dims_N_ids);
        const auto dims_K = get_container_subset(B_dims_N_K, B_dims_K_ids);

        // naive tensor B[N0, N1, N2, ..., K0, K1, K2...] Discriptor
        const auto B_grid_desc_Ns_Ks =
            ck_tile::make_naive_tensor_descriptor(B_dims_N_K, B_strides_N_K);

        // transformed tensor to flatten N and K dimensions  [N_total = N0 * N1 * N2 * ... , K_total
        // = K0 * K1 * K2 * ...]
        const auto B_grid_desc_Nflat_Kflat = ck_tile::transform_tensor_descriptor(
            B_grid_desc_Ns_Ks,
            make_tuple(make_merge_transform(dims_N), make_merge_transform(dims_K)),
            make_tuple(B_dims_N_ids, B_dims_K_ids),
            make_tuple(sequence<0>{}, sequence<1>{}));

        return B_grid_desc_Nflat_Kflat;
    }

    // E[G0, G1, ..., M0, M1, M2, ..., N0, N1, N2, ...]
    CK_TILE_HOST static constexpr auto
    Make_E_GridDescriptor_M_N(const std::vector<ck_tile::index_t>& E_dims    = {},
                              const std::vector<ck_tile::index_t>& E_strides = {})
    {
        const auto to_tuple = [&](auto& vec, auto start, auto end) {
            return generate_tuple([&](auto i) { return vec[start + i]; }, number<end - start>{});
        };

        // Remove G dimensions
        const auto E_dims_M_N =
            to_tuple(E_dims, number<NumDimG>{}, number<NumDimG + NumDimM + NumDimN>{});
        const auto E_strides_M_N =
            to_tuple(E_strides, number<NumDimG>{}, number<NumDimG + NumDimM + NumDimN>{});

        // dimension Ids for M and N
        constexpr auto E_dims_M_ids = typename arithmetic_sequence_gen<0, NumDimM, 1>::type{};
        constexpr auto E_dims_N_ids =
            typename arithmetic_sequence_gen<NumDimM, NumDimM + NumDimN, 1>::type{};

        // Dimensions for M and N
        const auto dims_M = get_container_subset(E_dims_M_N, E_dims_M_ids);
        const auto dims_N = get_container_subset(E_dims_M_N, E_dims_N_ids);

        // naive tensor E[M0, M1, M2, ..., N0, N1, N2...] Discriptor
        const auto E_grid_desc_Ms_Ns =
            ck_tile::make_naive_tensor_descriptor(E_dims_M_N, E_strides_M_N);

        // transformed tensor to flatten M and N dimensions   [M_total = M0 * M1 * M2 * ... ,
        // N_total = N0 * N1 * N2 * ...]
        const auto E_grid_desc_Mflat_Nflat = ck_tile::transform_tensor_descriptor(
            E_grid_desc_Ms_Ns,
            make_tuple(make_merge_transform(dims_M), make_merge_transform(dims_N)),
            make_tuple(E_dims_M_ids, E_dims_N_ids),
            make_tuple(sequence<0>{}, sequence<1>{}));

        return E_grid_desc_Mflat_Nflat;
    }

    CK_TILE_HOST static constexpr KernelArgs
    MakeKernelArgs(const BatchedContractionHostArgs<NumDTensor>& host_args)
    {
        const auto expected_A_dims = NumDimG + NumDimM + NumDimK;
        const auto expected_B_dims = NumDimG + NumDimN + NumDimK;
        const auto expected_E_dims = NumDimG + NumDimM + NumDimN;

        if(host_args.A_dims.size() != expected_A_dims ||
           host_args.A_strides.size() != expected_A_dims)
        {
            throw std::invalid_argument("A dimension size mismatch");
        }
        if(host_args.B_dims.size() != expected_B_dims ||
           host_args.B_strides.size() != expected_B_dims)
        {
            throw std::invalid_argument("B dimension size mismatch");
        }
        if(host_args.E_dims.size() != expected_E_dims ||
           host_args.E_strides.size() != expected_E_dims)
        {
            throw std::invalid_argument("E dimension size mismatch");
        }

        for(ck_tile::index_t d = 0; d < NumDTensor; ++d)
        {
            if(host_args.Ds_dims[d].size() != expected_E_dims ||
               host_args.Ds_strides[d].size() != expected_E_dims)
            {
                throw std::invalid_argument("D dimension size mismatch");
            }
        }

        KernelArgs kargs;
        kargs.a_ptr   = host_args.a_ptr;
        kargs.b_ptr   = host_args.b_ptr;
        kargs.ds_ptr  = host_args.ds_ptr;
        kargs.e_ptr   = host_args.e_ptr;
        kargs.k_batch = host_args.k_batch;

        for(ck_tile::index_t i = 0; i < NumDimG; ++i)
        {
            kargs.G_dims[i]      = host_args.A_dims[i];
            kargs.G_strides_A[i] = host_args.A_strides[i];
            kargs.G_strides_B[i] = host_args.B_strides[i];
            kargs.G_strides_E[i] = host_args.E_strides[i];
        }
        for(ck_tile::index_t i = 0; i < NumDimM; ++i)
        {
            kargs.M_dims[i] = host_args.A_dims[NumDimG + i];
            if(kargs.M_dims[i] != host_args.E_dims[NumDimG + i])
            {
                throw std::invalid_argument("M dimension mismatch between A and E tensors");
            }
        }
        for(ck_tile::index_t i = 0; i < NumDimN; ++i)
        {
            kargs.N_dims[i] = host_args.B_dims[NumDimG + i];
            if(kargs.N_dims[i] != host_args.E_dims[NumDimG + NumDimM + i])
            {
                throw std::invalid_argument("N dimension mismatch between B and E tensors");
            }
        }
        for(ck_tile::index_t i = 0; i < NumDimK; ++i)
        {
            kargs.K_dims[i] = host_args.A_dims[NumDimG + NumDimM + i];
            if(kargs.K_dims[i] != host_args.B_dims[NumDimG + NumDimN + i])
            {
                throw std::invalid_argument("K dimension mismatch between A and B tensors");
            }
        }

        kargs.G_total = host_args.G_total;
        kargs.M_total = host_args.M_total;
        kargs.N_total = host_args.N_total;
        kargs.K_total = host_args.K_total;

        kargs.stride_A = kargs.K_total;
        kargs.stride_B = kargs.K_total;
        kargs.stride_E = kargs.N_total;

        for(ck_tile::index_t d = 0; d < NumDTensor; ++d)
        {
            for(ck_tile::index_t i = 0; i < NumDimG; ++i)
            {
                kargs.G_strides_Ds[d][i] = host_args.Ds_strides[d][i];
            }
            kargs.stride_Ds[d] = kargs.N_total; // D tensors same shape as E
        }

        return kargs;
    }
    struct BatchOffsets
    {
        ck_tile::index_t a;
        ck_tile::index_t b;
        ck_tile::index_t e;
        std::array<ck_tile::index_t, NumDTensor> ds;
    };

    CK_TILE_DEVICE constexpr auto
    CalculateOffsetFromFlatIndex(ck_tile::index_t flat_g_idx,
                                 const ck_tile::index_t* G_dims,
                                 const ck_tile::index_t* G_strides) const
    {
        ck_tile::index_t offset    = 0;
        ck_tile::index_t remaining = flat_g_idx;
        // Iterate from last to first dimension
        for(int i = NumDimG - 1; i >= 0; --i)
        {
            const ck_tile::index_t idx = remaining % G_dims[i];
            offset += idx * G_strides[i];
            remaining /= G_dims[i];
        }
        return offset;
    }

    CK_TILE_DEVICE constexpr auto CalculateAllBatchOffsets(ck_tile::index_t flat_g_index,
                                                           const KernelArgs& kargs) const
    {
        BatchOffsets offsets;
        offsets.a = CalculateOffsetFromFlatIndex(flat_g_index, kargs.G_dims, kargs.G_strides_A);
        offsets.b = CalculateOffsetFromFlatIndex(flat_g_index, kargs.G_dims, kargs.G_strides_B);
        offsets.e = CalculateOffsetFromFlatIndex(flat_g_index, kargs.G_dims, kargs.G_strides_E);
        static_for<0, NumDTensor, 1>{}([&](auto i) {
            offsets.ds[i] = CalculateOffsetFromFlatIndex(
                flat_g_index, kargs.G_dims, kargs.G_strides_Ds[i].data());
        });
        return offsets;
    }

    CK_TILE_DEVICE void operator()(const KernelArgs& kargs) const
    {

        const auto [iM, iN] =
            TilePartitioner{kargs.M_total, kargs.N_total}.GetOutputTileIndex(blockIdx.x);
        const ck_tile::index_t i_m =
            __builtin_amdgcn_readfirstlane(iM * TilePartitioner::MPerBlock);
        const ck_tile::index_t i_n =
            __builtin_amdgcn_readfirstlane(iN * TilePartitioner::NPerBlock);

        const auto i_batch_flat = __builtin_amdgcn_readfirstlane(blockIdx.y);
        const auto i_splitk     = __builtin_amdgcn_readfirstlane(blockIdx.z);

        const auto offsets = CalculateAllBatchOffsets(i_batch_flat, kargs);

        const ADataType* a_ptr = static_cast<const ADataType*>(kargs.a_ptr) + offsets.a;
        const BDataType* b_ptr = static_cast<const BDataType*>(kargs.b_ptr) + offsets.b;
        EDataType* e_ptr       = static_cast<EDataType*>(kargs.e_ptr) + offsets.e;

        std::array<const void*, NumDTensor> ds_batch_ptr;
        static_for<0, NumDTensor, 1>{}([&](auto i) {
            ds_batch_ptr[i] =
                static_cast<const void*>(static_cast<const char*>(kargs.ds_ptr[i]) + offsets.ds[i]);
        });

        typename UniversalGemmKernel::KernelArgs gemm_kargs{{a_ptr},
                                                            {b_ptr},
                                                            {ds_batch_ptr},
                                                            e_ptr,
                                                            kargs.M_total,
                                                            kargs.N_total,
                                                            kargs.K_total,
                                                            {kargs.stride_A},
                                                            {kargs.stride_B},
                                                            {kargs.stride_Ds},
                                                            kargs.stride_E,
                                                            kargs.k_batch};

        const typename UniversalGemmKernel::SplitKBatchOffset splitk_batch_offset(gemm_kargs,
                                                                                  i_splitk);

        const ADataType* a_ptr_final = a_ptr + splitk_batch_offset.as_k_split_offset[0];
        const BDataType* b_ptr_final = b_ptr + splitk_batch_offset.bs_k_split_offset[0];

        // if(threadIdx.x == 0 && blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0)
        // {
        //     printf("BatchedContractionKernel: NumDTensor=%ld\n", static_cast<long>(NumDTensor));
        //     for (ck_tile::index_t d = 0; d < NumDTensor; ++d)
        //     {
        //         // printf("ds_batch_ptr[%ld]: %p\n", static_cast<long>(d), ds_batch_ptr[d]);
        //         // Print first few values of each D tensor
        //         const float* d_ptr = static_cast<const float*>(ds_batch_ptr[d]);
        //         printf("ds[%ld][0]=%f ds[%ld][1]=%f ds[%ld][2]=%f\n",
        //             static_cast<long>(d), d_ptr[0],
        //             static_cast<long>(d), d_ptr[1],
        //             static_cast<long>(d), d_ptr[2]);
        //     }
        // }

        __shared__ char smem_ptr[GetSmemSize()];

        UniversalGemmKernel::RunGemm({a_ptr_final},
                                     {b_ptr_final},
                                     {ds_batch_ptr},
                                     e_ptr,
                                     smem_ptr,
                                     gemm_kargs,
                                     splitk_batch_offset,
                                     i_m,
                                     i_n);
    }
};

} // namespace ck_tile
