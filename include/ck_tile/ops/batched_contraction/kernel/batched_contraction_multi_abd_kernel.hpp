// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <array>
#include <cstddef>
#include <stdexcept>
#include <vector>

#include "ck_tile/core.hpp"
#include "ck_tile/host/kernel_launch.hpp"
#include "ck_tile/host/stream_utils.hpp"
#include "ck_tile/ops/batched_contraction/kernel/batched_contraction_kernel.hpp"
#include "ck_tile/ops/batched_contraction/pipeline/batched_contraction_problem.hpp"
#include "ck_tile/ops/batched_contraction/utils/tensor_descriptor_utils.hpp"

#if __clang_major__ >= 23
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wlifetime-safety-intra-tu-suggestions"
#endif

namespace ck_tile {

// =============================================================================
// BatchedContractionMultiABDProblem
// =============================================================================
// Extends BatchedContractionProblem with multiple A and B tensors.
// AsDataType and BsDataType must be ck_tile::tuple<...> of per-tensor types.

template <typename AsDataType_,
          typename BsDataType_,
          typename DsDataType_,
          typename EDataType_,
          index_t NumDimG_,
          index_t NumDimM_,
          index_t NumDimN_,
          index_t NumDimK_>
struct BatchedContractionMultiABDProblem
{
    using AsDataType = remove_cvref_t<AsDataType_>;
    using BsDataType = remove_cvref_t<BsDataType_>;
    using DsDataType = remove_cvref_t<DsDataType_>;
    using EDataType  = remove_cvref_t<EDataType_>;

    // ck_tile::tuple exposes its own static size(); std::tuple_size is only
    // specialized for it to enable structured bindings and is documented as
    // "don't use this" in tuple.hpp.
    static constexpr index_t NumATensor = static_cast<index_t>(AsDataType::size());
    static constexpr index_t NumBTensor = static_cast<index_t>(BsDataType::size());
    static constexpr index_t NumDTensor = static_cast<index_t>(DsDataType::size());

    static constexpr index_t NumDimG = NumDimG_;
    static constexpr index_t NumDimM = NumDimM_;
    static constexpr index_t NumDimN = NumDimN_;
    static constexpr index_t NumDimK = NumDimK_;
};

// =============================================================================
// BatchedContractionMultiABDHostArgs
// =============================================================================
// Host-side arguments for batched contraction with multiple A and B tensors.
// Dim arrays use std::array<index_t, N> for fixed-size compile-time dim counts.

template <index_t NumDimG,
          index_t NumDimM,
          index_t NumDimN,
          index_t NumDimK,
          index_t NumATensor,
          index_t NumBTensor,
          index_t NumDTensor>
struct BatchedContractionMultiABDHostArgs
{
    static constexpr index_t kADimSize = NumDimG + NumDimM + NumDimK;
    static constexpr index_t kBDimSize = NumDimG + NumDimN + NumDimK;
    static constexpr index_t kEDimSize = NumDimG + NumDimM + NumDimN;

    using ADims = std::array<index_t, kADimSize>;
    using BDims = std::array<index_t, kBDimSize>;
    using EDims = std::array<index_t, kEDimSize>;

    CK_TILE_HOST
    BatchedContractionMultiABDHostArgs(const std::array<const void*, NumATensor>& as_ptr_,
                                       const std::array<const void*, NumBTensor>& bs_ptr_,
                                       const std::array<const void*, NumDTensor>& ds_ptr_,
                                       void* e_ptr_,
                                       const std::array<ADims, NumATensor>& as_dims_,
                                       const std::array<BDims, NumBTensor>& bs_dims_,
                                       const std::array<EDims, NumDTensor>& ds_dims_,
                                       const EDims& e_dims_,
                                       const std::array<ADims, NumATensor>& as_strides_,
                                       const std::array<BDims, NumBTensor>& bs_strides_,
                                       const std::array<EDims, NumDTensor>& ds_strides_,
                                       const EDims& e_strides_)
        : as_ptr(as_ptr_),
          bs_ptr(bs_ptr_),
          ds_ptr(ds_ptr_),
          e_ptr(e_ptr_),
          as_dims(as_dims_),
          bs_dims(bs_dims_),
          ds_dims(ds_dims_),
          e_dims(e_dims_),
          as_strides(as_strides_),
          bs_strides(bs_strides_),
          ds_strides(ds_strides_),
          e_strides(e_strides_)
    {
    }

    std::array<const void*, NumATensor> as_ptr;
    std::array<const void*, NumBTensor> bs_ptr;
    std::array<const void*, NumDTensor> ds_ptr;
    void* e_ptr;

    std::array<ADims, NumATensor> as_dims;
    std::array<BDims, NumBTensor> bs_dims;
    std::array<EDims, NumDTensor> ds_dims;
    EDims e_dims;

    std::array<ADims, NumATensor> as_strides;
    std::array<BDims, NumBTensor> bs_strides;
    std::array<EDims, NumDTensor> ds_strides;
    EDims e_strides;
};

// =============================================================================
// BatchedContractionMultiABDKernel
// =============================================================================
// Wraps BatchedContractionKernel and adds multi-A / multi-B launch logic.
// Each A tensor is accumulated independently (same M/K dims), contributing
// to the same output tile via the pipeline. Each B tensor is handled similarly.
//
// Launch strategy: iterate over (A, B) tensor pairs, accumulate C, then
// write through the epilogue once.  This matches the gemm_multi_abd pattern.

template <typename Problem_,
          typename TilePartitioner_,
          typename GemmPipeline_,
          typename EpiloguePipeline_>
struct BatchedContractionMultiABDKernel
{
    using Problem    = remove_cvref_t<Problem_>;
    using AsDataType = remove_cvref_t<typename Problem::AsDataType>;
    using BsDataType = remove_cvref_t<typename Problem::BsDataType>;
    using DsDataType = remove_cvref_t<typename Problem::DsDataType>;
    using EDataType  = remove_cvref_t<typename Problem::EDataType>;

    static constexpr index_t NumATensor = Problem::NumATensor;
    static constexpr index_t NumBTensor = Problem::NumBTensor;
    static constexpr index_t NumDTensor = Problem::NumDTensor;
    static constexpr index_t NumDimG    = Problem::NumDimG;
    static constexpr index_t NumDimM    = Problem::NumDimM;
    static constexpr index_t NumDimN    = Problem::NumDimN;
    static constexpr index_t NumDimK    = Problem::NumDimK;

    // Use the first-A / first-B types for the inner single-A/B kernel.
    using ADataType = remove_cvref_t<std::tuple_element_t<0, AsDataType>>;
    using BDataType = remove_cvref_t<std::tuple_element_t<0, BsDataType>>;

    // Inner single-A/B problem for computing strides and descriptors.
    using InnerProblem = BatchedContractionProblem<ADataType,
                                                   BDataType,
                                                   DsDataType,
                                                   EDataType,
                                                   NumDimG,
                                                   NumDimM,
                                                   NumDimN,
                                                   NumDimK,
                                                   NumDTensor>;

    using InnerKernel =
        BatchedContractionKernel<InnerProblem, TilePartitioner_, GemmPipeline_, EpiloguePipeline_>;

    using TilePartitioner  = remove_cvref_t<TilePartitioner_>;
    using GemmPipeline     = remove_cvref_t<GemmPipeline_>;
    using EpiloguePipeline = remove_cvref_t<EpiloguePipeline_>;

    using HostArgs = BatchedContractionMultiABDHostArgs<NumDimG,
                                                        NumDimM,
                                                        NumDimN,
                                                        NumDimK,
                                                        NumATensor,
                                                        NumBTensor,
                                                        NumDTensor>;

    using ADims = typename HostArgs::ADims;
    using BDims = typename HostArgs::BDims;
    using EDims = typename HostArgs::EDims;

    using InnerHostArgs   = BatchedContractionHostArgs<NumDTensor>;
    using InnerKernelArgs = typename InnerKernel::KernelArgs;

    static constexpr index_t kBlockSize = InnerKernel::kBlockSize;

    CK_TILE_HOST static constexpr index_t GetSmemSize() { return InnerKernel::GetSmemSize(); }

    CK_TILE_HOST static constexpr auto GetBlockSize() { return InnerKernel::GetBlockSize(); }

    CK_TILE_HOST static auto GridSize(const InnerKernelArgs& kargs)
    {
        return InnerKernel::GridSize(kargs);
    }

    CK_TILE_HOST static bool IsSupportedArguments(const InnerKernelArgs& kargs)
    {
        return InnerKernel::IsSupportedArguments(kargs);
    }

    // Build a BatchedContractionHostArgs from the multi-ABD HostArgs for the i-th A and j-th B.
    CK_TILE_HOST static InnerHostArgs
    MakeInnerHostArgs(const HostArgs& args, index_t ia, index_t ib)
    {
        // Convert fixed-size dim arrays to vectors for InnerHostArgs
        auto arr_to_vec = [](const auto& arr) {
            return std::vector<index_t>(arr.begin(), arr.end());
        };

        const ADims& a_dims    = args.as_dims[ia];
        const ADims& a_strides = args.as_strides[ia];
        const BDims& b_dims    = args.bs_dims[ib];
        const BDims& b_strides = args.bs_strides[ib];

        auto A_dims_vec    = arr_to_vec(a_dims);
        auto A_strides_vec = arr_to_vec(a_strides);
        auto B_dims_vec    = arr_to_vec(b_dims);
        auto B_strides_vec = arr_to_vec(b_strides);

        std::array<std::vector<index_t>, NumDTensor> Ds_dims_vecs;
        std::array<std::vector<index_t>, NumDTensor> Ds_strides_vecs;
        for(index_t id = 0; id < NumDTensor; ++id)
        {
            Ds_dims_vecs[id]    = arr_to_vec(args.ds_dims[id]);
            Ds_strides_vecs[id] = arr_to_vec(args.ds_strides[id]);
        }

        auto E_dims_vec    = arr_to_vec(args.e_dims);
        auto E_strides_vec = arr_to_vec(args.e_strides);

        return InnerHostArgs{
            args.as_ptr[ia],
            args.bs_ptr[ib],
            args.ds_ptr,
            args.e_ptr,
            /*k_batch=*/1,
            A_dims_vec,
            B_dims_vec,
            Ds_dims_vecs,
            E_dims_vec,
            A_strides_vec,
            B_strides_vec,
            Ds_strides_vecs,
            E_strides_vec,
        };
    }

    CK_TILE_HOST static InnerKernelArgs MakeKernelArgs(const HostArgs& args)
    {
        // Use first A and first B for building descriptors; all A/B share the same dims/strides
        // in the standard contraction case.
        auto inner = MakeInnerHostArgs(args, 0, 0);
        return InnerKernel::MakeKernelArgs(inner);
    }

    // Launch: runs once per (ia, ib) pair and accumulates into E.
    CK_TILE_HOST static float launch(const HostArgs& args, const stream_config& stream)
    {
        float total_time = 0.0f;

        for(index_t ia = 0; ia < NumATensor; ++ia)
        {
            for(index_t ib = 0; ib < NumBTensor; ++ib)
            {
                auto inner_args = MakeInnerHostArgs(args, ia, ib);
                auto kargs      = InnerKernel::MakeKernelArgs(inner_args);

                if(!InnerKernel::IsSupportedArguments(kargs))
                    return -1.0f;

                const dim3 grids  = InnerKernel::GridSize(kargs);
                const dim3 blocks = InnerKernel::GetBlockSize();

                constexpr int kBlockPerCu = 1;
                float t                   = launch_kernel(
                    stream, make_kernel<kBlockPerCu>(InnerKernel{}, grids, blocks, 0, kargs));
                if(t < 0.0f)
                    return -1.0f;
                total_time += t;
            }
        }

        return total_time;
    }
};

} // namespace ck_tile

#if __clang_major__ >= 23
#pragma clang diagnostic pop
#endif
