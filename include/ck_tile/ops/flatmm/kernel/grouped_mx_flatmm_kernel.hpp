// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <stdexcept>
#include <string>

#include "ck_tile/core.hpp"
#include "ck_tile/ops/common.hpp"
#include "ck_tile/ops/flatmm/kernel/mx_flatmm_kernel.hpp"
#include "ck_tile/ops/flatmm/kernel/grouped_flatmm_kernel.hpp"

#if __clang_major__ >= 23
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wlifetime-safety-intra-tu-suggestions"
#endif
namespace ck_tile {

template <typename TilePartitioner_, typename MXFlatmmPipeline_, typename EpiloguePipeline_>
struct GroupedMXFlatmmKernel
    : MXFlatmmKernel<TilePartitioner_, MXFlatmmPipeline_, EpiloguePipeline_>
{
    using UnderlyingGemmKernel =
        MXFlatmmKernel<TilePartitioner_, MXFlatmmPipeline_, EpiloguePipeline_>;

    using TilePartitioner  = remove_cvref_t<TilePartitioner_>;
    using MXFlatmmPipeline = remove_cvref_t<MXFlatmmPipeline_>;
    using EpiloguePipeline = remove_cvref_t<EpiloguePipeline_>;

    using ADataType  = remove_cvref_t<typename MXFlatmmPipeline::ADataType>;
    using BDataType  = remove_cvref_t<typename MXFlatmmPipeline::BDataType>;
    using CDataType  = remove_cvref_t<typename EpiloguePipeline::ODataType>;
    using DsLayout   = remove_cvref_t<typename EpiloguePipeline::DsLayout>;
    using DsDataType = remove_cvref_t<typename EpiloguePipeline::DsDataType>;

    static constexpr index_t NumDTensor = DsDataType::size();
    static constexpr index_t kBlockSize = MXFlatmmPipeline_::BlockSize;

    static_assert(DsLayout::size() == DsDataType::size(),
                  "The size of DsLayout and DsDataType should be the same");

    [[nodiscard]] CK_TILE_HOST static const std::string GetName()
    {
        return concat('_',
                      "grouped_mx_flatmm",
                      gemm_prec_str<ADataType, BDataType>(),
                      MXFlatmmPipeline::GetName());
    }

    template <class ScaleM, class ScaleN, index_t NumDTensor_ = 0>
    CK_TILE_HOST static auto
    GridSize(const GroupedFlatmmHostArgs<ScaleM, ScaleN, NumDTensor_>& kernelArgs)
    {
        hipDeviceProp_t prop;
        int deviceId = 0;

        const int block_size     = UnderlyingGemmKernel::BlockSize().x;
        int dyn_smem_size        = 0;
        int maxActiveBlocksPerCU = 0;

        if(hipGetDeviceProperties(&prop, deviceId) != hipSuccess)
            throw std::runtime_error(std::string("hipGetDeviceProperties failed: ") +
                                     hipGetErrorName(hipGetLastError()));

        if(hipOccupancyMaxActiveBlocksPerMultiprocessor(
               &maxActiveBlocksPerCU,
               reinterpret_cast<void*>(kentry<1,
                                              GroupedMXFlatmmKernel,
                                              GroupedFlatmmHostArgs<ScaleM, ScaleN, NumDTensor_>>),
               block_size,
               dyn_smem_size) != hipSuccess)
            throw std::runtime_error(
                std::string("hipOccupancyMaxActiveBlocksPerMultiprocessor failed: ") +
                hipGetErrorName(hipGetLastError()));

        const int persistent_block_size = prop.multiProcessorCount * maxActiveBlocksPerCU;

        if(kernelArgs.k_batch != 1)
            throw std::runtime_error(
                "GroupedMXFlatmmKernel only supports k_batch == 1 (Split-K is not supported).");
        return dim3(persistent_block_size, 1, kernelArgs.k_batch);
    }

    template <typename HostArgs>
    CK_TILE_HOST static constexpr auto MakeKernelArgs(const HostArgs& hostArgs)
    {
        return hostArgs;
    }

    template <class ScaleM, class ScaleN, index_t NumDTensor_ = 0>
    CK_TILE_DEVICE void operator()(GroupedFlatmmHostArgs<ScaleM, ScaleN, NumDTensor_> kargs) const
    {
        index_t group_idx        = 0;
        index_t block_linear_idx = blockIdx.x;
        index_t total_block_cnt  = gridDim.x;

        UnderlyingGemmKernel underlying_kernel{};
        for(; group_idx < kargs.group_count; ++group_idx)
        {
            const index_t M               = kargs.M[group_idx];
            const index_t N               = kargs.N[group_idx];
            const index_t group_block_cnt = TilePartitioner::GridSize(M, N);

            while(block_linear_idx < group_block_cnt)
            {
                FlatmmKernelArgs<ScaleM, ScaleN, NumDTensor_> impl_kargs{
                    kargs.a_ptr[group_idx],
                    kargs.b_shuffle_ptr[group_idx],
                    kargs.ds_ptr,
                    kargs.c_ptr[group_idx],
                    kargs.M[group_idx],
                    kargs.N[group_idx],
                    kargs.K[group_idx],
                    kargs.stride_A[group_idx],
                    kargs.stride_B[group_idx],
                    kargs.stride_Ds,
                    kargs.stride_C[group_idx],
                    kargs.k_batch,
                    kargs.scale_m[group_idx],
                    kargs.scale_n[group_idx]};
                underlying_kernel(impl_kargs, block_linear_idx);
                block_linear_idx += total_block_cnt;
            }
            block_linear_idx -= group_block_cnt;
        }
    }
};

} // namespace ck_tile

#if __clang_major__ >= 23
#pragma clang diagnostic pop
#endif
