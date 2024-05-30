// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2023, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include <iostream>
#include <sstream>

#include "ck/utility/common_header.hpp"
#include "ck/tensor_description/tensor_descriptor.hpp"
#include "ck/tensor_description/tensor_descriptor_helper.hpp"
#include "ck/tensor_operation/gpu/device/tensor_layout.hpp"
#include "ck/tensor_operation/gpu/device/device_tall_and_skinny_gemm.hpp"
#include "ck/tensor_operation/gpu/device/gemm_specialization.hpp"
#include "ck/tensor_operation/gpu/grid/gridwise_tall_and_skinny_gemm_splitk.hpp"
#include "ck/host_utility/device_prop.hpp"
#include "ck/host_utility/kernel_launch.hpp"

namespace ck {
namespace tensor_operation {
namespace device {

template <
    typename ADataType,
    typename BDataType,
    typename CDataType,
    typename AccDataType,
    typename ALayout,
    typename BLayout,
    typename CLayout,
    typename AElementwiseOperation,
    typename BElementwiseOperation,
    typename CElementwiseOperation,
    GemmSpecialization GemmSpec,
    index_t BlockSize,
    index_t MPerBlock,
    index_t NPerBlock,
    index_t K0PerBlock,
    index_t K1,
    index_t MPerThread,
    index_t NPerThread,
    index_t KPerThread,
    typename ABlockTransferThreadSliceLengths_KBatch_K0_M0_M1_K1,
    typename ABlockTransferThreadClusterLengths_KBatch_K0_M0_M1_K1,
    typename ABlockTransferThreadClusterArrangeOrder,
    typename ABlockTransferSrcAccessOrder,
    typename ABlockTransferSrcVectorTensorLengths_KBatch_K0_M0_M1_K1,
    typename ABlockTransferSrcVectorTensorContiguousDimOrder,
    typename ABlockTransferDstVectorTensorLengths_KBatch_K0_M0_M1_K1,
    typename BThreadTransferSrcDstAccessOrder,
    index_t BThreadTransferSrcVectorDim,
    index_t BThreadTransferSrcScalarPerVector,
    typename CThreadTransferSrcDstAccessOrder,
    index_t CThreadTransferSrcDstVectorDim,
    index_t CThreadTransferDstScalarPerVector,
    enable_if_t<
        is_same_v<AElementwiseOperation, ck::tensor_operation::element_wise::PassThrough> &&
            is_same_v<BElementwiseOperation, ck::tensor_operation::element_wise::PassThrough> &&
            is_same_v<CElementwiseOperation, ck::tensor_operation::element_wise::PassThrough>,
        bool> = false>
struct deviceTsmmDl : public DeviceTsmm<ALayout,
                                        BLayout,
                                        CLayout,
                                        ADataType,
                                        BDataType,
                                        CDataType,
                                        AElementwiseOperation,
                                        BElementwiseOperation,
                                        CElementwiseOperation>

{
    static constexpr auto I0 = Number<0>{};
    static constexpr auto I1 = Number<1>{};
    static constexpr auto I2 = Number<2>{};
    static constexpr auto I3 = Number<3>{};
    static constexpr auto I4 = Number<4>{};
    static constexpr auto I5 = Number<5>{};

    // GridwiseTsmm
    using GridwiseTsmm =
        GridwiseTsmmDl_km_kn_mn<BlockSize,
                                ADataType,
                                AccDataType,
                                CDataType,
                                ALayout,
                                BLayout,
                                CLayout,
                                GemmSpec,
                                MPerBlock,
                                NPerBlock,
                                K0PerBlock,
                                K1,
                                MPerThread,
                                NPerThread,
                                KPerThread,
                                ABlockTransferThreadSliceLengths_KBatch_K0_M0_M1_K1,
                                ABlockTransferThreadClusterLengths_KBatch_K0_M0_M1_K1,
                                ABlockTransferThreadClusterArrangeOrder,
                                ABlockTransferSrcAccessOrder,
                                ABlockTransferSrcVectorTensorLengths_KBatch_K0_M0_M1_K1,
                                ABlockTransferSrcVectorTensorContiguousDimOrder,
                                ABlockTransferDstVectorTensorLengths_KBatch_K0_M0_M1_K1,
                                BThreadTransferSrcDstAccessOrder,
                                BThreadTransferSrcVectorDim,
                                BThreadTransferSrcScalarPerVector,
                                CThreadTransferSrcDstAccessOrder,
                                CThreadTransferSrcDstVectorDim,
                                CThreadTransferDstScalarPerVector>;

    using DefaultBlock2CTileMap = typename GridwiseTsmm::DefaultBlock2CTileMap;
    using Argument              = typename GridwiseTsmm::Argument;
    // Invoker
    struct Invoker : public BaseInvoker
    {

        float Run(const Argument& karg, const StreamConfig& stream_config = StreamConfig{})
        {

            const index_t grid_size = GridwiseTsmm::CalculateGridSize(karg.M, karg.N, karg.k_batch);
            // const auto b2c_map      = DefaultBlock2CTileMap{};

            const auto K0 = karg.K0;

            const bool has_main_k_block_loop = GridwiseTsmm::CalculateHasMainKBlockLoop(K0);
            const bool has_triple_tail_k_block_loop =
                GridwiseTsmm::CalculateHasTripleTailKBlockLoop(K0);

            float ave_time = 0;

            if(karg.k_batch > 1)
                hipGetErrorString(hipMemset(karg.p_c_grid, 0, karg.M * karg.N * sizeof(CDataType)));

            if(has_main_k_block_loop && has_triple_tail_k_block_loop)
            {
                if(karg.k_batch == 1)
                {

                    const auto kernel = kernel_tsmm_dl_v1r3<GridwiseTsmm,
                                                            ADataType,
                                                            CDataType,
                                                            InMemoryDataOperationEnum::Set,
                                                            true,
                                                            true,
                                                            DefaultBlock2CTileMap>; // //
                    ave_time          = launch_and_time_kernel(
                        stream_config, kernel, dim3(grid_size), dim3(BlockSize), 0, karg);
                }
                else
                {
                    const auto kernel = kernel_tsmm_dl_v1r3<GridwiseTsmm,
                                                            ADataType,
                                                            CDataType,
                                                            InMemoryDataOperationEnum::AtomicAdd,
                                                            true,
                                                            true,
                                                            DefaultBlock2CTileMap>; // //
                    ave_time          = launch_and_time_kernel(
                        stream_config, kernel, dim3(grid_size), dim3(BlockSize), 0, karg);
                }
            }
            else if(has_main_k_block_loop && !has_triple_tail_k_block_loop)
            {

                if(karg.k_batch == 1)
                {
                    const auto kernel = kernel_tsmm_dl_v1r3<GridwiseTsmm,
                                                            ADataType,
                                                            CDataType,
                                                            InMemoryDataOperationEnum::Set,
                                                            true,
                                                            false,
                                                            DefaultBlock2CTileMap>; // //
                    ave_time          = launch_and_time_kernel(
                        stream_config, kernel, dim3(grid_size), dim3(BlockSize), 0, karg);
                }
                else
                {
                    const auto kernel = kernel_tsmm_dl_v1r3<GridwiseTsmm,
                                                            ADataType,
                                                            CDataType,
                                                            InMemoryDataOperationEnum::AtomicAdd,
                                                            true,
                                                            false,
                                                            DefaultBlock2CTileMap>; // //
                    ave_time          = launch_and_time_kernel(
                        stream_config, kernel, dim3(grid_size), dim3(BlockSize), 0, karg);
                }
            }
            else if(!has_main_k_block_loop && has_triple_tail_k_block_loop)
            {
                if(karg.k_batch == 1)
                {
                    const auto kernel = kernel_tsmm_dl_v1r3<GridwiseTsmm,
                                                            ADataType,
                                                            CDataType,
                                                            InMemoryDataOperationEnum::Set,
                                                            false,
                                                            true,
                                                            DefaultBlock2CTileMap>; // //
                    ave_time          = launch_and_time_kernel(
                        stream_config, kernel, dim3(grid_size), dim3(BlockSize), 0, karg);
                }
                else
                {
                    const auto kernel = kernel_tsmm_dl_v1r3<GridwiseTsmm,
                                                            ADataType,
                                                            CDataType,
                                                            InMemoryDataOperationEnum::AtomicAdd,
                                                            false,
                                                            true,
                                                            DefaultBlock2CTileMap>; // //
                    ave_time          = launch_and_time_kernel(
                        stream_config, kernel, dim3(grid_size), dim3(BlockSize), 0, karg);
                }
            }
            else
            {
                if(karg.k_batch == 1)
                {
                    const auto kernel = kernel_tsmm_dl_v1r3<GridwiseTsmm,
                                                            ADataType,
                                                            CDataType,
                                                            InMemoryDataOperationEnum::Set,
                                                            false,
                                                            false,
                                                            DefaultBlock2CTileMap>; // //
                    ave_time          = launch_and_time_kernel(
                        stream_config, kernel, dim3(grid_size), dim3(BlockSize), 0, karg);
                }
                else
                {
                    const auto kernel = kernel_tsmm_dl_v1r3<GridwiseTsmm,
                                                            ADataType,
                                                            CDataType,
                                                            InMemoryDataOperationEnum::AtomicAdd,
                                                            false,
                                                            false,
                                                            DefaultBlock2CTileMap>; // //
                    ave_time          = launch_and_time_kernel(
                        stream_config, kernel, dim3(grid_size), dim3(BlockSize), 0, karg);
                }
            }

            return ave_time;
        }

        // polymorphic
        float Run(const BaseArgument* p_arg,
                  const StreamConfig& stream_config = StreamConfig{}) override
        {
            return Run(*dynamic_cast<const Argument*>(p_arg), stream_config);
        }
    };

    static constexpr bool IsValidCompilationParameter()
    {
        // TODO: properly implement this check
        return true;
    }
    // //
    static bool IsSupportedArgument(const Argument& arg)
    {
        if(ck::get_device_name() == "gfx906" || ck::get_device_name() == "gfx1030" ||
           ck::get_device_name() == "gfx908" || ck::get_device_name() == "gfx90a" ||
           ck::get_device_name() == "gfx1100" || ck::get_device_name() == "gfx1101" ||
           ck::get_device_name() == "gfx1102")
        {
            return GridwiseTsmm::CheckValidity(arg);
        }
        else
        {
            return false;
        }
    }
    // //
    // polymorphic
    bool IsSupportedArgument(const BaseArgument* p_arg) override
    {
        return IsSupportedArgument(*dynamic_cast<const Argument*>(p_arg));
    }

    static auto MakeArgument(const ADataType* p_a,
                             const BDataType* p_b,
                             CDataType* p_c,
                             index_t M,
                             index_t N,
                             index_t K,
                             index_t StrideA,
                             index_t StrideB,
                             index_t StrideC,
                             AElementwiseOperation,
                             BElementwiseOperation,
                             CElementwiseOperation,
                             index_t KBatch) // //
    {
        return Argument{p_a,
                        p_b,
                        p_c,
                        M,
                        N,
                        K,
                        StrideA,
                        StrideB,
                        StrideC,
                        // GridwiseTsmm::CalculateMPadded(M),
                        // GridwiseTsmm::CalculateNPadded(N),
                        // GridwiseTsmm::CalculateKPadded(K, KBatch),
                        GridwiseTsmm::CalculateK0(K, KBatch),
                        KBatch}; // //
    }

    static auto MakeInvoker() { return Invoker{}; }

    // polymorphic
    std::unique_ptr<BaseArgument> MakeArgumentPointer(const void* p_a,
                                                      const void* p_b,
                                                      void* p_c,
                                                      index_t M,
                                                      index_t N,
                                                      index_t K,
                                                      index_t StrideA,
                                                      index_t StrideB,
                                                      index_t StrideC,
                                                      AElementwiseOperation,
                                                      BElementwiseOperation,
                                                      CElementwiseOperation,
                                                      ck::index_t KBatch = 1) override // //
    {

        return std::make_unique<Argument>(static_cast<const ADataType*>(p_a),
                                          static_cast<const BDataType*>(p_b),
                                          static_cast<CDataType*>(p_c),
                                          M,
                                          N,
                                          K,
                                          StrideA,
                                          StrideB,
                                          StrideC,
                                          //   GridwiseTsmm::CalculateMPadded(M),
                                          //   GridwiseTsmm::CalculateNPadded(N),
                                          //   GridwiseTsmm::CalculateKPadded(K, KBatch),
                                          GridwiseTsmm::CalculateK0(K, KBatch),
                                          KBatch); // //
    }

    // polymorphic
    std::unique_ptr<BaseInvoker> MakeInvokerPointer() override
    {
        return std::make_unique<Invoker>(Invoker{});
    }

    // polymorphic
    std::string GetTypeString() const override
    {
        auto str = std::stringstream();

        // clang-format off
        str << "deviceTsmmDl"
            << "<"
            << BlockSize << ", "
            << MPerBlock << ", "
            << NPerBlock << ", "
            << K0PerBlock << ", "
            << K1 << ", "
            << MPerThread << ", "
            << NPerThread << ", "
            << KPerThread
            << ">";
        // clang-format on

        return str.str();
    }
};

} // namespace device
} // namespace tensor_operation
} // namespace ck

