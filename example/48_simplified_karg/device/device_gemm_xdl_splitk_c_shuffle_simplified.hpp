// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2022, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include <iostream>
#include <sstream>

#include "ck/utility/common_header.hpp"
#include "ck/tensor_description/tensor_descriptor.hpp"
#include "ck/tensor_description/tensor_descriptor_helper.hpp"
#include "ck/tensor_operation/gpu/device/tensor_layout.hpp"
#include "ck/tensor_operation/gpu/device/device_gemm_splitk.hpp"
#include "ck/tensor_operation/gpu/device/gemm_specialization.hpp"
//#include "ck/tensor_operation/gpu/grid/gridwise_gemm_xdlops_v2r4r2.hpp"
#include "ck/host_utility/device_prop.hpp"
#include "ck/host_utility/kernel_launch.hpp"
#include "grid/gridwise_gemm_xdlops_v2r4r2_simplified.hpp"

namespace ck {
namespace tensor_operation {
namespace device {

template <typename ADataType,
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
          ck::index_t BlockSize,
          ck::index_t MPerBlock,
          ck::index_t NPerBlock,
          ck::index_t K0PerBlock,
          ck::index_t K1,
          ck::index_t MPerXDL,
          ck::index_t NPerXDL,
          ck::index_t MXdlPerWave,
          ck::index_t NXdlPerWave,
          typename ABlockTransferThreadClusterLengths_K0_M_K1,
          typename ABlockTransferThreadClusterArrangeOrder,
          typename ABlockTransferSrcAccessOrder,
          ck::index_t ABlockTransferSrcVectorDim,
          ck::index_t ABlockTransferSrcScalarPerVector,
          ck::index_t ABlockTransferDstScalarPerVector_K1,
          bool ABlockLdsAddExtraM,
          typename BBlockTransferThreadClusterLengths_K0_N_K1,
          typename BBlockTransferThreadClusterArrangeOrder,
          typename BBlockTransferSrcAccessOrder,
          ck::index_t BBlockTransferSrcVectorDim,
          ck::index_t BBlockTransferSrcScalarPerVector,
          ck::index_t BBlockTransferDstScalarPerVector_K1,
          bool BBlockLdsAddExtraN,
          index_t CShuffleMRepeatPerShuffle,
          index_t CShuffleNRepeatPerShuffle,
          typename CBlockTransferClusterLengths_MBlock_MPerBlock_NBlock_NPerBlock,
          index_t CBlockTransferScalarPerVector_NWaveNPerXDL>
struct DeviceGemmXdlSplitKCShuffleSimplified : public DeviceGemmSplitK<ALayout,
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

    // template<bool USE_ATOMIC = true>
    // struct DataOperationSelector{
    //     static constexpr InMemoryDataOperationEnum value = InMemoryDataOperationEnum::AtomicAdd;
    // };
    // template<>
    // struct DataOperationSelector<false>{
    //     static constexpr InMemoryDataOperationEnum value = InMemoryDataOperationEnum::Set;
    // };

    using GridwiseGemm = GridwiseGemm_bk0mk1_bk0nk1_mn_xdlops_v2r4r2_simplified<
        BlockSize,
        ADataType, // TODO: distinguish A/B datatype
        AccDataType,
        CDataType,
        ALayout,
        BLayout,
        CLayout,
        // AGridDesc_K0_M_K1,
        // BGridDesc_K0_N_K1,
        // CGridDesc_M_N,
        AElementwiseOperation,
        BElementwiseOperation,
        CElementwiseOperation,
        GemmSpec,
        MPerBlock,
        NPerBlock,
        K0PerBlock,
        MPerXDL,
        NPerXDL,
        K1,
        MXdlPerWave,
        NXdlPerWave,
        ABlockTransferThreadClusterLengths_K0_M_K1,
        ABlockTransferThreadClusterArrangeOrder,
        ABlockTransferSrcAccessOrder,
        ABlockTransferSrcVectorDim,
        ABlockTransferSrcScalarPerVector,
        ABlockTransferDstScalarPerVector_K1,
        false, // AThreadTransferSrcResetCoordinateAfterRun,
        ABlockLdsAddExtraM,
        BBlockTransferThreadClusterLengths_K0_N_K1,
        BBlockTransferThreadClusterArrangeOrder,
        BBlockTransferSrcAccessOrder,
        BBlockTransferSrcVectorDim,
        BBlockTransferSrcScalarPerVector,
        BBlockTransferDstScalarPerVector_K1,
        false, // BThreadTransferSrcResetCoordinateAfterRun,
        BBlockLdsAddExtraN,
        CShuffleMRepeatPerShuffle,
        CShuffleNRepeatPerShuffle,
        CBlockTransferScalarPerVector_NWaveNPerXDL,
        CBlockTransferClusterLengths_MBlock_MPerBlock_NBlock_NPerBlock>;

    using Argument = typename GridwiseGemm::Argument;

    // Invoker
    struct Invoker : public BaseInvoker
    {

        void Print(const Argument& arg) { arg.Print(); }

        float Run(const Argument& arg, const StreamConfig& stream_config = StreamConfig{})
        {
            if(stream_config.log_level_ > 0)
            {
                Print(arg);
            }

            const auto kbatch = arg.k_batch;

            if(!GridwiseGemm::CheckValidity(arg))
            {
                throw std::runtime_error(
                    "wrong! GridwiseGemm_bk0mk1_bk0nk1_mn_xdlops_v2r4r2_simplified has invalid "
                    "setting");
            }

            const index_t grid_size = math::integer_divide_ceil(arg.M, MPerBlock) *
                                      math::integer_divide_ceil(arg.N, NPerBlock) * arg.k_batch;

            const auto K0 = arg.K0;

            const bool has_main_k0_block_loop = GridwiseGemm::CalculateHasMainK0BlockLoop(K0);

            float ave_time = 0;

            const auto Run = [&](const auto& kernel) {
                if(kbatch > 1)
                    hipGetErrorString(
                        hipMemset(arg.p_c_grid, 0, arg.M * arg.N * sizeof(CDataType)));

                ave_time = launch_and_time_kernel(
                    stream_config, kernel, dim3(grid_size), dim3(BlockSize), 0, arg);
            };

            if(has_main_k0_block_loop)
            {
                if(kbatch == 1)
                {
                    const auto kernel =
                        kernel_gemm_xdlops_v2r4r2_simplified<GridwiseGemm,
                                                             ADataType, // TODO: distiguish A/B
                                                                        // datatype
                                                             CDataType,
                                                             AElementwiseOperation,
                                                             BElementwiseOperation,
                                                             CElementwiseOperation,
                                                             true,
                                                             InMemoryDataOperationEnum::Set>;

                    Run(kernel);
                }
                else
                {
                    const auto kernel =
                        kernel_gemm_xdlops_v2r4r2_simplified<GridwiseGemm,
                                                             ADataType, // TODO: distiguish A/B
                                                                        // datatype
                                                             CDataType,
                                                             AElementwiseOperation,
                                                             BElementwiseOperation,
                                                             CElementwiseOperation,
                                                             true,
                                                             InMemoryDataOperationEnum::AtomicAdd>;

                    Run(kernel);
                }
            }
            else
            {
                if(kbatch == 1)
                {
                    const auto kernel =
                        kernel_gemm_xdlops_v2r4r2_simplified<GridwiseGemm,
                                                             ADataType, // TODO: distiguish A/B
                                                                        // datatype
                                                             CDataType,
                                                             AElementwiseOperation,
                                                             BElementwiseOperation,
                                                             CElementwiseOperation,
                                                             false,
                                                             InMemoryDataOperationEnum::Set>;

                    Run(kernel);
                }
                else
                {
                    const auto kernel =
                        kernel_gemm_xdlops_v2r4r2_simplified<GridwiseGemm,
                                                             ADataType, // TODO: distiguish A/B
                                                                        // datatype
                                                             CDataType,
                                                             AElementwiseOperation,
                                                             BElementwiseOperation,
                                                             CElementwiseOperation,
                                                             false,
                                                             InMemoryDataOperationEnum::AtomicAdd>;

                    Run(kernel);
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

    static bool IsSupportedArgument(const Argument& arg)
    {
        return GridwiseGemm::CheckValidity(arg);
    }

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
                             // AElementwiseOperation a_element_op,
                             // BElementwiseOperation b_element_op,
                             // CElementwiseOperation c_element_op,
                             index_t KBatch)
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
                        GridwiseGemm::CalculateMPadded(M),
                        GridwiseGemm::CalculateNPadded(N),
                        GridwiseGemm::CalculateKPadded(K),
                        GridwiseGemm::CalculateK0(K, KBatch),
                        KBatch};
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
                                                      ck::index_t KBatch = 1) override
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
                                          GridwiseGemm::CalculateMPadded(M),
                                          GridwiseGemm::CalculateNPadded(N),
                                          GridwiseGemm::CalculateKPadded(K),
                                          GridwiseGemm::CalculateK0(K, KBatch),
                                          KBatch);
    }

    // polymorphic
    std::unique_ptr<BaseInvoker> MakeInvokerPointer() override
    {
        return std::make_unique<Invoker>(Invoker{});
    }

    template <typename Layout>
    struct LStr
    {
        static std::string Get() { return ""; }
    };

    template <>
    struct LStr<ck::tensor_layout::gemm::RowMajor>
    {
        static std::string Get() { return "R"; }
    };

    template <>
    struct LStr<ck::tensor_layout::gemm::ColumnMajor>
    {
        static std::string Get() { return "C"; }
    };

    // polymorphic
    std::string GetTypeString() const override
    {
        auto str = std::stringstream();

        // clang-format off
        str << "DeviceGemmXdlSplitKCShuffleSimplified_"
            << getGemmSpecializationString(GemmSpec) << "_"
            << LStr<ALayout>::Get() << LStr<BLayout>::Get() << LStr<CLayout>::Get() << "_"
            << "B" << BlockSize << "_"
            << "Vec" << ABlockTransferSrcScalarPerVector << "x"
            << BBlockTransferSrcScalarPerVector << "x"
            << CBlockTransferScalarPerVector_NWaveNPerXDL << "_"
            << MPerBlock << "x"
            << NPerBlock << "x"
            << K0PerBlock << "x"
            << K1 ;
        // clang-format on

        return str.str();
    }
};

} // namespace device
} // namespace tensor_operation
} // namespace ck
