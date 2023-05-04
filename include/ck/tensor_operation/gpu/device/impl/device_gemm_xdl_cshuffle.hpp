// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2022, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include <iostream>
#include <sstream>

#include "ck/utility/common_header.hpp"
#include "ck/tensor_description/tensor_descriptor.hpp"
#include "ck/tensor_description/tensor_descriptor_helper.hpp"
#include "ck/tensor_operation/gpu/device/tensor_layout.hpp"
#include "ck/tensor_operation/gpu/device/device_gemm.hpp"
#include "ck/tensor_operation/gpu/device/gemm_specialization.hpp"
#include "ck/tensor_operation/gpu/grid/gridwise_gemm_xdl_cshuffle_v1.hpp"
#include "ck/host_utility/device_prop.hpp"
#include "ck/host_utility/kernel_launch.hpp"

namespace ck {
namespace tensor_operation {
namespace device {

// Note: inter-wave loop scheduler is rolled out to c-shuffle version first. Becuase non c-shuffle
// version currently has compiler issues with register spill which further causes validation
// failures.
template <typename ALayout,
          typename BLayout,
          typename CLayout,
          typename ADataType,
          typename BDataType,
          typename CDataType,
          typename GemmAccDataType,
          typename CShuffleDataType,
          typename AElementwiseOperation,
          typename BElementwiseOperation,
          typename CElementwiseOperation,
          GemmSpecialization GemmSpec,
          index_t NumGemmKPrefetchStage,
          index_t BlockSize,
          index_t MPerBlock,
          index_t NPerBlock,
          index_t KPerBlock,
          index_t AK1,
          index_t BK1,
          index_t MPerXDL,
          index_t NPerXDL,
          index_t MXdlPerWave,
          index_t NXdlPerWave,
          typename ABlockTransferThreadClusterLengths_AK0_M_AK1,
          typename ABlockTransferThreadClusterArrangeOrder,
          typename ABlockTransferSrcAccessOrder,
          index_t ABlockTransferSrcVectorDim,
          index_t ABlockTransferSrcScalarPerVector,
          index_t ABlockTransferDstScalarPerVector_AK1,
          bool ABlockLdsExtraM,
          typename BBlockTransferThreadClusterLengths_BK0_N_BK1,
          typename BBlockTransferThreadClusterArrangeOrder,
          typename BBlockTransferSrcAccessOrder,
          index_t BBlockTransferSrcVectorDim,
          index_t BBlockTransferSrcScalarPerVector,
          index_t BBlockTransferDstScalarPerVector_BK1,
          bool BBlockLdsExtraN,
          index_t CShuffleMXdlPerWavePerShuffle,
          index_t CShuffleNXdlPerWavePerShuffle,
          typename CShuffleBlockTransferClusterLengths_MBlock_MPerBlock_NBlock_NPerBlock,
          index_t CShuffleBlockTransferScalarPerVector_NPerBlock,
          LoopScheduler LoopSched     = make_default_loop_scheduler(),
          PipelineVersion PipelineVer = PipelineVersion::v1>
struct DeviceGemm_Xdl_CShuffle : public DeviceGemm<ALayout,
                                                   BLayout,
                                                   CLayout,
                                                   ADataType,
                                                   BDataType,
                                                   CDataType,
                                                   AElementwiseOperation,
                                                   BElementwiseOperation,
                                                   CElementwiseOperation>
{
    using DeviceOp = DeviceGemm_Xdl_CShuffle;

    static constexpr auto I0 = Number<0>{};
    static constexpr auto I1 = Number<1>{};
    static constexpr auto I2 = Number<2>{};

    // GridwiseGemm
    using GridwiseGemm = GridwiseGemm_k0mk1_k0nk1_mn_xdl_cshuffle_v1<
        ALayout,
        BLayout,
        CLayout,
        ADataType, // TODO: distinguish A/B datatype
        GemmAccDataType,
        CShuffleDataType,
        CDataType,
        AElementwiseOperation,
        BElementwiseOperation,
        CElementwiseOperation,
        GemmSpec,
        InMemoryDataOperationEnum::Set,
        NumGemmKPrefetchStage,
        BlockSize,
        MPerBlock,
        NPerBlock,
        KPerBlock,
        AK1,
        BK1,
        MPerXDL,
        NPerXDL,
        MXdlPerWave,
        NXdlPerWave,
        ABlockTransferThreadClusterLengths_AK0_M_AK1,
        ABlockTransferThreadClusterArrangeOrder,
        ABlockTransferSrcAccessOrder,
        ABlockTransferSrcVectorDim,
        ABlockTransferSrcScalarPerVector,
        ABlockTransferDstScalarPerVector_AK1,
        false,
        ABlockLdsExtraM,
        BBlockTransferThreadClusterLengths_BK0_N_BK1,
        BBlockTransferThreadClusterArrangeOrder,
        BBlockTransferSrcAccessOrder,
        BBlockTransferSrcVectorDim,
        BBlockTransferSrcScalarPerVector,
        BBlockTransferDstScalarPerVector_BK1,
        false,
        BBlockLdsExtraN,
        CShuffleMXdlPerWavePerShuffle,
        CShuffleNXdlPerWavePerShuffle,
        CShuffleBlockTransferClusterLengths_MBlock_MPerBlock_NBlock_NPerBlock,
        CShuffleBlockTransferScalarPerVector_NPerBlock,
        LoopSched,
        PipelineVer>;

    using AGridDesc_AK0_M_AK1 =
        decltype(GridwiseGemm::MakeAGridDescriptor_AK0_M_AK1(1, 1, 1, 1, 1, 1));
    using BGridDesc_BK0_N_BK1 =
        decltype(GridwiseGemm::MakeBGridDescriptor_BK0_N_BK1(1, 1, 1, 1, 1, 1));
    using CGridDesc_M_N = decltype(GridwiseGemm::MakeCGridDescriptor_M_N(1, 1, 1, 1, 1));

    // Argument
    struct Argument : public BaseArgument
    {
        __host__ Argument(const ADataType* p_a_grid_,
                          const BDataType* p_b_grid_,
                          CDataType* p_c_grid_,
                          index_t M_,
                          index_t N_,
                          index_t K_,
                          index_t StrideA_,
                          index_t StrideB_,
                          index_t StrideC_)
            : p_a_grid{p_a_grid_},
              p_b_grid{p_b_grid_},
              p_c_grid{p_c_grid_},
              M{M_},
              N{N_},
              K{K_},
              StrideA{StrideA_},
              StrideB{StrideB_},
              StrideC{StrideC_},
              MPadded{GridwiseGemm::CalculateMPadded(M_)},
              NPadded{GridwiseGemm::CalculateNPadded(N_)},
              KPadded{GridwiseGemm::CalculateKPadded(K_)},
              AK0{GridwiseGemm::CalculateAK0(K_)},
              BK0{GridwiseGemm::CalculateBK0(K_)},
              a_grid_desc_ak0_m_ak1{
                  GridwiseGemm::MakeAGridDescriptor_AK0_M_AK1(M_,
                                                              GridwiseGemm::CalculateMPadded(M_),
                                                              K_,
                                                              GridwiseGemm::CalculateKPadded(K_),
                                                              StrideA_,
                                                              GridwiseGemm::CalculateAK0(K_))},
              b_grid_desc_bk0_n_bk1{
                  GridwiseGemm::MakeBGridDescriptor_BK0_N_BK1(K_,
                                                              GridwiseGemm::CalculateKPadded(K_),
                                                              N_,
                                                              GridwiseGemm::CalculateNPadded(N_),
                                                              StrideB_,
                                                              GridwiseGemm::CalculateBK0(K_))},
              c_grid_desc_m_n{
                  GridwiseGemm::MakeCGridDescriptor_M_N(M_,
                                                        GridwiseGemm::CalculateMPadded(M_),
                                                        N_,
                                                        GridwiseGemm::CalculateNPadded(N_),
                                                        StrideC_)}
        {
        }

        __host__ __device__ void Print() const
        {
            std::cout << "arg {"
                      << "M:" << M << ", "
                      << "N:" << N << ", "
                      << "K:" << K << ", "
                      << "SA:" << StrideA << ", "
                      << "SB:" << StrideB << ", "
                      << "SC:" << StrideC << ", "
                      << "MP:" << MPadded << ", "
                      << "NP:" << NPadded << ", "
                      << "KP:" << KPadded << ", "
                      << "AK0:" << AK0 << ", "
                      << "BK0:" << BK0 << "}" << std::endl;
        }

        __host__ __device__ Argument(const Argument&) = default;

        __host__ __device__ ~Argument() override {}

        //  private:
        const ADataType* p_a_grid;
        const BDataType* p_b_grid;
        CDataType* p_c_grid;
        index_t M;
        index_t N;
        index_t K;
        index_t StrideA;
        index_t StrideB;
        index_t StrideC;
        index_t MPadded;
        index_t NPadded;
        index_t KPadded;
        index_t AK0;
        index_t BK0;
        AGridDesc_AK0_M_AK1 a_grid_desc_ak0_m_ak1;
        BGridDesc_BK0_N_BK1 b_grid_desc_bk0_n_bk1;
        CGridDesc_M_N c_grid_desc_m_n;
    };

    // Invoker
    struct Invoker : public BaseInvoker
    {
        using Argument = DeviceOp::Argument;

        void Print(const Argument& karg) { karg.Print(); }

        float Run(const Argument& karg, const StreamConfig& stream_config = StreamConfig{})
        {
            if(stream_config.log_level_ > 0)
            {
                Print(karg);
            }

            if(!GridwiseGemm::CheckValidity(karg))
            {
                throw std::runtime_error("wrong! GridwiseGemm has invalid setting");
            }

            index_t gdx, gdy, gdz;
            std::tie(gdx, gdy, gdz) = GridwiseGemm::CalculateGridSize(karg.M, karg.N);

            const auto K = GridwiseGemm::CalculateAK0(karg.K) * AK1;

            float ave_time = 0;

            if(GridwiseGemm::CalculateHasMainKBlockLoop(K))
            {
                const auto kernel =
                    kernel_gemm_xdl_cshuffle_v1_simplified<GridwiseGemm, Argument, true>;

                ave_time = launch_and_time_kernel(
                    stream_config, kernel, dim3(gdx, gdy, gdz), dim3(BlockSize), 0, karg);
            }
            else
            {
                const auto kernel =
                    kernel_gemm_xdl_cshuffle_v1_simplified<GridwiseGemm, Argument, false>;
                ave_time = launch_and_time_kernel(
                    stream_config, kernel, dim3(gdx, gdy, gdz), dim3(BlockSize), 0, karg);
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

    static bool IsSupportedArgument(const Argument& karg)
    {
        return GridwiseGemm::CheckValidity(karg);
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
                             index_t StrideC)
    {
        return Argument{p_a, p_b, p_c, M, N, K, StrideA, StrideB, StrideC};
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
                                                      CElementwiseOperation) override
    {
        return std::make_unique<Argument>(static_cast<const ADataType*>(p_a),
                                          static_cast<const BDataType*>(p_b),
                                          static_cast<CDataType*>(p_c),
                                          M,
                                          N,
                                          K,
                                          StrideA,
                                          StrideB,
                                          StrideC);
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

        std::map<LoopScheduler, std::string> LoopSchedToString{
            {LoopScheduler::Default, "Default"}, {LoopScheduler::Interwave, "Interwave"}};

        std::map<PipelineVersion, std::string> PipelineVersionToString{{PipelineVersion::v1, "v1"},
                                                                       {PipelineVersion::v2, "v2"}};

        // clang-format off
        str << "DeviceGemm_Xdl_CShuffle"
            << "<"
            << BlockSize << ", "
            << MPerBlock << ", "
            << NPerBlock << ", "
            << KPerBlock << ", "
            << AK1 << ", "
            << BK1 << ", "
            << MPerXDL << ", "
            << NPerXDL << ", "
            << MXdlPerWave << ", "
            << NXdlPerWave << ", "
            << ABlockTransferSrcScalarPerVector << ", "
            << BBlockTransferSrcScalarPerVector << ", "
            << CShuffleMXdlPerWavePerShuffle << ", "
            << CShuffleNXdlPerWavePerShuffle
            << ">"
            << " LoopScheduler: "
            << LoopSchedToString[LoopSched] << ", "
            << "PipelineVersion: "
            << PipelineVersionToString[PipelineVer];;
        // clang-format on

        return str.str();
    }
};

} // namespace device
} // namespace tensor_operation
} // namespace ck
