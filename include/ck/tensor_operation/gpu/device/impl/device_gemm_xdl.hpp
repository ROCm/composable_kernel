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
#include "ck/tensor_operation/gpu/grid/gridwise_gemm_xdlops_v2r3.hpp"
#include "ck/host_utility/device_prop.hpp"
#include "ck/host_utility/kernel_launch.hpp"

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
          ck::index_t CThreadTransferSrcDstVectorDim,
          ck::index_t CThreadTransferDstScalarPerVector,
          ck::index_t NumPrefetch         = 1,
          ck::LoopScheduler LoopSched     = make_default_loop_scheduler(),
          ck::PipelineVersion PipelineVer = ck::PipelineVersion::v1>
struct DeviceGemmXdl : public DeviceGemm<ALayout,
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

    static constexpr auto K1Number = Number<K1>{};

    // GridwiseGemm
    using GridwiseGemm = GridwiseGemm_k0mk1_k0nk1_mn_xdlops_v2r3<
        BlockSize,
        ADataType, // TODO: distinguish A/B datatype
        AccDataType,
        CDataType,
        InMemoryDataOperationEnum::Set,
        ALayout,
        BLayout,
        CLayout,
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
        Sequence<0, 2, 4, 5, 6, 1, 3, 7>, // CThreadTransferSrcDstAccessOrder,
        CThreadTransferSrcDstVectorDim,
        CThreadTransferDstScalarPerVector,
        NumPrefetch,
        LoopSched,
        PipelineVer>;

    using AGridDesc_K0_M_K1 = decltype(GridwiseGemm::MakeAGridDescriptor_K0_M_K1(1, 1, 1, 1));
    using BGridDesc_K0_N_K1 = decltype(GridwiseGemm::MakeBGridDescriptor_K0_N_K1(1, 1, 1, 1));
    using CGridDesc_M_N     = decltype(GridwiseGemm::MakeCGridDescriptor_M_N(1, 1, 1, 1, 1));

    // Argument
    struct Argument : public BaseArgument
    {
        Argument(const ADataType* p_a_grid,
                 const BDataType* p_b_grid,
                 CDataType* p_c_grid,
                 index_t M_,
                 index_t N_,
                 index_t K_,
                 index_t StrideA_,
                 index_t StrideB_,
                 index_t StrideC_)
            : p_a_grid_{p_a_grid},
              p_b_grid_{p_b_grid},
              p_c_grid_{p_c_grid},
              M{M_},
              N{N_},
              K{K_},
              StrideA{StrideA_},
              StrideB{StrideB_},
              StrideC{StrideC_},
              MPadded{GridwiseGemm::CalculateMPadded(M_)},
              NPadded{GridwiseGemm::CalculateNPadded(N_)},
              a_grid_desc_k0_m_k1{},
              c_grid_desc_m_n{}
        {
            // Print();

            a_grid_desc_k0_m_k1 = GridwiseGemm::MakeAGridDescriptor_K0_M_K1(M, MPadded, K, StrideA);
            c_grid_desc_m_n =
                GridwiseGemm::MakeCGridDescriptor_M_N(M, MPadded, N, NPadded, StrideC);
        }

        __host__ __device__ void Print() const
        {
            printf("M = %d, N = %d, K = %d, "
                   "SA = %d, SB = %d, SC = %d, "
                   "MP = %d, NP = %d\n",
                   M,
                   N,
                   K,
                   StrideA,
                   StrideB,
                   StrideC,
                   MPadded,
                   NPadded);
        }

        //  private:
        const ADataType* p_a_grid_;
        const BDataType* p_b_grid_;
        CDataType* p_c_grid_;
        index_t M;
        index_t N;
        index_t K;
        index_t StrideA;
        index_t StrideB;
        index_t StrideC;
        index_t MPadded;
        index_t NPadded;
        AGridDesc_K0_M_K1 a_grid_desc_k0_m_k1;
        CGridDesc_M_N c_grid_desc_m_n;
    };

    // Invoker
    struct Invoker : public BaseInvoker
    {
        using Argument = DeviceGemmXdl::Argument;

        float Run(const Argument& arg, const StreamConfig& stream_config = StreamConfig{})
        {
#if DEBUG_LOG
            {
                // std::cout << "arg.a_grid_desc_k0_m_k1_{" <<
                // arg.a_grid_desc_k0_m_k1_.GetLength(I0)
                //           << ", " << arg.a_grid_desc_k0_m_k1_.GetLength(I1) << ", "
                //           << arg.a_grid_desc_k0_m_k1_.GetLength(I2) << "}" << std::endl;

                // std::cout << "arg.b_grid_desc_k0_n_k1_{" <<
                // arg.b_grid_desc_k0_n_k1_.GetLength(I0)
                //           << ", " << arg.b_grid_desc_k0_n_k1_.GetLength(I1) << ", "
                //           << arg.b_grid_desc_k0_n_k1_.GetLength(I2) << "}" << std::endl;

                // std::cout << "arg.c_grid_desc_m_n_{ " << arg.c_grid_desc_m_n_.GetLength(I0) << ",
                // "
                //           << arg.c_grid_desc_m_n_.GetLength(I1) << "}" << std::endl;
            }
#endif

            if(!GridwiseGemm::CheckValidity(arg))
            {
                throw std::runtime_error(
                    "wrong! GridwiseGemm_k0mk1_k0nk1_mn_xdlops_v2r3 has invalid setting");
            }

            index_t gdx, gdy, gdz;
            std::tie(gdx, gdy, gdz) = GridwiseGemm::CalculateGridSize(arg.M, arg.N);

            float ave_time = 0;

            if(GridwiseGemm::CalculateHasMainKBlockLoop(arg.K))
            {
                const auto kernel = kernel_gemm_xdlops_v2r3<GridwiseGemm, Argument, true>;

                ave_time = launch_and_time_kernel(stream_config,
                                                  kernel,
                                                  dim3(gdx, gdy, gdz),
                                                  dim3(BlockSize),
                                                  0,
                                                  arg.p_a_grid_,
                                                  arg.p_b_grid_,
                                                  arg.p_c_grid_,
                                                  arg);
            }
            else
            {
                const auto kernel = kernel_gemm_xdlops_v2r3<GridwiseGemm, Argument, false>;

                ave_time = launch_and_time_kernel(stream_config,
                                                  kernel,
                                                  dim3(gdx, gdy, gdz),
                                                  dim3(BlockSize),
                                                  0,
                                                  arg.p_a_grid_,
                                                  arg.p_b_grid_,
                                                  arg.p_c_grid_,
                                                  arg);
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
        if(ck::get_device_name() == "gfx908")
        {
            if constexpr(!(is_same_v<AccDataType, float> || is_same_v<AccDataType, float> ||
                           is_same_v<AccDataType, int32_t>))
            {
                return false;
            }
        }
        else if(ck::get_device_name() == "gfx90a" || ck::get_device_name() == "gfx940")
        {
            if constexpr(!(is_same_v<AccDataType, float> || is_same_v<AccDataType, float> ||
                           is_same_v<AccDataType, int32_t> || is_same_v<AccDataType, double>))
            {
                return false;
            }
        }
        else
        {
            return false;
        }

        if(arg.K % K1 != 0)
        {
            return false;
        }

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
                             AElementwiseOperation,
                             BElementwiseOperation,
                             CElementwiseOperation)
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
        str << "DeviceGemmXdl"
            << "<"
            << BlockSize << ", "
            << MPerBlock << ", "
            << NPerBlock << ", "
            << K0PerBlock << ", "
            << K1 << ", "
            << MPerXDL << ", "
            << NPerXDL << ", "
            << MXdlPerWave << ", "
            << NXdlPerWave << ", "
            << ABlockTransferSrcScalarPerVector << ", "
            << ABlockTransferDstScalarPerVector_K1 << ", "
            << BBlockTransferSrcScalarPerVector << ", "
            << BBlockTransferDstScalarPerVector_K1
            << ">"
            << " NumPrefetch: "
            << NumPrefetch << ", "
            << "LoopScheduler: "
            << LoopSchedToString[LoopSched] << ", "
            << "PipelineVersion: "
            << PipelineVersionToString[PipelineVer];
        // clang-format on

        return str.str();
    }
};

} // namespace device
} // namespace tensor_operation
} // namespace ck
