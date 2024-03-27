// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2023, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include <iostream>
#include <sstream>

#include "ck/utility/common_header.hpp"
#include "ck/tensor_description/tensor_descriptor.hpp"
#include "ck/tensor_description/tensor_descriptor_helper.hpp"
#include "ck/tensor_operation/gpu/device/tensor_layout.hpp"
#include "ck/tensor_operation/gpu/device/device_gemm_splitk.hpp"
#include "ck/tensor_operation/gpu/device/gemm_specialization.hpp"
#include "ck/tensor_operation/gpu/grid/gridwise_gemm_xdlops_v2r4r2.hpp"
#include "ck/host_utility/device_prop.hpp"
#include "ck/host_utility/kernel_launch.hpp"

#include "ck/utility/reduction_enums.hpp"
#include "ck/tensor_operation/gpu/device/reduction_operator_mapping.hpp"
#include "ck/tensor_operation/gpu/device/impl/device_reduce_threadwise.hpp"

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
          index_t CBlockTransferScalarPerVector_NWaveNPerXDL,
          typename ComputeType        = CDataType,
          PipelineVersion PipelineVer = PipelineVersion::v1,
          LoopScheduler LoopSched     = make_default_loop_scheduler(),
          typename LDSTypeA           = ComputeType,
          typename LDSTypeB           = ComputeType,
          typename CReduceType        = CDataType>

struct DeviceGemmXdlSplitKReduce : public DeviceGemmSplitK<ALayout,
                                                           BLayout,
                                                           CLayout,
                                                           ADataType,
                                                           BDataType,
                                                           CDataType,
                                                           AElementwiseOperation,
                                                           BElementwiseOperation,
                                                           CElementwiseOperation,
                                                           ComputeType>
{
    static constexpr auto I0 = Number<0>{};
    static constexpr auto I1 = Number<1>{};
    static constexpr auto I2 = Number<2>{};
    static constexpr auto I3 = Number<3>{};

    // TODO: should be exposed as Tparams.
    static constexpr index_t NumGemmKPrefetchStage = 1;

    using ComputeTypeA = ComputeType;
    using ComputeTypeB = ComputeType;

#if 0
    static constexpr index_t CBlockTransferScalarPerVector_NWaveNPerXDL_ =
        CBlockTransferScalarPerVector_NWaveNPerXDL > 1
            ? (CBlockTransferScalarPerVector_NWaveNPerXDL / 2)
            : 1;
#else
    static constexpr index_t CBlockTransferScalarPerVector_NWaveNPerXDL_ =
        CBlockTransferScalarPerVector_NWaveNPerXDL;
#endif

    static constexpr index_t MAX_K_BATCH = 30;

    using GridwiseGemm = GridwiseGemm_bk0mk1_bk0nk1_mn_xdlops_v2r4r2<
        BlockSize,
        ADataType,
        BDataType,
        AccDataType,
        CReduceType,
        ALayout,
        BLayout,
        CLayout,
        AElementwiseOperation,
        BElementwiseOperation,
        CElementwiseOperation,
        GemmSpec,
        NumGemmKPrefetchStage,
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
        CBlockTransferScalarPerVector_NWaveNPerXDL_,
        CBlockTransferClusterLengths_MBlock_MPerBlock_NBlock_NPerBlock,
        LoopSched,
        PipelineVer,
        ComputeTypeA,
        ComputeTypeB,
        LDSTypeA,
        LDSTypeB,
        false // disable atomic
        >;

    using PassThrough = ck::tensor_operation::element_wise::PassThrough;
    using ReduceAdd   = ck::reduce::Add;

    using DeviceReduceInstance =
        DeviceReduceThreadWise<CReduceType, // InDataType,
                               AccDataType, // AccDataType,
                               CDataType,   // OutDataType,
                               4,           // Rank
                               1,           // NumReduceDim
                               ReduceAdd,
                               PassThrough,
                               PassThrough,
                               false, // PropagateNan,
                               false, // OutputIndex,
                               false,
                               false, // HaveIndexInputIfOutputIndex
                               64,    // BlockSize_,
                               CBlockTransferScalarPerVector_NWaveNPerXDL,  // MThreadSliceSize_,
                               1,                                           // KThreadSliceSize_,
                               0,                                           // InSrcVectorDim_,
                               CBlockTransferScalarPerVector_NWaveNPerXDL_, // InSrcVectorSize_,
                               CBlockTransferScalarPerVector_NWaveNPerXDL   // OutDstVectorSize_
                               >;

    struct Argument : public GridwiseGemm::Argument
    {
        Argument(const ADataType* p_a_grid_,
                 const BDataType* p_b_grid_,
                 CDataType* p_c_grid_,
                 index_t M_,
                 index_t N_,
                 index_t K_,
                 index_t StrideA_,
                 index_t StrideB_,
                 index_t StrideC_,
                 index_t MPadded_,
                 index_t NPadded_,
                 index_t KPadded_,
                 index_t K0Padded_,
                 index_t k_batch_,
                 AElementwiseOperation a_element_op_,
                 BElementwiseOperation b_element_op_,
                 CElementwiseOperation c_element_op_)
            : GridwiseGemm::Argument(p_a_grid_,
                                     p_b_grid_,
                                     nullptr, // p_c_grid_,
                                     M_,
                                     N_,
                                     K_,
                                     StrideA_,
                                     StrideB_,
                                     StrideC_,
                                     MPadded_,
                                     NPadded_,
                                     KPadded_,
                                     K0Padded_,
                                     k_batch_),
              p_c_grid{p_c_grid_},
              a_element_op(a_element_op_),
              b_element_op(b_element_op_),
              c_element_op(c_element_op_)
        {
        }

        CDataType* p_c_grid;

        AElementwiseOperation a_element_op;
        BElementwiseOperation b_element_op;
        CElementwiseOperation c_element_op;
    };

    using DefaultBlock2CTileMap = typename GridwiseGemm::DefaultBlock2CTileMap;

    // Invoker
    struct Invoker : public BaseInvoker
    {

        void Print(const Argument& karg) { karg.Print(); }

        float RunReduce(const Argument& arg, const StreamConfig& stream_config = StreamConfig{})
        {

            const index_t KBatch                  = arg.k_batch;
            const index_t NBlock                  = arg.N / NPerBlock;
            std::array<ck::index_t, 4> in_lengths = {arg.M, NBlock, KBatch, NPerBlock};
            std::array<ck::index_t, 4> in_strides = {
                NBlock * KBatch * NPerBlock, KBatch * NPerBlock, NPerBlock, 1};

            std::array<ck::index_t, 3> out_lengths = {arg.M, NBlock, NPerBlock};
            std::array<ck::index_t, 3> out_strides = {NBlock * NPerBlock, NPerBlock, 1};

            std::array<int, 1> reduce_dims{2};

            auto reduce = DeviceReduceInstance{};

            auto argument_ptr = reduce.MakeArgumentPointer(in_lengths,
                                                           in_strides,
                                                           out_lengths,
                                                           out_strides,
                                                           reduce_dims,
                                                           1.0,
                                                           0,
                                                           arg.p_workspace_,
                                                           nullptr,
                                                           arg.p_c_grid,
                                                           nullptr,
                                                           PassThrough{},
                                                           PassThrough{});

            auto invoker_ptr = reduce.MakeInvokerPointer();

            float ave_time = 0;

            if(reduce.IsSupportedArgument(argument_ptr.get()))
            {
                ave_time = invoker_ptr->Run(argument_ptr.get(), stream_config);
            }
            else
            {
                throw std::runtime_error(
                    "The runtime parameters seems not supported by the device instance, exiting!");
            }

            std::size_t num_bytes =
                arg.M * arg.N * KBatch * sizeof(CReduceType) + arg.M * arg.N * sizeof(CDataType);

            float gb_per_sec = num_bytes / 1.E6 / ave_time;

            std::cout << "Perf: " << ave_time << " ms, " << gb_per_sec << " GB/s" << std::endl;

            return ave_time;
        }

        float Run(const Argument& karg_, const StreamConfig& stream_config = StreamConfig{})
        {
            if(stream_config.log_level_ > 0)
            {
                Print(karg_);
            }

            auto karg = karg_;

            const auto kbatch = karg.k_batch;

            if(karg.p_workspace_ == nullptr ||
               karg.workspace_size_ <
                   (sizeof(CReduceType) * karg.MPadded * karg.NPadded * karg.k_batch))
            {
                throw std::runtime_error("please set workspace_ptr and workspaec_size properly!");
            }

            if(kbatch > 1)
            {
                auto p_gemm_arg_      = dynamic_cast<typename GridwiseGemm::Argument*>(&karg);
                p_gemm_arg_->p_c_grid = static_cast<CReduceType*>(karg.p_workspace_);
            }
            else
            {
                auto p_gemm_arg_      = dynamic_cast<typename GridwiseGemm::Argument*>(&karg);
                p_gemm_arg_->p_c_grid = static_cast<CReduceType*>(karg.p_c_grid);
            }

            if(!GridwiseGemm::CheckValidity(karg))
            {
                throw std::runtime_error(
                    "wrong! GridwiseGemm_bk0mk1_bk0nk1_mn_xdlops_v2r4r2 has invalid "
                    "setting");
            }

            const auto b2c_map = DefaultBlock2CTileMap{};
            index_t gdx, gdy, gdz;
            std::tie(gdx, gdy, gdz) = b2c_map.CalculateGridSize(karg.M, karg.N, karg.k_batch);
            const auto K0Padded     = karg.K0Padded;

            const bool has_main_k0_block_loop = GridwiseGemm::CalculateHasMainK0BlockLoop(K0Padded);

            float ave_time = 0;

            const auto Run = [&](const auto& kernel) {
                ave_time =
                    launch_and_time_kernel(stream_config,
                                           kernel,
                                           dim3(gdx, gdy, gdz),
                                           dim3(BlockSize),
                                           0,
                                           static_cast<typename GridwiseGemm::Argument>(karg),
                                           b2c_map,
                                           karg.a_element_op,
                                           karg.b_element_op,
                                           karg.c_element_op);
            };

            if(has_main_k0_block_loop)
            {
                const auto kernel =
                    kernel_gemm_xdlops_v2r4r2_simplified<GridwiseGemm,
                                                         true,
                                                         InMemoryDataOperationEnum::Set,
                                                         DefaultBlock2CTileMap,
                                                         AElementwiseOperation,
                                                         BElementwiseOperation,
                                                         CElementwiseOperation>;

                Run(kernel);
            }
            else
            {
                const auto kernel =
                    kernel_gemm_xdlops_v2r4r2_simplified<GridwiseGemm,
                                                         false,
                                                         InMemoryDataOperationEnum::Set,
                                                         DefaultBlock2CTileMap,
                                                         AElementwiseOperation,
                                                         BElementwiseOperation,
                                                         CElementwiseOperation>;

                Run(kernel);
            }

            if(kbatch > 1)
                ave_time += RunReduce(karg, stream_config);

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
        if(!ck::is_xdl_supported())
        {
            return false;
        }

        if(karg.p_workspace_ == nullptr)
        {
            std::cerr << "karg.p_workspace_ == nullpr" << std::endl;
            return false;
        }

        if(karg.workspace_size_ <
           (sizeof(CReduceType) * karg.MPadded * karg.NPadded * karg.k_batch))
        {
            std::cerr << "workspace_size_ = " << karg.workspace_size_ << " less than "
                      << sizeof(CReduceType) * karg.MPadded * karg.NPadded * karg.k_batch
                      << std::endl;
            return false;
        }

        if(karg.N % NPerBlock != 0)
            return false;

        if(karg.k_batch > MAX_K_BATCH)
            return false;

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
                             index_t StrideC,
                             AElementwiseOperation a_element_op,
                             BElementwiseOperation b_element_op,
                             CElementwiseOperation c_element_op,
                             index_t KBatch)
    {
        return Argument(p_a,
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
                        GridwiseGemm::CalculateKPadded(K, KBatch),
                        GridwiseGemm::CalculateK0Padded(K, KBatch),
                        KBatch,
                        a_element_op,
                        b_element_op,
                        c_element_op);
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
                                                      AElementwiseOperation a_element_op,
                                                      BElementwiseOperation b_element_op,
                                                      CElementwiseOperation c_element_op,
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
                                          GridwiseGemm::CalculateKPadded(K, KBatch),
                                          GridwiseGemm::CalculateK0Padded(K, KBatch),
                                          KBatch,
                                          a_element_op,
                                          b_element_op,
                                          c_element_op);
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

        str << GridwiseGemm::GetTypeString()
            << "_Reduce, LoopScheduler: " << LoopSchedToString[LoopSched]
            << ", PipelineVersion: " << PipelineVersionToString[PipelineVer];

        return str.str();
    }

    size_t GetWorkSpaceSize(const BaseArgument* p_arg) const override
    {
        auto arg = *dynamic_cast<const Argument*>(p_arg);

        return arg.MPadded * arg.NPadded * sizeof(CReduceType) * MAX_K_BATCH;
    }

    void SetWorkSpaceSize(BaseArgument* p_arg, const std::size_t workspace_size) const override
    {
        auto p_arg_             = dynamic_cast<Argument*>(p_arg);
        p_arg_->workspace_size_ = workspace_size;
    }

    void SetWorkSpacePointer(BaseArgument* p_arg,
                             void* p_workspace,
                             const StreamConfig& = StreamConfig{}) const override
    {
        auto p_arg_          = dynamic_cast<Argument*>(p_arg);
        p_arg_->p_workspace_ = p_workspace;
    }
};

} // namespace device
} // namespace tensor_operation
} // namespace ck
