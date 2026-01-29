// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <sstream>
#include <initializer_list>
#include <cstdlib>

#include "ck/ck.hpp"
#include "ck/tensor_operation/gpu/device/impl/device_batched_gemm_gemm_wmma_cshuffle_v3_common.hpp"
#include "ck/tensor_operation/gpu/device/device_batched_gemm_gemm.hpp"
#include "ck/tensor_operation/gpu/device/gemm_specialization.hpp"
#include "ck/tensor_operation/gpu/grid/gridwise_batched_gemm_gemm_wmma_cshuffle_v3.hpp"
#include "ck/utility/tuple.hpp"

namespace ck {
namespace tensor_operation {
namespace device {

// Computes C = A  * B0 * B1
//         MN = MK * KL * LN
//              ^^^^^^ (Acc0)
//              ^^^^^^^^^^^ (Acc1)
template <typename ALayout,
          typename B0layout,
          typename B1Layout,
          typename CLayout,
          typename ADataType,
          typename B0DataType,
          typename B1DataType,
          typename CDataType,
          typename AccDataType,
          typename CShuffleDataType,
          typename AElementwiseOperation,
          typename B0ElementwiseOperation,
          typename AccElementwiseOperation,
          typename B1ElementwiseOperation,
          typename CElementwiseOperation,
          GemmSpecialization GemmSpec,
          ck::index_t BlockSize,
          ck::index_t MPerBlock,
          ck::index_t LPerBlock,     // Gemm0NPerBlock
          ck::index_t KPerBlock,     // Gemm0KPerBlock
          ck::index_t NPerBlock,     // Gemm1NPerBlock
          ck::index_t LTilePerBlock, // Gemm1KPerBlock
          ck::index_t AK1,
          ck::index_t BK1,
          ck::index_t L1,       // B1K1
          ck::index_t MPerWmma, // Gemm0/1 MPerWmma
          ck::index_t LPerWmma, // Gemm0/1 NPerWmma
          ck::index_t MRepeat,  // Gemm0/1 MWmmaPerWave or Mrepeat
          ck::index_t LRepeat,  // Gemm0 NWmmaPerWave or Nrepeat
          ck::index_t NRepeat,  // Gemm1 NWmmaPerWave or Nrepeat
          typename ABlockTransferThreadClusterLengths_K0_M_K1,
          typename ABlockTransferThreadClusterArrangeOrder,
          typename ABlockTransferSrcAccessOrder,
          ck::index_t ABlockTransferSrcVectorDim,
          ck::index_t ABlockTransferSrcScalarPerVector,
          ck::index_t ABlockTransferDstScalarPerVector_K1,
          bool ABlockLdsAddExtraM,
          typename B0BlockTransferThreadClusterLengths_K0_L_K1,
          typename B0BlockTransferThreadClusterArrangeOrder,
          typename B0BlockTransferSrcAccessOrder,
          ck::index_t B0BlockTransferSrcVectorDim,
          ck::index_t B0BlockTransferSrcScalarPerVector,
          ck::index_t B0BlockTransferDstScalarPerVector_K1,
          bool B0BlockLdsAddExtraL,
          typename B1BlockTransferThreadClusterLengths_L0_N_L1,
          typename B1BlockTransferThreadClusterArrangeOrder,
          typename B1BlockTransferSrcAccessOrder,
          ck::index_t B1BlockTransferSrcVectorDim,
          ck::index_t B1BlockTransferSrcScalarPerVector,
          ck::index_t B1BlockTransferDstScalarPerVector_L1,
          bool B1BlockLdsAddExtraN,
          index_t CShuffleMRepeatPerShuffle,
          index_t CShuffleNRepeatPerShuffle,
          typename CShuffleBlockTransferClusterLengths_MBlock_MPerBlock_NBlock_NPerBlock,
          index_t CShuffleBlockTransferScalarPerVector_NPerBlock,
          BlockGemmPipelineScheduler BlkGemmPipeSched = BlockGemmPipelineScheduler::Intrawave,
          BlockGemmPipelineVersion BlkGemmPipelineVer = BlockGemmPipelineVersion::v1>
struct DeviceBatchedGemmGemm_Wmma_CShuffleV3 : public DeviceBatchedGemmGemm<ALayout,
                                                                            B0layout,
                                                                            B1Layout,
                                                                            CLayout,
                                                                            ADataType,
                                                                            B0DataType,
                                                                            B1DataType,
                                                                            CDataType,
                                                                            AElementwiseOperation,
                                                                            B0ElementwiseOperation,
                                                                            AccElementwiseOperation,
                                                                            B1ElementwiseOperation,
                                                                            CElementwiseOperation>
{
    using DeviceOp = DeviceBatchedGemmGemm_Wmma_CShuffleV3;

    static constexpr auto I0 = Number<0>{};

    // To match XDL implementation NPerWmma (A.k.a Gemm1 NPerWmma) is set equal
    // to LPerWmma (A.k.a Gemm0 NPerWmma).
    static constexpr index_t NPerWmma = LPerWmma;

    using DeviceGemmGemmCommonBase =
        DeviceGemmGemm_Wmma_CShuffleV3_Common<DeviceOp,
                                              GemmSpec,
                                              ALayout,
                                              B0layout,
                                              Tuple<>, // D0sLayout
                                              B1Layout,
                                              Tuple<>, // D1sLayout
                                              CLayout,
                                              BlockSize,
                                              MPerBlock,
                                              LPerBlock,
                                              KPerBlock,
                                              NPerBlock,
                                              ADataType,
                                              B0DataType,
                                              B1DataType,
                                              AccDataType,
                                              CDataType,
                                              Tuple<>, // D0sDataType
                                              Tuple<>, // D1sDataType
                                              AElementwiseOperation,
                                              B0ElementwiseOperation,
                                              AccElementwiseOperation,
                                              B1ElementwiseOperation,
                                              CElementwiseOperation,
                                              AK1,
                                              BK1,
                                              L1,
                                              MPerWmma,
                                              LPerWmma,
                                              BlkGemmPipelineVer,
                                              ABlockTransferSrcVectorDim,
                                              ABlockTransferSrcScalarPerVector,
                                              B0BlockTransferSrcVectorDim,
                                              B0BlockTransferSrcScalarPerVector,
                                              B1BlockTransferSrcVectorDim,
                                              B1BlockTransferSrcScalarPerVector,
                                              ck::index_t{}, // CDE0BlockTransferSrcScalarPerVector
                                              CShuffleBlockTransferScalarPerVector_NPerBlock,
                                              false>; // IsMultiD

    // GridwiseOp
    using GridwiseOp = GridwiseBatchedGemmGemm_wmma_cshuffle_v3<
        // DataType Family
        ADataType,
        B0DataType,
        Tuple<>,     // Ds0DataType
        AccDataType, // Acc0DataType
        B1DataType,
        Tuple<>,     // Ds1DataType
        AccDataType, // Acc1DataType
        CShuffleDataType,
        CDataType,
        // ElementwiseOp Family
        AElementwiseOperation,
        B0ElementwiseOperation,
        AccElementwiseOperation,
        B1ElementwiseOperation,
        CElementwiseOperation,
        InMemoryDataOperationEnum::Set,
        // InMemory Data Descriptor
        typename DeviceGemmGemmCommonBase::AGridDesc,
        typename DeviceGemmGemmCommonBase::B0GridDesc,
        Tuple<>, // Ds0GridDesc
        typename DeviceGemmGemmCommonBase::B1GridDesc,
        Tuple<>, // Ds1GridDesc
        typename DeviceGemmGemmCommonBase::CGridDesc_M_N,
        // Tiling Family
        MPerBlock,
        LPerBlock,
        KPerBlock,
        AK1,
        BK1,
        NPerBlock,
        LTilePerBlock,
        L1,
        MPerWmma,
        LPerWmma,
        NPerWmma,
        MRepeat,
        LRepeat,
        NRepeat,
        // ThreadCluster Family
        BlockSize,
        ABlockTransferThreadClusterLengths_K0_M_K1,
        ABlockTransferThreadClusterArrangeOrder,
        ABlockTransferSrcAccessOrder,
        ABlockTransferSrcVectorDim,
        ABlockTransferSrcScalarPerVector,
        ABlockTransferDstScalarPerVector_K1,
        true,
        ABlockLdsAddExtraM,
        B0BlockTransferThreadClusterLengths_K0_L_K1,
        B0BlockTransferThreadClusterArrangeOrder,
        B0BlockTransferSrcAccessOrder,
        B0BlockTransferSrcVectorDim,
        B0BlockTransferSrcScalarPerVector,
        B0BlockTransferDstScalarPerVector_K1,
        true,
        B0BlockLdsAddExtraL,
        1, // CDE0BlockTransferSrcScalarPerVector
        B1BlockTransferThreadClusterLengths_L0_N_L1,
        B1BlockTransferThreadClusterArrangeOrder,
        B1BlockTransferSrcAccessOrder,
        B1BlockTransferSrcVectorDim,
        B1BlockTransferSrcScalarPerVector,
        B1BlockTransferDstScalarPerVector_L1,
        false,
        B1BlockLdsAddExtraN,
        CShuffleMRepeatPerShuffle,
        CShuffleNRepeatPerShuffle,
        CShuffleBlockTransferClusterLengths_MBlock_MPerBlock_NBlock_NPerBlock,
        CShuffleBlockTransferScalarPerVector_NPerBlock,
        DeviceGemmGemmCommonBase::GridDescriptorCreator::Transform::matrix_padder.PadN,
        BlkGemmPipeSched,
        BlkGemmPipelineVer>;

    using DeviceGemmGemmCommon = DeviceGemmGemm_Wmma_CShuffleV3_Common_Invoker_Arg<
        DeviceOp,
        GemmSpec,
        ALayout,
        B0layout,
        Tuple<>, // D0sLayout
        B1Layout,
        Tuple<>, // D1sLayout
        CLayout,
        BlockSize,
        MPerBlock,
        LPerBlock,
        KPerBlock,
        NPerBlock,
        ADataType,
        B0DataType,
        B1DataType,
        AccDataType,
        CDataType,
        Tuple<>, // D0sDataType,
        Tuple<>, // D1sDataType,
        AElementwiseOperation,
        B0ElementwiseOperation,
        AccElementwiseOperation,
        B1ElementwiseOperation,
        CElementwiseOperation,
        AK1,
        BK1,
        L1,
        MPerWmma,
        LPerWmma,
        BlkGemmPipelineVer,
        ABlockTransferSrcVectorDim,
        ABlockTransferSrcScalarPerVector,
        B0BlockTransferSrcVectorDim,
        B0BlockTransferSrcScalarPerVector,
        B1BlockTransferSrcVectorDim,
        B1BlockTransferSrcScalarPerVector,
        ck::index_t{}, // CDE0BlockTransferSrcScalarPerVector
        CShuffleBlockTransferScalarPerVector_NPerBlock,
        false>; // IsMultiD
    // Invoker
    using Invoker = typename DeviceGemmGemmCommon::Invoker;

    // Argument
    using Argument = typename DeviceGemmGemmCommon::Argument;

    static bool IsSupportedArgument(const Argument& arg)
    {
        return DeviceGemmGemmCommon::IsSupportedArgument(arg);
    }
    // polymorphic
    bool IsSupportedArgument(const BaseArgument* p_arg) override
    {
        return DeviceGemmGemmCommon::IsSupportedArgument(*dynamic_cast<const Argument*>(p_arg));
    }

    // polymorphic
    std::unique_ptr<BaseArgument> MakeArgumentPointer(const void* p_a,
                                                      const void* p_b0,
                                                      const void* p_b1,
                                                      void* p_c,
                                                      ck::index_t M,
                                                      ck::index_t N,
                                                      ck::index_t K,
                                                      ck::index_t O,
                                                      ck::index_t Batch,
                                                      ck::index_t StrideA,
                                                      ck::index_t StrideB0,
                                                      ck::index_t StrideB1,
                                                      ck::index_t StrideC,
                                                      ck::index_t BatchStrideA,
                                                      ck::index_t BatchStrideB0,
                                                      ck::index_t BatchStrideB1,
                                                      ck::index_t BatchStrideC,
                                                      AElementwiseOperation a_element_op,
                                                      B0ElementwiseOperation b0_element_op,
                                                      AccElementwiseOperation acc_element_op,
                                                      B1ElementwiseOperation b1_element_op,
                                                      CElementwiseOperation c_element_op) override
    {

        std::array<const void*, DeviceGemmGemmCommonBase::NumD0Tensor> p_d0_grid{};
        std::array<const void*, DeviceGemmGemmCommonBase::NumD1Tensor> p_d1_grid{};
        std::array<index_t, DeviceGemmGemmCommonBase::NumD0Tensor> StrideD0s{}, BatchStrideD0s{};
        std::array<index_t, DeviceGemmGemmCommonBase::NumD1Tensor> StrideD1s, BatchStrideD1s{};
        return std::make_unique<Argument>(static_cast<const ADataType*>(p_a),
                                          static_cast<const B0DataType*>(p_b0),
                                          p_d0_grid,
                                          static_cast<const B1DataType*>(p_b1),
                                          p_d1_grid,
                                          static_cast<CDataType*>(p_c),
                                          M,
                                          N,
                                          K,
                                          O,
                                          Batch,
                                          StrideA,
                                          StrideB0,
                                          StrideD0s,
                                          StrideB1,
                                          StrideD1s,
                                          StrideC,
                                          BatchStrideA,
                                          BatchStrideB0,
                                          BatchStrideD0s,
                                          BatchStrideB1,
                                          BatchStrideD1s,
                                          BatchStrideC,
                                          a_element_op,
                                          b0_element_op,
                                          acc_element_op,
                                          b1_element_op,
                                          c_element_op);
    }

    static auto MakeInvoker() { return Invoker{}; }

    // polymorphic
    std::unique_ptr<BaseInvoker> MakeInvokerPointer() override
    {
        return std::make_unique<Invoker>(Invoker{});
    }

    template <typename T>
    static constexpr const char* DataTypeToString()
    {
        if constexpr(std::is_same_v<T, float>)
        {
            return "fp32";
        }
        else if constexpr(std::is_same_v<T, ck::half_t>)
        {
            return "fp16";
        }
        else if constexpr(std::is_same_v<T, ck::bhalf_t>)
        {
            return "bf16";
        }
        else if constexpr(std::is_same_v<T, ck::f8_t>)
        {
            return "fp8";
        }
        else if constexpr(std::is_same_v<T, ck::bf8_t>)
        {
            return "bf8";
        }
        else if constexpr(std::is_same_v<T, int32_t>)
        {
            return "int32";
        }
        else if constexpr(std::is_same_v<T, int8_t>)
        {
            return "int8";
        }
        else if constexpr(std::is_same_v<T, ck::int4_t>)
        {
            return "int4";
        }
        else
        {
            return "unknown";
        }
    }

    // polymorphic
    std::string GetTypeString() const override
    {
        auto str = std::stringstream();

        std::map<BlockGemmPipelineScheduler, std::string> BlkGemmPipelineSchedulerToString{
            {BlockGemmPipelineScheduler::Intrawave, "Intrawave"},
            {BlockGemmPipelineScheduler::Interwave, "Interwave"}};

        std::map<BlockGemmPipelineVersion, std::string> BlkGemmPipelineVersionToString{
            {BlockGemmPipelineVersion::v1, "v1"},
            {BlockGemmPipelineVersion::v2, "v2"},
            {BlockGemmPipelineVersion::v3, "v3"},
            {BlockGemmPipelineVersion::v4, "v4"},
            {BlockGemmPipelineVersion::v5, "v5"}};

        // clang-format off
        str << "DeviceBatchedGemmGemm_Wmma_CShuffleV3"
            << "<"
            << ALayout::name[0]
            << B0layout::name[0]
            << B1Layout::name[0]
            << CLayout::name[0] << ", "
            << "A " << DataTypeToString<ADataType>() << ", "
            << "B0 " << DataTypeToString<B0DataType>() << ", "
            << "B1 " << DataTypeToString<B1DataType>() << ", "
            << "C " << DataTypeToString<CDataType>() << ", "
            << "Acc " << DataTypeToString<AccDataType>() << ", "
            << "Cshuf " << DataTypeToString<CShuffleDataType>() << ", "
            << BlockSize << ", "
            << MPerBlock << ", "
            << LPerBlock << ", "
            << KPerBlock << ", "
            << AK1 << ", "
            << BK1 << ", "
            << MPerBlock << ", "
            << NPerBlock << ", "
            << LTilePerBlock << ", "
            << L1 << ", "
            << getGemmSpecializationString(GemmSpec)
            << ">"
            << "BlkGemmPipelineScheduler: "
            << BlkGemmPipelineSchedulerToString[BlkGemmPipeSched] << ", "
            << "BlkGemmPipelineVersion: "
            << BlkGemmPipelineVersionToString[BlkGemmPipelineVer] << ", "
            << "BlkGemmPipelinePrefetchStages: "
            << GridwiseOp::BlockwiseGemmPipe::PrefetchStages;
        // clang-format on

        return str.str();
    }
};

} // namespace device
} // namespace tensor_operation
} // namespace ck
