// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <sstream>
#include <initializer_list>
#include <cstdlib>

#include "ck/ck.hpp"
#include "ck/tensor_operation/gpu/device/impl/device_batched_gemm_gemm_wmma_cshuffle_v3_common.hpp"
#include "ck/tensor_operation/gpu/device/device_batched_gemm_multiple_d_gemm_multiple_d.hpp"
#include "ck/tensor_operation/gpu/device/gemm_specialization.hpp"
#include "ck/tensor_operation/gpu/grid/gridwise_batched_gemm_gemm_wmma_cshuffle_v3.hpp"

namespace ck {
namespace tensor_operation {
namespace device {

// Computes:
//         Acc = Acc_Op(A_Op(A) * B0_Op(B0), D0_0, D0_1, ...)
//         E = CDE1_Op(Acc_Op(Acc0) * B1_Op(B1), D1_0, D1_1, ...)
//
// Dimensions:
//         A        = MK
//         B0       = KL
//         Acc/D0s  = ML
//         B1       = LN
//         E/D1s    = MN
template <typename ALayout,
          typename B0layout,
          typename D0sLayout,
          typename B1Layout,
          typename D1sLayout,
          typename E1Layout,
          typename ADataType,
          typename B0DataType,
          typename D0sDataType,
          typename B1DataType,
          typename D1sDataType,
          typename E1DataType,
          typename AccDataType,
          typename CShuffleDataType,
          typename AElementwiseOperation,
          typename B0ElementwiseOperation,
          typename AccElementwiseOperation,
          typename B1ElementwiseOperation,
          typename CDE1ElementwiseOperation,
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
          ck::index_t CDE0BlockTransferSrcScalarPerVector,
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
struct DeviceBatchedGemmMultipleDGemmMultipleD_Wmma_CShuffleV3
    : public DeviceBatchedGemmMultipleDGemmMultipleD<ALayout,
                                                     B0layout,
                                                     D0sLayout,
                                                     B1Layout,
                                                     D1sLayout,
                                                     E1Layout,
                                                     ADataType,
                                                     B0DataType,
                                                     D0sDataType,
                                                     B1DataType,
                                                     D1sDataType,
                                                     E1DataType,
                                                     AElementwiseOperation,
                                                     B0ElementwiseOperation,
                                                     AccElementwiseOperation,
                                                     B1ElementwiseOperation,
                                                     CDE1ElementwiseOperation>
{
    using DeviceOp = DeviceBatchedGemmMultipleDGemmMultipleD_Wmma_CShuffleV3;

    static constexpr index_t NumD0Tensor = D0sDataType::Size();
    static constexpr index_t NumD1Tensor = D1sDataType::Size();

    // To match XDL implementation NPerWmma (A.k.a Gemm1 NPerWmma) is set equal
    // to LPerWmma (A.k.a Gemm0 NPerWmma).
    static constexpr index_t NPerWmma = LPerWmma;

    using DeviceGemmGemmCommonBase =
        DeviceGemmGemm_Wmma_CShuffleV3_Common<DeviceOp,
                                              GemmSpec,
                                              ALayout,
                                              B0layout,
                                              D0sLayout,
                                              B1Layout,
                                              D1sLayout,
                                              E1Layout,
                                              BlockSize,
                                              MPerBlock,
                                              LPerBlock,
                                              KPerBlock,
                                              NPerBlock,
                                              ADataType,
                                              B0DataType,
                                              B1DataType,
                                              AccDataType,
                                              E1DataType,
                                              D0sDataType,
                                              D1sDataType,
                                              AElementwiseOperation,
                                              B0ElementwiseOperation,
                                              AccElementwiseOperation,
                                              B1ElementwiseOperation,
                                              CDE1ElementwiseOperation,
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
                                              CDE0BlockTransferSrcScalarPerVector,
                                              CShuffleBlockTransferScalarPerVector_NPerBlock,
                                              true>; // IsMultiD

    // GridwiseOp
    using GridwiseOp = GridwiseBatchedGemmGemm_wmma_cshuffle_v3<
        // DataType Family
        ADataType,
        B0DataType,
        D0sDataType,
        AccDataType, // Acc0DataType
        B1DataType,
        D1sDataType,
        AccDataType, // Acc1DataType
        CShuffleDataType,
        E1DataType,
        // ElementwiseOp Family
        AElementwiseOperation,
        B0ElementwiseOperation,
        AccElementwiseOperation,
        B1ElementwiseOperation,
        CDE1ElementwiseOperation,
        InMemoryDataOperationEnum::Set,
        // InMemory Data Descriptor
        typename DeviceGemmGemmCommonBase::AGridDesc,
        typename DeviceGemmGemmCommonBase::B0GridDesc,
        typename DeviceGemmGemmCommonBase::D0sGridDesc,
        typename DeviceGemmGemmCommonBase::B1GridDesc,
        typename DeviceGemmGemmCommonBase::D1sGridDesc,
        typename DeviceGemmGemmCommonBase::E1GridDesc,
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
        CDE0BlockTransferSrcScalarPerVector,
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
        D0sLayout,
        B1Layout,
        D1sLayout,
        E1Layout,
        BlockSize,
        MPerBlock,
        LPerBlock,
        KPerBlock,
        NPerBlock,
        ADataType,
        B0DataType,
        B1DataType,
        AccDataType,
        E1DataType,
        D0sDataType,
        D1sDataType,
        AElementwiseOperation,
        B0ElementwiseOperation,
        AccElementwiseOperation,
        B1ElementwiseOperation,
        CDE1ElementwiseOperation,
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
        CDE0BlockTransferSrcScalarPerVector,
        CShuffleBlockTransferScalarPerVector_NPerBlock,
        true>; // IsMultiD
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

    static auto MakeArgument(const ADataType* p_a0,
                             const B0DataType* p_b0,
                             std::array<const void*, NumD0Tensor> p_d0s,
                             const B1DataType* p_b1,
                             std::array<const void*, NumD1Tensor> p_d1s,
                             E1DataType* p_e1,
                             index_t MRaw,
                             index_t NRaw,
                             index_t KRaw,
                             index_t Gemm1NRaw,
                             index_t Batch,
                             index_t StrideA0,
                             index_t StrideB0,
                             std::array<index_t, NumD0Tensor> StrideD0s,
                             index_t StrideB1,
                             std::array<index_t, NumD1Tensor> StrideD1s,
                             index_t StrideE1,
                             index_t BatchStrideA0,
                             index_t BatchStrideB0,
                             std::array<index_t, NumD0Tensor> BatchStrideD0s,
                             index_t BatchStrideB1,
                             std::array<index_t, NumD1Tensor> BatchStrideD1s,
                             index_t BatchStrideE1,
                             AElementwiseOperation a0_element_op,
                             B0ElementwiseOperation b0_element_op,
                             AccElementwiseOperation cde0_element_op,
                             B1ElementwiseOperation b1_element_op,
                             CDE1ElementwiseOperation cde1_element_op)
    {
        return Argument{p_a0,          p_b0,
                        p_d0s,         p_b1,
                        p_d1s,         p_e1,
                        MRaw,          NRaw,
                        KRaw,          Gemm1NRaw,
                        Batch,         StrideA0,
                        StrideB0,      StrideD0s,
                        StrideB1,      StrideD1s,
                        StrideE1,      BatchStrideA0,
                        BatchStrideB0, BatchStrideD0s,
                        BatchStrideB1, BatchStrideD1s,
                        BatchStrideE1, a0_element_op,
                        b0_element_op, cde0_element_op,
                        b1_element_op, cde1_element_op};
    }

    // polymorphic
    std::unique_ptr<BaseArgument>
    MakeArgumentPointer(const void* p_a,
                        const void* p_b0,
                        std::array<const void*, NumD0Tensor> p_d0s,
                        const void* p_b1,
                        std::array<const void*, NumD1Tensor> p_d1s,
                        void* p_c,
                        ck::index_t M,
                        ck::index_t N,
                        ck::index_t K,
                        ck::index_t O,
                        ck::index_t Batch,
                        ck::index_t StrideA,
                        ck::index_t StrideB0,
                        std::array<index_t, NumD0Tensor> StrideD0s,
                        ck::index_t StrideB1,
                        std::array<index_t, NumD1Tensor> StrideD1s,
                        ck::index_t StrideE1,
                        ck::index_t BatchStrideA,
                        ck::index_t BatchStrideB0,
                        std::array<index_t, NumD0Tensor> BatchStrideD0s,
                        ck::index_t BatchStrideB1,
                        std::array<index_t, NumD1Tensor> BatchStrideD1s,
                        ck::index_t BatchStrideE1,
                        AElementwiseOperation a_element_op,
                        B0ElementwiseOperation b0_element_op,
                        AccElementwiseOperation acc_element_op,
                        B1ElementwiseOperation b1_element_op,
                        CDE1ElementwiseOperation c_element_op) override
    {
        return std::make_unique<Argument>(static_cast<const ADataType*>(p_a),
                                          static_cast<const B0DataType*>(p_b0),
                                          p_d0s,
                                          static_cast<const B1DataType*>(p_b1),
                                          p_d1s,
                                          static_cast<E1DataType*>(p_c),
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
                                          StrideE1,
                                          BatchStrideA,
                                          BatchStrideB0,
                                          BatchStrideD0s,
                                          BatchStrideB1,
                                          BatchStrideD1s,
                                          BatchStrideE1,
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

    template <typename DataTypes>
    std::string DataTypeTupleToString() const
    {
        const auto string_types = generate_tuple(
            [&](auto i) {
                using ElementType = remove_cvref_t<tuple_element_t<i.value, DataTypes>>;
                return DataTypeToString<ElementType>();
            },
            Number<DataTypes::Size()>{});

        return TupleReduce<0, DataTypes::Size()>(
            [&](std::string s, std::string a) { return a + ", " + s; }, string_types);
    };

    template <typename Layouts>
    std::string LayoutTupleToString() const
    {
        const auto string_layouts = generate_tuple(
            [&](auto i) {
                using ElementLayout = remove_cvref_t<tuple_element_t<i.value, Layouts>>;
                return std::string(1, ElementLayout::name[0]);
            },
            Number<Layouts::Size()>{});

        return TupleReduce<0, Layouts::Size()>([&](std::string s, std::string a) { return a + s; },
                                               string_layouts);
    };

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
        str << "DeviceBatchedGemmMultipleDGemmMultipleD_Wmma_CShuffleV3"
            << "<A/B0/B1/E: "
            << ALayout::name[0]
            << B0layout::name[0]
            << B1Layout::name[0]
            << E1Layout::name[0]  << ", "
            << "D0s: " << LayoutTupleToString<D0sLayout>() << " "
            << "D1s: " << LayoutTupleToString<D1sLayout>()
            << ", "
            << "A " << DataTypeToString<ADataType>() << ", "
            << "B0 " << DataTypeToString<B0DataType>() << ", "
            << "D0s (" << DataTypeTupleToString<D0sDataType>() << "), "
            << "B1 " << DataTypeToString<B1DataType>() << ", "
            << "D1s (" << DataTypeTupleToString<D1sDataType>() << "), "
            << "E1 " << DataTypeToString<E1DataType>() << ", "
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
            << getGemmSpecializationString(GemmSpec) << ", "
            << "BlkGemmPipelineScheduler: "
            << BlkGemmPipelineSchedulerToString[BlkGemmPipeSched] << ", "
            << "BlkGemmPipelineVersion: "
            << BlkGemmPipelineVersionToString[BlkGemmPipelineVer] << ", "
            << "BlkGemmPipelinePrefetchStages: "
            << GridwiseOp::BlockwiseGemmPipe::PrefetchStages 
            << ">";
        // clang-format on

        return str.str();
    }
};

} // namespace device
} // namespace tensor_operation
} // namespace ck
