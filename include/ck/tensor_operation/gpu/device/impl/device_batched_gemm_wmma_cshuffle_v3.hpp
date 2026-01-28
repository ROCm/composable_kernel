// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <iostream>
#include <sstream>

#include "ck/utility/common_header.hpp"
#include "ck/tensor_description/tensor_descriptor.hpp"
#include "ck/tensor_description/tensor_descriptor_helper.hpp"
#include "ck/tensor_operation/gpu/device/tensor_layout.hpp"
#include "ck/tensor_operation/gpu/device/device_batched_gemm.hpp"
#include "ck/tensor_operation/gpu/device/gemm_specialization.hpp"
#include "ck/tensor_operation/gpu/grid/gridwise_gemm_wmma_cshuffle_v3.hpp"
#include "ck/tensor_operation/gpu/device/impl/device_batched_gemm_wmma_cshuffle_v3_common.hpp"

namespace ck {
namespace tensor_operation {
namespace device {

/// @brief \"Universal\" Batched GEMM operation without SplitK support.
///
/// @par Overview
///         This GEMM operation implements the following mathematical equation:
///         C{G,M,N} = C_op(A_op(A{G,M,K}) * B_op(B{G,K,N}))
///         Where A, B are input tensors and C is the output tensor. The A/B/C_op are
///         elementwise operations applied to the A, B, and C tensors, respectively.
///         The \"universal\" gemm comes with multiple pipelines optimized for different usage
///         scenarios. That's why it's called \"universal\". It's universal through its design
///         and versatilty.
///
/// @note   This Kernel implementation currently does not support the SplitK algorithm.
///
/// @tparam ALayout     A tensor data layout.
/// @tparam BLayout     B tensor data layout.
/// @tparam CLayout     C tensor data layout.
/// @tparam ADataType   A tensor data type.
/// @tparam BDataType   B tensor data type.
/// @tparam CDataType   C tensor data type.
/// @tparam AccDataType The accumulation data type related to the hardware
///                         matrix-multiplication instruction.
/// @tparam CShuffleDataType The data type used to store matrix-multiplication results into
///                          LDS memory during \"CShuffle\" data layout optimization.
/// @tparam AElementwiseOperation Elementwise operation applied to the A input tensor elements.
/// @tparam BElementwiseOperation Elementwise operation applied to the B input tensor elements.
/// @tparam CElementwiseOperation Elementwise operation applied to the C output tensor
///                               (after GEMM).
/// @tparam GemmSpec    Determines used "padding" version.
/// @tparam BlockSize   The number of threads within workgroup.
/// @tparam MPerBlock   The input/output data tile size in the M dimension.
/// @tparam NPerBlock   The input/output data tile size in the N dimension.
/// @tparam KPerBlock   The input data tile size in the K dimension.
/// @tparam AK1         The vector load size from global memory for A tensor.
/// @tparam BK1         The vector load size from global memory for B tensor.
/// @tparam MPerWmma    M size of Wave Matrix Multiply Accumulate (WMMA) instruction.
/// @tparam NPerWmma    N size of Wave Matrix Multiply Accumulate (WMMA) instruction.
/// @tparam MRepeat     The number of iterations in the M dimension over output tile per wavefront.
/// @tparam NRepeat     The number of iterations in the N dimension over output tile per wavefront.
/// @tparam ABlockTransferThreadClusterLengths_AK0_M_AK1 Spatial thread distribution over the input
///                                                      data. Can be interpreted as the answer
///                                                      to the question, "How many threads can be
///                                                      arranged on each input data axis?"
/// @tparam ABlockTransferThreadClusterArrangeOrder The order of thread spatial distribution over
///                                                 the input tensor dimension. Can be interpreted
///                                                 as the answer to the question: "In which
///                                                 order to spread threads through tensor axes?".
/// @tparam ABlockTransferSrcAccessOrder The order of accessing input tensor axes. Can be
///                                      interpreted as the answer to the question "Which dimension
///                                      to read first? And which next?" etc.
/// @tparam ABlockTransferSrcVectorDim   The index of axis on which we could do vectorized memory
///                                      access - the one with contiguous memory.
/// @tparam ABlockTransferSrcScalarPerVector The size of vector access instruction - the number of
///                                          elements accessed per thread per instruction.
/// @tparam ABlockTransferDstScalarPerVector_AK1 The size of vectorized store into LDS memory.
/// @tparam ABlockLdsExtraM                      Whether to use padding for LDS or not. With
///                                              universal GEMM there's no need for padding.
/// @tparam BBlockTransferThreadClusterLengths_BK0_N_BK1 Spatial thread distribution over the input
///                                                      data. Can be interpreted as the answer
///                                                      to the question: "How many threads to
///                                                      arrange on each input data axis?"
/// @tparam BBlockTransferThreadClusterArrangeOrder The order of thread spatial distribution over
///                                                 the input tensor dimension. Can be interpreted
///                                                 as the answer to the question: "In which
///                                                 order to spread threads through tensor axes?".
/// @tparam BBlockTransferSrcAccessOrder he order of accessing input tensor axes. Can be
///                                      interpreted as the answer to the question "Which dimension
///                                      to read first? And which next?" etc.
/// @tparam BBlockTransferSrcVectorDim  The index of axis on which we could do vectorized memory
///                                      access - the one with contiguous memory.
/// @tparam BBlockTransferSrcScalarPerVector The size of vector access instruction - the number of
///                                          elements accessed per thread per instruction.
/// @tparam BBlockTransferDstScalarPerVector_BK1 The size of vectorized store into LDS memory.
/// @tparam BBlockLdsExtraN                      Whether to use padding for LDS or not. With
///                                              universal GEMM there's no need for padding.
/// @tparam CShuffleMRepeatPerShuffle   The number of matrix-multiplication instructions
///                                         results to process per wave per iteration of CShuffle
///                                         in M dimension.
/// @tparam CShuffleNRepeatPerShuffle   The number of matrix-multiplication instructions
///                                         results to process per wave per iteration of CShuffle
///                                         in N dimension.
/// @tparam CShuffleBlockTransferClusterLengths_MBlock_MPerBlock_NBlock_NPerBlock The spatial
///                                         thread distribution used for storing data into output
///                                         tensor across output data layout dimensions.
/// @tparam CShuffleBlockTransferScalarPerVector_NPerBlock The size of vectorized memory access.
///                                         Used when storing data to output tensor.
/// @tparam BlkGemmPipeSched    The version of blockwise-gemm pipeline scheduler (interwave or
///                             intrawave).
/// @tparam BlkGemmPipelineVer  The version of blockwise-gemm pipeline.
/// @tparam ComputeTypeA    Data type used for A input of hardware matrix-multiplication
///                         instructions.
/// @tparam ComputeTypeB    Data type used for B input of hardware matrix-multiplication
///                         instructions.
/// @tparam PermuteA            Whether the A input tensor has gridwise-gemm friendly data layout
///                             in global memory. Currently not supported!
/// @tparam PermuteB            Whether the B input tensor has gridwise-gemm friendly data layout
///                             in global memory (pre-shuffled). Currently not supported!
template <typename ALayout,
          typename BLayout,
          typename CLayout,
          typename ADataType,
          typename BDataType,
          typename CDataType,
          typename AccDataType,
          typename CShuffleDataType,
          typename AElementwiseOperation,
          typename BElementwiseOperation,
          typename CElementwiseOperation,
          GemmSpecialization GemmSpec,
          index_t BlockSize,
          index_t MPerBlock,
          index_t NPerBlock,
          index_t KPerBlock,
          index_t AK1,
          index_t BK1,
          index_t MPerWmma,
          index_t NPerWmma,
          index_t MRepeat,
          index_t NRepeat,
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
          index_t CShuffleMRepeatPerShuffle,
          index_t CShuffleNRepeatPerShuffle,
          typename CShuffleBlockTransferClusterLengths_MBlock_MPerBlock_NBlock_NPerBlock,
          index_t CShuffleBlockTransferScalarPerVector_NPerBlock,
          BlockGemmPipelineScheduler BlkGemmPipeSched = BlockGemmPipelineScheduler::Intrawave,
          BlockGemmPipelineVersion BlkGemmPipelineVer = BlockGemmPipelineVersion::v1,
          typename ComputeTypeA                       = CDataType,
          typename ComputeTypeB                       = ComputeTypeA,
          bool PermuteA                               = false,
          bool PermuteB                               = false>
struct DeviceBatchedGemm_Wmma_CShuffleV3 : public DeviceBatchedGemm<ALayout,
                                                                    BLayout,
                                                                    CLayout,
                                                                    ADataType,
                                                                    BDataType,
                                                                    CDataType,
                                                                    AElementwiseOperation,
                                                                    BElementwiseOperation,
                                                                    CElementwiseOperation>
{
    // We are inheriting from DeviceBatchedGemm and this base class does not support permuteA and
    // permuteB arguments so for now we are not including this functionality.
    static_assert(PermuteA == false,
                  "Permute A functionality not supported by DeviceBatchedGemm operations.\n");
    static_assert(PermuteB == false,
                  "Permute B functionality not supported by DeviceBatchedGemm operations.\n");

    // GridwiseGemm
    using GridwiseGemm = GridwiseGemm_wmma_cshuffle_v3<
        ALayout,
        BLayout,
        Tuple<>, // DsLayout
        CLayout,
        Tuple<ADataType>,
        Tuple<BDataType>,
        AccDataType,
        CShuffleDataType,
        Tuple<>, // DsDataType
        CDataType,
        AElementwiseOperation,
        BElementwiseOperation,
        CElementwiseOperation,
        GemmSpec,
        BlockSize,
        MPerBlock,
        NPerBlock,
        KPerBlock,
        AK1,
        BK1,
        MPerWmma,
        NPerWmma,
        MRepeat,
        NRepeat,
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
        CShuffleMRepeatPerShuffle,
        CShuffleNRepeatPerShuffle,
        CShuffleBlockTransferClusterLengths_MBlock_MPerBlock_NBlock_NPerBlock,
        Sequence<CShuffleBlockTransferScalarPerVector_NPerBlock>,
        BlkGemmPipeSched,
        BlkGemmPipelineVer,
        ComputeTypeA,
        ComputeTypeB,
        false,  // PermuteA not supported by DeviceBatchedGemm base class.
        false>; // PermuteB not supported by DeviceBatchedGemm base class.

    using DeviceGemmCommon = DeviceBatchedGemm_Wmma_CShuffleV3_Common<
        GridwiseGemm,
        Tuple<ADataType>,
        Tuple<BDataType>,
        CDataType,
        MPerBlock,
        NPerBlock,
        KPerBlock,
        BlockSize,
        AK1,
        BK1,
        GemmSpec,
        Sequence<CShuffleBlockTransferScalarPerVector_NPerBlock>,
        BlkGemmPipeSched,
        BlkGemmPipelineVer,
        ComputeTypeA,
        ComputeTypeB,
        false, // IsBScaled
        AElementwiseOperation,
        BElementwiseOperation,
        CElementwiseOperation>;

    // Argument
    using Argument = typename DeviceGemmCommon::Argument;

    // Invoker
    using Invoker = typename DeviceGemmCommon::Invoker;

    // polymorphic
    bool IsSupportedArgument(const BaseArgument* p_arg) override
    {
        return DeviceGemmCommon::IsSupportedArgument(*dynamic_cast<const Argument*>(p_arg));
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
                             index_t BatchStrideA,
                             index_t BatchStrideB,
                             index_t BatchStrideC,
                             index_t Batch,
                             AElementwiseOperation,
                             BElementwiseOperation,
                             CElementwiseOperation)
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
                        BatchStrideA,
                        BatchStrideB,
                        BatchStrideC,
                        Batch,
                        1, /* KBatch */
                        AElementwiseOperation{},
                        BElementwiseOperation{},
                        CElementwiseOperation{}};
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
                                                      index_t BatchStrideA,
                                                      index_t BatchStrideB,
                                                      index_t BatchStrideC,
                                                      index_t Batch,
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
                                          StrideC,
                                          BatchStrideA,
                                          BatchStrideB,
                                          BatchStrideC,
                                          Batch,
                                          1,
                                          AElementwiseOperation{},
                                          BElementwiseOperation{},
                                          CElementwiseOperation{}); // KBatch
    }

    // polymorphic
    std::unique_ptr<BaseInvoker> MakeInvokerPointer() override
    {
        return std::make_unique<Invoker>(Invoker{});
    }

    // polymorphic
    std::string GetTypeString() const override
    {
        return DeviceGemmCommon::template GetTypeString<MPerWmma,
                                                        NPerWmma,
                                                        MRepeat,
                                                        NRepeat,
                                                        ABlockTransferSrcScalarPerVector,
                                                        BBlockTransferSrcScalarPerVector,
                                                        ALayout,
                                                        BLayout,
                                                        CLayout>();
    }
    REGISTER_EXTRA_PRINTING_METHODS
};

} // namespace device
} // namespace tensor_operation
} // namespace ck
