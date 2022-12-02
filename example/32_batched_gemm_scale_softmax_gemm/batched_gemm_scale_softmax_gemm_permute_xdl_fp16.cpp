// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2022, Advanced Micro Devices, Inc. All rights reserved.

/*
Gemm + Softmax + Gemm fused operation. Computes C_g_m_o = Softmax(A_g_m_k * B0_g_k_n) * B1_g_n_o
                                                                  |-----------------|
                                                                          Gemm0
                                                          |-------------------------------------|
                                                                          Gemm1
*/
#pragma clang diagnostic ignored "-Wunused-variable"
#include <iostream>
#include <numeric>
#include <initializer_list>
#include <cstdlib>

#include "ck/ck.hpp"
#include "ck/tensor_operation/gpu/device/gemm_specialization.hpp"
#include "ck/tensor_operation/gpu/device/tensor_specialization.hpp"
#include "ck/tensor_operation/gpu/device/impl/device_batched_gemm_softmax_gemm_permute_xdl_cshuffle.hpp"
#include "ck/tensor_operation/gpu/element/element_wise_operation.hpp"

#include "ck/library/utility/check_err.hpp"
#include "ck/library/utility/device_memory.hpp"
#include "ck/library/utility/host_tensor.hpp"
#include "ck/library/utility/host_tensor_generator.hpp"
#include "ck/library/utility/literals.hpp"
#include "ck/library/reference_tensor_operation/cpu/reference_batched_gemm.hpp"
#include "ck/library/reference_tensor_operation/cpu/reference_softmax.hpp"

template <ck::index_t... Is>
using S = ck::Sequence<Is...>;

using F16 = ck::half_t;
using F32 = float;

using PassThrough = ck::tensor_operation::element_wise::PassThrough;

using ADataType        = F16;
using B0DataType       = F16;
using B1DataType       = F16;
using AccDataType      = F32;
using CShuffleDataType = F32;
using CDataType        = F16;
using Acc0BiasDataType = ck::Tuple<>;
using Acc1BiasDataType = ck::Tuple<>;

static constexpr ck::index_t NumDimG = 2;
static constexpr ck::index_t NumDimM = 1;
static constexpr ck::index_t NumDimN = 1;
static constexpr ck::index_t NumDimK = 1;
static constexpr ck::index_t NumDimO = 1;

using AElementOp    = PassThrough;
using B0ElementOp   = PassThrough;
using Acc0ElementOp = ck::tensor_operation::element_wise::Scale;
using B1ElementOp   = PassThrough;
using CElementOp    = PassThrough;

static constexpr auto GemmSpec = ck::tensor_operation::device::GemmSpecialization::Default;
static constexpr auto MaskingSpec =
    ck::tensor_operation::device::MaskingSpecialization::MaskDisabled;

static constexpr auto TensorSpecA  = ck::tensor_operation::device::TensorSpecialization::Default;
static constexpr auto TensorSpecB0 = ck::tensor_operation::device::TensorSpecialization::Default;
static constexpr auto TensorSpecB1 = ck::tensor_operation::device::TensorSpecialization::Default;
static constexpr auto TensorSpecC  = ck::tensor_operation::device::TensorSpecialization::Default;

using DeviceGemmInstance =
    ck::tensor_operation::device::DeviceBatchedGemmSoftmaxGemmPermute_Xdl_CShuffle<
        NumDimG,
        NumDimM,
        NumDimN,
        NumDimK,
        NumDimO,
        ck::half_t,
        ck::half_t,
        ck::half_t,
        ck::half_t,
        ck::Tuple<>,
        ck::Tuple<>,
        float,
        float, // CShuffleDType,
        ck::tensor_operation::element_wise::PassThrough,
        ck::tensor_operation::element_wise::PassThrough,
        ck::tensor_operation::element_wise::Scale,
        ck::tensor_operation::element_wise::PassThrough,
        ck::tensor_operation::element_wise::PassThrough,
        GemmSpec,
        ck::tensor_operation::device::TensorSpecialization::Default,
        ck::tensor_operation::device::TensorSpecialization::Default,
        ck::tensor_operation::device::TensorSpecialization::Default,
        ck::tensor_operation::device::TensorSpecialization::Default,
        1,
        256,                       // block_size
        64,                        // m_per_block
        256,                       // n_per_block
        32,                        // k_per_block
        64,                        // Gemm1NPerBlock
        32,                        // Gemm1KPerBlock
        8,                         // ak1
        8,                         // bk1
        2,                         // b1k1
        16,                        // m_per_xdl
        16,                        // n_per_xdl
        1,                         // m_xdl_per_wave
        16,                        // n_xdl_per_wave
        4,                         // Gemm1NXdlPerWave
        ck::Sequence<4, 64, 1>,    // thread_cluster_length
        ck::Sequence<1, 0, 2>,     // thread_cluster_arrange_order
        ck::Sequence<1, 0, 2>,     // src_access_order
        2,                         // src_vector_dim
        8,                         // src_scalar_per_vector
        8,                         // dst_scalar_per_vector
        1,                         // add_extra_dim
        ck::Sequence<4, 64, 1>,    // thread_cluster_length
        ck::Sequence<1, 0, 2>,     // thread_cluster_arrange_order
        ck::Sequence<1, 0, 2>,     // src_access_order
        2,                         // src_vector_dim
        8,                         // src_scalar_per_vector
        8,                         // dst_scalar_per_vector
        1,                         // add_extra_dim
        ck::Sequence<16, 16, 1>,   // thread_cluster_length
        ck::Sequence<0, 2, 1>,     // thread_cluster_arrange_order
        ck::Sequence<0, 2, 1>,     // src_access_order
        1,                         // src_vector_dim
        4,                         // src_scalar_per_vector
        2,                         // dst_scalar_per_vector
        0,                         // add_extra_dim
        1,                         // m_xdl_per_wave
        4,                         // n_xdl_per_wave
        ck::Sequence<1, 32, 1, 8>, // m_n_block_wave_per_xdl
        8,                         // scalar_per_vector
        ck::tensor_operation::device::MaskingSpecialization::MaskDisabled>; // causal_mask

// Ref Gemm0: fp16 in, fp32 out
using ReferenceGemm0Instance = ck::tensor_operation::host::ReferenceBatchedGemm<ADataType,
                                                                                B0DataType,
                                                                                AccDataType,
                                                                                AccDataType,
                                                                                AElementOp,
                                                                                B0ElementOp,
                                                                                Acc0ElementOp>;

// Ref Softmax: fp32 in, fp16 out
using ReferenceSoftmaxInstance =
    ck::tensor_operation::host::ReferenceSoftmax<AccDataType, ADataType, AccDataType>;

// Ref Gemm1: fp16 in, fp16 out
using ReferenceGemm1Instance = ck::tensor_operation::host::ReferenceBatchedGemm<ADataType,
                                                                                B1DataType,
                                                                                CDataType,
                                                                                AccDataType,
                                                                                AElementOp,
                                                                                B1ElementOp,
                                                                                CElementOp>;

#include "run_batched_gemm_scale_softmax_gemm_permute.inc"

int main(int argc, char* argv[]) { return run(argc, argv); }
