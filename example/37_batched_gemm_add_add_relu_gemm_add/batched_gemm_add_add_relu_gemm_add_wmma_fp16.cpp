// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

/*
Computes C_m_o = Relu(A0[m, k] * B0[n, k] + D00[m, n] + D01[mn]) * B1[n, o] + D1[m, o]
*/

#include <iostream>
#include <numeric>
#include <initializer_list>
#include <cstdlib>

#include "ck/ck.hpp"
#include "ck/tensor_operation/gpu/device/gemm_specialization.hpp"
#include "ck/tensor_operation/gpu/device/impl/device_batched_gemm_multiple_d_gemm_multiple_d_wmma_cshuffle_v3.hpp"
#include "ck/tensor_operation/gpu/element/binary_element_wise_operation.hpp"

#include "ck/library/utility/check_err.hpp"
#include "ck/library/utility/device_memory.hpp"
#include "ck/library/utility/host_tensor.hpp"
#include "ck/library/utility/host_tensor_generator.hpp"
#include "ck/library/utility/literals.hpp"
#include "ck/library/reference_tensor_operation/cpu/reference_batched_gemm.hpp"

#include "element_ops.h"

using ::ck::DeviceMem;
using ::ck::HostTensorDescriptor;
using ::ck::Tensor;

template <ck::index_t... Is>
using S = ck::Sequence<Is...>;

using F16 = ck::half_t;
using F32 = float;

using Row = ck::tensor_layout::gemm::RowMajor;
using Col = ck::tensor_layout::gemm::ColumnMajor;

using PassThrough = ck::tensor_operation::element_wise::PassThrough;

using A0DataType       = F16;
using B0DataType       = F16;
using D00DataType      = F16;
using D01DataType      = F16;
using B1DataType       = F16;
using D1DataType       = F16;
using AccDataType      = F32;
using CShuffleDataType = F32;
using E1DataType       = F16;

using A0Layout  = Row;
using B0Layout  = Col;
using D00Layout = Row;
using D01Layout = Row;
using B1Layout  = Row;
using D1Layout  = Row;
using E1Layout  = Row;

using A0ElementOp   = PassThrough;
using B0ElementOp   = PassThrough;
using CDE0ElementOp = AddAddRelu;
using A1ElementOp   = PassThrough;
using B1ElementOp   = PassThrough;
using CDE1ElementOp = ck::tensor_operation::element_wise::Add;

constexpr auto GemmSpec = ck::tensor_operation::device::GemmSpecialization::Default;

using DeviceGemmInstance =
    ck::tensor_operation::device::DeviceBatchedGemmMultipleDGemmMultipleD_Wmma_CShuffleV3<
        A0Layout,
        B0Layout,
        ck::Tuple<D00Layout, D01Layout>,
        B1Layout,
        ck::Tuple<D1Layout>,
        E1Layout,
        A0DataType,
        B0DataType,
        ck::Tuple<D00DataType, D01DataType>,
        B1DataType,
        ck::Tuple<D1DataType>,
        E1DataType,
        AccDataType,
        CShuffleDataType,
        A0ElementOp,
        B0ElementOp,
        CDE0ElementOp,
        B1ElementOp,
        CDE1ElementOp,
        GemmSpec,

        32, // BlockSize
        16, // MPerBlock
        64, // LPerBlock
        64, // KPerBlock
        64, // NPerBlock (Gemm1NPerBlock)
        64, // LTilePerBlock (Gemm1KPerBlock)
        8,  // AK1
        8,  // BK1
        8,  // L1 (B1K1)
        16, // MPerWmma
        16, // LPerWmma
        1,  // MRepeat
        4,  // LRepeat (Gemm0NRepeat)
        4,  // NRepeat (Gemm1NRepeat)

        S<2, 16, 1>,    // ABlockTransferThreadClusterLengths_K0_M_K1
        S<1, 0, 2>,     // ABlockTransferThreadClusterArrangeOrder
        S<1, 0, 2>,     // ABlockTransferSrcAccessOrder
        2,              // ABlockTransferSrcVectorDim
        8,              // ABlockTransferSrcScalarPerVector
        8,              // ABlockTransferDstScalarPerVector_K1
        false,          // ABlockLdsAddExtraM
        S<2, 16, 1>,    // B0BlockTransferThreadClusterLengths_K0_L_K1
        S<1, 0, 2>,     // B0BlockTransferThreadClusterArrangeOrder
        S<1, 0, 2>,     // B0BlockTransferSrcAccessOrder
        2,              // B0BlockTransferSrcVectorDim
        8,              // B0BlockTransferSrcScalarPerVector
        8,              // B0BlockTransferDstScalarPerVector_K1
        false,          // B0BlockLdsAddExtraL
        4,              // CDE0BlockTransferSrcScalarPerVector
        S<2, 16, 1>,    // B1BlockTransferThreadClusterLengths_L0_N_L1
        S<0, 2, 1>,     // B1BlockTransferThreadClusterArrangeOrder
        S<0, 2, 1>,     // B1BlockTransferSrcAccessOrder
        1,              // B1BlockTransferSrcVectorDim
        4,              // B1BlockTransferSrcScalarPerVector
        2,              // B1BlockTransferDstScalarPerVector_L1
        true,           // B1BlockLdsAddExtraN
        1,              // CShuffleMRepeatPerShuffle
        2,              // CShuffleNRepeatPerShuffle
        S<1, 16, 1, 2>, // CShuffleBlockTransferClusterLengths_MBlock_MPerBlock_NBlock_NPerBlock
        8>;             // CShuffleBlockTransferScalarPerVector_NPerBlock

#include "batched_gemm_multiple_d_gemm_multiple_d.inc"
int main(int argc, char* argv[]) { return run_example(argc, argv); }
