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
#include "ck/tensor_operation/gpu/device/impl/device_batched_gemm_multiple_d_gemm_multiple_d_xdl_cshuffle.hpp"
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

using A0DataType        = F16;
using B0DataType        = F16;
using AccDataType       = F32;
using D00DataType       = F16;
using D01DataType       = F16;
using B1DataType        = F16;
using C1ShuffleDataType = F32;
using D1DataType        = F16;
using E1DataType        = F16;

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

static constexpr bool PadGemm0M = false;
static constexpr bool PadGemm0N = false;
static constexpr bool PadGemm0K = false;
static constexpr bool PadGemm1N = false;
static constexpr bool PadGemm1K = false;

using DeviceGemmInstance =
    ck::tensor_operation::device::DeviceBatchedGemmMultipleDGemmMultipleD_Xdl_CShuffle<
        A0Layout,
        B0Layout,
        ck::Tuple<D00Layout, D01Layout>,
        B1Layout,
        ck::Tuple<D1Layout>,
        E1Layout,
        A0DataType,
        B0DataType,
        AccDataType,
        ck::Tuple<D00DataType, D01DataType>,
        B1DataType,
        AccDataType,
        C1ShuffleDataType,
        ck::Tuple<D1DataType>,
        E1DataType,
        A0ElementOp,
        B0ElementOp,
        CDE0ElementOp,
        B1ElementOp,
        CDE1ElementOp,
        PadGemm0M,
        PadGemm0N,
        PadGemm0K,
        PadGemm1N,
        PadGemm1K,
        1,
        256,
        128,         // MPerBlock
        128,         // NPerBlock
        32,          // KPerBlock
        128,         // Gemm1NPerBlock
        32,          // Gemm1KPerBlock
        8,           // AK1
        8,           // BK1
        2,           // B1K1
        16,          // MPerXDL
        16,          // NPerXDL
        2,           // MXdlPerWave
        8,           // NXdlPerWave
        8,           // Gemm1NXdlPerWave
        S<4, 64, 1>, // ABlockTransfer
        S<1, 0, 2>,
        S<1, 0, 2>,
        2,
        8,
        8,
        true,
        S<4, 64, 1>, // BBlockTransfer
        S<1, 0, 2>,
        S<1, 0, 2>,
        2,
        8,
        8,
        true,
        9,           // D0sTransferSrcVectorDim
        4,           // D0sTransferSrcScalaerPerVector
        S<8, 32, 1>, // B1BlockTransfer
        S<0, 2, 1>,
        S<0, 2, 1>,
        1,
        4,
        2,
        false,
        1,              // CShuffleMXdlPerWavePerShuffle
        2,              // CShuffleNXdlPerWavePerShuffle
        S<1, 32, 1, 8>, // CShuffleBlockTransferClusterLengths_MBlock_MPerBlock_NBlock_NPerBlock
        4>;             // CShuffleBlockTransferScalarPerVector_NPerBlock

#include "batched_gemm_multiple_d_gemm_multiple_d.inc"
int main(int argc, char* argv[]) { return run_example(argc, argv); }
