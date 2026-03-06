// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <iostream>
#include <numeric>
#include <initializer_list>
#include <cstdlib>

#include "ck/ck.hpp"
#include "ck/tensor_operation/gpu/device/gemm_specialization.hpp"
#include "ck/tensor_operation/gpu/device/impl/device_batched_contraction_multiple_d_wmma_cshuffle.hpp"
#include "ck/tensor_operation/gpu/element/element_wise_operation.hpp"

#include "ck/library/utility/check_err.hpp"
#include "ck/library/utility/device_memory.hpp"
#include "ck/library/utility/host_tensor.hpp"
#include "ck/library/utility/host_tensor_generator.hpp"
#include "ck/library/utility/numeric.hpp"

using ::ck::DeviceMem;
using ::ck::HostTensorDescriptor;
using ::ck::make_ParallelTensorFunctor;
using ::ck::Tensor;

using Row    = ck::tensor_layout::gemm::RowMajor;
using Bypass = ck::tensor_layout::BypassLayoutVerification;

template <ck::index_t... Is>
using S = ck::Sequence<Is...>;

using F16 = ck::half_t;
using F32 = float;

using PassThrough = ck::tensor_operation::element_wise::PassThrough;
using Add         = ck::tensor_operation::element_wise::Add;

using ADataType        = F16;
using BDataType        = F16;
using AccDataType      = F32;
using CShuffleDataType = F16;
using DDataType        = F16;
using DsDataType       = ck::Tuple<DDataType>;
using EDataType        = F16;

static constexpr ck::index_t NumDimG = 2;
static constexpr ck::index_t NumDimM = 2;
static constexpr ck::index_t NumDimN = 2;
static constexpr ck::index_t NumDimK = 1;

using AElementOp   = ck::tensor_operation::element_wise::PassThrough;
using BElementOp   = ck::tensor_operation::element_wise::PassThrough;
using CDEElementOp = ck::tensor_operation::element_wise::Add;

static constexpr auto GemmSpec = ck::tensor_operation::device::GemmSpecialization::MNKPadding;

static constexpr auto ASpec  = ck::tensor_operation::device::TensorSpecialization::Default;
static constexpr auto BSpec  = ck::tensor_operation::device::TensorSpecialization::Default;
static constexpr auto DESpec = ck::tensor_operation::device::TensorSpecialization::Default;

using DeviceOpInstanceKKNN =
    ck::tensor_operation::device::DeviceBatchedContractionMultipleD_Wmma_CShuffle<NumDimG,
                                                                                  NumDimM,
                                                                                  NumDimN,
                                                                                  NumDimK,
                                                                                  ADataType,
                                                                                  BDataType,
                                                                                  AccDataType,
                                                                                  CShuffleDataType,
                                                                                  DsDataType,
                                                                                  EDataType,
                                                                                  AElementOp,
                                                                                  BElementOp,
                                                                                  CDEElementOp,
                                                                                  GemmSpec,
                                                                                  ASpec,
                                                                                  BSpec,
                                                                                  DESpec,
                                                                                  1,
                                                                                  128,
                                                                                  64,
                                                                                  64,
                                                                                  64,
                                                                                  4,
                                                                                  16,
                                                                                  16,
                                                                                  1,
                                                                                  4,
                                                                                  S<4, 32, 1>,
                                                                                  S<1, 0, 2>,
                                                                                  S<1, 0, 2>,
                                                                                  2,
                                                                                  4,
                                                                                  4,
                                                                                  false,
                                                                                  S<4, 32, 1>,
                                                                                  S<1, 0, 2>,
                                                                                  S<1, 0, 2>,
                                                                                  2,
                                                                                  4,
                                                                                  4,
                                                                                  false,
                                                                                  1,
                                                                                  1,
                                                                                  S<1, 64, 1, 2>,
                                                                                  8>;

using DeviceOpInstance = DeviceOpInstanceKKNN;

#include "run_batched_gemm_bias_e_permute_example.inc"
int main(int argc, char* argv[]) { return !run_batched_gemm_bias_e_permute_example(argc, argv); }
