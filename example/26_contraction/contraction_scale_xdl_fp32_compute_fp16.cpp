// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "ck/ck.hpp"
#include "ck/tensor_operation/gpu/element/element_wise_operation.hpp"

#include "common_instances.hpp"

using ADataType        = F32;
using BDataType        = F32;
using AccDataType      = F32;
using CShuffleDataType = F32;
using DsDataType       = ck::Tuple<>;
using EDataType        = F32;
using ComputeDataType  = F16;

static constexpr ck::index_t NumDimM = 2;
static constexpr ck::index_t NumDimN = 2;
static constexpr ck::index_t NumDimK = 2;

using AElementOp   = ck::tensor_operation::element_wise::PassThrough;
using BElementOp   = ck::tensor_operation::element_wise::PassThrough;
using CDEElementOp = ck::tensor_operation::element_wise::Scale;

// Instantiate DeviceOpInstance for all four layout variants (KK, KN, MK, MN).
// See common_instances.hpp for macro definition and available BASE/SUFFIX options.
CK_CONTRACTION_DEVICE_OP_INSTANCES(Generic, N);

#include "run_contraction_scale_example.inc"

int main(int argc, char* argv[]) { return run_contraction_scale_example(argc, argv); }
