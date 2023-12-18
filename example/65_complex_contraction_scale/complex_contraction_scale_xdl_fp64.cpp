// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2023, Advanced Micro Devices, Inc. All rights reserved.

#include "ck/ck.hpp"
#include "ck/tensor_operation/gpu/element/element_wise_operation.hpp"

#include "common_instances.hpp"

using ADataType        = F64;
using BDataType        = F64;
using AccDataType      = F64;
using CShuffleDataType = F64;
using DsDataType       = ck::Tuple<>;
using EDataType        = F64;
using ComputeDataType  = F64;

static constexpr ck::index_t NumDimM = 2;
static constexpr ck::index_t NumDimN = 2;
static constexpr ck::index_t NumDimK = 2;

using AElementOp         = ck::tensor_operation::element_wise::PassThrough;
using BElementOp         = ck::tensor_operation::element_wise::PassThrough;
using CDEElementOp       = ck::tensor_operation::element_wise::Bilinear;
using CDEElementOp_Scale = ck::tensor_operation::element_wise::Scale;

using DeviceOpInstanceKKN = DeviceOpInstanceKK_Generic<NumDimM,
                                                       NumDimN,
                                                       NumDimK,
                                                       ADataType,
                                                       BDataType,
                                                       AccDataType,
                                                       CShuffleDataType,
                                                       DsDataType,
                                                       EDataType,
                                                       ComputeDataType,
                                                       AElementOp,
                                                       BElementOp,
                                                       CDEElementOp>;

using DeviceOpInstanceKKN_Scale = DeviceOpInstanceKK_Generic<NumDimM,
                                                             NumDimN,
                                                             NumDimK,
                                                             ADataType,
                                                             BDataType,
                                                             AccDataType,
                                                             CShuffleDataType,
                                                             DsDataType,
                                                             EDataType,
                                                             ComputeDataType,
                                                             AElementOp,
                                                             BElementOp,
                                                             CDEElementOp_Scale>;

using DeviceOpInstanceKNN = DeviceOpInstanceKN_Generic<NumDimM,
                                                       NumDimN,
                                                       NumDimK,
                                                       ADataType,
                                                       BDataType,
                                                       AccDataType,
                                                       CShuffleDataType,
                                                       DsDataType,
                                                       EDataType,
                                                       ComputeDataType,
                                                       AElementOp,
                                                       BElementOp,
                                                       CDEElementOp>;

using DeviceOpInstanceMKN = DeviceOpInstanceMK_Generic<NumDimM,
                                                       NumDimN,
                                                       NumDimK,
                                                       ADataType,
                                                       BDataType,
                                                       AccDataType,
                                                       CShuffleDataType,
                                                       DsDataType,
                                                       EDataType,
                                                       ComputeDataType,
                                                       AElementOp,
                                                       BElementOp,
                                                       CDEElementOp>;

using DeviceOpInstanceMNN = DeviceOpInstanceMN_Generic<NumDimM,
                                                       NumDimN,
                                                       NumDimK,
                                                       ADataType,
                                                       BDataType,
                                                       AccDataType,
                                                       CShuffleDataType,
                                                       DsDataType,
                                                       EDataType,
                                                       ComputeDataType,
                                                       AElementOp,
                                                       BElementOp,
                                                       CDEElementOp>;

using DeviceOpInstance       = DeviceOpInstanceKKN;
using DeviceOpInstance_Scale = DeviceOpInstanceKKN_Scale;

#include "run_complex_contraction_scale_example.inc"

int main(int argc, char* argv[]) { return run_complex_contraction_scale_example(argc, argv); }
