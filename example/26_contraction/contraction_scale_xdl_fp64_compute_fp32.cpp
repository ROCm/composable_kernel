// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2023, Advanced Micro Devices, Inc. All rights reserved.

#include "ck/ck.hpp"
#include "ck/tensor_operation/gpu/element/element_wise_operation.hpp"

#include "common_instances.hpp"

using ADataType        = F64;
using BDataType        = F64;
using AccDataType      = F32;
using CShuffleDataType = F64;
using DsDataType       = ck::Tuple<>;
using EDataType        = F64;
using ComputeDataType  = F32;

static constexpr ck::index_t NumDimM = 2;
static constexpr ck::index_t NumDimN = 2;
static constexpr ck::index_t NumDimK = 2;

using AElementOp   = ck::tensor_operation::element_wise::PassThrough;
using BElementOp   = ck::tensor_operation::element_wise::PassThrough;
using CDEElementOp = ck::tensor_operation::element_wise::Scale;

using DeviceOpInstanceKKN = DeviceOpInstanceKK_FP64<NumDimM,
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

using DeviceOpInstanceKNN = DeviceOpInstanceKN_FP64<NumDimM,
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

using DeviceOpInstanceMKN = DeviceOpInstanceMK_FP64<NumDimM,
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

using DeviceOpInstanceMNN = DeviceOpInstanceMN_FP64<NumDimM,
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

using DeviceOpInstance = DeviceOpInstanceKKN;

#include "run_contraction_scale_example.inc"

int main(int argc, char* argv[]) { return run_contraction_scale_example(argc, argv); }
