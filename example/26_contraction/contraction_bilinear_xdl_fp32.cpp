// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2023, Advanced Micro Devices, Inc. All rights reserved.

#include "ck/ck.hpp"
#include "ck/tensor_operation/gpu/element/element_wise_operation.hpp"

#include "common_instances.hpp"

using ADataType        = F32;
using BDataType        = F32;
using AccDataType      = F32;
using CShuffleDataType = F32;
using DDataType        = F32;
using DsDataType       = ck::Tuple<DDataType>;
using EDataType        = F32;
using ComputeDataType  = F32;

static constexpr ck::index_t NumDimM = 2;
static constexpr ck::index_t NumDimN = 2;
static constexpr ck::index_t NumDimK = 2;

using AElementOp   = ck::tensor_operation::element_wise::PassThrough;
using BElementOp   = ck::tensor_operation::element_wise::PassThrough;
using CDEElementOp = ck::tensor_operation::element_wise::Bilinear;

using DeviceOpInstanceKKNN = DeviceOpInstanceKK_Generic<NumDimM,
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

using DeviceOpInstanceKNNN = DeviceOpInstanceKN_Generic<NumDimM,
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

using DeviceOpInstanceMKNN = DeviceOpInstanceMK_Generic<NumDimM,
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

using DeviceOpInstanceMNNN = DeviceOpInstanceMN_Generic<NumDimM,
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

using DeviceOpInstance = DeviceOpInstanceKKNN;

#include "run_contraction_bilinear_example.inc"

int main(int argc, char* argv[]) { return run_contraction_bilinear_example(argc, argv); }
