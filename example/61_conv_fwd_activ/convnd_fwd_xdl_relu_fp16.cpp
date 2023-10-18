// SPDX-License-Identifier: MIT
// Copyright (c) 2023, Advanced Micro Devices, Inc. All rights reserved.

#include "convnd_fwd_activ_common.hpp"

using OutElementOp = ck::tensor_operation::element_wise::Relu;

using DeviceGroupedConvNDFwdActivInstance = DeviceGroupedConvNDFwdInstance<OutElementOp>;
#include "run_convnd_fwd_activ_example.inc"

int main(int argc, char* argv[]) { return run_convnd_fwd_example(argc, argv) ? 0 : 1; }
