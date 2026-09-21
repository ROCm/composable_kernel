// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2023, Advanced Micro Devices, Inc. All rights reserved.

#include "common.hpp"
#include "ck/library/reference_tensor_operation/cpu/reference_mx_gemm.hpp"

#include "mx_gemm_xdl_benchmark_instance.hpp"

#include "run_mx_gemm_example_v3.inc"

namespace ck::tensor_operation::device::instance {

extern void add_mx_gemm_xdl_benchmark_instances(mx_gemm_xdl_benchmark_instances& instances);

bool init_opt_ptrs(mx_gemm_xdl_benchmark_instances& op_ptrs)
{
    add_mx_gemm_xdl_benchmark_instances(op_ptrs);
    return true;
}

} // namespace ck::tensor_operation::device::instance

int main(int argc, char* argv[])
{

    mx_gemm_xdl_benchmark_instances op_ptrs;
    return !ck::tensor_operation::device::instance::init_opt_ptrs(op_ptrs) ||
           !run_mx_gemm_splitk_example<false>(op_ptrs, argc, argv);
}
