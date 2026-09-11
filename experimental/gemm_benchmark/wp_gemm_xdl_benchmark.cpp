// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2023, Advanced Micro Devices, Inc. All rights reserved.

#include "common.hpp"
#include "ck/library/reference_tensor_operation/cpu/reference_gemm.hpp"

#include "wp_gemm_xdl_benchmark_instance.hpp"

#include "run_gemm_example_v3.inc"

namespace ck::tensor_operation::device::instance {

extern void add_wp_gemm_xdl_benchmark_instances(wp_gemm_xdl_benchmark_instances& instances);

bool init_opt_ptrs(wp_gemm_xdl_benchmark_instances& op_ptrs)
{
    add_wp_gemm_xdl_benchmark_instances(op_ptrs);
    return true;
}

} // namespace ck::tensor_operation::device::instance

int main(int argc, char* argv[])
{
    wp_gemm_xdl_benchmark_instances op_ptrs;
    return !ck::tensor_operation::device::instance::init_opt_ptrs(op_ptrs) ||
           !run_gemm_splitk_example<true>(op_ptrs, argc, argv);
}
