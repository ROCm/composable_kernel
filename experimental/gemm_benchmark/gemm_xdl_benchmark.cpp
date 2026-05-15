// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2023, Advanced Micro Devices, Inc. All rights reserved.

#include "common.hpp"

#include "gemm_xdl_benchmark_instance.hpp"

#include "run_gemm_example_v3.inc"

namespace ck::tensor_operation::device::instance {

extern void add_gemm_xdl_benchmark_instances(gemm_xdl_benchmark_instances& instances);

bool init_opt_ptrs(gemm_xdl_benchmark_instances& op_ptrs)
{
    add_gemm_xdl_benchmark_instances(op_ptrs);
    return true;
}

} // namespace ck::tensor_operation::device::instance

int main(int argc, char* argv[])
{
    gemm_xdl_benchmark_instances op_ptrs;
    return !ck::tensor_operation::device::instance::init_opt_ptrs(op_ptrs) ||
           !run_gemm_splitk_example<false>(op_ptrs, argc, argv);
}
