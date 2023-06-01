// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2022, Advanced Micro Devices, Inc. All rights reserved.

#include <iostream>

#include "profiler/profile_batched_gemm_impl.hpp"

#include "ck/library/tensor_operation_instance/gpu/batched_gemm_multi_d.hpp"

namespace {
using ADataType = ck::half_t;
using BDataType = ck::half_t;
using CDataType = ck::half_t;

using Row = ck::tensor_layout::gemm::RowMajor;
using Col = ck::tensor_layout::gemm::ColumnMajor;

using Empty_Tuple = ck::Tuple<>;

using PassThrough = ck::tensor_operation::element_wise::PassThrough;
} // namespace

int main()
{
    int M          = 512;
    int N          = 256;
    int K          = 128;
    int BatchCount = 3;

    bool pass = true;

    using namespace ck::tensor_operation::device;

    pass = pass && ck::profiler::profile_batched_gemm_impl<ADataType,
                                                           BDataType,
                                                           CDataType,
                                                           Row,
                                                           Row,
                                                           Row,
                                                           PassThrough,
                                                           PassThrough,
                                                           PassThrough,
                                                           DeviceBatchedGemmMultiD<Row,
                                                                                   Row,
                                                                                   Empty_Tuple,
                                                                                   Row,
                                                                                   ADataType,
                                                                                   BDataType,
                                                                                   Empty_Tuple,
                                                                                   CDataType,
                                                                                   PassThrough,
                                                                                   PassThrough,
                                                                                   PassThrough>>(
                       true, 1, false, 1, M, N, K, K, N, N, M * K, K * N, M * N, BatchCount);

    pass = pass && ck::profiler::profile_batched_gemm_impl<ADataType,
                                                           BDataType,
                                                           CDataType,
                                                           Row,
                                                           Col,
                                                           Row,
                                                           PassThrough,
                                                           PassThrough,
                                                           PassThrough,
                                                           DeviceBatchedGemmMultiD<Row,
                                                                                   Col,
                                                                                   Empty_Tuple,
                                                                                   Row,
                                                                                   ADataType,
                                                                                   BDataType,
                                                                                   Empty_Tuple,
                                                                                   CDataType,
                                                                                   PassThrough,
                                                                                   PassThrough,
                                                                                   PassThrough>>(
                       true, 1, false, 1, M, N, K, K, K, N, M * K, K * N, M * N, BatchCount);

    pass = pass && ck::profiler::profile_batched_gemm_impl<ADataType,
                                                           BDataType,
                                                           CDataType,
                                                           Col,
                                                           Row,
                                                           Row,
                                                           PassThrough,
                                                           PassThrough,
                                                           PassThrough,
                                                           DeviceBatchedGemmMultiD<Col,
                                                                                   Row,
                                                                                   Empty_Tuple,
                                                                                   Row,
                                                                                   ADataType,
                                                                                   BDataType,
                                                                                   Empty_Tuple,
                                                                                   CDataType,
                                                                                   PassThrough,
                                                                                   PassThrough,
                                                                                   PassThrough>>(
                       true, 1, false, 1, M, N, K, M, N, N, M * K, K * N, M * N, BatchCount);

    pass = pass && ck::profiler::profile_batched_gemm_impl<ADataType,
                                                           BDataType,
                                                           CDataType,
                                                           Col,
                                                           Col,
                                                           Row,
                                                           PassThrough,
                                                           PassThrough,
                                                           PassThrough,
                                                           DeviceBatchedGemmMultiD<Col,
                                                                                   Col,
                                                                                   Empty_Tuple,
                                                                                   Row,
                                                                                   ADataType,
                                                                                   BDataType,
                                                                                   Empty_Tuple,
                                                                                   CDataType,
                                                                                   PassThrough,
                                                                                   PassThrough,
                                                                                   PassThrough>>(
                       true, 1, false, 1, M, N, K, M, K, N, M * K, K * N, M * N, BatchCount);

    std::cout << "test BatchedGEMMMultiD fp16: " << (pass ? "Pass" : "Fail") << std::endl;
    return pass ? 0 : 1;
}
