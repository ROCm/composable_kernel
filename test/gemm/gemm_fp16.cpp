// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2022, Advanced Micro Devices, Inc. All rights reserved.

#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <numeric>
#include <tuple>
#include <vector>

#include "ck/ck.hpp"
#include "ck/tensor_operation/gpu/device/tensor_layout.hpp"
#include "ck/tensor_operation/gpu/device/gemm_specialization.hpp"
#include "ck/tensor_operation/gpu/device/device_gemm.hpp"
#include "ck/tensor_operation/gpu/element/element_wise_operation.hpp"

#include "ck/library/tensor_operation_instance/gpu/gemm.hpp"

#include "ck/library/utility/check_err.hpp"
#include "ck/library/utility/device_memory.hpp"
#include "ck/library/utility/host_tensor.hpp"
#include "ck/library/utility/host_tensor_generator.hpp"
#include "ck/library/reference_tensor_operation/cpu/reference_gemm.hpp"

#include "test/gemm/gemm_util.hpp"

int main()
{
    using ADataType   = ck::half_t;
    using BDataType   = ck::half_t;
    using CDataType   = ck::half_t;
    using AccDataType = float;

    using Row = ck::tensor_layout::gemm::RowMajor;
    using Col = ck::tensor_layout::gemm::ColumnMajor;

    using PassThrough = ck::tensor_operation::element_wise::PassThrough;

    auto test = [&](auto a_layout, auto b_layout, auto c_layout) {
        bool pass = true;

        using DeviceOp = ck::tensor_operation::device::DeviceGemm<decltype(a_layout),
                                                                  decltype(b_layout),
                                                                  decltype(c_layout),
                                                                  ADataType,
                                                                  BDataType,
                                                                  CDataType,
                                                                  PassThrough,
                                                                  PassThrough,
                                                                  PassThrough>;

        const auto gemmPtrs =
            ck::tensor_operation::device::instance::DeviceOperationInstanceFactory<
                DeviceOp>::GetInstances();

        for(auto& gemmPtr : gemmPtrs)
        {
            pass &= ck::gemm_util::TestGemm<std::unique_ptr<DeviceOp>,
                                            ADataType,
                                            BDataType,
                                            CDataType,
                                            AccDataType,
                                            decltype(a_layout),
                                            decltype(b_layout),
                                            decltype(c_layout),
                                            PassThrough,
                                            PassThrough,
                                            PassThrough>{}(gemmPtr);
        }

        return pass;
    };

    bool pass = test(Row{}, Row{}, Row{}) && test(Row{}, Col{}, Row{}) &&
                test(Col{}, Row{}, Row{}) && test(Col{}, Col{}, Row{});

    std::cout << "TestGemm ..... " << (pass ? "SUCCESS" : "FAILURE") << std::endl;
    return pass ? 0 : 1;
}
