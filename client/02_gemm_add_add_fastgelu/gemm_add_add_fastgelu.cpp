// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2022, Advanced Micro Devices, Inc. All rights reserved.

#include <iomanip>
#include <vector>
#include <iostream>

#include "ck/ck.hpp"
#include "ck/tensor_operation/gpu/device/tensor_layout.hpp"
#include "ck/tensor_operation/gpu/device/device_gemm_multiple_d.hpp"
#include "ck/tensor_operation/gpu/element/element_wise_operation.hpp"

struct SimpleDeviceMem
{
    SimpleDeviceMem() = delete;

    SimpleDeviceMem(std::size_t mem_size) : p_mem_{}
    {
        (void)hipMalloc(static_cast<void**>(&p_mem_), mem_size);
    }

    void* GetDeviceBuffer() { return p_mem_; }

    ~SimpleDeviceMem() { (void)hipFree(p_mem_); }

    void* p_mem_;
};

namespace ck {
namespace tensor_operation {
namespace device {
namespace device_gemm_instance {

using DeviceGemmAddAddFastGeluPtr = ck::tensor_operation::device::DeviceGemmMultipleDPtr<
    2,
    ck::tensor_operation::element_wise::PassThrough,
    ck::tensor_operation::element_wise::PassThrough,
    ck::tensor_operation::element_wise::AddAddFastGelu>;

void add_device_gemm_add_add_fastgelu_xdl_c_shuffle_f16_f16_f16_mk_kn_mn_instances(
    std::vector<DeviceGemmAddAddFastGeluPtr>&);
void add_device_gemm_add_add_fastgelu_xdl_c_shuffle_f16_f16_f16_mk_nk_mn_instances(
    std::vector<DeviceGemmAddAddFastGeluPtr>&);
void add_device_gemm_add_add_fastgelu_xdl_c_shuffle_f16_f16_f16_km_kn_mn_instances(
    std::vector<DeviceGemmAddAddFastGeluPtr>&);
void add_device_gemm_add_add_fastgelu_xdl_c_shuffle_f16_f16_f16_km_nk_mn_instances(
    std::vector<DeviceGemmAddAddFastGeluPtr>&);

} // namespace device_gemm_instance
} // namespace device
} // namespace tensor_operation
} // namespace ck

template <typename ADataType,
          typename BDataType,
          typename AccDataType,
          typename D0DataType,
          typename D1DataType,
          typename EDataType,
          typename ALayout,
          typename BLayout,
          typename D0Layout,
          typename D1Layout,
          typename ELayout>
void client_gemm_add_add_fastgelu_impl(
    int M, int N, int K, int StrideA, int StrideB, int StrideD0, int StrideD1, int StrideE)
{
    using PassThrough    = ck::tensor_operation::element_wise::PassThrough;
    using AddAddFastGelu = ck::tensor_operation::element_wise::AddAddFastGelu;

    using AElementOp   = PassThrough;
    using BElementOp   = PassThrough;
    using CDEElementOp = AddAddFastGelu;

    const auto a_element_op   = AElementOp{};
    const auto b_element_op   = BElementOp{};
    const auto cde_element_op = CDEElementOp{};

    // add device GEMM instances
    std::vector<ck::tensor_operation::device::device_gemm_instance::DeviceGemmAddAddFastGeluPtr>
        device_op_ptrs;

    if(std::is_same<ADataType, ck::half_t>::value && std::is_same<BDataType, ck::half_t>::value &&
       std::is_same<EDataType, ck::half_t>::value)
    {
        if(std::is_same<ALayout, ck::tensor_layout::gemm::RowMajor>::value &&
           std::is_same<BLayout, ck::tensor_layout::gemm::RowMajor>::value &&
           std::is_same<ELayout, ck::tensor_layout::gemm::RowMajor>::value)
        {
            ck::tensor_operation::device::device_gemm_instance::
                add_device_gemm_add_add_fastgelu_xdl_c_shuffle_f16_f16_f16_mk_kn_mn_instances(
                    device_op_ptrs);
        }
        else if(std::is_same<ALayout, ck::tensor_layout::gemm::RowMajor>::value &&
                std::is_same<BLayout, ck::tensor_layout::gemm::ColumnMajor>::value &&
                std::is_same<ELayout, ck::tensor_layout::gemm::RowMajor>::value)
        {
            ck::tensor_operation::device::device_gemm_instance::
                add_device_gemm_add_add_fastgelu_xdl_c_shuffle_f16_f16_f16_mk_nk_mn_instances(
                    device_op_ptrs);
        }
        else if(std::is_same<ALayout, ck::tensor_layout::gemm::ColumnMajor>::value &&
                std::is_same<BLayout, ck::tensor_layout::gemm::RowMajor>::value &&
                std::is_same<ELayout, ck::tensor_layout::gemm::RowMajor>::value)
        {
            ck::tensor_operation::device::device_gemm_instance::
                add_device_gemm_add_add_fastgelu_xdl_c_shuffle_f16_f16_f16_km_kn_mn_instances(
                    device_op_ptrs);
        }
        else if(std::is_same<ALayout, ck::tensor_layout::gemm::ColumnMajor>::value &&
                std::is_same<BLayout, ck::tensor_layout::gemm::ColumnMajor>::value &&
                std::is_same<ELayout, ck::tensor_layout::gemm::RowMajor>::value)
        {
            ck::tensor_operation::device::device_gemm_instance::
                add_device_gemm_add_add_fastgelu_xdl_c_shuffle_f16_f16_f16_km_nk_mn_instances(
                    device_op_ptrs);
        }
    }

    std::cout << "found " << device_op_ptrs.size() << " instances" << std::endl;

    auto f_matrix_space_size =
        [](std::size_t nRow, std::size_t nCol, std::size_t stride, auto layout) {
            using Layout = decltype(layout);

            if(std::is_same<Layout, ck::tensor_layout::gemm::RowMajor>::value)
            {
                return (nRow - 1) * stride + nCol;
            }
            else
            {
                return (nCol - 1) * stride + nRow;
            }
        };

    SimpleDeviceMem a_device_buf(sizeof(ADataType) * f_matrix_space_size(M, K, StrideA, ALayout{}));
    SimpleDeviceMem b_device_buf(sizeof(BDataType) * f_matrix_space_size(K, N, StrideB, BLayout{}));
    SimpleDeviceMem d0_m_n_device_buf(sizeof(D0DataType) *
                                      f_matrix_space_size(M, N, StrideD0, D0Layout{}));
    SimpleDeviceMem d1_m_n_device_buf(sizeof(D1DataType) *
                                      f_matrix_space_size(M, N, StrideD1, D1Layout{}));
    SimpleDeviceMem e_device_buf(sizeof(EDataType) * f_matrix_space_size(M, N, StrideE, ELayout{}));

    std::string best_device_op_name;
    float best_ave_time   = 0;
    float best_tflops     = 0;
    float best_gb_per_sec = 0;

    // profile device operation instances
    for(auto& device_op_ptr : device_op_ptrs)
    {
        auto argument_ptr = device_op_ptr->MakeArgumentPointer(
            a_device_buf.GetDeviceBuffer(),
            b_device_buf.GetDeviceBuffer(),
            std::array<const void*, 2>{d0_m_n_device_buf.GetDeviceBuffer(),
                                       d1_m_n_device_buf.GetDeviceBuffer()},
            static_cast<EDataType*>(e_device_buf.GetDeviceBuffer()),
            M,
            N,
            K,
            StrideA,
            StrideB,
            std::array<ck::index_t, 2>{StrideD0, StrideD1},
            StrideE,
            a_element_op,
            b_element_op,
            cde_element_op);

        auto invoker_ptr = device_op_ptr->MakeInvokerPointer();

        std::string device_op_name = device_op_ptr->GetTypeString();

        if(device_op_ptr->IsSupportedArgument(argument_ptr.get()))
        {
            float ave_time = invoker_ptr->Run(argument_ptr.get(), StreamConfig{nullptr, true});

            std::size_t flop = std::size_t(2) * M * N * K;

            std::size_t num_btype =
                sizeof(ADataType) * M * K + sizeof(BDataType) * K * N + sizeof(EDataType) * M * N;

            float tflops = static_cast<float>(flop) / 1.E9 / ave_time;

            float gb_per_sec = num_btype / 1.E6 / ave_time;

            std::cout << "Perf: " << std::setw(10) << ave_time << " ms, " << tflops << " TFlops, "
                      << gb_per_sec << " GB/s, " << device_op_name << std::endl;

            if(tflops > best_tflops)
            {
                best_device_op_name = device_op_name;
                best_tflops         = tflops;
                best_ave_time       = ave_time;
                best_gb_per_sec     = gb_per_sec;
            }
        }
        else
        {
            std::cout << device_op_name << " does not support this problem" << std::endl;
        }
    }

    std::cout << "Best Perf: " << best_ave_time << " ms, " << best_tflops << " TFlops, "
              << best_gb_per_sec << " GB/s, " << best_device_op_name << std::endl;
}

int main(int argc, char* argv[])
{
    // GEMM shape
    ck::index_t M = 3840;
    ck::index_t N = 4096;
    ck::index_t K = 4096;

    ck::index_t StrideA  = 4096;
    ck::index_t StrideB  = 4096;
    ck::index_t StrideD0 = 0;
    ck::index_t StrideD1 = 4096;
    ck::index_t StrideE  = 4096;

    if(argc == 1)
    {
        // use default case
    }
    else if(argc == 7)
    {
        M = std::stoi(argv[1]);
        N = std::stoi(argv[2]);
        K = std::stoi(argv[3]);

        StrideA  = std::stoi(argv[4]);
        StrideB  = std::stoi(argv[5]);
        StrideD0 = std::stoi(argv[6]);
        StrideD1 = std::stoi(argv[7]);
        StrideE  = std::stoi(argv[8]);
    }
    else
    {
        printf("arg1 to 8: M (256x), N(128x), K(32x), StrideA, StrideB, StrideD0, StrideD1, "
               "StrideE\n");
        exit(0);
    }

    using F16 = ck::half_t;
    using F32 = float;

    using Row = ck::tensor_layout::gemm::RowMajor;
    using Col = ck::tensor_layout::gemm::ColumnMajor;

    client_gemm_add_add_fastgelu_impl<F16, F16, F32, F16, F16, F16, Row, Col, Row, Row, Row>(
        M, N, K, StrideA, StrideB, StrideD0, StrideD1, StrideE);

    return 0;
}
