// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <iostream>
#include <string>
#include <fstream>
#include <stdexcept>
#include <iomanip>
#include <vector>
#include <memory>

#include "ck_tile/core.hpp"
#include "ck_tile/host.hpp"
#include "common/utils.hpp"
#include "grouped_gemm_common.hpp"

#if __clang_major__ >= 23
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wlifetime-safety-intra-tu-suggestions"
#endif
// Data types and Layouts are defined by the generated kernel headers
// No hardcoded type definitions here to avoid conflicts

struct GroupedGemmProblem
{
    int group_count_;
    int kbatch_;
    std::vector<int> Ms_, Ns_, Ks_;
    std::vector<int> stride_As_, stride_Bs_, stride_Cs_;

    std::string dtype_a_, dtype_b_, dtype_acc_, dtype_c_;
    std::string layout_a_, layout_b_, layout_c_;

    friend std::ostream& operator<<(std::ostream& os, const GroupedGemmProblem& problem)
    {
        os << "{\n"
           << "   \"group_count\":" << problem.group_count_ << ",\n"
           << "   \"kbatch\":" << problem.kbatch_ << ",\n"
           << "   \"dtype_a\":\"" << problem.dtype_a_ << "\",\n"
           << "   \"dtype_b\":\"" << problem.dtype_b_ << "\",\n"
           << "   \"dtype_acc\":\"" << problem.dtype_acc_ << "\",\n"
           << "   \"dtype_c\":\"" << problem.dtype_c_ << "\",\n"
           << "   \"layout_a\":\"" << problem.layout_a_ << "\",\n"
           << "   \"layout_b\":\"" << problem.layout_b_ << "\",\n"
           << "   \"layout_c\":\"" << problem.layout_c_ << "\"\n"
           << "}";
        return os;
    }
};

/// @brief Function to compare the results of the device and host computations for a single group
bool compare_single(std::string instanceName,
                    ck_tile::index_t K,
                    ck_tile::index_t kbatch,
                    ck_tile::HostTensor<CDataType>& c_m_n_dev_result,
                    ck_tile::HostTensor<CDataType>& c_m_n_host_result)
{
    const float max_accumulated_value =
        *std::max_element(c_m_n_host_result.mData.begin(), c_m_n_host_result.mData.end());
    const auto rtol_atol = calculate_rtol_atol<ADataType, BDataType, AccDataType, CDataType>(
        K, kbatch, max_accumulated_value);
    bool pass = ck_tile::check_err(c_m_n_dev_result,
                                   c_m_n_host_result,
                                   "Error: Incorrect results!",
                                   rtol_atol.at(ck_tile::number<0>{}),
                                   rtol_atol.at(ck_tile::number<1>{}));

    std::cout << "For " << instanceName << " Relative error threshold is "
              << rtol_atol.at(ck_tile::number<0>{}) << " Absolute error threshold is "
              << rtol_atol.at(ck_tile::number<1>{}) << std::endl;
    std::cout << "The verification result is:" << (pass ? "correct" : "fail") << std::endl;

    return pass;
}

/// @brief Function to compare grouped gemm results across all groups
bool compare_grouped(std::string instanceName,
                     const GroupedGemmProblem& problem,
                     std::vector<ck_tile::HostTensor<CDataType>>& c_dev_results,
                     std::vector<ck_tile::HostTensor<CDataType>>& c_host_results)
{
    bool pass = true;
    for(int i = 0; i < problem.group_count_; ++i)
    {
        pass &= compare_single(instanceName + "[" + std::to_string(i) + "]",
                               problem.Ks_[i],
                               problem.kbatch_,
                               c_dev_results[i],
                               c_host_results[i]);
    }
    return pass;
}

/// @brief Function to get the kernel output with reference implementation on CPU/GPU for all groups
void gemm_host_reference_grouped(int verify,
                                 const GroupedGemmProblem& problem,
                                 std::vector<ck_tile::HostTensor<ADataType>>& a_tensors,
                                 std::vector<ck_tile::HostTensor<BDataType>>& b_tensors,
                                 std::vector<ck_tile::HostTensor<CDataType>>& c_host_results,
                                 std::vector<std::unique_ptr<ck_tile::DeviceMem>>& a_dev_bufs,
                                 std::vector<std::unique_ptr<ck_tile::DeviceMem>>& b_dev_bufs)
{
    const int group_count = problem.group_count_;

    if(verify == 1)
    {
        for(int i = 0; i < group_count; ++i)
        {
            c_host_results[i].SetZero();
            ck_tile::reference_gemm<ADataType, BDataType, AccDataType, CDataType>(
                a_tensors[i], b_tensors[i], c_host_results[i]);
        }
    }
    else if(verify == 2)
    {
        for(int i = 0; i < group_count; ++i)
        {
            if constexpr(std::is_same_v<BDataType, ck_tile::pk_int4_t>)
            {
                b_dev_bufs[i]->ToDevice(b_tensors[i].data());
            }

            ck_tile::DeviceMem c_gpu_buf_ref(c_host_results[i].get_element_space_size_in_bytes());
            c_host_results[i].SetZero();
            c_gpu_buf_ref.SetZero();

            ADataType* d_A = static_cast<ADataType*>(a_dev_bufs[i]->GetDeviceBuffer());
            BDataType* d_B = static_cast<BDataType*>(b_dev_bufs[i]->GetDeviceBuffer());
            CDataType* d_C = static_cast<CDataType*>(c_gpu_buf_ref.GetDeviceBuffer());

            ck_tile::reference_gemm_gpu<ADataType,
                                        BDataType,
                                        AccDataType,
                                        CDataType,
                                        ALayout,
                                        BLayout,
                                        CLayout>(d_A,
                                                 d_B,
                                                 d_C,
                                                 problem.Ms_[i],
                                                 problem.Ns_[i],
                                                 problem.Ks_[i],
                                                 problem.stride_As_[i],
                                                 problem.stride_Bs_[i],
                                                 problem.stride_Cs_[i]);

            c_gpu_buf_ref.FromDevice(c_host_results[i].data());
        }
    }
}
#if __clang_major__ >= 23
#pragma clang diagnostic pop
#endif
