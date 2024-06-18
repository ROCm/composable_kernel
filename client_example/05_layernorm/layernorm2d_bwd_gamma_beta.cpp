// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2024, Advanced Micro Devices, Inc. All rights reserved.

#include <iomanip>
#include <vector>
#include <iostream>

#include "ck/ck.hpp"
#include "ck/tensor_operation/gpu/device/device_normalization_bwd_gamma_beta.hpp"

#include "ck/library/tensor_operation_instance/gpu/layernorm_bwd_gamma_beta.hpp"

using DYDataType         = float;
using XDataType          = float;
using GammaDataType      = float;
using MeanInvStdDataType = float;
using DGammaDataType     = float;
using DBetaDataType      = float;

constexpr int Rank         = 2;
constexpr int NumReduceDim = 1;

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

int main(int argc, char* argv[])
{
    ck::index_t M = 1024;
    ck::index_t N = 1024;

    SimpleDeviceMem dy_dev(sizeof(DYDataType) * M * N);
    SimpleDeviceMem x_dev(sizeof(XDataType) * M * N);
    SimpleDeviceMem mean_dev(sizeof(MeanInvStdDataType) * M);
    SimpleDeviceMem inv_std_dev(sizeof(MeanInvStdDataType) * M);
    SimpleDeviceMem dgamma_dev(sizeof(DGammaDataType) * N);
    SimpleDeviceMem dbeta_dev(sizeof(DBetaDataType) * N);

    using DeviceOp =
        ck::tensor_operation::device::DeviceNormalizationBwdGammaBeta<DYDataType,
                                                                      XDataType,
                                                                      MeanInvStdDataType,
                                                                      DGammaDataType,
                                                                      DBetaDataType,
                                                                      Rank,
                                                                      NumReduceDim>;

    // get device op instances
    const auto op_ptrs = ck::tensor_operation::device::instance::DeviceOperationInstanceFactory<
        DeviceOp>::GetInstances();

    std::cout << "found " << op_ptrs.size() << " instances" << std::endl;

    std::string best_op_name;
    bool found            = false;
    int best_op_id        = -1;
    float best_ave_time   = std::numeric_limits<float>::max();
    float best_gb_per_sec = 0;

    // profile device operation instances
    std::cout << "Run all instances and do timing" << std::endl;

    std::size_t num_bytes = sizeof(DYDataType) * M * N + sizeof(XDataType) * M * N +
                            sizeof(MeanInvStdDataType) * M * 2 + sizeof(DGammaDataType) * N +
                            sizeof(DBetaDataType) * N;

    for(int i = 0; i < op_ptrs.size(); ++i)
    {
        auto& op_ptr = op_ptrs[i];

        auto argument_ptr = op_ptr->MakeArgumentPointer({M, N}, // inLengths
                                                        {N, 1}, // dyStrides
                                                        {N, 1}, // xStrides
                                                        {1, 0}, // meanStrides
                                                        {1, 0}, // invStdStrides
                                                        {N},    // outLengths
                                                        {1},    // dgammaStrides
                                                        {1},    // dbetaStrides
                                                        {0},    // reduceDims
                                                        dy_dev.GetDeviceBuffer(),
                                                        x_dev.GetDeviceBuffer(),
                                                        mean_dev.GetDeviceBuffer(),
                                                        inv_std_dev.GetDeviceBuffer(),
                                                        dgamma_dev.GetDeviceBuffer(),
                                                        dbeta_dev.GetDeviceBuffer());

        auto invoker_ptr = op_ptr->MakeInvokerPointer();

        std::string op_name = op_ptr->GetTypeString();

        if(op_ptr->IsSupportedArgument(argument_ptr.get()))
        {
            size_t workspace_sz = op_ptr->GetWorkSpaceSize(argument_ptr.get());
            SimpleDeviceMem workspace(workspace_sz);
            op_ptr->SetWorkSpacePointer(argument_ptr.get(), workspace.GetDeviceBuffer());

            float ave_time   = invoker_ptr->Run(argument_ptr.get(), StreamConfig{nullptr, true});
            float gb_per_sec = num_bytes / 1.E6 / ave_time;

            std::cout << "Perf: " << std::setw(10) << ave_time << " ms, " << gb_per_sec << " GB/s, "
                      << op_name << std::endl;

            if(ave_time < best_ave_time)
            {
                found           = true;
                best_op_id      = i;
                best_op_name    = op_name;
                best_ave_time   = ave_time;
                best_gb_per_sec = gb_per_sec;
            }
        }
        else
        {
            std::cout << op_name << " does not support this problem" << std::endl;
        }
    }

    std::cout << "Best Perf: " << best_ave_time << " ms, " << best_gb_per_sec << " GB/s, "
              << best_op_name << std::endl;

    // run the best intance
    if(found)
    {
        auto& op_ptr = op_ptrs[best_op_id];
        std::cout << "Run the best instance without timing: " << op_ptr->GetTypeString()
                  << std::endl;

        auto argument_ptr = op_ptr->MakeArgumentPointer({M, N}, // inLengths
                                                        {N, 1}, // dyStrides
                                                        {N, 1}, // xStrides
                                                        {1, 0}, // meanStrides
                                                        {1, 0}, // invStdStrides
                                                        {N},    // outLengths
                                                        {1},    // dgammaStrides
                                                        {1},    // dbetaStrides
                                                        {0},    // reduceDims
                                                        dy_dev.GetDeviceBuffer(),
                                                        x_dev.GetDeviceBuffer(),
                                                        mean_dev.GetDeviceBuffer(),
                                                        inv_std_dev.GetDeviceBuffer(),
                                                        dgamma_dev.GetDeviceBuffer(),
                                                        dbeta_dev.GetDeviceBuffer());

        auto invoker_ptr = op_ptr->MakeInvokerPointer();

        if(op_ptr->IsSupportedArgument(argument_ptr.get()))
        {
            size_t workspace_sz = op_ptr->GetWorkSpaceSize(argument_ptr.get());
            SimpleDeviceMem workspace(workspace_sz);
            op_ptr->SetWorkSpacePointer(argument_ptr.get(), workspace.GetDeviceBuffer());

            invoker_ptr->Run(argument_ptr.get(), StreamConfig{nullptr, false});
        }

        std::cout << "Done" << std::endl;
    }

    return 0;
}
