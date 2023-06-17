// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2023, Advanced Micro Devices, Inc. All rights reserved.

#include <iomanip>
#include <vector>
#include <iostream>

#include "ck/ck.hpp"
#include "ck/tensor_operation/gpu/device/tensor_layout.hpp"
#include "ck/tensor_operation/gpu/device/device_normalization.hpp"
#include "ck/tensor_operation/gpu/element/element_wise_operation.hpp"

#include "ck/library/tensor_operation_instance/gpu/normalization_swish.hpp"

using XDataType       = ck::half_t;
using GammaDataType   = float;
using BetaDataType    = float;
using YDataType       = ck::half_t;
using ComputeDataType = float;
using Swish           = ck::tensor_operation::element_wise::Swish;

constexpr int Rank         = 5;
constexpr int NumReduceDim = 3;

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
    ck::index_t N = 32;
    ck::index_t H = 16;
    ck::index_t W = 16;
    ck::index_t G = 64;
    ck::index_t C = 128;

    std::size_t xy_size         = N * H * W * G * C;
    std::size_t gamma_beta_size = G * C;

    std::vector<ck::index_t> xy_strides         = {H * W * G * C, W * G * C, G * C, C, 1};
    std::vector<ck::index_t> gamma_beta_strides = {0, 0, 0, C, 1};

    SimpleDeviceMem x_device_buf(sizeof(XDataType) * xy_size);
    SimpleDeviceMem gamma_device_buf(sizeof(GammaDataType) * gamma_beta_size);
    SimpleDeviceMem beta_device_buf(sizeof(BetaDataType) * gamma_beta_size);
    SimpleDeviceMem y_device_buf(sizeof(YDataType) * xy_size);

    using DeviceOp = ck::tensor_operation::device::DeviceNormalization<XDataType,
                                                                       GammaDataType,
                                                                       BetaDataType,
                                                                       ComputeDataType,
                                                                       YDataType,
                                                                       Swish,
                                                                       Rank,
                                                                       NumReduceDim>;

    // get device op instances
    const auto op_ptrs = ck::tensor_operation::device::instance::DeviceOperationInstanceFactory<
        DeviceOp>::GetInstances();

    std::cout << "found " << op_ptrs.size() << " instances" << std::endl;

    const auto& generic_op_ptr = op_ptrs[0];

    auto generic_argument_ptr =
        generic_op_ptr->MakeArgumentPointer({N, H, W, G, C},    // lengths
                                            xy_strides,         // xStrides
                                            gamma_beta_strides, // gammaStrides
                                            gamma_beta_strides, // betaStrides
                                            xy_strides,         // yStrides
                                            {1, 2, 4},          // reduceDims
                                            1e-6,
                                            x_device_buf.GetDeviceBuffer(),
                                            gamma_device_buf.GetDeviceBuffer(),
                                            beta_device_buf.GetDeviceBuffer(),
                                            y_device_buf.GetDeviceBuffer(),
                                            nullptr,
                                            nullptr,
                                            Swish{});

    if(!generic_op_ptr->IsSupportedArgument(generic_argument_ptr.get()))
    {
        throw std::runtime_error(
            "The generic kernel instance should be able to support any input shapes");
    };

    std::string best_op_name;
    bool found            = false;
    int best_op_id        = -1;
    float best_ave_time   = std::numeric_limits<float>::max();
    float best_gb_per_sec = 0;

    // profile device operation instances
    std::cout << "Run all instances and do timing" << std::endl;

    for(int i = 0; i < op_ptrs.size(); ++i)
    {
        auto& op_ptr      = op_ptrs[i];
        auto argument_ptr = op_ptr->MakeArgumentPointer({N, H, W, G, C},    // lengths
                                                        xy_strides,         // xStrides
                                                        gamma_beta_strides, // gammaStrides
                                                        gamma_beta_strides, // betaStrides
                                                        xy_strides,         // yStrides
                                                        {1, 2, 4},          // reduceDims
                                                        1e-6,
                                                        x_device_buf.GetDeviceBuffer(),
                                                        gamma_device_buf.GetDeviceBuffer(),
                                                        beta_device_buf.GetDeviceBuffer(),
                                                        y_device_buf.GetDeviceBuffer(),
                                                        nullptr,
                                                        nullptr,
                                                        Swish{});

        auto invoker_ptr = op_ptr->MakeInvokerPointer();

        std::string op_name = op_ptr->GetTypeString();

        if(op_ptr->IsSupportedArgument(argument_ptr.get()))
        {
            float ave_time = invoker_ptr->Run(argument_ptr.get(), StreamConfig{nullptr, true});

            std::size_t num_byte =
                sizeof(XDataType) * xy_size + sizeof(GammaDataType) * gamma_beta_size +
                sizeof(BetaDataType) * gamma_beta_size + sizeof(YDataType) * xy_size;

            float gb_per_sec = num_byte / 1.E6 / ave_time;

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

    // run the best intance
    if(found)
    {
        std::cout << "Best Perf: " << best_ave_time << " ms, " << best_gb_per_sec << " GB/s, "
                  << best_op_name << std::endl;

        auto& op_ptr = op_ptrs[best_op_id];
        std::cout << "Run the best instance without timing: " << op_ptr->GetTypeString()
                  << std::endl;

        auto argument_ptr = op_ptr->MakeArgumentPointer({N, H, W, G, C},    // lengths
                                                        xy_strides,         // xStrides
                                                        gamma_beta_strides, // gammaStrides
                                                        gamma_beta_strides, // betaStrides
                                                        xy_strides,         // yStrides
                                                        {1, 2, 4},          // reduceDims
                                                        1e-6,
                                                        x_device_buf.GetDeviceBuffer(),
                                                        gamma_device_buf.GetDeviceBuffer(),
                                                        beta_device_buf.GetDeviceBuffer(),
                                                        y_device_buf.GetDeviceBuffer(),
                                                        nullptr,
                                                        nullptr,
                                                        Swish{});

        auto invoker_ptr = op_ptr->MakeInvokerPointer();

        if(op_ptr->IsSupportedArgument(argument_ptr.get()))
        {
            invoker_ptr->Run(argument_ptr.get(), StreamConfig{nullptr, false});
        }

        std::cout << "Done" << std::endl;
    }

    return 0;
}
