// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2022, Advanced Micro Devices, Inc. All rights reserved.

#include <iomanip>
#include <iostream>
#include <vector>

#include "ck/ck.hpp"
#include "ck/library/tensor_operation_instance/gpu/convolution_forward.hpp"
#include "ck/tensor_operation/gpu/device/tensor_layout.hpp"
#include "ck/tensor_operation/gpu/device/device_conv_fwd.hpp"
#include "ck/tensor_operation/gpu/element/element_wise_operation.hpp"

using InDataType  = ck::half_t;
using WeiDataType = ck::half_t;
using OutDataType = ck::half_t;

using InLayout    = ck::tensor_layout::convolution::NHWC;
using WeiLayout   = ck::tensor_layout::convolution::KYXC;
using OutLayout   = ck::tensor_layout::convolution::NHWK;
using PassThrough = ck::tensor_operation::element_wise::PassThrough;

static constexpr ck::index_t NumDimSpatial = 2;
static constexpr ck::index_t N             = 16;
static constexpr ck::index_t K             = 32;
static constexpr ck::index_t C             = 3;
static constexpr ck::index_t Y             = 3;
static constexpr ck::index_t X             = 3;
static constexpr ck::index_t Hi            = 224;
static constexpr ck::index_t Wi            = 224;
static constexpr ck::index_t Ho            = 113;
static constexpr ck::index_t Wo            = 113;

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
    std::vector<ck::index_t> in_spatial_lengths{Hi, Wi};
    std::vector<ck::index_t> filter_spatial_lengths{Y, X};
    std::vector<ck::index_t> out_spatial_lengths{Ho, Wo};
    std::vector<ck::index_t> filter_strides{2, 2};
    std::vector<ck::index_t> filter_dilations{1, 1};
    std::vector<ck::index_t> input_left_pads{2, 2};
    std::vector<ck::index_t> input_right_pads{2, 2};

    SimpleDeviceMem in(sizeof(InDataType) * N * Hi * Wi * C);
    SimpleDeviceMem wei(sizeof(WeiDataType) * K * Y * X * C);
    SimpleDeviceMem out(sizeof(OutDataType) * N * Ho * Wo * K);

    using DeviceOp = ck::tensor_operation::device::DeviceConvFwd<NumDimSpatial,
                                                                 InLayout,
                                                                 WeiLayout,
                                                                 OutLayout,
                                                                 InDataType,
                                                                 WeiDataType,
                                                                 OutDataType,
                                                                 PassThrough,
                                                                 PassThrough,
                                                                 PassThrough>;
    // get device op instances
    const auto op_ptrs = ck::tensor_operation::device::instance::DeviceOperationInstanceFactory<
        DeviceOp>::GetInstances();

    std::cout << "found " << op_ptrs.size() << " instances" << std::endl;

    std::string best_op_name;
    int best_op_id        = -1;
    float best_avg_time   = std::numeric_limits<float>::max();
    float best_gb_per_sec = 0;
    float best_tflops     = 0;

    // profile device operation instances
    std::cout << "Run all instances and do timing" << std::endl;

    for(int i = 0; i < op_ptrs.size(); ++i)
    {
        auto& op_ptr        = op_ptrs[i];
        auto argument_ptr   = op_ptr->MakeArgumentPointer(in.GetDeviceBuffer(),
                                                        wei.GetDeviceBuffer(),
                                                        out.GetDeviceBuffer(),
                                                        N,
                                                        K,
                                                        C,
                                                        in_spatial_lengths,
                                                        filter_spatial_lengths,
                                                        out_spatial_lengths,
                                                        filter_strides,
                                                        filter_dilations,
                                                        input_left_pads,
                                                        input_right_pads,
                                                        PassThrough{},
                                                        PassThrough{},
                                                        PassThrough{});
        auto invoker_ptr    = op_ptr->MakeInvokerPointer();
        std::string op_name = op_ptr->GetTypeString();

        if(op_ptr->IsSupportedArgument(argument_ptr.get()))
        {
            float avg_time = invoker_ptr->Run(argument_ptr.get(), StreamConfig{nullptr, true});

            std::size_t flop      = 2 * N * K * C * Ho * Wo * Y * X;
            std::size_t num_bytes = sizeof(InDataType) * N * Hi * Wi * C +
                                    sizeof(WeiDataType) * K * Y * X * C +
                                    sizeof(OutDataType) * N * Ho * Wo * K;

            float tflops     = static_cast<float>(flop) / 1.E9 / avg_time;
            float gb_per_sec = num_bytes / 1.E6 / avg_time;

            std::cout << "Perf: " << std::setw(10) << avg_time << " ms, " << tflops << " TFlops, "
                      << gb_per_sec << " GB/s, " << op_name << std::endl;

            if(tflops > best_tflops)
            {
                best_op_id      = i;
                best_op_name    = op_name;
                best_avg_time   = avg_time;
                best_gb_per_sec = gb_per_sec;
                best_tflops     = tflops;
            }
        }
        else
        {
            std::cout << op_name << " does not support this problem" << std::endl;
        }
    }

    std::cout << "Best Perf: " << std::setw(10) << best_avg_time << " ms, " << best_tflops
              << " TFlops, " << best_gb_per_sec << " GB/s, " << best_op_name << std::endl;

    // run the best intance
    {
        auto& op_ptr = op_ptrs[best_op_id];
        std::cout << "Run the best instance without timing: " << op_ptr->GetTypeString()
                  << std::endl;
        auto argument_ptr = op_ptr->MakeArgumentPointer(in.GetDeviceBuffer(),
                                                        wei.GetDeviceBuffer(),
                                                        out.GetDeviceBuffer(),
                                                        N,
                                                        K,
                                                        C,
                                                        in_spatial_lengths,
                                                        filter_spatial_lengths,
                                                        out_spatial_lengths,
                                                        filter_strides,
                                                        filter_dilations,
                                                        input_left_pads,
                                                        input_right_pads,
                                                        PassThrough{},
                                                        PassThrough{},
                                                        PassThrough{});

        auto invoker_ptr = op_ptr->MakeInvokerPointer();

        if(op_ptr->IsSupportedArgument(argument_ptr.get()))
        {
            invoker_ptr->Run(argument_ptr.get(), StreamConfig{nullptr, false});
        }

        std::cout << "Done" << std::endl;
    }

    return 0;
}