// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2023, Advanced Micro Devices, Inc. All rights reserved.

#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <numeric>

#include "ck/ck.hpp"
#include "ck/library/tensor_operation_instance/gpu/grouped_convolution_backward_weight.hpp"
#include "ck/tensor_operation/gpu/device/tensor_layout.hpp"
#include "ck/tensor_operation/gpu/device/device_conv_fwd.hpp"
#include "ck/tensor_operation/gpu/element/element_wise_operation.hpp"

using PassThrough = ck::tensor_operation::element_wise::PassThrough;

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

template <ck::index_t NumDimSpatial>
std::size_t GetFlops(ck::index_t G,
                     ck::index_t N,
                     ck::index_t K,
                     ck::index_t C,
                     const std::array<ck::index_t, NumDimSpatial>& output_spatial_lengths,
                     const std::array<ck::index_t, NumDimSpatial>& filter_spatial_lengths)
{
    // 2 * G * N * K * C * <output spatial lengths product> * <filter spatial lengths product>
    return static_cast<std::size_t>(2) * G * N * K * C *
           std::accumulate(std::begin(output_spatial_lengths),
                           std::end(output_spatial_lengths),
                           static_cast<std::size_t>(1),
                           std::multiplies<>()) *
           std::accumulate(std::begin(filter_spatial_lengths),
                           std::end(filter_spatial_lengths),
                           static_cast<std::size_t>(1),
                           std::multiplies<>());
}

template <typename InDataType, ck::index_t NumDimSpatial>
std::size_t GetInputByte(ck::index_t G,
                         ck::index_t N,
                         ck::index_t C,
                         const std::array<ck::index_t, NumDimSpatial>& input_spatial_lengths)
{
    // sizeof(InDataType) * (G * N * C * <input spatial lengths product>) +
    return sizeof(InDataType) * (G * N * C *
                                 std::accumulate(std::begin(input_spatial_lengths),
                                                 std::end(input_spatial_lengths),
                                                 static_cast<std::size_t>(1),
                                                 std::multiplies<>()));
}

template <typename WeiDataType, ck::index_t NumDimSpatial>
std::size_t GetWeightByte(ck::index_t G,
                          ck::index_t K,
                          ck::index_t C,
                          const std::array<ck::index_t, NumDimSpatial>& filter_spatial_lengths)
{
    // sizeof(WeiDataType) * (G * K * C * <filter spatial lengths product>) +
    return sizeof(WeiDataType) * (G * K * C *
                                  std::accumulate(std::begin(filter_spatial_lengths),
                                                  std::end(filter_spatial_lengths),
                                                  static_cast<std::size_t>(1),
                                                  std::multiplies<>()));
}

template <typename OutDataType, ck::index_t NumDimSpatial>
std::size_t GetOutputByte(ck::index_t G,
                          ck::index_t N,
                          ck::index_t K,
                          const std::array<ck::index_t, NumDimSpatial>& output_spatial_lengths)
{
    // sizeof(OutDataType) * (G * N * K * <output spatial lengths product>);
    return sizeof(OutDataType) * (G * N * K *
                                  std::accumulate(std::begin(output_spatial_lengths),
                                                  std::end(output_spatial_lengths),
                                                  static_cast<std::size_t>(1),
                                                  std::multiplies<std::size_t>()));
}

template <ck::index_t NumDimSpatial,
          typename InDataType,
          typename WeiDataType,
          typename OutDataType,
          typename InLayout,
          typename WeiLayout,
          typename OutLayout>
bool run_grouped_conv_bwd_weight(
    const ck::index_t G,
    const ck::index_t N,
    const ck::index_t K,
    const ck::index_t C,
    const std::array<ck::index_t, NumDimSpatial>& input_spatial_lengths,
    const std::array<ck::index_t, NumDimSpatial>& filter_spatial_lengths,
    const std::array<ck::index_t, NumDimSpatial>& output_spatial_lengths,
    const std::array<ck::index_t, NumDimSpatial + 3>& input_strides,
    const std::array<ck::index_t, NumDimSpatial + 3>& output_strides,
    const std::array<ck::index_t, NumDimSpatial>& conv_filter_strides,
    const std::array<ck::index_t, NumDimSpatial>& conv_filter_dilations,
    const std::array<ck::index_t, NumDimSpatial>& input_left_pads,
    const std::array<ck::index_t, NumDimSpatial>& input_right_pads)
{

    ck::index_t split_k = 2;
    SimpleDeviceMem in(GetInputByte<InDataType, NumDimSpatial>(G, N, C, input_spatial_lengths));
    SimpleDeviceMem wei(GetWeightByte<WeiDataType, NumDimSpatial>(G, K, C, filter_spatial_lengths));
    SimpleDeviceMem out(GetOutputByte<OutDataType, NumDimSpatial>(G, N, K, output_spatial_lengths));

    using DeviceOp = ck::tensor_operation::device::DeviceGroupedConvBwdWeight<NumDimSpatial,
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
                                                        G,
                                                        N,
                                                        K,
                                                        C,
                                                        input_spatial_lengths,
                                                        filter_spatial_lengths,
                                                        output_spatial_lengths,
                                                        input_strides,
                                                        output_strides,
                                                        conv_filter_strides,
                                                        conv_filter_dilations,
                                                        input_left_pads,
                                                        input_right_pads,
                                                        PassThrough{},
                                                        PassThrough{},
                                                        PassThrough{},
                                                        split_k);
        auto invoker_ptr    = op_ptr->MakeInvokerPointer();
        std::string op_name = op_ptr->GetTypeString();

        if(op_ptr->IsSupportedArgument(argument_ptr.get()))
        {
            float avg_time = invoker_ptr->Run(argument_ptr.get(), StreamConfig{nullptr, true});

            std::size_t flop =
                GetFlops<NumDimSpatial>(G, N, K, C, output_spatial_lengths, filter_spatial_lengths);
            std::size_t num_bytes =
                GetInputByte<InDataType, NumDimSpatial>(G, N, C, input_spatial_lengths) +
                GetWeightByte<WeiDataType, NumDimSpatial>(G, K, C, filter_spatial_lengths) +
                GetOutputByte<OutDataType, NumDimSpatial>(G, N, K, output_spatial_lengths);

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
            std::cerr << op_name << " does not support this problem" << std::endl;
        }
    }

    if(best_op_id < 0)
    {
        std::cerr << "no suitable instance" << std::endl;
        return false;
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
                                                        G,
                                                        N,
                                                        K,
                                                        C,
                                                        input_spatial_lengths,
                                                        filter_spatial_lengths,
                                                        output_spatial_lengths,
                                                        input_strides,
                                                        output_strides,
                                                        conv_filter_strides,
                                                        conv_filter_dilations,
                                                        input_left_pads,
                                                        input_right_pads,
                                                        PassThrough{},
                                                        PassThrough{},
                                                        PassThrough{},
                                                        split_k);
        auto invoker_ptr  = op_ptr->MakeInvokerPointer();

        if(op_ptr->IsSupportedArgument(argument_ptr.get()))
        {
            invoker_ptr->Run(argument_ptr.get(), StreamConfig{nullptr, false});
        }

        std::cout << "Done" << std::endl;
    }

    return true;
}
