// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <numeric>
#include <vector>

#include "ck/ck.hpp"
#include "ck/library/tensor_operation_instance/gpu/grouped_convolution_backward_data.hpp"
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

int main()
{
    std::array<ck::index_t, NumDimSpatial + 3> in_lengths{G, N, C, Di, Hi, Wi};
    std::array<ck::index_t, NumDimSpatial + 3> in_strides{
        C, Di * Hi * Wi * G * C, 1, Hi * Wi * G * C, Wi * G * C, G * C};

    std::array<ck::index_t, NumDimSpatial + 3> wei_lengths{G, K, C, Z, Y, X};
    std::array<ck::index_t, NumDimSpatial + 3> wei_strides{
        K * Z * Y * X * C, Z * Y * X * C, 1, Y * X * C, X * C, C};

    std::array<ck::index_t, NumDimSpatial + 3> out_lengths{G, N, K, Do, Ho, Wo};
    std::array<ck::index_t, NumDimSpatial + 3> out_strides{
        K, Do * Ho * Wo * G * K, 1, Ho * Wo * G * K, Wo * G * K, G * K};

    std::array<ck::index_t, NumDimSpatial> filter_strides{1, 1, 1};
    std::array<ck::index_t, NumDimSpatial> filter_dilations{1, 1, 1};
    std::array<ck::index_t, NumDimSpatial> input_left_pads{1, 1, 1};
    std::array<ck::index_t, NumDimSpatial> input_right_pads{1, 1, 1};

    SimpleDeviceMem in(sizeof(InDataType) * G * N * Di * Hi * Wi * C);
    SimpleDeviceMem wei(sizeof(WeiDataType) * G * K * Z * Y * X * C);
    SimpleDeviceMem out(sizeof(OutDataType) * G * N * Do * Ho * Wo * K);

    // do conv
    auto conv     = DeviceConvInstance{};

    auto argument_ptr   = conv.MakeArgumentPointer(out.GetDeviceBuffer(),
                                                        wei.GetDeviceBuffer(),
                                                          {},
                                                        in.GetDeviceBuffer(),
                                                        out_lengths,
                                                        out_strides,
                                                        wei_lengths,
                                                        wei_strides,
                                                          {},
                                                          {},
                                                        in_lengths,
                                                        in_strides,
                                                        filter_strides,
                                                        filter_dilations,
                                                        input_left_pads,
                                                        input_right_pads,
                                                        PassThrough{},
                                                        PassThrough{},
                                                        PassThrough{});
        auto invoker_ptr    = conv.MakeInvokerPointer();
        std::string op_name = conv.GetTypeString();

        if(conv.IsSupportedArgument(argument_ptr.get()))
        {
            float avg_time = invoker_ptr->Run(argument_ptr.get(), StreamConfig{nullptr, true});

            std::size_t flop      = std::size_t(2) * G * N * K * C * Do * Ho * Wo * Y * X;
            std::size_t num_bytes = sizeof(InDataType) * G * N * Di * Hi * Wi * C +
                                    sizeof(WeiDataType) * G * K * Z * Y * X * C +
                                    sizeof(OutDataType) * G * N * Do * Ho * Wo * K;

            float tflops     = static_cast<float>(flop) / 1.E9 / avg_time;
            float gb_per_sec = num_bytes / 1.E6 / avg_time;

            std::cout << "Perf: " << std::setw(10) << avg_time << " ms, " << tflops << " TFlops, "
                      << gb_per_sec << " GB/s, " << op_name << std::endl;
        }
        else
        {
            std::cerr << op_name << " does not support this problem" << std::endl;
        }
    
}
