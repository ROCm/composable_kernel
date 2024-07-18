// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2023, Advanced Micro Devices, Inc. All rights reserved.

#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <numeric>
#include <vector>

#include "ck/ck.hpp"
#include "ck/library/tensor_operation_instance/gpu/grouped_convolution_forward.hpp"
#include "ck/tensor_operation/gpu/device/tensor_layout.hpp"
#include "ck/tensor_operation/gpu/element/element_wise_operation.hpp"

using InDataType  = ck::half_t;
using WeiDataType = ck::half_t;
using OutDataType = ck::half_t;

using InLayout    = ck::tensor_layout::convolution::GNWC;
using WeiLayout   = ck::tensor_layout::convolution::GKXC;
using OutLayout   = ck::tensor_layout::convolution::GNWK;
using PassThrough = ck::tensor_operation::element_wise::PassThrough;

static constexpr ck::index_t NumDimSpatial = 1;
static constexpr ck::index_t G             = 32;
static constexpr ck::index_t N             = 256;
static constexpr ck::index_t K             = 192;
static constexpr ck::index_t C             = 192;
static constexpr ck::index_t X             = 3;
static constexpr ck::index_t Wi            = 28;
static constexpr ck::index_t Wo            = 28;

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
    std::array<ck::index_t, NumDimSpatial + 3> in_lengths{G, N, Wi, C};
    std::array<ck::index_t, NumDimSpatial + 3> in_strides{0, 0, 0, 1};

    std::array<ck::index_t, NumDimSpatial + 3> wei_lengths{G, K, X, C};
    std::array<ck::index_t, NumDimSpatial + 3> wei_strides{0, 0, 0, 1};

    std::array<ck::index_t, NumDimSpatial + 3> out_lengths{G, N, Wo, K};
    std::array<ck::index_t, NumDimSpatial + 3> out_strides{0, 0, 0, 1};

    std::partial_sum(rbegin(in_lengths),
                     std::prev(rend(in_lengths)),
                     std::next(rbegin(in_strides)),
                     std::multiplies<>{});
    std::partial_sum(rbegin(wei_lengths),
                     std::prev(rend(wei_lengths)),
                     std::next(rbegin(wei_strides)),
                     std::multiplies<>{});
    std::partial_sum(rbegin(out_lengths),
                     std::prev(rend(out_lengths)),
                     std::next(rbegin(out_strides)),
                     std::multiplies<>{});

    // transpose GNWC/GKXC/GNWK to GNCW/GKCX/GNCW
    std::rotate(rbegin(in_lengths),
                std::next(rbegin(in_lengths)),
                std::next(rbegin(in_lengths), NumDimSpatial + 1));
    std::rotate(rbegin(in_strides),
                std::next(rbegin(in_strides)),
                std::next(rbegin(in_strides), NumDimSpatial + 1));
    std::rotate(rbegin(wei_lengths),
                std::next(rbegin(wei_lengths)),
                std::next(rbegin(wei_lengths), NumDimSpatial + 1));
    std::rotate(rbegin(wei_strides),
                std::next(rbegin(wei_strides)),
                std::next(rbegin(wei_strides), NumDimSpatial + 1));
    std::rotate(rbegin(out_lengths),
                std::next(rbegin(out_lengths)),
                std::next(rbegin(out_lengths), NumDimSpatial + 1));
    std::rotate(rbegin(out_strides),
                std::next(rbegin(out_strides)),
                std::next(rbegin(out_strides), NumDimSpatial + 1));

    std::array<ck::index_t, NumDimSpatial> filter_strides{1};
    std::array<ck::index_t, NumDimSpatial> filter_dilations{1};
    std::array<ck::index_t, NumDimSpatial> input_left_pads{1};
    std::array<ck::index_t, NumDimSpatial> input_right_pads{1};

    SimpleDeviceMem in(sizeof(InDataType) * G * N * Wi * C);
    SimpleDeviceMem wei(sizeof(WeiDataType) * G * K * X * C);
    SimpleDeviceMem out(sizeof(OutDataType) * G * N * Wo * K);

    using DeviceOp = ck::tensor_operation::device::DeviceGroupedConvFwdMultipleD<NumDimSpatial,
                                                                                 InLayout,
                                                                                 WeiLayout,
                                                                                 ck::Tuple<>,
                                                                                 OutLayout,
                                                                                 InDataType,
                                                                                 WeiDataType,
                                                                                 ck::Tuple<>,
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
                                                        {},
                                                        out.GetDeviceBuffer(),
                                                        in_lengths,
                                                        in_strides,
                                                        wei_lengths,
                                                        wei_strides,
                                                        {},
                                                        {},
                                                        out_lengths,
                                                        out_strides,
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

            std::size_t flop      = std::size_t(2) * G * N * K * C * Wo * X;
            std::size_t num_bytes = sizeof(InDataType) * G * N * Wi * C +
                                    sizeof(WeiDataType) * G * K * X * C +
                                    sizeof(OutDataType) * G * N * Wo * K;

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
        return EXIT_FAILURE;
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
                                                        {},
                                                        out.GetDeviceBuffer(),
                                                        in_lengths,
                                                        in_strides,
                                                        wei_lengths,
                                                        wei_strides,
                                                        {},
                                                        {},
                                                        out_lengths,
                                                        out_strides,
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
}
