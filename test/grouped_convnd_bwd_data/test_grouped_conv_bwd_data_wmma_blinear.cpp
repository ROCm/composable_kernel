// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <algorithm>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <typeinfo>
#include <gtest/gtest.h>

#include "ck/ck.hpp"
#include "ck/tensor_operation/gpu/device/tensor_layout.hpp"

#include "ck/tensor_operation/gpu/element/element_wise_operation.hpp"

#include "ck/library/tensor_operation_instance/gpu/grouped_convolution_backward_data_bilinear.hpp"

#include "ck/library/utility/check_err.hpp"
#include "ck/library/utility/device_memory.hpp"
#include "ck/library/utility/host_tensor.hpp"
#include "ck/library/utility/host_tensor_generator.hpp"
#include "ck/library/utility/convolution_parameter.hpp"
#include "ck/library/utility/convolution_host_tensor_descriptor_helper.hpp"
#include "ck/library/reference_tensor_operation/cpu/reference_conv_bwd_data.hpp"

template <typename Tuple>
class TestGroupedConvndBwdDataWmma : public ::testing::Test
{
   using InDataType  = ck::half_t;
using WeiDataType = ck::half_t;
using OutDataType = ck::half_t;
using ComputeDataType = InDataType;
using AccDataType = float;




using InLayout    = ck::tensor_layout::convolution::NDHWGC;
using WeiLayout   = ck::tensor_layout::convolution::GKZYXC;
using OutLayout   = ck::tensor_layout::convolution::NDHWGK;
using PassThrough = ck::tensor_operation::element_wise::PassThrough;
using Bilinear = ck::tensor_operation::element_wise::Bilinear;

static constexpr ck::index_t NumDimSpatial = 3;
static constexpr ck::index_t G             = 2;
static constexpr ck::index_t N             = 16;
static constexpr ck::index_t K             = 16;
static constexpr ck::index_t C             = 16;
static constexpr ck::index_t Z             = 3;
static constexpr ck::index_t Y             = 3;
static constexpr ck::index_t X             = 3;
static constexpr ck::index_t Di            = 14;
static constexpr ck::index_t Hi            = 14;
static constexpr ck::index_t Wi            = 14;
static constexpr ck::index_t Do            = 14;
static constexpr ck::index_t Ho            = 14;
static constexpr ck::index_t Wo            = 14;


public: std::vector<ck::utils::conv::ConvParam> conv_param;
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

  bool PerformBwdData()
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

    using DeviceOp =
        ck::tensor_operation::device::DeviceGroupedConvBwdDataMultipleD<NumDimSpatial,
                                                                        OutLayout,
                                                                        WeiLayout,
                                                                        ck::Tuple<InLayout>,
                                                                        InLayout,
                                                                        OutDataType,
                                                                        WeiDataType,
                                                                        ck::Tuple<InDataType>,
                                                                        InDataType,
                                                                        PassThrough,
                                                                        PassThrough,
                                                                        Bilinear>;

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

    for(size_t i = 0; i < op_ptrs.size(); ++i)
    {
        auto& op_ptr        = op_ptrs[i];
        auto argument_ptr   = op_ptr->MakeArgumentPointer(out.GetDeviceBuffer(),
                                                        wei.GetDeviceBuffer(),
                                                          {in.GetDeviceBuffer()},
                                                        in.GetDeviceBuffer(),
                                                        out_lengths,
                                                        out_strides,
                                                        wei_lengths,
                                                        wei_strides,
                                                          {in_lengths},
                                                          {in_strides},
                                                        in_lengths,
                                                        in_strides,
                                                        filter_strides,
                                                        filter_dilations,
                                                        input_left_pads,
                                                        input_right_pads,
                                                        PassThrough{},
                                                        PassThrough{},
                                                        Bilinear{2.f, 2.f});
        auto invoker_ptr    = op_ptr->MakeInvokerPointer();
        std::string op_name = op_ptr->GetTypeString();

        if(op_ptr->IsSupportedArgument(argument_ptr.get()))
        {
            float avg_time = invoker_ptr->Run(argument_ptr.get(), StreamConfig{nullptr, true});

            std::size_t flop = std::size_t(2) * G * N * K * C * Do * Ho * Wo * Y * X +
                               3 * G * N * Di * Hi * Wi * C;
            std::size_t num_bytes = 2 * sizeof(InDataType) * G * N * Di * Hi * Wi * C +
                                    sizeof(WeiDataType) * G * K * Z * Y * X * C +
                                    sizeof(OutDataType) * G * N * Do * Ho * Wo * K;

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
        auto argument_ptr = op_ptr->MakeArgumentPointer(out.GetDeviceBuffer(),
                                                        wei.GetDeviceBuffer(),
                                                        {in.GetDeviceBuffer()},
                                                        in.GetDeviceBuffer(),
                                                        out_lengths,
                                                        out_strides,
                                                        wei_lengths,
                                                        wei_strides,
                                                        {in_lengths},
                                                        {in_strides},
                                                        in_lengths,
                                                        in_strides,
                                                        filter_strides,
                                                        filter_dilations,
                                                        input_left_pads,
                                                        input_right_pads,
                                                        PassThrough{},
                                                        PassThrough{},
                                                        Bilinear{2.f, 2.f});

        auto invoker_ptr = op_ptr->MakeInvokerPointer();

        if(op_ptr->IsSupportedArgument(argument_ptr.get()))
        {
            invoker_ptr->Run(argument_ptr.get(), StreamConfig{nullptr, false});
        }

        std::cout << "Done" << std::endl;
    


       bool passed =op_ptr->IsSupportedArgument(argument_ptr.get());

        std::cout << "passed  value is:" << passed << std::endl;
        return passed;
       
        
    }
    
}

     public: bool Run()
    {
      
        bool pass = true;
   pass        = pass && PerformBwdData();
            std::cout<< "pass value is :" << pass << std::endl;
        EXPECT_TRUE(pass);
        return pass;
    
    }

};

using namespace ck::tensor_layout::convolution;


using GNDHWC = ck::tensor_layout::convolution::GNDHWC;
using NDHWGC = ck::tensor_layout::convolution::NDHWGC;

using GKZYXC = ck::tensor_layout::convolution::GKZYXC;

using GNDHWK = ck::tensor_layout::convolution::GNDHWK;
using NDHWGK = ck::tensor_layout::convolution::NDHWGK;
using NGKDHW = ck::tensor_layout::convolution::NGKDHW;

using NGCDHW = ck::tensor_layout::convolution::NGCDHW;

using KernelTypes3d = ::testing::Types<std::tuple<float, GNDHWK, GKZYXC, GNDHWC>,
                                       std::tuple<ck::half_t, GNDHWK, GKZYXC, GNDHWC>,
                                       std::tuple<ck::bhalf_t, GNDHWK, GKZYXC, GNDHWC>,
                                       std::tuple<float, NGKDHW, GKZYXC, NGCDHW>,
                                       std::tuple<ck::half_t, NGKDHW, GKZYXC, NGCDHW>,
                                       std::tuple<ck::bhalf_t, NGKDHW, GKZYXC, NGCDHW>,
                                       std::tuple<float, NGKDHW, GKCZYX, NGCDHW>,
                                       std::tuple<ck::half_t, NGKDHW, GKCZYX, NGCDHW>,
                                       std::tuple<ck::bhalf_t, NGKDHW, GKCZYX, NGCDHW>,
                                       std::tuple<float, NDHWGK, GKZYXC, NDHWGC>,
                                       std::tuple<ck::half_t, NDHWGK, GKZYXC, NDHWGC>,
                                       std::tuple<ck::bhalf_t, NDHWGK, GKZYXC, NDHWGC>>;
        
template <typename Tuple>
class TestGroupedConvndBwdDataWmma3d : public TestGroupedConvndBwdDataWmma<Tuple>
{
};

TYPED_TEST_SUITE(TestGroupedConvndBwdDataWmma3d, KernelTypes3d);

TYPED_TEST(TestGroupedConvndBwdDataWmma3d, Test3D)
{

    this->conv_param.push_back(
        {3, 2, 16, 128, 256, {1, 1, 1}, {7, 7, 7}, {2, 2, 2}, {1, 1, 1}, {0, 0, 0}, {0, 0, 0}});
            this->conv_param.push_back(
        {3, 2, 2, 128, 256, {3, 3, 3}, {14, 14, 3}, {1, 1, 1}, {1, 1, 1}, {1, 1, 1}, {1, 1, 1}});
    this->conv_param.push_back(
        {3, 2, 32, 128, 256, {1, 1, 1}, {3, 3, 3}, {1, 1, 1}, {1, 1, 1}, {0, 0, 0}, {0, 0, 0}});
    this->conv_param.push_back(
        {3, 1, 1, 1, 32, {3, 3, 3}, {32, 32, 32}, {1, 1, 1}, {1, 1, 1}, {1, 1, 1}, {1, 1, 1}});
    this->conv_param.push_back(
        {3, 1, 1, 64, 3, {3, 3, 3}, {32, 32, 32}, {1, 1, 1}, {1, 1, 1}, {1, 1, 1}, {1, 1, 1}});
    this->conv_param.push_back(
        {3, 1, 1, 1, 1, {3, 3, 3}, {32, 32, 32}, {1, 1, 1}, {1, 1, 1}, {1, 1, 1}, {1, 1, 1}});
   this-> Run();


}

int main(int argc, char** argv)
{
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}

