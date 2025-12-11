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


using ::ck::DeviceMem;
using ::ck::HostTensorDescriptor;
using ::ck::Tensor;

template <typename Tuple>
class TestGroupedConvndBwdData : public ::testing::Test
{
    protected:
    using F16 = ck::half_t;
    using InDataType  = std::tuple_element_t<0, Tuple>;;
     using WeiDataType  =std::tuple_element_t<0, Tuple>;;
      using OutDataType  = std::tuple_element_t<0, Tuple>;;
      using ComputeDataType = InDataType;
using InLayout    = std::tuple_element_t<3, Tuple>;;
using WeiLayout   = std::tuple_element_t<2, Tuple>;;
using OutLayout   = std::tuple_element_t<1, Tuple>;;
    using WeiElementOp  = ck::tensor_operation::element_wise::PassThrough;
    using InElementOp = ck::tensor_operation::element_wise::Bilinear;
    using OutElementOp = ck::tensor_operation::element_wise::PassThrough;
    using PassThrough = ck::tensor_operation::element_wise::PassThrough;

    using Bilinear = ck::tensor_operation::element_wise::Bilinear;
    static constexpr ck::index_t NDimSpatial = 3;
    static constexpr float alpha             = 2.f;
     static constexpr float beta             = 2.f;


    std::vector<ck::utils::conv::ConvParam> conv_params;
    std::vector<ck::index_t> split_ks{1};

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
   
    bool PerformConvDataScale(ck::utils::conv::ConvParam& conv_param, const ck::index_t split_k)
    {
        bool passed = true;
      
           const auto out_g_n_k_wos_desc =
            ck::utils::conv::make_output_host_tensor_descriptor_g_n_k_wos_packed<OutLayout>(
                conv_param);

        const auto wei_g_k_c_xs_desc =
            ck::utils::conv::make_weight_host_tensor_descriptor_g_k_c_xs_packed<WeiLayout>(
                conv_param);

        const auto in_g_n_c_wis_desc =
            ck::utils::conv::make_input_host_tensor_descriptor_g_n_c_wis_packed<InLayout>(
                conv_param);

        Tensor<WeiDataType> wei(wei_g_k_c_xs_desc);
        Tensor<OutDataType> out(out_g_n_k_wos_desc);
        Tensor<InDataType> in_host(in_g_n_c_wis_desc);
        Tensor<InDataType> in_device(in_g_n_c_wis_desc);

        std::cout << "in: " << in_host.mDesc << std::endl;
        std::cout << "wei: " << wei.mDesc << std::endl;
        std::cout << "out: " << out.mDesc << std::endl;

        SimpleDeviceMem in_device_buf(sizeof(InDataType)* in_host.mDesc.GetElementSpaceSize());
        SimpleDeviceMem out_device_buf(sizeof(OutDataType)* out.mDesc.GetElementSpaceSize());
        SimpleDeviceMem wei_device_buf(sizeof(WeiDataType) * wei.mDesc.GetElementSpaceSize());

       std::array<ck::index_t, NDimSpatial + 3> out_lengths{};
        std::array<ck::index_t, NDimSpatial + 3> out_strides{};
        std::array<ck::index_t, NDimSpatial + 3> wei_lengths{};
        std::array<ck::index_t, NDimSpatial + 3> wei_strides{};
        std::array<ck::index_t, NDimSpatial + 3> in_lengths{};
        std::array<ck::index_t, NDimSpatial + 3> in_strides{};
        std::array<ck::index_t, NDimSpatial> conv_filter_strides{};
        std::array<ck::index_t, NDimSpatial> conv_filter_dilations{};
        std::array<ck::index_t, NDimSpatial> input_left_pads{};
        std::array<ck::index_t, NDimSpatial> input_right_pads{};
      
        
        auto copy = [](const auto& x, auto& y) { ck::ranges::copy(x, y.begin()); };

        copy(out_g_n_k_wos_desc.GetLengths(), out_lengths);
        copy(out_g_n_k_wos_desc.GetStrides(), out_strides);
        copy(wei_g_k_c_xs_desc.GetLengths(), wei_lengths);
        copy(wei_g_k_c_xs_desc.GetStrides(), wei_strides);
        copy(in_g_n_c_wis_desc.GetLengths(), in_lengths);
        copy(in_g_n_c_wis_desc.GetStrides(), in_strides);
        copy(conv_param.conv_filter_strides_, conv_filter_strides);
        copy(conv_param.conv_filter_dilations_, conv_filter_dilations);
        copy(conv_param.input_left_pads_, input_left_pads);
        copy(conv_param.input_right_pads_, input_right_pads);
   
    using DeviceOp = ck::tensor_operation::device::DeviceGroupedConvBwdDataMultipleD<NDimSpatial,
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

        int num_kernel = 0;

        for(std::size_t i = 0; i < op_ptrs.size(); ++i)
        {
            auto& op_ptr      = op_ptrs[i];
            auto argument_ptr = op_ptr->MakeArgumentPointer(out_device_buf.GetDeviceBuffer(),
                                                        wei_device_buf.GetDeviceBuffer(),
                                                        {in_device_buf.GetDeviceBuffer()},
                                                        in_device_buf.GetDeviceBuffer(),
                                                        out_lengths,
                                                        out_strides,
                                                        wei_lengths,
                                                        wei_strides,
                                                          {in_lengths},
                                                          {in_strides},
                                                        in_lengths,
                                                        in_strides,
                                                        conv_filter_strides,
                                                        conv_filter_dilations,
                                                        input_left_pads,
                                                        input_right_pads,
                                                        PassThrough{},
                                                        PassThrough{},
                                                        Bilinear{alpha,beta});

            DeviceMem workspace_buf(op_ptr->GetWorkSpaceSize(argument_ptr.get()));
            op_ptr->SetWorkSpacePointer(argument_ptr.get(), workspace_buf.GetDeviceBuffer());

            auto invoker_ptr    = op_ptr->MakeInvokerPointer();
            std::string op_name = op_ptr->GetTypeString();

            if(op_ptr->IsSupportedArgument(argument_ptr.get()))
            {
                num_kernel++;
                float avg_time = invoker_ptr->Run(argument_ptr.get(), StreamConfig{nullptr, true});
               // wei_device_buf.FromDevice(in_device.mData.data());

                using AccDataType = float;
                float max_accumulated_value =
                    *std::max_element(in_host.mData.begin(), in_host.mData.end());

                const ck::index_t num_accums         = out.GetElementSize() / conv_param.K_;
                const ck::index_t num_accums_split_k = split_k;
                double rtol =
                    ck::utils::get_relative_threshold<InDataType, WeiDataType, AccDataType>(
                        num_accums / num_accums_split_k);
                double atol =
                    ck::utils::get_absolute_threshold<InDataType, WeiDataType, AccDataType>(
                        max_accumulated_value / num_accums_split_k,
                        num_accums / num_accums_split_k);

                // Calculate error due to split_k accumulation
                auto rtol_split_k =
                    ck::utils::get_relative_threshold<InDataType, InDataType, InDataType>(
                        num_accums_split_k);
                auto atol_split_k =
                    ck::utils::get_absolute_threshold<InDataType, InDataType, InDataType>(
                        max_accumulated_value, num_accums_split_k);
                // Use higher threshold
                rtol = std::max(rtol, rtol_split_k);
                atol = std::max(atol, atol_split_k);

                passed &= ck::utils::check_err(
                    in_device, in_host, "Error: incorrect results!", rtol, atol);

                std::size_t flop =
                    conv_param.GetFlops() +
                    3 * conv_param.GetOutputByte<InDataType>() / sizeof(InDataType);
                std::size_t num_bytes = conv_param.GetByte<InDataType, WeiDataType, OutDataType>() +
                                        conv_param.GetOutputByte<InDataType>();

                float tflops     = static_cast<float>(flop) / 1.E9 / avg_time;
                float gb_per_sec = num_bytes / 1.E6 / avg_time;

                std::cout << "Perf: " << std::setw(10) << avg_time << " ms, " << tflops
                          << " TFlops, " << gb_per_sec << " GB/s, " << op_name << std::endl;
            }
            else
            {
                std::cerr << op_name << " does not support this problem" << std::endl;
            }
        }

        printf("\033[36mvalids: %d\033[0m\n", num_kernel);
        return passed;
    }

    void Run()
    {
        EXPECT_FALSE(conv_params.empty());
        bool pass = true;

        for(auto split_k : split_ks)
        {
            for(auto& param : conv_params)
            {
                pass = pass && PerformConvDataScale(param, split_k);
            }
        }
        EXPECT_TRUE(pass);
    }
};

template <typename Tuple>
class TestGroupedConvndBwdData3d : public TestGroupedConvndBwdData<Tuple>
{
};

using NDHWGC = ck::tensor_layout::convolution::NDHWGC;
using GKZYXC = ck::tensor_layout::convolution::GKZYXC;
using NDHWGK = ck::tensor_layout::convolution::NDHWGK;


using KernelTypes3d = ::testing::Types<
                                       std::tuple<ck::half_t, NDHWGK, GKZYXC, NDHWGC>,
                                       std::tuple<ck::bhalf_t, NDHWGK, GKZYXC, NDHWGC>>;

TYPED_TEST_SUITE(TestGroupedConvndBwdData3d, KernelTypes3d);

TYPED_TEST(TestGroupedConvndBwdData3d, Test3D)
{
this->conv_params.push_back({3, 3, 16, 96, 96,   {1, 3, 3}, {2, 48, 32}, {1, 1, 1}, {1, 1, 1}, {0, 1, 1}, {0, 1, 1}});
this->conv_params.push_back({3, 1, 16, 288, 288, {2, 1, 1}, {2, 48, 32}, {2, 1, 1}, {1, 1, 1}, {0, 0, 0}, {0, 0, 0}});
this->conv_params.push_back({3, 3, 16, 96, 96,   {3, 1, 1}, {2, 48, 32}, {1, 1, 1}, {1, 1, 1}, {1, 0, 0}, {1, 0, 0}});
this->conv_params.push_back({3, 3, 16, 96, 96,   {1, 3, 3}, {4, 48, 32}, {1, 1, 1}, {1, 1, 1}, {0, 1, 1}, {0, 1, 1}});
this->conv_params.push_back({3, 1, 16, 288, 288, {2, 1, 1}, {4, 48, 32}, {2, 1, 1}, {1, 1, 1}, {0, 0, 0}, {0, 0, 0}});
this->conv_params.push_back({3, 3, 16, 96, 96,   {3, 1, 1}, {4, 48, 32}, {1, 1, 1}, {1, 1, 1}, {1, 0, 0}, {1, 0, 0}});
this->conv_params.push_back({3, 3, 16, 96, 96,   {1, 3, 3}, {8, 48, 32}, {1, 1, 1}, {1, 1, 1}, {0, 1, 1}, {0, 1, 1}});
this->conv_params.push_back({3, 1, 16, 288, 288, {2, 1, 1}, {8, 48, 32}, {2, 1, 1}, {1, 1, 1}, {0, 0, 0}, {0, 0, 0}});
this->conv_params.push_back({3, 3, 16, 96, 96,   {3, 1, 1}, {8, 48, 32}, {1, 1, 1}, {1, 1, 1}, {1, 0, 0}, {1, 0, 0}});
    this->Run();
}
