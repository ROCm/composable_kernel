// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2023, Advanced Micro Devices, Inc. All rights reserved.

#include <cstdlib>
#include <initializer_list>
#include <iostream>
#include <numeric>

#include "ck/ck.hpp"
#include "ck/tensor_operation/gpu/device/convolution_backward_data_specialization.hpp"
#include "ck/tensor_operation/gpu/device/tensor_layout.hpp"
#include "ck/tensor_operation/gpu/element/element_wise_operation.hpp"

#include "ck/library/reference_tensor_operation/cpu/reference_avgpool_bwd.hpp"
#include "ck/library/utility/algorithm.hpp"
#include "ck/library/utility/check_err.hpp"
#include "ck/library/utility/convolution_host_tensor_descriptor_helper.hpp"
#include "ck/library/utility/convolution_parameter.hpp"
#include "ck/library/utility/device_memory.hpp"
#include "ck/library/utility/host_tensor.hpp"
#include "ck/library/utility/host_tensor_generator.hpp"

using DOutDataType = float;
using DInDataType  = float;

using PassThrough = ck::tensor_operation::element_wise::PassThrough;

bool pool3d_bwd_test(bool do_verification,
                     bool time_kernel,
                     ck::index_t N,
                     ck::index_t C,
                     ck::index_t Di,
                     ck::index_t Hi,
                     ck::index_t Wi,
                     std::vector<ck::index_t> window_lengths,
                     std::vector<ck::index_t> window_strides,
                     std::vector<ck::index_t> window_dilations,
                     std::vector<ck::index_t> dinput_left_pads,
                     std::vector<ck::index_t> dinput_right_pads)
{
    auto OutSpatialLength = [&](auto InSpatialLength, int index) {
        ck::index_t left_pad   = dinput_left_pads[index];
        ck::index_t right_pad  = dinput_right_pads[index];
        ck::index_t window_len = window_lengths[index];
        ck::index_t stride     = window_strides[index];
        return (InSpatialLength + left_pad + right_pad - window_len) / stride + 1;
    };

    ck::index_t Do = OutSpatialLength(Di, 0);
    ck::index_t Ho = OutSpatialLength(Hi, 1);
    ck::index_t Wo = OutSpatialLength(Wi, 1);

    Tensor<DOutDataType> dout(HostTensorDescriptor({N, C, Do, Ho, Wo}));
    Tensor<DInDataType> din_dev(HostTensorDescriptor({N, C, Di, Hi, Wi}));
    Tensor<DInDataType> din_host(HostTensorDescriptor({N, C, Di, Hi, Wi}));

    std::cout << "dout: " << dout.mDesc << std::endl;
    std::cout << "din_host: " << din_host.mDesc << std::endl;

    dout.GenerateTensorValue(GeneratorTensor_3<DOutDataType>{0.0, 1.0});

    if(do_verification)
    {
        auto ref_pool =
            ck::tensor_operation::host::ReferenceAvgPoolBwd<3, DInDataType, DOutDataType>();

        auto ref_invoker = ref_pool.MakeInvoker();

        auto ref_argument = ref_pool.MakeArgument(din_host,
                                                  dout,
                                                  window_lengths,
                                                  window_strides,
                                                  window_dilations,
                                                  dinput_left_pads,
                                                  dinput_right_pads);

        ref_invoker.Run(ref_argument);
    }

    // TODO - full example
    ck::ignore = time_kernel;
    return 0;
}

int main()
{
    std::vector<ck::index_t> window_lengths    = {5, 5, 5};
    std::vector<ck::index_t> window_strides    = {2, 2, 2};
    std::vector<ck::index_t> window_dilations  = {1, 1, 1};
    std::vector<ck::index_t> dinput_left_pads  = {0, 0, 0};
    std::vector<ck::index_t> dinput_right_pads = {0, 0, 0};

    ck::index_t N  = 1;
    ck::index_t C  = 16;
    ck::index_t Di = 40;
    ck::index_t Hi = 40;
    ck::index_t Wi = 40;

    pool3d_bwd_test(true,
                    false,
                    N,
                    C,
                    Di,
                    Hi,
                    Wi,
                    window_lengths,
                    window_strides,
                    window_dilations,
                    dinput_left_pads,
                    dinput_right_pads);
}
