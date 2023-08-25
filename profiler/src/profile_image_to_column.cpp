// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2023, Advanced Micro Devices, Inc. All rights reserved.

#include <iostream>
#include <numeric>
#include <initializer_list>
#include <cstdlib>

#include "profiler/profile_image_to_column_impl.hpp"
#include "profiler_operation_registry.hpp"

namespace {

enum struct ConvLayout
{
    NHWC, // 0
};

enum struct DataType
{
    F32_F32,   // 0
    F16_F16,   // 1
    BF16_BF16, // 2
    INT8_INT8, // 3
};

#define OP_NAME "image_to_column"
#define OP_DESC "Image To Column"

static void print_helper_msg()
{
    std::cout
        // clang-format off
        << "arg1: tensor operation (" OP_NAME ": " OP_DESC ")\n"
        << "arg2: data type (0: Input fp32, Weight fp32, Output fp32\n"
        << "                 1: Input fp16, Weight fp16, Output fp16\n"
        << "                 2: Input bf16, Weight bf16, Output bf16\n"
        << "                 3: Input int8, Weight int8, Output int8)\n"
        << "arg3: tensor layout (0: Input[N, Hi, Wi, C], Output[N * Ho * Wo, Y * X * C])\n"
        << "arg4: verification (0: no, 1: yes)\n"
        << "arg5: initialization (0: no init, 1: integer value, 2: decimal value)\n"
        << "arg6: print tensor value (0: no; 1: yes)\n"
        << "arg7: time kernel (0: no, 1: yes)\n"
        << ck::utils::conv::get_conv_param_parser_helper_msg() << std::endl;
    // clang-format on
}

} // namespace

int profile_image_to_column(int argc, char* argv[])
{
    // 8 for control, 1 for num_dim_spatial
    if(argc < 9)
    {
        print_helper_msg();
        return 1;
    }

    const auto data_type       = static_cast<DataType>(std::stoi(argv[2]));
    const auto layout          = static_cast<ConvLayout>(std::stoi(argv[3]));
    const bool do_verification = std::stoi(argv[4]);
    const int init_method      = std::stoi(argv[5]);
    const bool do_log          = std::stoi(argv[6]);
    const bool time_kernel     = std::stoi(argv[7]);
    const int num_dim_spatial  = std::stoi(argv[8]);

    // 8 for control, 1 for num_dim_spatial, 4 for G/N/K/C, and 6 * num_dim_spatial
    if(argc != 8 + 1 + 4 + 6 * num_dim_spatial)
    {
        print_helper_msg();
        return 1;
    }

    const auto params = ck::utils::conv::parse_conv_param(num_dim_spatial, 9, argv);

    using F32  = float;
    using F16  = ck::half_t;
    using BF16 = ck::bhalf_t;
    using INT8 = int8_t;

    using namespace ck::tensor_layout::convolution;

    constexpr auto I1 = ck::Number<1>{};
    constexpr auto I2 = ck::Number<2>{};
    constexpr auto I3 = ck::Number<3>{};

    auto profile = [&](auto num_dim_spatial_tmp, auto in_layout, auto in_type, auto out_type) {
        constexpr ck::index_t NDimSpatial = num_dim_spatial_tmp.value;

        using InLayout = decltype(in_layout);

        using InDataType  = decltype(in_type);
        using OutDataType = decltype(out_type);

        bool pass = ck::profiler::
            profile_image_to_column_impl<NDimSpatial, InLayout, InDataType, OutDataType>(
                do_verification, init_method, do_log, time_kernel, params);

        return pass ? 0 : 1;
    };

    // NHWC
    if(layout == ConvLayout::NHWC)
    {
        if(num_dim_spatial == 1)
        {
            if(data_type == DataType::F32_F32)
            {
                return profile(I1, GNWC{}, F32{}, F32{});
            }
            else if(data_type == DataType::F16_F16)
            {
                return profile(I1, GNWC{}, F16{}, F16{});
            }
            else if(data_type == DataType::BF16_BF16)
            {
                return profile(I1, GNWC{}, BF16{}, BF16{});
            }
            else if(data_type == DataType::INT8_INT8)
            {
                return profile(I1, GNWC{}, INT8{}, INT8{});
            }
        }
        else if(num_dim_spatial == 2)
        {
            if(data_type == DataType::F32_F32)
            {
                return profile(I2, GNHWC{}, F32{}, F32{});
            }
            else if(data_type == DataType::F16_F16)
            {
                return profile(I2, GNHWC{}, F16{}, F16{});
            }
            else if(data_type == DataType::BF16_BF16)
            {
                return profile(I2, GNHWC{}, BF16{}, BF16{});
            }
            else if(data_type == DataType::INT8_INT8)
            {
                return profile(I2, GNHWC{}, INT8{}, INT8{});
            }
        }
        else if(num_dim_spatial == 3)
        {
            if(data_type == DataType::F32_F32)
            {
                return profile(I3, GNDHWC{}, F32{}, F32{});
            }
            else if(data_type == DataType::F16_F16)
            {
                return profile(I3, GNDHWC{}, F16{}, F16{});
            }
            else if(data_type == DataType::BF16_BF16)
            {
                return profile(I3, GNDHWC{}, BF16{}, BF16{});
            }
            else if(data_type == DataType::INT8_INT8)
            {
                return profile(I3, GNDHWC{}, INT8{}, INT8{});
            }
        }
    }

    std::cout << "this data_type & layout is not implemented" << std::endl;

    return 1;
}

REGISTER_PROFILER_OPERATION(OP_NAME, OP_DESC, profile_image_to_column);
