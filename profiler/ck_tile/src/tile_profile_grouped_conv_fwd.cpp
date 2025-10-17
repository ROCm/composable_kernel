// SPDX-License-Identifier: MIT
// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.

#include <cstdlib>
#include <initializer_list>
#include <iostream>
#include <numeric>

#include "tile_profile_grouped_conv_fwd_impl.hpp"
#include "tile_profiler_operation_registry.hpp"

// CK Tile library dependencies
#include "ck_tile/core/numeric/integral_constant.hpp"
#include "ck_tile/ops/common/tensor_layout.hpp"

namespace {

enum struct ConvLayout
{
    GNCHW_GKCYX_GNKHW, // 0
    GNHWC_GKYXC_GNHWK, // 1
    NHWGC_GKYXC_NHWGK, // 2
    NGCHW_GKYXC_NGKHW, // 3
    NGCHW_GKCYX_NGKHW, // 4
};

enum struct ConvDataType
{
    F32_F32_F32,        // 0
    F16_F16_F16,        // 1
    BF16_F32_BF16,      // 2
    F16_F16_F16_BF8_F8, // 3
    I8_I8_I8,           // 4
    BF16_BF16_BF16,     // 5
    F32_F32_F32_TF32,   // 6
};

#define OP_NAME "grouped_conv_fwd"
#define OP_DESC "Grouped Convolution Forward"

static void print_helper_msg()
{
    std::string conv_param_parser_helper_msg;

    conv_param_parser_helper_msg += "Following arguments (depending on number of spatial dims):\n"
           " Number of spatial dimensions (1=Conv1d, 2=Conv2d, 3=Conv3d)\n"
           " G, N, K, C, \n"
           " <filter spatial dimensions>, (ie Y, X for 2D)\n"
           " <input image spatial dimensions>, (ie Hi, Wi for 2D)\n"
           " <strides>, (ie Sy, Sx for 2D)\n"
           " <dilations>, (ie Dy, Dx for 2D)\n"
           " <left padding>, (ie LeftPy, LeftPx for 2D)\n"
           " <right padding>, (ie RightPy, RightPx for 2D)\n";

    std::cout
        // clang-format off
        << "arg1: tensor operation (" OP_NAME ": " OP_DESC ")\n"
        << "arg2: data type (0: Input fp32, Weight fp32, Output fp32\n"
        << "                 1: Input fp16, Weight fp16, Output fp16\n"
        << "                 2: Input bf16, Weight bf16, Output bf16\n"
        << "                 3: Input int8, Weight int8, Output int8\n"
        << "                 4: Input fp8, Weight fp8, Output fp8\n"
        << "                 5: Input bf8, Weight bf8, Output fp8\n"
        << "                 6: Input fp8, Weight bf8, Output fp8\n"
        << "                 7: Input bf8, Weight fp8, Output fp8\n"
        << "                 8: Input fp32, Weight fp32, Output fp32, Compute tf32)\n"
        << "arg3: tensor layout (0: Input[G, N, Hi, Wi, C], Weight[G, K, Y, X, C], Output[G, N, Ho, Wo, K]\n"
        << "                     1: Input[N, Hi, Wi, G, C], Weight[G, K, Y, X, C], Output[N, Ho, Wo, G, K]\n"
        << "                     2: Input[N, G, C, Hi, Wi], Weight[G, K, Y, X, C], Output[N, "
        "G, K, Ho, Wo]\n"
        << "                     3: Input[N, G, C, Hi, Wi], Weight[G, K, C, Y, X], Output[N, "
        "G, K, Ho, Wo])\n"
        << "arg4: indexing data type (0: 32-bit, 1: 64-bit)\n"
        << "arg5: verification (0: no, 1: yes)\n"
        << "arg6: initialization (0: no init, 1: integer value, 2: decimal value)\n"
        << "arg7: print tensor value (0: no; 1: yes)\n"
        << "arg8: time kernel (0: no, 1: yes)\n"
        << conv_param_parser_helper_msg << std::endl;
    // clang-format on
}

} // namespace

int tile_profile_grouped_conv_fwd(int argc, char* argv[])
{
    // 8 for control, 1 for num_dim_spatial
    if(argc < 10)
    {
        print_helper_msg();
        return 1;
    }

    const auto data_type       = static_cast<ConvDataType>(std::stoi(argv[2]));
    const auto layout          = static_cast<ConvLayout>(std::stoi(argv[3]));
    const bool do_verification = std::stoi(argv[5]);
    const int init_method      = std::stoi(argv[6]);
    const bool do_log          = std::stoi(argv[7]);
    const bool time_kernel     = std::stoi(argv[8]);
    const int num_dim_spatial  = std::stoi(argv[9]);

    // 9 for control, 1 for num_dim_spatial, 4 for G/N/K/C, and 6 * num_dim_spatial
    if(argc != 9 + 1 + 4 + 6 * num_dim_spatial)
    {
        print_helper_msg();
        return 1;
    }

    const auto params = ck_tile::conv::parse_conv_param(num_dim_spatial, 10, argv);
    constexpr ck_tile::index_t k_batch = 1;

    using F32  = float;
    using F16  = ck_tile::half_t;
    using BF16 = ck_tile::bfloat16_t;
    using F8   = ck_tile::fp8_t;
    using BF8  = ck_tile::bf8_t;
#if defined(__gfx942__)
    using TF32 = ck::tf32_t;
#endif

    using NHWGC  = ck_tile::tensor_layout::convolution::NHWGC;
    using NDHWGC = ck_tile::tensor_layout::convolution::NDHWGC;

    using GKYXC  = ck_tile::tensor_layout::convolution::GKYXC;
    using GKZYXC = ck_tile::tensor_layout::convolution::GKZYXC;

    using NHWGK  = ck_tile::tensor_layout::convolution::NHWGK;
    using NDHWGK = ck_tile::tensor_layout::convolution::NDHWGK;

    constexpr auto I2 = ck_tile::number<2>{};
    constexpr auto I3 = ck_tile::number<3>{};

    auto profile = [&](auto num_dim_spatial_tmp,
                       auto in_layout,
                       auto wei_layout,
                       auto out_layout,
                       auto in_type,
                       auto wei_type,
                       auto out_type,
                       auto compute_type_a,
                       auto compute_type_b) {
        constexpr ck_tile::index_t NDimSpatial = num_dim_spatial_tmp.value;

        using InLayout  = decltype(in_layout);
        using WeiLayout = decltype(wei_layout);
        using OutLayout = decltype(out_layout);

        using InDataType  = decltype(in_type);
        using WeiDataType = decltype(wei_type);
        using OutDataType = decltype(out_type);

        using ComputeTypeA = decltype(compute_type_a);
        using ComputeTypeB = decltype(compute_type_b);

        bool pass = ck_tile::profiler::profile_grouped_conv_fwd_impl<NDimSpatial,
                                                                       InLayout,
                                                                       WeiLayout,
                                                                       OutLayout,
                                                                       InDataType,
                                                                       WeiDataType,
                                                                       OutDataType,
                                                                       ComputeTypeA,
                                                                       ComputeTypeB>(
            do_verification, init_method, do_log, time_kernel, params, k_batch);

        return pass ? 0 : 1;
    };

    if(num_dim_spatial == 2 && layout == ConvLayout::NHWGC_GKYXC_NHWGK)
    {
        if(data_type == ConvDataType::F32_F32_F32)
        {
            return profile(I2, NHWGC{}, GKYXC{}, NHWGK{}, F32{}, F32{}, F32{}, F32{}, F32{});
        }
        if(data_type == ConvDataType::F16_F16_F16)
        {
            return profile(I2, NHWGC{}, GKYXC{}, NHWGK{}, F16{}, F16{}, F16{}, F16{}, F16{});
        }
        if(data_type == ConvDataType::BF16_F32_BF16)
        {
            // fp32 atomic add is used for weight tensor in bf16 kernel
            return profile(I2, NHWGC{}, GKYXC{}, NHWGK{}, BF16{}, F32{}, BF16{}, BF16{}, BF16{});
        }
        if(data_type == ConvDataType::BF16_BF16_BF16)
        {
            return profile(I2, NHWGC{}, GKYXC{}, NHWGK{}, BF16{}, BF16{}, BF16{}, BF16{}, BF16{});
        }
        else if(data_type == ConvDataType::F32_F32_F32_TF32)
        {
#if defined(__gfx942__)
            return profile(I2, NHWGC{}, GKYXC{}, NHWGK{}, F32{}, F32{}, F32{}, TF32{}, TF32{});
#endif
        }
    }
    
    if(num_dim_spatial == 3 && layout == ConvLayout::NHWGC_GKYXC_NHWGK)
    {
        if(data_type == ConvDataType::F32_F32_F32)
        {
            return profile(I3, NDHWGC{}, GKZYXC{}, NDHWGK{}, F32{}, F32{}, F32{}, F32{}, F32{});
        }
        if(data_type == ConvDataType::F16_F16_F16)
        {
            return profile(I3, NDHWGC{}, GKZYXC{}, NDHWGK{}, F16{}, F16{}, F16{}, F16{}, F16{});
        }
        if(data_type == ConvDataType::BF16_F32_BF16)
        {
            // fp32 atomic add is used for weight tensor in bf16 kernel
            return profile(I3, NDHWGC{}, GKZYXC{}, NDHWGK{}, BF16{}, F32{}, BF16{}, BF16{}, BF16{});
        }
        if(data_type == ConvDataType::BF16_BF16_BF16)
        {
            return profile(
                I3, NDHWGC{}, GKZYXC{}, NDHWGK{}, BF16{}, BF16{}, BF16{}, BF16{}, BF16{});
        }
        if(data_type == ConvDataType::F16_F16_F16_BF8_F8)
        {
            return profile(I3, NDHWGC{}, GKZYXC{}, NDHWGK{}, F16{}, F16{}, F16{}, BF8{}, F8{});
        }
        else if(data_type == ConvDataType::I8_I8_I8)
        {
            return profile(
                I3, NDHWGC{}, GKZYXC{}, NDHWGK{}, int8_t{}, int8_t{}, int8_t{}, int8_t{}, int8_t{});
        }
        else if(data_type == ConvDataType::F32_F32_F32_TF32)
        {
#if defined(__gfx942__)
            return profile(I3, NDHWGC{}, GKZYXC{}, NDHWGK{}, F32{}, F32{}, F32{}, TF32{}, TF32{});
#endif
        }
    }

    std::cout << "this data_type & layout is not implemented" << std::endl;

    return 1;
}

REGISTER_PROFILER_OPERATION(OP_NAME, OP_DESC, tile_profile_grouped_conv_fwd);
