// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <iostream>
#include <numeric>
#include <initializer_list>
#include <cstdlib>

#include "ck_tile/builder/testing/conv/ck_tile.hpp"
#include "ck_tile/host/device_prop.hpp"
#ifdef CK_TILE_DISPATCHER
#include "profiler/grouped_convolution_forward_tile_dispatcher_algs.hpp"
#endif
#include "profiler/tile_profiler_utils.hpp"

#include "profiler_operation_registry.hpp"

namespace {

enum struct ConvLayout
{
    GNHWC_GKYXC_GNHWK, // 0
    NHWGC_GKYXC_NHWGK, // 1
    NGCHW_GKYXC_NGKHW, // 2
    NGCHW_GKCYX_NGKHW, // 3
};

enum struct ConvDataType
{
    F32_F32_F32,      // 0
    F16_F16_F16,      // 1
    BF16_BF16_BF16,   // 2
    INT8_INT8_INT8,   // 3
    F8_F8_F8,         // 4
    BF8_BF8_F8,       // 5
    F8_BF8_F8,        // 6
    BF8_F8_F8,        // 7
    F32_F32_F32_TF32, // 8
};

enum struct IndexType
{
    INDEX_T,      // 0
    LONG_INDEX_T, // 1
};

static void print_helper_msg()
{
    std::cout
        // clang-format off
        << "arg1: tensor operation (grouped_conv_fwd : Grouped Convolution Forward)\n"
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
        << "Following arguments (depending on number of spatial dims):\n"
         <<  " Number of spatial dimensions (1=Conv1d, 2=Conv2d, 3=Conv3d)\n"
         <<  " G, N, K, C, \n"
         <<  " <filter spatial dimensions>, (ie Y, X for 2D)\n"
         <<  " <input image spatial dimensions>, (ie Hi, Wi for 2D)\n"
         <<  " <strides>, (ie Sy, Sx for 2D)\n"
         <<  " <dilations>, (ie Dy, Dx for 2D)\n"
         <<  " <left padding>, (ie LeftPy, LeftPx for 2D)\n"
         <<  " <right padding>, (ie RightPy, RightPx for 2D)\n" << std::endl;
    // clang-format on
}

namespace ckb = ck_tile::builder;
namespace ckt = ck_tile::builder::test;
namespace ckp = ck_tile::builder::profiling;

template <auto SIGNATURE>
int call_profiler(const ckt::Args<SIGNATURE>& args, bool do_verification, bool time_kernel)
{
    auto inputs  = alloc_inputs(args);
    auto outputs = alloc_outputs(args);
    ckt::init_inputs(args, inputs.get());

    std::cout << args.make_input_descriptor() << std::endl;
    std::cout << args.make_weight_descriptor() << std::endl;
    std::cout << args.make_output_descriptor() << std::endl;
    float avg_time;
    std::string op_name;
    bool valid;
    std::tie(valid, avg_time, op_name) =
        ckp::run_grouped_conv_forward_tile_algs(args,
                                                inputs.get(),
                                                outputs.get(),
                                                ck_tile::stream_config{nullptr,
                                                                       time_kernel,
                                                                       0 /*log_level*/,
                                                                       5 /*cold_iters*/,
                                                                       50 /*nrepeat_*/,
                                                                       true /*is_gpu_timer_*/,
                                                                       time_kernel /*flush_cache*/},
                                                do_verification);
    if(time_kernel)
    {
        std::cout << "Best configuration parameters:" << "\nname: " << op_name
                  << "\navg_time: " << avg_time << std::endl;
    }
    return !valid;
}

#define OP_NAME "grouped_conv_fwd_tile"
#define OP_DESC "Grouped Convolution Forward (CK Tile)"

} // namespace

int profile_grouped_conv_fwd_tile(int argc, char* argv[])
{
    // 8 for control, 1 for num_dim_spatial
    if(argc < 10)
    {
        print_helper_msg();
        return 1;
    }

    const auto data_type                   = static_cast<ConvDataType>(std::stoi(argv[2]));
    const auto layout                      = static_cast<ConvLayout>(std::stoi(argv[3]));
    const auto index_type                  = static_cast<IndexType>(std::stoi(argv[4]));
    const bool do_verification             = std::stoi(argv[5]);
    [[maybe_unused]] const int init_method = std::stoi(argv[6]);
    [[maybe_unused]] const bool do_log     = std::stoi(argv[7]);
    const bool time_kernel                 = std::stoi(argv[8]);
    const int num_dim_spatial              = std::stoi(argv[9]);

    // 9 for control, 1 for num_dim_spatial, 4 for G/N/K/C, and 6 * num_dim_spatial
    if(argc != 9 + 1 + 4 + 6 * num_dim_spatial)
    {
        print_helper_msg();
        return 1;
    }

    std::cout << "IMPORTANT: Generate instances using: python "
                 "experimental/builder/src/generate_instances.py --mode=profiler and rerun cmake"
              << std::endl;

    const auto params = ck::utils::conv::parse_conv_param(num_dim_spatial, 10, argv);

    if(index_type == IndexType::LONG_INDEX_T)
    {
        std::cout << "this indexing data type is not implemented" << std::endl;
        return 1;
    }

    if(layout == ConvLayout::NHWGC_GKYXC_NHWGK)
    {
        if(num_dim_spatial == 2)
        {
            if(data_type == ConvDataType::F32_F32_F32)
            {
                constexpr auto SIGNATURE = ckp::SIGNATURE_NHWGC_FP32_FWD;
                return call_profiler<SIGNATURE>(
                    ckp::parse_conv_args<SIGNATURE>(10, argv), do_verification, time_kernel);
            }
            else if(data_type == ConvDataType::F16_F16_F16)
            {
                constexpr auto SIGNATURE = ckp::SIGNATURE_NHWGC_FP16_FWD;
                return call_profiler<SIGNATURE>(
                    ckp::parse_conv_args<SIGNATURE>(10, argv), do_verification, time_kernel);
            }
            else if(data_type == ConvDataType::BF16_BF16_BF16)
            {
                constexpr auto SIGNATURE = ckp::SIGNATURE_NHWGC_BF16_FWD;
                return call_profiler<SIGNATURE>(
                    ckp::parse_conv_args<SIGNATURE>(10, argv), do_verification, time_kernel);
            }
        }
        else if(num_dim_spatial == 3)
        {
            if(data_type == ConvDataType::F32_F32_F32)
            {
                constexpr auto SIGNATURE = ckp::SIGNATURE_NDHWGC_FP32_FWD;
                return call_profiler<SIGNATURE>(
                    ckp::parse_conv_args<SIGNATURE>(10, argv), do_verification, time_kernel);
            }
            else if(data_type == ConvDataType::F16_F16_F16)
            {
                constexpr auto SIGNATURE = ckp::SIGNATURE_NDHWGC_FP16_FWD;
                return call_profiler<SIGNATURE>(
                    ckp::parse_conv_args<SIGNATURE>(10, argv), do_verification, time_kernel);
            }
            else if(data_type == ConvDataType::BF16_BF16_BF16)
            {
                constexpr auto SIGNATURE = ckp::SIGNATURE_NDHWGC_BF16_FWD;
                return call_profiler<SIGNATURE>(
                    ckp::parse_conv_args<SIGNATURE>(10, argv), do_verification, time_kernel);
            }
        }
    }
    else if(layout == ConvLayout::NGCHW_GKYXC_NGKHW || layout == ConvLayout::NGCHW_GKCYX_NGKHW)
    {
        if(num_dim_spatial == 2)
        {
            if(data_type == ConvDataType::F32_F32_F32)
            {
                constexpr auto SIGNATURE = ckp::SIGNATURE_NGCHW_FP32_FWD;
                return call_profiler<SIGNATURE>(
                    ckp::parse_conv_args<SIGNATURE>(10, argv), do_verification, time_kernel);
            }
            else if(data_type == ConvDataType::F16_F16_F16)
            {
                constexpr auto SIGNATURE = ckp::SIGNATURE_NGCHW_FP16_FWD;
                return call_profiler<SIGNATURE>(
                    ckp::parse_conv_args<SIGNATURE>(10, argv), do_verification, time_kernel);
            }
            else if(data_type == ConvDataType::BF16_BF16_BF16)
            {
                constexpr auto SIGNATURE = ckp::SIGNATURE_NGCHW_BF16_FWD;
                return call_profiler<SIGNATURE>(
                    ckp::parse_conv_args<SIGNATURE>(10, argv), do_verification, time_kernel);
            }
        }
    }

    std::cout << "this data_type & layout is not implemented" << std::endl;

    return 1;
}

REGISTER_PROFILER_OPERATION(OP_NAME, OP_DESC, profile_grouped_conv_fwd_tile);
