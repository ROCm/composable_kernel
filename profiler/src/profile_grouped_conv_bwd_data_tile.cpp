// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <iostream>
#include <numeric>
#include <initializer_list>
#include <cstdlib>
#include <string>

#include "ck_tile/builder/testing/conv/ck_tile.hpp"
#include "ck_tile/host/device_prop.hpp"
#ifdef CK_TILE_DISPATCHER
#include "profiler/grouped_convolution_backward_data_tile_dispatcher_algs.hpp"
#endif
#include "profiler/tile_profiler_utils.hpp"
#include "profiler/profiler_arg_utils.hpp"

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
    F32_F32_F32_TF32, // 3
};

#define OP_NAME "grouped_conv_bwd_data_tile"
#define OP_DESC "Grouped Convolution Backward Data (CK Tile)"

static void print_helper_msg()
{
    std::cout
        // clang-format off
        << "arg1: tensor operation (" OP_NAME ": " OP_DESC ")\n"
        << "arg2: data type (0: Output fp32, Weight fp32, Input fp32\n"
        << "                 1: Output fp16, Weight fp16, Input fp16\n"
        << "                 2: Output bf16, Weight bf16, Input bf16\n"
        << "                 3: Output fp32, Weight fp32, Input fp32, Compute tf32)\n"
        << "arg3: tensor layout (0: Output[G, N, Ho, Wo, C], Weight[G, K, Y, X, C], Input[G, N, Hi, Wi, K]\n"
        << "                     1: Output[N, Ho, Wo, G, C], Weight[G, K, Y, X, C], Input[N, Hi, Wi, G, K])\n"
        << "                     2: Output[N, G, C, Ho, Wo], Weight[G, K, Y, X, C], Input[N, G, K, Hi, Wi])\n"
        << "                     3: Output[N, G, C, Ho, Wo], Weight[G, K, C, Y, X], Input[N, G, K, Hi, Wi])\n"
        << "arg4: verification (0: no, 1: yes)\n"
        << "arg5: initialization (0: no init, 1: integer value, 2: decimal value)\n"
        << "arg6: print tensor value (0: no; 1: yes)\n"
        << "arg7: time kernel (0: no, 1: yes)\n"
        << ck::utils::conv::get_conv_param_parser_helper_msg() << std::endl
        << "Last argument: split-K (0: internally computed split-K value; 1, 2, 4, 8, 16, 32, 64, 128: set k batches explicitly)\n"
        << "\nOptional arguments:\n"
        << "  --instance <id>      Run only the specified instance (0-indexed among valid instances)\n";
    // clang-format on
}

namespace ckb = ck_tile::builder;
namespace ckt = ck_tile::builder::test;
namespace ckp = ck_tile::builder::profiling;

template <auto SIGNATURE>
int call_profiler(const ckt::Args<SIGNATURE>& args,
                  const std::string& split_k,
                  bool do_verification,
                  bool time_kernel,
                  ck_tile::index_t instance_index)
{
    auto inputs  = ckt::alloc_inputs(args);
    auto outputs = ckt::alloc_outputs(args);
    ckt::init_inputs(args, inputs.get());

    std::cout << args.make_input_descriptor() << std::endl;
    std::cout << args.make_weight_descriptor() << std::endl;
    std::cout << args.make_output_descriptor() << std::endl;
    auto&& [valid, avg_time, op_name, best_split_k, best_instance_index] =
        ckp::run_grouped_conv_backward_data_tile_algs(
            args,
            split_k,
            instance_index,
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
        std::cout << "\nBest configuration parameters:" << "\n\tname: " << op_name << " (instance "
                  << best_instance_index << ")" << "\n\tavg_time: " << avg_time << ", SplitK "
                  << best_split_k << std::endl;
    }
    return !valid;
}

} // namespace

int profile_grouped_conv_bwd_data_tile(int argc, char* argv[])
{
    // Parse optional named arguments first
    ck_tile::index_t instance_index = -1;
    bool dummy;
    ck::profiler::parse_named_args(argc, argv, instance_index, dummy);
    const int named_arg_count = ck::profiler::count_named_args(argc, argv);

    // Adjust argc for positional argument checking
    const int positional_argc = argc - named_arg_count;

    // 8 for control, 1 for num_dim_spatial
    if(positional_argc < 9)
    {
        print_helper_msg();
        return 1;
    }

    const auto data_type       = static_cast<ConvDataType>(std::stoi(argv[2]));
    const auto layout          = static_cast<ConvLayout>(std::stoi(argv[3]));
    const bool do_verification = std::stoi(argv[4]);
    const bool time_kernel     = std::stoi(argv[7]);
    const int num_dim_spatial  = std::stoi(argv[8]);

    // 8 for control, 1 for num_dim_spatial, 4 for G/N/K/C, and 6 * num_dim_spatial, 1 for split-K
    if(positional_argc != 8 + 1 + 4 + 6 * num_dim_spatial + 1)
    {
        print_helper_msg();
        return 1;
    }

    constexpr ck_tile::index_t conv_params_start_idx = 9;
    const auto params =
        ck::utils::conv::parse_conv_param(num_dim_spatial, conv_params_start_idx, argv);
    std::cout << params << std::endl;

    auto split_k = std::string(argv[8 + 1 + 4 + 6 * num_dim_spatial]);

    // The bwd data profiler in old CK uses -1 to loop over all split-K values.
    // We want to have the same API for backward compatibility, but we need to convert it to "all"
    // for the new API.
    if(split_k == "-1")
    {
        split_k = "all";
    }

    if(layout == ConvLayout::NHWGC_GKYXC_NHWGK)
    {
        if(num_dim_spatial == 2)
        {
            if(data_type == ConvDataType::F16_F16_F16)
            {
                constexpr auto SIGNATURE = ckp::SIGNATURE_NHWGC_FP16_BWD_DATA;
                return call_profiler<SIGNATURE>(
                    ckp::parse_conv_args<SIGNATURE>(conv_params_start_idx, argv),
                    split_k,
                    do_verification,
                    time_kernel,
                    instance_index);
            }
            else if(data_type == ConvDataType::BF16_BF16_BF16)
            {
                constexpr auto SIGNATURE = ckp::SIGNATURE_NHWGC_BF16_BWD_DATA;
                return call_profiler<SIGNATURE>(
                    ckp::parse_conv_args<SIGNATURE>(conv_params_start_idx, argv),
                    split_k,
                    do_verification,
                    time_kernel,
                    instance_index);
            }
            else if(data_type == ConvDataType::F32_F32_F32)
            {
                constexpr auto SIGNATURE = ckp::SIGNATURE_NHWGC_FP32_BWD_DATA;
                return call_profiler<SIGNATURE>(
                    ckp::parse_conv_args<SIGNATURE>(conv_params_start_idx, argv),
                    split_k,
                    do_verification,
                    time_kernel,
                    instance_index);
            }
        }
        else if(num_dim_spatial == 3)
        {
            if(data_type == ConvDataType::F16_F16_F16)
            {
                constexpr auto SIGNATURE = ckp::SIGNATURE_NDHWGC_FP16_BWD_DATA;
                return call_profiler<SIGNATURE>(
                    ckp::parse_conv_args<SIGNATURE>(conv_params_start_idx, argv),
                    split_k,
                    do_verification,
                    time_kernel,
                    instance_index);
            }
            else if(data_type == ConvDataType::BF16_BF16_BF16)
            {
                constexpr auto SIGNATURE = ckp::SIGNATURE_NDHWGC_BF16_BWD_DATA;
                return call_profiler<SIGNATURE>(
                    ckp::parse_conv_args<SIGNATURE>(conv_params_start_idx, argv),
                    split_k,
                    do_verification,
                    time_kernel,
                    instance_index);
            }
            else if(data_type == ConvDataType::F32_F32_F32)
            {
                constexpr auto SIGNATURE = ckp::SIGNATURE_NDHWGC_FP32_BWD_DATA;
                return call_profiler<SIGNATURE>(
                    ckp::parse_conv_args<SIGNATURE>(conv_params_start_idx, argv),
                    split_k,
                    do_verification,
                    time_kernel,
                    instance_index);
            }
        }
    }

    std::cout << "this data_type & layout is not implemented" << std::endl;

    return 1;
}

REGISTER_PROFILER_OPERATION(OP_NAME, OP_DESC, profile_grouped_conv_bwd_data_tile);
