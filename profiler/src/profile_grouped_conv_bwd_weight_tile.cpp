// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <iostream>
#include <numeric>
#include <initializer_list>
#include <cstdlib>

#include "ck_tile/builder/testing/conv/ck_tile.hpp"
#include "ck_tile/host/device_prop.hpp"
#ifdef CK_TILE_DISPATCHER
#include "profiler/grouped_convolution_backward_weight_tile_dispatcher_algs.hpp"
#endif
#include "profiler/tile_profiler_utils.hpp"

#include "profiler_operation_registry.hpp"

namespace {

enum struct ConvLayout
{
    GNCHW_GKCYX_GNKHW, // 0
    GNHWC_GKYXC_GNHWK, // 1
    NHWGC_GKYXC_NHWGK, // 2
    NGCHW_GKYXC_NGKHW, // 3
    NGCHW_GKCYX_NGKHW, // 4
};

std::ostream& operator<<([[clang::lifetimebound]] std::ostream& os, const ConvLayout& layout)
{
    using ck::operator<<;
    switch(layout)
    {
    case ConvLayout::GNCHW_GKCYX_GNKHW:
        os << "Input[G, N, C, Hi, Wi], Weight[G, K, C, Y, X], Output[G, N, K, Ho, Wo]";
        break;
    case ConvLayout::GNHWC_GKYXC_GNHWK:
        os << "Input[G, N, Hi, Wi, C], Weight[G, K, Y, X, C], Output[G, N, Ho, Wo, K]";
        break;
    case ConvLayout::NHWGC_GKYXC_NHWGK:
        os << "Input[N, Hi, Wi, G, C], Weight[G, K, Y, X, C], Output[N, Ho, Wo, G, K]";
        break;
    case ConvLayout::NGCHW_GKYXC_NGKHW:
        os << "Input[N, G, C, Hi, Wi], Weight[G, K, Y, X, C], Output[N, G, K, Ho, Wo]";
        break;
    case ConvLayout::NGCHW_GKCYX_NGKHW:
        os << "Input[N, G, C, Hi, Wi], Weight[G, K, C, Y, X], Output[N, G, K, Ho, Wo]";
        break;
    default: os << "unknown layout";
    }

    return os;
}

enum struct ConvDataType
{
    F32_F32_F32,          // 0
    F16_F16_F16,          // 1
    BF16_FP32_BF16,       // 2
    F16_F16_F16_GEMM_BF8, // 3
    INT8_INT8_INT8,       // 4
    BF16_BF16_BF16,       // 5
    F32_F32_F32_COMP_TF32 // 6
};

std::ostream& operator<<([[clang::lifetimebound]] std::ostream& os, const ConvDataType& data_type)
{
    using ck::operator<<;
    switch(data_type)
    {
    case ConvDataType::F32_F32_F32: os << "Input fp32, Weight fp32, Output fp32"; break;
    case ConvDataType::F16_F16_F16: os << "Input fp16, Weight fp16, Output fp16"; break;
    case ConvDataType::BF16_FP32_BF16: os << "Input bf16, Weight fp32, Output bf16"; break;
    case ConvDataType::F16_F16_F16_GEMM_BF8:
        os << "Input fp16, Weight fp16, Output fp16, Gemm bf8@fp8";
        break;
    case ConvDataType::INT8_INT8_INT8: os << "Input int8, Weight int8, Output int8"; break;
    case ConvDataType::BF16_BF16_BF16: os << "Input bf16, Weight bf16, Output bf16"; break;
    case ConvDataType::F32_F32_F32_COMP_TF32:
        os << "Input fp32, Weight fp32, Output fp32, Compute tf32";
        break;
    default: os << "unknown data type";
    }

    return os;
}

#define OP_NAME "grouped_conv_bwd_weight_tile"
#define OP_DESC "Grouped Convolution Backward Weight (CK Tile)"

static void print_helper_msg()
{
    std::cout << "arg1: tensor operation (" OP_NAME ": " OP_DESC ")\n"
              << "arg2: data type (0: Input fp32, Weight fp32, Output fp32\n"
              << "                 1: Input fp16, Weight fp16, Output fp16\n"
              << "                 2: Input bf16, Weight fp32, Output bf16\n"
              << "                 3: Input fp16, Weight fp16, Output fp16, Gemm bf8@fp8\n"
              << "                 4: Input int8, Weight int8, Output int8\n"
              << "                 5: Input bf16, Weight bf16, Output bf16\n"
              << "                 6: Input fp32, Weight fp32, Output fp32, Compute tf32)\n"
              << "arg3: tensor layout (0: Input[G, N, C, Hi, Wi], Weight[G, K, C, Y, X], Output[G, "
                 "N, K, Ho, Wo]\n"
              << "                     1: Input[G, N, Hi, Wi, C], Weight[G, K, Y, X, C], Output[G, "
                 "N, Ho, Wo, K]\n"
              << "                     2: Input[N, Hi, Wi, G, C], Weight[G, K, Y, X, C], Output[N, "
                 "Ho, Wo, G, K]\n"
              << "                     3: Input[N, G, C, Hi, Wi], Weight[G, K, Y, X, C], Output[N, "
                 "G, K, Ho, Wo]\n"
              << "                     4: Input[N, G, C, Hi, Wi], Weight[G, K, C, Y, X], Output[N, "
                 "G, K, Ho, Wo]\n"
              << "arg4: verification (0: no, 1: yes)\n"
              << "arg5: initialization (0: no init, 1: integer value, 2: decimal value)\n"
              << "arg6: print tensor value (0: no; 1: yes)\n"
              << "arg7: time kernel (0: no, 1: yes)\n"
              << ck::utils::conv::get_conv_param_parser_helper_msg()
              << " SplitK (-1 for internally computed split-K value, positive value to set k "
                 "batches explicitly, or 'all' to test all internal split-K values)\n"
              << std::endl;
}

namespace ckb = ck_tile::builder;
namespace ckt = ck_tile::builder::test;
namespace ckp = ck_tile::builder::profiling;

template <auto SIGNATURE>
int call_profiler(const ckt::Args<SIGNATURE>& args,
                  const std::string& split_k,
                  bool do_verification,
                  bool time_kernel)
{
    auto inputs  = ckt::alloc_inputs(args);
    auto outputs = ckt::alloc_outputs(args);
    ckt::init_inputs(args, inputs.get());

    std::cout << args.make_input_descriptor() << std::endl;
    std::cout << args.make_weight_descriptor() << std::endl;
    std::cout << args.make_output_descriptor() << std::endl;
    auto&& [valid, avg_time, op_name, best_split_k] =
        ckp::run_grouped_conv_backward_weight_tile_algs(
            args,
            split_k,
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
        std::cout << "\nBest configuration parameters:" << "\n\tname: " << op_name
                  << "\n\tavg_time: " << avg_time << ", SplitK " << best_split_k << std::endl;
    }
    return !valid;
}

} // namespace

int profile_grouped_conv_bwd_weight_tile(int argc, char* argv[])
{
    // 8 for control, 1 for num_dim_spatial
    if(argc < 9)
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
    if(argc != 8 + 1 + 4 + 6 * num_dim_spatial + 1)
    {
        print_helper_msg();
        return 1;
    }

    constexpr ck_tile::index_t conv_params_start_idx = 9;

    std::cout << "IMPORTANT: Generate instances using: python "
                 "experimental/builder/src/generate_instances.py --mode=profiler and rerun cmake"
              << std::endl;

    std::cout << "Data type: " << data_type << std::endl;
    std::cout << "Layout: " << layout << std::endl;
    const auto params =
        ck::utils::conv::parse_conv_param(num_dim_spatial, conv_params_start_idx, argv);
    std::cout << params << std::endl;

    const std::string& split_k = std::string(argv[8 + 1 + 4 + 6 * num_dim_spatial]);
    std::cout << "Split-K: " << split_k << std::endl;

    if(layout == ConvLayout::NHWGC_GKYXC_NHWGK)
    {
        if(num_dim_spatial == 2)
        {
            if(data_type == ConvDataType::F16_F16_F16)
            {
                constexpr auto SIGNATURE = ckp::SIGNATURE_NHWGC_FP16_BWD_WEIGHT;
                return call_profiler<SIGNATURE>(
                    ckp::parse_conv_args<SIGNATURE>(conv_params_start_idx, argv),
                    split_k,
                    do_verification,
                    time_kernel);
            }
            else if(data_type == ConvDataType::BF16_BF16_BF16)
            {
                constexpr auto SIGNATURE = ckp::SIGNATURE_NHWGC_BF16_BWD_WEIGHT;
                return call_profiler<SIGNATURE>(
                    ckp::parse_conv_args<SIGNATURE>(conv_params_start_idx, argv),
                    split_k,
                    do_verification,
                    time_kernel);
            }
            else if(data_type == ConvDataType::F32_F32_F32)
            {
                constexpr auto SIGNATURE = ckp::SIGNATURE_NHWGC_FP32_BWD_WEIGHT;
                return call_profiler<SIGNATURE>(
                    ckp::parse_conv_args<SIGNATURE>(conv_params_start_idx, argv),
                    split_k,
                    do_verification,
                    time_kernel);
            }
        }
        else if(num_dim_spatial == 3)
        {
            if(data_type == ConvDataType::F16_F16_F16)
            {
                constexpr auto SIGNATURE = ckp::SIGNATURE_NDHWGC_FP16_BWD_WEIGHT;
                return call_profiler<SIGNATURE>(
                    ckp::parse_conv_args<SIGNATURE>(conv_params_start_idx, argv),
                    split_k,
                    do_verification,
                    time_kernel);
            }
            else if(data_type == ConvDataType::BF16_BF16_BF16)
            {
                constexpr auto SIGNATURE = ckp::SIGNATURE_NDHWGC_BF16_BWD_WEIGHT;
                return call_profiler<SIGNATURE>(
                    ckp::parse_conv_args<SIGNATURE>(conv_params_start_idx, argv),
                    split_k,
                    do_verification,
                    time_kernel);
            }
            else if(data_type == ConvDataType::F32_F32_F32)
            {
                constexpr auto SIGNATURE = ckp::SIGNATURE_NDHWGC_FP32_BWD_WEIGHT;
                return call_profiler<SIGNATURE>(
                    ckp::parse_conv_args<SIGNATURE>(conv_params_start_idx, argv),
                    split_k,
                    do_verification,
                    time_kernel);
            }
        }
    }

    std::cout << "this data_type & layout is not implemented" << std::endl;

    return 1;
}

REGISTER_PROFILER_OPERATION(OP_NAME, OP_DESC, profile_grouped_conv_bwd_weight_tile);
