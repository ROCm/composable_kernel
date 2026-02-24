// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <iostream>
#include <numeric>
#include <initializer_list>
#include <cstdlib>

#include "profiler/profile_grouped_conv_fwd_convscale_add_impl.hpp"
#include "profiler_operation_registry.hpp"

using F8  = ck::f8_t;
using F32 = float;

namespace {

enum struct ConvLayout
{
    NDHWGC_GKZYXC_NDHWGK, // 0
    // NDHWGK_GKZYXC_NDHWGK, // 1
    // NHWGC_GKYXC_NHWGK,    // 2
    // NHWGK_GKYXC_NHWGK,    // 3
    // NWGC_GKXC_NWGK,       // 4
    // NWGK_GKXC_NWGK,       // 5
};

enum struct ConvDataType
{
    F8_F8_F8, // 0
};

enum struct IndexType
{
    INDEX_T,      // 0
    LONG_INDEX_T, // 1
};

#define OP_NAME "grouped_conv_fwd_convscale_add"
#define OP_DESC "Grouped Convolution Forward ConvScaleAdd"

static void print_helper_msg()
{
    std::cout
        // clang-format off
        << "arg1: tensor operation (" OP_NAME ": " OP_DESC ")\n"
        << "arg2: data type (0: Input f8, Weight f8, Output f8\n"
        << "arg3: tensor layout (0: Input[N, Di, Hi, Wi, G, C], Weight[G, K, Z, Y, X, C], Output[N, Do, Ho, Wo, G, K])\n"
        << "arg4: index type (0: INDEX_T, 1: LONG_INDEX_T)\n"
        << "arg5: verification (0: no, 1: yes)\n"
        << "arg6: initialization (0: no init, 1: integer value, 2: decimal value)\n"
        << "arg7: print tensor value (0: no; 1: yes)\n"
        << "arg8: time kernel (0=no, 1=yes)\n"
        << ck::utils::conv::get_conv_param_parser_helper_msg() << std::endl;
    // clang-format on
}

int profile_grouped_conv_fwd_convscale_add(int argc, char* argv[])
{
    // 8 for control, 1 for num_dim_spatial
    if(argc < 10)
    {
        print_helper_msg();
        return 1;
    }

    const auto data_type       = static_cast<ConvDataType>(std::stoi(argv[2]));
    const auto layout          = static_cast<ConvLayout>(std::stoi(argv[3]));
    const auto index_type      = static_cast<IndexType>(std::stoi(argv[4]));
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

    const auto params = ck::utils::conv::parse_conv_param(num_dim_spatial, 10, argv);

    if(index_type != IndexType::INDEX_T)
    {
        std::cout << "this indexing data type is not implemented" << std::endl;
        return 1;
    }

    using F32 = float;

    using GKZYXC = ck::tensor_layout::convolution::GKZYXC;
    using NDHWGC = ck::tensor_layout::convolution::NDHWGC;
    using NDHWGK = ck::tensor_layout::convolution::NDHWGK;

    constexpr auto I3 = ck::Number<3>{};

    auto profile = [&](auto num_dim_spatial_tmp,
                       auto in_layout,
                       auto wei_layout,
                       auto d_layout,
                       auto out_layout,
                       auto in_type,
                       auto wei_type,
                       auto d_type,
                       auto out_type,
                       auto a_compute_type,
                       auto b_compute_type) {
        constexpr ck::index_t NDimSpatial = num_dim_spatial_tmp.value;

        using InLayout  = decltype(in_layout);
        using WeiLayout = decltype(wei_layout);
        using DLayout   = decltype(d_layout);
        using OutLayout = decltype(out_layout);

        using InDataType  = decltype(in_type);
        using WeiDataType = decltype(wei_type);
        using DDataType   = decltype(d_type);
        using OutDataType = decltype(out_type);

        using AComputeType = decltype(a_compute_type);
        using BComputeType = decltype(b_compute_type);

        const auto convscaleadd_op = ck::tensor_operation::element_wise::ConvScaleAdd{};

        bool pass = ck::profiler::profile_grouped_conv_fwd_convscale_add_impl<NDimSpatial,
                                                                              InLayout,
                                                                              WeiLayout,
                                                                              DLayout,
                                                                              OutLayout,
                                                                              InDataType,
                                                                              WeiDataType,
                                                                              DDataType,
                                                                              OutDataType,
                                                                              AComputeType,
                                                                              BComputeType,
                                                                              ck::index_t>(
            do_verification, init_method, do_log, time_kernel, params, convscaleadd_op);

        return pass ? 0 : 1;
    };

    if(num_dim_spatial == 3 && layout == ConvLayout::NDHWGC_GKZYXC_NDHWGK)
    {

        if(data_type == ConvDataType::F8_F8_F8)
        {
            return profile(
                I3, NDHWGC{}, GKZYXC{}, NDHWGK{}, NDHWGK{}, F8{}, F8{}, F32{}, F8{}, F8{}, F8{});
            // I3, NDHWGC{}, GKZYXC{}, NDHWGK{}, NDHWGK{}, F8{}, F8{}, F32{}, F8{}, F32{}, F32{});
        }
    }

    std::cout << "this data_type & layout is not implemented" << std::endl;

    return 1;
}

} // namespace

REGISTER_PROFILER_OPERATION(OP_NAME, OP_DESC, profile_grouped_conv_fwd_convscale_add);
