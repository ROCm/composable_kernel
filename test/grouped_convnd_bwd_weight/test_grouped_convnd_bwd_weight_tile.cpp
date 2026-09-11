// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <cstdlib>
#include <iostream>
#include <initializer_list>
#include <vector>
#include <gtest/gtest.h>

#include "ck_tile/builder/testing/conv/ck_tile.hpp"
#include "ck_tile/host/device_prop.hpp"
#ifdef CK_TILE_DISPATCHER
#include "profiler/grouped_convolution_backward_weight_tile_dispatcher_algs.hpp"
#else
#include "profiler/grouped_convolution_backward_weight_tile_algs.hpp"
#endif

static ck::index_t args_mask      = 0xffff;
static ck::index_t instance_index = -1;

namespace ckb = ck_tile::builder;
namespace ckt = ck_tile::builder::test;
namespace ckp = ck_tile::builder::profiling;

template <ck_tile::index_t num_spatial_dim_,
          ckb::DataType data_type_,
          ckb::DataType acc_data_type_,
          ckb::TensorLayout in_layout_,
          ckb::TensorLayout wei_layout_,
          ckb::TensorLayout out_layout_>
struct SignatureDetails
{
    static constexpr ck_tile::index_t num_spatial_dim = num_spatial_dim_;
    static constexpr ckb::DataType data_type          = data_type_;
    static constexpr ckb::DataType acc_data_type      = acc_data_type_;
    static constexpr ckb::TensorLayout in_layout      = in_layout_;
    static constexpr ckb::TensorLayout wei_layout     = wei_layout_;
    static constexpr ckb::TensorLayout out_layout     = out_layout_;
};

template <typename SignatureDetailsType>
class TestGroupedConvndBwdWeightTile : public ::testing::Test
{
    protected:
    static constexpr auto SIGNATURE =
        ckt::ConvSignature{.spatial_dim            = SignatureDetailsType::num_spatial_dim,
                           .direction              = ckb::ConvDirection::BACKWARD_WEIGHT,
                           .data_type              = SignatureDetailsType::data_type,
                           .accumulation_data_type = SignatureDetailsType::acc_data_type,
                           .input  = {.config = {.layout = SignatureDetailsType::in_layout}},
                           .weight = {.config = {.layout = SignatureDetailsType::wei_layout}},
                           .output = {.config = {.layout = SignatureDetailsType::out_layout}}};

    std::vector<ckt::Args<SIGNATURE>> conv_args;
    std::vector<std::string> split_ks{"-1", "1", "2", "64"};

    template <ck::index_t NDimSpatial>
    void Run()
    {
        ASSERT_FALSE(conv_args.empty());
        bool pass = true;
        for(size_t i = 0; i < conv_args.size(); i++)
        {
            for(auto& split_k : split_ks)
            {
                if((args_mask & (1 << i)) == 0)
                {
                    continue;
                }
                auto& args = conv_args[i];

                auto inputs  = alloc_inputs(args);
                auto outputs = alloc_outputs(args);
                ckt::init_tensor_buffer_uniform_int(
                    inputs.get().input, args.make_input_descriptor(), -5, 5);
                ckt::init_tensor_buffer_uniform_int(
                    inputs.get().output, args.make_output_descriptor(), -5, 5);

                std::cout << args.make_input_descriptor() << std::endl;
                std::cout << args.make_weight_descriptor() << std::endl;
                std::cout << args.make_output_descriptor() << std::endl;
                [[maybe_unused]] auto&& [case_passed, avg_time, op_name, best_split_k] =

                    ckp::run_grouped_conv_backward_weight_tile_algs(
                        args,
                        split_k,
                        inputs.get(),
                        outputs.get(),
                        ck_tile::stream_config{nullptr, false /*time_kernel*/});

                pass = pass && case_passed;
            }
        }
        EXPECT_TRUE(pass);
    }

    void conv_args_append(std::size_t,
                          std::size_t G,
                          std::size_t N,
                          std::size_t K,
                          std::size_t C,
                          const std::vector<std::size_t>& filter_spatial_lengths,
                          const std::vector<std::size_t>& input_spatial_lengths,
                          const std::vector<std::size_t>& conv_filter_strides,
                          const std::vector<std::size_t>& conv_filter_dilations,
                          const std::vector<std::size_t>& input_left_pads,
                          const std::vector<std::size_t>& input_right_pads)
    {
        ckt::Args<SIGNATURE> args = {
            .lengths =
                {
                    .batch_size      = N,
                    .groups          = G,
                    .input_channels  = C,
                    .output_channels = K,
                    .image = ckt::filter_extent_from_vector<SignatureDetailsType::num_spatial_dim>(
                        input_spatial_lengths),
                    .filter = ckt::filter_extent_from_vector<SignatureDetailsType::num_spatial_dim>(
                        filter_spatial_lengths),
                },
            .filter_strides = ckt::filter_extent_from_vector<SignatureDetailsType::num_spatial_dim>(
                conv_filter_strides),
            .filter_dilation =
                ckt::filter_extent_from_vector<SignatureDetailsType::num_spatial_dim>(
                    conv_filter_dilations),
            .input_left_pad = ckt::filter_extent_from_vector<SignatureDetailsType::num_spatial_dim>(
                input_left_pads),
            .input_right_pad =
                ckt::filter_extent_from_vector<SignatureDetailsType::num_spatial_dim>(
                    input_right_pads),
            .a_elementwise_op   = {},
            .b_elementwise_op   = {},
            .cde_elementwise_op = {},
        };
        conv_args.push_back(args);
    }
};

using KernelTypes2d = ::testing::Types<SignatureDetails<2,
                                                        ckb::DataType::FP32,
                                                        ckb::DataType::FP32,
                                                        ckb::TensorLayout::NHWGC,
                                                        ckb::TensorLayout::GKYXC,
                                                        ckb::TensorLayout::NHWGK>,
                                       SignatureDetails<2,
                                                        ckb::DataType::FP16,
                                                        ckb::DataType::FP32,
                                                        ckb::TensorLayout::NHWGC,
                                                        ckb::TensorLayout::GKYXC,
                                                        ckb::TensorLayout::NHWGK>,
                                       SignatureDetails<2,
                                                        ckb::DataType::BF16,
                                                        ckb::DataType::FP32,
                                                        ckb::TensorLayout::NHWGC,
                                                        ckb::TensorLayout::GKYXC,
                                                        ckb::TensorLayout::NHWGK>>;

using KernelTypes3d = ::testing::Types<SignatureDetails<3,
                                                        ckb::DataType::FP32,
                                                        ckb::DataType::FP32,
                                                        ckb::TensorLayout::NDHWGC,
                                                        ckb::TensorLayout::GKZYXC,
                                                        ckb::TensorLayout::NDHWGK>,
                                       SignatureDetails<3,
                                                        ckb::DataType::FP16,
                                                        ckb::DataType::FP32,
                                                        ckb::TensorLayout::NDHWGC,
                                                        ckb::TensorLayout::GKZYXC,
                                                        ckb::TensorLayout::NDHWGK>,
                                       SignatureDetails<3,
                                                        ckb::DataType::BF16,
                                                        ckb::DataType::FP32,
                                                        ckb::TensorLayout::NDHWGC,
                                                        ckb::TensorLayout::GKZYXC,
                                                        ckb::TensorLayout::NDHWGK>>;

template <typename SignatureDetailsType>
class TestGroupedConvndBwdWeightTile2d : public TestGroupedConvndBwdWeightTile<SignatureDetailsType>
{
};

template <typename SignatureDetailsType>
class TestGroupedConvndBwdWeightTile3d : public TestGroupedConvndBwdWeightTile<SignatureDetailsType>
{
};

TYPED_TEST_SUITE(TestGroupedConvndBwdWeightTile2d, KernelTypes2d);
TYPED_TEST_SUITE(TestGroupedConvndBwdWeightTile3d, KernelTypes3d);

TYPED_TEST(TestGroupedConvndBwdWeightTile2d, Test2D)
{
    this->conv_args.clear();
    this->conv_args_append(2, 2, 64, 4, 4, {1, 1}, {7, 7}, {1, 1}, {1, 1}, {0, 0}, {0, 0});
    this->conv_args_append(2, 2, 64, 128, 256, {1, 1}, {7, 7}, {2, 2}, {1, 1}, {0, 0}, {0, 0});
    this->conv_args_append(2, 2, 64, 3, 3, {1, 1}, {7, 7}, {1, 1}, {1, 1}, {0, 0}, {0, 0});
    this->conv_args_append(2, 2, 64, 5, 5, {1, 1}, {7, 7}, {1, 1}, {1, 1}, {0, 0}, {0, 0});
    this->conv_args_append(2, 2, 4, 128, 256, {3, 3}, {14, 14}, {1, 1}, {1, 1}, {1, 1}, {1, 1});
    this->conv_args_append(2, 2, 128, 128, 256, {1, 1}, {3, 3}, {1, 1}, {1, 1}, {0, 0}, {0, 0});
    this->conv_args_append(2, 1, 1, 1, 32, {3, 3}, {32, 32}, {1, 1}, {1, 1}, {1, 1}, {1, 1});
    this->conv_args_append(2, 1, 1, 64, 3, {3, 3}, {32, 32}, {1, 1}, {1, 1}, {1, 1}, {1, 1});
    this->conv_args_append(2, 1, 1, 1, 1, {3, 3}, {32, 32}, {1, 1}, {1, 1}, {1, 1}, {1, 1});
    this->conv_args_append(2, 16, 16, 1, 1, {3, 3}, {28, 28}, {2, 2}, {1, 1}, {1, 1}, {1, 1});
    this->template Run<2>();
}

TYPED_TEST(TestGroupedConvndBwdWeightTile3d, Test3D)
{
    this->conv_args.clear();
    this->conv_args_append(
        3, 2, 16, 128, 256, {1, 1, 1}, {7, 7, 7}, {2, 2, 2}, {1, 1, 1}, {0, 0, 0}, {0, 0, 0});
    this->conv_args_append(
        3, 2, 2, 128, 256, {3, 3, 3}, {14, 14, 3}, {1, 1, 1}, {1, 1, 1}, {1, 1, 1}, {1, 1, 1});
    this->conv_args_append(
        3, 2, 32, 128, 256, {1, 1, 1}, {3, 3, 3}, {1, 1, 1}, {1, 1, 1}, {0, 0, 0}, {0, 0, 0});
    this->conv_args_append(
        3, 1, 1, 1, 32, {3, 3, 3}, {16, 16, 16}, {1, 1, 1}, {1, 1, 1}, {1, 1, 1}, {1, 1, 1});
    this->conv_args_append(
        3, 1, 1, 64, 3, {3, 3, 3}, {14, 14, 14}, {1, 1, 1}, {1, 1, 1}, {1, 1, 1}, {1, 1, 1});
    this->conv_args_append(
        3, 1, 1, 1, 1, {3, 3, 3}, {18, 18, 18}, {1, 1, 1}, {1, 1, 1}, {1, 1, 1}, {1, 1, 1});
    this->conv_args_append(
        3, 16, 16, 1, 1, {3, 3, 3}, {28, 28, 28}, {2, 2, 2}, {1, 1, 1}, {1, 1, 1}, {1, 1, 1});
    this->template Run<3>();
}

int main(int argc, char** argv)
{
    testing::InitGoogleTest(&argc, argv);
    if(argc == 1) {}
    else if(argc == 3)
    {
        args_mask      = strtol(argv[1], nullptr, 0);
        instance_index = atoi(argv[2]);
    }
    else
    {
        std::cout << "Usage of " << argv[0] << std::endl;
        std::cout << "Arg1,2: args_mask instance_index(-1 means all)" << std::endl;
    }
    return RUN_ALL_TESTS();
}
