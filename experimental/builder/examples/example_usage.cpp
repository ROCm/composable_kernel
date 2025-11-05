// Copyright (C) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

/**
 * Example: Using ConvBuilder-generated kernel instances
 * 
 * This example demonstrates how to use the automatically generated
 * convolution kernel instances from the ConvBuilder system.
 */

#include <iostream>
#include "ck_tile/builder/conv_builder.hpp"
#include "experimental/builder/test/impl/conv_signature_types.hpp"
#include "experimental/builder/test/impl/conv_algorithm_types.hpp"

using namespace ck_tile::builder;
using namespace ck_tile::builder::test;

int main() {
    // Example 1: Define a convolution signature at compile-time
    constexpr ConvSignature my_signature = {
        .spatial_dim = 2,
        .direction = ConvDirection::FORWARD,
        .layout = GroupConvLayout{._2d = GroupConvLayout2D::GNHWC_GKYXC_GNHWK},
        .data_type = DataType::FP16,
        .elementwise_operation = ElementwiseOperation::PASS_THROUGH
    };

    // Example 2: Define an algorithm at compile-time (Standard XDL)
    constexpr ConvAlgorithm_DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle my_algorithm = {
        .thread_block = {
            .block_size = 256,
            .tile_size = {.m = 128, .n = 128, .k = 32}
        },
        .gridwise_gemm = {
            .ak1 = 8,
            .bk1 = 8,
            .m_per_xdl = 32,
            .n_per_xdl = 32,
            .m_xdl_per_wave = 2,
            .n_xdl_per_wave = 2
        },
        .block_transfer = {
            .block_transfer_a = {.k0 = 4, .m_n = 64, .k1 = 1},
            .block_transfer_b = {.k0 = 4, .m_n = 64, .k1 = 1},
            .thread_cluster_dims_c = {
                .m_block = 1,
                .m_wave_per_xdl = 32,
                .n_block = 1,
                .n_wave_per_xdl = 8
            },
            .lds_transfer_a = {
                .src_vector_dim = 2,
                .src_scalar_per_vector = 8,
                .lds_dst_scalar_per_vector = 8,
                .is_direct_load = false,
                .lds_padding = true
            },
            .lds_transfer_b = {
                .src_vector_dim = 2,
                .src_scalar_per_vector = 8,
                .lds_dst_scalar_per_vector = 8,
                .is_direct_load = false,
                .lds_padding = true
            },
            .epilogue_c = {
                .m_per_wave_per_shuffle = 1,
                .n_per_wave_per_shuffle = 1,
                .scalar_per_vector = 8
            },
            .block_transfer_access_order_a = {.order = {1, 0, 2}},
            .block_transfer_access_order_b = {.order = {1, 0, 2}},
            .src_access_order_a = {.order = {1, 0, 2}},
            .src_access_order_b = {.order = {1, 0, 2}}
        },
        .fwd_specialization = ConvFwdSpecialization::DEFAULT,
        .gemm_specialization = GemmSpecialization::MNKPadding,
        .num_gemm_k_prefetch_stages = 1,
        .num_groups_to_merge = 1,
        .loop_scheduler = LoopScheduler::DEFAULT
    };

    // Example 3: Instantiate ConvBuilder with signature and algorithm
    using MyConvBuilder = ConvBuilder<my_signature, my_algorithm>;
    using MyKernelInstance = MyConvBuilder::Instance;

    std::cout << "ConvBuilder instantiation successful!\n";
    std::cout << "Kernel instance type created.\n";

    // Example 4: Alternatively, include pre-generated instances
    // #include "experimental/builder/generated/conv_instances_batch_00.cpp"
    // using namespace ck_tile::builder::generated::batch_0;
    // using PreGeneratedKernel = Instance_0;  // Use pre-generated instance

    return 0;
}

/*
 * To build this example:
 * 
 * mkdir -p build && cd build
 * cmake .. -DBUILD_TESTING=ON
 * make example_conv_builder_usage
 * ./bin/example_conv_builder_usage
 */
