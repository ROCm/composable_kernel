// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <gtest/gtest.h>
#include <vector>
#include <cmath>
#include <tuple>
#include <iostream>
#include <cstring>

#include "ck_tile/core.hpp"
#include "ck_tile/host.hpp"
#include "ck_tile/ops/mhc.hpp"
#include "ck_tile/host/kernel_launch.hpp"

template <typename Tuple>
class TestCkTileMHC : public ::testing::Test
{
    // protected:
    // using XDataType               = std::tuple_element_t<0, Tuple>;
    // using ComputeDataType         = std::tuple_element_t<1, Tuple>;
    // using YDataType               = std::tuple_element_t<2, Tuple>;
    // using ReduceOpsType           = std::tuple_element_t<3, Tuple>;
    // using ElementwiseOpsType      = std::tuple_element_t<4, Tuple>;
    // using AccumulatorOpsType      = std::tuple_element_t<5, Tuple>;
    // using InterBlockReduceOpsType = std::tuple_element_t<6, Tuple>;
    // using BlockWarps_             = std::tuple_element_t<7, Tuple>;
    // using BlockTile_              = std::tuple_element_t<8, Tuple>;
    // using WarpTile_               = std::tuple_element_t<9, Tuple>;
    // using ThreadTile_             = std::tuple_element_t<10, Tuple>;

    // using TestReduce2dShape =
    //     ck_tile::Reduce2dShape<BlockWarps_, BlockTile_, WarpTile_, ThreadTile_>;

    // template <std::size_t InputDim, typename KeptDimSeq, typename ReduceDimSeq>
    // void RunGenericTest(const std::vector<ck_tile::index_t>& input_shape,
    //                     const std::vector<ck_tile::index_t>& input_strides,
    //                     const std::vector<ck_tile::index_t>& output_shape,
    //                     const std::vector<ck_tile::index_t>& output_strides,
    //                     ck_tile::index_t kept_dim_len_prod,
    //                     ck_tile::index_t total_reduce_elements,
    //                     KeptDimSeq kept_dims,
    //                     ReduceDimSeq reduce_dims)
    void RunGenericTest()
    {

        // Test parameters
        const int B  = 8;     // Batch size
        const int n  = 4;     // Expansion rate (aka streams)
        const int C  = 256;   // Output layer dim
        const int nC = n * C; // Total input dimension

        const int output_dim = 2 * n + n * n; // 2n + n^2 = 8 + 16 = 24 for n=4

        // Allocate host tensors
        ck_tile::HostTensor<float> h_x({B, nC});              // Input [B, nC]
        ck_tile::HostTensor<float> h_phi({nC, output_dim});   // Weights [nC, 2n+n^2]
        ck_tile::HostTensor<float> h_output({B, output_dim}); // Output [B, 2n+n^2]

        // Initialize with random data
        ck_tile::FillUniformDistribution<float>{-1.0f, 1.0f}(h_x);
        ck_tile::FillUniformDistribution<float>{-0.5f, 0.5f}(h_phi);
        h_output.SetZero();

        // Allocate device memory
        ck_tile::DeviceMem d_x_mem(h_x.get_element_space_size_in_bytes());
        ck_tile::DeviceMem d_phi_mem(h_phi.get_element_space_size_in_bytes());
        ck_tile::DeviceMem d_output_mem(h_output.get_element_space_size_in_bytes());

        // Copy data to device
        d_x_mem.ToDevice(h_x.data());
        d_phi_mem.ToDevice(h_phi.data());
        d_output_mem.ToDevice(h_output.data());

        // Kernel launch configuration
        const ck_tile::index_t kBlockSize      = 256; // 256 threads per block
        const ck_tile::index_t kGridSize       = B;   // One block per batch element
        constexpr ck_tile::index_t kBlockPerCu = 1;

        // TODO: Define Problem and Policy types
        // using Problem = ck_tile::MHCProblem<...>;
        // using Kernel = ck_tile::ManifoldConstrainedHyperConnection<Problem, Policy>;

        std::cout << "Launching MHC kernel with:" << std::endl;
        std::cout << "  Batch size (B): " << B << std::endl;
        std::cout << "  Expansion factor (n): " << n << std::endl;
        std::cout << "  Channels per stream (C): " << C << std::endl;
        std::cout << "  Input dimension (nC): " << nC << std::endl;
        std::cout << "  Output dimension (2n+n²): " << output_dim << std::endl;
        std::cout << "  Grid size: " << kGridSize << std::endl;
        std::cout << "  Block size: " << kBlockSize << std::endl;

        // Kernel launch
        /*
        ck_tile::launch_kernel(
            ck_tile::stream_config{nullptr, false, 0},
            ck_tile::make_kernel<kBlockPerCu>(
                Kernel{},
                kGridSize,
                kBlockSize,
                0,  // shared memory size
                static_cast<float*>(d_x_mem.GetDeviceBuffer()),
                static_cast<float*>(d_phi_mem.GetDeviceBuffer()),
                static_cast<float*>(d_output_mem.GetDeviceBuffer()),
                B, n, C));
        */

        // Copy results back to host
        // d_output_mem.FromDevice(h_output.data());

        // TODO: Add reference computation and validation

        // auto h_ys = ck_tile::generate_tuple(
        //     [&output_shape, &output_strides](auto /*i*/) {
        //         return ck_tile::HostTensor<YDataType>(output_shape, output_strides);
        //     },
        //     ck_tile::number<number_operations>{});

        // auto h_ys_ref = ck_tile::generate_tuple(
        //     [&output_shape, &output_strides](auto /*i*/) {
        //         return ck_tile::HostTensor<YDataType>(output_shape, output_strides);
        //     },
        //     ck_tile::number<number_operations>{});

        // ck_tile::FillUniformDistribution<XDataType>{-5.f, 5.f}(h_x);

        // ck_tile::static_for<0, number_operations, 1>{}([&](auto i) {
        //     h_ys.template at<i>().SetZero();
        //     h_ys_ref.template at<i>().SetZero();
        // });

        // auto output_number_elements = [&output_shape]() {
        //     ck_tile::index_t prod = 1;
        //     for(auto len : output_shape)
        //         prod *= len;
        //     return prod;
        // }();

        // auto output_buffer_size =
        //     number_operations * h_ys.get(ck_tile::number<0>{}).get_element_space_size_in_bytes();
        // ck_tile::DeviceMem d_x_mem(h_x.get_element_space_size_in_bytes());
        // ck_tile::DeviceMem d_y_mem(output_buffer_size);

        // std::vector<YDataType> h(number_operations * output_number_elements);

        // // Init the output data with identity values respective to each reduce op
        // ck_tile::static_for<0, number_operations, 1>{}([&](auto i) {
        //     constexpr auto op       = ReduceOpsType{}.at(i);
        //     const auto identity_val = op.template GetIdentityValue<YDataType>();
        //     std::fill(h.begin() + i * output_number_elements,
        //               h.begin() + (i + 1) * output_number_elements,
        //               identity_val);
        // });

        // d_x_mem.ToDevice(h_x.data());
        // d_y_mem.ToDevice(h.data());

        // using Problem = ck_tile::Reduce2dProblem<XDataType,
        //                                          ComputeDataType,
        //                                          YDataType,
        //                                          TestReduce2dShape,
        //                                          ReduceOpsType,
        //                                          KeptDimSeq,
        //                                          ReduceDimSeq,
        //                                          InputDim>;

        // using Kernel = ck_tile::MultiReduceMultiblock<Problem>;

        // // Launch configuration
        // const ck_tile::index_t kBlockSize      = Kernel::BlockSize();
        // constexpr ck_tile::index_t kBlockPerCu = 1;

        // auto elementwise_ops =
        //     make_elementwise_ops_tuple(total_reduce_elements, ElementwiseOpsType{});
        // auto accumulator_ops =
        //     make_elementwise_ops_tuple(total_reduce_elements, AccumulatorOpsType{});

        // auto [num_block_tile_iterations, block_group_size] =
        //     typename Kernel::TilePartitioner{total_reduce_elements}.GetBlockGroupParams();

        // std::cout << "Block group size: " << block_group_size
        //           << ", Num block tile iterations: " << num_block_tile_iterations
        //           << ", Reduce total length: " << total_reduce_elements << std::endl;

        // ck_tile::index_t kGridSize =
        //     ((kept_dim_len_prod + TestReduce2dShape::Block_M - 1) / TestReduce2dShape::Block_M) *
        //     block_group_size;

        // // Generic helper to create tuple from vector based on compile-time size
        // auto make_shape_tuple = []<std::size_t N>(const std::vector<ck_tile::index_t>& vec) {
        //     return [&vec]<std::size_t... I>(std::index_sequence<I...>) {
        //         return ck_tile::make_tuple(vec[I]...);
        //     }(std::make_index_sequence<N>{});
        // };

        // auto input_shape_tuple   = make_shape_tuple.template operator()<InputDim>(input_shape);
        // auto input_strides_tuple = make_shape_tuple.template operator()<InputDim>(input_strides);

        // if(!Kernel::IsSupportedArgument()) // TODO
        // {
        // }

        // ck_tile::launch_kernel(
        //     ck_tile::stream_config{nullptr, false, 0},
        //     ck_tile::make_kernel<kBlockPerCu>(Kernel{},
        //                                       kGridSize,
        //                                       kBlockSize,
        //                                       0,
        //                                       static_cast<XDataType*>(d_x_mem.GetDeviceBuffer()),
        //                                       static_cast<YDataType*>(d_y_mem.GetDeviceBuffer()),
        //                                       input_shape_tuple,
        //                                       input_strides_tuple,
        //                                       kept_dims,
        //                                       reduce_dims,
        //                                       output_number_elements,
        //                                       elementwise_ops,
        //                                       accumulator_ops,
        //                                       InterBlockReduceOpsType{}));

        // TODO: Reference computation + Transfer data back to host
        EXPECT_TRUE(true);
    }
};
