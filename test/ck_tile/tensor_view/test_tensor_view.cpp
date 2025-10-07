// SPDX-License-Identifier: MIT
// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.

#include <hip/hip_runtime.h>
#include <gtest/gtest.h>
#include <vector>

#include "ck_tile/core.hpp"
#include "ck_tile/core/tensor/tensor_view.hpp"
#include "ck_tile/core/tensor/tensor_descriptor.hpp"
#include "ck_tile/core/tensor/tensor_coordinate.hpp"
#include "ck_tile/core/tensor/tile_window.hpp"
#include "ck_tile/host/hip_check_error.hpp"
#include "ck_tile/core/tensor/store_tile.hpp"
#include "ck_tile/core/algorithm/static_encoding_pattern.hpp"

using namespace ck_tile;

class TestTensorView : public ::testing::Test
{
    protected:
    void SetUp() override {}
    void TearDown() override {}
};

template <typename TensorView, typename MultiIndex>
__global__ void test_tensor_view_kernel(TensorView tw, MultiIndex idx_top, int* output, bool debug)
{
    const auto& tensor_desc = tw.get_tensor_descriptor();
    const auto idx_ndim     = make_tensor_coordinate(tensor_desc, idx_top);

    const index_t n_rows = tensor_desc.get_length(number<0>{});
    const index_t n_cols = tensor_desc.get_length(number<1>{});

    for(int i = 0; i < n_rows; ++i)
    {
        for(int j = 0; j < n_cols; ++j)
        {
            const index_t idx           = i * n_cols + j;
            const index_t linear_offset = tensor_desc.calculate_offset(make_multi_index(i, j));
            const int element = tw.template get_vectorized_elements<int>(idx_ndim, linear_offset);
            output[idx]       = element;
            if(debug)
            {
                printf("tw(%d,%d) = %d\n", i, j, element);
            }
        }
    }
}

template <typename TensorDesc, typename MultiIndex>
auto run_tensor_view_test(const TensorDesc& tensor_desc,
                          const MultiIndex& base_addr,
                          const std::vector<int>& data_host,
                          const std::vector<int>& expected_output_host,
                          bool debug = false)
{
    int* data_device;
    hip_check_error(hipMalloc(&data_device, data_host.size() * sizeof(int)));
    hip_check_error(hipMemcpy(
        data_device, data_host.data(), data_host.size() * sizeof(int), hipMemcpyHostToDevice));

    std::vector<int> output_host(expected_output_host.size(), 0);
    int* output_device;
    hip_check_error(hipMalloc(&output_device, expected_output_host.size() * sizeof(int)));
    hip_check_error(hipMemset(output_device, 0, expected_output_host.size() * sizeof(int)));

    const auto tw =
        make_tensor_view<address_space_enum::generic,
                         memory_operation_enum::set,
                         amd_buffer_coherence_enum::coherence_default>(data_device, tensor_desc);

    EXPECT_EQ(tw.get_num_of_dimension(), 2);

    test_tensor_view_kernel<<<1, 1>>>(tw, base_addr, output_device, debug);
    hip_check_error(hipMemcpy(output_host.data(),
                              output_device,
                              output_host.size() * sizeof(int),
                              hipMemcpyDeviceToHost));

    EXPECT_EQ(output_host, expected_output_host);

    hip_check_error(hipFree(data_device));
    hip_check_error(hipFree(output_device));

    return tw;
}

TEST_F(TestTensorView, BasicAccess1)
{
    // clang-format off
    std::vector<int> data_host = 
      {
        11, 12, 13, 14, 15,
        21, 22, 23, 24, 25,
        31, 32, 33, 34, 35,
        41, 42, 43, 44, 45,
        51, 52, 53, 54, 55,
        61, 62, 63, 64, 65,
        71, 72, 73, 74, 75,
        81, 82, 83, 84, 85
      };
    // clang-format on

    /*
    Base_ptr = &memory[0]
    Stride to next row = 5
    Stride to next column = 2
    Tensor Lengths: 2 rows, 3 columns
    Linear index = Base_ptr + i * 5 + j * 2, for multi-index (i,j).

    Memory Array: [0] [1] [2] [3] [4] [5] [6] [7] [8] [9] [10] [11] [12] [13] [14] ...
                   |           |           |           |             |
    Tensor Access:
    tensor(0,0) = memory[0*5 + 0*2] = memory[0] = 11
    tensor(0,1) = memory[0*5 + 1*2] = memory[2] = 13
    tensor(0,2) = memory[0*5 + 2*2] = memory[4] = 15
    tensor(1,0) = memory[1*5 + 0*2] = memory[5] = 21
    tensor(1,1) = memory[1*5 + 1*2] = memory[7] = 23
    tensor(1,2) = memory[1*5 + 2*2] = memory[9] = 25
    */
    constexpr auto base_addr   = make_multi_index(number<0>{}, number<0>{});
    constexpr auto tensor_desc = make_naive_tensor_descriptor(make_tuple(number<2>{}, number<3>{}),
                                                              make_tuple(number<5>{}, number<2>{}));

    // clang-format off
    const std::vector<int> expected_output_host = 
      {
        11, 13, 15,
        21, 23, 25
      };
    // clang-format on

    run_tensor_view_test(tensor_desc, base_addr, data_host, expected_output_host);
}

TEST_F(TestTensorView, BasicAccess2)
{
    // clang-format off
    std::vector<int> data_host = 
      {
        11, 12, 13, 14, 15,
        21, 22, 23, 24, 25,
        31, 32, 33, 34, 35,
        41, 42, 43, 44, 45,
        51, 52, 53, 54, 55,
        61, 62, 63, 64, 65,
        71, 72, 73, 74, 75,
        81, 82, 83, 84, 85
      };
    // clang-format on

    /*
    Stride to next row = 3
    Stride to next column = 4
    Tensor Lengths: 2 rows, 2 columns
    Global offset = 1 x stride_rows (3) + 2 x stride_columns (4) = 11
    Linear index = Global offset + i * stride_rows + j * stride_cols, for multi-index (i,j).

    Tensor Access:
    tensor(0,0) = memory[11 + 0*3 + 0*4] = memory[11] = 32
    tensor(0,1) = memory[11 + 0*3 + 1*4] = memory[15] = 41
    tensor(1,0) = memory[11 + 1*3 + 0*4] = memory[14] = 35
    tensor(1,1) = memory[11 + 1*3 + 1*4] = memory[18] = 44
    */
    constexpr auto base_addr   = make_multi_index(number<1>{}, number<2>{});
    constexpr auto tensor_desc = make_naive_tensor_descriptor(make_tuple(number<2>{}, number<2>{}),
                                                              make_tuple(number<3>{}, number<4>{}));

    // clang-format off
    const std::vector<int> expected_output_host = 
      {
        data_host[11], data_host[15],
        data_host[14], data_host[18]
      };
    // clang-format on

    run_tensor_view_test(tensor_desc, base_addr, data_host, expected_output_host);
}

TEST_F(TestTensorView, BasicAccess3)
{
    // clang-format off
    std::vector<int> data_host = 
      {
        11, 12, 13, 14, 15,
        21, 22, 23, 24, 25,
        31, 32, 33, 34, 35,
        41, 42, 43, 44, 45,
        51, 52, 53, 54, 55,
        61, 62, 63, 64, 65,
        71, 72, 73, 74, 75,
        81, 82, 83, 84, 85
      };
    // clang-format on

    /*
    Stride to next row = 5
    Stride to next column = 1
    Tensor Lengths: 3 rows, 3 columns
    Global offset = 4 x stride_rows (5) + 1 x stride_columns (1) = 21
    Linear index = Global offset + i * stride_rows + j * stride_cols, for multi-index (i,j).

    Tensor Access:
    tensor(0,0) = memory[21 + 0*3 + 0*1] = memory[21] = 52
    tensor(0,1) = memory[21 + 0*3 + 1*1] = memory[22] = 53
    tensor(0,2) = memory[21 + 0*3 + 2*1] = memory[23] = 54
    tensor(1,0) = memory[21 + 1*5 + 0*1] = memory[26] = 62
    tensor(1,1) = memory[21 + 1*5 + 1*1] = memory[27] = 63
    tensor(1,2) = memory[21 + 1*5 + 2*1] = memory[28] = 64
    tensor(2,0) = memory[21 + 2*5 + 0*1] = memory[31] = 62
    tensor(2,1) = memory[21 + 2*5 + 1*1] = memory[32] = 63
    tensor(2,2) = memory[21 + 2*5 + 2*1] = memory[33] = 64
    */
    constexpr auto base_addr   = make_multi_index(number<4>{}, number<1>{});
    constexpr auto tensor_desc = make_naive_tensor_descriptor(make_tuple(number<3>{}, number<3>{}),
                                                              make_tuple(number<5>{}, number<1>{}));

    const std::vector<int> expected_output_host = {52, 53, 54, 62, 63, 64, 72, 73, 74};

    run_tensor_view_test(tensor_desc, base_addr, data_host, expected_output_host);
}

__global__ void test_static_distributed_tensor_kernel(int* output)
{
    constexpr index_t MIterPerWarp = 2;
    constexpr index_t NIterPerWarp = 2;
    constexpr index_t MWarp        = 1;
    constexpr index_t NWarp        = 1;

    constexpr auto c_block_dstr_encoding = tile_distribution_encoding<
        sequence<>,
        tuple<sequence<MIterPerWarp, MWarp>, sequence<NIterPerWarp, NWarp>>,
        tuple<sequence<1, 2>>,
        tuple<sequence<1, 1>>,
        sequence<1, 2>,
        sequence<0, 0>>{};
    constexpr auto c_block_dstr = make_static_tile_distribution(c_block_dstr_encoding);
    auto c_block_tensor         = make_static_distributed_tensor<int>(c_block_dstr);

    *output = c_block_tensor.get_num_of_dimension();
}

TEST_F(TestTensorView, StaticDistributedTensor)
{
    int* output_device;
    hip_check_error(hipMalloc(&output_device, sizeof(int)));
    test_static_distributed_tensor_kernel<<<1, 1>>>(output_device);

    int output_host;
    hip_check_error(hipMemcpy(&output_host, output_device, sizeof(int), hipMemcpyDeviceToHost));
    EXPECT_GT(output_host, 0);
    hip_check_error(hipFree(output_device));
}

template <typename DistributedIndex>
__device__ void print_distributed_index(const DistributedIndex& idx)
{
    printf("[");
    for(auto i = 0; i < idx.impl_.size(); i++)
    {
        printf("%d", idx.impl_[i]);
        if(i < idx.impl_.size() - 1)
        {
            printf(", ");
        }
    }
    printf("]");
}

__global__ void test_4x4_matrix_2x2_blocks_modify_input_kernel(int* input, int* output, bool)
{
    constexpr index_t global_shape_0 = 4;
    constexpr index_t global_shape_1 = 4;

    // Tile distribution parameters
    constexpr index_t MRepeat          = 1;
    constexpr index_t NRepeat          = 1;
    constexpr index_t MWarpPerBlock    = 1;
    constexpr index_t NWarpPerBlock    = 1;
    constexpr index_t MThreadPerWarp   = 2;
    constexpr index_t NThreadPerWarp   = 2;
    constexpr index_t MVectorPerThread = 2;
    constexpr index_t NVectorPerThread = 2;

    // Tile distribution encoding for 4x4 matrix as 2x2 blocks
    constexpr auto encoding = tile_distribution_encoding<
        sequence<>, // No reduction dims
        tuple<
            // [H1_0, H1_1, H1_2, H1_3]
            sequence<MRepeat, MWarpPerBlock, MThreadPerWarp, MVectorPerThread>, // M-dim: 1 rep, 1
                                                                                // warp, 2 threads,
                                                                                // 2 elements per
                                                                                // thread
            // [H2_0, H2_1, H2_2, H2_3]
            sequence<NRepeat, NWarpPerBlock, NThreadPerWarp, NVectorPerThread>>, // N-dim: 1 rep, 1
                                                                                 // warp, 2 threads,
                                                                                 // 2 elements per
                                                                                 // thread
        // P minor and major combined:
        // P1 -> (H1_1, H2_1) and P2 -> (H1_2, H2_2)
        tuple<sequence<1, 2>, sequence<1, 2>>, // P major(Warp) -> H mapping
        tuple<sequence<1, 1>, sequence<2, 2>>, // P minor(Thread) -> H mapping
        // Combined mapping
        // First row: Y -> {H1,H2} mapping
        // Second row: which in H dim (0,1,2,3) we map Y to
        // Y0 -> H1_0, Y1 -> H1_3, Y2 -> H2_0, Y3 -> H2_3
        sequence<1, 1, 2, 2>,    // Trivial since we have only warp
        sequence<0, 3, 0, 3>>{}; // Map thread id to number of elements per thread (Hi_3)

    auto distribution = make_static_tile_distribution(encoding);

    constexpr auto hs_lengths_0 = encoding.hs_lengthss_[number<0>{}];
    constexpr auto hs_lengths_1 = encoding.hs_lengthss_[number<1>{}];

    constexpr index_t x0_size = reduce_on_sequence(hs_lengths_0, multiplies{}, number<1>{});
    constexpr index_t x1_size = reduce_on_sequence(hs_lengths_1, multiplies{}, number<1>{});

    if(threadIdx.x == 0 && blockIdx.x == 0)
    {
        printf("\n- Tile distribution created:\n");
        printf("  X dimensions: %d\n", distribution.get_num_of_dimension_x());
        printf("  Y dimensions: %d\n", distribution.get_num_of_dimension_y());
        printf("  P dimensions: %d\n", distribution.get_num_of_dimension_p());
        printf("  X lengths: [%d, %d]\n", x0_size, x1_size);
    }
    block_sync_lds();

    auto global_view = make_naive_tensor_view_packed<address_space_enum::global>(
        input, make_tuple(global_shape_0, global_shape_1));

    const auto window_lengths = make_tuple(x0_size, x1_size);
    auto tile_window          = make_tile_window(global_view,
                                        window_lengths,
                                                 {0, 0}, // Window origin as initializer list
                                        distribution);
    auto distributed_tensor   = tile_window.load();

    // Create output tensor view
    auto output_global_view = make_naive_tensor_view_packed<address_space_enum::global>(
        output, make_tuple(global_shape_0, global_shape_1));

    // Create output tile window with the same distribution
    auto output_tile_window = make_tile_window(output_global_view,
                                               window_lengths,
                                               {0, 0}, // Same window origin
                                               distribution);

    // Create a new distributed tensor for output (copy from input or modify)
    auto output_distributed_tensor = distributed_tensor; // Copy the loaded data

    // Modify the data before storing
    sweep_tile(output_distributed_tensor,
               [&](auto idx) { output_distributed_tensor(idx) = distributed_tensor(idx) * 2; });

    // Store the distributed tensor to the output
    output_tile_window.store(output_distributed_tensor);
}

TEST_F(TestTensorView, StaticDistributedTensor4x4Matrix2x2Blocks_modify_input)
{
    // clang-format off
    std::vector<int> data_host = 
      {
        1, 2, 3 ,4,
        5, 6, 7, 8, 
        9, 10, 11, 12,
        13, 14, 15, 16
      };
    // clang-format on

    constexpr int total_elements = 16; // 4x4 matrix = 16 elements
    std::vector<int> output_host(total_elements, 0);
    int* output_device;

    int* input_device;
    hip_check_error(hipMalloc(&input_device, data_host.size() * sizeof(int)));
    hip_check_error(hipMemcpy(
        input_device, data_host.data(), data_host.size() * sizeof(int), hipMemcpyHostToDevice));

    hip_check_error(hipMalloc(&output_device, total_elements * sizeof(int)));
    hip_check_error(hipMemset(output_device, 0, total_elements * sizeof(int)));

    // Run kernel with debug output
    const dim3 block_dim(4); // 4 threads to cover 4 blocks
    const dim3 grid_dim(1);
    test_4x4_matrix_2x2_blocks_modify_input_kernel<<<grid_dim, block_dim>>>(
        input_device, output_device, true);
    hip_check_error(hipDeviceSynchronize());

    // Copy results back
    hip_check_error(hipMemcpy(
        output_host.data(), output_device, total_elements * sizeof(int), hipMemcpyDeviceToHost));

    // Verify the 4x4 matrix is correctly organized as 2x2 blocks
    // Expected matrix:
    //  2  4  6  8
    //  10 12 14 16
    //  18 20 22 24
    //  26 28 30 32

    std::vector<int> expected_output = {2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32};

    EXPECT_EQ(output_host, expected_output);

    hip_check_error(hipFree(output_device));
    hip_check_error(hipFree(input_device));
}

__global__ void test_4x4_matrix_get_2x2_blocks_kernel(int* input, int* output)
{
    constexpr index_t global_shape_0 = 4;
    constexpr index_t global_shape_1 = 4;

    // Tile distribution parameters
    constexpr index_t MRepeat          = 1;
    constexpr index_t NRepeat          = 1;
    constexpr index_t MWarpPerBlock    = 1;
    constexpr index_t NWarpPerBlock    = 1;
    constexpr index_t MThreadPerWarp   = 2;
    constexpr index_t NThreadPerWarp   = 2;
    constexpr index_t MVectorPerThread = 2;
    constexpr index_t NVectorPerThread = 2;

    // Tile distribution encoding for 4x4 matrix as 2x2 blocks
    constexpr auto encoding = tile_distribution_encoding<
        sequence<>, // No reduction dims
        tuple<
            // [H1_0, H1_1, H1_2, H1_3]
            sequence<MRepeat, MWarpPerBlock, MThreadPerWarp, MVectorPerThread>, // M-dim: 1 rep, 1
                                                                                // warp, 2 threads,
                                                                                // 2 elements per
                                                                                // thread
            // [H2_0, H2_1, H2_2, H2_3]
            sequence<NRepeat, NWarpPerBlock, NThreadPerWarp, NVectorPerThread>>, // N-dim: 1 rep, 1
                                                                                 // warp, 2 threads,
                                                                                 // 2 elements per
                                                                                 // thread
        // P minor and major combined:
        // P1 -> (H1_1, H2_1) and P2 -> (H1_2, H2_2)
        tuple<sequence<1, 2>, sequence<1, 2>>, // P major(Warp) -> H mapping
        tuple<sequence<1, 1>, sequence<2, 2>>, // P minor(Thread) -> H mapping
        // Combined mapping
        // First row: Y -> {H1,H2} mapping
        // Second row: which in H dim (0,1,2,3) we map Y to
        // Y0 -> H1_0, Y1 -> H1_3, Y2 -> H2_0, Y3 -> H2_3
        sequence<1, 1, 2, 2>,    // Trivial since we have only warp
        sequence<0, 3, 0, 3>>{}; // Map thread id to number of elements per thread (Hi_3)

    auto distribution = make_static_tile_distribution(encoding);

    constexpr auto hs_lengths_0 = encoding.hs_lengthss_[number<0>{}];
    constexpr auto hs_lengths_1 = encoding.hs_lengthss_[number<1>{}];

    constexpr index_t x0_size = reduce_on_sequence(hs_lengths_0, multiplies{}, number<1>{});
    constexpr index_t x1_size = reduce_on_sequence(hs_lengths_1, multiplies{}, number<1>{});

    auto global_view = make_naive_tensor_view_packed<address_space_enum::global>(
        input, make_tuple(global_shape_0, global_shape_1));

    const auto window_lengths = make_tuple(x0_size, x1_size);
    auto tile_window          = make_tile_window(global_view,
                                        window_lengths,
                                                 {0, 0}, // Window origin as initializer list
                                        distribution);
    auto distributed_tensor   = tile_window.load();

    // Up to this point, we have set-up the distributed tensor for the 4x4 matrix
    // similar to the output of the MFMA when 2 conv groups are merged.
    // We want to copy only the diagonal 2x2 blocks to the output, similar to the epilogue
    // part of batched iGEMM for 2 conv groups.

    // Create output encoding for 4x2 matrix (2 diagonal blocks stacked vertically)
    // The threads access the output distributed tensor in a sliced manner
    // T0 -> row 0
    // T1 -> row 1 (this is masked out when copying from input)
    // T2 -> row 2 (this is masked out when copying from input)
    // T3 -> row 3
    // Each thread copies 2 elements (a row of a 2x2 block)
    constexpr auto output_encoding =
        tile_distribution_encoding<sequence<>,
                                   tuple<
                                       // M dimension: 2 diagonal blocks * 2 rows each = 4 total
                                       // rows
                                       sequence<1, 1, 4, 1>,
                                       // N dimension: 2 columns per block
                                       sequence<1, 1, 1, 2>>,
                                   tuple<sequence<1, 2>, sequence<1, 2>>,
                                   tuple<sequence<1, 1>, sequence<2, 2>>,
                                   sequence<1, 1, 1, 2>,
                                   sequence<0, 3, 0, 3>>{};

    auto output_distribution       = make_static_tile_distribution(output_encoding);
    auto output_distributed_tensor = make_static_distributed_tensor<int>(output_distribution);
    auto output_global_view =
        make_naive_tensor_view_packed<address_space_enum::global>(output, make_tuple(4, 2));

    auto get_block_number = [&]() -> ck_tile::tuple<index_t, index_t> {
        constexpr index_t m_size = 2;
        constexpr index_t n_size = 2;
        const auto x_space_coord = distribution.calculate_index();
        const index_t m_block    = x_space_coord[0] / m_size;
        const index_t n_block    = x_space_coord[1] / n_size;
        return make_tuple(m_block, n_block);
    };

    auto mask = [&]() -> bool {
        // Return true only for the diagonal blocks.
        const auto blockId = get_block_number();
        return blockId[number<0>{}] == blockId[number<1>{}];
    };

    auto get_output_row_offset = [&](auto row) -> index_t {
        const auto blockId      = get_block_number();
        const auto block_id_row = blockId[number<0>{}];
        return row - block_id_row;
    };

    // Because we copy one row at the time, we need to loop over the 2 rows of the 2x2 blocks.
    static_for<0, 2, 1>{}([&](auto row) {
        // We mask out the threads that do not contribute to the diagonal blocks.
        if(mask())
        {
            output_distributed_tensor.get_thread_buffer() =
                distributed_tensor.get_y_sliced_thread_data(
                    sequence<0, 0, row, 0>{},
                    sequence<1, 1, 1, 2>{}); // copy one row of a 2x2 block

            const auto row_offset   = get_output_row_offset(row);
            auto output_tile_window = make_tile_window(
                output_global_view, make_tuple(4, 2), {row_offset, 0}, output_distribution);

            output_tile_window.store(output_distributed_tensor);
        }
    });
}

TEST_F(TestTensorView, StaticDistributedTensor4x4Matrix2x2Blocks_get_sub_blocks)
{
    // clang-format off
    std::vector<int> data_host = 
      {
        1, 2, 3 ,4,
        5, 6, 7, 8, 
        9, 10, 11, 12,
        13, 14, 15, 16
      };
    // clang-format on

    constexpr int total_elements = 8; // 2 times 2 x 2 matrix = 8 elements
    std::vector<int> output_host(total_elements, 0);
    int* output_device;

    int* input_device;
    hip_check_error(hipMalloc(&input_device, data_host.size() * sizeof(int)));
    hip_check_error(hipMemcpy(
        input_device, data_host.data(), data_host.size() * sizeof(int), hipMemcpyHostToDevice));

    hip_check_error(hipMalloc(&output_device, total_elements * sizeof(int)));
    hip_check_error(hipMemset(output_device, 0, total_elements * sizeof(int)));

    // Run kernel with debug output
    const dim3 block_dim(4); // 4 threads to cover 4 blocks
    const dim3 grid_dim(1);

    test_4x4_matrix_get_2x2_blocks_kernel<<<grid_dim, block_dim>>>(input_device, output_device);
    hip_check_error(hipDeviceSynchronize());

    // Copy results back
    hip_check_error(hipMemcpy(
        output_host.data(), output_device, total_elements * sizeof(int), hipMemcpyDeviceToHost));

    // Verify the 4x4 matrix is correctly organized as 2x2 blocks
    // Expected matrix:
    //  1  2
    //  5  6
    //  11 12
    //  15 16

    std::vector<int> expected_output = {1, 2, 5, 6, 11, 12, 15, 16};

    EXPECT_EQ(output_host, expected_output);

    hip_check_error(hipFree(output_device));
    hip_check_error(hipFree(input_device));
}

__global__ void test_4x4_matrix_get_2x2_blocks_with_sfc_and_lds_kernel(int* input, int* output)
{
    constexpr index_t MPerBlock = 4;
    constexpr index_t NPerBlock = 4;

    // Tile distribution encoding for 4x4 matrix as 2x2 blocks
    constexpr auto encoding = tile_distribution_encoding<
        sequence<>, // No reduction dims
        tuple<
            // [H1_0, H1_1, H1_2, H1_3]
            sequence<1, 1, 2, 2>, // M-dim: 1 rep, 1 warp, 2 threads, 2 elements per thread
            // [H2_0, H2_1, H2_2, H2_3]
            sequence<1, 1, 2, 2>>, // N-dim: 1 rep, 1 warp, 2 threads, 2 elements per thread
        // P minor and major combined:
        // P1 -> (H1_1, H2_1) and P2 -> (H1_2, H2_2)
        tuple<sequence<1, 2>, sequence<1, 2>>, // P major(Warp) -> H mapping
        tuple<sequence<1, 1>, sequence<2, 2>>, // P minor(Thread) -> H mapping
        // Combined mapping
        // First row: Y -> {H1,H2} mapping
        // Second row: which in H dim (0,1,2,3) we map Y to
        // Y0 -> H1_0, Y1 -> H1_3, Y2 -> H2_0, Y3 -> H2_3
        sequence<1, 1, 2, 2>,    // Trivial since we have only warp
        sequence<0, 3, 0, 3>>{}; // Map thread id to number of elements per thread (Hi_3)

    auto distribution = make_static_tile_distribution(encoding);

    auto global_view = make_naive_tensor_view_packed<address_space_enum::global>(
        input, make_tuple(MPerBlock, NPerBlock));

    const auto window_lengths = make_tuple(MPerBlock, NPerBlock);
    auto input_tile_window    = make_tile_window(global_view,
                                              window_lengths,
                                                 {0, 0}, // Window origin as initializer list
                                              distribution);
    auto input_tensor         = input_tile_window.load();

    // Up to this point, we have set-up the distributed tensor for the 4x4 matrix
    // similar to the output of the MFMA when 2 conv groups are merged.
    // We want to copy only the diagonal 2x2 blocks to the output, similar to the epilogue
    // part of batched iGEMM for 2 conv groups.
    constexpr index_t NumGroupsToMerge = 2; // Number of merged groups

    // Create tensor descriptor for the output 4x2 matrix (2 diagonal blocks stacked vertically)
    auto output_view = make_naive_tensor_view_packed<address_space_enum::global>(
        output, make_tuple(MPerBlock, NPerBlock / NumGroupsToMerge));

    auto output_window =
        make_tile_window(output_view,
                         make_tuple(number<MPerBlock>{}, number<NPerBlock / NumGroupsToMerge>{}),
                         {0, 0}); // We have only threadblock

    //------------------------------------------------------------
    // CShuffle epilogue similation
    //------------------------------------------------------------

    // Allocate and prepare LDS
    __shared__ char p_smem[MPerBlock * NPerBlock * sizeof(int)];

    // Initialize the LDS to zero
    if(threadIdx.x == 0 && blockIdx.x == 0)
    {
        int* lds_data = reinterpret_cast<int*>(p_smem);
        for(index_t i = 0; i < MPerBlock; i++)
        {
            for(index_t j = 0; j < NPerBlock; j++)
            {
                lds_data[i * NPerBlock + j] = 0;
            }
        }
    }
    __syncthreads();

    constexpr index_t MPerIterationShuffle = 2;
    constexpr index_t NPerIterationShuffle = 2;

    auto lds_tile = make_static_distributed_tensor<int>(distribution);

    constexpr auto lds_block_desc =
        make_naive_tensor_descriptor(make_tuple(number<MPerBlock>{}, number<NPerBlock>{}),
                                     make_tuple(number<NPerBlock>{}, number<1>{}));

    auto o_lds_block =
        make_tensor_view<address_space_enum::lds>(reinterpret_cast<int*>(p_smem), lds_block_desc);

    auto in_lds_window =
        make_tile_window(o_lds_block,
                         make_tuple(number<MPerIterationShuffle>{}, number<NPerIterationShuffle>{}),
                         {0, 0},
                         distribution);

    // Row major thread to data mapping in LDS
    auto out_lds_window =
        make_tile_window(o_lds_block,
                         make_tuple(number<MPerIterationShuffle>{}, number<NPerIterationShuffle>{}),
                         {0, 0});

    // Set-up traversing the 2x2 blocks
    using SFC = space_filling_curve<sequence<MPerBlock, NPerBlock>,
                                    sequence<0, 1>,
                                    sequence<MPerIterationShuffle, NPerIterationShuffle>,
                                    false>;

    using SFC_dram = space_filling_curve<sequence<MPerBlock, NPerBlock / NumGroupsToMerge>,
                                         sequence<0, 1>,
                                         sequence<MPerIterationShuffle, NPerIterationShuffle>,
                                         false>;

    using TileEncodingPattern =
        tile_distribution_encoding_pattern_2d<4, // Block size
                                              MPerIterationShuffle,
                                              NPerIterationShuffle,
                                              2, // Vector size
                                              tile_distribution_pattern::sparse_row,
                                              1>; // Number of wave groups

    constexpr auto output_tile_distribution =
        TileEncodingPattern::make_2d_static_tile_distribution();

    // Copy the tile at one go from register to LDS.
    block_sync_lds();

    lds_tile.get_thread_buffer() = input_tensor.get_y_sliced_thread_data(
        sequence<0, 0, 0, 0>{}, sequence<1, 1, MPerIterationShuffle, MPerIterationShuffle>{});

    store_tile(in_lds_window, lds_tile);
    block_sync_lds();

    // Print the contents of LDS
    if(threadIdx.x == 0 && blockIdx.x == 0)
    {
        printf("LDS contents:\n");
        int* lds_data = reinterpret_cast<int*>(p_smem);
        for(index_t i = 0; i < 4; i++)
        {
            for(index_t j = 0; j < 4; j++)
            {
                printf("%3d ", lds_data[i * 4 + j]);
            }
            printf("\n");
        }
    }

    // For the output tensor, we need to copy only the diagonal 2x2 blocks to global memory.
    static_for<0, NumGroupsToMerge, 1>{}([&](auto group) {
        auto out_tensor = load_tile(make_tile_window(out_lds_window, output_tile_distribution));

        store_tile(output_window, out_tensor);

        // Print the output tensor contents.
        __syncthreads();
        if(threadIdx.x == 0 && blockIdx.x == 0)
        {

            printf("Output tensor contents after loading group %d:\n", group.value);
            for(index_t i = 0; i < 4; i++)
            {
                for(index_t j = 0; j < 2; j++)
                {
                    printf("%3d", output[i * 2 + j]);
                }
                printf("\n");
            }
        }
        __syncthreads();

        // Moving output window works correctly.
        if constexpr(group != NumGroupsToMerge - 1)
        {
            constexpr auto step = SFC_dram::get_forward_step(group);
            move_tile_window(output_window, {step.at(number<0>{}), step.at(number<1>{})});

            constexpr auto iAccess      = number<group * NumGroupsToMerge + group>{};
            constexpr auto next_iAccess = number<(group + 1) * NumGroupsToMerge + (group + 1)>{};
            constexpr auto step_lds     = SFC::get_step_between(iAccess, next_iAccess);
            move_tile_window(out_lds_window, {step_lds.at(number<0>{}), step_lds.at(number<1>{})});
        }
    });
}

TEST_F(TestTensorView, StaticDistributedTensor4x4Matrix2x2Blocks_get_sub_blocks_SFC)
{
    // clang-format off
    std::vector<int> data_host = 
      {
        1, 2, 3 ,4,
        5, 6, 7, 8, 
        9, 10, 11, 12,
        13, 14, 15, 16
      };
    // clang-format on

    constexpr int total_elements = 8; // 2 times 2 x 2 matrix = 8 elements
    std::vector<int> output_host(total_elements, 0);
    int* output_device;

    int* input_device;
    hip_check_error(hipMalloc(&input_device, data_host.size() * sizeof(int)));
    hip_check_error(hipMemcpy(
        input_device, data_host.data(), data_host.size() * sizeof(int), hipMemcpyHostToDevice));

    hip_check_error(hipMalloc(&output_device, total_elements * sizeof(int)));
    hip_check_error(hipMemset(output_device, 0, total_elements * sizeof(int)));

    // Run kernel with debug output
    const dim3 block_dim(4); // 4 threads to cover 4 blocks
    const dim3 grid_dim(1);

    test_4x4_matrix_get_2x2_blocks_with_sfc_and_lds_kernel<<<grid_dim, block_dim>>>(input_device,
                                                                                    output_device);
    hip_check_error(hipDeviceSynchronize());

    // Copy results back
    hip_check_error(hipMemcpy(
        output_host.data(), output_device, total_elements * sizeof(int), hipMemcpyDeviceToHost));

    // Verify the 4x4 matrix is correctly organized as 2x2 blocks
    // Expected matrix:
    //  1  2
    //  5  6
    //  11 12
    //  15 16

    std::vector<int> expected_output = {1, 2, 5, 6, 11, 12, 15, 16};

    EXPECT_EQ(output_host, expected_output);

    hip_check_error(hipFree(output_device));
    hip_check_error(hipFree(input_device));
}
