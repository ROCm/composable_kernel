#include <hip/hip_runtime.h>
#include <gtest/gtest.h>
#include <vector>

#include "ck_tile/core.hpp"
#include "ck_tile/core/tensor/tensor_view.hpp"
#include "ck_tile/core/tensor/tensor_descriptor.hpp"
#include "ck_tile/core/tensor/tensor_coordinate.hpp"
#include "ck_tile/core/tensor/tile_window.hpp"
#include "ck_tile/host/hip_check_error.hpp"

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
    const auto idx_ndim = make_tensor_coordinate(tensor_desc, idx_top);

    const index_t n_rows = tensor_desc.get_length(number<0>{});
    const index_t n_cols = tensor_desc.get_length(number<1>{});

    for (int i = 0; i < n_rows; ++i)
    {
        for (int j = 0; j < n_cols; ++j)
        {
            const index_t idx = i * n_cols + j;
            const index_t linear_offset = tensor_desc.calculate_offset(make_multi_index(i, j));
            const int element = tw.template get_vectorized_elements<int>(idx_ndim, linear_offset);
            output[idx] = element;
            if (debug)
            {
              printf("tw(%d,%d) = %d\n", i, j, element);
            }
        }
    }
}

// template <typename TensorView, typename WindowLengths, typename MultiIndex>
// __global__ void test_tile_window_kernel(
//     TensorView tw, WindowLengths window_lengths, MultiIndex origin, int* output, bool debug)
// {
//     auto tile_window = make_tile_window(tw, window_lengths, origin);
//     const index_t n_rows = window_lengths[number<0>{}];
//     const index_t n_cols = window_lengths[number<1>{}];

//     const auto tile_data = tile_window.load();

//     for (auto i = 0; i < n_rows; ++i)
//     {
//       for (auto j = 0; j < n_cols; ++j)
//       {
//         const index_t idx = i * n_cols + j;
//         const int element = tile_data.at(number<idx>{});
//         output[idx] = element;
//         if (debug)
//         {
//           printf("tile_window(%d,%d) = %d\n", i, j, element);
//         }
//       }
//     }

// }

template <typename TensorDesc, typename MultiIndex>
auto run_tensor_view_test(const TensorDesc& tensor_desc, 
              const MultiIndex& base_addr,
              const std::vector<int>& data_host, 
              const std::vector<int>& expected_output_host,
              bool debug = false)
{
    int* data_device;
    hip_check_error(hipMalloc(&data_device, data_host.size() * sizeof(int)));
    hip_check_error(hipMemcpy(data_device, data_host.data(), data_host.size() * sizeof(int), 
                                                                        hipMemcpyHostToDevice));

    std::vector<int> output_host(expected_output_host.size(), 0);
    int* output_device;
    hip_check_error(hipMalloc(&output_device, expected_output_host.size() * sizeof(int)));
    hip_check_error(hipMemset(output_device, 0, expected_output_host.size() * sizeof(int)));
    
    const auto tw = make_tensor_view<
        address_space_enum::generic,
        memory_operation_enum::set,
        amd_buffer_coherence_enum::coherence_default>(
      data_device, 
      tensor_desc
    );

    EXPECT_EQ(tw.get_num_of_dimension(), 2);

    test_tensor_view_kernel<<<1, 1>>>(tw, base_addr, output_device, debug);
    hip_check_error(hipMemcpy(
      output_host.data(), output_device, output_host.size() * sizeof(int), hipMemcpyDeviceToHost));

    EXPECT_EQ(output_host, expected_output_host);

    hip_check_error(hipFree(data_device));
    hip_check_error(hipFree(output_device));

    return tw;
}

// template <typename TensorView, typename WindowLengths, typename MultiIndex>
// auto run_tile_window_test(
//               const TensorView tensor_view, 
//               const WindowLengths window_lengths,
//               const MultiIndex& origin,
//               const std::vector<int>& expected_output_host,
//               bool debug = true)
// {
//     std::vector<int> output_host(expected_output_host.size(), 0);
//     int* output_device;
//     hip_check_error(hipMalloc(&output_device, expected_output_host.size() * sizeof(int)));
//     hip_check_error(hipMemset(output_device, 0, expected_output_host.size() * sizeof(int)));

//     test_tile_window_kernel<<<1, 1>>>(tensor_view, window_lengths, origin, output_device, debug);
//     hip_check_error(hipMemcpy(
//       output_host.data(), output_device, output_host.size() * sizeof(int), hipMemcpyDeviceToHost));

//     EXPECT_EQ(output_host, expected_output_host);
// }

TEST_F(TestTensorView, BasicAccess1)
{
    // clang format-off
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
    // clang format-on

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
    constexpr auto tensor_desc = make_naive_tensor_descriptor(
      make_tuple(number<2>{}, number<3>{}),
      make_tuple(number<5>{}, number<2>{})  
    );
    
    // clang format-off
    const std::vector<int> expected_output_host = 
      {
        11, 13, 15,
        21, 23, 25
      };
    // clang format-on

    run_tensor_view_test(tensor_desc, base_addr, data_host, expected_output_host);
}

TEST_F(TestTensorView, BasicAccess2)
{
    // clang format-off
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
    // clang format-on

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
    constexpr auto tensor_desc = make_naive_tensor_descriptor(
      make_tuple(number<2>{}, number<2>{}),
      make_tuple(number<3>{}, number<4>{})  
    );
    
    // clang format-off
    const std::vector<int> expected_output_host = 
      {
        data_host[11], data_host[15],
        data_host[14], data_host[18]
      };
    // clang format-on
    
    run_tensor_view_test(tensor_desc, base_addr, data_host, expected_output_host);
}

TEST_F(TestTensorView, BasicAccess3)
{
    // clang format-off
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
    // clang format-on

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
    constexpr auto base_addr = make_multi_index(number<4>{}, number<1>{});
    constexpr auto tensor_desc = make_naive_tensor_descriptor(
      make_tuple(number<3>{}, number<3>{}),
      make_tuple(number<5>{}, number<1>{})  
    );
    
    const std::vector<int> expected_output_host = 
      {
        52, 53, 54,
        62, 63, 64,
        72, 73, 74
      };
    
    run_tensor_view_test(tensor_desc, base_addr, data_host, expected_output_host);
}

// TEST_F(TestTensorView, CreateTileWindow)
// {
//     // clang format-off
//     std::vector<int> data_host = 
//       {
//         11, 12, 13, 14, 15,
//         21, 22, 23, 24, 25,
//         31, 32, 33, 34, 35,
//         41, 42, 43, 44, 45,
//         51, 52, 53, 54, 55,
//         61, 62, 63, 64, 65,
//         71, 72, 73, 74, 75,
//         81, 82, 83, 84, 85
//       };
//     // clang format-on

//     /*
//     Create a view to to the full data
//     */
//     constexpr auto base_addr = make_multi_index(number<0>{}, number<0>{});
//     constexpr auto tensor_desc = make_naive_tensor_descriptor(
//       make_tuple(number<8>{}, number<5>{}),
//       make_tuple(number<5>{}, number<1>{})  
//     );
    
//     const auto& tw_full = run_tensor_view_test(tensor_desc, base_addr, data_host, data_host);

//     const std::vector<int> expected_output_host = 
//       {
//         51, 52, 53, 54, 55,
//         61, 62, 63, 64, 65,
//         71, 72, 73, 74, 75,
//         81, 82, 83, 84, 85
//       };
    
//     run_tile_window_test(
//       tw_full,          // tensor view to the original data
//       // Create a tile_window to the bottom half of the tensor_view. 
//       make_tuple(5, 4),                 // window lengths
//       make_multi_index(4, 0),           // origin in the tensor view
//       expected_output_host);   
// }

__global__ void test_static_distributed_tensor_kernel(int* output)
{
  constexpr index_t MIterPerWarp = 2;
  constexpr index_t NIterPerWarp = 2;
  constexpr index_t MWarp = 1;
  constexpr index_t NWarp = 1;

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
  test_static_distributed_tensor_kernel<<<1,1>>>(output_device);

  int output_host;
  hip_check_error(hipMemcpy(
      &output_host, output_device, sizeof(int), hipMemcpyDeviceToHost));
  EXPECT_GT(output_host, 0);
  hip_check_error(hipFree(output_device));
}

template <typename DistributedIndex>
__device__ void print_distributed_index(const DistributedIndex& idx)
{
    printf("[");
    for (auto i=0; i < idx.impl_.size(); i++) 
    {
        printf("%d", idx.impl_[i]);
        if (i < idx.impl_.size() - 1) 
        {
            printf(", ");
        }
    }
    printf("]");
}

__global__ void test_4x4_matrix_2x2_blocks_kernel(int* input, int* output, bool)
{
    constexpr index_t x0_size = 4;
    constexpr index_t x1_size = 4;
    auto global_view = make_naive_tensor_view_packed<address_space_enum::global>(
            input_data, make_tuple(x0_size, x1_size));

    constexpr index_t MRepeat = 1;
    constexpr index_t NRepeat = 1;
    constexpr index_t MWarpPerBlock = 1;
    constexpr index_t NWarpPerBlock = 1;
    constexpr index_t MThreadPerWarp = 2;
    constexpr index_t NThreadPerWarp = 2;
    constexpr index_t MVectorPerThread = 2;
    constexpr index_t NVectorPerThread = 2;

    // Tile distribution encoding for 4x4 matrix as 2x2 blocks
    constexpr auto matrix_4x4_dstr_encoding = tile_distribution_encoding<
        sequence<>,                                                      // No reduction dims
        tuple<
            // [H1_0, H1_1, H1_2, H1_3]
            sequence<MRepeat, MWarpPerBlock, MThreadPerWarp, MVectorPerThread>,  // M-dim: 1 rep, 1 warp, 2 threads, 2 elements per thread
            // [H2_0, H2_1, H2_2, H2_3]
            sequence<NRepeat, NWarpPerBlock, NThreadPerWarp, NVectorPerThread>>, // N-dim: 1 rep, 1 warp, 2 threads, 2 elements per thread
        // P minor and major combined:
        // P1 -> (H1_1, H2_1) and P2 -> (H1_2, H2_2)
        tuple<sequence<1, 2>, sequence<1,2>>,                                    // P major(Warp) -> H mapping
        tuple<sequence<1, 1>, sequence<2,2>>,                                    // P minor(Thread) -> H mapping
        // Combined mapping
        // First row: Y -> {H1,H2} mapping
        // Second row: which in H dim (0,1,2,3) we map Y to
        // Y0 -> H1_0, Y1 -> H1_3, Y2 -> H2_0, Y3 -> H2_3 
        sequence<1, 1, 2, 2>,                                                    // Trivial since we have only warp
        sequence<0, 3, 0, 3>>{};                                                 // Map thread id to number of elements per thread (Hi_3)

    auto distribution = make_static_tile_distribution(matrix_4x4_dstr_encoding);

    const auto window_lengths = make_tuple(x0_size, x1_size);
    auto tile_window = make_tile_window(global_view,
                                            window_lengths,
                                            {0, 0}, // Window origin as initializer list
                                            distribution);
    auto distributed_tensor = tile_window.load();



    constexpr auto matrix_4x4_dstr = make_static_tile_distribution(matrix_4x4_dstr_encoding);
    auto distributed_matrix = make_static_distributed_tensor<int>(matrix_4x4_dstr);

    // Initialize the 4x4 matrix with values 1-16
    constexpr auto matrix_spans = distributed_matrix.get_distributed_spans();

    // Initialize matrix with row-major values (1-16)
    sweep_tile_span(matrix_spans[number<0>{}], [&](auto idx0) {
        sweep_tile_span(matrix_spans[number<1>{}], [&](auto idx1) {
            constexpr auto distributed_idx = make_tuple(idx0, idx1);
            
            // Get the actual matrix coordinates from distributed indices
            const auto x_indices = get_x_indices_from_distributed_indices(
                matrix_4x4_dstr, distributed_idx);
            
            // Calculate value: row * 4 + col + 1 (for 1-16 numbering)
            const int row = x_indices[number<0>{}];
            const int col = x_indices[number<1>{}];
            const int value = row * 4 + col + 1;
            
            distributed_matrix(distributed_idx) = value;
            
            // if (debug) 
            // {
            //   // Ensure that we get some sensible output from different threads
            //   if (threadIdx.x == 0)
            //   {
            //     printf("DistributedIdx (thread 0): (");
            //     print_distributed_index(idx0);
            //     printf(", ");
            //     print_distributed_index(idx1);
            //     printf(") -> Matrix[%d,%d] = %d\n", row, col, value);
            //   }
            //   else if (threadIdx.x == 1)
            //   {
            //     printf("DistributedIdx (thread 1): (");
            //     print_distributed_index(idx0);
            //     printf(", ");
            //     print_distributed_index(idx1);
            //     printf(") -> Matrix[%d,%d] = %d\n", row, col, value);
            //   }
            //   else if (threadIdx.x == 2)
            //   {
            //     printf("DistributedIdx (thread 2): (");
            //     print_distributed_index(idx0);
            //     printf(", ");
            //     print_distributed_index(idx1);
            //     printf(") -> Matrix[%d,%d] = %d\n", row, col, value);
            //   }
            //   else if (threadIdx.x == 3)
            //   {
            //     printf("DistributedIdx (thread 3): (");
            //     print_distributed_index(idx0);
            //     printf(", ");
            //     print_distributed_index(idx1);
            //     printf(") -> Matrix[%d,%d] = %d\n", row, col, value);
            //   }
            // }
        });
    });

    // Ensure all threads have completed initialization
    __syncthreads();

    // Access 2x2 blocks and store results
    int output_idx = 0;
    // Block (0,0): top-left 2x2
    for (int block_row = 0; block_row < 2; block_row++) {
        for (int block_col = 0; block_col < 2; block_col++) {
            for (int i = 0; i < 2; i++) {
                for (int j = 0; j < 2; j++) {
                    const int row = block_row * 2 + i;
                    const int col = block_col * 2 + j;
                    
                    // Find the distributed indices for this matrix position
                    bool found = false;
                    int value = 0;
                    
                    sweep_tile_span(matrix_spans[number<0>{}], [&](auto idx0) {
                        sweep_tile_span(matrix_spans[number<1>{}], [&](auto idx1) {
                            if (!found) {
                                constexpr auto distributed_idx = make_tuple(idx0, idx1);
                                const auto x_indices = get_x_indices_from_distributed_indices(
                                    matrix_4x4_dstr, distributed_idx);
                                
                                if (x_indices[number<0>{}] == row && 
                                    x_indices[number<1>{}] == col) {
                                    value = distributed_matrix[distributed_idx];
                                    found = true;
                                }
                            }
                        });
                    });
                    
                    output[output_idx++] = value;
                    
                    // if (debug) 
                    // {
                    //     printf("Block(%d,%d)[%d,%d] (thread %u) = Matrix[%d,%d] = %d\n", 
                    //       block_row, block_col, i, j, threadIdx.x, row, col, value);
                    // }
                }
            }
        }
    }
}

TEST_F(TestTensorView, StaticDistributedTensor4x4Matrix2x2Blocks)
{
    // clang format-off
    std::vector<int> data_host = 
      {
        1, 2, 3 ,4,
        5, 6, 7, 8, 
        9, 10, 11, 12,
        13, 14, 15, 16
      };
    // clang format-on

    constexpr int total_elements = 16; // 4x4 matrix = 16 elements
    std::vector<int> output_host(total_elements, 0);
    int* output_device;
    
    int* input_device;
    hip_check_error(hipMalloc(&input_device, data_host.size() * sizeof(int)));
    hip_check_error(hipMemcpy(input_device, data_host.data(), data_host.size() * sizeof(int), hipMemcpyHostToDevice));

    hip_check_error(hipMalloc(&output_device, total_elements * sizeof(int)));
    hip_check_error(hipMemset(output_device, 0, total_elements * sizeof(int)));
    
    // Run kernel with debug output
    const dim3 block_dim(4); // 4 threads to cover 4 blocks
    const dim3 grid_dim(1);
    test_4x4_matrix_2x2_blocks_kernel<<<grid_dim, block_dim>>>(input_device, output_device, true);
    hip_check_error(hipDeviceSynchronize());
    
    // Copy results back
    hip_check_error(hipMemcpy(
        output_host.data(), output_device, total_elements * sizeof(int), hipMemcpyDeviceToHost));
    
    // Verify the 4x4 matrix is correctly organized as 2x2 blocks
    // Expected matrix:
    //  1  2  3  4
    //  5  6  7  8  
    //  9 10 11 12
    // 13 14 15 16
    
    // Block (0,0): [1,2; 5,6]
    // Block (0,1): [3,4; 7,8]  
    // Block (1,0): [9,10; 13,14]
    // Block (1,1): [11,12; 15,16]
    
    std::vector<int> expected_output = {
        // Block (0,0)
        1, 2, 5, 6,
        // Block (0,1)  
        3, 4, 7, 8,
        // Block (1,0)
        9, 10, 13, 14,
        // Block (1,1)
        11, 12, 15, 16
    };
    
    EXPECT_EQ(output_host, expected_output);
    
    hip_check_error(hipFree(output_device));
    hip_check_error(hipFree(input_device));
}

// Additional test to show slicing functionality
// __global__ void test_matrix_slicing_kernel(int* output)
// {
//     constexpr index_t MIterPerWarp = 2;
//     constexpr index_t NIterPerWarp = 2; 
//     constexpr index_t MWarp = 1;
//     constexpr index_t NWarp = 1;

//     constexpr auto matrix_dstr_encoding = tile_distribution_encoding<
//         sequence<>,
//         tuple<sequence<MIterPerWarp, MWarp>, sequence<NIterPerWarp, NWarp>>,
//         tuple<sequence<2, 1>>,
//         tuple<sequence<1, 1>>,
//         sequence<1, 2>,
//         sequence<0, 0>>{};

//     constexpr auto matrix_dstr = make_static_tile_distribution(matrix_dstr_encoding);
//     auto distributed_matrix = make_static_distributed_tensor<int>(matrix_dstr);
    
//     // Initialize with simple values
//     distributed_matrix.initialize(42);
    
//     // Extract a 2x2 slice from the top-left corner  
//     auto slice_data = distributed_matrix.get_y_sliced_thread_data(
//         sequence<0, 0>{},    // slice origins
//         sequence<2, 2>{}     // slice lengths  
//     );
    
//     // Store slice size in output
//     output[0] = slice_data.size();
// }

// TEST_F(TestTensorView, StaticDistributedTensorSlicing)
// {
//     int* output_device;
//     hip_check_error(hipMalloc(&output_device, sizeof(int)));
    
//     test_matrix_slicing_kernel<<<1, 32>>>(output_device);
//     hip_check_error(hipDeviceSynchronize());
    
//     int slice_size;
//     hip_check_error(hipMemcpy(&slice_size, output_device, sizeof(int), hipMemcpyDeviceToHost));
    
//     EXPECT_GT(slice_size, 0);
    
//     hipFree(output_device);
// }
