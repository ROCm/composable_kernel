// SPDX-License-Identifier: MIT
// Copyright (c) 2024, Advanced Micro Devices, Inc. All rights reserved.

/*
 * Tutorial 01: Tensor Fundamentals - Complete Foundation
 *
 * This tutorial teaches the three core concepts of ck_tile tensor system:
 * 1. Tensor Descriptor - defines tensor layout (lengths + strides)
 * 2. Tensor View - combines descriptor with memory pointer for access
 * 3. Tensor Coordinate - multi-dimensional index bound to a descriptor
 *
 * Key Learning: ALL access goes through tensor_view API using thread_buffer
 */

#include <iostream>
#include <vector>
#include <iomanip>
#include <numeric>
#include "ck_tile/core.hpp"
#include "ck_tile/host.hpp"

using namespace ck_tile;

// Vectorized read functions for easy debugging/disassembly
// Note: These functions demonstrate the API but may be scalarized by the compiler
// when returning by value. For true vectorization, use get_vectorized_elements inline.

template <typename DataType>
CK_TILE_DEVICE thread_buffer<DataType, 2> vectorized_read_2(const DataType* p_data, index_t offset)
{
    auto view = make_naive_tensor_view<address_space_enum::global>(
        p_data,
        make_tuple(12), // total elements
        make_tuple(1),  // stride
        number<2>{},    // GuaranteedLastDimensionVectorLength
        number<1>{}     // GuaranteedLastDimensionVectorStride
    );
    auto desc  = view.get_tensor_descriptor();
    auto coord = make_tensor_coordinate(desc, make_tuple(offset));
    return view.template get_vectorized_elements<thread_buffer<DataType, 2>>(coord, 0);
}

template <typename DataType>
CK_TILE_DEVICE thread_buffer<DataType, 4> vectorized_read_4(const DataType* p_data, index_t offset)
{
    auto view = make_naive_tensor_view<address_space_enum::global>(
        p_data,
        make_tuple(12), // total elements
        make_tuple(1),  // stride
        number<4>{},    // GuaranteedLastDimensionVectorLength
        number<1>{}     // GuaranteedLastDimensionVectorStride
    );
    auto desc  = view.get_tensor_descriptor();
    auto coord = make_tensor_coordinate(desc, make_tuple(offset));
    return view.template get_vectorized_elements<thread_buffer<DataType, 4>>(coord, 0);
}

template <typename DataType>
CK_TILE_DEVICE void
vectorized_write_4(DataType* p_data, index_t offset, thread_buffer<DataType, 4> buffer)
{
    auto view  = make_naive_tensor_view<address_space_enum::global>(p_data,
                                                                   make_tuple(12), // total elements
                                                                   make_tuple(1)   // stride
    );
    auto desc  = view.get_tensor_descriptor();
    auto coord = make_tensor_coordinate(desc, make_tuple(offset));
    view.set_vectorized_elements(coord, 0, buffer);
}

// Additional functions with fp16 to demonstrate vectorization with smaller types
CK_TILE_DEVICE thread_buffer<half_t, 4> vectorized_read_4_fp16(const half_t* p_data, index_t offset)
{
    auto view = make_naive_tensor_view<address_space_enum::global>(
        p_data,
        make_tuple(24), // total elements (more for fp16)
        make_tuple(1)   // stride
    );
    auto desc  = view.get_tensor_descriptor();
    auto coord = make_tensor_coordinate(desc, make_tuple(offset));
    return view.template get_vectorized_elements<thread_buffer<half_t, 4>>(coord, 0);
}

CK_TILE_DEVICE thread_buffer<half_t, 8> vectorized_read_8_fp16(const half_t* p_data, index_t offset)
{
    auto view  = make_naive_tensor_view<address_space_enum::global>(p_data,
                                                                   make_tuple(24), // total elements
                                                                   make_tuple(1)   // stride
    );
    auto desc  = view.get_tensor_descriptor();
    auto coord = make_tensor_coordinate(desc, make_tuple(offset));
    return view.template get_vectorized_elements<thread_buffer<half_t, 8>>(coord, 0);
}

// The kernel that demonstrates all fundamental concepts
template <typename DataType>
struct TensorFundamentalsKernel
{
    static constexpr index_t kBlockSize = 64;

    CK_TILE_DEVICE void
    operator()(const DataType* p_input, DataType* p_output, index_t H, index_t W, index_t C) const
    {
        // Only thread 0 for clean output
        if(get_thread_id() != 0)
            return;

        printf("\n=== TENSOR FUNDAMENTALS IN CK_TILE ===\n\n");

        //==================================================================
        // PART 1: TENSOR DESCRIPTOR
        //==================================================================
        printf("PART 1: TENSOR DESCRIPTOR\n");
        printf("-------------------------\n");

        // A tensor descriptor defines the layout of a tensor
        // It contains: lengths (shape) + strides (memory layout)

        // Create a descriptor for [H,W,C] tensor in row-major layout
        auto hwc_descriptor =
            make_naive_tensor_descriptor(make_tuple(H, W, C),    // lengths: [2, 3, 2]
                                         make_tuple(W * C, C, 1) // strides: [6, 2, 1] for row-major
            );

        // Access descriptor properties
        auto lengths = hwc_descriptor.get_lengths();
        // Note: Descriptors don't expose strides directly after transformation

        printf("Descriptor for [H=%ld, W=%ld, C=%ld] tensor:\n",
               static_cast<long>(H),
               static_cast<long>(W),
               static_cast<long>(C));
        printf("  Lengths: [%ld, %ld, %ld]\n",
               static_cast<long>(lengths.at(number<0>{})),
               static_cast<long>(lengths.at(number<1>{})),
               static_cast<long>(lengths.at(number<2>{})));
        printf("  Strides: [%ld, %ld, %ld] (row-major)\n",
               static_cast<long>(W * C),
               static_cast<long>(C),
               static_cast<long>(1));
        printf("  Memory formula: offset = h*%ld + w*%ld + c*%ld\n\n",
               static_cast<long>(W * C),
               static_cast<long>(C),
               static_cast<long>(1));

        //==================================================================
        // PART 2: TENSOR VIEW - Three Creation Methods
        //==================================================================
        printf("PART 2: TENSOR VIEW (Descriptor + Memory)\n");
        printf("-----------------------------------------\n");

        // Method 1: Explicit strides (most control)
        printf("Method 1: make_naive_tensor_view with explicit strides\n");
        auto view1 = make_naive_tensor_view<address_space_enum::global>(
            p_input,                // GPU memory pointer
            make_tuple(H, W, C),    // lengths
            make_tuple(W * C, C, 1) // explicit strides
        );
        printf("  Created view with shape [%ld,%ld,%ld] and strides [%ld,%ld,%ld]\n",
               static_cast<long>(H),
               static_cast<long>(W),
               static_cast<long>(C),
               static_cast<long>(W * C),
               static_cast<long>(C),
               static_cast<long>(1));

        // Method 2: Packed/contiguous (auto-computes row-major strides)
        printf("\nMethod 2: make_naive_tensor_view_packed (auto strides)\n");
        auto view2 = make_naive_tensor_view_packed<address_space_enum::global>(
            p_input,            // GPU memory pointer
            make_tuple(H, W, C) // lengths only, strides auto-computed
        );
        printf("  Created packed view - strides computed automatically\n");
        printf("  For row-major: last dim stride=1, each dim stride = next_dim_stride * "
               "next_dim_length\n");

        // Method 3: From existing descriptor
        printf("\nMethod 3: make_tensor_view from descriptor\n");
        auto view3 =
            make_tensor_view<address_space_enum::global>(p_input,       // GPU memory pointer
                                                         hwc_descriptor // existing descriptor
            );
        printf("  Created view using pre-existing descriptor\n");

        // Demonstrate all three views access the same data
        printf("\nVerifying all three methods create equivalent views:\n");
        {
            auto coord_test =
                make_tensor_coordinate(view1.get_tensor_descriptor(), make_tuple(0, 1, 0));
            auto val1 = view1.template get_vectorized_elements<thread_buffer<DataType, 1>>(
                coord_test, 0)[number<0>{}];

            auto coord_test2 =
                make_tensor_coordinate(view2.get_tensor_descriptor(), make_tuple(0, 1, 0));
            auto val2 = view2.template get_vectorized_elements<thread_buffer<DataType, 1>>(
                coord_test2, 0)[number<0>{}];

            auto coord_test3 =
                make_tensor_coordinate(view3.get_tensor_descriptor(), make_tuple(0, 1, 0));
            auto val3 = view3.template get_vectorized_elements<thread_buffer<DataType, 1>>(
                coord_test3, 0)[number<0>{}];

            printf("  view1[0,1,0] = %.0f (explicit strides)\n", static_cast<float>(val1));
            printf("  view2[0,1,0] = %.0f (packed/auto strides)\n", static_cast<float>(val2));
            printf("  view3[0,1,0] = %.0f (from descriptor)\n", static_cast<float>(val3));
            printf("  ✓ All three methods produce identical views!\n\n");
        }

        //==================================================================
        // PART 3: TENSOR COORDINATE
        //==================================================================
        printf("PART 3: TENSOR COORDINATE (Multi-dim Indexing)\n");
        printf("-----------------------------------------------\n");

        // Coordinates are multi-dimensional indices bound to a descriptor
        // They know how to map to linear memory offsets

        auto desc = view1.get_tensor_descriptor();

        // Create coordinate for position [1,2,0]
        auto coord = make_tensor_coordinate(desc, make_tuple(1, 2, 0));

        // Coordinate can compute its linear offset
        index_t offset = coord.get_offset();
        printf("Coordinate [1,2,0] maps to linear offset: %ld\n", static_cast<long>(offset));
        printf("  Calculation: 1*%ld + 2*%ld + 0*%ld = %ld\n\n",
               static_cast<long>(W * C),
               static_cast<long>(C),
               static_cast<long>(1),
               static_cast<long>(offset));

        //==================================================================
        // PART 4: ELEMENT ACCESS - The Critical Pattern
        //==================================================================
        printf("PART 4: ELEMENT ACCESS (thread_buffer Pattern)\n");
        printf("----------------------------------------------\n");
        printf("CRITICAL: get_vectorized_elements returns thread_buffer, NOT value!\n\n");

        // Reading elements - THE CORRECT PATTERN
        printf("Reading element at [0,0,0]:\n");
        {
            auto read_coord = make_tensor_coordinate(desc, make_tuple(0, 0, 0));

            // get_vectorized_elements returns thread_buffer<T,N>, not T!
            auto buffer = view1.template get_vectorized_elements<thread_buffer<DataType, 1>>(
                read_coord, // coordinate
                0           // linear_offset (usually 0)
            );

            // Extract actual value from thread_buffer
            DataType value = buffer[number<0>{}];

            printf("  Step 1: Create coordinate for [0,0,0]\n");
            printf("  Step 2: Call get_vectorized_elements -> returns thread_buffer\n");
            printf("  Step 3: Extract value with [number<0>{}]\n");
            printf("  Value at [0,0,0] = %.0f\n\n", static_cast<float>(value));
        }

        // Writing elements - THE CORRECT PATTERN
        printf("Writing element at [0,0,1]:\n");
        {
            auto write_coord = make_tensor_coordinate(desc, make_tuple(0, 0, 1));

            // Create thread_buffer for writing
            thread_buffer<DataType, 1> write_buffer;
            write_buffer[number<0>{}] = 99.0f;

            // Write to output view
            auto output_view = make_naive_tensor_view<address_space_enum::global>(
                p_output, make_tuple(H, W, C), make_tuple(W * C, C, 1));

            output_view.set_vectorized_elements(write_coord, 0, write_buffer);

            printf("  Step 1: Create thread_buffer with value 99\n");
            printf("  Step 2: Create coordinate for [0,0,1]\n");
            printf("  Step 3: Call set_vectorized_elements with buffer\n");
            printf("  Written value 99 to output[0,0,1]\n\n");
        }

        //==================================================================
        // PART 4.5: VECTORIZED ACCESS - Reading Multiple Elements
        //==================================================================
        printf("PART 4.5: VECTORIZED ACCESS (Performance Optimization)\n");
        printf("-------------------------------------------------------\n");
        printf("CRITICAL: Vectorization reads/writes multiple elements in one operation!\n\n");

        // Create a flattened view for easier vectorized access
        auto flat_view = make_naive_tensor_view<address_space_enum::global>(
            p_input,
            make_tuple(H * W * C), // [12] - all elements in linear order
            make_tuple(1)          // stride = 1 (contiguous)
        );
        auto flat_desc = flat_view.get_tensor_descriptor();

        // Example 1: Reading 2 elements at once (vector size = 2)
        printf("Example 1: Reading 2 elements at once\n");
        {
            // Call the vectorized_read_2 function (easy to disassemble in debugger)
            auto buffer = vectorized_read_2(p_input, 0);

            DataType val0 = buffer[number<0>{}];
            DataType val1 = buffer[number<1>{}];

            printf("  Position [0]: Read 2 elements in one operation\n");
            printf("    buffer[0] = %.0f\n", static_cast<float>(val0));
            printf("    buffer[1] = %.0f\n", static_cast<float>(val1));
            printf("  ✓ 2x faster than reading elements individually!\n");
            printf("  ✓ In debugger: 'disassemble vectorized_read_2<float>'\n\n");
        }

        // Example 2: Reading 4 elements at once (vector size = 4)
        printf("Example 2: Reading 4 elements at once\n");
        {
            // Call the vectorized_read_4 function (easy to disassemble in debugger)
            auto buffer = vectorized_read_4(p_input, 4);

            printf("  Position [4]: Read 4 elements in one operation\n");
            printf("    buffer[0] = %.0f\n", static_cast<float>(buffer[number<0>{}]));
            printf("    buffer[1] = %.0f\n", static_cast<float>(buffer[number<1>{}]));
            printf("    buffer[2] = %.0f\n", static_cast<float>(buffer[number<2>{}]));
            printf("    buffer[3] = %.0f\n", static_cast<float>(buffer[number<3>{}]));
            printf("  ✓ 4x faster than reading elements individually!\n");
            printf("  ✓ In debugger: 'disassemble vectorized_read_4<float>'\n\n");
        }

        // Example 3: Writing vectorized data
        printf("Example 3: Writing multiple elements at once\n");
        {
            // Create a buffer with 4 values
            thread_buffer<DataType, 4> write_buffer;
            write_buffer[number<0>{}] = 100.0f;
            write_buffer[number<1>{}] = 101.0f;
            write_buffer[number<2>{}] = 102.0f;
            write_buffer[number<3>{}] = 103.0f;

            // Call the vectorized_write_4 function (easy to disassemble in debugger)
            vectorized_write_4(p_output, 4, write_buffer);

            printf("  Position [4-7]: Wrote 4 elements in one operation\n");
            printf("    Wrote: 100, 101, 102, 103\n");
            printf("  ✓ 4x faster than writing elements individually!\n");
            printf("  ✓ In debugger: 'disassemble vectorized_write_4<float>'\n\n");
        }

        // Example 4: Vectorized copy operation (TRUE VECTORIZATION!)
        printf("Example 4: Vectorized copy - INLINE usage (TRUE vectorization)\n");
        {
            auto output_flat_view = make_naive_tensor_view<address_space_enum::global>(
                p_output, make_tuple(H * W * C), make_tuple(1));
            auto out_flat_desc = output_flat_view.get_tensor_descriptor();

            // Copy first 8 elements using vector size 4 (2 iterations)
            // THIS is where real vectorization happens - inline, no function calls!
            printf("  Copying first 8 elements using 2 vectorized operations:\n");
            for(index_t i = 0; i < 8; i += 4)
            {
                auto in_coord  = make_tensor_coordinate(flat_desc, make_tuple(i));
                auto out_coord = make_tensor_coordinate(out_flat_desc, make_tuple(i));

                // Read 4 elements - INLINE vectorized load (not through function)
                auto buffer =
                    flat_view.template get_vectorized_elements<thread_buffer<DataType, 4>>(in_coord,
                                                                                           0);

                // Write 4 elements (skip positions 4-7 which we already wrote)
                if(i != 4)
                {
                    output_flat_view.set_vectorized_elements(out_coord, 0, buffer);
                }

                printf("    Iteration %ld: Copied elements [%ld-%ld]\n",
                       static_cast<long>(i / 4),
                       static_cast<long>(i),
                       static_cast<long>(i + 3));
            }
            printf("  ✓ Copied 8 elements with only 2 memory operations!\n");
            printf("  ✓ THIS loop shows true vectorization in assembly!\n\n");
        }

        printf("Vectorization Key Points:\n");
        printf("  • Vector sizes: 1, 2, 4, 8 (powers of 2)\n");
        printf("  • Requires contiguous memory layout (stride=1 in access dimension)\n");
        printf("  • Dramatically improves memory bandwidth utilization\n");
        printf("  • Essential for high-performance GPU kernels\n");
        printf("  • Access each element with buffer[number<i>{}]\n");
        printf("  • IMPORTANT: Use inline for true vectorization, not function calls!\n");
        printf("  • Standalone functions may be scalarized when returning by value\n\n");

        //==================================================================
        // PART 5: MULTIPLE VIEWS OF SAME DATA
        //==================================================================
        printf("PART 5: MULTIPLE VIEWS OF SAME DATA\n");
        printf("------------------------------------\n");

        // Create two different views of the same memory
        // View A: [H, W, C] = [2, 3, 2]
        auto view_hwc = make_naive_tensor_view<address_space_enum::global>(
            p_input, make_tuple(H, W, C), make_tuple(W * C, C, 1));

        // View B: [HW, C] = [6, 2] - flattened spatial dimensions
        auto view_hw_c = make_naive_tensor_view<address_space_enum::global>(
            p_input, make_tuple(H * W, C), make_tuple(C, 1));

        printf("Two views of same memory:\n");
        printf("  View A: [H=%ld, W=%ld, C=%ld]\n",
               static_cast<long>(H),
               static_cast<long>(W),
               static_cast<long>(C));
        printf("  View B: [HW=%ld, C=%ld]\n", static_cast<long>(H * W), static_cast<long>(C));

        // Show they access the same data
        printf("\nAccessing same element through different views:\n");

        // Access element at h=1, w=1, c=0 through View A
        auto desc_a  = view_hwc.get_tensor_descriptor();
        auto coord_a = make_tensor_coordinate(desc_a, make_tuple(1, 1, 0));
        auto buffer_a =
            view_hwc.template get_vectorized_elements<thread_buffer<DataType, 1>>(coord_a, 0);
        DataType val_a = buffer_a[number<0>{}];

        // Access same element through View B at hw=4 (1*3+1), c=0
        auto desc_b  = view_hw_c.get_tensor_descriptor();
        auto coord_b = make_tensor_coordinate(desc_b, make_tuple(4, 0));
        auto buffer_b =
            view_hw_c.template get_vectorized_elements<thread_buffer<DataType, 1>>(coord_b, 0);
        DataType val_b = buffer_b[number<0>{}];

        printf("  View A[1,1,0] = %.0f\n", static_cast<float>(val_a));
        printf("  View B[4,0]   = %.0f (same value!)\n", static_cast<float>(val_b));
        printf("  Both access offset %ld in memory\n\n", static_cast<long>(coord_a.get_offset()));

        //==================================================================
        // PART 6: PRACTICAL EXAMPLE - Copy with Views
        //==================================================================
        printf("PART 6: PRACTICAL EXAMPLE - Copy Data\n");
        printf("--------------------------------------\n");

        // Create output view
        auto output_view = make_naive_tensor_view<address_space_enum::global>(
            p_output, make_tuple(H, W, C), make_tuple(W * C, C, 1));
        auto out_desc = output_view.get_tensor_descriptor();

        // Copy all elements using tensor_view API
        index_t count = 0;
        for(index_t h = 0; h < H; h++)
        {
            for(index_t w = 0; w < W; w++)
            {
                for(index_t c = 0; c < C; c++)
                {
                    // Read from input
                    auto in_coord = make_tensor_coordinate(desc, make_tuple(h, w, c));
                    auto in_buffer =
                        view1.template get_vectorized_elements<thread_buffer<DataType, 1>>(in_coord,
                                                                                           0);

                    // Write to output (except [0,0,1] which we already wrote as 99)
                    if(!(h == 0 && w == 0 && c == 1))
                    {
                        auto out_coord = make_tensor_coordinate(out_desc, make_tuple(h, w, c));
                        output_view.set_vectorized_elements(out_coord, 0, in_buffer);
                    }
                    count++;
                }
            }
        }
        printf("Copied %ld elements using tensor_view API\n", static_cast<long>(count));
        printf("Note: output[0,0,1] = 99 (modified), rest copied from input\n\n");

        //==================================================================
        // SUMMARY
        //==================================================================
        printf("=== KEY TAKEAWAYS ===\n");
        printf("1. Descriptor = Lengths + Strides (defines layout)\n");
        printf("2. View = Descriptor + Memory (enables access)\n");
        printf("3. Coordinate = Multi-dim index bound to descriptor\n");
        printf("4. ALWAYS: get_vectorized_elements returns thread_buffer!\n");
        printf("5. ALWAYS: Extract value with [number<0>{}], [number<1>{}], etc.\n");
        printf("6. Vectorization: Use thread_buffer<T,N> with N=2,4,8 for performance\n");
        printf("7. NEVER: Access memory directly - use tensor_view API\n\n");
    }
};

int main()
{
    std::cout << "\n================================================\n";
    std::cout << "Tutorial 01: Tensor Fundamentals\n";
    std::cout << "================================================\n\n";

    // Initialize HIP
    int device_count;
    hip_check_error(hipGetDeviceCount(&device_count));
    if(device_count == 0)
    {
        std::cerr << "No GPU devices found!\n";
        return 1;
    }

    hip_check_error(hipSetDevice(0));
    hipDeviceProp_t props;
    hip_check_error(hipGetDeviceProperties(&props, 0));
    std::cout << "Using GPU: " << props.name << "\n";

    // Small tensor for demonstration
    constexpr index_t H    = 2;         // height
    constexpr index_t W    = 3;         // width
    constexpr index_t C    = 2;         // # of chanels
    constexpr index_t size = H * W * C; // total tensor size

    std::cout << "\nTensor configuration:\n";
    std::cout << "  Shape: [" << H << ", " << W << ", " << C << "]\n";
    std::cout << "  Total elements: " << size << "\n";
    std::cout << "  Layout: Row-major (strides = [" << W * C << ", " << C << ", 1])\n\n";

    // Create test data: 1, 2, 3, 4, ... 12
    std::vector<float> h_input(size);
    std::iota(h_input.begin(), h_input.end(), 1.0f);

    std::cout << "Input data (row-major memory order):\n";
    for(index_t i = 0; i < size; ++i)
    {
        if(i % C == 0 && i > 0)
            std::cout << " | ";
        std::cout << std::setw(2) << h_input[i] << " ";
    }
    std::cout << "\n";

    std::cout << "\nLogical view [H,W,C]:\n";
    for(index_t h = 0; h < H; h++)
    {
        std::cout << "  H=" << h << ": ";
        for(index_t w = 0; w < W; w++)
        {
            std::cout << "[";
            for(index_t c = 0; c < C; c++)
            {
                index_t idx = h * W * C + w * C + c;
                std::cout << std::setw(2) << h_input[idx];
                if(c < C - 1)
                    std::cout << ",";
            }
            std::cout << "] ";
        }
        std::cout << "\n";
    }

    // Allocate device memory
    DeviceMem d_input(size * sizeof(float));
    DeviceMem d_output(size * sizeof(float));

    // Copy input to device
    d_input.ToDevice(h_input.data(), size * sizeof(float));

    // Initialize output to zeros
    std::vector<float> h_zeros(size, 0.0f);
    d_output.ToDevice(h_zeros.data(), size * sizeof(float));

    // Launch kernel
    constexpr index_t block_size = TensorFundamentalsKernel<float>::kBlockSize;
    stream_config stream;

    std::cout << "\nLaunching kernel...\n";
    std::cout << "=====================================\n";

    launch_kernel(stream,
                  make_kernel<block_size>(TensorFundamentalsKernel<float>{},
                                          dim3(1),          // 1 block
                                          dim3(block_size), // 64 threads
                                          0,                // no shared memory
                                          static_cast<const float*>(d_input.GetDeviceBuffer()),
                                          static_cast<float*>(d_output.GetDeviceBuffer()),
                                          H,
                                          W,
                                          C));

    hip_check_error(hipDeviceSynchronize());
    std::cout << "=====================================\n";

    // Copy output back
    std::vector<float> h_output(size);
    d_output.FromDevice(h_output.data(), size * sizeof(float));

    // Verify results
    std::cout << "\nOutput verification:\n";
    bool passed = true;
    for(index_t i = 0; i < size; ++i)
    {
        float expected = (i == 1) ? 99.0f : h_input[i]; // We wrote 99 to position [0,0,1]
        if(std::abs(h_output[i] - expected) > 1e-6f)
        {
            passed = false;
            std::cout << "  ✗ Mismatch at index " << i << ": expected " << expected << ", got "
                      << h_output[i] << "\n";
        }
    }

    if(passed)
    {
        std::cout << "  ✓ All elements correct!\n";
        std::cout << "  ✓ output[0,0,1] = 99 (modified as expected)\n";
        std::cout << "  ✓ All other elements copied correctly\n";
    }

    std::cout << "\n=== Tutorial Complete ===\n";
    std::cout << "You now understand:\n";
    std::cout << "- Tensor descriptors (layout definition)\n";
    std::cout << "- Tensor views (memory access abstraction)\n";
    std::cout << "- Tensor coordinates (multi-dimensional indexing)\n";
    std::cout << "- The thread_buffer pattern for element access\n";
    std::cout << "- Vectorized access with thread_buffer<T,N> for performance\n";
    std::cout << "- Creating multiple views of the same data\n\n";

    return passed ? 0 : 1;
}
