// SPDX-License-Identifier: MIT
// Copyright (c) Advanced Micro Devices, Inc. All rights reserved.

#include "ck_tile/host.hpp"
#include "ck_tile/core.hpp"
#include <cstring>
#include <iostream>
#include <vector>

namespace ck_tile {

struct MergeTransformExample
{
    CK_TILE_DEVICE void operator()() const
    {
        if(threadIdx.x == 0 && blockIdx.x == 0)
        {
            printf("\n=== Merge Transform Example (Device Kernel) ===\n");
            
            // Create merge transform for 4x5 tensor (20 elements total)
            auto transform = make_merge_transform(make_tuple(4, 5));
            
            printf("\nMerge Transform created for 4x5 tensor:\n");
            printf("- Lower dimensions: 2 (4x5)\n");
            printf("- Upper dimensions: 1 (20 elements linear)\n");
            printf("- Transform type: Multiple lower dimensions → Single upper dimension\n\n");
            
            // Test 1: Manual forward calculation - 2D coordinate → Linear index
            // For merge transform, the formula is: linear = row * columns + col
            printf("=== Forward Transform: 2D coordinate → Linear index (Manual) ===\n");
            
            // Test coordinate [2, 3]
            int row = 2, col = 3;
            int linear_manual = row * 5 + col;  // Manual calculation: 2*5 + 3 = 13
            
            printf("2D coord [%d, %d] → Linear index %d\n", row, col, linear_manual);
            printf("Calculation: %d×5 + %d = %d\n\n", row, col, linear_manual);
            
            // Test 2: Using transform to verify - Inverse transform - Linear index → 2D coordinate
            printf("=== Inverse Transform: Linear index → 2D coordinate (Using transform) ===\n");
            
            multi_index<1> upper_coord;
            upper_coord[number<0>{}] = 13;
            
            multi_index<2> lower_coord;
            transform.calculate_lower_index(lower_coord, upper_coord);
            
            printf("Linear index %d → 2D coord [%d, %d]\n",
                   static_cast<int>(upper_coord[number<0>{}]),
                   static_cast<int>(lower_coord[number<0>{}]),
                   static_cast<int>(lower_coord[number<1>{}]));
            printf("Calculation: 13 ÷ 5 = %d remainder %d\n\n",
                   static_cast<int>(lower_coord[number<0>{}]),
                   static_cast<int>(lower_coord[number<1>{}]));
            
            // Test 3: More examples showing both directions
            printf("=== Additional Examples (Both Directions) ===\n");
            
            // Test several coordinates - forward (manual calculation)
            int test_coords[][2] = {{0, 0}, {0, 4}, {1, 0}, {3, 4}};
            int num_tests = sizeof(test_coords) / sizeof(test_coords[0]);
            
            printf("Forward (2D → Linear):\n");
            for(int i = 0; i < num_tests; i++)
            {
                int test_row = test_coords[i][0];
                int test_col = test_coords[i][1];
                int test_linear = test_row * 5 + test_col;
                
                printf("  2D [%d, %d] → Linear %d\n", test_row, test_col, test_linear);
            }
            
            printf("\nInverse (Linear → 2D):\n");
            // Test several linear indices - inverse using transform
            int test_linear[] = {0, 5, 10, 19};
            int num_linear_tests = sizeof(test_linear) / sizeof(test_linear[0]);
            
            for(int i = 0; i < num_linear_tests; i++)
            {
                multi_index<1> test_upper;
                test_upper[number<0>{}] = test_linear[i];
                
                multi_index<2> test_lower;
                transform.calculate_lower_index(test_lower, test_upper);
                
                printf("  Linear %d → 2D [%d, %d]\n",
                       test_linear[i],
                       static_cast<int>(test_lower[number<0>{}]),
                       static_cast<int>(test_lower[number<1>{}]));
            }
            
            // Test 4: Verify round-trip consistency
            printf("\n=== Round-trip Verification ===\n");
            printf("Testing coordinate [2, 3]:\n");
            
            // Forward: [2,3] → 13
            int orig_row = 2, orig_col = 3;
            int computed_linear = orig_row * 5 + orig_col;
            
            // Inverse: 13 → [2,3]
            multi_index<1> round_trip_upper;
            round_trip_upper[number<0>{}] = computed_linear;
            
            multi_index<2> round_trip_lower;
            transform.calculate_lower_index(round_trip_lower, round_trip_upper);
            
            printf("  Original 2D: [%d, %d]\n", orig_row, orig_col);
            printf("  Forward → Linear: %d\n", computed_linear);
            printf("  Inverse → 2D: [%d, %d]\n",
                   static_cast<int>(round_trip_lower[number<0>{}]),
                   static_cast<int>(round_trip_lower[number<1>{}]));
            
            bool is_consistent = (orig_row == static_cast<int>(round_trip_lower[number<0>{}])) &&
                               (orig_col == static_cast<int>(round_trip_lower[number<1>{}]));
            printf("  Round-trip consistent: %s\n", is_consistent ? "YES" : "NO");
            
            printf("\n=== Merge Transform Example Complete ===\n");
        }
    }
};

struct UnmergeTransformExample
{
    CK_TILE_DEVICE void operator()() const
    {
        if(threadIdx.x == 0 && blockIdx.x == 0)
        {
            printf("\n=== Unmerge Transform Example (Device Kernel) ===\n");
            
            // Create unmerge transform for 3x4x2 tensor (24 elements total)
            auto transform = make_unmerge_transform(make_tuple(3, 4, 2));
            
            printf("\nUnmerge Transform created for 3x4x2 tensor:\n");
            printf("- Lower dimensions: 1 (24 elements linear)\n");
            printf("- Upper dimensions: 3 (3x4x2)\n");
            printf("- Transform type: Multiple upper dimensions → Single lower dimension\n\n");
            
            // Test 1: Using transform - Inverse: Upper (3D) → Lower (1D)
            printf("=== Inverse Transform: 3D coordinate → Linear index (Using transform) ===\n");
            
            multi_index<3> upper_coord;
            upper_coord[number<0>{}] = 1;
            upper_coord[number<1>{}] = 3;
            upper_coord[number<2>{}] = 0;
            
            multi_index<1> lower_coord;
            transform.calculate_lower_index(lower_coord, upper_coord);
            
            printf("3D coord [%d, %d, %d] → Linear index %d\n",
                   static_cast<int>(upper_coord[number<0>{}]),
                   static_cast<int>(upper_coord[number<1>{}]),
                   static_cast<int>(upper_coord[number<2>{}]),
                   static_cast<int>(lower_coord[number<0>{}]));
            printf("Calculation: %d×8 + %d×2 + %d = %d\n",
                   static_cast<int>(upper_coord[number<0>{}]),
                   static_cast<int>(upper_coord[number<1>{}]),
                   static_cast<int>(upper_coord[number<2>{}]),
                   static_cast<int>(lower_coord[number<0>{}]));
            
            // Test 2: Manual forward: Lower (1D) → Upper (3D)
            printf("\n=== Forward Transform: Linear index → 3D coordinate (Manual) ===\n");
            // Test 2: Manual forward: Lower (1D) → Upper (3D)
            printf("\n=== Forward Transform: Linear index → 3D coordinate (Manual) ===\n");
            
            // Take linear index 14 and manually compute 3D coordinates
            int linear_idx = 14;
            int rev_dim0 = linear_idx / (4 * 2);      // 14 / 8 = 1
            int remainder = linear_idx % (4 * 2);     // 14 % 8 = 6
            int rev_dim1 = remainder / 2;             // 6 / 2 = 3
            int rev_dim2 = remainder % 2;             // 6 % 2 = 0
            
            printf("Linear index %d → 3D coord [%d, %d, %d]\n", 
                   linear_idx, rev_dim0, rev_dim1, rev_dim2);
            printf("Calculation: 14 ÷ 8 = %d remainder %d, then %d ÷ 2 = %d remainder %d\n\n",
                   rev_dim0, remainder, remainder, rev_dim1, rev_dim2);
            
            // Test 3: More examples showing the transform direction
            printf("=== Additional Examples ===\n");
            
            // Test several 3D coordinates - using transform (Inverse: 3D → Linear)
            int test_coords[][3] = {{0, 0, 0}, {1, 0, 0}, {1, 3, 0}, {2, 3, 1}};
            int num_tests = sizeof(test_coords) / sizeof(test_coords[0]);
            
            printf("Inverse Transform (3D → Linear):\n");
            for(int i = 0; i < num_tests; i++)
            {
                multi_index<3> test_upper;
                test_upper[number<0>{}] = test_coords[i][0];
                test_upper[number<1>{}] = test_coords[i][1];
                test_upper[number<2>{}] = test_coords[i][2];
                
                multi_index<1> test_lower;
                transform.calculate_lower_index(test_lower, test_upper);
                
                printf("  3D [%d, %d, %d] → Linear %d\n",
                       test_coords[i][0], test_coords[i][1], test_coords[i][2],
                       static_cast<int>(test_lower[number<0>{}]));
            }
            
            printf("\nForward Transform (Linear → 3D):\n");
            // Test several linear indices - manual forward calculation
            int test_linear_vals[] = {0, 8, 14, 23};
            int num_linear_tests = sizeof(test_linear_vals) / sizeof(test_linear_vals[0]);
            
            for(int i = 0; i < num_linear_tests; i++)
            {
                int lin_val = test_linear_vals[i];
                int calc_dim0 = lin_val / 8;
                int calc_remainder = lin_val % 8;
                int calc_dim1 = calc_remainder / 2;
                int calc_dim2 = calc_remainder % 2;
                
                printf("  Linear %d → 3D [%d, %d, %d]\n", 
                       lin_val, calc_dim0, calc_dim1, calc_dim2);
            }
            
            // Test 5: Verify round-trip consistency
            printf("\n=== Round-trip Verification ===\n");
            printf("Testing coordinate [1, 3, 0]:\n");
            
            // Forward: [1,3,0] → 14 using transform
            multi_index<3> orig_upper;
            orig_upper[number<0>{}] = 1;
            orig_upper[number<1>{}] = 3;
            orig_upper[number<2>{}] = 0;
            
            multi_index<1> computed_lower;
            transform.calculate_lower_index(computed_lower, orig_upper);
            
            // Reverse: 14 → [1,3,0] using manual calculation
            int linear_result = static_cast<int>(computed_lower[number<0>{}]);
            int back_dim0 = linear_result / 8;
            int back_remainder = linear_result % 8;
            int back_dim1 = back_remainder / 2;
            int back_dim2 = back_remainder % 2;
            
            printf("  Original 3D: [%d, %d, %d]\n",
                   static_cast<int>(orig_upper[number<0>{}]),
                   static_cast<int>(orig_upper[number<1>{}]),
                   static_cast<int>(orig_upper[number<2>{}]));
            printf("  Transform → Linear: %d\n", linear_result);
            printf("  Manual reverse → 3D: [%d, %d, %d]\n", back_dim0, back_dim1, back_dim2);
            
            bool is_consistent = (static_cast<int>(orig_upper[number<0>{}]) == back_dim0) &&
                               (static_cast<int>(orig_upper[number<1>{}]) == back_dim1) &&
                               (static_cast<int>(orig_upper[number<2>{}]) == back_dim2);
            printf("  Round-trip consistent: %s\n", is_consistent ? "YES" : "NO");
            
            printf("\n=== Unmerge Transform Example Complete ===\n");
        }
    }
};

struct EmbedTransformExample
{
    CK_TILE_DEVICE void operator()() const
    {
        if(threadIdx.x == 0 && blockIdx.x == 0)
        {
            printf("\n=== Embed Transform Example (Device Kernel) ===\n");
            
            // Create embed transform for 2x3 tensor with custom strides [12, 1]
            auto transform = make_embed_transform(make_tuple(2, 3), make_tuple(12, 1));
            
            printf("\nEmbed Transform created for 2x3 tensor with strides [12, 1]:\n");
            printf("- Lower dimensions: 1 (linear index)\n");
            printf("- Upper dimensions: 2 (2x3 with custom strides)\n");
            printf("- Transform type: Single lower dimension → Multiple upper dimensions (strided)\n\n");
            
            // Test 1: Manual forward: Lower (1D) → Upper (2D)
            printf("=== Forward Transform: Linear index → 2D coordinate (Manual) ===\n");
            
            // Test linear index 14
            int linear_idx = 14;
            int row = linear_idx / 12;        // 14 / 12 = 1
            int remainder = linear_idx % 12;  // 14 % 12 = 2
            int col = remainder / 1;          // 2 / 1 = 2
            
            printf("Linear index %d → 2D coord [%d, %d]\n", linear_idx, row, col);
            printf("Calculation: 14 ÷ 12 = %d, remainder = %d\n\n", row, remainder);
            
            // Test 2: Using transform - Inverse: Upper (2D) → Lower (1D)
            printf("=== Inverse Transform: 2D coordinate → Linear index (Using transform) ===\n");
            
            multi_index<2> upper_coord;
            upper_coord[number<0>{}] = 1;
            upper_coord[number<1>{}] = 2;
            
            multi_index<1> lower_coord;
            transform.calculate_lower_index(lower_coord, upper_coord);
            
            printf("2D coord [%d, %d] → Linear index %d\n",
                   static_cast<int>(upper_coord[number<0>{}]),
                   static_cast<int>(upper_coord[number<1>{}]),
                   static_cast<int>(lower_coord[number<0>{}]));
            printf("Calculation: %d×12 + %d×1 = %d\n\n",
                   static_cast<int>(upper_coord[number<0>{}]),
                   static_cast<int>(upper_coord[number<1>{}]),
                   static_cast<int>(lower_coord[number<0>{}]));
            
            // Test 3: More examples showing both directions
            printf("=== Additional Examples ===\n");
            
            // Test several 2D coordinates - using transform (Inverse: 2D → Linear)
            int test_coords[][2] = {{0, 0}, {0, 2}, {1, 0}, {1, 2}};
            int num_tests = sizeof(test_coords) / sizeof(test_coords[0]);
            
            printf("Inverse Transform (2D → Linear):\n");
            for(int i = 0; i < num_tests; i++)
            {
                multi_index<2> test_upper;
                test_upper[number<0>{}] = test_coords[i][0];
                test_upper[number<1>{}] = test_coords[i][1];
                
                multi_index<1> test_lower;
                transform.calculate_lower_index(test_lower, test_upper);
                
                printf("  2D [%d, %d] → Linear %d\n",
                       test_coords[i][0], test_coords[i][1],
                       static_cast<int>(test_lower[number<0>{}]));
            }
            
            printf("\nForward Transform (Linear → 2D):\n");
            // Test several linear indices - manual forward calculation
            int test_linear_vals[] = {0, 2, 12, 14};
            int num_linear_tests = sizeof(test_linear_vals) / sizeof(test_linear_vals[0]);
            
            for(int i = 0; i < num_linear_tests; i++)
            {
                int lin_val = test_linear_vals[i];
                int calc_row = lin_val / 12;
                int calc_remainder = lin_val % 12;
                int calc_col = calc_remainder / 1;
                
                printf("  Linear %d → 2D [%d, %d]\n", 
                       lin_val, calc_row, calc_col);
            }
            
            // Test 4: Verify round-trip consistency
            printf("\n=== Round-trip Verification ===\n");
            printf("Testing coordinate [1, 2]:\n");
            
            // Forward: [1,2] → 14 using transform (inverse)
            multi_index<2> orig_upper;
            orig_upper[number<0>{}] = 1;
            orig_upper[number<1>{}] = 2;
            
            multi_index<1> computed_lower;
            transform.calculate_lower_index(computed_lower, orig_upper);
            
            // Reverse: 14 → [1,2] using manual calculation
            int linear_result = static_cast<int>(computed_lower[number<0>{}]);
            int back_row = linear_result / 12;
            int back_remainder = linear_result % 12;
            int back_col = back_remainder / 1;
            
            printf("  Original 2D: [%d, %d]\n",
                   static_cast<int>(orig_upper[number<0>{}]),
                   static_cast<int>(orig_upper[number<1>{}]));
            printf("  Transform → Linear: %d\n", linear_result);
            printf("  Manual reverse → 2D: [%d, %d]\n", back_row, back_col);
            
            bool is_consistent = (static_cast<int>(orig_upper[number<0>{}]) == back_row) &&
                               (static_cast<int>(orig_upper[number<1>{}]) == back_col);
            printf("  Round-trip consistent: %s\n", is_consistent ? "YES" : "NO");
            
            printf("\n=== Embed Transform Example Complete ===\n");
        }
    }
};

struct ReplicateTransformExample
{
    CK_TILE_DEVICE void operator()() const
    {
        if(threadIdx.x == 0 && blockIdx.x == 0)
        {
            printf("\n=== Replicate Transform Example (Device Kernel) ===\n");
            
            // Create replicate transform for 3x4 broadcasted dimensions
            auto transform = make_replicate_transform(make_tuple(3, 4));
            
            printf("\nReplicate Transform created for 3x4 broadcasting:\n");
            printf("- Lower dimensions: 0 (scalar value)\n");
            printf("- Upper dimensions: 2 (3x4 broadcasted space)\n");
            printf("- Transform type: Scalar → Multi-dimensional broadcasting\n\n");
            
            // Test 1: Inverse: Upper (2D) → Lower (0D) - Always empty
            printf("=== Inverse Transform: 2D coordinate → Scalar (Using transform) ===\n");
            printf("Any 2D coordinate maps to an empty scalar coordinate\n");
            
            // Test several 2D coordinates - all map to empty scalar
            int test_coords[][2] = {{0, 0}, {1, 2}, {2, 3}, {1, 1}};
            int num_tests = sizeof(test_coords) / sizeof(test_coords[0]);
            
            for(int i = 0; i < num_tests; i++)
            {
                multi_index<2> upper_coord;
                upper_coord[number<0>{}] = test_coords[i][0];
                upper_coord[number<1>{}] = test_coords[i][1];
                
                multi_index<0> lower_coord;  // Empty coordinate (0 dimensions)
                transform.calculate_lower_index(lower_coord, upper_coord);
                
                printf("  2D [%d, %d] → Empty scalar [] (always empty)\n",
                       test_coords[i][0], test_coords[i][1]);
            }
            
            // Test 2: Manual forward explanation - Conceptual only
            printf("\n=== Forward Transform: Scalar → 2D coordinate (Conceptual) ===\n");
            printf("Broadcasting concept: Single scalar value appears at ALL positions\n");
            printf("- Scalar value logically broadcasts to every coordinate in 3x4 space\n");
            printf("- No actual coordinate calculation needed - same value everywhere\n\n");
            
            // Show broadcasting behavior conceptually
            printf("Broadcasting visualization:\n");
            printf("  Scalar value → appears at [0,0], [0,1], [0,2], [0,3]\n");
            printf("                 appears at [1,0], [1,1], [1,2], [1,3]\n");
            printf("                 appears at [2,0], [2,1], [2,2], [2,3]\n");
            printf("  Total positions: 3×4 = 12 positions, all contain same scalar value\n\n");
            
            // Test 3: Boundary checks and validity
            printf("=== Transform Properties ===\n");
            
            // All coordinates within bounds are valid for replicate
            int boundary_coords[][2] = {{0, 0}, {2, 3}, {1, 2}};
            int num_boundary_tests = sizeof(boundary_coords) / sizeof(boundary_coords[0]);
            
            printf("Coordinate validity (all should be valid for broadcasting):\n");
            for(int i = 0; i < num_boundary_tests; i++)
            {
                multi_index<2> test_upper;
                test_upper[number<0>{}] = boundary_coords[i][0];
                test_upper[number<1>{}] = boundary_coords[i][1];
                
                // For replicate transform, all coordinates within bounds are valid
                bool is_valid = (boundary_coords[i][0] >= 0 && boundary_coords[i][0] < 3) &&
                               (boundary_coords[i][1] >= 0 && boundary_coords[i][1] < 4);
                
                printf("  Coord [%d, %d]: %s\n",
                       boundary_coords[i][0], boundary_coords[i][1],
                       is_valid ? "Valid" : "Invalid");
            }
            
            // Test 4: Understanding the broadcasting semantics
            printf("\n=== Broadcasting Semantics ===\n");
            printf("ReplicateTransform characteristics:\n");
            printf("- Maps 0-dimensional (scalar) space to N-dimensional space\n");
            printf("- Every position in upper space maps to same scalar value\n");
            printf("- Essential for GEMM bias terms, constant broadcasting\n");
            printf("- Zero memory overhead - logical replication only\n\n");
            
            // Test 5: Demonstrate the key insight
            printf("=== Key Insight ===\n");
            printf("Forward direction (conceptual): Scalar → All 2D positions\n");
            printf("  - Input: single scalar value\n");
            printf("  - Output: same value appears at every coordinate [i,j] where 0≤i<3, 0≤j<4\n");
            printf("  - No coordinate transformation needed - same value everywhere\n\n");
            
            printf("Inverse direction (using transform):\n");
            printf("  - Input: any coordinate [i,j] in upper space\n");
            printf("  - Output: empty coordinate [] (representing the scalar)\n");
            printf("  - All upper coordinates map to the same scalar source\n\n");
            
            // Test 6: Practical usage context
            printf("=== Practical Usage ===\n");
            printf("Common applications:\n");
            printf("1. GEMM operations: C = A × B + bias (bias broadcasted)\n");
            printf("2. Elementwise operations: tensor + scalar\n");
            printf("3. Neural networks: adding bias terms\n");
            printf("4. Initialization: filling tensors with constant values\n\n");
            
            printf("Example: bias term in neural network\n");
            printf("  - bias: scalar or 1D vector\n");
            printf("  - activations: [batch_size, features] = [256, 128]\n");
            printf("  - replicate bias across all [batch, feature] positions\n");
            printf("  - Result: bias added to every activation\n");
            
            printf("\n=== Replicate Transform Example Complete ===\n");
        }
    }
};

struct OffsetTransformExample
{
    CK_TILE_DEVICE void operator()() const
    {
        if(threadIdx.x == 0 && blockIdx.x == 0)
        {
            printf("\n=== Offset Transform Example (Device Kernel) ===\n");
            
            // Create offset transform for sub-region access
            // CK Tile offset: lower = upper + offset
            // Maps upper space [0, 47] to lower space [16, 63] (adding offset 16)
            auto transform = make_offset_transform(48, 16);
            
            printf("\nOffset Transform created for sub-region access:\n");
            printf("- Upper dimensions: 1 (original indices [0, 47])\n");
            printf("- Lower dimensions: 1 (translated indices [16, 63])\n");
            printf("- Transform type: Translation by constant offset (+16)\n");
            printf("- CK Tile formula: lower = upper + offset\n\n");
            
            // Test 1: Inverse: Upper (original) → Lower (translated) using transform
            printf("=== Using Transform: Original index → Translated index ===\n");
            
            multi_index<1> upper_coord;
            upper_coord[number<0>{}] = 5;  // Original index 5
            
            multi_index<1> lower_coord;
            transform.calculate_lower_index(lower_coord, upper_coord);
            
            printf("Original index %d → Translated index %d\n",
                   static_cast<int>(upper_coord[number<0>{}]),
                   static_cast<int>(lower_coord[number<0>{}]));
            printf("Calculation: %d + 16 = %d\n",
                   static_cast<int>(upper_coord[number<0>{}]),
                   static_cast<int>(lower_coord[number<0>{}]));
            
            // Test 2: Manual reverse calculation (if we needed it)
            printf("\n=== Manual Reverse: Translated index → Original index ===\n");
            
            int translated_idx = 21;
            int original_idx = translated_idx - 16;  // Subtract offset
            
            printf("Translated index %d → Original index %d\n", translated_idx, original_idx);
            printf("Calculation: %d - 16 = %d\n", translated_idx, original_idx);
            
            // Test 3: Multiple examples showing the transform
            printf("\n=== Additional Examples ===\n");
            
            // Test several original indices using transform (Original → Translated)
            int test_original_indices[] = {0, 10, 20, 47};
            int num_tests = sizeof(test_original_indices) / sizeof(test_original_indices[0]);
            
            printf("Using Transform (Original → Translated):\n");
            for(int i = 0; i < num_tests; i++)
            {
                multi_index<1> test_upper;
                test_upper[number<0>{}] = test_original_indices[i];
                
                multi_index<1> test_lower;
                transform.calculate_lower_index(test_lower, test_upper);
                
                printf("  Original %d → Translated %d\n",
                       test_original_indices[i],
                       static_cast<int>(test_lower[number<0>{}]));
            }
            
            printf("\nManual Reverse (Translated → Original):\n");
            // Test several translated indices - manual reverse calculation
            int test_translated_indices[] = {16, 26, 36, 63};
            int num_trans_tests = sizeof(test_translated_indices) / sizeof(test_translated_indices[0]);
            
            for(int i = 0; i < num_trans_tests; i++)
            {
                int trans_idx = test_translated_indices[i];
                int orig_idx = trans_idx - 16;  // Subtract offset
                
                printf("  Translated %d → Original %d\n", trans_idx, orig_idx);
            }
            
            // Test 4: Boundary conditions
            printf("\n=== Boundary Conditions ===\n");
            printf("Original space: [0, 47] (48 elements)\n");
            printf("Translated space: [16, 63] (offset +16)\n");
            
            // Test boundaries using transform
            int boundary_original_indices[] = {0, 47};  // Start and end of original space
            
            for(int i = 0; i < 2; i++)
            {
                multi_index<1> boundary_upper;
                boundary_upper[number<0>{}] = boundary_original_indices[i];
                
                multi_index<1> boundary_lower;
                transform.calculate_lower_index(boundary_lower, boundary_upper);
                
                printf("  Original boundary %d → Translated %d\n",
                       boundary_original_indices[i],
                       static_cast<int>(boundary_lower[number<0>{}]));
            }
            
            // Test 5: Round-trip verification
            printf("\n=== Round-trip Verification ===\n");
            printf("Testing original index 10:\n");
            
            // Start with original index 10
            int original_start = 10;
            
            // Forward: original → translated (using transform)
            multi_index<1> orig_coord;
            orig_coord[number<0>{}] = original_start;
            
            multi_index<1> trans_coord;
            transform.calculate_lower_index(trans_coord, orig_coord);
            
            int computed_translated = static_cast<int>(trans_coord[number<0>{}]);
            
            // Reverse: translated → original (manual)
            int back_to_original = computed_translated - 16;
            
            printf("  Original: %d\n", original_start);
            printf("  Transform → Translated: %d\n", computed_translated);
            printf("  Manual reverse → Original: %d\n", back_to_original);
            
            bool is_consistent = (original_start == back_to_original);
            printf("  Round-trip consistent: %s\n", is_consistent ? "YES" : "NO");
            
            // Test 6: Practical usage explanation
            printf("\n=== Practical Usage ===\n");
            printf("Offset transforms are essential for:\n");
            printf("1. Tile-based algorithms: mapping logical tiles to buffer positions\n");
            printf("2. Memory management: translating local indices to global buffer positions\n");
            printf("3. GPU kernels: accessing sub-regions within larger allocated buffers\n");
            printf("4. Sliding window operations: translating window-local to global coordinates\n\n");
            
            printf("Example: Processing 48-element logical tiles in a larger buffer\n");
            printf("  - Logical tile indices: [0,47] (what the algorithm works with)\n");
            printf("  - Buffer positions: [16,63] (where data actually lives)\n");
            printf("  - Offset transform maps logical → buffer positions\n");
            printf("  - Algorithm uses index 0, accesses buffer position 16\n");
            printf("  - Algorithm uses index 47, accesses buffer position 63\n");
            
            printf("\n=== Offset Transform Example Complete ===\n");
        }
    }
};

struct PassthroughTransformExample
{
    CK_TILE_DEVICE void operator()() const
    {
        if(threadIdx.x == 0 && blockIdx.x == 0)
        {
            printf("\n=== PassThrough Transform Example (Device Kernel) ===\n");
            
            // Create pass-through transform for identity mapping
            // CK Tile passthrough: lower = upper (perfect identity)
            // Maps space [0, 59] to space [0, 59] (no change)
            auto transform = make_pass_through_transform(60);
            
            printf("\nPassThrough Transform created for identity mapping:\n");
            printf("- Upper dimensions: 1 (original indices [0, 59])\n");
            printf("- Lower dimensions: 1 (identical indices [0, 59])\n");
            printf("- Transform type: Perfect identity (no-op)\n");
            printf("- CK Tile formula: lower = upper\n\n");
            
            // Test 1: Forward: Upper → Lower using transform (identity)
            printf("=== Using Transform: Upper → Lower (Identity) ===\n");
            
            multi_index<1> upper_coord;
            upper_coord[number<0>{}] = 25;  // Index 25
            
            multi_index<1> lower_coord;
            transform.calculate_lower_index(lower_coord, upper_coord);
            
            printf("Upper index %d → Lower index %d\n",
                   static_cast<int>(upper_coord[number<0>{}]),
                   static_cast<int>(lower_coord[number<0>{}]));
            printf("Calculation: %d (unchanged)\n",
                   static_cast<int>(upper_coord[number<0>{}]));
            
            // Test 2: Multiple examples showing the identity transform
            printf("\n=== Additional Examples ===\n");
            
            // Test several indices using transform (Upper → Lower identity)
            int test_indices[] = {0, 10, 25, 45, 59};
            int num_tests = sizeof(test_indices) / sizeof(test_indices[0]);
            
            printf("Using Transform (Upper → Lower Identity):\n");
            for(int i = 0; i < num_tests; i++)
            {
                multi_index<1> test_upper;
                test_upper[number<0>{}] = test_indices[i];
                
                multi_index<1> test_lower;
                transform.calculate_lower_index(test_lower, test_upper);
                
                printf("  Upper %d → Lower %d (unchanged)\n",
                       test_indices[i],
                       static_cast<int>(test_lower[number<0>{}]));
            }
            
            // Test 3: Boundary conditions
            printf("\n=== Boundary Conditions ===\n");
            printf("Space: [0, 59] (60 elements)\n");
            printf("Identity mapping: no coordinate change\n");
            
            // Test boundaries using transform
            int boundary_indices[] = {0, 59};  // Start and end of space
            
            for(int i = 0; i < 2; i++)
            {
                multi_index<1> boundary_upper;
                boundary_upper[number<0>{}] = boundary_indices[i];
                
                multi_index<1> boundary_lower;
                transform.calculate_lower_index(boundary_lower, boundary_upper);
                
                printf("  Boundary %d → %d (identity)\n",
                       boundary_indices[i],
                       static_cast<int>(boundary_lower[number<0>{}]));
            }
            
            // Test 4: Round-trip verification
            printf("\n=== Round-trip Verification ===\n");
            printf("Testing index 35:\n");
            
            // Start with index 35
            int original_index = 35;
            
            // Forward: upper → lower (using transform)
            multi_index<1> orig_coord;
            orig_coord[number<0>{}] = original_index;
            
            multi_index<1> result_coord;
            transform.calculate_lower_index(result_coord, orig_coord);
            
            int computed_result = static_cast<int>(result_coord[number<0>{}]);
            
            printf("  Original: %d\n", original_index);
            printf("  Transform → Result: %d\n", computed_result);
            
            bool is_identity = (original_index == computed_result);
            printf("  Perfect identity: %s\n", is_identity ? "YES" : "NO");
            
            // Test 5: Practical usage explanation
            printf("\n=== Practical Usage ===\n");
            printf("PassThrough transforms are essential for:\n");
            printf("1. Placeholder in transformation chains: maintaining dimensions\n");
            printf("2. Template completion: filling required transform slots\n");
            printf("3. Optimization targets: compiler can eliminate identity operations\n");
            printf("4. Dimension preservation: keeping coordinate spaces unchanged\n\n");
            
            printf("Example: Multi-transform composition\n");
            printf("  - Some dimensions need complex transformations\n");
            printf("  - Other dimensions pass through unchanged\n");
            printf("  - PassThrough maintains dimensional consistency\n");
            printf("  - Zero runtime cost (optimized away)\n");
            
            // Test 6: Performance characteristics
            printf("\n=== Performance Characteristics ===\n");
            printf("PassThrough transforms have unique properties:\n");
            printf("- Zero computational cost: idx_low = idx_up\n");
            printf("- Compile-time optimization: often eliminated entirely\n");
            printf("- Memory efficiency: no additional storage needed\n");
            printf("- Perfect cache behavior: no coordinate translation overhead\n");
            
            printf("\n=== PassThrough Transform Example Complete ===\n");
        }
    }
};

struct PadTransformExample
{
    CK_TILE_DEVICE void operator()() const
    {
        if(threadIdx.x == 0 && blockIdx.x == 0)
        {
            printf("\n=== Pad Transform Example (Device Kernel) ===\n");
            
            // Create pad transform for boundary handling
            // CK Tile pad: lower = upper - left_pad
            // Maps upper space [0, 4] to lower space [-1, 3] (where valid range is [0, 2])
            auto transform = make_pad_transform(3, 1, 1);  // low_length=3, left_pad=1, right_pad=1
            
            printf("\nPad Transform created for boundary handling:\n");
            printf("- Upper dimensions: 1 (padded indices [0, 4])\n");
            printf("- Lower dimensions: 1 (original indices [0, 2], with padding region)\n");
            printf("- Transform type: Coordinate translation with padding boundaries\n");
            printf("- CK Tile formula: lower = upper - left_pad\n\n");
            
            // Test 1: Forward: Upper (padded) → Lower (original) using transform
            printf("=== Using Transform: Padded index → Original index ===\n");
            
            multi_index<1> upper_coord;
            upper_coord[number<0>{}] = 2;  // Padded index 2
            
            multi_index<1> lower_coord;
            transform.calculate_lower_index(lower_coord, upper_coord);
            
            printf("Padded index %d → Original index %d\n",
                   static_cast<int>(upper_coord[number<0>{}]),
                   static_cast<int>(lower_coord[number<0>{}]));
            printf("Calculation: %d - 1 = %d\n",
                   static_cast<int>(upper_coord[number<0>{}]),
                   static_cast<int>(lower_coord[number<0>{}]));
            
            // Test 2: Multiple examples showing padding behavior
            printf("\n=== Additional Examples ===\n");
            
            // Test several padded indices using transform (Padded → Original)
            int test_padded_indices[] = {0, 1, 2, 3, 4};  // Full padded range
            int num_tests = sizeof(test_padded_indices) / sizeof(test_padded_indices[0]);
            
            printf("Using Transform (Padded → Original):\n");
            for(int i = 0; i < num_tests; i++)
            {
                multi_index<1> test_upper;
                test_upper[number<0>{}] = test_padded_indices[i];
                
                multi_index<1> test_lower;
                transform.calculate_lower_index(test_lower, test_upper);
                
                int original_idx = static_cast<int>(test_lower[number<0>{}]);
                bool is_valid = (original_idx >= 0 && original_idx < 3);  // Valid range [0,2]
                
                printf("  Padded %d → Original %d (%s)\n",
                       test_padded_indices[i],
                       original_idx,
                       is_valid ? "valid" : "padding");
            }
            
            // Test 3: Boundary analysis
            printf("\n=== Boundary Analysis ===\n");
            printf("Original space: [0, 2] (3 elements)\n");
            printf("Padded space: [0, 4] (5 elements with left_pad=1, right_pad=1)\n");
            printf("Padding regions: [0] = left padding, [4] = right padding\n");
            printf("Valid data region in padded space: [1, 3]\n");
            
            // Test 4: Padding region identification
            printf("\n=== Padding Region Identification ===\n");
            
            for(int padded_idx = 0; padded_idx <= 4; padded_idx++)
            {
                multi_index<1> pad_upper;
                pad_upper[number<0>{}] = padded_idx;
                
                multi_index<1> pad_lower;
                transform.calculate_lower_index(pad_lower, pad_upper);
                
                int orig_idx = static_cast<int>(pad_lower[number<0>{}]);
                
                const char* region_type;
                if (orig_idx < 0) {
                    region_type = "left padding";
                } else if (orig_idx >= 3) {
                    region_type = "right padding";
                } else {
                    region_type = "valid data";
                }
                
                printf("  Padded[%d] → Original[%d] (%s)\n",
                       padded_idx, orig_idx, region_type);
            }
            
            // Test 5: Practical usage explanation
            printf("\n=== Practical Usage ===\n");
            printf("Pad transforms are essential for:\n");
            printf("1. Convolution operations: handling boundary conditions\n");
            printf("2. Stencil computations: extending data access beyond boundaries\n");
            printf("3. Image processing: border handling with padding strategies\n");
            printf("4. Memory access: safe boundary checking in kernels\n\n");
            
            printf("Example: 3x3 convolution on 3-element 1D data\n");
            printf("  - Original data: [A, B, C] (indices 0,1,2)\n");
            printf("  - Padded data: [P, A, B, C, P] (indices 0,1,2,3,4)\n");
            printf("  - Convolution can safely access indices 0-4\n");
            printf("  - Pad transform maps padded indices to original data locations\n");
            
            // Test 6: Performance characteristics
            printf("\n=== Performance Characteristics ===\n");
            printf("Pad transforms have important properties:\n");
            printf("- Boundary checking: enables safe out-of-bounds access\n");
            printf("- Memory efficiency: no actual data duplication\n");
            printf("- Conditional access: can be used with validity checks\n");
            printf("- Zero-copy operation: logical padding without data movement\n");
            
            printf("\n=== Pad Transform Example Complete ===\n");
        }
    }
};

} // namespace ck_tile

int main()
{
    std::cout << "\n=== CK Tile Transform Examples ===\n" << std::endl;
    
    // Run Merge Transform Example
    std::cout << "Running Merge Transform Example:" << std::endl;
    std::cout << "- Forward: 2D coordinates → Linear index" << std::endl;
    std::cout << "- Inverse: Linear index → 2D coordinates" << std::endl;
    std::cout << "- For a 4x5 tensor (20 elements total)\n" << std::endl;

    constexpr ck_tile::index_t kBlockSize  = 128;
    constexpr ck_tile::index_t kBlockPerCu = 1;
    constexpr ck_tile::index_t kGridSize   = 1;

    using MergeKernel = ck_tile::MergeTransformExample;
    float merge_time = launch_kernel(ck_tile::stream_config{nullptr, true, 0, 0, 1},
                                     ck_tile::make_kernel<kBlockSize, kBlockPerCu>(
                                         MergeKernel{},
                                         kGridSize,
                                         kBlockSize,
                                         0));

    std::cout << "\nMerge kernel execution completed. Average time: " << merge_time << " ms" << std::endl;

    // Run Unmerge Transform Example
    std::cout << "\n" << std::string(60, '=') << std::endl;
    std::cout << "Running Unmerge Transform Example:" << std::endl;
    std::cout << "- Forward: Linear index → 3D coordinates" << std::endl;
    std::cout << "- Inverse: 3D coordinates → Linear index" << std::endl;
    std::cout << "- For a 3x4x2 tensor (24 elements total)\n" << std::endl;

    using UnmergeKernel = ck_tile::UnmergeTransformExample;
    float unmerge_time = launch_kernel(ck_tile::stream_config{nullptr, true, 0, 0, 1},
                                       ck_tile::make_kernel<kBlockSize, kBlockPerCu>(
                                           UnmergeKernel{},
                                           kGridSize,
                                           kBlockSize,
                                           0));

    std::cout << "\nUnmerge kernel execution completed. Average time: " << unmerge_time << " ms" << std::endl;

    // Run Embed Transform Example
    std::cout << "\n" << std::string(60, '=') << std::endl;
    std::cout << "Running Embed Transform Example:" << std::endl;
    std::cout << "- Forward: Linear index → 2D coordinates (with custom strides)" << std::endl;
    std::cout << "- Inverse: 2D coordinates → Linear index" << std::endl;
    std::cout << "- For a 2x3 tensor with strides [12, 1]\n" << std::endl;

    using EmbedKernel = ck_tile::EmbedTransformExample;
    float embed_time = launch_kernel(ck_tile::stream_config{nullptr, true, 0, 0, 1},
                                     ck_tile::make_kernel<kBlockSize, kBlockPerCu>(
                                         EmbedKernel{},
                                         kGridSize,
                                         kBlockSize,
                                         0));

    std::cout << "\nEmbed kernel execution completed. Average time: " << embed_time << " ms" << std::endl;

    // Run Replicate Transform Example
    std::cout << "\n" << std::string(60, '=') << std::endl;
    std::cout << "Running Replicate Transform Example:" << std::endl;
    std::cout << "- Forward: Scalar → 2D coordinates (broadcasting)" << std::endl;
    std::cout << "- Inverse: 2D coordinates → Scalar" << std::endl;
    std::cout << "- For 3x4 broadcasting (scalar to all positions)\n" << std::endl;

    using ReplicateKernel = ck_tile::ReplicateTransformExample;
    float replicate_time = launch_kernel(ck_tile::stream_config{nullptr, true, 0, 0, 1},
                                         ck_tile::make_kernel<kBlockSize, kBlockPerCu>(
                                             ReplicateKernel{},
                                             kGridSize,
                                             kBlockSize,
                                             0));

    std::cout << "\nReplicate kernel execution completed. Average time: " << replicate_time << " ms" << std::endl;

    // Run Offset Transform Example
    std::cout << "\n" << std::string(60, '=') << std::endl;
    std::cout << "Running Offset Transform Example:" << std::endl;
    std::cout << "- Forward: Sub-region index → Buffer index (with offset)" << std::endl;
    std::cout << "- Inverse: Buffer index → Sub-region index" << std::endl;
    std::cout << "- For 48-element sub-region at offset 16\n" << std::endl;

    using OffsetKernel = ck_tile::OffsetTransformExample;
    float offset_time = launch_kernel(ck_tile::stream_config{nullptr, true, 0, 0, 1},
                                      ck_tile::make_kernel<kBlockSize, kBlockPerCu>(
                                          OffsetKernel{},
                                          kGridSize,
                                          kBlockSize,
                                          0));

    std::cout << "\nOffset kernel execution completed. Average time: " << offset_time << " ms" << std::endl;

    // Run PassThrough Transform Example
    std::cout << "\n" << std::string(60, '=') << std::endl;
    std::cout << "Running PassThrough Transform Example:" << std::endl;
    std::cout << "- Forward: Index → Same index (identity)" << std::endl;
    std::cout << "- Inverse: Same index → Index (identity)" << std::endl;
    std::cout << "- For 60-element space with perfect identity mapping\n" << std::endl;

    using PassthroughKernel = ck_tile::PassthroughTransformExample;
    float passthrough_time = launch_kernel(ck_tile::stream_config{nullptr, true, 0, 0, 1},
                                           ck_tile::make_kernel<kBlockSize, kBlockPerCu>(
                                               PassthroughKernel{},
                                               kGridSize,
                                               kBlockSize,
                                               0));

    std::cout << "\nPassThrough kernel execution completed. Average time: " << passthrough_time << " ms" << std::endl;

    // Run Pad Transform Example
    std::cout << "\n" << std::string(60, '=') << std::endl;
    std::cout << "Running Pad Transform Example:" << std::endl;
    std::cout << "- Forward: Padded index → Original index (with boundary handling)" << std::endl;
    std::cout << "- Inverse: Original index → Padded index (logical)" << std::endl;
    std::cout << "- For 3-element data with left_pad=1, right_pad=1\n" << std::endl;

    using PadKernel = ck_tile::PadTransformExample;
    float pad_time = launch_kernel(ck_tile::stream_config{nullptr, true, 0, 0, 1},
                                   ck_tile::make_kernel<kBlockSize, kBlockPerCu>(
                                       PadKernel{},
                                       kGridSize,
                                       kBlockSize,
                                       0));

    std::cout << "\nPad kernel execution completed. Average time: " << pad_time << " ms" << std::endl;
    std::cout << "\nAll transform examples completed successfully!" << std::endl;

    return 0;
}
