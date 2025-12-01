// SPDX-License-Identifier: MIT
// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include <numeric>
#include "ck/utility/common_header.hpp"
#include "ck/tensor_operation/gpu/device/impl/device_grouped_conv_utils.hpp"

namespace ck {
namespace tensor_operation {
namespace device {

// Check if a tensor descriptor has compact layout
// Compact means: GetElementSpaceSize() == product of all dimension lengths
// Non-compact descriptors have complex transform pipelines that may not support split-k hack
template <typename Descriptor>
bool IsDescriptorCompact(const Descriptor& desc)
{
    // Calculate product of all dimensions
    long_index_t dims_product  = 1;
    constexpr index_t num_dims = Descriptor::GetNumOfDimension();

    // Use template recursion to multiply all dimension lengths
    static_for<0, num_dims, 1>{}(
        [&](auto i) { dims_product *= static_cast<long_index_t>(desc.GetLength(i)); });

    return desc.GetElementSpaceSize() == dims_product;
}

// Determine split-k hack eligibility for descriptor pair
// This checks all the conditions required for safely using the split-k offset hack
template <index_t NDimSpatial, typename InLayout, typename WeiLayout, typename OutLayout>
struct SplitKHackEligibility
{
    template <typename ADescriptor, typename BDescriptor>
    static auto
    Check(const ADescriptor& a_desc,
          const BDescriptor& b_desc,
          index_t k_batch,
          index_t Conv_N,
          const std::array<index_t, NDimSpatial>& output_spatial_lengths,
          index_t k_block_size) // K0PerBlock*K1 for v1, K0PerBlock for v3, KPerBlock for two-stage
    {
        // Only enable hack if k_batch > 1
        if(k_batch <= 1)
        {
            return std::make_pair(false, false);
        }

        // Calculate output spatial product
        const index_t output_spatial_acum = std::accumulate(output_spatial_lengths.begin(),
                                                            output_spatial_lengths.end(),
                                                            index_t{1},
                                                            std::multiplies<index_t>());

        // Check various divisibility and layout requirements
        const bool is_k_not_paded = (Conv_N * output_spatial_acum) % (k_block_size * k_batch) == 0;

        const bool can_divide_n_spatial_by_k_batch = (Conv_N * output_spatial_acum) % k_batch == 0;

        const bool can_divide_n_by_k_batch = Conv_N % k_batch == 0;

        const bool is_correct_layout =
            is_NSpatialGC_GKSpatial_NSpatialGK<InLayout, WeiLayout, OutLayout>();

        const bool is_a_stride_divisible = a_desc.GetElementSpaceSize() % k_batch == 0;

        const bool is_b_stride_divisible = b_desc.GetElementSpaceSize() % k_batch == 0;

        // Check descriptor compactness
        const bool is_a_compact = IsDescriptorCompact(a_desc);
        const bool is_b_compact = IsDescriptorCompact(b_desc);

        // Determine hack flags based on all conditions
        const bool split_k_offset_a_hack = can_divide_n_spatial_by_k_batch && is_k_not_paded &&
                                           is_correct_layout && is_a_stride_divisible &&
                                           is_a_compact;

        const bool split_k_offset_b_hack = can_divide_n_by_k_batch && is_k_not_paded &&
                                           is_correct_layout && is_b_stride_divisible &&
                                           is_b_compact;

        return std::make_pair(split_k_offset_a_hack, split_k_offset_b_hack);
    }
};

} // namespace device
} // namespace tensor_operation
} // namespace ck
