// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck/ck.hpp"
#include "ck/tensor_operation/gpu/device/tensor_layout.hpp"
#include <vector>

namespace ck {
namespace ref {

// Helper function to compute layout-aware strides for convolution tensors
// For channel-last layouts (GNHWC, GKYXC, GNHWK): C/K is the innermost dimension
// For channel-first layouts (GNCDHW, GKCZYX, GNKDHW): spatial dimensions are innermost
inline std::vector<index_t> compute_conv_tensor_strides(const std::vector<index_t>& lengths,
                                                        index_t ndim_spatial,
                                                        bool channel_last)
{
    std::vector<index_t> strides(lengths.size());

    if(channel_last)
    {
        // Channel-last layout: spatial dimensions come before C/K in memory
        // lengths[0] = G, lengths[1] = N/K, lengths[2] = C/K, lengths[3...] = spatial
        // Memory order: G, N/K, spatial..., C/K
        strides[2]     = 1; // C/K is innermost
        index_t stride = static_cast<index_t>(lengths[2]);

        // Spatial dimensions in reverse order
        for(int i = ndim_spatial + 2; i >= 3; --i)
        {
            strides[i] = stride;
            stride *= lengths[i];
        }

        // N/K
        strides[1] = stride;
        stride *= lengths[1];

        // G
        strides[0] = stride;
    }
    else
    {
        // Row-major layout (channel-first or fallback)
        // Memory order follows index order: G, N/K, C/K, spatial...
        index_t stride = 1;
        for(int i = lengths.size() - 1; i >= 0; --i)
        {
            strides[i] = stride;
            stride *= lengths[i];
        }
    }

    return strides;
}

// Template helper to detect if a layout is channel-last (C or K as innermost dimension)
template <typename Layout>
constexpr bool is_channel_last_layout()
{
    using namespace ck::tensor_layout::convolution;

    // Input layouts with C last
    if constexpr(std::is_same_v<Layout, NWC> || std::is_same_v<Layout, NHWC> ||
                 std::is_same_v<Layout, NDHWC> || std::is_same_v<Layout, GNWC> ||
                 std::is_same_v<Layout, GNHWC> || std::is_same_v<Layout, GNDHWC> ||
                 std::is_same_v<Layout, NWGC> || std::is_same_v<Layout, NHWGC> ||
                 std::is_same_v<Layout, NDHWGC>)
    {
        return true;
    }
    // Weight layouts with C last
    else if constexpr(std::is_same_v<Layout, KXC> || std::is_same_v<Layout, KYXC> ||
                      std::is_same_v<Layout, KZYXC> || std::is_same_v<Layout, GKXC> ||
                      std::is_same_v<Layout, GKYXC> || std::is_same_v<Layout, GKZYXC> ||
                      std::is_same_v<Layout, KXGC> || std::is_same_v<Layout, KYXGC> ||
                      std::is_same_v<Layout, KZYXGC>)
    {
        return true;
    }
    // Output layouts with K last
    else if constexpr(std::is_same_v<Layout, NWK> || std::is_same_v<Layout, NHWK> ||
                      std::is_same_v<Layout, NDHWK> || std::is_same_v<Layout, GNWK> ||
                      std::is_same_v<Layout, GNHWK> || std::is_same_v<Layout, GNDHWK> ||
                      std::is_same_v<Layout, NWGK> || std::is_same_v<Layout, NHWGK> ||
                      std::is_same_v<Layout, NDHWGK>)
    {
        return true;
    }
    else
    {
        return false;
    }
}

} // namespace ref
} // namespace ck
