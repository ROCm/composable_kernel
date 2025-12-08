// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck/ck.hpp"
#include "ck/tensor_operation/gpu/device/tensor_layout.hpp"
#include "ck/library/reference_tensor_operation/gpu/conv_common.hpp"
#include <vector>

namespace ck {
namespace ref {
namespace layout_utils {

// Helper to compute permutation from one layout to another
// This is a compile-time mapping from layout types to dimension orderings

// Dimension indices for 6D tensors (grouped convolutions)
// For input: G=0, N=1, C=2, D=3, H=4, W=5
// For weight: G=0, K=1, C=2, Z=3, Y=4, X=5
// For output: G=0, N=1, K=2, D=3, H=4, W=5

// ===== INPUT LAYOUT MAPPINGS =====

// GNCDHW: [G, N, C, D, H, W] = [0, 1, 2, 3, 4, 5] (canonical order)
template <typename Layout>
struct InputLayoutTrait;

template <>
struct InputLayoutTrait<tensor_layout::convolution::GNCDHW>
{
    static constexpr int num_dims = 6;
    static std::vector<int> dim_order() { return {0, 1, 2, 3, 4, 5}; } // G,N,C,D,H,W
};

template <>
struct InputLayoutTrait<tensor_layout::convolution::NDHWGC>
{
    static constexpr int num_dims = 6;
    static std::vector<int> dim_order() { return {1, 3, 4, 5, 0, 2}; } // N,D,H,W,G,C
};

template <>
struct InputLayoutTrait<tensor_layout::convolution::GNDHWC>
{
    static constexpr int num_dims = 6;
    static std::vector<int> dim_order() { return {0, 1, 3, 4, 5, 2}; } // G,N,D,H,W,C
};

template <>
struct InputLayoutTrait<tensor_layout::convolution::NGCDHW>
{
    static constexpr int num_dims = 6;
    static std::vector<int> dim_order() { return {1, 0, 2, 3, 4, 5}; } // N,G,C,D,H,W
};

// 2D variants (GNCHW, NHWGC, etc.)
template <>
struct InputLayoutTrait<tensor_layout::convolution::GNCHW>
{
    static constexpr int num_dims = 5;
    static std::vector<int> dim_order() { return {0, 1, 2, 3, 4}; } // G,N,C,H,W
};

template <>
struct InputLayoutTrait<tensor_layout::convolution::NHWGC>
{
    static constexpr int num_dims = 5;
    static std::vector<int> dim_order() { return {1, 3, 4, 0, 2}; } // N,H,W,G,C
};

template <>
struct InputLayoutTrait<tensor_layout::convolution::GNHWC>
{
    static constexpr int num_dims = 5;
    static std::vector<int> dim_order() { return {0, 1, 3, 4, 2}; } // G,N,H,W,C
};

template <>
struct InputLayoutTrait<tensor_layout::convolution::NGCHW>
{
    static constexpr int num_dims = 5;
    static std::vector<int> dim_order() { return {1, 0, 2, 3, 4}; } // N,G,C,H,W
};

// 1D variants
template <>
struct InputLayoutTrait<tensor_layout::convolution::GNCW>
{
    static constexpr int num_dims = 4;
    static std::vector<int> dim_order() { return {0, 1, 2, 3}; } // G,N,C,W
};

template <>
struct InputLayoutTrait<tensor_layout::convolution::NWGC>
{
    static constexpr int num_dims = 4;
    static std::vector<int> dim_order() { return {1, 3, 0, 2}; } // N,W,G,C
};

template <>
struct InputLayoutTrait<tensor_layout::convolution::GNWC>
{
    static constexpr int num_dims = 4;
    static std::vector<int> dim_order() { return {0, 1, 3, 2}; } // G,N,W,C
};

template <>
struct InputLayoutTrait<tensor_layout::convolution::NGCW>
{
    static constexpr int num_dims = 4;
    static std::vector<int> dim_order() { return {1, 0, 2, 3}; } // N,G,C,W
};

// ===== WEIGHT LAYOUT MAPPINGS =====

template <typename Layout>
struct WeightLayoutTrait;

template <>
struct WeightLayoutTrait<tensor_layout::convolution::GKCZYX>
{
    static constexpr int num_dims = 6;
    static std::vector<int> dim_order() { return {0, 1, 2, 3, 4, 5}; } // G,K,C,Z,Y,X
};

template <>
struct WeightLayoutTrait<tensor_layout::convolution::KZYXGC>
{
    static constexpr int num_dims = 6;
    static std::vector<int> dim_order() { return {1, 3, 4, 5, 0, 2}; } // K,Z,Y,X,G,C
};

template <>
struct WeightLayoutTrait<tensor_layout::convolution::GKZYXC>
{
    static constexpr int num_dims = 6;
    static std::vector<int> dim_order() { return {0, 1, 3, 4, 5, 2}; } // G,K,Z,Y,X,C
};

// 2D variants
template <>
struct WeightLayoutTrait<tensor_layout::convolution::GKCYX>
{
    static constexpr int num_dims = 5;
    static std::vector<int> dim_order() { return {0, 1, 2, 3, 4}; } // G,K,C,Y,X
};

template <>
struct WeightLayoutTrait<tensor_layout::convolution::KYXGC>
{
    static constexpr int num_dims = 5;
    static std::vector<int> dim_order() { return {1, 3, 4, 0, 2}; } // K,Y,X,G,C
};

template <>
struct WeightLayoutTrait<tensor_layout::convolution::GKYXC>
{
    static constexpr int num_dims = 5;
    static std::vector<int> dim_order() { return {0, 1, 3, 4, 2}; } // G,K,Y,X,C
};

// 1D variants
template <>
struct WeightLayoutTrait<tensor_layout::convolution::GKCX>
{
    static constexpr int num_dims = 4;
    static std::vector<int> dim_order() { return {0, 1, 2, 3}; } // G,K,C,X
};

template <>
struct WeightLayoutTrait<tensor_layout::convolution::KXGC>
{
    static constexpr int num_dims = 4;
    static std::vector<int> dim_order() { return {1, 3, 0, 2}; } // K,X,G,C
};

template <>
struct WeightLayoutTrait<tensor_layout::convolution::GKXC>
{
    static constexpr int num_dims = 4;
    static std::vector<int> dim_order() { return {0, 1, 3, 2}; } // G,K,X,C
};

// ===== OUTPUT LAYOUT MAPPINGS =====

template <typename Layout>
struct OutputLayoutTrait;

template <>
struct OutputLayoutTrait<tensor_layout::convolution::GNKDHW>
{
    static constexpr int num_dims = 6;
    static std::vector<int> dim_order() { return {0, 1, 2, 3, 4, 5}; } // G,N,K,D,H,W
};

template <>
struct OutputLayoutTrait<tensor_layout::convolution::NDHWGK>
{
    static constexpr int num_dims = 6;
    static std::vector<int> dim_order() { return {1, 3, 4, 5, 0, 2}; } // N,D,H,W,G,K
};

template <>
struct OutputLayoutTrait<tensor_layout::convolution::GNDHWK>
{
    static constexpr int num_dims = 6;
    static std::vector<int> dim_order() { return {0, 1, 3, 4, 5, 2}; } // G,N,D,H,W,K
};

template <>
struct OutputLayoutTrait<tensor_layout::convolution::NGKDHW>
{
    static constexpr int num_dims = 6;
    static std::vector<int> dim_order() { return {1, 0, 2, 3, 4, 5}; } // N,G,K,D,H,W
};

// 2D variants
template <>
struct OutputLayoutTrait<tensor_layout::convolution::GNKHW>
{
    static constexpr int num_dims = 5;
    static std::vector<int> dim_order() { return {0, 1, 2, 3, 4}; } // G,N,K,H,W
};

template <>
struct OutputLayoutTrait<tensor_layout::convolution::NHWGK>
{
    static constexpr int num_dims = 5;
    static std::vector<int> dim_order() { return {1, 3, 4, 0, 2}; } // N,H,W,G,K
};

template <>
struct OutputLayoutTrait<tensor_layout::convolution::GNHWK>
{
    static constexpr int num_dims = 5;
    static std::vector<int> dim_order() { return {0, 1, 3, 4, 2}; } // G,N,H,W,K
};

template <>
struct OutputLayoutTrait<tensor_layout::convolution::NGKHW>
{
    static constexpr int num_dims = 5;
    static std::vector<int> dim_order() { return {1, 0, 2, 3, 4}; } // N,G,K,H,W
};

// 1D variants
template <>
struct OutputLayoutTrait<tensor_layout::convolution::GNKW>
{
    static constexpr int num_dims = 4;
    static std::vector<int> dim_order() { return {0, 1, 2, 3}; } // G,N,K,W
};

template <>
struct OutputLayoutTrait<tensor_layout::convolution::NWGK>
{
    static constexpr int num_dims = 4;
    static std::vector<int> dim_order() { return {1, 3, 0, 2}; } // N,W,G,K
};

template <>
struct OutputLayoutTrait<tensor_layout::convolution::GNWK>
{
    static constexpr int num_dims = 4;
    static std::vector<int> dim_order() { return {0, 1, 3, 2}; } // G,N,W,K
};

template <>
struct OutputLayoutTrait<tensor_layout::convolution::NGKW>
{
    static constexpr int num_dims = 4;
    static std::vector<int> dim_order() { return {1, 0, 2, 3}; } // N,G,K,W
};

// Helper function to compute permutation from source layout to destination layout
// Given source_order and dest_order, compute perm such that dest[i] = src[perm[i]]
inline std::vector<int> compute_permutation(const std::vector<int>& src_order,
                                            const std::vector<int>& dst_order)
{
    int num_dims = src_order.size();
    std::vector<int> perm(num_dims);

    // For each destination position, find which source position has that dimension
    for(int dst_pos = 0; dst_pos < num_dims; ++dst_pos)
    {
        int dst_dim = dst_order[dst_pos]; // Which dimension goes in dst position dst_pos
        // Find which source position contains this dimension
        for(int src_pos = 0; src_pos < num_dims; ++src_pos)
        {
            if(src_order[src_pos] == dst_dim)
            {
                perm[dst_pos] = src_pos;
                break;
            }
        }
    }

    return perm;
}

// Canonical dimension ordering for naive kernels
// Input:  NDHWGC (naive kernel uses flat C, but we separate into G and C_per_group)
// Weight: KZYXGC (naive kernel uses flat C, but we separate into G and C_per_group)
// Output: NDHWGK (naive kernel uses flat K, but we separate into G and K_per_group)

inline std::vector<int> get_naive_input_order_3d()
{
    return {1, 3, 4, 5, 0, 2}; // N,D,H,W,G,C
}

inline std::vector<int> get_naive_input_order_2d()
{
    return {1, 3, 4, 0, 2}; // N,H,W,G,C
}

inline std::vector<int> get_naive_input_order_1d()
{
    return {1, 3, 0, 2}; // N,W,G,C
}

inline std::vector<int> get_naive_weight_order_3d()
{
    return {1, 3, 4, 5, 0, 2}; // K,Z,Y,X,G,C
}

inline std::vector<int> get_naive_weight_order_2d()
{
    return {1, 3, 4, 0, 2}; // K,Y,X,G,C
}

inline std::vector<int> get_naive_weight_order_1d()
{
    return {1, 3, 0, 2}; // K,X,G,C
}

inline std::vector<int> get_naive_output_order_3d()
{
    return {1, 3, 4, 5, 0, 2}; // N,D,H,W,G,K
}

inline std::vector<int> get_naive_output_order_2d()
{
    return {1, 3, 4, 0, 2}; // N,H,W,G,K
}

inline std::vector<int> get_naive_output_order_1d()
{
    return {1, 3, 0, 2}; // N,W,G,K
}

// Helper functions to build dimension vectors for transformations
// These separate G and C_per_group (or K_per_group) as distinct dimensions

inline std::vector<index_t> build_input_dims(const ConvDims& dims, index_t NDimSpatial)
{
    index_t C_per_group = dims.C / dims.G;
    if(NDimSpatial == 3)
        return {dims.G, dims.N, C_per_group, dims.Di, dims.Hi, dims.Wi};
    else if(NDimSpatial == 2)
        return {dims.G, dims.N, C_per_group, dims.Hi, dims.Wi};
    else // 1D
        return {dims.G, dims.N, C_per_group, dims.Wi};
}

inline std::vector<index_t> build_weight_dims(const ConvDims& dims, index_t NDimSpatial)
{
    index_t K_per_group = dims.K / dims.G;
    index_t C_per_group = dims.C / dims.G;
    if(NDimSpatial == 3)
        return {dims.G, K_per_group, C_per_group, dims.Z, dims.Y, dims.X};
    else if(NDimSpatial == 2)
        return {dims.G, K_per_group, C_per_group, dims.Y, dims.X};
    else // 1D
        return {dims.G, K_per_group, C_per_group, dims.X};
}

inline std::vector<index_t> build_output_dims(const ConvDims& dims, index_t NDimSpatial)
{
    index_t K_per_group = dims.K / dims.G;
    if(NDimSpatial == 3)
        return {dims.G, dims.N, K_per_group, dims.Do, dims.Ho, dims.Wo};
    else if(NDimSpatial == 2)
        return {dims.G, dims.N, K_per_group, dims.Ho, dims.Wo};
    else // 1D
        return {dims.G, dims.N, K_per_group, dims.Wo};
}

// Helper functions to build naive kernel dimension vectors (for transformation targets)
inline std::vector<index_t> build_naive_input_dims(const ConvDims& dims, index_t NDimSpatial)
{
    index_t C_per_group = dims.C / dims.G;
    if(NDimSpatial == 3)
        return {dims.N, dims.Di, dims.Hi, dims.Wi, dims.G, C_per_group};
    else if(NDimSpatial == 2)
        return {dims.N, dims.Hi, dims.Wi, dims.G, C_per_group};
    else // 1D
        return {dims.N, dims.Wi, dims.G, C_per_group};
}

inline std::vector<index_t> build_naive_weight_dims(const ConvDims& dims, index_t NDimSpatial)
{
    index_t K_per_group = dims.K / dims.G;
    index_t C_per_group = dims.C / dims.G;
    if(NDimSpatial == 3)
        return {K_per_group, dims.Z, dims.Y, dims.X, dims.G, C_per_group};
    else if(NDimSpatial == 2)
        return {K_per_group, dims.Y, dims.X, dims.G, C_per_group};
    else // 1D
        return {K_per_group, dims.X, dims.G, C_per_group};
}

inline std::vector<index_t> build_naive_output_dims(const ConvDims& dims, index_t NDimSpatial)
{
    index_t K_per_group = dims.K / dims.G;
    if(NDimSpatial == 3)
        return {dims.N, dims.Do, dims.Ho, dims.Wo, dims.G, K_per_group};
    else if(NDimSpatial == 2)
        return {dims.N, dims.Ho, dims.Wo, dims.G, K_per_group};
    else // 1D
        return {dims.N, dims.Wo, dims.G, K_per_group};
}

} // namespace layout_utils
} // namespace ref
} // namespace ck
