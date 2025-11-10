// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2025, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck_tile/core/config.hpp"
#include "ck_tile/core/numeric/integer.hpp"
#include "ck_tile/core/numeric/integral_constant.hpp"
#include "ck_tile/core/numeric/numeric.hpp"
#include "ck_tile/core/utility/type_traits.hpp"
#include "ck_tile/core/tensor/tensor_descriptor.hpp"
#include "ck_tile/core/tensor/tile_distribution.hpp"
#include "ck_tile/core/arch/arch.hpp"

namespace ck_tile
{

// A static distributed tensor backed by shared memory (LDS),
// with indexing defined by a naive packed tensor descriptor of the Y-dimensions
// of the provided StaticTileDistribution.
//
// Notes:
// - This does NOT allocate LDS. Callers must provide a __shared__-allocated buffer
//   of size get_lds_buffer_size() elements of DataType.
// - Index mapping uses a naive packed descriptor derived from the Y lengths of the
//   distribution to compute a linear offset into LDS.
// - Distributed indexing API mirrors static_distributed_tensor (operator[]/operator()).
//
// Example (inside __global__ kernel):
//   using Dist = decltype(make_static_tile_distribution(my_encoding{}));
//   using Tensor = ck_tile::static_distributed_lds_tensor<float, Dist>;
//
//   // Compile-time LDS size in elements
//   __shared__ float lds[Tensor::get_lds_buffer_size()];
//
//   Tensor t{lds};
//   // write using distributed indices (compile-time indices)
//   t(ck_tile::detail::make_tile_distributed_index(ck_tile::sequence<0>{},
//                                                  ck_tile::sequence<0>{})) = 1.0f;
//   ck_tile::block_sync_lds();
//
//   // read back similarly
//   auto v = t[ck_tile::detail::make_tile_distributed_index(ck_tile::sequence<0>{},
//                                                           ck_tile::sequence<0>{})];
//
template <typename DataType_, typename StaticTileDistribution_>
struct static_distributed_lds_tensor
{
    using DataType               = remove_cvref_t<DataType_>;
    using StaticTileDistribution = remove_cvref_t<StaticTileDistribution_>;

    static_assert(StaticTileDistribution::is_static(),
                  "StaticTileDistribution must be fully known at compile time");

    // Build a naive packed Y descriptor based on the distribution's Y lengths.
    // This mirrors the usage pattern from static_distributed_tensor where offsets are
    // computed through the Ys-to-D descriptor, but here we explicitly construct a naive
    // packed descriptor to satisfy the requirement.
    using YsDescriptorOriginal =
        remove_cvref_t<decltype(StaticTileDistribution{}.get_ys_to_d_descriptor())>;

    using YsNaivePackedDescriptor =
        remove_cvref_t<decltype(make_naive_tensor_descriptor_packed(YsDescriptorOriginal{}.get_lengths()))>;

    using ThreadTensorDesc = YsNaivePackedDescriptor;

    static constexpr index_t PackedSize =
        ck_tile::numeric_traits<remove_cvref_t<DataType>>::PackedSize;

    static constexpr index_t kLdsElementSpaceSize = ThreadTensorDesc{}.get_element_space_size();
    static_assert(kLdsElementSpaceSize > 0, "Make sure tile distribution is valid for LDS");

    // Number of DataType elements required in LDS buffer (considering PackedSize for DataType)
    CK_TILE_HOST_DEVICE static constexpr index_t get_lds_buffer_size()
    {
        return kLdsElementSpaceSize / PackedSize;
    }

    CK_TILE_HOST_DEVICE static constexpr auto get_num_of_dimension()
    {
        return StaticTileDistribution::get_num_of_dimension_x();
    }

    CK_TILE_HOST_DEVICE static constexpr auto get_lengths()
    {
        return StaticTileDistribution::get_lengths();
    }

    CK_TILE_HOST_DEVICE static constexpr auto get_tile_distribution()
    {
        return StaticTileDistribution{};
    }

    CK_TILE_HOST_DEVICE static constexpr auto get_distributed_spans()
    {
        return StaticTileDistribution::get_distributed_spans();
    }

    // Construct with an LDS pointer (caller-provided __shared__ storage).
    CK_TILE_HOST_DEVICE explicit constexpr static_distributed_lds_tensor(DataType* lds_ptr)
        : p_lds_{lds_ptr}
    {
    }

    // Bind or rebind LDS pointer at runtime if needed.
    CK_TILE_DEVICE void bind(DataType* lds_ptr) { p_lds_ = lds_ptr; }

    CK_TILE_DEVICE constexpr const DataType* data() const { return p_lds_; }
    CK_TILE_DEVICE constexpr DataType* data() { return p_lds_; }

    // Distributed indexing: const access
    template <typename TileDistributedIndices>
    CK_TILE_HOST_DEVICE constexpr const DataType& operator[](TileDistributedIndices) const
    {
        static_assert(is_static_v<TileDistributedIndices>,
                      "TileDistributedIndices must be static");
        constexpr auto y_idx = get_tile_distribution().get_y_indices_from_distributed_indices(
            TileDistributedIndices{});
        constexpr index_t linear = ThreadTensorDesc{}.calculate_offset(y_idx) / PackedSize;
        return p_lds_[linear];
    }

    // Distributed indexing: mutable access
    template <typename TileDistributedIndices>
    CK_TILE_HOST_DEVICE constexpr DataType& operator()(TileDistributedIndices)
    {
        static_assert(is_static_v<TileDistributedIndices>,
                      "TileDistributedIndices must be static");
        constexpr auto y_idx = get_tile_distribution().get_y_indices_from_distributed_indices(
            TileDistributedIndices{});
        constexpr index_t linear = ThreadTensorDesc{}.calculate_offset(y_idx) / PackedSize;
        return p_lds_[linear];
    }

    // Returns the naive Y-descriptor (type-only, constexpr-friendly)
    CK_TILE_HOST_DEVICE static constexpr auto get_y_naive_packed_descriptor()
    {
        return ThreadTensorDesc{};
    }

private:
    DataType* p_lds_ = nullptr; // shared (LDS) pointer provided by caller
};

// Helper: make function
template <typename DataType, typename StaticTileDistribution>
CK_TILE_HOST_DEVICE constexpr auto
make_static_distributed_lds_tensor(DataType* lds_ptr, const StaticTileDistribution&)
{
    return static_distributed_lds_tensor<remove_cvref_t<DataType>,
                                         remove_cvref_t<StaticTileDistribution>>{lds_ptr};
}

// Utility to query required LDS size (in elements of DataType) for a given distribution
template <typename DataType, typename StaticTileDistribution>
CK_TILE_HOST_DEVICE constexpr index_t
get_required_lds_size_elems(const StaticTileDistribution&)
{
    using tensor_t = static_distributed_lds_tensor<remove_cvref_t<DataType>,
                                                   remove_cvref_t<StaticTileDistribution>>;
    return tensor_t::get_lds_buffer_size();
}

} // namespace ck_tile
