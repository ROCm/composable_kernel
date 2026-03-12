// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/core/numeric/integer.hpp"
#include "ck_tile/core/tensor/static_distributed_tensor.hpp"
namespace ck_tile {

/**
 * @brief Compress A vector for 2:4 structured sparsity instruction by moving all non-zero
 * elements into lower part of a_vec to half its effective size.
 * @param a_vec Vector to be compressed.
 * @tparam ADataType The data type of a_vec
 * @tparam CompressedSize The target compression size
 * @tparam AVec The vector type of a_vec (deduced)
 * @return Packed 32‑bit word containing **CompressedSize** 2‑bit fields.
 *         Each field encodes the original position (0–3) of the corresponding
 *         non‑zero element in the input. If fewer than CompressedSize
 *         non‑zeros are found, remaining fields default to 2 (see below).
 */
template <typename ADataType, index_t CompressedSize, typename AVec>
static CK_TILE_DEVICE int32_t compress_a_impl(AVec& a_vec)
{
    // idx holds one 2‑bit index per output element (total CompressedSize entries).
    // It is initialized to the pattern 0b10 for every field. This matches
    // what the hardware expects when there are fewer than two non‑zero values
    // in a 4‑element group – the unused output is treated as coming from slot 2.
    // The loop below will clear and set each field as real non‑zeros are seen.
    int32_t idx = 0;
    static_for<0, CompressedSize, 1>{}([&](auto k) { idx |= (2 << (2 * k)); });

    static_for<0, CompressedSize / 2, 1>{}([&](auto i) {
        ADataType nonzero_elems[2] = {a_vec[i * 4 + 2], a_vec[i * 4 + 3]};
        int32_t non_zero_pos       = 0;

        static_for<0, 3, 1>{}([&](auto j) {
            if(a_vec[i * 4 + j] != 0.0f)
            {
                nonzero_elems[non_zero_pos] = a_vec[i * 4 + j];
                // clear the two‑bit field for this output and insert j
                idx &= ~(0b11 << 2 * (i * 2 + non_zero_pos));
                idx |= j << 2 * (i * 2 + non_zero_pos);
                ++non_zero_pos;
            }
        });
        a_vec[i * 2]     = nonzero_elems[0];
        a_vec[i * 2 + 1] = nonzero_elems[1];
    });

    return idx;
}

template <typename WarpGemmAttribute_>
struct WarpGemmSmfmacImpl
{
    using WarpGemmAttribute = remove_cvref_t<WarpGemmAttribute_>;

    static constexpr index_t kM = WarpGemmAttribute::kM;
    static constexpr index_t kN = WarpGemmAttribute::kN;
    static constexpr index_t kK = WarpGemmAttribute::kK;
    /// @brief The number of elements in K dimension processed by single thread in wavefront.
    ///
    /// @note  Note that WarpGemm may run MFMA instruction multiple times (on different K).
    ///        In such situation this value reflects this fact.
    static constexpr index_t kKPerThread = WarpGemmAttribute::kKPerThread;

    using ADataType = typename WarpGemmAttribute::ADataType;
    using BDataType = typename WarpGemmAttribute::BDataType;
    using CDataType = typename WarpGemmAttribute::CDataType;

    using AWarpDstrEncoding = typename WarpGemmAttribute::AWarpDstrEncoding;
    using BWarpDstrEncoding = typename WarpGemmAttribute::BWarpDstrEncoding;
    using CWarpDstrEncoding = typename WarpGemmAttribute::CWarpDstrEncoding;

    using AWarpDstr = remove_cvref_t<decltype(make_static_tile_distribution(AWarpDstrEncoding{}))>;
    using BWarpDstr = remove_cvref_t<decltype(make_static_tile_distribution(BWarpDstrEncoding{}))>;
    using CWarpDstr = remove_cvref_t<decltype(make_static_tile_distribution(CWarpDstrEncoding{}))>;

    using AWarpTensor = static_distributed_tensor<ADataType, AWarpDstr>;
    using BWarpTensor = static_distributed_tensor<BDataType, BWarpDstr>;
    using CWarpTensor = static_distributed_tensor<CDataType, CWarpDstr>;

    CK_TILE_HOST_DEVICE static constexpr auto get_num_of_access()
    {
        return WarpGemmAttribute_::get_num_of_access();
    }

    template <index_t CompressedSize, typename AVec>
    CK_TILE_DEVICE int32_t compress_a_vec(AVec& a_vec)
    {
        return compress_a_impl<ADataType, CompressedSize>(a_vec);
    }

    template <typename CTensor, typename ATensor, typename BTensor, bool post_nop_ = false>
    CK_TILE_DEVICE void
    operator()(CTensor& c, const ATensor& a, const BTensor& b, bool_constant<post_nop_> = {}) const
    {
        static_assert(detail::is_similiar_distributed_tensor_v<CTensor, CWarpTensor> &&
                      detail::is_similiar_distributed_tensor_v<ATensor, AWarpTensor> &&
                      detail::is_similiar_distributed_tensor_v<BTensor, BWarpTensor>);
        constexpr auto CompressionRatio = WarpGemmAttribute::kCompressionRatio;

        using AVec = ext_vector_t<ADataType, ATensor::get_thread_buffer_size()>;
        static constexpr index_t CompressedSize =
            ATensor::get_thread_buffer_size() / CompressionRatio;
        using AVecCompressed = ext_vector_t<ADataType, CompressedSize>;
        using BVec           = ext_vector_t<BDataType, BTensor::get_thread_buffer_size()>;
        using CVec           = ext_vector_t<CDataType, CTensor::get_thread_buffer_size()>;

        constexpr auto I0 = number<0>{};

        auto a_vec       = a.get_thread_buffer().template get_as<AVec>()[I0];
        const auto b_vec = b.get_thread_buffer().template get_as<BVec>()[I0];
        auto c_vec       = c.get_thread_buffer().template get_as<CVec>()[I0];

        const int32_t idx = compress_a_vec<CompressedSize>(a_vec);

        static_assert(CompressedSize == 4);
        // @TODO can we simply set a_vec_pruned to a_vec[0:3]?
        const AVecCompressed a_vec_pruned = {a_vec[0], a_vec[1], a_vec[2], a_vec[3]};

        // c_vec += a_vec * b_vec[idx]
        WarpGemmAttribute{}(c_vec, a_vec_pruned, b_vec, idx, bool_constant<post_nop_>{});

        c.get_thread_buffer().template set_as<CVec>(I0, c_vec);
    }
};

} // namespace ck_tile
