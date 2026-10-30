// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/ops/epilogue/default_2d_epilogue.hpp"

namespace ck_tile {

/**
 * @brief Problem description of LdsShuffle2DEpilogue.
 *
 * @note The warp count, not the block size, is the template parameter: get_warp_size()
 *       is 64 on the host, where the kernel symbol is formed, and 32 on a wave32 device.
 */
template <typename AccDataType_,
          typename ODataType_,
          index_t kNumWarps_,
          index_t kMPerBlock_,
          index_t kNPerBlock_,
          bool kPadM_,
          bool kPadN_>
struct LdsShuffle2DEpilogueProblem
{
    using AccDataType                   = remove_cvref_t<AccDataType_>;
    using ODataType                     = remove_cvref_t<ODataType_>;
    static constexpr index_t kNumWarps  = kNumWarps_;
    static constexpr index_t kMPerBlock = kMPerBlock_;
    static constexpr index_t kNPerBlock = kNPerBlock_;
    static constexpr bool kPadM         = kPadM_;
    static constexpr bool kPadN         = kPadN_;
    static constexpr index_t NumDTensor = 0;
};

/**
 * @brief Stores an M x N row-major tile through LDS so that the global stores are
 *        coalesced.
 *
 * @details In the wave32 WMMA C layout a lane owns 8 contiguous elements of every 16, so
 *          one store instruction touches 16 rows, each stride_o apart. Transposing the
 *          tile through LDS lets every store instruction cover whole rows. CShuffleEpilogue
 *          does the same for the XDL C layout, which this layout does not fit.
 *
 *          Row widths whose vectors cannot tile a wavefront (e.g. d = 96, 160, 192) keep
 *          the direct store of Default2DEpilogue and need no LDS.
 */
template <typename Problem_, typename Policy_ = void>
struct LdsShuffle2DEpilogue
{
    using Problem     = remove_cvref_t<Problem_>;
    using AccDataType = remove_cvref_t<typename Problem::AccDataType>;
    using ODataType   = remove_cvref_t<typename Problem::ODataType>;

    static constexpr index_t kNumWarps  = Problem::kNumWarps;
    static constexpr index_t kMPerBlock = Problem::kMPerBlock;
    static constexpr index_t kNPerBlock = Problem::kNPerBlock;
    static constexpr bool kPadM         = Problem::kPadM;
    static constexpr bool kPadN         = Problem::kPadN;

    static_assert(Problem::NumDTensor == 0, "D-tensor chaining is not implemented");
    static_assert(sizeof(ODataType) <= 16, "ODataType wider than 16 B is not supported");

    // widest ds_/buffer_ access: 16 bytes
    static constexpr index_t kVectorMax = 16 / static_cast<index_t>(sizeof(ODataType));

    // A vector store carries one out-of-bounds flag, taken from its first element. A padded
    // N is only guaranteed to be a multiple of kPadNAlignment (the qr_tdm dvcheck in
    // fmha_fwd.py), so the vector must divide it or the tail spills into the next row.
    static constexpr index_t kPadNAlignment = 8;
    static constexpr index_t kVector =
        kPadN ? (kVectorMax < kPadNAlignment ? kVectorMax : kPadNAlignment) : kVectorMax;

    // LDS row pitch with row_bytes % 128 == 32, so the rows a wave writes start on four
    // different bank offsets.
    static constexpr index_t kRowBytes = kNPerBlock * static_cast<index_t>(sizeof(ODataType));
    static constexpr index_t kPadBytes = ((32 - kRowBytes) % 128 + 128) % 128;
    static constexpr index_t kNLds =
        kNPerBlock + kPadBytes / static_cast<index_t>(sizeof(ODataType));

    static_assert(kNPerBlock % kVector == 0, "kNPerBlock must be a multiple of the vector size");
    static_assert(kPadBytes % static_cast<index_t>(sizeof(ODataType)) == 0,
                  "LDS row pad must be a whole number of elements");
    static_assert((kRowBytes + kPadBytes) % 128 == 32, "LDS row pitch is not bank-staggered");
    static_assert((kNLds * sizeof(ODataType)) % 16 == 0,
                  "LDS row pitch must stay 16-byte aligned for b128 access");

    /**
     * @brief Whether the thread-raked re-read tiles the block: the vectors of a row must
     *        tile a wavefront and the rows per pass must divide kMPerBlock. Evaluated with
     *        the compiling side's warp size, as the distribution itself is.
     */
    CK_TILE_HOST_DEVICE static constexpr bool IsShuffleTileable()
    {
        constexpr index_t warp_size = get_warp_size();
        constexpr index_t largest   = kMPerBlock * kNPerBlock / (kNumWarps * warp_size);
        constexpr index_t x1        = kVector < largest ? kVector : largest;
        if constexpr(x1 < 1 || kNPerBlock % x1 != 0)
        {
            return false;
        }
        else
        {
            constexpr index_t x0 = warp_size < kNPerBlock / x1 ? warp_size : kNPerBlock / x1;
            constexpr index_t y1 = warp_size / x0;
            return x0 * y1 == warp_size && kNPerBlock % (x0 * x1) == 0 &&
                   kMPerBlock % (y1 * kNumWarps) == 0;
        }
    }

    CK_TILE_HOST_DEVICE static constexpr index_t GetSmemSize()
    {
        return IsShuffleTileable() ? static_cast<index_t>(kMPerBlock * kNLds * sizeof(ODataType))
                                   : 0;
    }

    // Default2DEpilogue's store also keeps the buffer resource of a padded window scalar.
    using DirectStore =
        Default2DEpilogue<Default2DEpilogueProblem<AccDataType, ODataType, kPadM, kPadN>>;

    // operator() takes the LDS block as a 4th argument (see detail::epilogue_uses_smem_v).
    static constexpr bool kUsesSmem = true;

    /**
     * @brief Stores o_acc_tile to o_dram_window_tmp, through LDS at p_smem when given.
     *
     * @details The path is chosen on the type of p_smem: the `nullptr` literal selects the
     *          direct store at compile time, so the unused path takes no registers. The
     *          parameter is defaulted, so three-argument callers of Default2DEpilogue keep
     *          working with the direct store.
     */
    template <typename ODramWindowTmp,
              typename OAccTile,
              typename DsDramWindows,
              typename PSmem = std::nullptr_t>
    CK_TILE_DEVICE auto operator()(ODramWindowTmp& o_dram_window_tmp,
                                   const OAccTile& o_acc_tile,
                                   const DsDramWindows&,
                                   PSmem p_smem = nullptr) const
    {
        if constexpr(std::is_same_v<PSmem, std::nullptr_t> || !IsShuffleTileable())
        {
            ignore = p_smem;
            DirectStore{}(o_dram_window_tmp, o_acc_tile, nullptr);
        }
        else
        {
            const auto lds_desc =
                make_naive_tensor_descriptor(make_tuple(number<kMPerBlock>{}, number<kNPerBlock>{}),
                                             make_tuple(number<kNLds>{}, number<1>{}),
                                             number<kVector>{},
                                             number<1>{});

            auto o_lds = make_tensor_view<address_space_enum::lds>(
                reinterpret_cast<ODataType*>(p_smem), lds_desc);

            auto in_lds_window =
                make_tile_window(o_lds,
                                 make_tuple(number<kMPerBlock>{}, number<kNPerBlock>{}),
                                 {0, 0},
                                 o_acc_tile.get_tile_distribution());

            // The pipeline's last TDM may still be writing LDS: drain the tensor and LDS
            // counters (the single-argument form skips dscnt) before overwriting it.
            s_wait_tensorcnt_barrier<0, 0>();
            store_tile(in_lds_window, cast_tile<ODataType>(o_acc_tile));
            block_sync_lds();

            using TileEncodingPattern =
                tile_distribution_encoding_pattern_2d<kNumWarps * get_warp_size(),
                                                      kMPerBlock,
                                                      kNPerBlock,
                                                      kVector,
                                                      tile_distribution_pattern::thread_raked>;

            constexpr auto dram_dist = TileEncodingPattern::make_2d_static_tile_distribution();

            auto out_lds_window = make_tile_window(
                o_lds, make_tuple(number<kMPerBlock>{}, number<kNPerBlock>{}), {0, 0}, dram_dist);

            DirectStore{}(o_dram_window_tmp, load_tile(out_lds_window), nullptr);
        }
    }
};

} // namespace ck_tile
