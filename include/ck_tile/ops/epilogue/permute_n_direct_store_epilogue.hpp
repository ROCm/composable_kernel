// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
#pragma once

#include "ck_tile/host/concat.hpp"
#include "ck_tile/core.hpp"
#include "ck_tile/ops/common/utils.hpp"
#include "ck_tile/ops/common/tensor_layout.hpp"
#include "ck_tile/ops/elementwise/unary_element_wise_operation.hpp"
#include "ck_tile/ops/gemm/warp/warp_gemm_dispatcher.hpp"

#include <type_traits>

namespace ck_tile {

/**
 * PermuteNDirectStoreEpilogue
 *
 * - No LDS, no barriers.
 * - Assumes the accumulator layout + TilePermuteN pre-processing makes it possible
 *   to store to DRAM after a small in-thread reorder.
 *
 * Recommended usage:
 *   - Use this epilogue only under strict compile-time constraints.
 *   - Otherwise fall back to LDS-based CShuffleEpilogue.
 */
template <typename Problem_, typename Policy_ = void>
struct PermuteNDirectStoreEpilogue
{
    using Problem     = remove_cvref_t<Problem_>;
    using AsDataType  = remove_cvref_t<typename Problem::AsDataType>;
    using BsDataType  = remove_cvref_t<typename Problem::BsDataType>;
    using AccDataType = remove_cvref_t<typename Problem::AccDataType>;
    using ODataType   = remove_cvref_t<typename Problem::ODataType>;
    using DsDataType  = remove_cvref_t<typename Problem::DsDataType>;
    using DsLayout    = remove_cvref_t<typename Problem::DsLayout>;
    using ELayout     = remove_cvref_t<typename Problem::ELayout>;
    using CDElementwise = remove_cvref_t<typename Problem::CDElementwise>;

    static constexpr bool ADataTypeIsTuple = is_detected<is_tuple, AsDataType>::value;
    static constexpr bool BDataTypeIsTuple = is_detected<is_tuple, BsDataType>::value;

    using AsDataTypeTuple = std::conditional_t<ADataTypeIsTuple,
                                               remove_cvref_t<AsDataType>,
                                               remove_cvref_t<tuple<AsDataType>>>;
    using BsDataTypeTuple = std::conditional_t<BDataTypeIsTuple,
                                               remove_cvref_t<BsDataType>,
                                               remove_cvref_t<tuple<BsDataType>>>;

    using ADataType = remove_cvref_t<std::tuple_element_t<number<0>{}, AsDataTypeTuple>>;
    using BDataType = remove_cvref_t<std::tuple_element_t<number<0>{}, BsDataTypeTuple>>;

    // Match CShuffleEpilogue’s type selection rules (for WQ kernels etc.)
    using ATypeToUse = std::conditional_t<std::is_same_v<ADataType, pk_int4_t> ||
                                              std::is_same_v<ADataType, pk_fp4_t>,
                                          BDataType,
                                          ADataType>;
    using BTypeToUse = std::conditional_t<std::is_same_v<BDataType, pk_int4_t> ||
                                              std::is_same_v<BDataType, pk_fp4_t> ||
                                              std::is_same_v<BDataType, pk_fp4_raw_t>,
                                          ADataType,
                                          BDataType>;

    static constexpr index_t kBlockSize          = Problem::kBlockSize;
    static constexpr index_t kMPerBlock          = Problem::kMPerBlock;
    static constexpr index_t kNPerBlock          = Problem::kNPerBlock;
    static constexpr index_t MWave               = Problem::MWave;
    static constexpr index_t NWave               = Problem::NWave;
    static constexpr index_t MPerXdl             = Problem::MPerXdl;
    static constexpr index_t NPerXdl             = Problem::NPerXdl;
    static constexpr index_t KPerXdl             = Problem::KPerXdl;
    static constexpr index_t isCTransposed       = Problem::isCTransposed;
    static constexpr index_t BlockedXDLN_PerWarp = Problem::BlockedXDLN_PerWarp;
    static constexpr index_t NumDTensor          = Problem::NumDTensor;

    static constexpr index_t MPerIteration = MPerXdl * MWave;
    static constexpr index_t NPerIteration = NPerXdl * NWave;
    static constexpr index_t MRepeat       = kMPerBlock / (MPerXdl * MWave);
    static constexpr index_t NRepeat       = kNPerBlock / (NPerXdl * NWave);

    CDElementwise elfunc_;

    CK_TILE_DEVICE PermuteNDirectStoreEpilogue(CDElementwise elfunc = CDElementwise{})
        : elfunc_(elfunc)
    {
    }

    [[nodiscard]] CK_TILE_HOST static const std::string GetName()
    {
        // clang-format off
        return concat('_', "PermuteNDirectStoreEpilogue",
                      concat('x', MWave, NWave),
                      concat('x', MPerXdl, NPerXdl, KPerXdl),
                      isCTransposed ? "CTransposed" : "CNotTransposed");
        // clang-format on
    }

    // Same EmptyScale / ScaleDataType pattern as CShuffleEpilogue
    struct EmptyScale {};

    template <typename, typename = void>
    struct ScaleDataType { using DataType = float; };

    template <typename T>
    struct ScaleDataType<T, std::void_t<typename T::DataType>> { using DataType = typename T::DataType; };

    // Helper: apply D tensors with the same elementwise as CShuffleEpilogue
    template <typename DramWindows, typename COutTensor>
    CK_TILE_DEVICE void apply_d_tensors(DramWindows& d_dram_windows, COutTensor& c_out_tensor) const
    {
        if constexpr(NumDTensor == 0)
        {
            // nothing
        }
        else
        {
            const auto ds_tensor = generate_tuple(
                [&](auto idx) { return load_tile(d_dram_windows[idx]); }, number<NumDTensor>{});

            const auto c_ds_tiles = concat_tuple_of_reference(
                tie(c_out_tensor, c_out_tensor),
                generate_tie([&](auto idx) -> const auto& { return ds_tensor[idx]; },
                             number<NumDTensor>{}));

            tile_elementwise_inout_unpack(elfunc_, c_ds_tiles);
        }
    }

    template <typename OutDramWindow, typename COutTensor>
    CK_TILE_DEVICE void store_to_dram(OutDramWindow& out_dram_window,
                                      const COutTensor& c_out_tensor) const
    {
        if constexpr(decltype(out_dram_window.get_bottom_tensor_view())::DstInMemOp ==
                     memory_operation_enum::set)
        {
            store_tile(out_dram_window, c_out_tensor);
        }
        else
        {
            update_tile(out_dram_window, c_out_tensor);
        }
    }

    /**
     * operator():
     * - out_dram_window: output tile window
     * - o_acc_tile: accumulator tile from mainloop
     * - ds_dram_windows: tuple of D tensor windows (optional)
     * - (no p_smem): this epilogue is LDS-free by construction
     * - scale_m / scale_n: optional scales (scalar or window)
     */
    template <typename ODramWindow,
              typename OAccTile,
              typename DsDramWindows,
              typename ScaleM = EmptyScale,
              typename ScaleN = EmptyScale>
    CK_TILE_DEVICE auto operator()(ODramWindow& out_dram_window,
                                   const OAccTile& o_acc_tile,
                                   const DsDramWindows& ds_dram_windows,
                                   const ScaleM& scale_m = {},
                                   const ScaleN& scale_n = {})
    {
        // ---------------------------
        // Contract checks (tighten these as you learn the exact invariants)
        // ---------------------------
        static_assert(std::is_same_v<ELayout, tensor_layout::gemm::RowMajor>,
                      "PermuteNDirectStoreEpilogue currently supports RowMajor output only.");

        // If 2D variants set this differently, you want an early, explicit failure (or route to LDS epilogue).
        static_assert(BlockedXDLN_PerWarp == 1,
                      "PermuteNDirectStoreEpilogue requires BlockedXDLN_PerWarp == 1. "
                      "Otherwise use LDS CShuffle epilogue or generalize the mapping.");

        // Warp gemm dispatcher info (to get CWarpDstr/CWarpTensor)
        using WG = WarpGemmDispatcher<ATypeToUse,
                                      BTypeToUse,
                                      AccDataType,
                                      MPerXdl,
                                      NPerXdl,
                                      KPerXdl,
                                      isCTransposed>;
        using CWarpDstr   = typename WG::CWarpDstr;
        using CWarpTensor = typename WG::CWarpTensor;

        static constexpr int RowsPerLane = CWarpTensor::get_thread_buffer_size();
        static_assert(MPerXdl % RowsPerLane == 0,
                      "PermuteNDirectStoreEpilogue: MPerXdl must be divisible by RowsPerLane.");

        // Build an explicit (M,N) factorization for the per-thread distribution
        constexpr int kM0 = MWave;
        constexpr int kM2 = RowsPerLane;
        constexpr int kM1 = MPerXdl / kM2;

        constexpr int kN0 = NWave;
        constexpr int kN1 = NPerXdl;
        constexpr int kN2 = NRepeat;

        using IntrThreadShuffleEncode =
            tile_distribution_encoding<sequence<>,
                                       tuple<sequence<kM0, kM1, kM2>, sequence<kN0, kN1, kN2>>,
                                       tuple<sequence<1, 2>, sequence<1, 2>>,
                                       tuple<sequence<0, 0>, sequence<1, 1>>,
                                       sequence<1, 2>,
                                       sequence<2, 2>>;

        constexpr auto dram_tile_distribution =
            make_static_tile_distribution(IntrThreadShuffleEncode{});

        // D tensors windows follow the same per-thread distribution
        auto d_dram_windows = generate_tuple(
            [&](auto idx) { return make_tile_window(ds_dram_windows[idx], dram_tile_distribution); },
            number<NumDTensor>{});

        // Used to interpret the accumulator slice buffer
        constexpr auto c_warp_y_lengths =
            to_sequence(CWarpDstr{}.get_ys_to_d_descriptor().get_lengths());
        constexpr auto c_warp_y_index_zeros = uniform_sequence_gen_t<CWarpDstr::NDimY, 0>{};

        auto shuffle_acc  = make_static_distributed_tensor<AccDataType>(dram_tile_distribution);
        auto c_out_tensor = make_static_distributed_tensor<ODataType>(dram_tile_distribution);

        // Optional scales
        constexpr bool has_scales =
            !std::is_same_v<ScaleM, EmptyScale> && !std::is_same_v<ScaleN, EmptyScale>;
        constexpr bool has_scalar_scales =
            std::is_same_v<ScaleM, AccDataType> && std::is_same_v<ScaleN, AccDataType>;

        using SMType = typename ScaleDataType<ScaleM>::DataType;
        using SNType = typename ScaleDataType<ScaleN>::DataType;

        auto sm_tile = make_static_distributed_tensor<SMType>(dram_tile_distribution);
        auto sn_tile = make_static_distributed_tensor<SNType>(dram_tile_distribution);

        auto scale_m_window = [&]() {
            if constexpr(has_scales && !has_scalar_scales)
                return make_tile_window(scale_m, dram_tile_distribution);
            else
                return EmptyScale{};
        }();

        auto scale_n_window = [&]() {
            if constexpr(has_scales && !has_scalar_scales)
                return make_tile_window(scale_n, dram_tile_distribution);
            else
                return EmptyScale{};
        }();

        // NOTE: This is the “fast path”: slice -> small in-thread permute -> optional D -> store.
        static_for<0, MRepeat, 1>{}([&](auto mIter) {
            // Slice accumulators for this M repeat (includes all NRepeat)
            shuffle_acc.get_thread_buffer() = o_acc_tile.get_y_sliced_thread_data(
                merge_sequences(sequence<mIter, 0>{}, c_warp_y_index_zeros),
                merge_sequences(sequence<1, NRepeat>{}, c_warp_y_lengths));

            // Load windowed scales for this slice, if provided
            if constexpr(has_scales && !has_scalar_scales)
            {
                sm_tile = load_tile(scale_m_window);
                sn_tile = load_tile(scale_n_window);
            }

            // In-thread reorder:
            //   src assumes shuffle_acc is laid out as NRepeat "planes" of size plane=c_warp_y_lengths.product()
            //   dst writes out as [RowsPerLane, NRepeat] interleaving to match store distribution.
            static_for<0, NRepeat, 1>{}([&](auto n_idx) {
                const index_t plane = c_warp_y_lengths.product();

                static_for<0, kM2, 1>{}([&](auto m_lane) {
                    const int src = n_idx * plane + m_lane;
                    const int dst = n_idx + m_lane * NRepeat;

                    AccDataType v = shuffle_acc.get_thread_buffer()[src];

                    if constexpr(has_scalar_scales)
                    {
                        v = static_cast<AccDataType>(v * scale_m * scale_n);
                    }
                    else if constexpr(has_scales && !has_scalar_scales)
                    {
                        const auto sm = static_cast<float>(sm_tile.get_thread_buffer()[dst]);
                        const auto sn = static_cast<float>(sn_tile.get_thread_buffer()[dst]);
                        v             = static_cast<AccDataType>(v * sm * sn);
                    }

                    c_out_tensor.get_thread_buffer()[dst] = type_convert<ODataType>(v);
                });
            });

            // Apply D tensors (bias/residual/etc.) if present
            apply_d_tensors(d_dram_windows, c_out_tensor);

            // Store/update to DRAM
            store_to_dram(out_dram_window, c_out_tensor);

            // Advance output + D windows by one M slice
            move_tile_window(out_dram_window, {number<MPerIteration>{}, number<0>{}});

            static_for<0, NumDTensor, 1>{}([&](auto idx) {
                move_tile_window(d_dram_windows[idx], {number<MPerIteration>{}, number<0>{}});
            });

            // IMPORTANT: also advance windowed scales if present (otherwise mIter>0 reuses scales)
            if constexpr(has_scales && !has_scalar_scales)
            {
                move_tile_window(scale_m_window, {number<MPerIteration>{}, number<0>{}});
                move_tile_window(scale_n_window, {number<MPerIteration>{}, number<0>{}});
            }
        });
    }
};

} // namespace ck_tile
