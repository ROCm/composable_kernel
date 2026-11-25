// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
#pragma once

#include <type_traits>
#include <utility>

#include "ck_tile/core.hpp"
#include "ck_tile/ops/gemm/block/block_gemm_areg_breg_creg_v1.hpp"
#include "ck_tile/ops/gemm/block/block_gemm_mx_tdm_scheduled_codegen.hpp"

namespace ck_tile {

/// The block GEMM of MxGemmPipelineAgBgCrCompTDMScheduled: BlockGemmARegBRegCRegV1 (PackMNIter,
/// one K step per sub-tile) on A and B tiles it reads itself from the pipeline's LDS buffers (A
/// and B in shared memory, C in registers).
///
/// The pipeline calls it per K step: LocalPrefetchA and LocalPrefetchB read a K step's A
/// fragments and scales and a range of its B fragments, and Run issues a range of the WMMAs of a
/// range of B fragments, as V1's hot loop does: half B issues each half's WMMAs after that half's
/// reads.
///
/// With PackMNIter, V1's M (N) index is (warp, repeat, atom row): the fragment order of the
/// policy's MakeLdsBlockDescriptor. The reads go through word windows whose distribution is
/// the word form of V1's warp encoding (one access per 16-byte chunk). The A and B tiles and the
/// accumulators keep each repeat's fragment as one vector of its warp tensor's words; element-wise
/// copies into byte thread buffers change the register allocation of the loop.
///
/// Policy is MxGemmPipelineAgBgCrCompTDMScheduledPolicy: the LDS layout, the read distributions
/// and the C distribution are its. V1Policy is the BlockGemmARegBRegCRegV1CustomPolicy its
/// GetBlockGemm builds.
template <typename Problem_, typename Policy_, typename V1Policy_>
struct BlockGemmMxAsmemBsmemCRegTDMScheduled
    : BlockGemmARegBRegCRegV1<remove_cvref_t<Problem_>, remove_cvref_t<V1Policy_>>
{
    using Base      = BlockGemmARegBRegCRegV1<remove_cvref_t<Problem_>, remove_cvref_t<V1Policy_>>;
    using Problem   = remove_cvref_t<Problem_>;
    using Policy    = remove_cvref_t<Policy_>;
    using WarpGemm  = typename Base::WarpGemm;
    using ADataType = typename Base::ADataType;
    using BDataType = typename Base::BDataType;

    static constexpr index_t kRm        = Base::MIterPerWarp;
    static constexpr index_t kRn        = Base::NIterPerWarp;
    static constexpr index_t kStepBytes = Policy::template GetStepBytes<Problem>();
    static_assert(Base::PackMNIter && Base::KPerSubTile == 1 && !Base::UseDefaultScheduler,
                  "comp_tdm_scheduled: V1 with PackMNIter, one K step per sub-tile");

    /// The accumulators: V1's C block tile, in fragment order.
    using AccTile = decltype(Base::MakeCBlockTile());
    /// The C block tile of the epilogue: the accumulators' thread buffer, labelled with tile rows
    /// and columns.
    using CBlockTile = decltype(make_static_distributed_tensor<float>(
        Policy::template MakeCBlockTileDistribution<Problem>()));
    static_assert(std::is_same_v<remove_cvref_t<decltype(AccTile{}.get_thread_buffer())>,
                                 remove_cvref_t<decltype(CBlockTile{}.get_thread_buffer())>>,
                  "comp_tdm_scheduled: the C block tile holds the C fragments of every repeat");

    /// Wave 0's window on A's or B's fragment of one K step at repeat 0 of buffer 0. The wave
    /// stays out of the window's per-lane coordinate so that its offset remains scalar.
    template <bool IsA>
    CK_TILE_DEVICE static auto MakeLdsWindow(CK_TILE_LDS_ADDR uint8_t* smem)
    {
        constexpr index_t rows = Policy::template GetFragmentHalves<Problem, IsA>() * 16;
        const auto view        = make_tensor_view<address_space_enum::lds>(
            c_style_pointer_cast<int32_t*>(smem),
            Policy::template MakeLdsBlockDescriptor<Problem, IsA>());
        return make_tile_window(view,
                                make_tuple(number<rows>{}, number<kStepBytes / 4>{}),
                                make_multi_index(0, 0),
                                Policy::template MakeLdsReadDistribution<Problem, IsA>());
    }
    using ALdsWindow = decltype(MakeLdsWindow<true>(nullptr));
    using BLdsWindow = decltype(MakeLdsWindow<false>(nullptr));

    /// A wave's window on A's or B's scales of K step 0 in buffer 0.
    template <bool IsA>
    CK_TILE_DEVICE static auto MakeLdsScaleWindow(CK_TILE_LDS_ADDR uint8_t* smem, index_t wave)
    {
        constexpr index_t atom    = Policy::template GetAtom<Problem, IsA>();
        constexpr index_t repeats = IsA ? kRm : kRn;
        constexpr index_t lanes   = Policy::template GetScaleLanes<Problem, IsA>();
        const auto view           = make_tensor_view<address_space_enum::lds>(
            c_style_pointer_cast<int32_t*>(smem),
            Policy::template MakeLdsScaleDescriptor<Problem, IsA>());
        return make_tile_window(view,
                                make_tuple(number<1>{}, number<lanes>{}, number<repeats>{}),
                                make_multi_index(0, wave * atom, 0),
                                Policy::template MakeLdsScaleReadDistribution<Problem, IsA>());
    }
    using AScaleWindow = decltype(MakeLdsScaleWindow<true>(nullptr, 0));
    using BScaleWindow = decltype(MakeLdsScaleWindow<false>(nullptr, 0));
    /// A K step's per-lane E8M0 words of A's or B's repeats: element r is repeat r's.
    using AScaleTile = decltype(std::declval<const AScaleWindow&>().load_with_offset(0));
    using BScaleTile = decltype(std::declval<const BScaleWindow&>().load_with_offset(0));

    /// One repeat's A or B fragment of a K step, as the words of V1's warp tensor.
    using AFrag = ext_vector_t<int32_t, sizeof(typename WarpGemm::AWarpTensor) / 4>;
    using BFrag = ext_vector_t<int32_t, sizeof(typename WarpGemm::BWarpTensor) / 4>;
    /// One WMMA's accumulators: slot mIter * kRn + nIter of the accumulators' thread buffer.
    using CVec = ext_vector_t<float, WarpGemm::CWarpTensor::get_thread_buffer_size()>;
    static_assert(AccTile::get_thread_buffer_size() == kRm * kRn * sizeof(CVec) / sizeof(float),
                  "comp_tdm_scheduled: the accumulators are one C fragment per WMMA");
    /// A K step's A fragments and the A and B scales.
    struct ATile
    {
        AFrag a[kRm];
        AScaleTile as;
        BScaleTile bs;
    };
    /// A K step's B fragments: all of them, or two halves with half-B.
    struct BTile
    {
        BFrag b[kRn];
    };

    CK_TILE_DEVICE constexpr BlockGemmMxAsmemBsmemCRegTDMScheduled() = default;

    /// `smem` is the pipeline's LDS; the wave's M and N position are the pipeline's.
    CK_TILE_DEVICE
    BlockGemmMxAsmemBsmemCRegTDMScheduled(CK_TILE_LDS_ADDR uint8_t* smem_,
                                          index_t wave_m_,
                                          index_t wave_n_)
        : wave_m(wave_m_),
          wave_n(wave_n_),
          a_window(MakeLdsWindow<true>(smem_)),
          b_window(MakeLdsWindow<false>(smem_)),
          as_window(MakeLdsScaleWindow<true>(smem_, wave_m_)),
          bs_window(MakeLdsScaleWindow<false>(smem_, wave_n_))
    {
        static_for<0, 2, 1>{}([&](auto parity) {
            const index_t buffer = parity * kBufferStride;
            origins[0][parity]   = detail::opaque_scalar(FragmentOrigin<true>(buffer));
            origins[1][parity]   = detail::opaque_scalar(FragmentOrigin<false>(buffer));
            origins[2][parity]   = detail::opaque_scalar(ScaleOrigin<true>(buffer));
            origins[3][parity]   = detail::opaque_scalar(ScaleOrigin<false>(buffer));
        });
    }

    /// A K step's scales and A fragments in LDS buffer `buffer` (a byte offset).
    CK_TILE_DEVICE ATile LocalPrefetchA(index_t buffer, index_t step) const
    {
        ATile h;
        h.as = LdsScales<true>(buffer, step);
        h.bs = LdsScales<false>(buffer, step);
        static_for<0, kRm, 1>{}([&](auto rm) { h.a[rm] = LdsFragment<true>(buffer, rm, step); });
        return h;
    }

    /// B fragments First .. First + Count - 1 of a K step into b's repeats First ...
    template <index_t First, index_t Count>
    CK_TILE_DEVICE void LocalPrefetchB(BTile& b, index_t buffer, index_t step) const
    {
        static_for<First, First + Count, 1>{}(
            [&](auto rn) { b.b[rn] = LdsFragment<false>(buffer, rn, step); });
    }

    /// WMMAs [IssueBegin, IssueEnd) of the kRm x NCount WMMAs of B fragments [NFirst, NFirst +
    /// NCount): half-B order (rn outer, rm inner) with NOuter, else rm outer. The body is V1's
    /// scaled hot loop with the K step's scales.
    template <index_t NFirst, index_t NCount, bool NOuter, index_t IssueBegin, index_t IssueEnd>
    CK_TILE_DEVICE void Run(AccTile& c, const ATile& a, const BTile& b) const
    {
        static_assert(0 <= IssueBegin && IssueBegin <= IssueEnd && IssueEnd <= kRm * NCount,
                      "comp_tdm_scheduled: the WMMA range is within the B fragments' WMMAs");
        using AWarpTensor = typename WarpGemm::AWarpTensor;
        using BWarpTensor = typename WarpGemm::BWarpTensor;
        using CWarpTensor = typename WarpGemm::CWarpTensor;

        constexpr index_t AScaleTypeVal =
            ScaleDataTypeToEnum<typename Problem::AScaleDataType>::value;
        constexpr index_t BScaleTypeVal =
            ScaleDataTypeToEnum<typename Problem::BScaleDataType>::value;

        static_for<IssueBegin, IssueEnd, 1>{}([&](auto i) {
            constexpr index_t issued = decltype(i)::value;
            constexpr index_t mIter  = NOuter ? issued % kRm : issued / NCount;
            constexpr index_t nIter  = NFirst + (NOuter ? issued / kRm : issued % NCount);

            AWarpTensor a_warp_tensor;
            a_warp_tensor.get_thread_buffer().template set_as<AFrag>(number<0>{}, a.a[mIter]);
            BWarpTensor b_warp_tensor;
            b_warp_tensor.get_thread_buffer().template set_as<BFrag>(number<0>{}, b.b[nIter]);
            auto& c_vec =
                c.get_thread_buffer().template get_as<CVec>()(number<mIter * kRn + nIter>{});
            CWarpTensor c_warp_tensor;
            c_warp_tensor.get_thread_buffer().template set_as<CVec>(number<0>{}, c_vec);

            WarpGemm{}
                .template operator()<AScaleDataType<AScaleTypeVal>, BScaleDataType<BScaleTypeVal>>(
                    c_warp_tensor,
                    a_warp_tensor,
                    b_warp_tensor,
                    a.as.get_thread_buffer()[number<mIter>{}],
                    a.bs.get_thread_buffer()[number<nIter>{}]);

            c_vec = c_warp_tensor.get_thread_buffer().template get_as<CVec>()[number<0>{}];
        });
    }

    /// The accumulators as the epilogue's C block tile: the same thread buffer, with V1's
    /// fragment-order rows and columns relabelled as tile rows and columns.
    CK_TILE_DEVICE static CBlockTile ToCBlockTile(const AccTile& acc)
    {
        CBlockTile c;
        c.get_thread_buffer() = acc.get_thread_buffer();
        return c;
    }

    private:
    static constexpr index_t kBufferStride = Policy::template GetLdsBufferStride<Problem>();

    /// The word offset, from a linear read window's, of the wave's repeat-0, K-step-0 fragment
    /// of A or B in LDS buffer `buffer` (a byte offset).
    template <bool IsA>
    CK_TILE_DEVICE index_t FragmentOrigin(index_t buffer) const
    {
        constexpr index_t atom    = Policy::template GetAtom<Problem, IsA>();
        constexpr index_t repeats = IsA ? kRm : kRn;
        constexpr index_t base    = Policy::template GetLdsOperand<Problem, IsA>().base / 4;
        constexpr index_t wave_words =
            Policy::template GetLdsRowOffset<Problem, IsA>(repeats * atom);
        return buffer / 4 + base + (IsA ? wave_m : wave_n) * wave_words;
    }

    /// The word offset, from a linear scale window's, of K step 0's scales of A or B in LDS
    /// buffer `buffer`.
    template <bool IsA>
    CK_TILE_DEVICE static index_t ScaleOrigin(index_t buffer)
    {
        return buffer / 4 + Policy::template GetLdsOperand<Problem, IsA, true>().base / 4;
    }

    /// origins[i][parity] for the LDS buffer at byte offset `buffer`.
    CK_TILE_DEVICE index_t Origin(index_t i, index_t buffer) const
    {
        return buffer == 0 ? origins[i][0] : origins[i][1];
    }

    /// A's or B's fragment of repeat r at K step `step`, as words. With linear reads the wave,
    /// buffer, repeat and step are a word offset from the window's; otherwise the window moves
    /// to the fragment's first row and the buffer and step are the offset.
    template <bool IsA>
    CK_TILE_DEVICE auto LdsFragment(index_t buffer, index_t r, index_t step) const
    {
        constexpr index_t atom    = Policy::template GetAtom<Problem, IsA>();
        constexpr index_t repeats = IsA ? kRm : kRn;
        constexpr index_t base    = Policy::template GetLdsOperand<Problem, IsA>().base / 4;
        const index_t wave_index  = IsA ? wave_m : wave_n;
        const auto& window        = [&]() -> const auto& {
            if constexpr(IsA)
                return a_window;
            else
                return b_window;
        }();
        const auto tile = [&] {
            if constexpr(Policy::template IsLdsReadLinear<Problem, IsA>())
            {
                constexpr auto repeat_offsets = [] {
                    array<index_t, repeats> offsets{};
                    for(index_t i = 0; i < repeats; ++i)
                        offsets[i] = Policy::template GetLdsRowOffset<Problem, IsA>(i * atom);
                    return offsets;
                }();
                return window.load_with_offset(Origin(IsA ? 0 : 1, buffer) + repeat_offsets[r] +
                                                   step * (kStepBytes / 4),
                                               number<-1>{},
                                               bool_constant<false>{});
            }
            else
            {
                auto moved = window;
                moved.set_window_origin(make_multi_index((wave_index * repeats + r) * atom, 0));
                return moved.load_with_offset(buffer / 4 + base + step * (kStepBytes / 4),
                                              number<-1>{},
                                              bool_constant<false>{});
            }
        }();
        using Vec = ext_vector_t<int32_t, decltype(tile)::get_thread_buffer_size()>;
        return tile.get_thread_buffer().template get_as<Vec>()[number<0>{}];
    }

    /// A's or B's per-lane E8M0 words of K step `step`. With linear reads the step is a word
    /// offset from the window's; otherwise the window moves to the step.
    template <bool IsA>
    CK_TILE_DEVICE auto LdsScales(index_t buffer, index_t step) const
    {
        constexpr auto op  = Policy::template GetLdsOperand<Problem, IsA, true>();
        const auto& window = [&]() -> const auto& {
            if constexpr(IsA)
                return as_window;
            else
                return bs_window;
        }();
        if constexpr(Policy::template IsLdsScaleReadLinear<Problem, IsA>())
        {
            constexpr index_t step_words =
                Policy::template MakeLdsScaleDescriptor<Problem, IsA>().calculate_offset(
                    make_multi_index(1, 0, 0));
            return window.load_with_offset(Origin(IsA ? 2 : 3, buffer) + step * step_words,
                                           number<-1>{},
                                           bool_constant<false>{});
        }
        else
        {
            constexpr index_t atom = Policy::template GetAtom<Problem, IsA>();
            auto moved             = window;
            moved.set_window_origin(make_multi_index(step, (IsA ? wave_m : wave_n) * atom, 0));
            return moved.load_with_offset(
                buffer / 4 + op.base / 4, number<-1>{}, bool_constant<false>{});
        }
    }

    index_t wave_m;
    index_t wave_n;
    /// Word origins of the linear reads, [A, B, A scales, B scales][buffer parity], from
    /// FragmentOrigin and ScaleOrigin. Opaque, so that each operand and buffer keeps one address
    /// VGPR and the reads' remaining offsets fit the 16-bit DS immediate; folded into each
    /// read's constant, the offsets pass 64 KiB and every read holds its own address VGPR.
    index_t origins[4][2];
    ALdsWindow a_window;
    BLdsWindow b_window;
    AScaleWindow as_window;
    BScaleWindow bs_window;
};

} // namespace ck_tile
