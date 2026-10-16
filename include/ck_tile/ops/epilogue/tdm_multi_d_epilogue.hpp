// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/ops/common/tensor_layout.hpp"
#include "ck_tile/ops/epilogue/tdm_epilogue.hpp"

namespace ck_tile {

// TDM epilogue with fused D tensors (E = CDElementwise(C, D0, D1, ...)).
//
// This is a separate type from TdmEpilogue on purpose: TdmEpilogue keeps rejecting D tensors so
// that no existing TdmEpilogue user (grouped convolution, mx flatmm, ...) silently starts to
// accept Ds. The LDS layout, the LDS size and the final TDM store are the same as TdmEpilogue;
// only the register stage differs: every D tile is loaded with the accumulator distribution and
// fused on the fp32 accumulator before the single cast to ODataType.
//
// Restrictions:
//  - NumDTensor > 0 (use TdmEpilogue for plain GEMM),
//  - row-major E and row-major Ds,
//  - no ScaleM/ScaleN,
//  - output window must use memory_operation_enum::set (no split-K accumulation). The TDM store
//    always overwrites, so any other memory operation traps at run time. A trap is used instead
//    of a static_assert because the kernel instantiates the atomic-add epilogue branch for
//    non-fp16/bf16 E even though GemmKernelMultiD rejects k_batch > 1 on the host.
//
// The Problem is reused from CShuffleEpilogueProblem.
template <typename Problem_>
struct TdmMultiDEpilogue : public TdmEpilogue<Problem_>
{
    using Base          = TdmEpilogue<Problem_>;
    using Problem       = typename Base::Problem;
    using AccDataType   = typename Base::AccDataType;
    using ODataType     = typename Base::ODataType;
    using DsDataType    = typename Base::DsDataType;
    using DsLayout      = typename Base::DsLayout;
    using ELayout       = typename Base::ELayout;
    using CDElementwise = typename Base::CDElementwise;
    using EmptyScale    = typename Base::EmptyScale;

    static constexpr index_t kBlockSize = Base::kBlockSize;
    static constexpr index_t kMPerBlock = Base::kMPerBlock;
    static constexpr index_t kNPerBlock = Base::kNPerBlock;
    static constexpr index_t NumDTensor = Base::NumDTensor;

    // Marker used by kernels to check that a TDM pipeline with D tensors is paired with this
    // epilogue.
    static constexpr bool kIsTdmMultiDEpilogue = true;

    static_assert(NumDTensor > 0, "TdmMultiDEpilogue requires at least one D tensor");
    static_assert(NumDTensor == DsLayout::size() && NumDTensor == DsDataType::size(),
                  "DsLayout and DsDataType must have NumDTensor elements");
    static_assert(std::is_same_v<ELayout, tensor_layout::gemm::RowMajor>,
                  "TdmMultiDEpilogue supports only row-major E");

    CK_TILE_HOST_DEVICE static constexpr bool AllDsRowMajor()
    {
        bool all_row = true;
        static_for<0, NumDTensor, 1>{}([&](auto i) {
            using DiLayout = remove_cvref_t<std::tuple_element_t<i.value, DsLayout>>;
            all_row        = all_row && std::is_same_v<DiLayout, tensor_layout::gemm::RowMajor>;
        });
        return all_row;
    }
    static_assert(AllDsRowMajor(), "TdmMultiDEpilogue supports only row-major D tensors");

    CK_TILE_HOST_DEVICE static constexpr index_t GetSmemSize() { return Base::GetSmemSize(); }

    template <typename ODramWindow,
              typename OAccTile,
              typename DsDramWindows,
              typename ScaleM = EmptyScale,
              typename ScaleN = EmptyScale>
    CK_TILE_DEVICE auto operator()(ODramWindow& out_dram_window,
                                   const OAccTile& o_acc_tile,
                                   const DsDramWindows& ds_dram_windows,
                                   void* p_smem,
                                   const ScaleM& scale_m = {},
                                   const ScaleN& scale_n = {})
    {
        ignore = scale_m;
        ignore = scale_n;
        static_assert(std::is_same_v<ScaleM, EmptyScale> && std::is_same_v<ScaleN, EmptyScale>,
                      "ScaleM and ScaleN must be EmptyScale with TdmMultiDEpilogue");
        static_assert(kBlockSize % get_warp_size() == 0, "BlockSize must be multiple of WarpSize");

        constexpr auto kOutMemOp =
            remove_cvref_t<decltype(out_dram_window.get_bottom_tensor_view())>::DstInMemOp;

        if constexpr(kOutMemOp != memory_operation_enum::set)
        {
            // The TDM store cannot accumulate; reaching this branch means a split-K launch that
            // the host side should have rejected.
            ignore = out_dram_window;
            ignore = o_acc_tile;
            ignore = ds_dram_windows;
            ignore = p_smem;
            __builtin_trap();
        }
        else
        {
            // Same LDS layout and TDM store distribution as TdmEpilogue (E is row-major).
            auto o_lds_block =
                impl::tdm_row_major_lds_view<ODataType, kMPerBlock, kNPerBlock>(p_smem);

            const auto acc_dstr = o_acc_tile.get_tile_distribution();

            auto in_lds_window =
                make_tile_window(o_lds_block,
                                 make_tuple(number<kMPerBlock>{}, number<kNPerBlock>{}),
                                 {0, 0},
                                 acc_dstr);

            auto out_lds_window =
                make_tile_window(o_lds_block,
                                 make_tuple(number<kMPerBlock>{}, number<kNPerBlock>{}),
                                 {0, 0},
                                 impl::tdm_wave_linear_distr<kBlockSize, kMPerBlock, kNPerBlock>());

            TDMConfig tdm_config;

            // Wait for all outstanding TDM loads of the main loop before LDS is reused.
            s_wait_tensorcnt_barrier<0 /*tensor_cnt*/, 0 /*lgkmcnt*/>();

            // D tiles are loaded after the barrier to keep their live ranges short.
            const auto d_tiles = generate_tuple(
                [&](auto i) { return load_tile(make_tile_window(ds_dram_windows[i], acc_dstr)); },
                number<NumDTensor>{});

            auto e_tile = make_static_distributed_tensor<ODataType>(acc_dstr);

            tile_elementwise_inout_unpack(
                CDElementwise{},
                concat_tuple_of_reference(
                    tie(e_tile, o_acc_tile),
                    generate_tie([&](auto i) -> const auto& { return d_tiles[i]; },
                                 number<NumDTensor>{})));

            store_tile(in_lds_window, e_tile);
            block_sync_lds();

            store_tile_tdm(tdm_config, out_dram_window, out_lds_window);
        }
    }
};

} // namespace ck_tile
