// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2026, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include <ck_tile/core.hpp>

#include "block_gemm_areg_bsmem_creg_v2_hack_0.hpp"
#include "block_gemm_areg_bsmem_trload_creg_v2_hack_1.hpp"
#include "hstu_attention_pipeline_policy_helper.hpp"

namespace ck_tile {

// Tdm-only P@V block gemm (gfx1250). Same A/B/C distributions as
// BlockGemmARegBSmemTrLoadCRegV2Hack_1 (it derives from it), but the V ds_load_tr reads run ahead
// of the WMMAs in groups: the (kIter, nIter) steps are flattened to kSteps and split into groups
// of kStepsPerGroup steps. The B tiles of the first kPrefetchDepth steps are loaded up front; then
// every group first issues, back to back, the loads of the steps kPrefetchDepth ahead of its own
// and then runs its own WMMAs. In ATT traces taken on gfx1250 while tuning this, a DS load issued
// right after a WMMA waited about 30 cycles (a measured value, not a documented latency), so this
// replaces one WMMA->DS switch per step by one per group. The sched_group_barriers pin that order
// and the sched_barrier(0) closing each group keeps later loads from being hoisted into it. That
// bounds the DS data in flight to kPrefetchDepth + kStepsPerGroup steps only while the WMMAs stay
// in their group: the barriers act within a basic block, and a cross-attention M128 build (ROCm
// 7.2) was seen to sink the Q@K WMMAs past the mask branch, behind all of the loads.
template <typename Problem_, typename Policy_, index_t kPrefetchDepth_, index_t kStepsPerGroup_>
struct BlockGemmARegBSmemTrLoadCRegTdmGroupPrefetch_1
    : public BlockGemmARegBSmemTrLoadCRegV2Hack_1<Problem_, Policy_>
{
    using Base           = BlockGemmARegBSmemTrLoadCRegV2Hack_1<Problem_, Policy_>;
    using Problem        = remove_cvref_t<Problem_>;
    using Policy         = remove_cvref_t<Policy_>;
    using ADataType      = remove_cvref_t<typename Problem::ADataType>;
    using BDataType      = remove_cvref_t<typename Problem::BDataType>;
    using CDataType      = remove_cvref_t<typename Problem::CDataType>;
    using BlockGemmShape = remove_cvref_t<typename Problem::BlockGemmShape>;

    using Base::MakeABlockTileDistribution;
    using Base::MakeCBlockTile;

    static constexpr index_t kBlockSize = Problem::kBlockSize;

    // The run_ahead() of the base reads ahead with its own depth and would silently ignore
    // kPrefetchDepth_ and kStepsPerGroup_, so it is hidden here.
    template <index_t kPrefetchRequested, typename... Args>
    CK_TILE_DEVICE void run_ahead(Args&&...) const = delete;

    // C += A * B
    template <typename CBlockTensor, typename ABlockTensorTmp, typename BBlockWindowTmp>
    CK_TILE_DEVICE void operator()(CBlockTensor& c_block_tensor,
                                   const ABlockTensorTmp& a_block_tensor_tmp,
                                   const BBlockWindowTmp& b_block_window_tmp) const
    {
        static_assert(
            std::is_same_v<ADataType, remove_cv_t<typename ABlockTensorTmp::DataType>> &&
                std::is_same_v<BDataType, remove_cv_t<typename BBlockWindowTmp::DataType>> &&
                std::is_same_v<CDataType, remove_cv_t<typename CBlockTensor::DataType>>,
            "wrong!");

        constexpr index_t MPerBlock = ABlockTensorTmp{}.get_lengths()[number<0>{}];
        constexpr index_t NPerBlock = BBlockWindowTmp{}.get_window_lengths()[number<1>{}];
        constexpr index_t KPerBlock = ABlockTensorTmp{}.get_lengths()[number<1>{}];

        static_assert(MPerBlock == BlockGemmShape::kM && NPerBlock == BlockGemmShape::kN &&
                          KPerBlock == BlockGemmShape::kK,
                      "wrong!");

        constexpr auto config = Policy::template GetWarpGemmMWarpNWarp<Problem>();

        using WG = remove_cvref_t<decltype(config.template at<0>())>;

        constexpr index_t MWarp = config.template at<1>();
        constexpr index_t NWarp = config.template at<2>();

        constexpr index_t MIterPerWarp = MPerBlock / (MWarp * WG::kM);
        constexpr index_t NIterPerWarp = NPerBlock / (NWarp * WG::kN);
        constexpr index_t KIterPerWarp = KPerBlock / WG::kK;

        constexpr index_t NPerBlockPerIter = NPerBlock / NIterPerWarp;
        constexpr index_t KPerBlockPerIter = KPerBlock / KIterPerWarp;

        constexpr index_t kSteps         = KIterPerWarp * NIterPerWarp;
        constexpr index_t kPrefetchDepth = kPrefetchDepth_ < kSteps ? kPrefetchDepth_ : kSteps;
        constexpr index_t kStepsPerGroup = kStepsPerGroup_ < kSteps ? kStepsPerGroup_ : kSteps;

        static_assert(kPrefetchDepth >= 1 && kStepsPerGroup >= 1 && kSteps % kStepsPerGroup == 0,
                      "kStepsPerGroup must divide the number of (kIter, nIter) steps");

        const index_t iNWarp = get_warp_id<false>() % NWarp;

        constexpr auto c_block_outer_dstr_encoding = tile_distribution_encoding<
            sequence<>,
            tuple<sequence<MIterPerWarp, MWarp>, sequence<NIterPerWarp, NWarp>>,
            tuple<sequence<1, 2>>,
            tuple<sequence<1, 1>>,
            sequence<1, 2>,
            sequence<0, 0>>{};

        constexpr auto c_block_dstr_encode = detail::make_embed_tile_distribution_encoding(
            c_block_outer_dstr_encoding, typename WG::CWarpDstrEncoding{});

        auto a_block_tensor = make_static_distributed_tensor<typename ABlockTensorTmp::DataType>(
            MakeABlockTileDistribution());

        a_block_tensor.get_thread_buffer() = a_block_tensor_tmp.get_thread_buffer();

        constexpr auto b_warp_dstr_encode =
            typename InputTileDistributionTraits<typename WG::BWarpDstrEncoding,
                                                 BDataType>::TransposedDstrEncode{};

        auto b_warp_window_tmp = make_tile_window(
            b_block_window_tmp.get_bottom_tensor_view(),
            make_tuple(number<WG::kK>{}, number<WG::kN>{}),
            b_block_window_tmp.get_window_origin() + multi_index<2>{0, iNWarp * WG::kN},
            make_static_tile_distribution(b_warp_dstr_encode));

        static_assert(
            std::is_same_v<remove_cvref_t<decltype(c_block_dstr_encode)>,
                           remove_cvref_t<decltype(CBlockTensor::get_tile_distribution()
                                                       .get_static_tile_distribution_encoding())>>,
            "wrong!");

        using AWarpDstr = typename WG::AWarpDstr;
        using CWarpDstr = typename WG::CWarpDstr;

        using AWarpTensor = typename WG::AWarpTensor;
        using CWarpTensor = typename WG::CWarpTensor;

        constexpr auto a_warp_y_lengths =
            to_sequence(AWarpDstr{}.get_ys_to_d_descriptor().get_lengths());
        constexpr auto c_warp_y_lengths =
            to_sequence(CWarpDstr{}.get_ys_to_d_descriptor().get_lengths());

        constexpr auto a_warp_y_index_zeros = uniform_sequence_gen_t<AWarpDstr::NDimY, 0>{};
        constexpr auto c_warp_y_index_zeros = uniform_sequence_gen_t<CWarpDstr::NDimY, 0>{};

        using b_warp_tensor_type = decltype(load_tile_transpose(b_warp_window_tmp));

        // number of ds_load_tr instructions per B warp tile (b128 each)
        constexpr index_t kDsPerStep =
            (b_warp_tensor_type::get_thread_buffer_size() * sizeof(BDataType) + 15) / 16;

        statically_indexed_array<b_warp_tensor_type, kSteps> b_warp_tensors;

        // All steps read through the one step-0 window: a step's B tile is the step-0 tile shifted
        // by a compile-time element offset of the (affine, unswizzled) Lds layout, so every
        // ds_load_tr shares the step-0 lane address and carries the step in its immediate offset
        // field instead of a per-step address VALU.
        auto load_step = [&](auto i_step) {
            constexpr index_t kIter = i_step / NIterPerWarp;
            constexpr index_t nIter = i_step % NIterPerWarp;

            // the warp windows start at iNWarp * kN columns
            constexpr index_t kStepOffsetB =
                detail::GetLdsAffineStepOffset<typename BBlockWindowTmp::BottomTensorView,
                                               WG::kK,
                                               WG::kN,
                                               kIter * KPerBlockPerIter,
                                               nIter * NPerBlockPerIter,
                                               NWarp,
                                               0,
                                               WG::kN>();

            // as in run_ahead(): static_move_ys = true also folds the access offsets inside the
            // warp tile into the immediate, which needs a V layout that is linear over every warp
            // tile read here (the hdim128 V layout pads after every row, so it is)
            static_assert(
                detail::IsLdsLinearOverTile<typename BBlockWindowTmp::BottomTensorView,
                                            WG::kK,
                                            WG::kN,
                                            kIter * KPerBlockPerIter,
                                            nIter * NPerBlockPerIter,
                                            NWarp,
                                            0,
                                            WG::kN>(),
                "static_move_ys needs an Lds layout that is linear over the V warp tiles");

            load_tile_transpose_with_offset(b_warp_tensors(i_step),
                                            b_warp_window_tmp,
                                            kStepOffsetB,
                                            number<-1>{},
                                            bool_constant<true>{},
                                            bool_constant<true>{});
        };

        // prologue: the first kPrefetchDepth steps, back to back
        static_for<0, kPrefetchDepth, 1>{}([&](auto i_step) { load_step(i_step); });

        __builtin_amdgcn_sched_barrier(0);

        static_for<0, kSteps / kStepsPerGroup, 1>{}([&](auto i_group) {
            constexpr index_t kFirstLoad = i_group * kStepsPerGroup + kPrefetchDepth;
            constexpr index_t kNumLoads =
                kFirstLoad >= kSteps
                    ? 0
                    : (kSteps - kFirstLoad < kStepsPerGroup ? kSteps - kFirstLoad : kStepsPerGroup);

            // the loads kPrefetchDepth steps ahead first, back to back ...
            static_for<0, kNumLoads, 1>{}(
                [&](auto i_load) { load_step(number<kFirstLoad + i_load>{}); });

            // ... then the WMMAs of this group
            static_for<0, kStepsPerGroup, 1>{}([&](auto i_step_in_group) {
                constexpr index_t i_step = i_group * kStepsPerGroup + i_step_in_group;
                constexpr auto kIter     = number<i_step / NIterPerWarp>{};
                constexpr auto nIter     = number<i_step % NIterPerWarp>{};

                static_for<0, MIterPerWarp, 1>{}([&](auto mIter) {
                    AWarpTensor a_warp_tensor;

                    a_warp_tensor.get_thread_buffer() = a_block_tensor.get_y_sliced_thread_data(
                        merge_sequences(sequence<mIter, kIter>{}, a_warp_y_index_zeros),
                        merge_sequences(sequence<1, 1>{}, a_warp_y_lengths));

                    CWarpTensor c_warp_tensor;

                    c_warp_tensor.get_thread_buffer() = c_block_tensor.get_y_sliced_thread_data(
                        merge_sequences(sequence<mIter, nIter>{}, c_warp_y_index_zeros),
                        merge_sequences(sequence<1, 1>{}, c_warp_y_lengths));

                    WG{}(c_warp_tensor, a_warp_tensor, b_warp_tensors[number<i_step>{}]);

                    c_block_tensor.set_y_sliced_thread_data(
                        merge_sequences(sequence<mIter, nIter>{}, c_warp_y_index_zeros),
                        merge_sequences(sequence<1, 1>{}, c_warp_y_lengths),
                        c_warp_tensor.get_thread_buffer());
                });
            });

            // keep the order: the DS loads of the group, then its WMMAs
            if constexpr(kNumLoads > 0)
                __builtin_amdgcn_sched_group_barrier(0x100, kNumLoads * kDsPerStep, 0);
            __builtin_amdgcn_sched_group_barrier(0x008, kStepsPerGroup * MIterPerWarp, 0);

            __builtin_amdgcn_sched_barrier(0);
        });
    }

    // C = A * B
    template <typename ABlockTensorTmp, typename BBlockWindowTmp>
    CK_TILE_DEVICE auto operator()(const ABlockTensorTmp& a_block_tensor_tmp,
                                   const BBlockWindowTmp& b_block_window_tmp) const
    {
        auto c_block_tensor = MakeCBlockTile();
        operator()(c_block_tensor, a_block_tensor_tmp, b_block_window_tmp);
        return c_block_tensor;
    }
};

// Tdm-only Q@K block gemm (gfx1250), the K-side counterpart of the P@V gemm above: same
// distributions as BlockGemmARegBSmemCRegV2Hack_0 (it derives from it), the K tiles read
// kPrefetchDepth (kIter, nIter) steps ahead in groups of kStepsPerGroup steps, each group first
// issuing its loads and then running its WMMAs (same step loop and the same limits as the P@V gemm
// above).
template <typename Problem_, typename Policy_, index_t kPrefetchDepth_, index_t kStepsPerGroup_>
struct BlockGemmARegBSmemCRegTdmGroupPrefetch_0
    : public BlockGemmARegBSmemCRegV2Hack_0<Problem_, Policy_>
{
    using Base           = BlockGemmARegBSmemCRegV2Hack_0<Problem_, Policy_>;
    using Problem        = remove_cvref_t<Problem_>;
    using Policy         = remove_cvref_t<Policy_>;
    using ADataType      = remove_cvref_t<typename Problem::ADataType>;
    using BDataType      = remove_cvref_t<typename Problem::BDataType>;
    using CDataType      = remove_cvref_t<typename Problem::CDataType>;
    using BlockGemmShape = remove_cvref_t<typename Problem::BlockGemmShape>;

    using Base::MakeABlockTileDistribution;
    using Base::MakeCBlockTile;

    static constexpr index_t kBlockSize = Problem::kBlockSize;

    // The run_ahead() of the base reads ahead with its own depth and would silently ignore
    // kPrefetchDepth_ and kStepsPerGroup_, so it is hidden here.
    template <index_t kPrefetchRequested, typename... Args>
    CK_TILE_DEVICE void run_ahead(Args&&...) const = delete;

    // C = A * B, written into c_block_tensor (the first k step overwrites C)
    template <typename CBlockTensor, typename ABlockTensorTmp, typename BBlockWindowTmp>
    CK_TILE_DEVICE void operator()(CBlockTensor& c_block_tensor,
                                   const ABlockTensorTmp& a_block_tensor_tmp,
                                   const BBlockWindowTmp& b_block_window_tmp) const
    {
        static_assert(
            std::is_same_v<ADataType, remove_cv_t<typename ABlockTensorTmp::DataType>> &&
                std::is_same_v<BDataType, remove_cv_t<typename BBlockWindowTmp::DataType>> &&
                std::is_same_v<CDataType, remove_cv_t<typename CBlockTensor::DataType>>,
            "wrong!");

        constexpr index_t MPerBlock = ABlockTensorTmp{}.get_lengths()[number<0>{}];
        constexpr index_t NPerBlock = BBlockWindowTmp{}.get_window_lengths()[number<0>{}];
        constexpr index_t KPerBlock = ABlockTensorTmp{}.get_lengths()[number<1>{}];

        static_assert(MPerBlock == BlockGemmShape::kM && NPerBlock == BlockGemmShape::kN &&
                          KPerBlock == BlockGemmShape::kK,
                      "wrong!");

        constexpr auto config = Policy::template GetWarpGemmMWarpNWarp<Problem>();

        using WG = remove_cvref_t<decltype(config.template at<0>())>;

        constexpr index_t MWarp = config.template at<1>();
        constexpr index_t NWarp = config.template at<2>();

        constexpr index_t MIterPerWarp = MPerBlock / (MWarp * WG::kM);
        constexpr index_t NIterPerWarp = NPerBlock / (NWarp * WG::kN);
        constexpr index_t KIterPerWarp = KPerBlock / WG::kK;

        constexpr index_t NPerBlockPerIter = NPerBlock / NIterPerWarp;
        constexpr index_t KPerBlockPerIter = KPerBlock / KIterPerWarp;

        constexpr index_t kSteps         = KIterPerWarp * NIterPerWarp;
        constexpr index_t kPrefetchDepth = kPrefetchDepth_ < kSteps ? kPrefetchDepth_ : kSteps;
        constexpr index_t kStepsPerGroup = kStepsPerGroup_ < kSteps ? kStepsPerGroup_ : kSteps;

        static_assert(kPrefetchDepth >= 1 && kStepsPerGroup >= 1 && kSteps % kStepsPerGroup == 0,
                      "kStepsPerGroup must divide the number of (kIter, nIter) steps");

        const index_t iNWarp = get_warp_id<false>() % NWarp;

        static_assert(NWarp == 1, "Check failed!");

        constexpr auto c_block_outer_dstr_encoding =
            tile_distribution_encoding<sequence<NWarp>,
                                       tuple<sequence<MIterPerWarp, MWarp>, sequence<NIterPerWarp>>,
                                       tuple<sequence<1, 0>>,
                                       tuple<sequence<1, 0>>,
                                       sequence<1, 2>,
                                       sequence<0, 0>>{};

        constexpr auto c_block_dstr_encode = detail::make_embed_tile_distribution_encoding(
            c_block_outer_dstr_encoding, typename WG::CWarpDstrEncoding{});

        auto a_block_tensor = make_static_distributed_tensor<typename ABlockTensorTmp::DataType>(
            MakeABlockTileDistribution());

        a_block_tensor.get_thread_buffer() = a_block_tensor_tmp.get_thread_buffer();

        auto b_warp_window_tmp = make_tile_window(
            b_block_window_tmp.get_bottom_tensor_view(),
            make_tuple(number<WG::kN>{}, number<WG::kK>{}),
            b_block_window_tmp.get_window_origin() + multi_index<2>{iNWarp * WG::kN, 0},
            make_static_tile_distribution(typename WG::BWarpDstrEncoding{}));

        static_assert(
            std::is_same_v<remove_cvref_t<decltype(c_block_dstr_encode)>,
                           remove_cvref_t<decltype(CBlockTensor::get_tile_distribution()
                                                       .get_static_tile_distribution_encoding())>>,
            "wrong!");

        using AWarpDstr = typename WG::AWarpDstr;
        using CWarpDstr = typename WG::CWarpDstr;

        using AWarpTensor = typename WG::AWarpTensor;
        using CWarpTensor = typename WG::CWarpTensor;

        constexpr auto a_warp_y_lengths =
            to_sequence(AWarpDstr{}.get_ys_to_d_descriptor().get_lengths());
        constexpr auto c_warp_y_lengths =
            to_sequence(CWarpDstr{}.get_ys_to_d_descriptor().get_lengths());

        constexpr auto a_warp_y_index_zeros = uniform_sequence_gen_t<AWarpDstr::NDimY, 0>{};
        constexpr auto c_warp_y_index_zeros = uniform_sequence_gen_t<CWarpDstr::NDimY, 0>{};

        using b_warp_tensor_type = decltype(load_tile(b_warp_window_tmp));

        constexpr index_t kDsPerStep =
            (b_warp_tensor_type::get_thread_buffer_size() * sizeof(BDataType) + 15) / 16;

        statically_indexed_array<b_warp_tensor_type, kSteps> b_warp_tensors;

        // As in the P@V gemm above: one step-0 window, the step folded into the ds_load
        // immediate offset.
        auto load_step = [&](auto i_step) {
            constexpr index_t kIter = i_step / NIterPerWarp;
            constexpr index_t nIter = i_step % NIterPerWarp;

            constexpr index_t kStepOffsetB =
                detail::GetLdsAffineStepOffset<typename BBlockWindowTmp::BottomTensorView,
                                               WG::kN,
                                               WG::kK,
                                               nIter * NPerBlockPerIter,
                                               kIter * KPerBlockPerIter>();

            b_warp_tensors(i_step) = b_warp_window_tmp.load_with_offset(number<kStepOffsetB>{});
        };

        static_for<0, kPrefetchDepth, 1>{}([&](auto i_step) { load_step(i_step); });

        __builtin_amdgcn_sched_barrier(0);

        static_for<0, kSteps / kStepsPerGroup, 1>{}([&](auto i_group) {
            constexpr index_t kFirstLoad = i_group * kStepsPerGroup + kPrefetchDepth;
            constexpr index_t kNumLoads =
                kFirstLoad >= kSteps
                    ? 0
                    : (kSteps - kFirstLoad < kStepsPerGroup ? kSteps - kFirstLoad : kStepsPerGroup);

            static_for<0, kNumLoads, 1>{}(
                [&](auto i_load) { load_step(number<kFirstLoad + i_load>{}); });

            static_for<0, kStepsPerGroup, 1>{}([&](auto i_step_in_group) {
                constexpr index_t i_step = i_group * kStepsPerGroup + i_step_in_group;
                constexpr auto kIter     = number<i_step / NIterPerWarp>{};
                constexpr auto nIter     = number<i_step % NIterPerWarp>{};

                static_for<0, MIterPerWarp, 1>{}([&](auto mIter) {
                    AWarpTensor a_warp_tensor;

                    a_warp_tensor.get_thread_buffer() = a_block_tensor.get_y_sliced_thread_data(
                        merge_sequences(sequence<mIter, kIter>{}, a_warp_y_index_zeros),
                        merge_sequences(sequence<1, 1>{}, a_warp_y_lengths));

                    CWarpTensor c_warp_tensor;

                    if constexpr(kIter == 0)
                    {
                        c_warp_tensor = WG{}(a_warp_tensor, b_warp_tensors[number<i_step>{}]);
                    }
                    else
                    {
                        c_warp_tensor.get_thread_buffer() = c_block_tensor.get_y_sliced_thread_data(
                            merge_sequences(sequence<mIter, nIter>{}, c_warp_y_index_zeros),
                            merge_sequences(sequence<1, 1>{}, c_warp_y_lengths));

                        WG{}(c_warp_tensor, a_warp_tensor, b_warp_tensors[number<i_step>{}]);
                    }

                    c_block_tensor.set_y_sliced_thread_data(
                        merge_sequences(sequence<mIter, nIter>{}, c_warp_y_index_zeros),
                        merge_sequences(sequence<1, 1>{}, c_warp_y_lengths),
                        c_warp_tensor.get_thread_buffer());
                });
            });

            if constexpr(kNumLoads > 0)
                __builtin_amdgcn_sched_group_barrier(0x100, kNumLoads * kDsPerStep, 0);
            __builtin_amdgcn_sched_group_barrier(0x008, kStepsPerGroup * MIterPerWarp, 0);

            __builtin_amdgcn_sched_barrier(0);
        });
    }

    // C = A * B
    template <typename ABlockTensorTmp, typename BBlockWindowTmp>
    CK_TILE_DEVICE auto operator()(const ABlockTensorTmp& a_block_tensor_tmp,
                                   const BBlockWindowTmp& b_block_window_tmp) const
    {
        auto c_block_tensor = MakeCBlockTile();
        operator()(c_block_tensor, a_block_tensor_tmp, b_block_window_tmp);
        return c_block_tensor;
    }
};

// As BlockGemmARegBSmemCRegTdmGroupPrefetch_0, but the A operand (Q) is read from Lds too: per
// (kIter, nIter) step the B tile is loaded kPrefetchDepth steps ahead, and the MIterPerWarp A
// tiles of a kIter are loaded together with the B tile of that kIter's first step, i.e. in the
// same group of DS loads. Only the A tiles of the current and the next kIter are live (2 x
// MIterPerWarp x 8 VGPRs) instead of the whole Q tile.
template <typename Problem_, typename Policy_, index_t kPrefetchDepth_, index_t kStepsPerGroup_>
struct BlockGemmASmemBSmemCRegTdmGroupPrefetch_0
    : public BlockGemmARegBSmemCRegV2Hack_0<Problem_, Policy_>
{
    using Base           = BlockGemmARegBSmemCRegV2Hack_0<Problem_, Policy_>;
    using Problem        = remove_cvref_t<Problem_>;
    using Policy         = remove_cvref_t<Policy_>;
    using ADataType      = remove_cvref_t<typename Problem::ADataType>;
    using BDataType      = remove_cvref_t<typename Problem::BDataType>;
    using CDataType      = remove_cvref_t<typename Problem::CDataType>;
    using BlockGemmShape = remove_cvref_t<typename Problem::BlockGemmShape>;

    using Base::MakeABlockTileDistribution;
    using Base::MakeCBlockTile;

    static constexpr index_t kBlockSize = Problem::kBlockSize;

    // The run_ahead() of the base reads ahead with its own depth and would silently ignore
    // kPrefetchDepth_ and kStepsPerGroup_, so it is hidden here.
    template <index_t kPrefetchRequested, typename... Args>
    CK_TILE_DEVICE void run_ahead(Args&&...) const = delete;

    // C = A * B, written into c_block_tensor (the first k step overwrites C)
    template <typename CBlockTensor, typename ABlockWindowTmp, typename BBlockWindowTmp>
    CK_TILE_DEVICE void operator()(CBlockTensor& c_block_tensor,
                                   const ABlockWindowTmp& a_block_window_tmp,
                                   const BBlockWindowTmp& b_block_window_tmp) const
    {
        static_assert(
            std::is_same_v<ADataType, remove_cv_t<typename ABlockWindowTmp::DataType>> &&
                std::is_same_v<BDataType, remove_cv_t<typename BBlockWindowTmp::DataType>> &&
                std::is_same_v<CDataType, remove_cv_t<typename CBlockTensor::DataType>>,
            "wrong!");

        constexpr index_t MPerBlock = ABlockWindowTmp{}.get_window_lengths()[number<0>{}];
        constexpr index_t NPerBlock = BBlockWindowTmp{}.get_window_lengths()[number<0>{}];
        constexpr index_t KPerBlock = ABlockWindowTmp{}.get_window_lengths()[number<1>{}];

        static_assert(MPerBlock == BlockGemmShape::kM && NPerBlock == BlockGemmShape::kN &&
                          KPerBlock == BlockGemmShape::kK,
                      "wrong!");

        constexpr auto config = Policy::template GetWarpGemmMWarpNWarp<Problem>();

        using WG = remove_cvref_t<decltype(config.template at<0>())>;

        constexpr index_t MWarp = config.template at<1>();
        constexpr index_t NWarp = config.template at<2>();

        constexpr index_t MIterPerWarp = MPerBlock / (MWarp * WG::kM);
        constexpr index_t NIterPerWarp = NPerBlock / (NWarp * WG::kN);
        constexpr index_t KIterPerWarp = KPerBlock / WG::kK;

        constexpr index_t NPerBlockPerIter = NPerBlock / NIterPerWarp;
        constexpr index_t KPerBlockPerIter = KPerBlock / KIterPerWarp;

        constexpr index_t kSteps         = KIterPerWarp * NIterPerWarp;
        constexpr index_t kPrefetchDepth = kPrefetchDepth_ < kSteps ? kPrefetchDepth_ : kSteps;
        constexpr index_t kStepsPerGroup = kStepsPerGroup_ < kSteps ? kStepsPerGroup_ : kSteps;

        static_assert(kPrefetchDepth >= 1 && kStepsPerGroup >= 1 && kSteps % kStepsPerGroup == 0,
                      "kStepsPerGroup must divide the number of (kIter, nIter) steps");

        static_assert(NWarp == 1, "Check failed!");

        // the warp's rows of A, as in MakeABlockDistributionEncode(): mIter * MWarp * kM +
        // iMWarp * kM + [0, kM)
        const index_t iMWarp = get_warp_id<false>() / NWarp;

        constexpr auto c_block_outer_dstr_encoding =
            tile_distribution_encoding<sequence<NWarp>,
                                       tuple<sequence<MIterPerWarp, MWarp>, sequence<NIterPerWarp>>,
                                       tuple<sequence<1, 0>>,
                                       tuple<sequence<1, 0>>,
                                       sequence<1, 2>,
                                       sequence<0, 0>>{};

        constexpr auto c_block_dstr_encode = detail::make_embed_tile_distribution_encoding(
            c_block_outer_dstr_encoding, typename WG::CWarpDstrEncoding{});

        auto a_warp_window_tmp = make_tile_window(
            a_block_window_tmp.get_bottom_tensor_view(),
            make_tuple(number<WG::kM>{}, number<WG::kK>{}),
            a_block_window_tmp.get_window_origin() + multi_index<2>{iMWarp * WG::kM, 0},
            make_static_tile_distribution(typename WG::AWarpDstrEncoding{}));

        auto b_warp_window_tmp =
            make_tile_window(b_block_window_tmp.get_bottom_tensor_view(),
                             make_tuple(number<WG::kN>{}, number<WG::kK>{}),
                             b_block_window_tmp.get_window_origin(),
                             make_static_tile_distribution(typename WG::BWarpDstrEncoding{}));

        static_assert(
            std::is_same_v<remove_cvref_t<decltype(c_block_dstr_encode)>,
                           remove_cvref_t<decltype(CBlockTensor::get_tile_distribution()
                                                       .get_static_tile_distribution_encoding())>>,
            "wrong!");

        using CWarpDstr = typename WG::CWarpDstr;

        using CWarpTensor = typename WG::CWarpTensor;

        constexpr auto c_warp_y_lengths =
            to_sequence(CWarpDstr{}.get_ys_to_d_descriptor().get_lengths());

        constexpr auto c_warp_y_index_zeros = uniform_sequence_gen_t<CWarpDstr::NDimY, 0>{};

        using a_warp_tensor_type = decltype(a_warp_window_tmp.load_with_offset(number<0>{}));
        using b_warp_tensor_type = decltype(b_warp_window_tmp.load_with_offset(number<0>{}));

        constexpr index_t kDsPerStepA =
            (a_warp_tensor_type::get_thread_buffer_size() * sizeof(ADataType) + 15) / 16;
        constexpr index_t kDsPerStepB =
            (b_warp_tensor_type::get_thread_buffer_size() * sizeof(BDataType) + 15) / 16;

        statically_indexed_array<a_warp_tensor_type, KIterPerWarp * MIterPerWarp> a_warp_tensors;
        statically_indexed_array<b_warp_tensor_type, kSteps> b_warp_tensors;

        // number of ds_load issued by load_step(i_step)
        constexpr auto ds_per_step = [](index_t i_step) constexpr {
            return kDsPerStepB + (i_step % NIterPerWarp == 0 ? MIterPerWarp * kDsPerStepA : 0);
        };

        auto load_step = [&](auto i_step) {
            constexpr index_t kIter = i_step / NIterPerWarp;
            constexpr index_t nIter = i_step % NIterPerWarp;

            if constexpr(nIter == 0)
            {
                static_for<0, MIterPerWarp, 1>{}([&](auto mIter) {
                    // the A windows of the M warps start at iMWarp * kM rows: the offset is
                    // checked from each of them
                    constexpr index_t kStepOffsetA =
                        detail::GetLdsAffineStepOffset<typename ABlockWindowTmp::BottomTensorView,
                                                       WG::kM,
                                                       WG::kK,
                                                       mIter * MWarp * WG::kM,
                                                       kIter * KPerBlockPerIter,
                                                       MWarp,
                                                       WG::kM,
                                                       0>();

                    a_warp_tensors(number<kIter * MIterPerWarp + mIter>{}) =
                        a_warp_window_tmp.load_with_offset(number<kStepOffsetA>{});
                });
            }

            constexpr index_t kStepOffsetB =
                detail::GetLdsAffineStepOffset<typename BBlockWindowTmp::BottomTensorView,
                                               WG::kN,
                                               WG::kK,
                                               nIter * NPerBlockPerIter,
                                               kIter * KPerBlockPerIter>();

            b_warp_tensors(i_step) = b_warp_window_tmp.load_with_offset(number<kStepOffsetB>{});
        };

        static_for<0, kPrefetchDepth, 1>{}([&](auto i_step) { load_step(i_step); });

        __builtin_amdgcn_sched_barrier(0);

        static_for<0, kSteps / kStepsPerGroup, 1>{}([&](auto i_group) {
            constexpr index_t kFirstLoad = i_group * kStepsPerGroup + kPrefetchDepth;
            constexpr index_t kNumLoads =
                kFirstLoad >= kSteps
                    ? 0
                    : (kSteps - kFirstLoad < kStepsPerGroup ? kSteps - kFirstLoad : kStepsPerGroup);

            constexpr index_t kNumDs = [=]() constexpr {
                index_t num_ds = 0;
                for(index_t i_load = 0; i_load < kNumLoads; ++i_load)
                    num_ds += ds_per_step(kFirstLoad + i_load);
                return num_ds;
            }();

            static_for<0, kNumLoads, 1>{}(
                [&](auto i_load) { load_step(number<kFirstLoad + i_load>{}); });

            static_for<0, kStepsPerGroup, 1>{}([&](auto i_step_in_group) {
                constexpr index_t i_step = i_group * kStepsPerGroup + i_step_in_group;
                constexpr auto kIter     = number<i_step / NIterPerWarp>{};
                constexpr auto nIter     = number<i_step % NIterPerWarp>{};

                static_for<0, MIterPerWarp, 1>{}([&](auto mIter) {
                    const auto& a_warp_tensor =
                        a_warp_tensors[number<kIter * MIterPerWarp + mIter>{}];

                    CWarpTensor c_warp_tensor;

                    if constexpr(kIter == 0)
                    {
                        c_warp_tensor = WG{}(a_warp_tensor, b_warp_tensors[number<i_step>{}]);
                    }
                    else
                    {
                        c_warp_tensor.get_thread_buffer() = c_block_tensor.get_y_sliced_thread_data(
                            merge_sequences(sequence<mIter, nIter>{}, c_warp_y_index_zeros),
                            merge_sequences(sequence<1, 1>{}, c_warp_y_lengths));

                        WG{}(c_warp_tensor, a_warp_tensor, b_warp_tensors[number<i_step>{}]);
                    }

                    c_block_tensor.set_y_sliced_thread_data(
                        merge_sequences(sequence<mIter, nIter>{}, c_warp_y_index_zeros),
                        merge_sequences(sequence<1, 1>{}, c_warp_y_lengths),
                        c_warp_tensor.get_thread_buffer());
                });
            });

            if constexpr(kNumLoads > 0)
                __builtin_amdgcn_sched_group_barrier(0x100, kNumDs, 0);
            __builtin_amdgcn_sched_group_barrier(0x008, kStepsPerGroup * MIterPerWarp, 0);

            __builtin_amdgcn_sched_barrier(0);
        });
    }

    // C = A * B
    template <typename ABlockWindowTmp, typename BBlockWindowTmp>
    CK_TILE_DEVICE auto operator()(const ABlockWindowTmp& a_block_window_tmp,
                                   const BBlockWindowTmp& b_block_window_tmp) const
    {
        auto c_block_tensor = MakeCBlockTile();
        operator()(c_block_tensor, a_block_window_tmp, b_block_window_tmp);
        return c_block_tensor;
    }
};

} // namespace ck_tile
