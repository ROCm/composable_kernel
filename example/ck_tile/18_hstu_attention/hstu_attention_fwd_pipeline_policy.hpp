// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2025, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include <ck_tile/core.hpp>
#include <ck_tile/ops/gemm/block/block_gemm_problem.hpp>
#include <ck_tile/ops/gemm/pipeline/tile_gemm_shape.hpp>
#include <ck_tile/ops/gemm/warp/warp_gemm_dispatcher.hpp>

#include <ck_tile/ops/gemm/block/block_gemm_areg_bsmem_creg_v2_custom_policy.hpp>
#include <ck_tile/ops/gemm/block/block_gemm_areg_bsmem_creg_one_warp_v1.hpp>

#include "block_gemm_areg_bsmem_creg_v2_hack_0.hpp"
#include "block_gemm_areg_bsmem_creg_v2_hack_1.hpp"
#include "block_gemm_areg_bsmem_trload_creg_v2_hack_1.hpp"

#include "hstu_attention_config.hpp"
#include "hstu_attention_kernel_util.hpp"
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
            constexpr index_t kOffset =
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
                                            kOffset,
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
            static_for<0, kNumLoads, 1>{}([&](auto j) { load_step(number<kFirstLoad + j>{}); });

            // ... then the WMMAs of this group
            static_for<0, kStepsPerGroup, 1>{}([&](auto j) {
                constexpr index_t i_step = i_group * kStepsPerGroup + j;
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

            constexpr index_t kOffset =
                detail::GetLdsAffineStepOffset<typename BBlockWindowTmp::BottomTensorView,
                                               WG::kN,
                                               WG::kK,
                                               nIter * NPerBlockPerIter,
                                               kIter * KPerBlockPerIter>();

            b_warp_tensors(i_step) = b_warp_window_tmp.load_with_offset(number<kOffset>{});
        };

        static_for<0, kPrefetchDepth, 1>{}([&](auto i_step) { load_step(i_step); });

        __builtin_amdgcn_sched_barrier(0);

        static_for<0, kSteps / kStepsPerGroup, 1>{}([&](auto i_group) {
            constexpr index_t kFirstLoad = i_group * kStepsPerGroup + kPrefetchDepth;
            constexpr index_t kNumLoads =
                kFirstLoad >= kSteps
                    ? 0
                    : (kSteps - kFirstLoad < kStepsPerGroup ? kSteps - kFirstLoad : kStepsPerGroup);

            static_for<0, kNumLoads, 1>{}([&](auto j) { load_step(number<kFirstLoad + j>{}); });

            static_for<0, kStepsPerGroup, 1>{}([&](auto j) {
                constexpr index_t i_step = i_group * kStepsPerGroup + j;
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

struct HstuAttentionFwdPipelineQRKSVSPolicy
{
    template <typename Problem, bool kUseDirectLds = false>
    CK_TILE_DEVICE static constexpr auto GetNumKLdsBuffers()
    {
        constexpr index_t n0_loops =
            Problem::HstuAttentionTileSetting::kN0 / Problem::HstuAttentionTileSetting::kN0Sub;

        if constexpr(kUseDirectLds)
            return n0_loops;
        else
            return min(n0_loops, 2);
    }

    template <typename Problem, bool kUseDirectLds = false>
    CK_TILE_DEVICE static constexpr auto GetNumVLdsBuffers()
    {
        constexpr index_t k1_loops =
            Problem::HstuAttentionTileSetting::kN0 / Problem::HstuAttentionTileSetting::kK1;

        if constexpr(kUseDirectLds)
            return k1_loops;
        else
            return min(k1_loops, 2);
    }

    // XOR swizzle unit (elements) of the direct-to-LDS K and V LDS layouts: the LDS read views
    // (MakeK/VLdsBlockDescriptor with kUseXor) and the kernels' K/V dram views
    // (make_direct_lds_xor_dram_view) use the same one
    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr index_t GetKDirectLdsSwizzleUnit()
    {
        // On gfx950, GetAlignmentK and ScalePerVector (normal read size) are actually same
        return lcm(GetAlignmentK<Problem, true /*kUseDirectLds*/>(),
                   GetQKWarpGemmBScalarPerVector<Problem>());
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr index_t GetVDirectLdsSwizzleUnit()
    {
        // On gfx950, AlignmentV is actually bigger than ScalePerVector (trload read size)
        return lcm(GetAlignmentV<Problem, true /*kUseTrLoad*/, true /*kUseDirectLds*/>(),
                   GetPVTWarpGemmBScalarPerVector<Problem, true /*kUseTrLoad*/>());
    }

    // [issues, warps, lanes * vector] view of one LDS buffer of kRows x kRowLength elements
    // starting at element kOffset of the LDS: the form the raw DMA takes to step M0.
    template <index_t kBlockSize,
              index_t kRows,
              index_t kRowLength,
              index_t kVector,
              index_t kOffset>
    CK_TILE_HOST_DEVICE static constexpr auto MakeDirectLdsRawStoreDescriptor()
    {
        constexpr index_t kWarpSize   = get_warp_size();
        constexpr index_t NumWarps    = kBlockSize / kWarpSize;
        constexpr index_t LanesPerRow = kRowLength / kVector;
        constexpr index_t RowsPerWarp = kWarpSize / LanesPerRow;
        constexpr index_t NumIssues   = kRows / (RowsPerWarp * NumWarps);

        static_assert(NumIssues * RowsPerWarp * NumWarps == kRows, "Check failed!");

        constexpr auto desc_0 = make_naive_tensor_descriptor_with_offset(
            make_tuple(number<NumIssues>{},
                       number<NumWarps>{},
                       number<RowsPerWarp>{},
                       number<LanesPerRow>{},
                       number<kVector>{}),
            make_tuple(number<NumWarps * RowsPerWarp * kRowLength>{},
                       number<RowsPerWarp * kRowLength>{},
                       number<kRowLength>{},
                       number<kVector>{},
                       number<1>{}),
            number<kOffset>{},
            number<kVector>{},
            number<1>{});

        return transform_tensor_descriptor(
            desc_0,
            make_tuple(make_pass_through_transform(number<NumIssues>{}),
                       make_pass_through_transform(number<NumWarps>{}),
                       make_merge_transform(make_tuple(
                           number<RowsPerWarp>{}, number<LanesPerRow>{}, number<kVector>{}))),
            make_tuple(sequence<0>{}, sequence<1>{}, sequence<2, 3, 4>{}),
            make_tuple(sequence<0>{}, sequence<1>{}, sequence<2>{}));
    }

    // kVector: the elements per DMA of each thread, i.e. the KPerThread of the K dram tile
    // distribution
    template <typename Problem, index_t kVector, index_t i_buf>
    CK_TILE_HOST_DEVICE static constexpr auto MakeKLdsRawStoreBlockDescriptor(number<i_buf>)
    {
        constexpr index_t kN0Sub     = Problem::HstuAttentionTileSetting::kN0Sub;
        constexpr index_t kQKHeaddim = Problem::HstuAttentionTileSetting::kQKHeaddim;

        return MakeDirectLdsRawStoreDescriptor<Problem::kBlockSize,
                                               kN0Sub,
                                               kQKHeaddim,
                                               kVector,
                                               i_buf * kN0Sub * kQKHeaddim>();
    }

    // the V buffers follow the K buffers; kVector is the NPerThread of the V dram tile distribution
    template <typename Problem, index_t kVector, index_t i_buf>
    CK_TILE_HOST_DEVICE static constexpr auto MakeVLdsRawStoreBlockDescriptor(number<i_buf>)
    {
        constexpr index_t kK1 = Problem::HstuAttentionTileSetting::kK1;
        constexpr index_t kN1 = Problem::HstuAttentionTileSetting::kN1;

        return MakeDirectLdsRawStoreDescriptor<Problem::kBlockSize,
                                               kK1,
                                               kN1,
                                               kVector,
                                               GetSmemSizeK<Problem, true /*kUseDirectLds*/>() /
                                                       sizeof(typename Problem::QKVDataType) +
                                                   i_buf * kK1 * kN1>();
    }

    template <typename Problem>
    CK_TILE_DEVICE static constexpr auto MakeQRegTileDistribution()
    {
        using BlockGemm = remove_cvref_t<decltype(GetQKBlockGemm<Problem>())>;

        return BlockGemm::template MakeABlockTileDistribution<
            Problem::HstuAttentionTileSetting::kM0,
            Problem::HstuAttentionTileSetting::kQKHeaddim>();
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr ck_tile::index_t GetQKWarpGemmBScalarPerVector()
    {
        using BlockGemm       = remove_cvref_t<decltype(GetQKBlockGemm<Problem>())>;
        constexpr auto config = BlockGemm::Policy::template GetWarpGemmMWarpNWarp<Problem>();
        using WG              = remove_cvref_t<decltype(config.template at<0>())>;

        using BEncoding = typename WG::BWarpDstrEncoding;
        return BEncoding::detail::ys_lengths_[BEncoding::NDimY - 1];
    };

    template <typename Problem, bool kUseTrLoad = false>
    CK_TILE_HOST_DEVICE static constexpr ck_tile::index_t GetPVTWarpGemmBScalarPerVector()
    {
        using BlockGemm       = remove_cvref_t<decltype(GetPVTBlockGemm<Problem, kUseTrLoad>())>;
        constexpr auto config = BlockGemm::Policy::template GetWarpGemmMWarpNWarp<Problem>();
        using WG              = remove_cvref_t<decltype(config.template at<0>())>;

        using BEncoding = typename WG::BWarpDstrEncoding;

        if constexpr(kUseTrLoad)
        {
            using BEncodingForTrLoad =
                typename InputTileDistributionTraits<BEncoding, typename BlockGemm::BDataType>::
                    TransposedDstrEncode;

            return BEncodingForTrLoad::detail::ys_lengths_[BEncoding::NDimY - 1];
        }
        else
        {
            return BEncoding::detail::ys_lengths_[BEncoding::NDimY - 1];
        }
    };

    template <typename Problem>
    CK_TILE_DEVICE static constexpr auto MakeBiasDramTileDistribution()
    {
        using BlockGemm = remove_cvref_t<decltype(GetQKBlockGemm<Problem>())>;

        constexpr auto bias_block_dstr_encode = BlockGemm::template MakeCBlockDistributionEncode<
            Problem::HstuAttentionTileSetting::kM0,
            Problem::HstuAttentionTileSetting::kN0>();
        constexpr auto bias_block_dstr = make_static_tile_distribution(bias_block_dstr_encode);

        return bias_block_dstr;
    }

    template <typename Problem>
    CK_TILE_DEVICE static constexpr auto GetAlignmentBias()
    {
        using BlockGemm       = remove_cvref_t<decltype(GetQKBlockGemm<Problem>())>;
        constexpr auto config = BlockGemm::Policy::template GetWarpGemmMWarpNWarp<Problem>();
        using WG              = remove_cvref_t<decltype(config.template at<0>())>;

        return WG::WarpGemmAttribute::Impl::kCM1PerLane;
    }

    template <typename Problem>
    CK_TILE_DEVICE static constexpr auto GetAlignmentQ()
    {
        constexpr index_t MaxVectorSize = 16 / sizeof(typename Problem::QKVDataType);

        using BlockGemm       = remove_cvref_t<decltype(GetQKBlockGemm<Problem>())>;
        constexpr auto config = BlockGemm::Policy::template GetWarpGemmMWarpNWarp<Problem>();
        using WG              = remove_cvref_t<decltype(config.template at<0>())>;

        return min(MaxVectorSize, WG::kK / WG::WarpGemmAttribute::Impl::kABKLane);
    }

    template <typename Problem, bool kUseDirectLds = false>
    CK_TILE_HOST_DEVICE static constexpr auto GetAlignmentK()
    {
        return Problem::GetKDramTileAccessMaxVectorSize(bool_constant<kUseDirectLds>{});
    }

    template <typename Problem, bool kUseTrLoad = false, bool kUseDirectLds = false>
    CK_TILE_HOST_DEVICE static constexpr auto GetAlignmentV()
    {
        if constexpr(!kUseTrLoad)
        { // special consideration when shuffling is required before storing V to LDS
            using VDataType = remove_cvref_t<typename Problem::QKVDataType>;

            constexpr index_t kBlockSize = Problem::kBlockSize;
            constexpr index_t kNPerBlock = Problem::HstuAttentionTileSetting::kN1;
            constexpr index_t kKPerBlock = Problem::HstuAttentionTileSetting::kK1;

            constexpr index_t ElemPerThread = kNPerBlock * kKPerBlock / kBlockSize;

            constexpr index_t kMaxVecLoad = Problem::GetVDramTileAccessMaxVectorSize();
            constexpr index_t kMinVecLoad = 4 / sizeof(VDataType);

            // try to avoid writing sub-dword to LDS due to poor performance
            constexpr index_t kVecLoad = ((ElemPerThread / kMaxVecLoad) >= kMinVecLoad)
                                             ? kMaxVecLoad
                                             : (ElemPerThread / kMinVecLoad);

            if constexpr(kVecLoad == 4 * (ElemPerThread / kVecLoad))
                return kVecLoad / 2;

            // special treatment for hdim96
            if constexpr((std::is_same_v<VDataType, half_t> || std::is_same_v<VDataType, bf16_t>) &&
                         kNPerBlock == 96)
            {
                // VectorSize 3 not permitted since buffer_load_dword requires a multiplier of dword
                if(kVecLoad == 3)
                    return 6;
            }

            return kVecLoad;
        }
        else
        {
            return Problem::GetVDramTileAccessMaxVectorSize(bool_constant<kUseDirectLds>{});
        };
    }

    // ---- Tdm staging: plain-Lds descriptors and their pad ------------------------------
    //
    // The Tdm engine writes K and V into Lds itself, in plain row-major order, so the
    // xor-swizzled layouts MakeK/VLdsBlockDescriptor() build are not available to a Tdm
    // staged buffer; bank conflicts are avoided with a row pad instead. These four entry
    // points are additive -- the existing descriptors are untouched and still serve the
    // Default and TrLoad pipelines.
    //
    // They live here rather than in the pipelines because detail::GetTdm*() only exist inside
    // #if HSTU_LDS_STAGING_THROUGH_TDM_AVAILABLE. A call spelled `detail::GetTdm...` in a
    // pipeline is a non-dependent qualified-id, so it is looked up when the template is
    // *defined* rather than instantiated -- merely including a Tdm pipeline header on
    // gfx94/gfx95 would then be a parse error, however carefully the call is guarded by
    // if constexpr. Going through the policy makes the name dependent on Problem and keeps
    // the #if on this side, which is the same shape the rest of this file already uses.

    // The (pad_interval, pad_amount) pair for the global -> Lds copy that stages K, already
    // encoded the way the Tdm descriptor carries it. MakeKLdsPlainBlockDescriptor() derives
    // the reader side of the same buffer from the same GetTdmLdsPaddingConfigForNormalRead()
    // call; the two are separate derivations and have been checked to agree, so an edit to
    // one must be mirrored in the other.
    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto GetTdmPadConfigK()
    {
#if HSTU_LDS_STAGING_THROUGH_TDM_AVAILABLE
        constexpr index_t kKPerBlock = Problem::HstuAttentionTileSetting::kQKHeaddim;

        using BlockGemm       = remove_cvref_t<decltype(GetQKBlockGemm<Problem>())>;
        constexpr auto config = BlockGemm::Policy::template GetWarpGemmMWarpNWarp<Problem>();
        using WG              = remove_cvref_t<decltype(config.template at<0>())>;

        constexpr auto lds_pad =
            detail::GetTdmLdsPaddingConfigForNormalRead<WG, false /*inputB*/, kKPerBlock>();

        return detail::GetTdmRawPaddingConfig<lds_pad.at(number<0>{}), lds_pad.at(number<1>{})>();
#else
        static_assert(sizeof(Problem) == 0,
                      "GetTdmPadConfigK() is only meaningful where Tdm staging is available");
        return make_tuple(number<0>{}, number<0>{});
#endif
    }

    // As GetTdmPadConfigK(), for the V copy. V is read back with ds_read_tr, so its pad comes
    // from GetTdmLdsPaddingConfigForTrLoadRead() and is wider than K's.
    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto GetTdmPadConfigV()
    {
#if HSTU_LDS_STAGING_THROUGH_TDM_AVAILABLE
        constexpr index_t kNPerBlock = Problem::HstuAttentionTileSetting::kN1;

        using BlockGemm = remove_cvref_t<decltype(GetPVTBlockGemm<Problem, true /*kUseTrLoad*/>())>;
        constexpr auto config = BlockGemm::Policy::template GetWarpGemmMWarpNWarp<Problem>();
        using WG              = remove_cvref_t<decltype(config.template at<0>())>;

        constexpr auto lds_pad =
            detail::GetTdmLdsPaddingConfigForTrLoadRead<WG, false /*inputB*/, kNPerBlock>();

        return detail::GetTdmRawPaddingConfig<lds_pad.at(number<0>{}), lds_pad.at(number<1>{})>();
#else
        static_assert(sizeof(Problem) == 0,
                      "GetTdmPadConfigV() is only meaningful where Tdm staging is available");
        return make_tuple(number<0>{}, number<0>{});
#endif
    }

    // Reader-side descriptor for one Tdm staged K tile, i.e. kN0 rows as n0_loops slices of
    // kN0Sub. The Tdm pipelines allocate two of these (see GetSmemSizePlain()).
    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto MakeKLdsPlainBlockDescriptor()
    {
#if HSTU_LDS_STAGING_THROUGH_TDM_AVAILABLE
        constexpr index_t n0_loops =
            Problem::HstuAttentionTileSetting::kN0 / Problem::HstuAttentionTileSetting::kN0Sub;
        constexpr index_t kNPerBlock = Problem::HstuAttentionTileSetting::kN0Sub;
        constexpr index_t kKPerBlock = Problem::HstuAttentionTileSetting::kQKHeaddim;

        using BlockGemm       = remove_cvref_t<decltype(GetQKBlockGemm<Problem>())>;
        constexpr auto config = BlockGemm::Policy::template GetWarpGemmMWarpNWarp<Problem>();
        using WG              = remove_cvref_t<decltype(config.template at<0>())>;

        constexpr auto pad_config =
            detail::GetTdmLdsPaddingConfigForNormalRead<WG, false /*inputB*/, kKPerBlock>();

        // The Tdm writer counts the pad interval from the Lds origin of each wave's own write
        // (see MakeKDramTdmTileDistribution()), so every wave has to start on the pad grid.
        constexpr index_t kNumWarps = Problem::kBlockSize / get_warp_size();
        static_assert(kNPerBlock / kNumWarps * kKPerBlock * sizeof(typename Problem::QKVDataType) %
                              pad_config.at(number<0>{}) ==
                          0,
                      "each wave's Tdm K write must start on the Lds pad grid");

        return detail::MakeRowMajorLdsPaddedBlockDescriptor<typename Problem::QKVDataType,
                                                            n0_loops,
                                                            kNPerBlock,
                                                            kKPerBlock,
                                                            pad_config.at(number<0>{}),
                                                            pad_config.at(number<1>{})>();
#else
        static_assert(sizeof(Problem) == 0,
                      "MakeKLdsPlainBlockDescriptor() is only meaningful where Tdm staging is "
                      "available");
        return 0;
#endif
    }

    // Reader-side descriptor for NumCopies Tdm staged V tiles stacked along the rows, i.e.
    // NumCopies * kN0 rows, each copy as k1_loops slices of kK1. V is always read back with
    // ds_read_tr, so the pad is the wider tr-load one. kNPerBlock (= kN1) is the row length of
    // the V tile, i.e. the descriptor's Cols, which is what the pad interval has to be derived
    // from.
    template <typename Problem, index_t NumCopies = 1>
    CK_TILE_HOST_DEVICE static constexpr auto MakeVLdsPlainBlockDescriptor()
    {
#if HSTU_LDS_STAGING_THROUGH_TDM_AVAILABLE
        constexpr index_t k1_loops =
            Problem::HstuAttentionTileSetting::kN0 / Problem::HstuAttentionTileSetting::kK1;
        constexpr index_t kNPerBlock = Problem::HstuAttentionTileSetting::kN1;
        constexpr index_t kKPerBlock = Problem::HstuAttentionTileSetting::kK1;

        using BlockGemm = remove_cvref_t<decltype(GetPVTBlockGemm<Problem, true /*kUseTrLoad*/>())>;
        constexpr auto config = BlockGemm::Policy::template GetWarpGemmMWarpNWarp<Problem>();
        using WG              = remove_cvref_t<decltype(config.template at<0>())>;

        constexpr auto pad_config =
            detail::GetTdmLdsPaddingConfigForTrLoadRead<WG, false /*inputB*/, kNPerBlock>();

        // As for K: each wave writes kKPerBlock / kNumWarps rows of the V tile (see
        // MakeVDramTdmTileDistribution()) and has to start on the pad grid.
        constexpr index_t kNumWarps = Problem::kBlockSize / get_warp_size();
        static_assert(kKPerBlock / kNumWarps * kNPerBlock * sizeof(typename Problem::QKVDataType) %
                              pad_config.at(number<0>{}) ==
                          0,
                      "each wave's Tdm V write must start on the Lds pad grid");

        return detail::MakeRowMajorLdsPaddedBlockDescriptor<typename Problem::QKVDataType,
                                                            1,
                                                            NumCopies * k1_loops * kKPerBlock,
                                                            kNPerBlock,
                                                            pad_config.at(number<0>{}),
                                                            pad_config.at(number<1>{})>();
#else
        static_assert(sizeof(Problem) == 0,
                      "MakeVLdsPlainBlockDescriptor() is only meaningful where Tdm staging is "
                      "available");
        return 0;
#endif
    }

    template <typename Problem, bool kUseDirectLds = false, bool kUseXor = false>
    CK_TILE_HOST_DEVICE static constexpr auto MakeKLdsBlockDescriptor()
    {
        constexpr index_t NumKLdsBuffers = GetNumKLdsBuffers<Problem, kUseDirectLds>();
        constexpr index_t kNPerBlock     = Problem::HstuAttentionTileSetting::kN0Sub;
        constexpr index_t kKPerBlock     = Problem::HstuAttentionTileSetting::kQKHeaddim;
        constexpr index_t kKVector       = GetAlignmentK<Problem, kUseDirectLds>();

        if constexpr(kUseDirectLds)
        {
            if constexpr(kUseXor) // true, used for reading from Lds
            {
                constexpr index_t kSwizzleUnit = GetKDirectLdsSwizzleUnit<Problem>();

                constexpr auto desc_native = detail::MakeSimplySwizzledNativeDesc<Problem,
                                                                                  NumKLdsBuffers,
                                                                                  kNPerBlock,
                                                                                  kKPerBlock,
                                                                                  kSwizzleUnit>();

                return transform_tensor_descriptor(
                    desc_native,
                    make_tuple(make_merge_transform(
                                   make_tuple(number<NumKLdsBuffers>{}, number<kNPerBlock>{})),
                               make_pass_through_transform(number<kKPerBlock>{})),
                    make_tuple(sequence<0, 1>{}, sequence<2>{}),
                    make_tuple(sequence<0>{}, sequence<1>{}));
            }
            else // false, used for direct-lds writing to Lds
            {
                return detail::
                    MakeRowMajorLdsPlainBlockDescriptor<NumKLdsBuffers, kNPerBlock, kKPerBlock>();
            }
        }
        else if constexpr(!detail::IsPerfectHeaddimSize(kKPerBlock))
        // for hdim96 and hdim160, use simplest layout
        {
            constexpr index_t SingleBufferSize = kNPerBlock * kKPerBlock;

            constexpr auto k_lds_block_desc_0 = make_naive_tensor_descriptor(
                make_tuple(number<NumKLdsBuffers>{}, number<kNPerBlock>{}, number<kKPerBlock>{}),
                make_tuple(number<SingleBufferSize>{}, number<kKPerBlock>{}, number<1>{}),
                number<kKVector>{},
                number<1>{});

            constexpr auto k_lds_block_desc = transform_tensor_descriptor(
                k_lds_block_desc_0,
                make_tuple(make_merge_transform(
                               make_tuple(number<NumKLdsBuffers>{}, number<kNPerBlock>{})),
                           make_pass_through_transform(number<kKPerBlock>{})),
                make_tuple(sequence<0, 1>{}, sequence<2>{}),
                make_tuple(sequence<0>{}, sequence<1>{}));

            return k_lds_block_desc;
        }
        else if constexpr(GetQKWarpGemmBScalarPerVector<Problem>() >= GetAlignmentK<Problem>())
        { // This path can only be reached if WarpGemm is 16x16x32 or 32x32x16
            using BlockGemm       = remove_cvref_t<decltype(GetQKBlockGemm<Problem>())>;
            constexpr auto config = BlockGemm::Policy::template GetWarpGemmMWarpNWarp<Problem>();
            using WG              = remove_cvref_t<decltype(config.template at<0>())>;

            constexpr auto kSwizzleUnit =
                detail::GetSwizzleUnitForNormalRead<WG, false /*inputB*/>();

            constexpr auto desc_native = detail::MakeSwizzledNativeDesc<Problem,
                                                                        NumKLdsBuffers,
                                                                        kNPerBlock,
                                                                        kKPerBlock,
                                                                        kSwizzleUnit>();

            return transform_tensor_descriptor(
                desc_native,
                make_tuple(make_merge_transform(
                               make_tuple(number<NumKLdsBuffers>{}, number<kNPerBlock>{})),
                           make_pass_through_transform(number<kKPerBlock>{})),
                make_tuple(sequence<0, 1>{}, sequence<2>{}),
                make_tuple(sequence<0>{}, sequence<1>{}));
        }
        else
        {
            constexpr index_t kDsReadVector = GetQKWarpGemmBScalarPerVector<Problem>();

#if defined(__hstu_gfx95__) || defined(__hstu_gfx125__)
            constexpr index_t BankSpanElements = 64 * 4 / sizeof(typename Problem::QKVDataType);
#else
            constexpr index_t BankSpanElements = 32 * 4 / sizeof(typename Problem::QKVDataType);
#endif
            // specially constructed lay-out, so that ds_write2_b64 can be used by the compiler
            // to avoid bank-conflict when writing to Lds
            if constexpr(kKPerBlock == BankSpanElements)
            { // This path is always chosen for hdim64 on gfx942, and it was found
              // this path could not completely remove bank-conflicts for hdim128
                constexpr auto kSwizzleUnit = kDsReadVector;

                constexpr auto desc_native = detail::MakeSwizzledNativeDesc<Problem,
                                                                            NumKLdsBuffers,
                                                                            kNPerBlock,
                                                                            kKPerBlock,
                                                                            kSwizzleUnit>();

                return transform_tensor_descriptor(
                    desc_native,
                    make_tuple(make_merge_transform(
                                   make_tuple(number<NumKLdsBuffers>{}, number<kNPerBlock>{})),
                               make_pass_through_transform(number<kKPerBlock>{})),
                    make_tuple(sequence<0, 1>{}, sequence<2>{}),
                    make_tuple(sequence<0>{}, sequence<1>{}));
            }
            else
            { // This path is chosen for hdim128 on gfx942
              // This path is able to completely remove bank-conflicts for kKPerBlock >=
              // BankSpanElements (kKPerBlock < BankSpanElements not tested)
                static_assert(kKVector == 2 * kDsReadVector, "Check failed!");

                constexpr index_t SingleBufferSize = kKPerBlock * kNPerBlock + kKPerBlock / 2;

                constexpr auto k_lds_block_desc_0 = make_naive_tensor_descriptor(
                    make_tuple(number<NumKLdsBuffers>{},
                               number<kKPerBlock / kKVector>{},
                               number<2>{},
                               number<kNPerBlock>{},
                               number<kDsReadVector>{}),
                    make_tuple(number<SingleBufferSize>{},
                               number<kNPerBlock * kKVector + kDsReadVector>{},
                               number<kNPerBlock * kDsReadVector>{},
                               number<kDsReadVector>{},
                               number<1>{}),
                    number<kDsReadVector>{},
                    number<1>{});

                constexpr auto k_lds_block_desc = transform_tensor_descriptor(
                    k_lds_block_desc_0,
                    make_tuple(make_merge_transform(
                                   make_tuple(number<NumKLdsBuffers>{}, number<kNPerBlock>{})),
                               make_merge_transform(make_tuple(number<kKPerBlock / kKVector>{},
                                                               number<2>{},
                                                               number<kDsReadVector>{}))),
                    make_tuple(sequence<0, 3>{}, sequence<1, 2, 4>{}),
                    make_tuple(sequence<0>{}, sequence<1>{}));

                return k_lds_block_desc;
            }
        };
    }

    // load_tile_tdm() does not take the hardware box shape as an argument: it recovers it from
    // the Dram window, by reading back the ys_to_d lengths of the window's tile distribution.
    // A Tdm load therefore needs a distribution whose Y dims describe one contiguous rectangle
    // per wave. The general-purpose MakeK/VDramTileDistribution() do not: they scatter lanes
    // across the head dim, which is what a register-staged buffer_load wants but is read back
    // by Tdm as a scattered box that the Lds side cannot interpret. These two give Tdm the
    // layout it needs instead -- the seqlen dim is split across waves and the head dim is left
    // whole, so the Y lengths come out (rows-per-wave, head dim).
    //
    // Both pass IsWarpLevelParallelOnly = true, so the single P dim indexes the warp rather
    // than the lane: every lane of a wave shares one P and the rectangle is owned by the wave
    // as a whole. That is also why these must only ever be fed to load_tile_tdm() -- under a
    // plain load_tile() every lane would materialise the entire rectangle.
    template <typename Problem>
    CK_TILE_DEVICE static constexpr auto MakeKDramTdmTileDistribution()
    {
        constexpr index_t kBlockSize = Problem::kBlockSize;
        constexpr index_t kNPerBlock = Problem::HstuAttentionTileSetting::kN0Sub;
        constexpr index_t kKPerBlock = Problem::HstuAttentionTileSetting::kQKHeaddim;

        constexpr index_t warpNum = kBlockSize / get_warp_size();

        static_assert(kNPerBlock % warpNum == 0,
                      "kN0Sub must be divisible by the number of warps for the Tdm K "
                      "distribution");

        return make_static_tile_distribution(
            tile_distribution_encoding<
                sequence<>,                                    // RsLengths: no replication
                tuple<sequence<warpNum, kNPerBlock / warpNum>, // HsLengthss[0]: seqlen, by wave
                      sequence<kKPerBlock>>,                   // HsLengthss[1]: hdim, whole
                tuple<sequence<1>>,                            // Ps2RHssMajor: P0 -> H[0]
                tuple<sequence<0>>,                            // Ps2RHssMinor: P0 -> warpNum
                sequence<1, 2>,                                // Ys2RHsMajor
                sequence<1, 0>>{},                             // Ys2RHsMinor
            bool_constant<true>{});                            // IsWarpLevelParallelOnly
    }

    // As MakeKDramTdmTileDistribution(), for the V copy. V's tile setting names its axes the
    // other way round -- kK1 is the seqlen dim and kN1 the head dim -- so this is the
    // transposed orientation, the one the ds_read_tr reader expects.
    template <typename Problem>
    CK_TILE_DEVICE static constexpr auto MakeVDramTdmTileDistribution()
    {
        constexpr index_t kBlockSize = Problem::kBlockSize;
        constexpr index_t kNPerBlock = Problem::HstuAttentionTileSetting::kN1;
        constexpr index_t kKPerBlock = Problem::HstuAttentionTileSetting::kK1;

        constexpr index_t warpNum = kBlockSize / get_warp_size();

        static_assert(kKPerBlock % warpNum == 0,
                      "kK1 must be divisible by the number of warps for the Tdm V "
                      "distribution");

        return make_static_tile_distribution(
            tile_distribution_encoding<
                sequence<>,                                    // RsLengths: no replication
                tuple<sequence<warpNum, kKPerBlock / warpNum>, // HsLengthss[0]: seqlen, by wave
                      sequence<kNPerBlock>>,                   // HsLengthss[1]: hdim, whole
                tuple<sequence<1>>,                            // Ps2RHssMajor: P0 -> H[0]
                tuple<sequence<0>>,                            // Ps2RHssMinor: P0 -> warpNum
                sequence<1, 2>,                                // Ys2RHsMajor
                sequence<1, 0>>{},                             // Ys2RHsMinor
            bool_constant<true>{});                            // IsWarpLevelParallelOnly
    }

    template <typename Problem, bool kUseDirectLds = false>
    CK_TILE_DEVICE static constexpr auto MakeKDramTileDistribution()
    {
        constexpr index_t kBlockSize = Problem::kBlockSize;
        constexpr index_t kNPerBlock = Problem::HstuAttentionTileSetting::kN0Sub;
        constexpr index_t kKPerBlock = Problem::HstuAttentionTileSetting::kQKHeaddim;

        constexpr index_t kKVector = GetAlignmentK<Problem, kUseDirectLds>();
        constexpr index_t OtherK   = kKPerBlock / kKVector;

        if constexpr((OtherK & (OtherK - 1)) == 0)
        { // for kKPerBlock=32,64,128,256 or kKPerBlock=96 && kKVector=6
            constexpr index_t KPerThread = kKVector;
            constexpr index_t KThreads   = OtherK;

            constexpr index_t NThreadPerWarp = get_warp_size() / KThreads;
            constexpr index_t NumWarps       = kBlockSize / get_warp_size();
            constexpr index_t NPerThread     = kNPerBlock / (NThreadPerWarp * NumWarps);

            return make_static_tile_distribution(
                tile_distribution_encoding<sequence<1>,
                                           tuple<sequence<NPerThread, NumWarps, NThreadPerWarp>,
                                                 sequence<KThreads, KPerThread>>,
                                           tuple<sequence<1>, sequence<1, 2>>,
                                           tuple<sequence<1>, sequence<2, 0>>,
                                           sequence<1, 2>,
                                           sequence<0, 1>>{});
        }
        else
        { // for kKPerBlock=160 or kKPerBlock=96 && kKVector=1/2/4/8
            constexpr index_t KPerThread    = kKVector;
            constexpr index_t KRepPerThread = (OtherK % 3 == 0) ? 3 : 5;
            constexpr index_t KThreads      = OtherK / KRepPerThread;

            constexpr index_t NThreadPerWarp = get_warp_size() / KThreads;
            constexpr index_t NumWarps       = kBlockSize / get_warp_size();
            constexpr index_t NPerThread     = kNPerBlock / (NThreadPerWarp * NumWarps);

            if constexpr(kUseDirectLds)
            {
                // [KThreadPerWarp, NThreads] should be contiguous dims on the 2D space
                return make_static_tile_distribution(
                    tile_distribution_encoding<sequence<1>,
                                               tuple<sequence<NPerThread, NumWarps, NThreadPerWarp>,
                                                     sequence<KThreads, KRepPerThread, kKVector>>,
                                               tuple<sequence<1>, sequence<1, 2>>,
                                               tuple<sequence<1>, sequence<2, 0>>,
                                               sequence<1, 2, 2>,
                                               sequence<0, 1, 2>>{});
            }
            else
            { // depending on the actual performance result, this branch might not be needed
                return make_static_tile_distribution(
                    tile_distribution_encoding<sequence<1>,
                                               tuple<sequence<NPerThread, NumWarps, NThreadPerWarp>,
                                                     sequence<KRepPerThread, KThreads, KPerThread>>,
                                               tuple<sequence<1>, sequence<1, 2>>,
                                               tuple<sequence<1>, sequence<2, 1>>,
                                               sequence<1, 2, 2>,
                                               sequence<0, 0, 2>>{});
            }
        };
    }

    template <typename Problem,
              bool kUseTrLoad    = false,
              bool kUseDirectLds = false,
              bool kUseXor       = false>
    CK_TILE_HOST_DEVICE static constexpr auto MakeVLdsBlockDescriptor()
    {
        constexpr index_t NumVLdsBuffers = GetNumVLdsBuffers<Problem, kUseDirectLds>();
        constexpr index_t kBlockSize     = Problem::kBlockSize;
        constexpr index_t kNPerBlock     = Problem::HstuAttentionTileSetting::kN1;
        constexpr index_t kKPerBlock     = Problem::HstuAttentionTileSetting::kK1;

        if constexpr(kUseDirectLds)
        {
            static_assert(kUseTrLoad, "Check failed!");

            if constexpr(kUseXor) // true, used for reading from Lds
            {
                constexpr index_t kSwizzleUnit = GetVDirectLdsSwizzleUnit<Problem>();

                constexpr auto desc_native = detail::MakeSimplySwizzledNativeDesc<Problem,
                                                                                  NumVLdsBuffers,
                                                                                  kKPerBlock, // kN
                                                                                  kNPerBlock, // kK
                                                                                  kSwizzleUnit>();

                // merge: NumVLdsBuffers * [kK1, kVHeaddim] -> [kN0, kVHeaddim]
                return transform_tensor_descriptor(
                    desc_native,
                    make_tuple(make_merge_transform(
                                   make_tuple(number<NumVLdsBuffers>{}, number<kKPerBlock>{})),
                               make_pass_through_transform(number<kNPerBlock>{})),
                    make_tuple(sequence<0, 1>{}, sequence<2>{}),
                    make_tuple(sequence<0>{}, sequence<1>{}));
            }
            else // false, used for direct-lds writing to Lds
            {
                return detail::MakeRowMajorLdsPlainBlockDescriptor<NumVLdsBuffers,
                                                                   kKPerBlock, // kN
                                                                   kNPerBlock /* kK */>();
            }
        }
        else if constexpr(!kUseTrLoad)
        {
            constexpr index_t N1 = GetAlignmentV<Problem>();

            constexpr index_t ElemPerThread = kNPerBlock * kKPerBlock / kBlockSize;

            // K2 is the vector size for storing shuffled tile to LDS
            constexpr index_t K2 = ElemPerThread / N1;

            constexpr index_t kDsReadVector = GetPVTWarpGemmBScalarPerVector<Problem>();

            static_assert(kDsReadVector >= K2, "Check failed!");

            using BlockGemm       = remove_cvref_t<decltype(GetPVTBlockGemm<Problem>())>;
            constexpr auto config = BlockGemm::Policy::template GetWarpGemmMWarpNWarp<Problem>();
            using WG              = remove_cvref_t<decltype(config.template at<0>())>;

            constexpr auto PaddingCfg =
                detail::GetLdsPaddingConfigForNormalRead<WG, false /*inputB */, kKPerBlock>();

            constexpr auto PadInterval = PaddingCfg[number<0>{}];
            constexpr auto PadLength   = PaddingCfg[number<1>{}];

            return detail::MakeRowMajorLdsPaddedBlockDescriptor<NumVLdsBuffers,
                                                                kNPerBlock,
                                                                kKPerBlock,
                                                                PadInterval,
                                                                PadLength>();
        }
        else
        {
            // With trload read,  16 threads per cycle access the [4Tl, 4Tm*4E] block and cross-bar
            // transpose it to [4E, 4Tm*4Tl] layout suitable for mfma. For hdim128, kK1=32, [32,
            // 128] = [2R*4Th*4Tl, 8R*4Tm*4E],  8R columns swizzled by 32 rows, for each value of
            // 8R, the 4Tl rows is mapped to separate bigger bank-groups (each has kSwizzleUnit
            // elements), able to guarantee 16 threads (4Tl*4Tm) in one cycle hitting to separate
            // smaller bank-groups (each has 4 elements, two dwords)
            using BlockGemm       = remove_cvref_t<decltype(GetPVTBlockGemm<Problem>())>;
            constexpr auto config = BlockGemm::Policy::template GetWarpGemmMWarpNWarp<Problem>();
            using WG              = remove_cvref_t<decltype(config.template at<0>())>;

            constexpr auto kSwizzleUnit =
                detail::GetSwizzleUnitForTrLoadRead<WG, false /*inputB*/>();

            constexpr auto desc_native = detail::MakeSwizzledNativeDesc<Problem,
                                                                        NumVLdsBuffers,
                                                                        kKPerBlock, // kN
                                                                        kNPerBlock, // kK
                                                                        kSwizzleUnit>();

            // merge: NumVLdsBuffers * [kK1, kVHeaddim] -> [kN0, kVHeaddim]
            return transform_tensor_descriptor(
                desc_native,
                make_tuple(make_merge_transform(
                               make_tuple(number<NumVLdsBuffers>{}, number<kKPerBlock>{})),
                           make_pass_through_transform(number<kNPerBlock>{})),
                make_tuple(sequence<0, 1>{}, sequence<2>{}),
                make_tuple(sequence<0>{}, sequence<1>{}));
        };
    }

    template <typename Problem, bool kUseTrLoad = false, bool kUseDirectLds = false>
    CK_TILE_DEVICE static constexpr auto MakeVDramTileDistribution()
    {
        constexpr index_t kBlockSize = Problem::kBlockSize;
        constexpr index_t kNPerBlock = Problem::HstuAttentionTileSetting::kN1;
        constexpr index_t kKPerBlock = Problem::HstuAttentionTileSetting::kK1;

        constexpr index_t kVector = GetAlignmentV<Problem, kUseTrLoad, kUseDirectLds>();

        if constexpr(!kUseTrLoad)
        {
            constexpr index_t NPerThread = kVector;
            constexpr index_t NThreads   = kNPerBlock / NPerThread;

            constexpr index_t ElemPerThread = kNPerBlock * kKPerBlock / kBlockSize;

            constexpr index_t KPerThread     = ElemPerThread / NPerThread;
            constexpr index_t KThreadPerWarp = get_warp_size() / NThreads;
            constexpr index_t NumWarps       = kBlockSize / get_warp_size();

            return make_static_tile_distribution(
                tile_distribution_encoding<sequence<1>,
                                           tuple<sequence<NThreads, NPerThread>,
                                                 sequence<NumWarps, KThreadPerWarp, KPerThread>>,
                                           tuple<sequence<2>, sequence<2, 1>>,
                                           tuple<sequence<0>, sequence<1, 0>>,
                                           sequence<2, 1>,
                                           sequence<2, 1>>{});
        }
        else
        {
            constexpr index_t OtherN = kNPerBlock / kVector;

            if constexpr((OtherN & (OtherN - 1)) == 0)
            { // for kNPerBlock=32,64,128,256 or kNPerBlock=96 && kVector=6
                constexpr index_t NPerThread = kVector;
                constexpr index_t NThreads   = OtherN;

                constexpr index_t KThreadPerWarp = get_warp_size() / NThreads;
                constexpr index_t NumWarps       = kBlockSize / get_warp_size();
                constexpr index_t KPerThread     = kKPerBlock / (KThreadPerWarp * NumWarps);

                return make_static_tile_distribution(
                    tile_distribution_encoding<sequence<1>,
                                               tuple<sequence<KPerThread, NumWarps, KThreadPerWarp>,
                                                     sequence<NThreads, NPerThread>>,
                                               tuple<sequence<1>, sequence<1, 2>>,
                                               tuple<sequence<1>, sequence<2, 0>>,
                                               sequence<1, 2>,
                                               sequence<0, 1>>{});
            }
            else
            { // for kNPerBlock=160 or kNPerBlock=96 && kVector=1/2/4/8
                constexpr index_t NPerThread    = kVector;
                constexpr index_t NRepPerThread = (OtherN % 3 == 0) ? 3 : 5;
                constexpr index_t NThreads      = OtherN / NRepPerThread;

                constexpr index_t KThreadPerWarp = get_warp_size() / NThreads;
                constexpr index_t NumWarps       = kBlockSize / get_warp_size();
                constexpr index_t KPerThread     = kKPerBlock / (KThreadPerWarp * NumWarps);

                if constexpr(kUseDirectLds)
                {
                    // [KThreadPerWarp, NThreads] should be contiguous dims on the 2D space
                    return make_static_tile_distribution(
                        tile_distribution_encoding<
                            sequence<1>,
                            tuple<sequence<KPerThread, NumWarps, KThreadPerWarp>,
                                  sequence<NThreads, NRepPerThread, NPerThread>>,
                            tuple<sequence<1>, sequence<1, 2>>,
                            tuple<sequence<1>, sequence<2, 0>>,
                            sequence<1, 2, 2>,
                            sequence<0, 1, 2>>{});
                }
                else
                { // depending on the actual performance result, this branch might not be needed
                    return make_static_tile_distribution(
                        tile_distribution_encoding<
                            sequence<1>,
                            tuple<sequence<KPerThread, NumWarps, KThreadPerWarp>,
                                  sequence<NRepPerThread, NThreads, NPerThread>>,
                            tuple<sequence<1>, sequence<1, 2>>,
                            tuple<sequence<1>, sequence<2, 1>>,
                            sequence<1, 2, 2>,
                            sequence<0, 0, 2>>{});
                }
            };
        };
    }

#if !HSTU_LDS_READ_WITH_TRANSPOSE_AVAILABLE
    template <typename Problem>
    CK_TILE_DEVICE static constexpr auto MakeShuffledVRegTileDistribution()
    {
        constexpr index_t kBlockSize = Problem::kBlockSize;
        constexpr index_t kNPerBlock = Problem::HstuAttentionTileSetting::kN1;
        constexpr index_t kKPerBlock = Problem::HstuAttentionTileSetting::kK1;

        constexpr index_t NPerThread = GetAlignmentV<Problem>();
        constexpr index_t NThreads   = kNPerBlock / NPerThread;

        constexpr index_t ElemPerThread = kNPerBlock * kKPerBlock / kBlockSize;

        constexpr index_t KPerThread     = ElemPerThread / NPerThread;
        constexpr index_t KThreadPerWarp = get_warp_size() / NThreads;
        constexpr index_t NumWarps       = kBlockSize / get_warp_size();

        return make_static_tile_distribution(
            tile_distribution_encoding<sequence<1>,
                                       tuple<sequence<NThreads, NPerThread>,
                                             sequence<NumWarps, KThreadPerWarp, KPerThread>>,
                                       tuple<sequence<2>, sequence<2, 1>>,
                                       tuple<sequence<0>, sequence<1, 0>>,
                                       sequence<1, 2>,
                                       sequence<1, 2>>{});
    }
#endif

    template <typename Problem>
    CK_TILE_DEVICE static constexpr index_t GetQKBlockGemmSingleRepM()
    {
        return Problem::HstuAttentionTileSetting::Gemm0BlockWarps::at(number<0>{}) *
               Problem::HstuAttentionTileSetting::Gemm0WarpTile::at(number<0>{});
    };

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto GetQKBlockGemm()
    {
        using GemmProblem = BlockGemmProblem<
            typename Problem::QKVDataType,
            typename Problem::QKVDataType,
            typename Problem::GemmAccDataType,
            Problem::kNumGemm0Warps * get_warp_size(),
            TileGemmShape<sequence<Problem::HstuAttentionTileSetting::kM0,
                                   Problem::HstuAttentionTileSetting::kN0Sub,
                                   Problem::HstuAttentionTileSetting::kQKHeaddim>,
                          typename Problem::HstuAttentionTileSetting::Gemm0BlockWarps,
                          typename Problem::HstuAttentionTileSetting::Gemm0WarpTile>>;

        auto warp_gemm = [&]() {
            if constexpr((std::is_same_v<typename Problem::QKVDataType, half_t> ||
                          std::is_same_v<typename Problem::QKVDataType, bf16_t>) &&
                         std::is_same_v<typename Problem::GemmAccDataType, float>)
            {
                constexpr index_t WarpGemmM =
                    Problem::HstuAttentionTileSetting::Gemm0WarpTile::at(number<0>{});
                constexpr index_t WarpGemmK =
                    Problem::HstuAttentionTileSetting::Gemm0WarpTile::at(number<2>{});

#if defined(__hstu_gfx125__)
                static_assert(WarpGemmM == 16 && WarpGemmK == 32, "Not supported WarpGemm sizes!");
#elif defined(__hstu_gfx95__)
                static_assert((WarpGemmM == 16 && WarpGemmK == 32) ||
                                  (WarpGemmM == 32 && WarpGemmK == 16),
                              "Not supported WarpGemm sizes!");
#else
                static_assert((WarpGemmM == 16 && (WarpGemmK == 16 || WarpGemmK == 32)) ||
                                  (WarpGemmM == 32 && (WarpGemmK == 8 || WarpGemmK == 16)),
                              "Not supported WarpGemm sizes!");
#endif

                return WarpGemmDispatcher<
                    typename Problem::QKVDataType,
                    typename Problem::QKVDataType,
                    typename Problem::GemmAccDataType,
                    Problem::HstuAttentionTileSetting::Gemm0WarpTile::at(number<0>{}),
                    Problem::HstuAttentionTileSetting::Gemm0WarpTile::at(number<1>{}),
                    Problem::HstuAttentionTileSetting::Gemm0WarpTile::at(number<2>{}),
                    true,
                    false,
                    false,
                    WGAttrNumAccessEnum::Default>{};
            }
            else
            {
                static_assert(false, "Not supported data types!");
            }
        }();

        using BlockGemmPolicy = BlockGemmARegBSmemCRegV2CustomPolicy<
            typename Problem::QKVDataType,
            typename Problem::QKVDataType,
            typename Problem::GemmAccDataType,
            typename Problem::HstuAttentionTileSetting::Gemm0BlockWarps,
            decltype(warp_gemm)>;

        if constexpr(1 < Problem::kNumGemm0Warps)
            return BlockGemmARegBSmemCRegV2Hack_0<GemmProblem, BlockGemmPolicy>{};
        else
            return BlockGemmARegBSmemCRegOneWarpV1<GemmProblem, BlockGemmPolicy>{};
    }

    template <typename Problem>
    CK_TILE_DEVICE static constexpr auto GetPVTBlockGemmSingleRepN()
    {
        return Problem::HstuAttentionTileSetting::Gemm1WarpTile::at(number<1>{}) *
               Problem::HstuAttentionTileSetting::Gemm1BlockWarps::at(number<1>{});
    };

    template <typename Problem, bool kUseTrLoad = false>
    CK_TILE_HOST_DEVICE static constexpr auto GetPVTBlockGemm()
    {
        using GemmProblem = BlockGemmProblem<
            typename Problem::QKVDataType,
            typename Problem::QKVDataType,
            typename Problem::GemmAccDataType,
            Problem::kNumGemm1Warps * get_warp_size(),
            TileGemmShape<sequence<Problem::HstuAttentionTileSetting::kM0,
                                   Problem::HstuAttentionTileSetting::kN1,
                                   Problem::HstuAttentionTileSetting::kK1>,
                          typename Problem::HstuAttentionTileSetting::Gemm1BlockWarps,
                          typename Problem::HstuAttentionTileSetting::Gemm1WarpTile>>;

        auto warp_gemm = [&]() {
            if constexpr((std::is_same_v<typename Problem::QKVDataType, half_t> ||
                          std::is_same_v<typename Problem::QKVDataType, bf16_t>) &&
                         std::is_same_v<typename Problem::GemmAccDataType, float>)
            {
                constexpr index_t WarpGemmM =
                    Problem::HstuAttentionTileSetting::Gemm1WarpTile::at(number<0>{});
                constexpr index_t WarpGemmK =
                    Problem::HstuAttentionTileSetting::Gemm1WarpTile::at(number<2>{});

#if defined(__hstu_gfx125__)
                static_assert(WarpGemmM == 16 && WarpGemmK == 32, "Not supported WarpGemm sizes!");
#elif defined(__hstu_gfx95__)
                static_assert((WarpGemmM == 16 && WarpGemmK == 32) ||
                                  (WarpGemmM == 32 && WarpGemmK == 16),
                              "Not supported WarpGemm sizes!");
#else
                static_assert((WarpGemmM == 16 && (WarpGemmK == 16 || WarpGemmK == 32)) ||
                                  (WarpGemmM == 32 && (WarpGemmK == 8 || WarpGemmK == 16)),
                              "Not supported WarpGemm sizes!");
#endif

                if constexpr((WarpGemmM == 16 && WarpGemmK == 32) ||
                             (WarpGemmM == 32 && WarpGemmK == 16))
                    return WarpGemmDispatcher<
                        typename Problem::QKVDataType,
                        typename Problem::QKVDataType,
                        typename Problem::GemmAccDataType,
                        Problem::HstuAttentionTileSetting::Gemm1WarpTile::at(number<0>{}),
                        Problem::HstuAttentionTileSetting::Gemm1WarpTile::at(number<1>{}),
                        Problem::HstuAttentionTileSetting::Gemm1WarpTile::at(number<2>{}),
                        true,
                        false,
                        false,
#if defined(__hstu_gfx95__)
                        WGAttrNumAccessEnum::Double
#else
                        WGAttrNumAccessEnum::Default
#endif
                        >{};
                else
                    return WarpGemmDispatcher<
                        typename Problem::QKVDataType,
                        typename Problem::QKVDataType,
                        typename Problem::GemmAccDataType,
                        Problem::HstuAttentionTileSetting::Gemm1WarpTile::at(number<0>{}),
                        Problem::HstuAttentionTileSetting::Gemm1WarpTile::at(number<1>{}),
                        Problem::HstuAttentionTileSetting::Gemm1WarpTile::at(number<2>{}),
                        true,
                        false,
                        false,
                        WGAttrNumAccessEnum::Default>{};
            }
            else
            {
                static_assert(false, "Not supported data types!");
            }
        }();

        using WarpGemm = remove_cvref_t<decltype(warp_gemm)>;

        using BlockGemmPolicy = BlockGemmARegBSmemCRegV2CustomPolicy<
            typename Problem::QKVDataType,
            typename Problem::QKVDataType,
            typename Problem::GemmAccDataType,
            typename Problem::HstuAttentionTileSetting::Gemm1BlockWarps,
            WarpGemm>;

        if constexpr(!kUseTrLoad)
        {
            return BlockGemmARegBSmemCRegV2Hack_1<GemmProblem, BlockGemmPolicy>{};
        }
        else
        {
            return BlockGemmARegBSmemTrLoadCRegV2Hack_1<GemmProblem, BlockGemmPolicy>{};
        };
    }

    // Tdm pipelines: the hdim128 tiles with more than one gemm0 warp (M64 and M128) get the
    // group-prefetch variant of the Q@K block gemm; everything else keeps GetQKBlockGemm().
    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto GetQKBlockGemmTdm()
    {
        using BaseGemm = remove_cvref_t<decltype(GetQKBlockGemm<Problem>())>;

        if constexpr(Problem::HstuAttentionTileSetting::kQKHeaddim == 128 &&
                     1 < Problem::kNumGemm0Warps)
        {
            constexpr bool kM128             = Problem::HstuAttentionTileSetting::kM0 >= 128;
            constexpr index_t kPrefetchDepth = kM128 ? (Problem::kUseSoftmax ? 8 : 4) : 8;
            constexpr index_t kStepsPerGroup = kM128 ? 4 : 8;

            return BlockGemmARegBSmemCRegTdmGroupPrefetch_0<typename BaseGemm::Problem,
                                                            typename BaseGemm::Policy,
                                                            kPrefetchDepth,
                                                            kStepsPerGroup>{};
        }
        else
            return BaseGemm{};
    }

    // Tdm pipelines: the hdim128 tiles get the group-prefetch variant of the trload P@V block
    // gemm (same distributions, V ds_load_tr issued in groups ahead of the WMMAs). kPrefetchDepth
    // is how many (kIter, nIter) B tiles are in flight ahead of the WMMAs, kStepsPerGroup how many
    // steps share one group of loads; both cost 8 VGPRs per step (16x32 bf16 per wave32).
    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto GetPVTBlockGemmTdm()
    {
        using BaseGemm = remove_cvref_t<decltype(GetPVTBlockGemm<Problem, true>())>;

        if constexpr(Problem::HstuAttentionTileSetting::kQKHeaddim == 128)
        {
            constexpr bool kM128             = Problem::HstuAttentionTileSetting::kM0 >= 128;
            constexpr index_t kPrefetchDepth = kM128 ? 4 : 8;
            constexpr index_t kStepsPerGroup = kM128 ? 2 : 8;

            return BlockGemmARegBSmemTrLoadCRegTdmGroupPrefetch_1<typename BaseGemm::Problem,
                                                                  typename BaseGemm::Policy,
                                                                  kPrefetchDepth,
                                                                  kStepsPerGroup>{};
        }
        else
            return BaseGemm{};
    }

    template <typename Problem, bool kUseTrLoad = false>
    CK_TILE_DEVICE static constexpr auto GetAlignmentO()
    {
        using BlockGemm       = remove_cvref_t<decltype(GetPVTBlockGemm<Problem, kUseTrLoad>())>;
        constexpr auto config = BlockGemm::Policy::template GetWarpGemmMWarpNWarp<Problem>();
        using WG              = remove_cvref_t<decltype(config.template at<0>())>;

        return WG::WarpGemmAttribute::Impl::kCM1PerLane;
    }

    template <typename Problem, bool kUseDirectLds = false>
    CK_TILE_HOST_DEVICE static constexpr ck_tile::index_t GetSmemSizeK()
    {
        constexpr auto actual_bytes =
            MakeKLdsBlockDescriptor<Problem, kUseDirectLds>().get_element_space_size() *
            sizeof(typename Problem::QKVDataType);

        return (actual_bytes + 63) / 64 * 64;
    };

    template <typename Problem, bool kPipelineUseTrLoad = false, bool kUseDirectLds = false>
    CK_TILE_HOST_DEVICE static constexpr ck_tile::index_t GetSmemSizeV()
    {
        constexpr auto actual_bytes =
            MakeVLdsBlockDescriptor<Problem, kPipelineUseTrLoad, kUseDirectLds>()
                .get_element_space_size() *
            sizeof(typename Problem::QKVDataType);

        return (actual_bytes + 63) / 64 * 64;
    };

#if defined(__hstu_gfx94__)
    template <typename Problem>
    CK_TILE_DEVICE static constexpr ck_tile::index_t IsFirstKBufferOverlapLastVBuffer()
    {
        constexpr index_t NumKLdsBuffers            = GetNumKLdsBuffers<Problem>();
        constexpr index_t NumVLdsBuffers            = GetNumVLdsBuffers<Problem>();
        constexpr index_t first_k_buff_upper_offset = GetSmemSizeK<Problem>() / NumKLdsBuffers;
        constexpr index_t last_v_buff_lower_offset =
            (GetSmemSizeV<Problem>() / NumVLdsBuffers) * (NumVLdsBuffers - 1);

        return first_k_buff_upper_offset > last_v_buff_lower_offset;
    };

    template <typename Problem>
    CK_TILE_DEVICE static constexpr ck_tile::index_t IsFirstVBufferOverlapLastKBuffer()
    {
        constexpr index_t NumKLdsBuffers            = GetNumKLdsBuffers<Problem>();
        constexpr index_t NumVLdsBuffers            = GetNumVLdsBuffers<Problem>();
        constexpr index_t first_v_buff_upper_offset = GetSmemSizeV<Problem>() / NumVLdsBuffers;
        constexpr index_t last_k_buff_lower_offset =
            (GetSmemSizeK<Problem>() / NumKLdsBuffers) * (NumKLdsBuffers - 1);

        return first_v_buff_upper_offset > last_k_buff_lower_offset;
    };
#endif

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr ck_tile::index_t GetSmemSizeDropout()
    {
        return 0;
    };

#if defined(__hstu_gfx94__)
    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr ck_tile::index_t GetSmemSize()
    {
        return max(GetSmemSizeK<Problem>(), GetSmemSizeV<Problem>()) +
               GetSmemSizeDropout<Problem>();
    }
#else
    template <typename Problem, bool kPipelineUseTrLoad = false, bool kUseDirectLds = false>
    CK_TILE_HOST_DEVICE static constexpr ck_tile::index_t GetSmemSize()
    {
        return GetSmemSizeK<Problem, kUseDirectLds>() +
               GetSmemSizeV<Problem, kPipelineUseTrLoad, kUseDirectLds>() +
               GetSmemSizeDropout<Problem>();
    }
#endif

    // Lds sizes for one Tdm staged K/V tile. Same rounding as GetSmemSizeK/V(), over the
    // plain descriptors.
    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr ck_tile::index_t GetSmemSizeKPlain()
    {
        constexpr auto actual_bytes =
            MakeKLdsPlainBlockDescriptor<Problem>().get_element_space_size() *
            sizeof(typename Problem::QKVDataType);

        return (actual_bytes + 63) / 64 * 64;
    };

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr ck_tile::index_t GetSmemSizeVPlain()
    {
        constexpr auto actual_bytes =
            MakeVLdsPlainBlockDescriptor<Problem>().get_element_space_size() *
            sizeof(typename Problem::QKVDataType);

        return (actual_bytes + 63) / 64 * 64;
    };

    // Lds layout of the Tdm pipelines: [K copy 0 | K copy 1 | V copy 0 | V copy 1 | dropout].
    // Each copy holds one whole kv-tile; one copy is read by the gemms of the current iteration
    // while Tdm fills the other with the next kv-tile.
    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr ck_tile::index_t GetSmemSizePlain()
    {
        return GetSmemSizeKPlain<Problem>() * 2 + GetSmemSizeVPlain<Problem>() * 2 +
               GetSmemSizeDropout<Problem>();
    }

    // Element offset of V copy 1 in the NumCopies = 2 V descriptor, i.e. where its row kN0 starts
    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr ck_tile::index_t GetVLdsCopy1Offset()
    {
        constexpr index_t offset = MakeVLdsPlainBlockDescriptor<Problem, 2>().calculate_offset(
            make_multi_index(Problem::HstuAttentionTileSetting::kN0, 0));

        static_assert(offset * sizeof(typename Problem::QKVDataType) ==
                          GetSmemSizeVPlain<Problem>(),
                      "V copy 1 must start where GetSmemSizePlain() puts it");

        return offset;
    }
};

} // namespace ck_tile
