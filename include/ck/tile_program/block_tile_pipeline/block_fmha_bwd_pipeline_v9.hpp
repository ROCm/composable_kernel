// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2024, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck/utility/common_header.hpp"
#include "ck/tensor_description/tensor_descriptor.hpp"
#include "ck/tensor_description/tensor_descriptor_helper.hpp"
#include "ck/tensor_description/tensor_adaptor.hpp"

#include "ck/tile_program/tile/tile_distribution.hpp"
#include "ck/tile_program/tile/load_tile.hpp"
#include "ck/tile_program/tile/store_tile.hpp"
#include "ck/tile_program/tile/tile_elementwise.hpp"
#include "ck/tile_program/tile/tile_gemm_shape.hpp"
#include "ck/tile_program/tile/slice_tile.hpp"
#include "ck/tile_program/warp_tile/warp_gemm.hpp"
#include "ck/tile_program/block_tile_pipeline/block_fmha_bwd_pipeline_default_policy.hpp"
#include "ck/tile_program/block_tile/block_reduce.hpp"
#include "ck/tile_program/tile/shuffle_distributed_tensor.hpp"

namespace ck {
namespace tile_program {
namespace block {

template <typename Problem, typename Policy = BlockFmhaBwdPipelineDefaultPolicy>
struct BlockFmhaBwdPipelineV9
{
    using QDataType    = remove_cvref_t<typename Problem::QDataType>;
    using KDataType    = remove_cvref_t<typename Problem::KDataType>;
    using VDataType    = remove_cvref_t<typename Problem::VDataType>;
    using GemmDataType = remove_cvref_t<typename Problem::GemmDataType>;
    using BiasDataType = remove_cvref_t<typename Problem::BiasDataType>;
    using LSEDataType  = remove_cvref_t<typename Problem::LSEDataType>;
    using AccDataType  = remove_cvref_t<typename Problem::AccDataType>;
    using DDataType    = remove_cvref_t<typename Problem::DDataType>;
    // using ZDataType        = remove_cvref_t<typename Problem::ZDataType>;
    using ODataType        = remove_cvref_t<typename Problem::ODataType>;
    using OGradDataType    = remove_cvref_t<typename Problem::OGradDataType>;
    using QGradDataType    = remove_cvref_t<typename Problem::QGradDataType>;
    using KGradDataType    = remove_cvref_t<typename Problem::KGradDataType>;
    using VGradDataType    = remove_cvref_t<typename Problem::VGradDataType>;
    using BiasGradDataType = remove_cvref_t<typename Problem::BiasGradDataType>;
    using FmhaMask         = remove_cvref_t<typename Problem::FmhaMask>;

    using BlockFmhaShape = remove_cvref_t<typename Problem::BlockFmhaShape>;

    static constexpr index_t kBlockPerCu = Problem::kBlockPerCu;
    static constexpr index_t kBlockSize  = Problem::kBlockSize;

    static constexpr index_t kM0        = BlockFmhaShape::kM0;
    static constexpr index_t kN0        = BlockFmhaShape::kN0;
    static constexpr index_t kK0        = BlockFmhaShape::kK0;
    static constexpr index_t kK1        = BlockFmhaShape::kK1;
    static constexpr index_t kK2        = BlockFmhaShape::kK2;
    static constexpr index_t kK3        = BlockFmhaShape::kK3;
    static constexpr index_t kK4        = BlockFmhaShape::kK4;
    static constexpr index_t kQKHeaddim = BlockFmhaShape::kQKHeaddim;
    static constexpr index_t kVHeaddim  = BlockFmhaShape::kVHeaddim;

    static constexpr bool kQLoadOnce      = BlockFmhaShape::kQLoadOnce;
    static constexpr bool kQTLoadOnce     = BlockFmhaShape::kQTLoadOnce;
    static constexpr bool kKLoadOnce      = BlockFmhaShape::kKLoadOnce;
    static constexpr bool kKTLoadOnce     = BlockFmhaShape::kKTLoadOnce;
    static constexpr bool kVLoadOnce      = BlockFmhaShape::kVLoadOnce;
    static constexpr bool kOGradLoadOnce  = BlockFmhaShape::kOGradLoadOnce;
    static constexpr bool kOGradTLoadOnce = BlockFmhaShape::kOGradTLoadOnce;

    static constexpr bool kIsGroupMode = Problem::kIsGroupMode;
    static constexpr bool kPadSeqLenQ  = Problem::kPadSeqLenQ;
    static constexpr bool kPadSeqLenK  = Problem::kPadSeqLenK;
    static constexpr bool kPadHeadDimQ = Problem::kPadHeadDimQ;
    static constexpr bool kPadHeadDimV = Problem::kPadHeadDimV;
    static constexpr bool kHasBias     = Problem::kHasBias;

    __host__ __device__ static constexpr ck::index_t GetSmemSize()
    {
        return Policy::template GetSmemSize<Problem>();
    }

    template <typename QDramBlockWindowTmp,
              typename QTDramBlockWindowTmp,
              typename KDramBlockWindowTmp,
              typename KTDramBlockWindowTmp,
              typename VDramBlockWindowTmp,
              typename BiasDramBlockWindowTmp,
              typename OGradDramBlockWindowTmp,
              typename OGradTDramBlockWindowTmp,
              typename LSEDramBlockWindowTmp,
              typename DDramBlockWindowTmp,
              typename QGradDramBlockWindowTmp,
              typename BiasGradDramBlockWindowTmp>
    __host__ __device__ auto
    operator()(const QDramBlockWindowTmp& q_dram_block_window_tmp,
               const QTDramBlockWindowTmp& qt_dram_block_window_tmp,
               const KDramBlockWindowTmp& k_dram_block_window_tmp,
               const KTDramBlockWindowTmp& kt_dram_block_window_tmp,
               const VDramBlockWindowTmp& v_dram_block_window_tmp,
               const BiasDramBlockWindowTmp& bias_dram_block_window_tmp,
               const OGradDramBlockWindowTmp& do_dram_block_window_tmp,
               const OGradTDramBlockWindowTmp& dot_dram_block_window_tmp,
               const LSEDramBlockWindowTmp& lse_dram_block_window_tmp,
               const DDramBlockWindowTmp& d_dram_block_window_tmp,
               const QGradDramBlockWindowTmp& dq_dram_block_window_tmp,
               const BiasGradDramBlockWindowTmp& dbias_dram_block_window_tmp,
               FmhaMask mask,
               float raw_scale,
#if CK_FMHA_FWD_FAST_EXP2
               float scale,
#endif
               void* smem_ptr) const
    {
        static_assert(
            is_same_v<QDataType, remove_cvref_t<typename QDramBlockWindowTmp::DataType>> &&
                is_same_v<QDataType, remove_cvref_t<typename QTDramBlockWindowTmp::DataType>> &&
                is_same_v<KDataType, remove_cvref_t<typename KDramBlockWindowTmp::DataType>> &&
                is_same_v<KDataType, remove_cvref_t<typename KTDramBlockWindowTmp::DataType>> &&
                is_same_v<VDataType, remove_cvref_t<typename VDramBlockWindowTmp::DataType>> &&
                is_same_v<OGradDataType,
                          remove_cvref_t<typename OGradDramBlockWindowTmp::DataType>> &&
                is_same_v<OGradDataType,
                          remove_cvref_t<typename OGradTDramBlockWindowTmp::DataType>> &&
                is_same_v<LSEDataType, remove_cvref_t<typename LSEDramBlockWindowTmp::DataType>> &&
                is_same_v<DDataType, remove_cvref_t<typename DDramBlockWindowTmp::DataType>> &&
                is_same_v<QGradDataType,
                          remove_cvref_t<typename QGradDramBlockWindowTmp::DataType>>,
            "wrong!");

        static_assert(kM0 == QDramBlockWindowTmp{}.GetWindowLengths()[Number<0>{}] &&
                          kQKHeaddim == QTDramBlockWindowTmp{}.GetWindowLengths()[Number<0>{}] &&
                          kN0 == KDramBlockWindowTmp{}.GetWindowLengths()[Number<0>{}] &&
                          kQKHeaddim == KTDramBlockWindowTmp{}.GetWindowLengths()[Number<0>{}] &&
                          kN0 == VDramBlockWindowTmp{}.GetWindowLengths()[Number<0>{}] &&
                          kM0 == BiasDramBlockWindowTmp{}.GetWindowLengths()[Number<0>{}] &&
                          kN0 == BiasDramBlockWindowTmp{}.GetWindowLengths()[Number<1>{}] &&
                          kM0 == OGradDramBlockWindowTmp{}.GetWindowLengths()[Number<0>{}] &&
                          kVHeaddim == OGradTDramBlockWindowTmp{}.GetWindowLengths()[Number<0>{}] &&
                          kM0 == LSEDramBlockWindowTmp{}.GetWindowLengths()[Number<0>{}] &&
                          kM0 == DDramBlockWindowTmp{}.GetWindowLengths()[Number<0>{}] &&
                          kM0 == QGradDramBlockWindowTmp{}.GetWindowLengths()[Number<0>{}] &&
                          kM0 == BiasGradDramBlockWindowTmp{}.GetWindowLengths()[Number<0>{}] &&
                          kN0 == BiasGradDramBlockWindowTmp{}.GetWindowLengths()[Number<1>{}],
                      "wrong!");

        // Q tile in LDS
        QDataType* q_lds_ptr = static_cast<QDataType*>(static_cast<void*>(
            static_cast<char*>(smem_ptr) + Policy::template GetSmemSizeK<Problem>() +
            Policy::template GetSmemSizeKT<Problem>()));
        auto q_lds           = make_tensor_view<AddressSpaceEnum::Lds>(
            q_lds_ptr, Policy::template MakeQLdsBlockDescriptor<Problem>());
        auto q_lds_window =
            make_tile_window(q_lds, make_tuple(Number<kM0>{}, Number<kK0>{}), {0, 0});

        // QT tile in LDS
        QDataType* qt_lds_ptr = static_cast<QDataType*>(static_cast<void*>(
            static_cast<char*>(smem_ptr) + Policy::template GetSmemSizeK<Problem>() +
            Policy::template GetSmemSizeKT<Problem>()));
        auto qt_lds           = make_tensor_view<AddressSpaceEnum::Lds>(
            qt_lds_ptr, Policy::template MakeQTLdsBlockDescriptor<Problem>());
        auto qt_lds_window =
            make_tile_window(qt_lds, make_tuple(Number<kQKHeaddim>{}, Number<kK3>{}), {0, 0});

        // K tile in LDS
        auto k_lds = make_tensor_view<AddressSpaceEnum::Lds>(
            reinterpret_cast<KDataType*>(smem_ptr),
            Policy::template MakeKLdsBlockDescriptor<Problem>());
        auto k_lds_window =
            make_tile_window(k_lds, make_tuple(Number<kN0>{}, Number<kQKHeaddim>{}), {0, 0});

        // KT tile in LDS
        KDataType* kt_lds_ptr = static_cast<KDataType*>(static_cast<void*>(
            static_cast<char*>(smem_ptr) + Policy::template GetSmemSizeK<Problem>()));
        auto kt_lds           = make_tensor_view<AddressSpaceEnum::Lds>(
            kt_lds_ptr, Policy::template MakeKTLdsBlockDescriptor<Problem>());
        auto kt_lds_window =
            make_tile_window(kt_lds, make_tuple(Number<kQKHeaddim>{}, Number<kN0>{}), {0, 0});

        // OGrad tile in LDS
        OGradDataType* do_lds_ptr = static_cast<OGradDataType*>(static_cast<void*>(
            static_cast<char*>(smem_ptr) + Policy::template GetSmemSizeK<Problem>() +
            Policy::template GetSmemSizeKT<Problem>()));
        auto do_lds               = make_tensor_view<AddressSpaceEnum::Lds>(
            do_lds_ptr, Policy::template MakeOGradLdsBlockDescriptor<Problem>());
        auto do_lds_window =
            make_tile_window(do_lds, make_tuple(Number<kM0>{}, Number<kK2>{}), {0, 0});

        // OGradT tile in LDS
        OGradDataType* dot_lds_ptr = static_cast<OGradDataType*>(static_cast<void*>(
            static_cast<char*>(smem_ptr) + Policy::template GetSmemSizeK<Problem>() +
            Policy::template GetSmemSizeKT<Problem>()));
        auto dot_lds               = make_tensor_view<AddressSpaceEnum::Lds>(
            dot_lds_ptr, Policy::template MakeOGradTLdsBlockDescriptor<Problem>());
        auto dot_lds_window =
            make_tile_window(dot_lds, make_tuple(Number<kVHeaddim>{}, Number<kK1>{}), {0, 0});

        // SGrad tile in LDS
        GemmDataType* ds_lds_ptr = static_cast<GemmDataType*>(static_cast<void*>(
            static_cast<char*>(smem_ptr) + Policy::template GetSmemSizeK<Problem>() +
            Policy::template GetSmemSizeKT<Problem>()));
        auto ds_lds              = make_tensor_view<AddressSpaceEnum::Lds>(
            ds_lds_ptr, Policy::template MakeSGradLdsBlockDescriptor<Problem>());
        auto ds_lds_window =
            make_tile_window(ds_lds, make_tuple(Number<kM0>{}, Number<kN0>{}), {0, 0});

        // BiasT/BiasGradT tile in LDS, use the same size and layout
        BiasDataType* biast_lds_ptr = static_cast<BiasDataType*>(static_cast<void*>(
            static_cast<char*>(smem_ptr) + Policy::template GetSmemSizeK<Problem>() +
            Policy::template GetSmemSizeKT<Problem>()));
        auto biast_lds              = make_tensor_view<AddressSpaceEnum::Lds>(
            biast_lds_ptr, Policy::template MakeBiasTLdsBlockDescriptor<Problem>());
        auto biast_lds_shuffle_window =
            make_tile_window(biast_lds, make_tuple(Number<kM0>{}, Number<kN0>{}), {0, 0});
        auto dbiast_lds_shuffle_window =
            make_tile_window(biast_lds,
                             make_tuple(Number<kM0>{}, Number<kN0>{}),
                             {0, 0},
                             Policy::template MakeShuffledBiasTileDistribution<Problem>());

        static_assert(std::is_same_v<BiasDataType, BiasGradDataType>,
                      "BiasDataType and BiasGradDataType should be the same!");

        // Block GEMM
        constexpr auto gemm_0 = Policy::template GetQKBlockGemm<Problem>();
        constexpr auto gemm_1 = Policy::template GetPTOGradTBlockGemm<Problem>();
        constexpr auto gemm_2 = Policy::template GetOGradVBlockGemm<Problem>();
        constexpr auto gemm_3 = Policy::template GetSGradTQTBlockGemm<Problem>();
        constexpr auto gemm_4 = Policy::template GetSGradKTBlockGemm<Problem>();

        auto v_dram_window = make_tile_window(
            v_dram_block_window_tmp.GetBottomTensorView(),
            v_dram_block_window_tmp.GetWindowLengths(),
            v_dram_block_window_tmp.GetWindowOrigin(),
            Policy::template MakeVInRegDramTileDistribution<Problem, decltype(gemm_2)>());

        auto v = load_tile(v_dram_window); // persistent V register tile

        using SPTBlockTileType     = decltype(gemm_0.MakeCBlockTile());
        using SPGradTBlockTileType = decltype(gemm_2.MakeCBlockTile());
        using QGradBlockTileType   = decltype(gemm_4.MakeCBlockTile());

        // init VGrad & KGrad
        auto dv_acc = decltype(gemm_1.MakeCBlockTile()){};
        auto dk_acc = decltype(gemm_3.MakeCBlockTile()){};

        clear_tile(dv_acc);
        clear_tile(dk_acc);

        auto k_dram_window = make_tile_window(
            k_dram_block_window_tmp.GetBottomTensorView(),
            k_dram_block_window_tmp.GetWindowLengths(),
            k_dram_block_window_tmp.GetWindowOrigin(),
            Policy::template MakeKDramTileDistribution<Problem>()); // K DRAM tile window for
                                                                    // load

        __builtin_amdgcn_sched_barrier(0);
        const auto k_origin = k_dram_window.GetWindowOrigin();
        const auto [seqlen_q_start, seqlen_q_end] =
            mask.GetTileRangeAlongY(k_origin.At(Number<0>{}), Number<kM0>{}, Number<kN0>{});

        const auto num_total_loop = math::integer_divide_ceil(seqlen_q_end - seqlen_q_start, kM0);

        // check early exit if masked and no work to do.
        if constexpr(FmhaMask::IsMasking)
        {
            if(num_total_loop <= 0)
            {
                // Note: here dk_acc&dv_acc are all cleard, return it
                // Note: v loaded but no fence, ignore it.
                return ck::make_tuple(dk_acc, dv_acc);
            }
        }

        auto k_block_tile = load_tile(k_dram_window);

        store_tile(k_lds_window, k_block_tile); // // persistent K in LDS

        auto kt_dram_block_window = kt_dram_block_window_tmp;

        auto kt_dram_window = make_tile_window(
            kt_dram_block_window.GetBottomTensorView(),
            kt_dram_block_window.GetWindowLengths(),
            kt_dram_block_window.GetWindowOrigin(),
            Policy::template MakeKTDramTileDistribution<Problem>()); // K^T DRAM tile window for
                                                                     // load

        auto kt_block_tile = load_tile(kt_dram_window);

        auto kt_shuffle_tmp = make_static_distributed_tensor<KDataType>(
            Policy::template MakeShuffledKTRegBlockDescriptor<Problem>());
        shuffle_distributed_tensor(kt_shuffle_tmp, kt_block_tile);

        store_tile(kt_lds_window, kt_shuffle_tmp); // persistent K^T in LDS

        auto q_dram_block_window = make_tile_window(q_dram_block_window_tmp.GetBottomTensorView(),
                                                    q_dram_block_window_tmp.GetWindowLengths(),
                                                    {seqlen_q_start, 0});

        auto qt_dram_block_window = make_tile_window(qt_dram_block_window_tmp.GetBottomTensorView(),
                                                     qt_dram_block_window_tmp.GetWindowLengths(),
                                                     {0, seqlen_q_start});

        auto do_dram_block_window = make_tile_window(do_dram_block_window_tmp.GetBottomTensorView(),
                                                     do_dram_block_window_tmp.GetWindowLengths(),
                                                     {seqlen_q_start, 0});

        auto dot_dram_block_window =
            make_tile_window(dot_dram_block_window_tmp.GetBottomTensorView(),
                             dot_dram_block_window_tmp.GetWindowLengths(),
                             {0, seqlen_q_start});

        auto dq_dram_block_window = make_tile_window(dq_dram_block_window_tmp.GetBottomTensorView(),
                                                     dq_dram_block_window_tmp.GetWindowLengths(),
                                                     {seqlen_q_start, 0});

        auto lse_dram_block_window =
            make_tile_window(lse_dram_block_window_tmp.GetBottomTensorView(),
                             lse_dram_block_window_tmp.GetWindowLengths(),
                             {seqlen_q_start});

        auto d_dram_block_window = make_tile_window(d_dram_block_window_tmp.GetBottomTensorView(),
                                                    d_dram_block_window_tmp.GetWindowLengths(),
                                                    {seqlen_q_start});

        const auto bias_origin = bias_dram_block_window_tmp.GetWindowOrigin();
        auto bias_dram_block_window =
            make_tile_window(bias_dram_block_window_tmp.GetBottomTensorView(),
                             bias_dram_block_window_tmp.GetWindowLengths(),
                             {seqlen_q_start, bias_origin.At(Number<1>{})}); // M/N

        const auto dbias_origin = dbias_dram_block_window_tmp.GetWindowOrigin();
        auto dbias_dram_block_window =
            make_tile_window(dbias_dram_block_window_tmp.GetBottomTensorView(),
                             dbias_dram_block_window_tmp.GetWindowLengths(),
                             {seqlen_q_start, dbias_origin.At(Number<1>{})}); // M/N

        auto qt_dram_window =
            make_tile_window(qt_dram_block_window.GetBottomTensorView(),
                             qt_dram_block_window.GetWindowLengths(),
                             qt_dram_block_window.GetWindowOrigin(),
                             Policy::template MakeQTDramTileDistribution<Problem>());

        auto dot_dram_window =
            make_tile_window(dot_dram_block_window.GetBottomTensorView(),
                             dot_dram_block_window.GetWindowLengths(),
                             dot_dram_block_window.GetWindowOrigin(),
                             Policy::template MakeOGradTDramTileDistribution<Problem>());

        auto lse_dram_window = make_tile_window(
            lse_dram_block_window.GetBottomTensorView(),
            lse_dram_block_window.GetWindowLengths(),
            lse_dram_block_window.GetWindowOrigin(),
            Policy::template MakeLSEDDramTileDistribution<Problem, decltype(gemm_0)>());

        auto d_dram_window = make_tile_window(
            d_dram_block_window.GetBottomTensorView(),
            d_dram_block_window.GetWindowLengths(),
            d_dram_block_window.GetWindowOrigin(),
            Policy::template MakeLSEDDramTileDistribution<Problem, decltype(gemm_0)>());

        auto bias_dram_window =
            make_tile_window(bias_dram_block_window.GetBottomTensorView(),
                             bias_dram_block_window.GetWindowLengths(),
                             bias_dram_block_window.GetWindowOrigin(),
                             Policy::template MakeBiasTileDistribution<Problem>());

        auto biast_lds_window = make_tile_window(
            biast_lds_shuffle_window.GetBottomTensorView(),
            biast_lds_shuffle_window.GetWindowLengths(),
            biast_lds_shuffle_window.GetWindowOrigin(),
            Policy::template MakeBiasTTileDistribution<Problem, decltype(gemm_0)>());

        index_t i_total_loops      = 0;
        constexpr index_t k0_loops = kQKHeaddim / kK0;
        constexpr index_t k1_loops = kM0 / kK1;
        constexpr index_t k2_loops = kVHeaddim / kK2;
        constexpr index_t k3_loops = kM0 / kK3;
        constexpr index_t k4_loops = kN0 / kK4;
        do
        {
            auto q_dram_window = make_tile_window(
                q_dram_block_window.GetBottomTensorView(),
                q_dram_block_window.GetWindowLengths(),
                q_dram_block_window.GetWindowOrigin(),
                Policy::template MakeQDramTileDistribution<Problem>()); // Q DRAM tile window for
                                                                        // load

            auto do_dram_window = make_tile_window(
                do_dram_block_window.GetBottomTensorView(),
                do_dram_block_window.GetWindowLengths(),
                do_dram_block_window.GetWindowOrigin(),
                Policy::template MakeOGradDramTileDistribution<Problem>()); // OGrad DRAM tile
                                                                            // window for load

            // STAGE 1, Q@K Gemm0
            auto st_acc = SPTBlockTileType{};

            auto q_block_tile = load_tile(q_dram_window);
            {
                move_tile_window(q_dram_window, {0, kK0});

                clear_tile(st_acc); // Initialize S^T

                store_tile(q_lds_window, q_block_tile);  // LDS write 0
                q_block_tile = load_tile(q_dram_window); // global read 1
            }

            if constexpr(kHasBias)
            {
                __builtin_amdgcn_sched_barrier(
                    0); // prevent from messing up the order of global loads
            }
            const auto bias_tile = load_tile(bias_dram_window); // load bias tile
            if constexpr(kHasBias)
            {
                __builtin_amdgcn_sched_barrier(
                    0); // prevent from messing up the order of global loads
            }

            if constexpr(k0_loops > 2)
            {
                static_for<0, k0_loops - 2, 1>{}([&](auto i_k0) {
                    block_sync_lds();
                    gemm_0(st_acc,
                           q_lds_window,
                           get_slice_tile(k_lds_window,
                                          Sequence<0, i_k0 * kK0>{},
                                          Sequence<kN0, (i_k0 + 1) * kK0>{}));
                    block_sync_lds();
                    move_tile_window(q_dram_window, {0, kK0});

                    store_tile(q_lds_window,
                               q_block_tile);                // LDS write i + 1
                    q_block_tile = load_tile(q_dram_window); // global read i + 2
                });
            }

            const auto dot_prefetch = load_tile(dot_dram_window); // prefetch load OGrad^T tile
            {                                                     // tail
                block_sync_lds();
                gemm_0(st_acc,
                       q_lds_window,
                       get_slice_tile(k_lds_window,
                                      Sequence<0, (k0_loops - 2) * kK0>{},
                                      Sequence<kN0, (k0_loops - 1) * kK0>{}));
                block_sync_lds();

                store_tile(q_lds_window, q_block_tile);
                block_sync_lds();

                gemm_0(st_acc,
                       q_lds_window,
                       get_slice_tile(k_lds_window,
                                      Sequence<0, (k0_loops - 1) * kK0>{},
                                      Sequence<kN0, k0_loops * kK0>{}));
            }

            // STAGE 2, Scale, Add bias, Mask, Softmax
            if constexpr(kHasBias)
            {
                block_sync_lds();
                auto bias_shuffle_tmp = make_static_distributed_tensor<BiasDataType>(
                    Policy::template MakeShuffledBiasTileDistribution<Problem>());
                shuffle_distributed_tensor(bias_shuffle_tmp, bias_tile);
                store_tile(biast_lds_shuffle_window, bias_shuffle_tmp);
                block_sync_lds();
                auto biast_tile = load_tile(biast_lds_window);
                tile_elementwise_inout(
                    [&](auto& x, const auto& y) {
#if !CK_FMHA_FWD_FAST_EXP2
                        x = raw_scale * x + type_convert<AccDataType>(y);
#else
                        x = scale * x + math::log2e_v<AccDataType> * type_convert<AccDataType>(y);
#endif
                    },
                    st_acc,
                    biast_tile);
                move_tile_window(bias_dram_window, {kM0, 0});
            }
            else
            {
#if !CK_FMHA_FWD_FAST_EXP2
                tile_elementwise_inout([&raw_scale](auto& x) { x = x * raw_scale; }, st_acc);
#endif
            }

            if constexpr(kPadSeqLenK || FmhaMask::IsMasking)
            {
                const auto q_origin      = q_dram_block_window.GetWindowOrigin();
                bool need_perpixel_check = mask.IsEdgeTile(q_origin.At(Number<0>{}),
                                                           k_origin.At(Number<0>{}),
                                                           Number<kM0>{},
                                                           Number<kN0>{});
                if(need_perpixel_check)
                {
                    set_tile_if(
                        st_acc, -NumericLimits<AccDataType>::Infinity(), [&](auto tile_idx) {
                            const auto row = q_origin.At(Number<0>{}) + tile_idx.At(Number<0>{});
                            const auto col = k_origin.At(Number<0>{}) + tile_idx.At(Number<1>{});
                            return mask.IsOutOfBound(row, col);
                        });
                }
            }

            const auto lse = load_tile(lse_dram_window);

            static const auto get_validated_lse = [](LSEDataType raw_lse) {
                if constexpr(kHasBias || FmhaMask::IsMasking)
                {
                    return raw_lse == -NumericLimits<LSEDataType>::Infinity()
                               ? type_convert<LSEDataType>(0.f)
                               : raw_lse;
                }
                else
                {
                    return raw_lse;
                }
            };

            auto pt                 = SPTBlockTileType{};
            constexpr auto pt_spans = decltype(pt)::GetDistributedSpans();
            sweep_tile_span(pt_spans[Number<0>{}], [&](auto idx0) {
                constexpr auto i_idx = make_tuple(idx0);
#if CK_FMHA_FWD_FAST_EXP2
                auto row_lse = math::log2e_v<LSEDataType> * get_validated_lse(lse[i_idx]);
#endif
                sweep_tile_span(pt_spans[Number<1>{}], [&](auto idx1) {
                    constexpr auto i_j_idx = make_tuple(idx0, idx1);
#if CK_FMHA_FWD_FAST_EXP2
                    if constexpr(kHasBias)
                    {
                        pt(i_j_idx) = math::exp2(st_acc[i_j_idx] - row_lse);
                    }
                    else
                    {
                        pt(i_j_idx) = math::exp2(scale * st_acc[i_j_idx] - row_lse);
                    }
#else
                    pt(i_j_idx) = math::exp(st_acc[i_j_idx] - get_validated_lse(lse[i_idx]));
#endif
                });
            });

            // STAGE 3, P^T@OGrad^T Gemm1
            block_sync_lds();
            {
                auto dot_shuffle_tmp = make_static_distributed_tensor<OGradDataType>(
                    Policy::template MakeShuffledOGradTRegBlockDescriptor<Problem>());
                shuffle_distributed_tensor(dot_shuffle_tmp, dot_prefetch);
                store_tile(dot_lds_window,
                           dot_shuffle_tmp); // store the prefetch
            }
            move_tile_window(dot_dram_window, {0, kK1});

            const auto pt_gemm = cast_tile<GemmDataType>(pt);

            if constexpr(k1_loops > 1)
            {
                static_for<0, k1_loops - 1, 1>{}([&](auto i_k1) {
                    const auto dot = load_tile(dot_dram_window); // load next OGrad^T
                    block_sync_lds();
                    gemm_1(dv_acc,
                           get_slice_tile(pt_gemm,
                                          Sequence<i_k1 * kK1, 0>{},
                                          Sequence<(i_k1 + 1) * kK1, kN0>{}),
                           dot_lds_window);
                    block_sync_lds();
                    auto dot_shuffle_tmp = make_static_distributed_tensor<OGradDataType>(
                        Policy::template MakeShuffledOGradTRegBlockDescriptor<Problem>());
                    shuffle_distributed_tensor(dot_shuffle_tmp, dot);
                    store_tile(dot_lds_window,
                               dot_shuffle_tmp); // store the prefetch

                    move_tile_window(dot_dram_window, {0, kK1});
                });
            }
            auto do_block_tile = load_tile(do_dram_window); // prefetch load OGrad tile
            // tail
            {
                block_sync_lds();
                gemm_1(dv_acc,
                       get_slice_tile(
                           pt_gemm, Sequence<(k1_loops - 1) * kK1, 0>{}, Sequence<kM0, kN0>{}),
                       dot_lds_window);
                block_sync_lds();
            }

            // STAGE 4, OGrad@V Gemm2
            auto dpt_acc = SPGradTBlockTileType{};

            {
                move_tile_window(do_dram_window, {0, kK2});

                clear_tile(dpt_acc); // Initialize PGrad^T

                store_tile(do_lds_window, do_block_tile);  // LDS write 0
                do_block_tile = load_tile(do_dram_window); // global read 1
            }

            if constexpr(k2_loops > 2)
            {
                static_for<0, k2_loops - 2, 1>{}([&](auto i_k2) {
                    block_sync_lds();
                    gemm_2(dpt_acc,
                           do_lds_window,
                           get_slice_tile(
                               v, Sequence<0, i_k2 * kK2>{}, Sequence<kN0, (i_k2 + 1) * kK2>{}));
                    block_sync_lds();
                    move_tile_window(do_dram_window, {0, kK2});

                    store_tile(do_lds_window,
                               do_block_tile);                 // LDS write i + 1
                    do_block_tile = load_tile(do_dram_window); // global read i + 2
                });
            }

            const auto qt_prefetch = load_tile(qt_dram_window); // prefetch load Q^T tile
            {                                                   // tail
                block_sync_lds();
                gemm_2(dpt_acc,
                       do_lds_window,
                       get_slice_tile(v,
                                      Sequence<0, (k2_loops - 2) * kK2>{},
                                      Sequence<kN0, (k2_loops - 1) * kK2>{}));
                block_sync_lds();

                store_tile(do_lds_window, do_block_tile);
                block_sync_lds();

                gemm_2(dpt_acc,
                       do_lds_window,
                       get_slice_tile(v,
                                      Sequence<0, (k2_loops - 1) * kK2>{},
                                      Sequence<kN0, k2_loops * kK2>{}));
            }

            // STAGE 5, P^T(PGrad^T - D)
            const auto d = load_tile(d_dram_window);

            auto dst                 = SPGradTBlockTileType{};
            constexpr auto dst_spans = decltype(dst)::GetDistributedSpans();
            sweep_tile_span(dst_spans[Number<0>{}], [&](auto idx0) {
                constexpr auto i_idx = make_tuple(idx0);
                sweep_tile_span(dst_spans[Number<1>{}], [&](auto idx1) {
                    constexpr auto i_j_idx = make_tuple(idx0, idx1);
                    dst(i_j_idx)           = pt[i_j_idx] * (dpt_acc[i_j_idx] - d[i_idx]);
                });
            });

            if constexpr(kHasBias)
            {
                const auto dbiast = cast_tile<BiasGradDataType>(dst);
                store_tile(biast_lds_shuffle_window, dbiast);
                block_sync_lds();
                auto dbiast_tile        = load_tile(dbiast_lds_shuffle_window);
                auto dbiast_shuffle_tmp = make_static_distributed_tensor<BiasGradDataType>(
                    Policy::template MakeBiasTileDistribution<Problem>());
                shuffle_distributed_tensor(dbiast_shuffle_tmp, dbiast_tile);
                store_tile(dbias_dram_block_window, dbiast_shuffle_tmp);
                move_tile_window(dbias_dram_block_window, {kM0, 0});
            }

            // STAGE 6, SGrad^T@Q^T Gemm3
            block_sync_lds();
            {
                auto qt_shuffle_tmp = make_static_distributed_tensor<QDataType>(
                    Policy::template MakeShuffledQTRegBlockDescriptor<Problem>());
                shuffle_distributed_tensor(qt_shuffle_tmp, qt_prefetch);
                store_tile(qt_lds_window,
                           qt_shuffle_tmp); // store the prefetch
            }
            move_tile_window(qt_dram_window, {0, kK3});

            const auto dst_gemm = cast_tile<GemmDataType>(dst);

            if constexpr(k3_loops > 1)
            {
                static_for<0, k3_loops - 1, 1>{}([&](auto i_k3) {
                    const auto qt = load_tile(qt_dram_window); // load next Q^T
                    block_sync_lds();
                    gemm_3(dk_acc,
                           get_slice_tile(dst_gemm,
                                          Sequence<i_k3 * kK3, 0>{},
                                          Sequence<(i_k3 + 1) * kK3, kN0>{}),
                           qt_lds_window);
                    block_sync_lds();
                    auto qt_shuffle_tmp = make_static_distributed_tensor<QDataType>(
                        Policy::template MakeShuffledQTRegBlockDescriptor<Problem>());
                    shuffle_distributed_tensor(qt_shuffle_tmp, qt);
                    store_tile(qt_lds_window,
                               qt_shuffle_tmp); // store the prefetch

                    move_tile_window(qt_dram_window, {0, kK3});
                });
            }
            // tail
            {
                block_sync_lds();
                gemm_3(dk_acc,
                       get_slice_tile(
                           dst_gemm, Sequence<(k3_loops - 1) * kK3, 0>{}, Sequence<kM0, kN0>{}),
                       qt_lds_window);
                block_sync_lds();
            }

            // STAGE 7, SGrad@K^T Gemm4
            store_tile(ds_lds_window, dst_gemm);

            auto dq_acc = QGradBlockTileType{};
            clear_tile(dq_acc); // Initialize QGrad

            block_sync_lds();

            static_for<0, k4_loops, 1>{}([&](auto i_k4) {
                gemm_4(dq_acc,
                       get_slice_tile(ds_lds_window,
                                      Sequence<0, i_k4 * kK4>{},
                                      Sequence<kM0, (i_k4 + 1) * kK4>{}),
                       get_slice_tile(kt_lds_window,
                                      Sequence<0, i_k4 * kK4>{},
                                      Sequence<kQKHeaddim, (i_k4 + 1) * kK4>{}));
            });

            // QGrad Scale
            tile_elementwise_inout([&raw_scale](auto& x) { x = x * raw_scale; }, dq_acc);
            const auto dq = cast_tile<QGradDataType>(dq_acc);
            update_tile(dq_dram_block_window, dq);

            // move tile windows
            move_tile_window(q_dram_block_window, {kM0, 0});
            move_tile_window(dq_dram_block_window, {kM0, 0});
            move_tile_window(do_dram_block_window, {kM0, 0});
            move_tile_window(lse_dram_window, {kM0});
            move_tile_window(d_dram_window, {kM0});
        } while(++i_total_loops < num_total_loop);

        // KGrad Scale
        tile_elementwise_inout([&raw_scale](auto& x) { x = x * raw_scale; }, dk_acc);

        return ck::make_tuple(dk_acc, dv_acc);
    }
};

} // namespace block
} // namespace tile_program
} // namespace ck
