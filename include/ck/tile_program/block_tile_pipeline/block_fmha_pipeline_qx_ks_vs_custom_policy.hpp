// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2023, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck/utility/common_header.hpp"
#include "ck/tensor_description/tensor_descriptor.hpp"
#include "ck/tensor_description/tensor_descriptor_helper.hpp"
#include "ck/tensor_description/tensor_adaptor.hpp"

#include "ck/tile_program/tile/tile_distribution.hpp"
#include "ck/tile_program/tile/tile_elementwise.hpp"
#include "ck/tile_program/tile/tile_gemm_shape.hpp"
#include "ck/tile_program/warp_tile/warp_gemm_dispatcher.hpp"
#include "ck/tile_program/block_tile_pipeline/block_gemm_pipeline_problem.hpp"
#include "ck/tile_program/block_tile/block_gemm_areg_bsmem_creg_v1.hpp"
#include "ck/tile_program/block_tile/block_gemm_areg_bsmem_creg_v1_custom_policy.hpp"
#include "ck/tile_program/block_tile/block_gemm_areg_bsmem_creg_v2.hpp"
#include "ck/tile_program/block_tile/block_gemm_areg_bsmem_creg_v2_custom_policy.hpp"
#include "ck/tile_program/block_tile/block_gemm_asmem_bsmem_creg_v1.hpp"
#include "ck/tile_program/block_tile/block_gemm_asmem_bsmem_creg_v1_custom_policy.hpp"
#include "ck/tensor_operation/gpu/device/tensor_layout.hpp"

// TODO: remove this
#define K_LDS_LOAD_USE_OFFSET_TRANSFORM 0

namespace ck {
namespace tile_program {
namespace block {

template <bool QLoadOnce_>
struct BlockFmhaPipelineQXCustomPolicy;

template <>
struct BlockFmhaPipelineQXCustomPolicy</* QLoadOnce = */ true>
{
    static constexpr bool QLoadOnce = true;

    template <typename Problem>
    __host__ __device__ static constexpr ck::index_t GetSmemSizeQ()
    {
        return 0;
    }

    // TODO: GetAlignment*() currently didn't consider if need padding or not
    //       so in pipeline still need check padding requirement
    template <typename Problem>
    __host__ __device__ static constexpr auto GetAlignmentQ()
    {
        using BlockGemm       = remove_cvref_t<decltype(GetQKBlockGemm<Problem>())>;
        constexpr auto config = BlockGemm::Policy::template GetWarpGemmMWarpNWarp<Problem>();
        using WG              = remove_cvref_t<decltype(config.template At<0>())>;
        return WG::kK / WG::WarpGemmAttribute::Impl::kABKLane;
    }

    template <typename Problem, typename BlockGemm>
    __host__ __device__ static constexpr auto MakeQDramTileDistribution()
    {
        constexpr auto config   = BlockGemm::Policy::template GetWarpGemmMWarpNWarp<Problem>();
        using WG                = remove_cvref_t<decltype(config.template At<0>())>;
        constexpr index_t MWarp = config.template At<1>();

        constexpr index_t kMPerBlock = Problem::BlockFmhaShape::kM0;
        constexpr index_t kKPerBlock = Problem::BlockFmhaShape::kK0BlockLength;

        constexpr index_t K2 = WG::kK / WG::WarpGemmAttribute::Impl::kABKLane;
        constexpr index_t K1 = WG::WarpGemmAttribute::Impl::kABKLane;
        constexpr index_t K0 = kKPerBlock / (K1 * K2);

        constexpr index_t M2 = WG::WarpGemmAttribute::Impl::kAMLane;
        constexpr index_t M1 = MWarp;
        constexpr index_t M0 = kMPerBlock / (M2 * M1);

        return make_static_tile_distribution(
            StaticTileDistributionEncoding<Sequence<1>,
                                           Tuple<Sequence<M0, M1, M2>, Sequence<K0, K1, K2>>,
                                           Tuple<Sequence<1>, Sequence<2, 1>>,
                                           Tuple<Sequence<1>, Sequence<1, 2>>,
                                           Sequence<1, 2, 2>,
                                           Sequence<0, 0, 2>>{});
    }

    template <typename Problem>
    __host__ __device__ static constexpr auto GetQKBlockGemm()
    {
        using BlockGemmProblem =
            BlockGemmPipelineProblem<typename Problem::QDataType,
                                     typename Problem::KDataType,
                                     typename Problem::SaccDataType,
                                     Problem::kBlockSize,
                                     TileGemmShape<Problem::BlockFmhaShape::kM0,
                                                   Problem::BlockFmhaShape::kN0,
                                                   Problem::BlockFmhaShape::kK0>>;

        constexpr auto warp_gemm = []() {
            if constexpr(is_same_v<typename Problem::QDataType, half_t> &&
                         is_same_v<typename Problem::KDataType, half_t> &&
                         is_same_v<typename Problem::SaccDataType, float>)
            {
                return warp::WarpGemmMfmaF16F16F32M16N16K32SwizzleBTransposedCDistribution{};
            }
            else if constexpr(is_same_v<typename Problem::QDataType, bhalf_t> &&
                              is_same_v<typename Problem::KDataType, bhalf_t> &&
                              is_same_v<typename Problem::SaccDataType, float>)
            {
                return warp::WarpGemmMfmaBf16Bf16F32M16N16K32SwizzleBTransposedCDistribution{};
            }
            else if constexpr(Problem::kIsFp8)
            {
                constexpr index_t swizzle_factor = 4; // TODO: hard coded here
                return warp::WarpGemmImpl<
                    warp::WarpGemmAtrributeMfmaIterateKAndTransposedCDistribution_SwizzleB<
                        warp::WarpGemmAttributeMfmaImpl_f32_32x32x16_f8_base<
                            typename Problem::QDataType,
                            typename Problem::KDataType>,
                        2,
                        swizzle_factor>>{};
            }
        }();

        using BlockGemmPolicy =
            BlockGemmARegBSmemCRegV2CustomPolicy<typename Problem::QDataType,
                                                 typename Problem::KDataType,
                                                 typename Problem::SaccDataType,
                                                 typename Problem::BlockFmhaShape::Gemm0BlockWarps,
                                                 decltype(warp_gemm)>;

        return BlockGemmARegBSmemCRegV2<BlockGemmProblem, BlockGemmPolicy>{};
    }
};

template <>
struct BlockFmhaPipelineQXCustomPolicy</* QLoadOnce = */ false>
{
    static constexpr bool QLoadOnce = false;

    template <typename Problem>
    __host__ __device__ static constexpr ck::index_t GetSmemSizeQ()
    {
        constexpr index_t lds_alignment = 16; // optional
        constexpr index_t q_smem_size =
            ck::math::integer_divide_ceil(
                sizeof(typename Problem::QDataType) *
                    MakeQLdsBlockDescriptor<Problem>().GetElementSpaceSize(),
                lds_alignment) *
            lds_alignment;
        return q_smem_size;
    }

    template <typename Problem>
    __host__ __device__ static constexpr auto GetAlignmentQ()
    {
        using QDataType = remove_cvref_t<typename Problem::QDataType>;
        return 16 / sizeof(QDataType);
    }

    template <typename Problem>
    __host__ __device__ static constexpr auto MakeQDramTileDistribution()
    {
        using QDataType = remove_cvref_t<typename Problem::QDataType>;

        constexpr index_t kBlockSize = Problem::kBlockSize;

        constexpr index_t kMPerBlock = Problem::BlockFmhaShape::kM0;
        constexpr index_t kKPerBlock = Problem::BlockFmhaShape::kK0;

        constexpr index_t K1 = 16 / sizeof(QDataType); // use dwordx4. TODO: change this
        constexpr index_t K0 = kKPerBlock / K1;
        constexpr index_t M2 = get_warp_size() / K0;
        constexpr index_t M1 = kBlockSize / get_warp_size();
        constexpr index_t M0 = kMPerBlock / (M2 * M1);

        return make_static_tile_distribution(
            StaticTileDistributionEncoding<Sequence<1>,
                                           Tuple<Sequence<M0, M1, M2>, Sequence<K0, K1>>,
                                           Tuple<Sequence<1>, Sequence<1, 2>>,
                                           Tuple<Sequence<1>, Sequence<2, 0>>,
                                           Sequence<1, 2>,
                                           Sequence<0, 1>>{});
    }

    // 3d + padding
    template <typename Problem>
    __host__ __device__ static constexpr auto MakeQLdsBlockDescriptor()
    {
        using QDataType = remove_cvref_t<typename Problem::QDataType>;

        constexpr index_t kMPerBlock = Problem::BlockFmhaShape::kM0;
        constexpr index_t kKPerBlock = Problem::BlockFmhaShape::kK0;
        constexpr index_t kKPack     = 16 / sizeof(QDataType);

        constexpr auto q_lds_block_desc_0 = make_naive_tensor_descriptor(
            make_tuple(Number<kKPerBlock / kKPack>{}, Number<kMPerBlock>{}, Number<kKPack>{}),
            make_tuple(Number<(kMPerBlock + 1) * kKPack>{}, Number<kKPack>{}, Number<1>{}),
            Number<8>{},
            Number<1>{});

        constexpr auto q_lds_block_desc = transform_tensor_descriptor(
            q_lds_block_desc_0,
            make_tuple(make_pass_through_transform(kMPerBlock),
                       make_merge_transform(make_tuple(kKPerBlock / kKPack, kKPack))),
            make_tuple(Sequence<1>{}, Sequence<0, 2>{}),
            make_tuple(Sequence<0>{}, Sequence<1>{}));

        return q_lds_block_desc;
    }

    template <typename Problem>
    __host__ __device__ static constexpr auto GetQKBlockGemm()
    {
        using BlockGemmProblem =
            BlockGemmPipelineProblem<typename Problem::QDataType,
                                     typename Problem::KDataType,
                                     typename Problem::SaccDataType,
                                     Problem::kBlockSize,
                                     TileGemmShape<Problem::BlockFmhaShape::kM0,
                                                   Problem::BlockFmhaShape::kN0,
                                                   Problem::BlockFmhaShape::kK0>>;

        constexpr auto warp_gemm = []() {
            if constexpr(is_same_v<typename Problem::QDataType, half_t> &&
                         is_same_v<typename Problem::KDataType, half_t> &&
                         is_same_v<typename Problem::SaccDataType, float>)
            {
                return warp::WarpGemmMfmaF16F16F32M16N16K32SwizzleBTransposedCDistribution{};
            }
            else if constexpr(is_same_v<typename Problem::QDataType, bhalf_t> &&
                              is_same_v<typename Problem::KDataType, bhalf_t> &&
                              is_same_v<typename Problem::SaccDataType, float>)
            {
                return warp::WarpGemmMfmaBf16Bf16F32M16N16K32SwizzleBTransposedCDistribution{};
            }
            else if constexpr(Problem::kIsFp8)
            {
                constexpr index_t swizzle_factor = 4; // TODO: hard coded here
                return warp::WarpGemmImpl<
                    warp::WarpGemmAtrributeMfmaIterateKAndTransposedCDistribution_SwizzleB<
                        warp::WarpGemmAttributeMfmaImpl_f32_32x32x16_f8_base<
                            typename Problem::QDataType,
                            typename Problem::KDataType>,
                        2,
                        swizzle_factor>>{};
            }
        }();

        using BlockGemmPolicy =
            BlockGemmASmemBSmemCRegV1CustomPolicy<typename Problem::QDataType,
                                                  typename Problem::KDataType,
                                                  typename Problem::SaccDataType,
                                                  typename Problem::BlockFmhaShape::Gemm0BlockWarps,
                                                  decltype(warp_gemm)>;

        return BlockGemmASmemBSmemCRegV1<BlockGemmProblem, BlockGemmPolicy>{};
    }
};

// This pipeline is qkv all located in LDS
template <bool QLoadOnce_,
          bool AsyncCopyK_,
          bool AsyncCopyV_,
          index_t NumPrefetchK_,
          index_t NumPrefetchV_>
struct BlockFmhaPipelineQXKSVSCustomPolicy : BlockFmhaPipelineQXCustomPolicy<QLoadOnce_>
{
    static constexpr bool AsyncCopyK = AsyncCopyK_;
    static constexpr bool AsyncCopyV = AsyncCopyV_; // TODO: this not supported yet

    static constexpr index_t NumPrefetchK = NumPrefetchK_;
    static constexpr index_t NumPrefetchV = NumPrefetchK_;

    using QXPolicy = BlockFmhaPipelineQXCustomPolicy<QLoadOnce_>;

    template <index_t k_prefetches_, index_t v_prefetches_, index_t k_loops_, index_t v_loops_>
    struct LdsBufferSequence
    {
        static constexpr auto Make()
        {
            return transform_sequences(
                [&](auto i) {
                    if(i < k_loops_)
                        return i % k_prefetches_;
                    return (i - k_loops_) % v_prefetches_;
                },
                typename arithmetic_sequence_gen<0, k_loops_ + v_loops_, 1>::type{});
        };

        using type = remove_cvref_t<decltype(Make())>;
    };
    // clang-format off
    template<> struct
    LdsBufferSequence<3, 3, 4, 4> { using type = Sequence<1, 2, 0, 1,   0, 1, 2, 0>; };

    template<> struct
    LdsBufferSequence<3, 3, 4, 2> { using type = Sequence<1, 2, 0, 1,   2, 0>; };

    template<> struct
    LdsBufferSequence<3, 3, 2, 4> { using type = Sequence<1, 2,         0, 1, 2, 0>; };

    template<> struct
    LdsBufferSequence<3, 3, 3, 3> { using type = Sequence<1, 2, 0,      1, 2, 0>; };

    template<> struct
    LdsBufferSequence<3, 3, 2, 2> { using type = Sequence<1, 2,         1, 0>;};
    // clang-format on

    template <typename Problem>
    __host__ __device__ static constexpr auto GetLdsBufferSequence()
    {
        using BlockFmhaShape = remove_cvref_t<typename Problem::BlockFmhaShape>;

        constexpr index_t kN0            = BlockFmhaShape::kN0;
        constexpr index_t kK0            = BlockFmhaShape::kK0;
        constexpr index_t kK1            = BlockFmhaShape::kK1;
        constexpr index_t kK0BlockLength = BlockFmhaShape::kK0BlockLength;

        constexpr index_t k0_loops = kK0BlockLength / kK0;
        constexpr index_t k1_loops = kN0 / kK1;

        return typename LdsBufferSequence<NumPrefetchK, NumPrefetchV, k0_loops, k1_loops>::type{};
    }

    template <typename Problem>
    __host__ __device__ static constexpr auto GetSmemKPackK()
    {
        // TODO: this is for 3d layout
        using KDataType = remove_cvref_t<typename Problem::KDataType>;
        return 16 / sizeof(KDataType);
    }

    template <typename Problem>
    __host__ __device__ static constexpr auto GetAlignmentK()
    {
        using KDataType = remove_cvref_t<typename Problem::KDataType>;
        if constexpr(AsyncCopyK)
        {
            return 4 / sizeof(KDataType);
        }
        else
        {
            return 16 / sizeof(KDataType);
        }
    }

    template <typename Problem>
    __host__ __device__ static constexpr auto GetSmemKPackV()
    {
        // TODO: this is for 3d layout
        using VDataType = remove_cvref_t<typename Problem::VDataType>;
        return 16 / sizeof(VDataType);
    }

    template <typename Problem>
    __host__ __device__ static constexpr auto GetAlignmentV()
    {
        using VLayout   = remove_cvref_t<typename Problem::BlockFmhaShape::VLayout>;
        using VDataType = remove_cvref_t<typename Problem::VDataType>;
        if constexpr(ck::is_same_v<VLayout, ck::tensor_layout::gemm::RowMajor>)
        {
            constexpr index_t kBlockSize   = Problem::kBlockSize;
            constexpr index_t kNPerBlock   = Problem::BlockFmhaShape::kN1;
            constexpr index_t kKPerBlock   = Problem::BlockFmhaShape::kK1;
            constexpr index_t total_pixels = kNPerBlock * kKPerBlock / kBlockSize;

            // TODO: not correct!
            if constexpr(total_pixels > 4)
                return 4;
            else
                return 2;
        }
        else
        {
            return 16 / sizeof(VDataType);
        }
    }

    template <typename Problem>
    __host__ __device__ static constexpr auto GetAlignmentBias()
    {
        using BlockGemm = remove_cvref_t<decltype(QXPolicy::template GetQKBlockGemm<Problem>())>;
        constexpr auto config = BlockGemm::Policy::template GetWarpGemmMWarpNWarp<Problem>();
        using WG              = remove_cvref_t<decltype(config.template At<0>())>;
        using CWarpDstr       = typename WG::CWarpDstr;
        constexpr auto vec =
            CWarpDstr{}.GetYs2DDescriptor().GetLengths().At(Number<CWarpDstr::NDimY - 1>{});
        return vec;
    }

    template <typename Problem>
    __host__ __device__ static constexpr auto GetAlignmentO()
    {
        using BlockGemm       = remove_cvref_t<decltype(GetKVBlockGemm<Problem>())>;
        constexpr auto config = BlockGemm::Policy::template GetWarpGemmMWarpNWarp<Problem>();
        using WG              = remove_cvref_t<decltype(config.template At<0>())>;
        using CWarpDstr       = typename WG::CWarpDstr;
        constexpr auto vec =
            CWarpDstr{}.GetYs2DDescriptor().GetLengths().At(Number<CWarpDstr::NDimY - 1>{});
        return vec;
    }

    template <typename Problem>
    __host__ __device__ static constexpr auto GetSingleSmemElementSpaceSize()
    {
        // this function assume K/V can share smem
        constexpr index_t SingleKSize = [&]() {
            if constexpr(!AsyncCopyK)
            {
                return MakeKLdsBlockDescriptor<Problem>().GetElementSpaceSize();
            }
            else
            {
                constexpr index_t kNPerBlock = Problem::BlockFmhaShape::kN0;
                constexpr index_t kKPerBlock = Problem::BlockFmhaShape::kK1;
                constexpr index_t NumWarps   = Problem::BlockFmhaShape::NumWarps;
                constexpr index_t warpSize   = ck::get_warp_size();

                constexpr index_t KPack   = GetSmemKPackK<Problem>(); // this is for lds
                constexpr index_t KVector = GetAlignmentK<Problem>(); // this is for global load
                constexpr index_t kPad    = KPack;

                static_assert(warpSize * KVector >= kKPerBlock &&
                              warpSize * KVector % kKPerBlock == 0);
                constexpr index_t LanesPerK  = kKPerBlock / KVector;
                constexpr index_t LaneGroups = warpSize / LanesPerK;
                constexpr index_t NumIssues  = kNPerBlock / (LaneGroups * NumWarps);

                return NumIssues * NumWarps * (warpSize * KVector + kPad);
            }
        }();

        constexpr index_t SingleVSize = [&]() {
            using VDataType                = remove_cvref_t<typename Problem::VDataType>;
            constexpr index_t Banks        = 32; // TODO: need change based on arch
            constexpr index_t PixelsPerRow = Banks * 4 / sizeof(VDataType);
            constexpr index_t kKPack       = GetSmemKPackK<Problem>();
            static_assert(PixelsPerRow % kKPack == 0);
            constexpr index_t NPerRow    = PixelsPerRow / kKPack;
            constexpr index_t kNPerBlock = Problem::BlockFmhaShape::kN1;
            constexpr index_t kKPerBlock = Problem::BlockFmhaShape::kK1;
            static_assert(kNPerBlock % NPerRow == 0);
            static_assert(kKPerBlock % kKPack == 0);

            return (kKPerBlock / kKPack) * (kNPerBlock / NPerRow) * (PixelsPerRow + kKPack);
        }();

        return math::max(SingleKSize, SingleVSize);
    }

    template <typename Problem, typename BlockGemm>
    __host__ __device__ static constexpr auto MakeQRegBlockDescriptor()
    {
        constexpr index_t kMPerBlock = Problem::BlockFmhaShape::kM0;
        constexpr index_t kKPerBlock = Problem::BlockFmhaShape::kK0BlockLength;

        constexpr auto config = BlockGemm::Policy::template GetWarpGemmMWarpNWarp<Problem>();

        using WG = remove_cvref_t<decltype(config.template At<0>())>;

        constexpr index_t MWarp = config.template At<1>();
        constexpr index_t NWarp = config.template At<2>();

        constexpr index_t MIterPerWarp = kMPerBlock / (MWarp * WG::kM);
        constexpr index_t KIterPerWarp = kKPerBlock / WG::kK;

        constexpr auto q_block_outer_dstr_encoding = StaticTileDistributionEncoding<
            Sequence<NWarp>,
            Tuple<Sequence<MIterPerWarp, MWarp>, Sequence<KIterPerWarp>>,
            Tuple<Sequence<1, 0>>,
            Tuple<Sequence<1, 0>>,
            Sequence<1, 2>,
            Sequence<0, 0>>{};

        constexpr auto q_block_dstr_encode = detail::make_embed_tile_distribution_encoding(
            q_block_outer_dstr_encoding, typename WG::AWarpDstrEncoding{});

        constexpr auto q_block_dstr = make_static_tile_distribution(q_block_dstr_encode);

        return q_block_dstr;
    }

    // TODO: this is used for non async copy desc. unify in the future
    template <typename Problem>
    __host__ __device__ static constexpr auto MakeKLdsBlockDescriptor()
    {
        constexpr index_t kNPerBlock = Problem::BlockFmhaShape::kN0;
        constexpr index_t kKPerBlock = Problem::BlockFmhaShape::kK1;
        constexpr index_t kKPack     = GetSmemKPackK<Problem>();

        constexpr auto k_lds_block_desc_0 = make_naive_tensor_descriptor(
            make_tuple(Number<kKPerBlock / kKPack>{}, Number<kNPerBlock>{}, Number<kKPack>{}),
            make_tuple(Number<(kNPerBlock + 1) * kKPack>{}, Number<kKPack>{}, Number<1>{}),
            Number<8>{},
            Number<1>{});

        constexpr auto k_lds_block_desc = transform_tensor_descriptor(
            k_lds_block_desc_0,
            make_tuple(make_pass_through_transform(kNPerBlock),
                       make_merge_transform(make_tuple(kKPerBlock / kKPack, kKPack))),
            make_tuple(Sequence<1>{}, Sequence<0, 2>{}),
            make_tuple(Sequence<0>{}, Sequence<1>{}));

        return k_lds_block_desc;
    }

    template <typename Problem, index_t IBuf = 0>
    __host__ __device__ static constexpr auto
        MakeKLdsStoreBlockDescriptor(Number<IBuf> = Number<0>{})
    {
        // K is always k-major, we use async-copy to load into LDS
        constexpr index_t kNPerBlock = Problem::BlockFmhaShape::kN0;
        constexpr index_t kKPerBlock = Problem::BlockFmhaShape::kK1;
        constexpr index_t kBlockSize = Problem::kBlockSize;
        constexpr index_t NumWarps   = Problem::BlockFmhaShape::NumWarps;
        constexpr index_t warpSize   = ck::get_warp_size();

        constexpr index_t KPack   = GetSmemKPackK<Problem>(); // this is for lds
        constexpr index_t KVector = GetAlignmentK<Problem>(); // this is for global load
        constexpr index_t kPad =
            KPack; // for async-copy, this pad is between warps. Optimize this for lds_read speed

        static_assert(warpSize * KVector >= kKPerBlock && warpSize * KVector % kKPerBlock == 0);
        constexpr index_t LanesPerK =
            kKPerBlock / KVector; // how many lane (within a wave) to load K
        constexpr index_t LaneGroups =
            warpSize /
            LanesPerK; // how many groups (within a wave), they may load different N, but same K
        constexpr index_t NumIssues = kNPerBlock / (LaneGroups * NumWarps);
        static_assert(NumIssues == kNPerBlock * kKPerBlock / (kBlockSize * KVector));

        constexpr auto k_lds_block_desc_0 = make_naive_tensor_descriptor_with_offset(
            make_tuple(Number<NumIssues>{},  // n0
                       Number<LaneGroups>{}, // n1
                       Number<NumWarps>{},   // n2
                       Number<LanesPerK>{},  // k0
                       Number<KVector>{}),   // k1
            make_tuple(Number<NumWarps*(warpSize * KVector + kPad)>{},
                       Number<kKPerBlock>{},
                       Number<warpSize * KVector + kPad>{},
                       Number<KVector>{},
                       Number<1>{}),
            Number<IBuf * GetSingleSmemElementSpaceSize<Problem>()>{},
            Number<KVector>{},
            Number<1>{});

        // TODO this layout is hard coded, and will be used in async copy buffer view load
        // in LDS the real layout is (bufs, N0, N2, N1*K0*K1)
        constexpr auto k_lds_block_desc_issues_warps_lanes = transform_tensor_descriptor(
            k_lds_block_desc_0,
            make_tuple(make_pass_through_transform(Number<NumIssues>{}),
                       make_pass_through_transform(Number<NumWarps>{}),
                       make_merge_transform(make_tuple(
                           Number<LaneGroups>{}, Number<LanesPerK>{}, Number<KVector>{}))),
            make_tuple(Sequence<0>{}, Sequence<2>{}, Sequence<1, 3, 4>{}),
            make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}));

        return k_lds_block_desc_issues_warps_lanes;
    }

#if K_LDS_LOAD_USE_OFFSET_TRANSFORM
    template <typename Problem, index_t IBuf = 0>
    __host__ __device__ static constexpr auto
        MakeKLdsLoadBlockDescriptor(Number<IBuf> = Number<0>{})
    {
        // K is always k-major, we use async-copy to load into LDS
        constexpr index_t kNPerBlock = Problem::BlockFmhaShape::kN0;
        constexpr index_t kKPerBlock = Problem::BlockFmhaShape::kK1;
        constexpr index_t kBlockSize = Problem::kBlockSize;
        constexpr index_t NumWarps   = Problem::BlockFmhaShape::NumWarps;
        constexpr index_t warpSize   = ck::get_warp_size();

        constexpr index_t KPack   = GetSmemKPackK<Problem>(); // this is for lds
        constexpr index_t KVector = GetAlignmentK<Problem>(); // this is for global load
        constexpr index_t kPad    = KPack; // for async-copy, this pad is between warps

        static_assert(warpSize * KVector >= kKPerBlock && warpSize * KVector % kKPerBlock == 0);
        constexpr index_t LanesPerK  = kKPerBlock / KVector; // within a wave
        constexpr index_t LaneGroups = warpSize / LanesPerK; // within a wave
        constexpr index_t NumIssues  = kNPerBlock / (LaneGroups * NumWarps);
        static_assert(NumIssues == kNPerBlock * kKPerBlock / (kBlockSize * KVector));

        constexpr auto k_lds_block_desc_0 = make_naive_tensor_descriptor_with_offset(
            make_tuple(Number<NumIssues>{},          // n0
                       Number<NumWarps>{},           // n2
                       Number<LaneGroups>{},         // n1
                       Number<kKPerBlock / KPack>{}, // k0
                       Number<KPack>{}),             // k1
            make_tuple(Number<NumWarps*(warpSize * KVector + kPad)>{},
                       Number<warpSize * KVector + kPad>{},
                       Number<kKPerBlock>{},
                       Number<KPack>{},
                       Number<1>{}),
            Number<IBuf * GetSingleSmemElementSpaceSize<Problem>()>{},
            Number<KPack>{},
            Number<1>{});

        constexpr auto k_lds_block_desc = transform_tensor_descriptor(
            k_lds_block_desc_0,
            make_tuple(
                make_merge_transform(
                    make_tuple(Number<NumIssues>{}, Number<LaneGroups>{}, Number<NumWarps>{})),
                make_merge_transform(make_tuple(Number<kKPerBlock / KPack>{}, Number<KPack>{}))),
            make_tuple(Sequence<0, 2, 1>{}, Sequence<3, 4>{}),
            make_tuple(Sequence<0>{}, Sequence<1>{}));

        return k_lds_block_desc;
    }
#else
    template <typename Problem>
    __host__ __device__ static constexpr auto MakeKLdsLoadBlockDescriptor()
    {
        // K is always k-major, we use async-copy to load into LDS
        constexpr index_t kNPerBlock = Problem::BlockFmhaShape::kN0;
        constexpr index_t kKPerBlock = Problem::BlockFmhaShape::kK1;
        constexpr index_t kBlockSize = Problem::kBlockSize;
        constexpr index_t NumWarps   = Problem::BlockFmhaShape::NumWarps;
        constexpr index_t warpSize   = ck::get_warp_size();

        constexpr index_t KPack   = GetSmemKPackK<Problem>(); // this is for lds
        constexpr index_t KVector = GetAlignmentK<Problem>(); // this is for global load
        constexpr index_t kPad    = KPack; // for async-copy, this pad is between warps

        static_assert(warpSize * KVector >= kKPerBlock && warpSize * KVector % kKPerBlock == 0);
        constexpr index_t LanesPerK  = kKPerBlock / KVector; // within a wave
        constexpr index_t LaneGroups = warpSize / LanesPerK; // within a wave
        constexpr index_t NumIssues  = kNPerBlock / (LaneGroups * NumWarps);
        static_assert(NumIssues == kNPerBlock * kKPerBlock / (kBlockSize * KVector));
        // constexpr index_t SingleKSize = NumIssues * NumWarps * (warpSize * KVector + kPad);
        // constexpr index_t SingleVSize = MakeVLdsBlockDescriptor<Problem>().GetElementSpaceSize();
        constexpr index_t BufferSize =
            GetSingleSmemElementSpaceSize<Problem>(); //  math::max(SingleKSize, SingleVSize);

        constexpr auto k_lds_block_desc_0 =
            make_naive_tensor_descriptor(make_tuple(Number<NumPrefetchK>{},       // num_buffers
                                                    Number<NumIssues>{},          // n0
                                                    Number<NumWarps>{},           // n2
                                                    Number<LaneGroups>{},         // n1
                                                    Number<kKPerBlock / KPack>{}, // k0
                                                    Number<KPack>{}),             // k1
                                         make_tuple(Number<BufferSize>{},
                                                    Number<NumWarps*(warpSize * KVector + kPad)>{},
                                                    Number<warpSize * KVector + kPad>{},
                                                    Number<kKPerBlock>{},
                                                    Number<KPack>{},
                                                    Number<1>{}),
                                         Number<KPack>{},
                                         Number<1>{});

        constexpr auto k_lds_block_desc = transform_tensor_descriptor(
            k_lds_block_desc_0,
            make_tuple(
                make_merge_transform(make_tuple(Number<NumPrefetchK>{},
                                                Number<NumIssues>{},
                                                Number<LaneGroups>{},
                                                Number<NumWarps>{})),
                make_merge_transform(make_tuple(Number<kKPerBlock / KPack>{}, Number<KPack>{}))),
            make_tuple(Sequence<0, 1, 3, 2>{}, Sequence<4, 5>{}),
            make_tuple(Sequence<0>{}, Sequence<1>{}));

        return k_lds_block_desc;
    }
#endif

    // 3d + padding
    template <typename Problem>
    __host__ __device__ static constexpr auto MakeVLdsBlockDescriptor()
    {
        using VDataType                = remove_cvref_t<typename Problem::VDataType>;
        constexpr index_t Banks        = 32; // TODO: need change based on arch
        constexpr index_t PixelsPerRow = Banks * 4 / sizeof(VDataType);
        constexpr index_t kKPack       = GetSmemKPackV<Problem>();
        static_assert(PixelsPerRow % kKPack == 0);
        constexpr index_t NPerRow    = PixelsPerRow / kKPack;
        constexpr index_t kNPerBlock = Problem::BlockFmhaShape::kN1;
        constexpr index_t kKPerBlock = Problem::BlockFmhaShape::kK1;
        static_assert(kNPerBlock % NPerRow == 0);
        static_assert(kKPerBlock % kKPack == 0);

        constexpr auto v_lds_block_desc_0 = make_naive_tensor_descriptor(
            make_tuple(Number<NumPrefetchV>{},
                       Number<kKPerBlock / kKPack>{},
                       Number<kNPerBlock / NPerRow>{},
                       Number<NPerRow>{},
                       Number<kKPack>{}),
            make_tuple(Number<GetSingleSmemElementSpaceSize<Problem>()>{},
                       Number<(kNPerBlock / NPerRow) * (PixelsPerRow + kKPack)>{},
                       Number<PixelsPerRow + kKPack>{},
                       Number<kKPack>{},
                       Number<1>{}),
            Number<kKPack>{},
            Number<1>{});

        constexpr auto v_lds_block_desc = transform_tensor_descriptor(
            v_lds_block_desc_0,
            make_tuple(
                make_merge_transform(make_tuple(
                    Number<NumPrefetchV>{}, Number<kNPerBlock / NPerRow>{}, Number<NPerRow>{})),
                make_merge_transform(make_tuple(Number<kKPerBlock / kKPack>{}, Number<kKPack>{}))),
            make_tuple(Sequence<0, 2, 3>{}, Sequence<1, 4>{}),
            make_tuple(Sequence<0>{}, Sequence<1>{}));

        return v_lds_block_desc;
    }

    template <typename Problem>
    __host__ __device__ static constexpr ck::index_t GetSmemSize()
    {
        // TODO: assume Q is in register
        // TODO: assume K/V has same data type
        constexpr index_t single_smem_size =
            GetSingleSmemElementSpaceSize<Problem>() * sizeof(typename Problem::KDataType);

        return QXPolicy::template GetSmemSizeQ<Problem>() +
               single_smem_size * math::max(NumPrefetchK, NumPrefetchV);
    }

    template <typename Problem>
    __host__ __device__ static constexpr auto MakeKDramTileDistribution()
    {
        if constexpr(!AsyncCopyK)
        {
            using KDataType = remove_cvref_t<typename Problem::KDataType>;

            constexpr index_t kBlockSize = Problem::kBlockSize;
            constexpr index_t kNPerBlock = Problem::BlockFmhaShape::kN0;
            constexpr index_t kKPerBlock = Problem::BlockFmhaShape::kK0;

            constexpr index_t K1 = 16 / sizeof(KDataType);
            constexpr index_t K0 = kKPerBlock / K1;
            constexpr index_t N2 = get_warp_size() / K0;
            constexpr index_t N1 = kBlockSize / get_warp_size();
            constexpr index_t N0 = kNPerBlock / (N2 * N1);

            return make_static_tile_distribution(
                StaticTileDistributionEncoding<Sequence<1>,
                                               Tuple<Sequence<N0, N1, N2>, Sequence<K0, K1>>,
                                               Tuple<Sequence<1>, Sequence<1, 2>>,
                                               Tuple<Sequence<1>, Sequence<2, 0>>,
                                               Sequence<1, 2>,
                                               Sequence<0, 1>>{});
        }
        else
        {
            constexpr index_t kNPerBlock = Problem::BlockFmhaShape::kN0;
            constexpr index_t kKPerBlock = Problem::BlockFmhaShape::kK1;
            constexpr index_t kBlockSize = Problem::kBlockSize;
            constexpr index_t NumWarps   = Problem::BlockFmhaShape::NumWarps;
            constexpr index_t warpSize   = ck::get_warp_size();

            constexpr index_t KVector = GetAlignmentK<Problem>(); // this is for global load

            static_assert(warpSize * KVector >= kKPerBlock && warpSize * KVector % kKPerBlock == 0);
            constexpr index_t LanesPerK  = kKPerBlock / KVector; // within a wave
            constexpr index_t LaneGroups = warpSize / LanesPerK; // within a wave
            constexpr index_t NumIssues  = kNPerBlock / (LaneGroups * NumWarps);
            static_assert(NumIssues == kNPerBlock * kKPerBlock / (kBlockSize * KVector));

            constexpr index_t N0 = NumIssues;
            constexpr index_t N1 = LaneGroups;
            constexpr index_t N2 = NumWarps;
            constexpr index_t K0 = LanesPerK;
            constexpr index_t K1 = KVector;

            return make_static_tile_distribution(
                StaticTileDistributionEncoding<Sequence<1>,
                                               Tuple<Sequence<N0, N1, N2>, Sequence<K0, K1>>,
                                               Tuple<Sequence<1>, Sequence<1, 2>>,
                                               Tuple<Sequence<2>, Sequence<1, 0>>,
                                               Sequence<1, 2>,
                                               Sequence<0, 1>>{});
        }
    }

    template <typename Problem>
    __device__ static constexpr auto MakeVDramTileDistribution()
    {
        using VDataType = remove_cvref_t<typename Problem::VDataType>;
        using VLayout   = remove_cvref_t<typename Problem::BlockFmhaShape::VLayout>;

        constexpr index_t kBlockSize = Problem::kBlockSize;
        constexpr index_t kNPerBlock = Problem::BlockFmhaShape::kN1;
        constexpr index_t kKPerBlock = Problem::BlockFmhaShape::kK1;

        if constexpr(ck::is_same_v<VLayout, ck::tensor_layout::gemm::RowMajor>)
        {
            constexpr index_t N1 = GetAlignmentV<Problem>();
            constexpr index_t N0 = kNPerBlock / N1; // P

            constexpr index_t total_pixels = kNPerBlock * kKPerBlock / kBlockSize;
            static_assert(total_pixels % N1 == 0); // TODO: this is not always true?
            constexpr index_t K3     = total_pixels / N1;
            constexpr index_t kKPack = GetSmemKPackV<Problem>();
            static_assert(kKPack % K3 == 0);
            constexpr index_t K2 = kKPack / K3; // TODO: this dimention could be outside single wave
            if constexpr(get_warp_size() % (K2 * N0) == 0)
            {
                constexpr index_t K1 = get_warp_size() / (K2 * N0);
                constexpr index_t K0 = kBlockSize / get_warp_size();
                static_assert(kKPerBlock == K0 * K1 * K2 * K3);
                return make_static_tile_distribution(
                    StaticTileDistributionEncoding<
                        Sequence<1>,
                        Tuple<Sequence<N0, N1>, Sequence<K0, K1, K2, K3>>,
                        Tuple<Sequence<2>, Sequence<2, 1, 2>>,
                        Tuple<Sequence<0>, Sequence<1, 0, 2>>,
                        Sequence<2, 1>,
                        Sequence<3, 1>>{});
            }
            else
            {
                constexpr index_t K1   = (K2 * N0) / get_warp_size();
                constexpr index_t K2_m = K2 / K1;
                constexpr index_t K0   = kBlockSize / get_warp_size() / K1;
                static_assert(kKPerBlock == K0 * K1 * K2_m * K3);
                return make_static_tile_distribution(
                    StaticTileDistributionEncoding<
                        Sequence<1>,
                        Tuple<Sequence<N0, N1>, Sequence<K0, K1, K2_m, K3>>,
                        Tuple<Sequence<2, 2>, Sequence<1, 2>>,
                        Tuple<Sequence<0, 1>, Sequence<0, 2>>,
                        Sequence<2, 1>,
                        Sequence<3, 1>>{});
            }
        }
        else
        {
            constexpr index_t K1 = 16 / sizeof(VDataType);
            constexpr index_t K0 = kKPerBlock / K1;
            constexpr index_t N2 = get_warp_size() / K0;
            constexpr index_t N1 = kBlockSize / get_warp_size();
            constexpr index_t N0 = kNPerBlock / (N2 * N1);
            static_assert(N0 != 0);

            return make_static_tile_distribution(
                StaticTileDistributionEncoding<Sequence<1>,
                                               Tuple<Sequence<N0, N1, N2>, Sequence<K0, K1>>,
                                               Tuple<Sequence<1>, Sequence<1, 2>>,
                                               Tuple<Sequence<1>, Sequence<2, 0>>,
                                               Sequence<1, 2>,
                                               Sequence<0, 1>>{});
        }
    }

    template <typename Problem, typename BlockGemm>
    __host__ __device__ static constexpr auto MakeBiasDramTileDistribution()
    {
        constexpr index_t MPerBlock = Problem::BlockFmhaShape::kM0;
        constexpr index_t NPerBlock = Problem::BlockFmhaShape::kN0;

        constexpr auto config = BlockGemm::Policy::template GetWarpGemmMWarpNWarp<Problem>();
        using WG              = remove_cvref_t<decltype(config.template At<0>())>;

        constexpr index_t MWarp = config.template At<1>();
        constexpr index_t NWarp = config.template At<2>();

        constexpr index_t MIterPerWarp = MPerBlock / (MWarp * WG::kM);
        constexpr index_t NIterPerWarp = NPerBlock / (NWarp * WG::kN);

        // Construct C-Block-Tensor
        constexpr auto c_block_outer_dstr_encoding = StaticTileDistributionEncoding<
            Sequence<>,
            Tuple<Sequence<MIterPerWarp, MWarp>, Sequence<NIterPerWarp, NWarp>>,
            Tuple<Sequence<1, 2>>,
            Tuple<Sequence<1, 1>>,
            Sequence<1, 2>,
            Sequence<0, 0>>{};

        constexpr auto c_block_dstr_encode = detail::make_embed_tile_distribution_encoding(
            c_block_outer_dstr_encoding, typename WG::CWarpDstrEncoding{});

        constexpr auto c_block_dstr = make_static_tile_distribution(c_block_dstr_encode);

        return c_block_dstr;
    }

    template <typename Problem>
    __host__ __device__ static constexpr auto MakeShuffledVRegBlockDescriptor()
    {
        // This descriptor only used when V layout is seqlen * hdim
        using VLayout = remove_cvref_t<typename Problem::BlockFmhaShape::VLayout>;
        static_assert(ck::is_same_v<VLayout, ck::tensor_layout::gemm::RowMajor>);
        constexpr index_t kBlockSize = Problem::kBlockSize;
        constexpr index_t kNPerBlock = Problem::BlockFmhaShape::kN1;
        constexpr index_t kKPerBlock = Problem::BlockFmhaShape::kK1;

        constexpr index_t N1           = GetAlignmentV<Problem>();
        constexpr index_t N0           = kNPerBlock / N1;
        constexpr index_t total_pixels = kNPerBlock * kKPerBlock / kBlockSize;
        static_assert(total_pixels % N1 == 0); // TODO: this is not always true?
        constexpr index_t K3     = total_pixels / N1;
        constexpr index_t kKPack = GetSmemKPackV<Problem>();
        static_assert(kKPack % K3 == 0);
        constexpr index_t K2 = kKPack / K3; // TODO: this dimention could be outside single wave
        if constexpr(get_warp_size() % (K2 * N0) == 0)
        {
            constexpr index_t K1 = get_warp_size() / (K2 * N0);
            constexpr index_t K0 = kBlockSize / get_warp_size();

            return make_static_tile_distribution(
                StaticTileDistributionEncoding<Sequence<1>,
                                               Tuple<Sequence<N0, N1>, Sequence<K0, K1, K2, K3>>,
                                               Tuple<Sequence<2>, Sequence<2, 1, 2>>,
                                               Tuple<Sequence<0>, Sequence<1, 0, 2>>,
                                               Sequence<1, 2>,
                                               Sequence<1, 3>>{});
        }
        else
        {
            constexpr index_t K1   = (K2 * N0) / get_warp_size();
            constexpr index_t K2_m = K2 / K1;
            constexpr index_t K0   = kBlockSize / get_warp_size() / K1;
            static_assert(kKPerBlock == K0 * K1 * K2_m * K3);
            return make_static_tile_distribution(
                StaticTileDistributionEncoding<Sequence<1>,
                                               Tuple<Sequence<N0, N1>, Sequence<K0, K1, K2_m, K3>>,
                                               Tuple<Sequence<2, 2>, Sequence<1, 2>>,
                                               Tuple<Sequence<0, 1>, Sequence<0, 2>>,
                                               Sequence<1, 2>,
                                               Sequence<1, 3>>{});
        }
    }

    template <typename Problem>
    __host__ __device__ static constexpr auto GetKVBlockGemm()
    {
        using BlockGemmProblem =
            BlockGemmPipelineProblem<typename Problem::PDataType,
                                     typename Problem::VDataType,
                                     typename Problem::OaccDataType,
                                     Problem::kBlockSize,
                                     TileGemmShape<Problem::BlockFmhaShape::kM0,
                                                   Problem::BlockFmhaShape::kN1,
                                                   Problem::BlockFmhaShape::kK1>>;

        auto warp_gemm = [&]() {
            if constexpr(Problem::kIsFp8)
            {
                return warp::WarpGemmImpl<
                    warp::WarpGemmAtrributeMfmaIterateKAndTransposedCDistribution<
                        warp::WarpGemmAttributeMfmaImpl_f32_32x32x16_f8_base<
                            typename Problem::PDataType,
                            typename Problem::VDataType>,
                        2>>{};
                // return
                // warp::WarpGemmImpl<warp::WarpGemmAtrributeMfmaTransposedCDistribution_SwizzleB<
                //         warp::WarpGemmAttributeMfmaImpl_f32_32x32x16_f8_base<typename
                //         Problem::PDataType, typename Problem::VDataType>>>{};
            }
            else
            {
                return ck::tile_program::warp::WarpGemmMfmaDispatcher<
                    typename Problem::PDataType,
                    typename Problem::VDataType,
                    typename Problem::OaccDataType,
                    Problem::BlockFmhaShape::Gemm1WarpTile::At(Number<0>{}),
                    Problem::BlockFmhaShape::Gemm1WarpTile::At(Number<1>{}),
                    Problem::BlockFmhaShape::Gemm1WarpTile::At(Number<2>{}),
                    true>{};
            }
        }();

        using WarpGemm = remove_cvref_t<decltype(warp_gemm)>;

        using BlockGemmPolicy =
            BlockGemmARegBSmemCRegV2CustomPolicy<typename Problem::PDataType,
                                                 typename Problem::VDataType,
                                                 typename Problem::OaccDataType,
                                                 typename Problem::BlockFmhaShape::Gemm1BlockWarps,
                                                 WarpGemm>;
        return BlockGemmARegBSmemCRegV2<BlockGemmProblem, BlockGemmPolicy>{};
    }
};

} // namespace block
} // namespace tile_program
} // namespace ck
