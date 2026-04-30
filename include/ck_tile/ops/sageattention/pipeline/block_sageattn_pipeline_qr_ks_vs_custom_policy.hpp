// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/ops/common/tensor_layout.hpp"
#include "ck_tile/ops/gemm/block/block_gemm_problem.hpp"
#include "ck_tile/ops/gemm/pipeline/gemm_pipeline_problem.hpp"
#include "ck_tile/ops/gemm/pipeline/tile_gemm_shape.hpp"
#include "ck_tile/ops/gemm/warp/warp_gemm.hpp"
#include "ck_tile/ops/gemm/warp/warp_gemm_dispatcher.hpp"
#include "ck_tile/ops/gemm/block/block_gemm_areg_bsmem_creg_v1_custom_policy.hpp"
#include "ck_tile/ops/gemm/block/block_gemm_areg_bsmem_creg_v2_custom_policy.hpp"
#include "ck_tile/ops/gemm/block/block_gemm_areg_bsmem_creg_v2.hpp"
#include "ck_tile/ops/gemm/block/block_gemm_areg_bsmem_creg_one_warp_v1.hpp"

namespace ck_tile {

template <typename T>
CK_TILE_HOST_DEVICE static constexpr index_t GetPackedSize()
{
    return numeric_traits<remove_cvref_t<T>>::PackedSize;
}

template <typename T>
CK_TILE_HOST_DEVICE static constexpr index_t GetLogicalVectorSize(index_t bytes)
{
    return (bytes / sizeof(remove_cvref_t<T>)) * GetPackedSize<T>();
}

template <typename Problem>
using SageAttnQKGemmQDataType =
    std::conditional_t<is_packed_type_v<remove_cvref_t<typename Problem::QDataType>>,
                       fp8_t,
                       remove_cvref_t<typename Problem::QDataType>>;

template <typename Problem>
using SageAttnQKGemmKDataType =
    std::conditional_t<is_packed_type_v<remove_cvref_t<typename Problem::KDataType>>,
                       fp8_t,
                       remove_cvref_t<typename Problem::KDataType>>;

template <bool QLoadOnce_>
struct BlockSageAttnPipelineQRCustomPolicy;

template <>
struct BlockSageAttnPipelineQRCustomPolicy</* QLoadOnce = */ true>
{
    static constexpr bool QLoadOnce = true;

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr ck_tile::index_t GetSmemSizeQ()
    {
        return 0;
    }

    // TODO: GetAlignment*() currently didn't consider if need padding or not
    //       so in pipeline still need check padding requirement
    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto GetAlignmentQ()
    {
        constexpr index_t MaxVectorSize = GetLogicalVectorSize<typename Problem::QDataType>(16);

        using BlockGemm       = remove_cvref_t<decltype(GetQKBlockGemm<Problem>())>;
        constexpr auto config = BlockGemm::Policy::template GetWarpGemmMWarpNWarp<Problem>();
        using WG              = remove_cvref_t<decltype(config.template at<0>())>;

        return min(MaxVectorSize, WG::kK / WG::WarpGemmAttribute::Impl::kABKLane);
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto MakeQRegTileDistribution()
    {
        using BlockGemm = remove_cvref_t<decltype(GetQKBlockGemm<Problem>())>;

        return BlockGemm::template MakeABlockTileDistribution<
            Problem::BlockSageAttnShape::kM0,
            Problem::BlockSageAttnShape::kSubQKHeaddim>();
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto GetQKBlockGemm()
    {
        using QKGemmQDataType = SageAttnQKGemmQDataType<Problem>;
        using QKGemmKDataType = SageAttnQKGemmKDataType<Problem>;
        // int8 MFMA accumulates to int32, but SaccDataType is float for softmax
        using GemmAccDataType =
            std::conditional_t<(std::is_same_v<QKGemmQDataType, int8_t> ||
                                std::is_same_v<QKGemmQDataType, signed char>) &&
                                   (std::is_same_v<QKGemmKDataType, int8_t> ||
                                    std::is_same_v<QKGemmKDataType, signed char>),
                               int32_t,
                               typename Problem::SaccDataType>;

        using GemmProblem =
            BlockGemmProblem<QKGemmQDataType,
                             QKGemmKDataType,
                             GemmAccDataType,
                             Problem::kNumGemm0Warps * get_warp_size(),
                             TileGemmShape<sequence<Problem::BlockSageAttnShape::kM0,
                                                    Problem::BlockSageAttnShape::kN0,
                                                    Problem::BlockSageAttnShape::kK0>,
                                           typename Problem::BlockSageAttnShape::Gemm0BlockWarps,
                                           typename Problem::BlockSageAttnShape::Gemm0WarpTile>>;

        constexpr auto warp_gemm = []() {
            if constexpr(get_warp_size() == 64 && std::is_same_v<QKGemmQDataType, fp8_t> &&
                         std::is_same_v<QKGemmKDataType, fp8_t> &&
                         std::is_same_v<typename Problem::SaccDataType, float>)
            {
                static_assert(Problem::BlockSageAttnShape::Gemm0WarpTile::at(number<0>{}) == 32);
                static_assert(Problem::BlockSageAttnShape::Gemm0WarpTile::at(number<1>{}) == 32);
                static_assert(Problem::BlockSageAttnShape::Gemm0WarpTile::at(number<2>{}) == 32);

                // TODO: hard coded here. Otherwise, it produces incorrect results
                constexpr index_t swizzle_factor = 4;
                return WarpGemmMfmaFp8Fp8F32M32N32K32SwizzleBTransposedCDistribution<
                    swizzle_factor>{};
            }
            else if constexpr(get_warp_size() == 64 &&
                              (std::is_same_v<QKGemmQDataType, int8_t> ||
                               std::is_same_v<QKGemmQDataType, signed char>) &&
                              (std::is_same_v<QKGemmKDataType, int8_t> ||
                               std::is_same_v<QKGemmKDataType, signed char>))
            {
                static_assert(Problem::BlockSageAttnShape::Gemm0WarpTile::at(number<0>{}) == 32);
                static_assert(Problem::BlockSageAttnShape::Gemm0WarpTile::at(number<1>{}) == 32);
                static_assert(Problem::BlockSageAttnShape::Gemm0WarpTile::at(number<2>{}) == 32);

                // Use special int8 MFMA with K iteration (similar to FP8)
                constexpr index_t swizzle_factor = 4;
                return WarpGemmMfmaI8I8I32M32N32K32SwizzleBTransposedCDistribution<
                    swizzle_factor>{};
            }
            else
            {
                constexpr bool SwizzleA =
                    Problem::BlockSageAttnShape::Gemm0WarpTile::at(number<0>{}) == 32;
                return WarpGemmDispatcher<
                    QKGemmQDataType,
                    QKGemmKDataType,
                    GemmAccDataType,
                    Problem::BlockSageAttnShape::Gemm0WarpTile::at(number<0>{}),
                    Problem::BlockSageAttnShape::Gemm0WarpTile::at(number<1>{}),
                    Problem::BlockSageAttnShape::Gemm0WarpTile::at(number<2>{}),
                    true, // TransposeC
                    SwizzleA>{};
            }
        }();

        using BlockGemmPolicy = BlockGemmARegBSmemCRegV2CustomPolicy<
            QKGemmQDataType,
            QKGemmKDataType,
            GemmAccDataType,
            typename Problem::BlockSageAttnShape::Gemm0BlockWarps,
            decltype(warp_gemm)>;

        if constexpr(1 < Problem::kNumGemm0Warps)
            return BlockGemmARegBSmemCRegV2<GemmProblem, BlockGemmPolicy>{};
        else
            return BlockGemmARegBSmemCRegOneWarpV1<GemmProblem, BlockGemmPolicy>{};
    }
};

// This pipeline is qkv all located in LDS
template <bool QLoadOnce_, bool AsyncCopy_, index_t NumPrefetchK_, index_t NumPrefetchV_>
struct BlockSageAttnPipelineQRKSVSCustomPolicy : BlockSageAttnPipelineQRCustomPolicy<QLoadOnce_>
{
    static constexpr bool AsyncCopy = AsyncCopy_;

    static constexpr index_t NumPrefetchK = NumPrefetchK_;
    static constexpr index_t NumPrefetchV = NumPrefetchV_;

    static constexpr index_t NumKVLdsBuffers = max(NumPrefetchK, NumPrefetchV);

    using QXPolicy = BlockSageAttnPipelineQRCustomPolicy<QLoadOnce_>;

    template <index_t k_prefetches_, index_t v_prefetches_, index_t k_loops_, index_t v_loops_>
    struct LdsBufferSequence
    {
        static constexpr index_t num_lds_buffers_ = max(k_prefetches_, v_prefetches_);
        static constexpr index_t ceil_ = ((v_loops_ - 1) / num_lds_buffers_) * num_lds_buffers_;

        // for qr_ks_vs_async, the Lds buffer assigned to last gemm_1 iteration of V should not
        // overlap with the Lds buffers used by first two gemm_0 iterations of K
        static constexpr auto Make()
        {
            // ensure v_loop_-1 is assigned to num_lds_buffers-1
            return transform_sequences(
                [&](auto i) {
                    if(i < k_loops_)
                        return i % num_lds_buffers_;
                    else
                        return ((num_lds_buffers_ - 1) + (i - k_loops_ + ceil_ - (v_loops_ - 1))) %
                               num_lds_buffers_;
                },
                typename arithmetic_sequence_gen<0, k_loops_ + v_loops_, 1>::type{});
        };

        using type = remove_cvref_t<decltype(Make())>;
    };

    // clang-format off
    template<> struct
    LdsBufferSequence<3, 3, 4, 4> { using type = sequence<1, 2, 0, 1,   0, 1, 2, 0>; };

    template<> struct
    LdsBufferSequence<3, 3, 4, 2> { using type = sequence<1, 2, 0, 1,   2, 0>; };

    template<> struct
    LdsBufferSequence<3, 3, 2, 4> { using type = sequence<1, 2,         0, 1, 2, 0>; };

    template<> struct
    LdsBufferSequence<3, 3, 3, 3> { using type = sequence<1, 2, 0,      1, 2, 0>; };

    template<> struct
    LdsBufferSequence<3, 3, 3, 4> { using type = sequence<1, 2, 0,      0, 1, 2, 0>; };

    template<> struct
    LdsBufferSequence<3, 3, 2, 2> { using type = sequence<1, 2,         1, 0>;};
    // clang-format on

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto GetLdsBufferSequence()
    {
        using BlockSageAttnShape = remove_cvref_t<typename Problem::BlockSageAttnShape>;

        constexpr index_t kN0        = BlockSageAttnShape::kN0;
        constexpr index_t kK0        = BlockSageAttnShape::kK0;
        constexpr index_t kK1        = BlockSageAttnShape::kK1;
        constexpr index_t kQKHeaddim = BlockSageAttnShape::kQKHeaddim;

        constexpr index_t k0_loops = kQKHeaddim / kK0;
        constexpr index_t k1_loops = kN0 / kK1;

        return typename LdsBufferSequence<NumPrefetchK, NumPrefetchV, k0_loops, k1_loops>::type{};
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto GetSmemKPackK()
    {
        // TODO: this is for 3d layout
        using KDataType = SageAttnQKGemmKDataType<Problem>;
        return GetLogicalVectorSize<KDataType>(16);
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto GetAlignmentK()
    {
        using KDataType = remove_cvref_t<typename Problem::KDataType>;
        if constexpr(AsyncCopy)
        {
#if defined(__gfx950__)
            constexpr index_t MaxLoadSizeInBytes = 4 * 4; // dwordx4
#else
            constexpr index_t MaxLoadSizeInBytes = 4; // dword
#endif

            return GetLogicalVectorSize<KDataType>(MaxLoadSizeInBytes);
        }
        else
        {
            constexpr index_t kBlockSize = Problem::kBlockSize;
            constexpr index_t kNPerBlock = Problem::BlockSageAttnShape::kN0;
            constexpr index_t kKPerBlock = Problem::BlockSageAttnShape::kK0;

            constexpr index_t MaxVectorSize = GetLogicalVectorSize<KDataType>(16);
            constexpr index_t ElemPerThread = (kNPerBlock * kKPerBlock) / kBlockSize;

            return min(MaxVectorSize, ElemPerThread);
        }
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto GetSmemKPackV()
    {
        // TODO: this is for 3d layout
        using VDataType                = remove_cvref_t<typename Problem::VDataType>;
        constexpr index_t kBlockSize   = Problem::kBlockSize;
        constexpr index_t kNPerBlock   = Problem::BlockSageAttnShape::kN1;
        constexpr index_t kKPerBlock   = Problem::BlockSageAttnShape::kK1;
        constexpr index_t total_pixels = kNPerBlock * kKPerBlock / kBlockSize;
        constexpr index_t kMaxVecLoad =
            min(total_pixels, static_cast<index_t>(16 / sizeof(VDataType)));

        return kMaxVecLoad;
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto GetAlignmentV()
    {
        using VLayout   = remove_cvref_t<typename Problem::BlockSageAttnShape::VLayout>;
        using VDataType = remove_cvref_t<typename Problem::VDataType>;
        constexpr index_t kBlockSize   = Problem::kBlockSize;
        constexpr index_t kNPerBlock   = Problem::BlockSageAttnShape::kN1;
        constexpr index_t kKPerBlock   = Problem::BlockSageAttnShape::kK1;
        constexpr index_t total_pixels = kNPerBlock * kKPerBlock / kBlockSize;
        constexpr index_t kMaxVecLoad =
            min(total_pixels, static_cast<index_t>(16 / sizeof(VDataType)));

        if constexpr(std::is_same_v<VLayout, ck_tile::tensor_layout::gemm::RowMajor>)
        {
            constexpr index_t kMinVecLoad = 4 / sizeof(VDataType);

            constexpr index_t kVecLoad = ((total_pixels / kMaxVecLoad) >= kMinVecLoad)
                                             ? kMaxVecLoad
                                             : (total_pixels / kMinVecLoad);

            return kVecLoad;
        }
        else
        {
            return kMaxVecLoad;
        }
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto GetAlignmentO()
    {
        using BlockGemm       = remove_cvref_t<decltype(GetKVBlockGemm<Problem>())>;
        constexpr auto config = BlockGemm::Policy::template GetWarpGemmMWarpNWarp<Problem>();
        using WG              = remove_cvref_t<decltype(config.template at<0>())>;

        constexpr index_t MaxVectorSize = 16 / sizeof(typename Problem::ODataType);
        return min(MaxVectorSize, WG::WarpGemmAttribute::Impl::kCM1PerLane);
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto GetSingleSmemElementSpaceSize()
    {
        // this function assume K/V can share smem
        constexpr index_t SingleKSize = [&]() {
            if constexpr(!AsyncCopy)
            {
                return MakeKLdsBlockDescriptor<Problem>().get_element_space_size();
            }
            else
            {
                constexpr index_t kNPerBlock = Problem::BlockSageAttnShape::kN0;
                constexpr index_t kKPerBlock = Problem::BlockSageAttnShape::kK0;
                constexpr index_t NumWarps   = Problem::BlockSageAttnShape::NumWarps;
                constexpr index_t WarpSize   = ck_tile::get_warp_size();

                constexpr index_t KPack   = GetSmemKPackK<Problem>(); // this is for lds
                constexpr index_t KVector = GetAlignmentK<Problem>(); // this is for global load
                constexpr index_t kPad    = KPack;

                static_assert(WarpSize * KVector >= kKPerBlock &&
                              WarpSize * KVector % kKPerBlock == 0);
                constexpr index_t LanesPerK  = kKPerBlock / KVector;
                constexpr index_t LaneGroups = WarpSize / LanesPerK;
                constexpr index_t NumIssues  = kNPerBlock / (LaneGroups * NumWarps);

                return NumIssues * NumWarps * (WarpSize * KVector + kPad);
            }
        }();

        constexpr index_t SingleVSize = [&]() {
            using VDataType                = remove_cvref_t<typename Problem::VDataType>;
            constexpr index_t Banks        = get_n_lds_banks();
            constexpr index_t PixelsPerRow = Banks * 4 / sizeof(VDataType);
            constexpr index_t kKPack       = GetSmemKPackV<Problem>();
            static_assert(PixelsPerRow % kKPack == 0);
            constexpr index_t NPerRow    = PixelsPerRow / kKPack;
            constexpr index_t kNPerBlock = Problem::BlockSageAttnShape::kN1;
            constexpr index_t kKPerBlock = Problem::BlockSageAttnShape::kK1;
            static_assert(kNPerBlock % NPerRow == 0);
            static_assert(kKPerBlock % kKPack == 0);

            return (kKPerBlock / kKPack) * (kNPerBlock / NPerRow) * (PixelsPerRow + kKPack);
        }();

        return max(SingleKSize, SingleVSize);
    }

    // TODO: this is used for non async copy desc. unify in the future
    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto MakeKLdsBlockDescriptor()
    {
        constexpr index_t kNPerBlock = Problem::BlockSageAttnShape::kN0;
        constexpr index_t kKPerBlock = Problem::BlockSageAttnShape::kK0;
        constexpr index_t kKPack     = GetSmemKPackK<Problem>();

        constexpr auto k_lds_block_desc_0 = make_naive_tensor_descriptor(
            make_tuple(number<kKPerBlock / kKPack>{}, number<kNPerBlock>{}, number<kKPack>{}),
            make_tuple(number<(kNPerBlock + 1) * kKPack>{}, number<kKPack>{}, number<1>{}),
            number<kKPack>{},
            number<1>{});

        constexpr auto k_lds_block_desc = transform_tensor_descriptor(
            k_lds_block_desc_0,
            make_tuple(
                make_pass_through_transform(number<kNPerBlock>{}),
                make_merge_transform(make_tuple(number<kKPerBlock / kKPack>{}, number<kKPack>{}))),
            make_tuple(sequence<1>{}, sequence<0, 2>{}),
            make_tuple(sequence<0>{}, sequence<1>{}));

        return k_lds_block_desc;
    }

    template <typename Problem, index_t IBuf = 0>
    CK_TILE_HOST_DEVICE static constexpr auto
    MakeKLdsStoreBlockDescriptor(number<IBuf> = number<0>{})
    {
        // K is always k-major, we use async-copy to load into LDS
        constexpr index_t kNPerBlock = Problem::BlockSageAttnShape::kN0;
        constexpr index_t kKPerBlock = Problem::BlockSageAttnShape::kK0;
        constexpr index_t kBlockSize = Problem::kBlockSize;
        constexpr index_t NumWarps   = Problem::BlockSageAttnShape::NumWarps;
        constexpr index_t WarpSize   = ck_tile::get_warp_size();

        constexpr index_t KPack   = GetSmemKPackK<Problem>(); // this is for lds
        constexpr index_t KVector = GetAlignmentK<Problem>(); // this is for global load
        constexpr index_t kPad =
            KPack; // for async-copy, this pad is between warps. Optimize this for lds_read speed

        static_assert(WarpSize * KVector >= kKPerBlock && WarpSize * KVector % kKPerBlock == 0);
        constexpr index_t LanesPerK =
            kKPerBlock / KVector; // how many lane (within a wave) to load K
        constexpr index_t LaneGroups =
            WarpSize /
            LanesPerK; // how many groups (within a wave), they may load different N, but same K
        constexpr index_t NumIssues = kNPerBlock / (LaneGroups * NumWarps);
        static_assert(NumIssues == kNPerBlock * kKPerBlock / (kBlockSize * KVector));

        constexpr auto k_lds_block_desc_0 = make_naive_tensor_descriptor_with_offset(
            make_tuple(number<NumIssues>{},  // n0
                       number<LaneGroups>{}, // n1
                       number<NumWarps>{},   // n2
                       number<LanesPerK>{},  // k0
                       number<KVector>{}),   // k1
            make_tuple(number<NumWarps*(WarpSize * KVector + kPad)>{},
                       number<kKPerBlock>{},
                       number<WarpSize * KVector + kPad>{},
                       number<KVector>{},
                       number<1>{}),
            number<IBuf * GetSingleSmemElementSpaceSize<Problem>()>{},
            number<KVector>{},
            number<1>{});

        // TODO this layout is hard coded, and will be used in async copy buffer view load
        // in LDS the real layout is (bufs, N0, N2, N1*K0*K1)
        constexpr auto k_lds_block_desc_issues_warps_lanes = transform_tensor_descriptor(
            k_lds_block_desc_0,
            make_tuple(make_pass_through_transform(number<NumIssues>{}),
                       make_pass_through_transform(number<NumWarps>{}),
                       make_merge_transform(make_tuple(
                           number<LaneGroups>{}, number<LanesPerK>{}, number<KVector>{}))),
            make_tuple(sequence<0>{}, sequence<2>{}, sequence<1, 3, 4>{}),
            make_tuple(sequence<0>{}, sequence<1>{}, sequence<2>{}));

        return k_lds_block_desc_issues_warps_lanes;
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto MakeKLdsLoadBlockDescriptor()
    {
        // K is always k-major, we use async-copy to load into LDS
        constexpr index_t kNPerBlock = Problem::BlockSageAttnShape::kN0;
        constexpr index_t kKPerBlock = Problem::BlockSageAttnShape::kK0;
        constexpr index_t kBlockSize = Problem::kBlockSize;
        constexpr index_t NumWarps   = Problem::BlockSageAttnShape::NumWarps;
        constexpr index_t WarpSize   = ck_tile::get_warp_size();

        constexpr index_t KPack   = GetSmemKPackK<Problem>(); // this is for lds
        constexpr index_t KVector = GetAlignmentK<Problem>(); // this is for global load
        constexpr index_t kPad    = KPack; // for async-copy, this pad is between warps

        static_assert(WarpSize * KVector >= kKPerBlock && WarpSize * KVector % kKPerBlock == 0);
        constexpr index_t LanesPerK  = kKPerBlock / KVector; // within a wave
        constexpr index_t LaneGroups = WarpSize / LanesPerK; // within a wave
        constexpr index_t NumIssues  = kNPerBlock / (LaneGroups * NumWarps);
        static_assert(NumIssues == kNPerBlock * kKPerBlock / (kBlockSize * KVector));
        // constexpr index_t SingleKSize = NumIssues * NumWarps * (WarpSize * KVector + kPad);
        // constexpr index_t SingleVSize =
        // MakeVLdsBlockDescriptor<Problem>().get_element_space_size();
        constexpr index_t BufferSize =
            GetSingleSmemElementSpaceSize<Problem>(); //  max(SingleKSize, SingleVSize);

        constexpr auto k_lds_block_desc_0 =
            make_naive_tensor_descriptor(make_tuple(number<NumKVLdsBuffers>{},    // num_buffers
                                                    number<NumIssues>{},          // n0
                                                    number<NumWarps>{},           // n2
                                                    number<LaneGroups>{},         // n1
                                                    number<kKPerBlock / KPack>{}, // k0
                                                    number<KPack>{}),             // k1
                                         make_tuple(number<BufferSize>{},
                                                    number<NumWarps*(WarpSize * KVector + kPad)>{},
                                                    number<WarpSize * KVector + kPad>{},
                                                    number<kKPerBlock>{},
                                                    number<KPack>{},
                                                    number<1>{}),
                                         number<KPack>{},
                                         number<1>{});

        constexpr auto k_lds_block_desc = transform_tensor_descriptor(
            k_lds_block_desc_0,
            make_tuple(
                make_merge_transform(make_tuple(number<NumKVLdsBuffers>{},
                                                number<NumIssues>{},
                                                number<LaneGroups>{},
                                                number<NumWarps>{})),
                make_merge_transform(make_tuple(number<kKPerBlock / KPack>{}, number<KPack>{}))),
            make_tuple(sequence<0, 1, 3, 2>{}, sequence<4, 5>{}),
            make_tuple(sequence<0>{}, sequence<1>{}));

        return k_lds_block_desc;
    }

    // 3d + padding
    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto MakeVLdsBlockDescriptor()
    {
        using VDataType                = remove_cvref_t<typename Problem::VDataType>;
        constexpr index_t Banks        = get_n_lds_banks();
        constexpr index_t PixelsPerRow = Banks * 4 / sizeof(VDataType);
        constexpr index_t kKPack       = GetSmemKPackV<Problem>();
        static_assert(PixelsPerRow % kKPack == 0);
        constexpr index_t NPerRow    = PixelsPerRow / kKPack;
        constexpr index_t kNPerBlock = Problem::BlockSageAttnShape::kN1;
        constexpr index_t kKPerBlock = Problem::BlockSageAttnShape::kK1;
        static_assert(kNPerBlock % NPerRow == 0);
        static_assert(kKPerBlock % kKPack == 0);

        constexpr auto v_lds_block_desc_0 = make_naive_tensor_descriptor(
            make_tuple(number<NumKVLdsBuffers>{},
                       number<kKPerBlock / kKPack>{},
                       number<kNPerBlock / NPerRow>{},
                       number<NPerRow>{},
                       number<kKPack>{}),
            make_tuple(number<GetSingleSmemElementSpaceSize<Problem>()>{},
                       number<(kNPerBlock / NPerRow) * (PixelsPerRow + kKPack)>{},
                       number<PixelsPerRow + kKPack>{},
                       number<kKPack>{},
                       number<1>{}),
            number<kKPack>{},
            number<1>{});

        constexpr auto v_lds_block_desc = transform_tensor_descriptor(
            v_lds_block_desc_0,
            make_tuple(
                make_merge_transform(make_tuple(
                    number<NumKVLdsBuffers>{}, number<kNPerBlock / NPerRow>{}, number<NPerRow>{})),
                make_merge_transform(make_tuple(number<kKPerBlock / kKPack>{}, number<kKPack>{}))),
            make_tuple(sequence<0, 2, 3>{}, sequence<1, 4>{}),
            make_tuple(sequence<0>{}, sequence<1>{}));

        return v_lds_block_desc;
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr ck_tile::index_t GetSmemSizeKV()
    {
        // TODO: assume Q is in register
        // TODO: assume K and V share smem buffers
        using KLdsDataType = SageAttnQKGemmKDataType<Problem>;
        constexpr index_t single_smem_size =
            GetSingleSmemElementSpaceSize<Problem>() * sizeof(KLdsDataType);

        return QXPolicy::template GetSmemSizeQ<Problem>() + single_smem_size * NumKVLdsBuffers;
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr ck_tile::index_t GetSmemSize()
    {
        return GetSmemSizeKV<Problem>();
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto MakeKDramTileDistribution()
    {
        if constexpr(!AsyncCopy)
        {
            using KDataType = remove_cvref_t<typename Problem::KDataType>;

            constexpr index_t kBlockSize = Problem::kBlockSize;
            constexpr index_t kNPerBlock = Problem::BlockSageAttnShape::kN0;
            constexpr index_t kKPerBlock = Problem::BlockSageAttnShape::kK0;

            constexpr index_t MaxVectorSize = GetLogicalVectorSize<KDataType>(16);
            constexpr index_t ElemPerThread = (kNPerBlock * kKPerBlock) / kBlockSize;

            constexpr index_t K1 = min(MaxVectorSize, ElemPerThread);
            constexpr index_t K0 = kKPerBlock / K1;
            constexpr index_t N2 = get_warp_size() / K0;
            constexpr index_t N1 = kBlockSize / get_warp_size();
            constexpr index_t N0 = kNPerBlock / (N2 * N1);

            return make_static_tile_distribution(
                tile_distribution_encoding<sequence<1>,
                                           tuple<sequence<N0, N1, N2>, sequence<K0, K1>>,
                                           tuple<sequence<1>, sequence<1, 2>>,
                                           tuple<sequence<1>, sequence<2, 0>>,
                                           sequence<1, 2>,
                                           sequence<0, 1>>{});
        }
        else
        {
            constexpr index_t kNPerBlock = Problem::BlockSageAttnShape::kN0;
            constexpr index_t kKPerBlock = Problem::BlockSageAttnShape::kK0;
            constexpr index_t kBlockSize = Problem::kBlockSize;
            constexpr index_t NumWarps   = Problem::BlockSageAttnShape::NumWarps;
            constexpr index_t WarpSize   = ck_tile::get_warp_size();

            constexpr index_t KVector = GetAlignmentK<Problem>(); // this is for global load

            static_assert(WarpSize * KVector >= kKPerBlock && WarpSize * KVector % kKPerBlock == 0);
            constexpr index_t LanesPerK  = kKPerBlock / KVector; // within a wave
            constexpr index_t LaneGroups = WarpSize / LanesPerK; // within a wave
            constexpr index_t NumIssues  = kNPerBlock / (LaneGroups * NumWarps);
            static_assert(NumIssues == kNPerBlock * kKPerBlock / (kBlockSize * KVector));

            constexpr index_t N0 = NumIssues;
            constexpr index_t N1 = LaneGroups;
            constexpr index_t N2 = NumWarps;
            constexpr index_t K0 = LanesPerK;
            constexpr index_t K1 = KVector;

            return make_static_tile_distribution(
                tile_distribution_encoding<sequence<1>,
                                           tuple<sequence<N0, N1, N2>, sequence<K0, K1>>,
                                           tuple<sequence<1>, sequence<1, 2>>,
                                           tuple<sequence<2>, sequence<1, 0>>,
                                           sequence<1, 2>,
                                           sequence<0, 1>>{});
        }
    }

    template <typename Problem>
    CK_TILE_DEVICE static constexpr auto MakeVDramTileDistribution()
    {
        using VLayout = remove_cvref_t<typename Problem::BlockSageAttnShape::VLayout>;

        constexpr index_t kBlockSize = Problem::kBlockSize;
        constexpr index_t kNPerBlock = Problem::BlockSageAttnShape::kN1;
        constexpr index_t kKPerBlock = Problem::BlockSageAttnShape::kK1;

        if constexpr(std::is_same_v<VLayout, ck_tile::tensor_layout::gemm::RowMajor>)
        {
            constexpr index_t N1 = GetAlignmentV<Problem>();
            constexpr index_t N0 = kNPerBlock / N1; // P

            constexpr index_t total_pixels = kNPerBlock * kKPerBlock / kBlockSize;
            constexpr index_t kKPack       = GetSmemKPackV<Problem>();
            constexpr index_t K3           = total_pixels / N1;
            constexpr index_t K2 = kKPack / K3; // TODO: this dimention could be outside single wave
            if constexpr(total_pixels % N1 != 0 || kKPack % K3 != 0) // if K2 or K3 is not divisible
            {
                static_assert(kNPerBlock % 16 == 0);
                constexpr index_t kNPack = kNPerBlock % 32 == 0 ? 32 : 16;
                constexpr index_t K0     = kBlockSize / get_warp_size();
                constexpr index_t N2     = 2;
                constexpr index_t N1_m   = kNPack / N2;
                constexpr index_t N0_m   = kNPerBlock / kNPack;
                constexpr index_t K1     = get_warp_size() / N1_m;
                constexpr index_t K2_m   = kKPerBlock / K1 / K0;
                return make_static_tile_distribution(
                    tile_distribution_encoding<
                        sequence<1>,
                        tuple<sequence<N0_m, N1_m, N2>, sequence<K0, K1, K2_m>>,
                        tuple<sequence<2>, sequence<2, 1>>, // K0, K1 N0
                        tuple<sequence<0>, sequence<1, 1>>,
                        sequence<1, 2, 1>, // N0 K2 N2
                        sequence<0, 2, 2>>{});
            }
            else if constexpr(get_warp_size() % (K2 * N0) == 0)
            {
                constexpr index_t K1 = get_warp_size() / (K2 * N0);
                constexpr index_t K0 = kBlockSize / get_warp_size();
                static_assert(kKPerBlock == K0 * K1 * K2 * K3);
                return make_static_tile_distribution(
                    tile_distribution_encoding<sequence<1>,
                                               tuple<sequence<N0, N1>, sequence<K0, K1, K2, K3>>,
                                               tuple<sequence<2>, sequence<2, 1, 2>>,
                                               tuple<sequence<0>, sequence<1, 0, 2>>,
                                               sequence<2, 1>,
                                               sequence<3, 1>>{});
            }
            else
            {
                constexpr index_t K1   = (K2 * N0) / get_warp_size();
                constexpr index_t K2_m = K2 / K1;
                constexpr index_t K0   = kBlockSize / get_warp_size() / K1;
                static_assert(kKPerBlock == K0 * K1 * K2_m * K3);
                return make_static_tile_distribution(
                    tile_distribution_encoding<sequence<1>,
                                               tuple<sequence<N0, N1>, sequence<K0, K1, K2_m, K3>>,
                                               tuple<sequence<2, 2>, sequence<1, 2>>,
                                               tuple<sequence<0, 1>, sequence<0, 2>>,
                                               sequence<2, 1>,
                                               sequence<3, 1>>{});
            }
        }
        else
        {
            constexpr index_t K1 = GetAlignmentV<Problem>();
            constexpr index_t K0 = kKPerBlock / K1;
            constexpr index_t N2 = get_warp_size() / K0;
            constexpr index_t N1 = kBlockSize / get_warp_size();
            static_assert(N2 != 0, "N2 is zero, which will lead to a division by zero error.");
            static_assert(N1 != 0, "N1 is zero, which will lead to a division by zero error.");
            constexpr index_t N0 = kNPerBlock / (N2 * N1);
            static_assert(N0 != 0);

            constexpr auto dstr = make_static_tile_distribution(
                tile_distribution_encoding<sequence<1>,
                                           tuple<sequence<N0, N1, N2>, sequence<K0, K1>>,
                                           tuple<sequence<1>, sequence<1, 2>>, // N1, N2 K0
                                           tuple<sequence<1>, sequence<2, 0>>,
                                           sequence<1, 2>, // N0 K1
                                           sequence<0, 1>>{});
            if constexpr(container_reduce(dstr.get_lengths(), std::multiplies<index_t>{}, 1) ==
                         kNPerBlock * kKPerBlock)
            {
                return dstr;
            }
            else
            {
                static_assert(kKPerBlock % 16 == 0);
                constexpr index_t kKPerIter = kKPerBlock % 32 == 0 ? 32 : 16;
                constexpr index_t K0_m      = kKPerBlock / kKPerIter;
                constexpr index_t K2        = 2;
                constexpr index_t K1_m      = kKPerIter / K2;
                constexpr index_t N2_m      = get_warp_size() / K1_m;
                constexpr index_t N0_m      = kNPerBlock / (N2_m * N1);
                constexpr auto dstr_m       = make_static_tile_distribution(
                    tile_distribution_encoding<
                              sequence<1>,
                              tuple<sequence<N0_m, N1, N2_m>, sequence<K0_m, K1_m, K2>>,
                              tuple<sequence<1>, sequence<1, 2>>, // N1, N2 K1
                              tuple<sequence<1>, sequence<2, 1>>,
                              sequence<2, 1, 2>, // K0 N0 K2
                              sequence<0, 0, 2>>{});
                static_assert(container_reduce(dstr_m.get_lengths(),
                                               std::multiplies<index_t>{},
                                               1) == kNPerBlock * kKPerBlock);
                return dstr_m;
            }
        }
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto MakeShuffledVRegBlockDescriptor()
    {
        // This descriptor only used when V layout is seqlen * hdim
        using VLayout = remove_cvref_t<typename Problem::BlockSageAttnShape::VLayout>;
        static_assert(std::is_same_v<VLayout, ck_tile::tensor_layout::gemm::RowMajor>);
        constexpr index_t kBlockSize = Problem::kBlockSize;
        constexpr index_t kNPerBlock = Problem::BlockSageAttnShape::kN1;
        constexpr index_t kKPerBlock = Problem::BlockSageAttnShape::kK1;

        constexpr index_t N1           = GetAlignmentV<Problem>();
        constexpr index_t N0           = kNPerBlock / N1;
        constexpr index_t total_pixels = kNPerBlock * kKPerBlock / kBlockSize;
        constexpr index_t K3           = total_pixels / N1;
        constexpr index_t kKPack       = GetSmemKPackV<Problem>();
        constexpr index_t K2 = kKPack / K3; // TODO: this dimention could be outside single wave
        if constexpr(total_pixels % N1 != 0 || kKPack % K3 != 0) // if K2 or K3 is not divisible
        {
            static_assert(kNPerBlock % 16 == 0);
            constexpr index_t kNPack = kNPerBlock % 32 == 0 ? 32 : 16;
            constexpr index_t K0     = kBlockSize / get_warp_size();
            constexpr index_t N2     = 2;
            constexpr index_t N1_m   = kNPack / N2;
            constexpr index_t N0_m   = kNPerBlock / kNPack;
            constexpr index_t K1     = get_warp_size() / N1_m;
            constexpr index_t K2_m   = kKPerBlock / K1 / K0;
            return make_static_tile_distribution(
                tile_distribution_encoding<sequence<1>,
                                           tuple<sequence<N0_m, N1_m, N2>, sequence<K0, K1, K2_m>>,
                                           tuple<sequence<2>, sequence<2, 1>>, // K0, K1 N0
                                           tuple<sequence<0>, sequence<1, 1>>,
                                           sequence<1, 1, 2>, // N0 K2 <-> N2
                                           sequence<0, 2, 2>>{});
        }
        else if constexpr(get_warp_size() % (K2 * N0) == 0)
        {
            constexpr index_t K1 = get_warp_size() / (K2 * N0);
            constexpr index_t K0 = kBlockSize / get_warp_size();

            return make_static_tile_distribution(
                tile_distribution_encoding<sequence<1>,
                                           tuple<sequence<N0, N1>, sequence<K0, K1, K2, K3>>,
                                           tuple<sequence<2>, sequence<2, 1, 2>>,
                                           tuple<sequence<0>, sequence<1, 0, 2>>,
                                           sequence<1, 2>,
                                           sequence<1, 3>>{});
        }
        else
        {
            constexpr index_t K1   = (K2 * N0) / get_warp_size();
            constexpr index_t K2_m = K2 / K1;
            constexpr index_t K0   = kBlockSize / get_warp_size() / K1;
            static_assert(kKPerBlock == K0 * K1 * K2_m * K3);
            return make_static_tile_distribution(
                tile_distribution_encoding<sequence<1>,
                                           tuple<sequence<N0, N1>, sequence<K0, K1, K2_m, K3>>,
                                           tuple<sequence<2, 2>, sequence<1, 2>>,
                                           tuple<sequence<0, 1>, sequence<0, 2>>,
                                           sequence<1, 2>,
                                           sequence<1, 3>>{});
        }
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto GetKVBlockGemm()
    {
        using GemmProblem =
            BlockGemmProblem<typename Problem::PDataType,
                             typename Problem::VDataType,
                             typename Problem::OaccDataType,
                             Problem::kNumGemm1Warps * get_warp_size(),
                             TileGemmShape<sequence<Problem::BlockSageAttnShape::kM0,
                                                    Problem::BlockSageAttnShape::kN1,
                                                    Problem::BlockSageAttnShape::kK1>,
                                           typename Problem::BlockSageAttnShape::Gemm1BlockWarps,
                                           typename Problem::BlockSageAttnShape::Gemm1WarpTile>>;

        auto warp_gemm = [&]() {
            if constexpr(get_warp_size() == 64 &&
                         std::is_same_v<typename Problem::PDataType, fp8_t> &&
                         std::is_same_v<typename Problem::VDataType, fp8_t> &&
                         std::is_same_v<typename Problem::OaccDataType, float>)
            {
                static_assert(Problem::BlockSageAttnShape::Gemm1WarpTile::at(number<0>{}) == 32);
                static_assert(Problem::BlockSageAttnShape::Gemm1WarpTile::at(number<1>{}) == 32);
                static_assert(Problem::BlockSageAttnShape::Gemm1WarpTile::at(number<2>{}) == 32);

                return WarpGemmMfmaFp8Fp8F32M32N32K32SwizzleBTransposedCDistribution<>{};
            }
            else
            {
                return WarpGemmDispatcher<
                    typename Problem::PDataType,
                    typename Problem::VDataType,
                    typename Problem::OaccDataType,
                    Problem::BlockSageAttnShape::Gemm1WarpTile::at(number<0>{}),
                    Problem::BlockSageAttnShape::Gemm1WarpTile::at(number<1>{}),
                    Problem::BlockSageAttnShape::Gemm1WarpTile::at(number<2>{}),
                    true>{};
            }
        }();

        using WarpGemm = remove_cvref_t<decltype(warp_gemm)>;

        using BlockGemmPolicy = BlockGemmARegBSmemCRegV2CustomPolicy<
            typename Problem::PDataType,
            typename Problem::VDataType,
            typename Problem::OaccDataType,
            typename Problem::BlockSageAttnShape::Gemm1BlockWarps,
            WarpGemm>;
        return BlockGemmARegBSmemCRegV2<GemmProblem, BlockGemmPolicy>{};
    }
};

} // namespace ck_tile
