// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2024, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/ops/fmha/pipeline/block_fmha_pipeline_qx_ks_vs_custom_policy.hpp"
#include "ck_tile/ops/gemm/block/block_gemm_asmem_bsmem_creg_v1_custom_policy.hpp"
#include "ck_tile/ops/gemm/block/block_gemm_asmem_bsmem_creg_v1.hpp"

namespace ck_tile {

// This pipeline is qkv all located in LDS
struct BlockFmhaFwdDecodePipelineQRKSVSDefaultPolicy
    : BlockFmhaPipelineQXKSVSCustomPolicy</* QLoadOnce = */ true,
                                          /* AsyncCopy = */ false,
                                          /* NumPrefetchK = */ 1,
                                          /* NumPrefetchV = */ 1>
{
    using BasePolicy = BlockFmhaPipelineQXKSVSCustomPolicy</* QLoadOnce = */ true,
                                                           /* AsyncCopy = */ false,
                                                           /* NumPrefetchK = */ 1,
                                                           /* NumPrefetchV = */ 1>;

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto GetAlignmentQ()
    {
        constexpr index_t kBlockSize = Problem::kBlockSize;
        constexpr index_t kMPerBlock = Problem::BlockFmhaShape::kM0;
        constexpr index_t kKPerBlock = Problem::BlockFmhaShape::kSubQKHeaddim;

        constexpr index_t MaxVectorSize = 16 / sizeof(typename Problem::QDataType);

        // this should align with MakeQDramTileDistribution()
        constexpr index_t ElemPerThread = (kMPerBlock * kKPerBlock) / kBlockSize;
        static_assert(0 < ElemPerThread);
        return min(ElemPerThread, MaxVectorSize);
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto GetAlignmentOacc()
    {
        using OaccDataType = remove_cvref_t<typename Problem::OaccDataType>;

        return static_cast<index_t>(16 / sizeof(OaccDataType));
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto MakeQDramTileDistribution()
    {
        using QDataType = remove_cvref_t<typename Problem::QDataType>;

        constexpr index_t kBlockSize = Problem::kBlockSize;
        constexpr index_t kMPerBlock = Problem::BlockFmhaShape::kM0;
        constexpr index_t kKPerBlock = Problem::BlockFmhaShape::kK0;

        constexpr index_t MaxVectorSize = 16 / sizeof(QDataType);

        constexpr index_t ElemPerThread = (kMPerBlock * kKPerBlock) / kBlockSize;
        static_assert(0 < ElemPerThread);
        constexpr index_t kMaxVecLoad = [&](){
            // Try dwordx4
            if constexpr (ElemPerThread % MaxVectorSize == 0){
                return MaxVectorSize;
            }
            // Try dwordx2
            else if constexpr (ElemPerThread % (MaxVectorSize / 2) == 0){
                return MaxVectorSize / 2;
            }
            // Try dword
            else if constexpr (ElemPerThread % (MaxVectorSize / 4) == 0){
                return MaxVectorSize / 4;
            }
            else{
                return 1;
            }
        }();

        constexpr index_t KPerThread     = kMaxVecLoad;
        // if false, we can not distribute the Ps thread evenly over Hs.
        constexpr bool KThreadEven = (ElemPerThread/KPerThread) % 2 == 0;
        constexpr index_t KThreads       = [&](){
            if constexpr (KThreadEven){
                return kKPerBlock / KPerThread;
            }
            else{
                // We move the odd factor to multiple instruction issue
                return kKPerBlock/ElemPerThread;
            }
        }();

        constexpr index_t KRepeat        = kKPerBlock / KThreads / KPerThread;
        static_assert(KRepeat == ElemPerThread/kMaxVecLoad);
        constexpr index_t MThreadPerWarp = get_warp_size() / KThreads;
        constexpr index_t NumWarps       = kBlockSize / get_warp_size();
        constexpr index_t MPerThread     = kMPerBlock / (MThreadPerWarp * NumWarps);

        return make_static_tile_distribution(
            tile_distribution_encoding<sequence<1>,
                                       tuple<sequence<MPerThread, NumWarps, MThreadPerWarp>,
                                             sequence<KRepeat, KThreads, KPerThread>>,
                                       tuple<sequence<1>, sequence<1, 2>>,
                                       tuple<sequence<1>, sequence<2, 1>>,
                                       sequence<2, 1, 2>,
                                       sequence<0, 0, 2>>{});
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto MakeKDramTileDistribution()
    {
        using KDataType = remove_cvref_t<typename Problem::KDataType>;

        constexpr index_t kBlockSize = Problem::kBlockSize;
        constexpr index_t kNPerBlock = Problem::BlockFmhaShape::kN0;
        constexpr index_t kKPerBlock = Problem::BlockFmhaShape::kK0;

        constexpr index_t MaxVectorSize = 16 / sizeof(KDataType);
        constexpr index_t ElemPerThread = (kNPerBlock * kKPerBlock) / kBlockSize;
        constexpr index_t kMaxVecLoad = [&](){
            // Try dwordx4
            if constexpr (ElemPerThread % MaxVectorSize == 0){
                return MaxVectorSize;
            }
            // Try dwordx2
            else if constexpr (ElemPerThread % (MaxVectorSize / 2) == 0){
                return MaxVectorSize / 2;
            }
            // Try dword
            else if constexpr (ElemPerThread % (MaxVectorSize / 4) == 0){
                return MaxVectorSize / 4;
            }
            else{
                return 1;
            }
        }();

        constexpr index_t KPerThread     = kMaxVecLoad;
        // if false, we can not distribute the Ps thread evenly over Hs.
        constexpr bool KThreadEven = (ElemPerThread/KPerThread) % 2 == 0;
        constexpr index_t KThreads       = [&](){
            if constexpr (KThreadEven){
                return kKPerBlock / KPerThread;
            }
            else{
                // We move the odd factor to multiple instruction issue
                return kKPerBlock/ElemPerThread;
            }}();

        constexpr index_t KRepeat        = kKPerBlock / KThreads / KPerThread;
        static_assert(KRepeat == ElemPerThread/kMaxVecLoad);
        constexpr index_t NThreadPerWarp = get_warp_size() / KThreads;
        constexpr index_t NumWarps       = kBlockSize / get_warp_size();
        constexpr index_t NPerThread     = kNPerBlock / (NThreadPerWarp * NumWarps);

        return make_static_tile_distribution(
            tile_distribution_encoding<sequence<1>,
                                       tuple<sequence<NPerThread, NumWarps, NThreadPerWarp>,
                                             sequence<KRepeat, KThreads, KPerThread>>,
                                       tuple<sequence<1>, sequence<1, 2>>,
                                       tuple<sequence<1>, sequence<2, 1>>,
                                       sequence<2, 1, 2>,
                                       sequence<0, 0, 2>>{});
    }

    template <typename Problem>
    CK_TILE_DEVICE static constexpr auto MakeVDramTileDistribution()
    {
        // constexpr index_t kBlockSize = Problem::kBlockSize;
        // constexpr index_t kNPerBlock = Problem::BlockFmhaShape::kN1;
        // constexpr index_t kKPerBlock = Problem::BlockFmhaShape::kK1;

        constexpr index_t N2 = 4; // Y 4 
        constexpr index_t N1 = 4; // P 2
        constexpr index_t N0 = 3; // Y 2
        
        constexpr index_t K2 = 2;  // Y
        constexpr index_t K1 = 16; // P
        constexpr index_t K0 = 1;  // P

        return make_static_tile_distribution(
            tile_distribution_encoding<sequence<1>,
                                       tuple<sequence<N0, N1, N2>, sequence<K0, K1, K2>>,
                                       tuple<sequence<2>, sequence<2, 1>>,
                                       tuple<sequence<0>, sequence<1, 1>>,
                                       sequence<1, 2, 1>,
                                       sequence<0, 2, 2>>{});
#if 0
        constexpr index_t kBlockSize = Problem::kBlockSize;
        constexpr index_t kNPerBlock = Problem::BlockFmhaShape::kN1;
        constexpr index_t kKPerBlock = Problem::BlockFmhaShape::kK1;

        constexpr index_t N1 = GetAlignmentV<Problem>(); // 8 
        constexpr index_t N0 = kNPerBlock / N1; // P     // 2

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
#endif
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto MakeShuffledVRegBlockDescriptor()
    {
        constexpr index_t N2 = 4; // Y 4 
        constexpr index_t N1 = 4; // P 2
        constexpr index_t N0 = 3; // Y 2
        
        constexpr index_t K2 = 2;  // Y
        constexpr index_t K1 = 16; // P
        constexpr index_t K0 = 1;  // P

        return make_static_tile_distribution(
            tile_distribution_encoding<sequence<1>,
                                       tuple<sequence<N0, N1, N2>, sequence<K0, K1, K2>>,
                                       tuple<sequence<2>, sequence<2, 1>>,
                                       tuple<sequence<0>, sequence<1, 1>>,
                                       sequence<1, 1, 2>,
                                       sequence<0, 2, 2>>{});
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto MakeQRegTileDistribution()
    {
        return BasePolicy::template MakeQRegTileDistribution<Problem>();
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto GetSmemKPackQ()
    {
        // TODO: this is for 3d layout
        using QDataType = remove_cvref_t<typename Problem::QDataType>;
        return static_cast<index_t>(16 / sizeof(QDataType));
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto MakeQLdsBlockDescriptor()
    {
        constexpr index_t kBlockSize = Problem::kBlockSize;
        constexpr index_t kMPerBlock = Problem::BlockFmhaShape::kM0;
        constexpr index_t kKPerBlock = Problem::BlockFmhaShape::kSubQKHeaddim;

        constexpr index_t ElemPerThread = (kMPerBlock * kKPerBlock) / kBlockSize;
        static_assert(0 < ElemPerThread);
        constexpr index_t kKPack = min(ElemPerThread, GetSmemKPackQ<Problem>());

        constexpr auto q_lds_block_desc_0 = make_naive_tensor_descriptor(
            make_tuple(number<kKPerBlock / kKPack>{}, number<kMPerBlock>{}, number<kKPack>{}),
            make_tuple(number<(kMPerBlock + 1) * kKPack>{}, number<kKPack>{}, number<1>{}),
            number<kKPack>{},
            number<1>{});

        constexpr auto q_lds_block_desc = transform_tensor_descriptor(
            q_lds_block_desc_0,
            make_tuple(
                make_pass_through_transform(number<kMPerBlock>{}),
                make_merge_transform(make_tuple(number<kKPerBlock / kKPack>{}, number<kKPack>{}))),
            make_tuple(sequence<1>{}, sequence<0, 2>{}),
            make_tuple(sequence<0>{}, sequence<1>{}));

        return q_lds_block_desc;
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto GetSmemNPackS()
    {
        using SDataType = remove_cvref_t<typename Problem::SaccDataType>;
        return static_cast<index_t>(16 / sizeof(SDataType));
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto MakeSLdsBlockDescriptor()
    {
        constexpr index_t kMPerBlock = Problem::BlockFmhaShape::kM0;
        constexpr index_t kNPerBlock = Problem::BlockFmhaShape::kN0;
        constexpr index_t kNPack     = GetSmemNPackS<Problem>();

        constexpr auto s_lds_block_desc_0 = make_naive_tensor_descriptor(
            make_tuple(number<kNPerBlock / kNPack>{}, number<kMPerBlock>{}, number<kNPack>{}),
            make_tuple(number<(kMPerBlock + 1) * kNPack>{}, number<kNPack>{}, number<1>{}),
            number<kNPack>{},
            number<1>{});

        constexpr auto s_lds_block_desc = transform_tensor_descriptor(
            s_lds_block_desc_0,
            make_tuple(
                make_pass_through_transform(number<kMPerBlock>{}),
                make_merge_transform(make_tuple(number<kNPerBlock / kNPack>{}, number<kNPack>{}))),
            make_tuple(sequence<1>{}, sequence<0, 2>{}),
            make_tuple(sequence<0>{}, sequence<1>{}));

        return s_lds_block_desc;
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto MakeSRegTileDistribution()
    {
        using BlockGemm = remove_cvref_t<decltype(GetKVBlockGemm<Problem>())>;

        constexpr auto config   = BlockGemm::Policy::template GetWarpGemmMWarpNWarp<Problem>();
        using WG                = remove_cvref_t<decltype(config.template at<0>())>;
        constexpr index_t MWarp = config.template at<1>();
        constexpr index_t NWarp = config.template at<2>();

        // static_assert(MWarp == 1, "Check failed!");

        constexpr index_t kMPerBlock = Problem::BlockFmhaShape::kM0;
        constexpr index_t kKPerBlock = Problem::BlockFmhaShape::kK1;
        constexpr index_t kTileK     = Problem::BlockFmhaShape::kN0;

        // K2 is equal to Impl::kABKPerLane * kKIterPerWarpGemm
        constexpr index_t K3 = WG::kK / WG::WarpGemmAttribute::Impl::kABKLane;
        constexpr index_t K2 = WG::WarpGemmAttribute::Impl::kABKLane;
        constexpr index_t K1 = kKPerBlock / (K2 * K3);
        constexpr index_t K0 = kTileK / kKPerBlock;
        constexpr index_t M2 = WG::WarpGemmAttribute::Impl::kAMLane;
        constexpr index_t M1 = MWarp;
        constexpr index_t M0 = kMPerBlock / (M2 * M1);

        constexpr auto s2_block_dstr_encoding =
            tile_distribution_encoding<sequence<NWarp>,
                                       tuple<sequence<M0, M1, M2>, sequence<K0, K1, K2, K3>>,
                                       tuple<sequence<1, 0>, sequence<2, 1>>,
                                       tuple<sequence<1, 0>, sequence<2, 2>>,
                                       sequence<1, 2, 2, 2>,
                                       sequence<0, 0, 1, 3>>{};

        constexpr auto s2_block_dstr = make_static_tile_distribution(s2_block_dstr_encoding);

        return s2_block_dstr;
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr ck_tile::index_t GetSmemSizeQ()
    {
        return MakeQLdsBlockDescriptor<Problem>().get_element_space_size() *
               sizeof(typename Problem::QDataType);
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr ck_tile::index_t GetSmemSizeK()
    {
        return MakeKLdsBlockDescriptor<Problem>().get_element_space_size() *
               sizeof(typename Problem::KDataType);
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr ck_tile::index_t GetSmemSizeV()
    {
        return MakeVLdsBlockDescriptor<Problem>().get_element_space_size() *
               sizeof(typename Problem::VDataType);
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr ck_tile::index_t GetSmemSizeS()
    {
        return MakeSLdsBlockDescriptor<Problem>().get_element_space_size() *
               sizeof(typename Problem::SaccDataType);
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr ck_tile::index_t GetSmemSize()
    {
        return max(GetSmemSizeQ<Problem>(), GetSmemSizeK<Problem>()) +
               max(GetSmemSizeV<Problem>(), GetSmemSizeS<Problem>());
    }
};

} // namespace ck_tile
