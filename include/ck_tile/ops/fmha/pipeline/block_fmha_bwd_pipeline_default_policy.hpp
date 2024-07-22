// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2024, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/ops/common/tensor_layout.hpp"
#include "ck_tile/ops/gemm/pipeline/block_gemm_pipeline_problem.hpp"
#include "ck_tile/ops/gemm/pipeline/tile_gemm_shape.hpp"
#include "ck_tile/ops/gemm/warp/warp_gemm.hpp"
#include "ck_tile/ops/gemm/warp/warp_gemm_dispatcher.hpp"
#include "ck_tile/ops/gemm/block/block_gemm_areg_bsmem_creg_v1_custom_policy.hpp"
#include "ck_tile/ops/gemm/block/block_gemm_areg_bsmem_creg_v1.hpp"
#include "ck_tile/ops/gemm/block/block_gemm_asmem_breg_creg_v1_custom_policy.hpp"
#include "ck_tile/ops/gemm/block/block_gemm_asmem_breg_creg_v1.hpp"
#include "ck_tile/ops/gemm/block/block_gemm_asmem_bsmem_creg_v1_custom_policy.hpp"
#include "ck_tile/ops/gemm/block/block_gemm_asmem_bsmem_creg_v1.hpp"

namespace ck_tile {

template <bool QLoadOnce_,
          bool QTLoadOnce_,
          bool KLoadOnce_,
          bool KTLoadOnce_,
          bool VLoadOnce_,
          bool OGradLoadOnce_,
          bool OGradTLoadOnce_>
struct BlockFmhaBwdPipelineDefaultPolicy
{
    static constexpr bool QLoadOnce =
        QLoadOnce_; // if q load whole block length (qkhdim) to LDS at once
    static constexpr bool QTLoadOnce =
        QTLoadOnce_; // if q^t load whole block length (qkhdim) to LDS at once
    static constexpr bool KLoadOnce =
        KLoadOnce_; // if k load whole block length (qkhdim) to LDS at once
    static constexpr bool KTLoadOnce =
        KTLoadOnce_; // if k^t load whole block length (qkhdim) to LDS at once
    static constexpr bool VLoadOnce =
        VLoadOnce_; // if v load whole block length (vhdim) to Vgprs at once
    static constexpr bool OGradLoadOnce =
        OGradLoadOnce_; // if do load whole block length (vhdim) to LDS at once
    static constexpr bool OGradTLoadOnce =
        OGradTLoadOnce_; // if do^t load whole block length (vhdim) to LDS at once

    // these are for global load
    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto GetAlignmentQ()
    {
        using QDataType = remove_cvref_t<typename Problem::QDataType>;
        return 16 / sizeof(QDataType);
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto GetAlignmentK()
    {
        using KDataType = remove_cvref_t<typename Problem::KDataType>;
        return 16 / sizeof(KDataType);
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto GetAlignmentV()
    {
        if constexpr(VLoadOnce)
        {
            using BlockGemm       = remove_cvref_t<decltype(GetOGradVBlockGemm<Problem>())>;
            constexpr auto config = BlockGemm::Policy::template GetWarpGemmMWarpNWarp<Problem>();
            using WG              = remove_cvref_t<decltype(config.template at<0>())>;
            return WG::kK / WG::WarpGemmAttribute::Impl::kABKLane;
        }
        else
        {
            using VDataType = remove_cvref_t<typename Problem::VDataType>;
            return 16 / sizeof(VDataType);
        }
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto GetAlignmentO()
    {
        using ODataType = remove_cvref_t<typename Problem::ODataType>;
        return 16 / sizeof(ODataType);
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto GetAlignmentOGrad()
    {
        using OGradDataType = remove_cvref_t<typename Problem::OGradDataType>;
        return 16 / sizeof(OGradDataType);
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto GetAlignmentQGrad()
    {
        using BlockGemm       = remove_cvref_t<decltype(GetSGradKTBlockGemm<Problem>())>;
        constexpr auto config = BlockGemm::Policy::template GetWarpGemmMWarpNWarp<Problem>();
        using WG              = remove_cvref_t<decltype(config.template at<0>())>;
        using CWarpDstr       = typename WG::CWarpDstr;
        constexpr auto vec =
            CWarpDstr{}.get_ys_to_d_descriptor().get_lengths().at(number<CWarpDstr::NDimY - 1>{});
        return vec;
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto GetAlignmentKGrad()
    {
        using BlockGemm       = remove_cvref_t<decltype(GetSGradTQTBlockGemm<Problem>())>;
        constexpr auto config = BlockGemm::Policy::template GetWarpGemmMWarpNWarp<Problem>();
        using WG              = remove_cvref_t<decltype(config.template at<0>())>;
        using CWarpDstr       = typename WG::CWarpDstr;
        constexpr auto vec =
            CWarpDstr{}.get_ys_to_d_descriptor().get_lengths().at(number<CWarpDstr::NDimY - 1>{});
        return vec;
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto GetAlignmentVGrad()
    {
        using BlockGemm       = remove_cvref_t<decltype(GetPTOGradTBlockGemm<Problem>())>;
        constexpr auto config = BlockGemm::Policy::template GetWarpGemmMWarpNWarp<Problem>();
        using WG              = remove_cvref_t<decltype(config.template at<0>())>;
        using CWarpDstr       = typename WG::CWarpDstr;
        constexpr auto vec =
            CWarpDstr{}.get_ys_to_d_descriptor().get_lengths().at(number<CWarpDstr::NDimY - 1>{});
        return vec;
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto GetTransposedAlignmentQ()
    {
        constexpr index_t kBlockSize = Problem::kBlockSize;
        constexpr index_t kNPerBlock = Problem::BlockFmhaShape::kQKHeaddim;
        constexpr index_t kKPerBlock = [&]() {
            if constexpr(QTLoadOnce)
                return Problem::BlockFmhaShape::kM0;
            else
                return Problem::BlockFmhaShape::kK3;
        }();
        constexpr index_t total_pixels = kNPerBlock * kKPerBlock / kBlockSize;

        // TODO: not correct!
        if constexpr(total_pixels > 4)
            return 4;
        else
            return 2;
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto GetTransposedAlignmentK()
    {
        constexpr index_t kBlockSize = Problem::kBlockSize;
        constexpr index_t kNPerBlock = Problem::BlockFmhaShape::kQKHeaddim;
        constexpr index_t kKPerBlock = [&]() {
            if constexpr(KTLoadOnce)
                return Problem::BlockFmhaShape::kN0;
            else
                return Problem::BlockFmhaShape::kK4;
        }();
        constexpr index_t total_pixels = kNPerBlock * kKPerBlock / kBlockSize;

        // TODO: not correct!
        if constexpr(total_pixels > 4)
            return 4;
        else
            return 2;
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto GetTransposedAlignmentOGrad()
    {
        constexpr index_t kBlockSize = Problem::kBlockSize;
        constexpr index_t kNPerBlock = Problem::BlockFmhaShape::kVHeaddim;
        constexpr index_t kKPerBlock = [&]() {
            if constexpr(OGradTLoadOnce)
                return Problem::BlockFmhaShape::kM0;
            else
                return Problem::BlockFmhaShape::kK1;
        }();
        constexpr index_t total_pixels = kNPerBlock * kKPerBlock / kBlockSize;

        // TODO: not correct!
        if constexpr(total_pixels > 4)
            return 4;
        else
            return 2;
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto GetTransposedAlignmentBias()
    {
        constexpr index_t kBlockSize = Problem::kBlockSize;
        constexpr index_t kMPerBlock = Problem::BlockFmhaShape::kM0;
        constexpr index_t kNPerBlock = Problem::BlockFmhaShape::kN0;

        constexpr index_t total_pixels = kMPerBlock * kNPerBlock / kBlockSize;

        // TODO: not correct!
        if constexpr(total_pixels > 32)
            return 8;
        else
            return 4;
    }

    // these are for lds
    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto GetSmemKPackQ()
    {
        // TODO: this is for 3d layout
        using QDataType = remove_cvref_t<typename Problem::QDataType>;
        return 16 / sizeof(QDataType);
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto GetSmemKPackK()
    {
        // TODO: this is for 3d layout
        using KDataType = remove_cvref_t<typename Problem::KDataType>;
        return 16 / sizeof(KDataType);
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto GetSmemKPackV()
    {
        // TODO: this is for 3d layout
        using VDataType = remove_cvref_t<typename Problem::VDataType>;
        return 16 / sizeof(VDataType);
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto GetSmemKPackBias()
    {
        // TODO: this is for 3d layout
        using BiasDataType = remove_cvref_t<typename Problem::BiasDataType>;
        return 16 / sizeof(BiasDataType);
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto GetSmemKPackOGrad()
    {
        // TODO: this is for 3d layout
        using OGradDataType = remove_cvref_t<typename Problem::OGradDataType>;
        return 16 / sizeof(OGradDataType);
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto GetSmemKPackSGrad()
    {
        // TODO: this is for 3d layout
        using GemmDataType = remove_cvref_t<typename Problem::GemmDataType>;
        return 16 / sizeof(GemmDataType);
    }

    template <typename Problem, typename BlockGemm>
    CK_TILE_HOST_DEVICE static constexpr auto MakeVInRegDramTileDistribution()
    {
        constexpr index_t kNPerBlock = Problem::BlockFmhaShape::kN0;
        constexpr index_t kKPerBlock = Problem::BlockFmhaShape::kVHeaddim;

        constexpr auto config = BlockGemm::Policy::template GetWarpGemmMWarpNWarp<Problem>();

        using WG = remove_cvref_t<decltype(config.template at<0>())>;

        constexpr index_t MWarp = config.template at<1>();
        constexpr index_t NWarp = config.template at<2>();

        constexpr index_t NIterPerWarp = kNPerBlock / (NWarp * WG::kN);
        constexpr index_t KIterPerWarp = kKPerBlock / WG::kK;

        constexpr auto v_block_outer_dstr_encoding =
            tile_distribution_encoding<sequence<MWarp>,
                                       tuple<sequence<NIterPerWarp, NWarp>, sequence<KIterPerWarp>>,
                                       tuple<sequence<0, 1>>,
                                       tuple<sequence<0, 1>>,
                                       sequence<1, 2>,
                                       sequence<0, 0>>{};

        constexpr auto v_block_dstr_encode = detail::make_embed_tile_distribution_encoding(
            v_block_outer_dstr_encoding, typename WG::BWarpDstrEncoding{});

        constexpr auto v_block_dstr = make_static_tile_distribution(v_block_dstr_encode);

        return v_block_dstr;
    }

    // 3d + padding
    template <index_t MNPerBlock, index_t KPerBlock, index_t KPack>
    CK_TILE_HOST_DEVICE static constexpr auto MakeXLdsBlockDescriptor()
    {
        constexpr auto x_lds_block_desc_0 = make_naive_tensor_descriptor(
            make_tuple(number<KPerBlock / KPack>{}, number<MNPerBlock>{}, number<KPack>{}),
            make_tuple(number<(MNPerBlock + 1) * KPack>{}, number<KPack>{}, number<1>{}),
            number<8>{},
            number<1>{});

        constexpr auto x_lds_block_desc = transform_tensor_descriptor(
            x_lds_block_desc_0,
            make_tuple(make_pass_through_transform(MNPerBlock),
                       make_merge_transform(make_tuple(KPerBlock / KPack, KPack))),
            make_tuple(sequence<1>{}, sequence<0, 2>{}),
            make_tuple(sequence<0>{}, sequence<1>{}));

        return x_lds_block_desc;
    }

    // 3d + padding
    template <index_t MNPerBlock, index_t KPerBlock, index_t KPack>
    CK_TILE_HOST_DEVICE static constexpr auto MakeXLdsBlockDescriptorAsXT()
    {
        constexpr auto x_lds_block_desc_0 = make_naive_tensor_descriptor(
            make_tuple(number<KPerBlock / KPack>{}, number<MNPerBlock>{}, number<KPack>{}),
            make_tuple(number<(MNPerBlock + 1) * KPack>{}, number<KPack>{}, number<1>{}),
            number<8>{},
            number<1>{});

        constexpr auto xt_lds_block_desc = transform_tensor_descriptor(
            x_lds_block_desc_0,
            make_tuple(make_pass_through_transform(MNPerBlock),
                       make_merge_transform(make_tuple(KPerBlock / KPack, KPack))),
            make_tuple(sequence<1>{}, sequence<0, 2>{}),
            make_tuple(sequence<1>{}, sequence<0>{}));

        return xt_lds_block_desc;
    }

    template <index_t MNPerBlock, index_t KPerBlock, index_t KPack, index_t PixelsPerRow>
    CK_TILE_HOST_DEVICE static constexpr auto MakeXTLdsBlockDescriptor()
    {
        static_assert(PixelsPerRow % KPack == 0);
        constexpr index_t NPerRow = PixelsPerRow / KPack;
        static_assert(MNPerBlock % NPerRow == 0);
        static_assert(KPerBlock % KPack == 0);

        constexpr auto xt_lds_block_desc_0 = make_naive_tensor_descriptor(
            make_tuple(number<KPerBlock / KPack>{},
                       number<MNPerBlock / NPerRow>{},
                       number<NPerRow>{},
                       number<KPack>{}),
            make_tuple(number<(MNPerBlock / NPerRow) * (PixelsPerRow + KPack)>{},
                       number<PixelsPerRow + KPack>{},
                       number<KPack>{},
                       number<1>{}),
            number<KPack>{},
            number<1>{});

        constexpr auto xt_lds_block_desc = transform_tensor_descriptor(
            xt_lds_block_desc_0,
            make_tuple(
                make_merge_transform(make_tuple(number<MNPerBlock / NPerRow>{}, number<NPerRow>{})),
                make_merge_transform(make_tuple(number<KPerBlock / KPack>{}, number<KPack>{}))),
            make_tuple(sequence<1, 2>{}, sequence<0, 3>{}),
            make_tuple(sequence<0>{}, sequence<1>{}));

        return xt_lds_block_desc;
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto MakeQLdsBlockDescriptor()
    {
        constexpr index_t kMPerBlock = Problem::BlockFmhaShape::kM0;
        constexpr index_t kKPerBlock = [&]() {
            if constexpr(QLoadOnce)
                return Problem::BlockFmhaShape::kQKHeaddim;
            else
                return Problem::BlockFmhaShape::kK0;
        }();
        constexpr index_t kKPack = GetSmemKPackQ<Problem>();

        return MakeXLdsBlockDescriptor<kMPerBlock, kKPerBlock, kKPack>();
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto MakeQLdsBlockDescriptorAsQT()
    {
        constexpr index_t kMPerBlock = Problem::BlockFmhaShape::kM0;
        constexpr index_t kKPerBlock = [&]() {
            if constexpr(QLoadOnce)
                return Problem::BlockFmhaShape::kQKHeaddim;
            else
                return Problem::BlockFmhaShape::kK0;
        }();
        constexpr index_t kKPack = GetSmemKPackQ<Problem>();

        return MakeXLdsBlockDescriptorAsXT<kMPerBlock, kKPerBlock, kKPack>();
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto MakeKLdsBlockDescriptor()
    {
        constexpr index_t kNPerBlock = Problem::BlockFmhaShape::kN0;
        constexpr index_t kKPerBlock = [&]() {
            if constexpr(KLoadOnce)
                return Problem::BlockFmhaShape::kQKHeaddim;
            else
                return Problem::BlockFmhaShape::kK0;
        }();
        constexpr index_t kKPack = GetSmemKPackK<Problem>();

        return MakeXLdsBlockDescriptor<kNPerBlock, kKPerBlock, kKPack>();
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto MakeKLdsBlockDescriptorAsKT()
    {
        constexpr index_t kNPerBlock = Problem::BlockFmhaShape::kN0;
        constexpr index_t kKPerBlock = [&]() {
            if constexpr(KLoadOnce)
                return Problem::BlockFmhaShape::kQKHeaddim;
            else
                return Problem::BlockFmhaShape::kK0;
        }();
        constexpr index_t kKPack = GetSmemKPackK<Problem>();

        return MakeXLdsBlockDescriptorAsXT<kNPerBlock, kKPerBlock, kKPack>();
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto MakeVLdsBlockDescriptor()
    {
        constexpr index_t kNPerBlock = Problem::BlockFmhaShape::kN0;
        constexpr index_t kKPerBlock = Problem::BlockFmhaShape::kK2;
        constexpr index_t kKPack     = GetSmemKPackV<Problem>();

        return MakeXLdsBlockDescriptor<kNPerBlock, kKPerBlock, kKPack>();
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto MakeOGradLdsBlockDescriptor()
    {
        constexpr index_t kMPerBlock = Problem::BlockFmhaShape::kM0;
        constexpr index_t kKPerBlock = [&]() {
            if constexpr(OGradLoadOnce)
                return Problem::BlockFmhaShape::kVHeaddim;
            else
                return Problem::BlockFmhaShape::kK2;
        }();
        constexpr index_t kKPack = GetSmemKPackOGrad<Problem>();

        return MakeXLdsBlockDescriptor<kMPerBlock, kKPerBlock, kKPack>();
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto MakeOGradLdsBlockDescriptorAsOGradT()
    {
        constexpr index_t kMPerBlock = Problem::BlockFmhaShape::kM0;
        constexpr index_t kKPerBlock = [&]() {
            if constexpr(OGradLoadOnce)
                return Problem::BlockFmhaShape::kVHeaddim;
            else
                return Problem::BlockFmhaShape::kK2;
        }();
        constexpr index_t kKPack = GetSmemKPackOGrad<Problem>();

        return MakeXLdsBlockDescriptorAsXT<kMPerBlock, kKPerBlock, kKPack>();
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto MakeSGradLdsBlockDescriptor()
    {
        constexpr index_t kMPerBlock = Problem::BlockFmhaShape::kM0;
        constexpr index_t kKPerBlock = Problem::BlockFmhaShape::kN0;
        constexpr index_t kKPack     = GetSmemKPackSGrad<Problem>();

        return MakeXLdsBlockDescriptor<kMPerBlock, kKPerBlock, kKPack>();
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto MakeQTLdsBlockDescriptor()
    {
        using QDataType                = remove_cvref_t<typename Problem::QDataType>;
        constexpr index_t Banks        = 32; // TODO: need change based on arch
        constexpr index_t PixelsPerRow = Banks * 4 / sizeof(QDataType);
        constexpr index_t kKPack       = GetSmemKPackQ<Problem>();
        constexpr index_t kNPerBlock   = Problem::BlockFmhaShape::kQKHeaddim;
        constexpr index_t kKPerBlock   = [&]() {
            if constexpr(QTLoadOnce)
                return Problem::BlockFmhaShape::kM0;
            else
                return Problem::BlockFmhaShape::kK3;
        }();

        return MakeXTLdsBlockDescriptor<kNPerBlock, kKPerBlock, kKPack, PixelsPerRow>();
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto MakeKTLdsBlockDescriptor()
    {
        using KDataType                = remove_cvref_t<typename Problem::KDataType>;
        constexpr index_t Banks        = 32; // TODO: need change based on arch
        constexpr index_t PixelsPerRow = Banks * 4 / sizeof(KDataType);
        constexpr index_t kKPack       = GetSmemKPackK<Problem>();
        constexpr index_t kNPerBlock   = Problem::BlockFmhaShape::kQKHeaddim;
        constexpr index_t kKPerBlock   = [&]() {
            if constexpr(KTLoadOnce)
                return Problem::BlockFmhaShape::kN0;
            else
                return Problem::BlockFmhaShape::kK4;
        }();

        return MakeXTLdsBlockDescriptor<kNPerBlock, kKPerBlock, kKPack, PixelsPerRow>();
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto MakeOGradTLdsBlockDescriptor()
    {
        using OGradDataType            = remove_cvref_t<typename Problem::OGradDataType>;
        constexpr index_t Banks        = 32; // TODO: need change based on arch
        constexpr index_t PixelsPerRow = Banks * 4 / sizeof(OGradDataType);
        constexpr index_t kKPack       = GetSmemKPackOGrad<Problem>();
        constexpr index_t kNPerBlock   = Problem::BlockFmhaShape::kVHeaddim;
        constexpr index_t kKPerBlock   = [&]() {
            if constexpr(OGradTLoadOnce)
                return Problem::BlockFmhaShape::kM0;
            else
                return Problem::BlockFmhaShape::kK1;
        }();

        return MakeXTLdsBlockDescriptor<kNPerBlock, kKPerBlock, kKPack, PixelsPerRow>();
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto MakeBiasTLdsBlockDescriptor()
    {
        using BiasDataType             = remove_cvref_t<typename Problem::BiasDataType>;
        constexpr index_t Banks        = 32; // TODO: need change based on arch
        constexpr index_t PixelsPerRow = Banks * 4 / sizeof(BiasDataType);
        constexpr index_t kKPack       = GetSmemKPackBias<Problem>();
        constexpr index_t kMPerBlock   = Problem::BlockFmhaShape::kM0;
        constexpr index_t kNPerBlock   = Problem::BlockFmhaShape::kN0;

        static_assert(PixelsPerRow % kKPack == 0);
        constexpr index_t NPerRow = PixelsPerRow / kKPack;
        static_assert(kNPerBlock % NPerRow == 0);
        static_assert(kMPerBlock % kKPack == 0);

        constexpr auto biast_lds_block_desc_0 = make_naive_tensor_descriptor(
            make_tuple(number<kMPerBlock / kKPack>{},
                       number<kNPerBlock / NPerRow>{},
                       number<NPerRow>{},
                       number<kKPack>{}),
            make_tuple(number<(kNPerBlock / NPerRow) * (PixelsPerRow + kKPack)>{},
                       number<PixelsPerRow + kKPack>{},
                       number<kKPack>{},
                       number<1>{}),
            number<kKPack>{},
            number<1>{});

        constexpr auto biast_lds_block_desc = transform_tensor_descriptor(
            biast_lds_block_desc_0,
            make_tuple(
                make_merge_transform(make_tuple(number<kNPerBlock / NPerRow>{}, number<NPerRow>{})),
                make_merge_transform(make_tuple(number<kMPerBlock / kKPack>{}, number<kKPack>{}))),
            make_tuple(sequence<1, 2>{}, sequence<0, 3>{}),
            make_tuple(sequence<1>{}, sequence<0>{}));

        return biast_lds_block_desc;
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr ck_tile::index_t GetSmemSizeQ()
    {
        constexpr index_t smem_size_q = sizeof(typename Problem::QDataType) *
                                        MakeQLdsBlockDescriptor<Problem>().get_element_space_size();
        return smem_size_q;
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr ck_tile::index_t GetSmemSizeQT()
    {
        constexpr index_t smem_size_qt = [&]() {
            if constexpr(QLoadOnce && !QTLoadOnce)
                return 0;
            else
                return sizeof(typename Problem::QDataType) *
                       MakeQTLdsBlockDescriptor<Problem>().get_element_space_size();
        }();
        return smem_size_qt;
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr ck_tile::index_t GetSmemSizeK()
    {
        constexpr index_t smem_size_k = sizeof(typename Problem::KDataType) *
                                        MakeKLdsBlockDescriptor<Problem>().get_element_space_size();
        return smem_size_k;
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr ck_tile::index_t GetSmemSizeKT()
    {
        constexpr index_t smem_size_kt = [&]() {
            if constexpr(KLoadOnce && !KTLoadOnce)
                return 0;
            else
                return sizeof(typename Problem::KDataType) *
                       MakeKTLdsBlockDescriptor<Problem>().get_element_space_size();
        }();
        return smem_size_kt;
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr ck_tile::index_t GetSmemSizeV()
    {
        constexpr index_t smem_size_v = [&]() {
            if constexpr(VLoadOnce)
                return 0;
            else
                return sizeof(typename Problem::VDataType) *
                       MakeVLdsBlockDescriptor<Problem>().get_element_space_size();
        }();
        return smem_size_v;
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr ck_tile::index_t GetSmemSizeOGrad()
    {
        constexpr index_t smem_size_do =
            sizeof(typename Problem::OGradDataType) *
            MakeOGradLdsBlockDescriptor<Problem>().get_element_space_size();
        return smem_size_do;
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr ck_tile::index_t GetSmemSizeOGradT()
    {
        constexpr index_t smem_size_dot = [&]() {
            if constexpr(OGradLoadOnce && !OGradTLoadOnce)
                return 0;
            else
                return sizeof(typename Problem::OGradDataType) *
                       MakeOGradTLdsBlockDescriptor<Problem>().get_element_space_size();
        }();
        return smem_size_dot;
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr ck_tile::index_t GetSmemSizeSGrad()
    {
        constexpr index_t smem_size_ds =
            sizeof(typename Problem::GemmDataType) *
            MakeSGradLdsBlockDescriptor<Problem>().get_element_space_size();
        return smem_size_ds;
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr ck_tile::index_t GetSmemSizeBias()
    {
        constexpr index_t smem_size_bias = [&]() {
            if constexpr(Problem::BiasEnum == BlockAttentionBiasEnum::ELEMENTWISE_BIAS)
                return sizeof(typename Problem::BiasDataType) *
                       MakeBiasTLdsBlockDescriptor<Problem>().get_element_space_size();
            else
                return 0;
        }();
        return smem_size_bias;
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr ck_tile::index_t GetSmemSize()
    {
        constexpr index_t smem_size_q         = GetSmemSizeQ<Problem>();
        constexpr index_t smem_size_qt        = GetSmemSizeQT<Problem>();
        constexpr index_t smem_size_k         = GetSmemSizeK<Problem>();
        constexpr index_t smem_size_kt        = GetSmemSizeKT<Problem>();
        constexpr index_t smem_size_v         = GetSmemSizeV<Problem>();
        constexpr index_t smem_size_do        = GetSmemSizeOGrad<Problem>();
        constexpr index_t smem_size_dot       = GetSmemSizeOGradT<Problem>();
        constexpr index_t smem_size_ds        = GetSmemSizeSGrad<Problem>();
        constexpr index_t smem_size_bias      = GetSmemSizeBias<Problem>();
        constexpr index_t smem_size_transpose = max(smem_size_ds, smem_size_bias);

        index_t smem_size = 0;

        if constexpr(QLoadOnce && OGradLoadOnce)
            smem_size += smem_size_q + smem_size_qt + smem_size_do + smem_size_dot +
                         smem_size_transpose; // 1~4 & 10
        else if(QLoadOnce && !OGradLoadOnce && !OGradTLoadOnce)
            smem_size += smem_size_q + smem_size_qt +
                         max(smem_size_do,
                             smem_size_dot,
                             smem_size_transpose); // 5/7/11 TODO: Multiple buffers strategy
        else if(!QLoadOnce && !QTLoadOnce && OGradLoadOnce)
            smem_size += smem_size_do + smem_size_dot +
                         max(smem_size_q,
                             smem_size_qt,
                             smem_size_transpose); // 6/8/12 TODO: Multiple buffers strategy
        else if(!QLoadOnce && !QTLoadOnce && !OGradLoadOnce && !OGradTLoadOnce)
            smem_size += max(smem_size_q,
                             smem_size_qt,
                             smem_size_do,
                             smem_size_dot,
                             smem_size_transpose); // 9/13 TODO: Multiple buffers strategy

        // 14/15 needs to be adjusted
        if constexpr(KLoadOnce)
            smem_size += (smem_size_k + smem_size_kt); // 1~13
        else
            smem_size =
                max(smem_size_k, smem_size_kt, smem_size); // 14/15 TODO: Multiple buffers strategy

        return max(smem_size, smem_size_v); // 15
    }

    template <typename Problem, typename BlockGemm>
    CK_TILE_HOST_DEVICE static constexpr auto MakeLSEDDramTileDistribution()
    {
        constexpr auto config   = BlockGemm::Policy::template GetWarpGemmMWarpNWarp<Problem>();
        using WG                = remove_cvref_t<decltype(config.template at<0>())>;
        constexpr index_t MWarp = config.template at<1>();
        constexpr index_t NWarp = config.template at<2>();

        constexpr index_t kMPerBlock = Problem::BlockFmhaShape::kM0;

        constexpr index_t N1 = WG::WarpGemmAttribute::Impl::kCNLane;
        constexpr index_t N0 = NWarp;

        constexpr index_t M4 = WG::WarpGemmAttribute::Impl::kCM1PerLane * 2;
        constexpr index_t M3 = WG::WarpGemmAttribute::Impl::kCMLane;
        constexpr index_t M2 = WG::WarpGemmAttribute::Impl::kCM0PerLane / 2;
        constexpr index_t M1 = MWarp;
        constexpr index_t M0 = kMPerBlock / (M1 * WG::WarpGemmAttribute::Impl::kM);

        return make_static_tile_distribution(
            tile_distribution_encoding<sequence<N0, N1>,
                                       tuple<sequence<M0, M1, M2, M3, M4>>,
                                       tuple<sequence<1, 0>, sequence<1, 0>>,
                                       tuple<sequence<1, 0>, sequence<3, 1>>,
                                       sequence<1, 1, 1>,
                                       sequence<0, 2, 4>>{});
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto MakeVDramTileDistribution()
    {
        using VDataType = remove_cvref_t<typename Problem::VDataType>;

        constexpr index_t kBlockSize = Problem::kBlockSize;

        constexpr index_t kNPerBlock = Problem::BlockFmhaShape::kN0;
        constexpr index_t kKPerBlock = Problem::BlockFmhaShape::kK2;

        constexpr index_t K1 = 16 / sizeof(VDataType);
        constexpr index_t K0 = kKPerBlock / K1;
        constexpr index_t N2 = get_warp_size() / K0;
        // coalesce reading for each blocks
        constexpr index_t N1 = kBlockSize / get_warp_size();
        constexpr index_t N0 = kNPerBlock / (N2 * N1);

        return make_static_tile_distribution(
            tile_distribution_encoding<sequence<>,
                                       tuple<sequence<N0, N1, N2>, sequence<K0, K1>>,
                                       tuple<sequence<1>, sequence<1, 2>>,
                                       tuple<sequence<1>, sequence<2, 0>>,
                                       sequence<1, 2>,
                                       sequence<0, 1>>{});
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto MakeQDramTileDistribution()
    {
        constexpr index_t kBlockSize = Problem::kBlockSize;

        constexpr index_t kMPerBlock = Problem::BlockFmhaShape::kM0;
        constexpr index_t kKPerBlock = [&]() {
            if constexpr(QLoadOnce)
                return Problem::BlockFmhaShape::kQKHeaddim;
            else
                return Problem::BlockFmhaShape::kK0;
        }();

        constexpr index_t K1 = GetAlignmentQ<Problem>();
        constexpr index_t K0 = kKPerBlock / K1;
        constexpr index_t M2 = get_warp_size() / K0;
        // coalesce reading for each blocks
        constexpr index_t M1 = kBlockSize / get_warp_size();
        constexpr index_t M0 = kMPerBlock / (M2 * M1);

        return make_static_tile_distribution(
            tile_distribution_encoding<sequence<>,
                                       tuple<sequence<M0, M1, M2>, sequence<K0, K1>>,
                                       tuple<sequence<1>, sequence<1, 2>>,
                                       tuple<sequence<1>, sequence<2, 0>>,
                                       sequence<1, 2>,
                                       sequence<0, 1>>{});
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto MakeKDramTileDistribution()
    {
        constexpr index_t kBlockSize = Problem::kBlockSize;

        constexpr index_t kNPerBlock = Problem::BlockFmhaShape::kN0;
        constexpr index_t kKPerBlock = [&]() {
            if constexpr(KLoadOnce)
                return Problem::BlockFmhaShape::kQKHeaddim;
            else
                return Problem::BlockFmhaShape::kK0;
        }();

        constexpr index_t K1 = GetAlignmentK<Problem>();
        constexpr index_t K0 = kKPerBlock / K1;
        constexpr index_t N2 = get_warp_size() / K0;
        // coalesce reading for each blocks
        constexpr index_t N1 = kBlockSize / get_warp_size();
        constexpr index_t N0 = kNPerBlock / (N2 * N1);

        return make_static_tile_distribution(
            tile_distribution_encoding<sequence<>,
                                       tuple<sequence<N0, N1, N2>, sequence<K0, K1>>,
                                       tuple<sequence<1>, sequence<1, 2>>,
                                       tuple<sequence<1>, sequence<2, 0>>,
                                       sequence<1, 2>,
                                       sequence<0, 1>>{});
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto MakeOGradDramTileDistribution()
    {
        constexpr index_t kBlockSize = Problem::kBlockSize;

        constexpr index_t kMPerBlock = Problem::BlockFmhaShape::kM0;
        constexpr index_t kKPerBlock = [&]() {
            if constexpr(OGradLoadOnce)
                return Problem::BlockFmhaShape::kVHeaddim;
            else
                return Problem::BlockFmhaShape::kK2;
        }();

        constexpr index_t K1 = GetAlignmentOGrad<Problem>();
        constexpr index_t K0 = kKPerBlock / K1;
        constexpr index_t M2 = get_warp_size() / K0;
        // coalesce reading for each blocks
        constexpr index_t M1 = kBlockSize / get_warp_size();
        constexpr index_t M0 = kMPerBlock / (M2 * M1);

        return make_static_tile_distribution(
            tile_distribution_encoding<sequence<>,
                                       tuple<sequence<M0, M1, M2>, sequence<K0, K1>>,
                                       tuple<sequence<1>, sequence<1, 2>>,
                                       tuple<sequence<1>, sequence<2, 0>>,
                                       sequence<1, 2>,
                                       sequence<0, 1>>{});
    }

    template <typename DataType, index_t MPerBlock, index_t KPerBlock>
    CK_TILE_HOST_DEVICE static constexpr auto MakePreXDramTileDistribution()
    {
        constexpr index_t K1 = 16 / sizeof(DataType);
        constexpr index_t K0 = KPerBlock / K1;
        constexpr index_t M2 = 1;
        constexpr index_t M1 = get_warp_size();
        constexpr index_t M0 = MPerBlock / M1;

        return make_static_tile_distribution(
            tile_distribution_encoding<sequence<>,
                                       tuple<sequence<M0, M1, M2>, sequence<K0, K1>>,
                                       tuple<sequence<1>, sequence<1>>,
                                       tuple<sequence<0>, sequence<1>>,
                                       sequence<1, 2, 2>,
                                       sequence<2, 0, 1>>{});
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto MakePreODramTileDistribution()
    {
        using ODataType = remove_cvref_t<typename Problem::ODataType>;

        constexpr index_t kBlockSize = Problem::kBlockSize;
        constexpr index_t kKPerBlock = Problem::kVHeaddim;

        return MakePreXDramTileDistribution<ODataType, kBlockSize, kKPerBlock>();
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto MakePreOGradDramTileDistribution()
    {
        using OGradDataType = remove_cvref_t<typename Problem::OGradDataType>;

        constexpr index_t kBlockSize = Problem::kBlockSize;
        constexpr index_t kKPerBlock = Problem::kVHeaddim;

        return MakePreXDramTileDistribution<OGradDataType, kBlockSize, kKPerBlock>();
    }

    template <typename Problem>
    CK_TILE_DEVICE static constexpr auto MakeQTDramTileDistribution()
    {
        constexpr index_t kBlockSize = Problem::kBlockSize;
        constexpr index_t kNPerBlock = Problem::BlockFmhaShape::kQKHeaddim;
        constexpr index_t kKPerBlock = [&]() {
            if constexpr(QTLoadOnce)
                return Problem::BlockFmhaShape::kM0;
            else
                return Problem::BlockFmhaShape::kK3;
        }();

        constexpr index_t N1 = GetTransposedAlignmentQ<Problem>();
        constexpr index_t N0 = kNPerBlock / N1; // P

        constexpr index_t total_pixels = kNPerBlock * kKPerBlock / kBlockSize;
        static_assert(total_pixels % N1 == 0); // TODO: this is not always true?
        constexpr index_t K3     = total_pixels / N1;
        constexpr index_t kKPack = GetSmemKPackQ<Problem>();
        static_assert(kKPack % K3 == 0);
        constexpr index_t K2 = kKPack / K3; // TODO: this dimention could be outside single wave
        constexpr index_t K1 = get_warp_size() / (K2 * N0);
        constexpr index_t K0 = kBlockSize / get_warp_size();
        static_assert(kKPerBlock == K0 * K1 * K2 * K3);

        return make_static_tile_distribution(
            tile_distribution_encoding<sequence<>,
                                       tuple<sequence<N0, N1>, sequence<K0, K1, K2, K3>>,
                                       tuple<sequence<2>, sequence<2, 1, 2>>,
                                       tuple<sequence<0>, sequence<1, 0, 2>>,
                                       sequence<2, 1>,
                                       sequence<3, 1>>{});
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto MakeShuffledQTRegBlockDescriptor()
    {
        constexpr index_t kBlockSize = Problem::kBlockSize;
        constexpr index_t kNPerBlock = Problem::BlockFmhaShape::kQKHeaddim;
        constexpr index_t kKPerBlock = [&]() {
            if constexpr(QTLoadOnce)
                return Problem::BlockFmhaShape::kM0;
            else
                return Problem::BlockFmhaShape::kK3;
        }();

        constexpr index_t N1           = GetTransposedAlignmentQ<Problem>();
        constexpr index_t N0           = kNPerBlock / N1;
        constexpr index_t total_pixels = kNPerBlock * kKPerBlock / kBlockSize;
        static_assert(total_pixels % N1 == 0); // TODO: this is not always true?
        constexpr index_t K3     = total_pixels / N1;
        constexpr index_t kKPack = GetSmemKPackQ<Problem>();
        static_assert(kKPack % K3 == 0);
        constexpr index_t K2 = kKPack / K3; // TODO: this dimention could be outside single wave
        constexpr index_t K1 = get_warp_size() / (K2 * N0);
        constexpr index_t K0 = kBlockSize / get_warp_size();

        return make_static_tile_distribution(
            tile_distribution_encoding<sequence<>,
                                       tuple<sequence<N0, N1>, sequence<K0, K1, K2, K3>>,
                                       tuple<sequence<2>, sequence<2, 1, 2>>,
                                       tuple<sequence<0>, sequence<1, 0, 2>>,
                                       sequence<1, 2>,
                                       sequence<1, 3>>{});
    }

    template <typename Problem>
    CK_TILE_DEVICE static constexpr auto MakeKTDramTileDistribution()
    {
        constexpr index_t kBlockSize = Problem::kBlockSize;
        constexpr index_t kNPerBlock = Problem::BlockFmhaShape::kQKHeaddim;
        constexpr index_t kKPerBlock = [&]() {
            if constexpr(KTLoadOnce)
                return Problem::BlockFmhaShape::kN0;
            else
                return Problem::BlockFmhaShape::kK4;
        }();

        constexpr index_t N1 = GetTransposedAlignmentK<Problem>();
        constexpr index_t N0 = kNPerBlock / N1; // P

        constexpr index_t total_pixels = kNPerBlock * kKPerBlock / kBlockSize;
        static_assert(total_pixels % N1 == 0); // TODO: this is not always true?
        constexpr index_t K3     = total_pixels / N1;
        constexpr index_t kKPack = GetSmemKPackK<Problem>();
        static_assert(kKPack % K3 == 0);
        constexpr index_t K2 = kKPack / K3; // TODO: this dimention could be outside single wave
        constexpr index_t K1 = get_warp_size() / (K2 * N0);
        constexpr index_t K0 = kBlockSize / get_warp_size();
        static_assert(kKPerBlock == K0 * K1 * K2 * K3);

        return make_static_tile_distribution(
            tile_distribution_encoding<sequence<>,
                                       tuple<sequence<N0, N1>, sequence<K0, K1, K2, K3>>,
                                       tuple<sequence<2>, sequence<2, 1, 2>>,
                                       tuple<sequence<0>, sequence<1, 0, 2>>,
                                       sequence<2, 1>,
                                       sequence<3, 1>>{});
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto MakeShuffledKTRegBlockDescriptor()
    {
        constexpr index_t kBlockSize = Problem::kBlockSize;
        constexpr index_t kNPerBlock = Problem::BlockFmhaShape::kQKHeaddim;
        constexpr index_t kKPerBlock = [&]() {
            if constexpr(KTLoadOnce)
                return Problem::BlockFmhaShape::kN0;
            else
                return Problem::BlockFmhaShape::kK4;
        }();

        constexpr index_t N1           = GetTransposedAlignmentK<Problem>();
        constexpr index_t N0           = kNPerBlock / N1;
        constexpr index_t total_pixels = kNPerBlock * kKPerBlock / kBlockSize;
        static_assert(total_pixels % N1 == 0); // TODO: this is not always true?
        constexpr index_t K3     = total_pixels / N1;
        constexpr index_t kKPack = GetSmemKPackK<Problem>();
        static_assert(kKPack % K3 == 0);
        constexpr index_t K2 = kKPack / K3; // TODO: this dimention could be outside single wave
        constexpr index_t K1 = get_warp_size() / (K2 * N0);
        constexpr index_t K0 = kBlockSize / get_warp_size();

        return make_static_tile_distribution(
            tile_distribution_encoding<sequence<>,
                                       tuple<sequence<N0, N1>, sequence<K0, K1, K2, K3>>,
                                       tuple<sequence<2>, sequence<2, 1, 2>>,
                                       tuple<sequence<0>, sequence<1, 0, 2>>,
                                       sequence<1, 2>,
                                       sequence<1, 3>>{});
    }

    template <typename Problem>
    CK_TILE_DEVICE static constexpr auto MakeOGradTDramTileDistribution()
    {
        constexpr index_t kBlockSize = Problem::kBlockSize;
        constexpr index_t kNPerBlock = Problem::BlockFmhaShape::kVHeaddim;
        constexpr index_t kKPerBlock = [&]() {
            if constexpr(OGradTLoadOnce)
                return Problem::BlockFmhaShape::kM0;
            else
                return Problem::BlockFmhaShape::kK1;
        }();

        constexpr index_t N1 = GetTransposedAlignmentOGrad<Problem>();
        constexpr index_t N0 = kNPerBlock / N1; // P

        constexpr index_t total_pixels = kNPerBlock * kKPerBlock / kBlockSize;
        static_assert(total_pixels % N1 == 0); // TODO: this is not always true?
        constexpr index_t K3     = total_pixels / N1;
        constexpr index_t kKPack = GetSmemKPackOGrad<Problem>();
        static_assert(kKPack % K3 == 0);
        constexpr index_t K2 = kKPack / K3; // TODO: this dimention could be outside single wave
        constexpr index_t K1 = get_warp_size() / (K2 * N0);
        constexpr index_t K0 = kBlockSize / get_warp_size();
        static_assert(kKPerBlock == K0 * K1 * K2 * K3);

        return make_static_tile_distribution(
            tile_distribution_encoding<sequence<>,
                                       tuple<sequence<N0, N1>, sequence<K0, K1, K2, K3>>,
                                       tuple<sequence<2>, sequence<2, 1, 2>>,
                                       tuple<sequence<0>, sequence<1, 0, 2>>,
                                       sequence<2, 1>,
                                       sequence<3, 1>>{});
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto MakeShuffledOGradTRegBlockDescriptor()
    {
        constexpr index_t kBlockSize = Problem::kBlockSize;
        constexpr index_t kNPerBlock = Problem::BlockFmhaShape::kVHeaddim;
        constexpr index_t kKPerBlock = [&]() {
            if constexpr(OGradTLoadOnce)
                return Problem::BlockFmhaShape::kM0;
            else
                return Problem::BlockFmhaShape::kK1;
        }();

        constexpr index_t N1           = GetTransposedAlignmentOGrad<Problem>();
        constexpr index_t N0           = kNPerBlock / N1;
        constexpr index_t total_pixels = kNPerBlock * kKPerBlock / kBlockSize;
        static_assert(total_pixels % N1 == 0); // TODO: this is not always true?
        constexpr index_t K3     = total_pixels / N1;
        constexpr index_t kKPack = GetSmemKPackOGrad<Problem>();
        static_assert(kKPack % K3 == 0);
        constexpr index_t K2 = kKPack / K3; // TODO: this dimention could be outside single wave
        constexpr index_t K1 = get_warp_size() / (K2 * N0);
        constexpr index_t K0 = kBlockSize / get_warp_size();

        return make_static_tile_distribution(
            tile_distribution_encoding<sequence<>,
                                       tuple<sequence<N0, N1>, sequence<K0, K1, K2, K3>>,
                                       tuple<sequence<2>, sequence<2, 1, 2>>,
                                       tuple<sequence<0>, sequence<1, 0, 2>>,
                                       sequence<1, 2>,
                                       sequence<1, 3>>{});
    }

    template <typename Problem>
    CK_TILE_DEVICE static constexpr auto MakeBiasTileDistribution()
    {
        constexpr index_t kBlockSize = Problem::kBlockSize;
        constexpr index_t kMPerBlock = Problem::BlockFmhaShape::kM0;
        constexpr index_t kNPerBlock = Problem::BlockFmhaShape::kN0;

        constexpr index_t N1 = GetTransposedAlignmentBias<Problem>();
        constexpr index_t N0 = kNPerBlock / N1; // P

        constexpr index_t total_pixels = kMPerBlock * kNPerBlock / kBlockSize;
        static_assert(total_pixels % N1 == 0); // TODO: this is not always true?
        constexpr index_t M3     = total_pixels / N1;
        constexpr index_t kKPack = GetSmemKPackBias<Problem>();
        static_assert(kKPack % M3 == 0);
        constexpr index_t M2 = kKPack / M3; // TODO: this dimention could be outside single wave
        constexpr index_t M1 = get_warp_size() / (M2 * N0);
        constexpr index_t M0 = kBlockSize / get_warp_size();
        static_assert(kMPerBlock == M0 * M1 * M2 * M3);

        return make_static_tile_distribution(
            tile_distribution_encoding<sequence<>,
                                       tuple<sequence<M0, M1, M2, M3>, sequence<N0, N1>>,
                                       tuple<sequence<1>, sequence<1, 2, 1>>,
                                       tuple<sequence<0>, sequence<1, 0, 2>>,
                                       sequence<1, 2>,
                                       sequence<3, 1>>{});
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto MakeShuffledBiasTileDistribution()
    {
        constexpr index_t kBlockSize = Problem::kBlockSize;
        constexpr index_t kMPerBlock = Problem::BlockFmhaShape::kM0;
        constexpr index_t kNPerBlock = Problem::BlockFmhaShape::kN0;

        constexpr index_t N1           = GetTransposedAlignmentBias<Problem>();
        constexpr index_t N0           = kNPerBlock / N1;
        constexpr index_t total_pixels = kMPerBlock * kNPerBlock / kBlockSize;
        static_assert(total_pixels % N1 == 0); // TODO: this is not always true?
        constexpr index_t M3     = total_pixels / N1;
        constexpr index_t kKPack = GetSmemKPackBias<Problem>();
        static_assert(kKPack % M3 == 0);
        constexpr index_t M2 = kKPack / M3; // TODO: this dimention could be outside single wave
        constexpr index_t M1 = get_warp_size() / (M2 * N0);
        constexpr index_t M0 = kBlockSize / get_warp_size();

        return make_static_tile_distribution(
            tile_distribution_encoding<sequence<>,
                                       tuple<sequence<M0, M1, M2, M3>, sequence<N0, N1>>,
                                       tuple<sequence<1>, sequence<1, 2, 1>>,
                                       tuple<sequence<0>, sequence<1, 0, 2>>,
                                       sequence<2, 1>,
                                       sequence<1, 3>>{});
    }

    template <typename BlockGemm>
    CK_TILE_HOST_DEVICE static constexpr auto MakeBiasTTileDistribution()
    {
        using c_block_tensor_type = decltype(BlockGemm{}.MakeCBlockTile());
        return c_block_tensor_type::get_tile_distribution();
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto GetQKBlockGemm()
    {
        using BlockGemmProblem =
            BlockGemmPipelineProblem<typename Problem::QDataType,
                                     typename Problem::KDataType,
                                     typename Problem::AccDataType,
                                     Problem::kBlockSize,
                                     TileGemmShape<Problem::BlockFmhaShape::kM0,
                                                   Problem::BlockFmhaShape::kN0,
                                                   Problem::BlockFmhaShape::kK0>>;

        constexpr auto warp_gemm = []() {
            if constexpr(std::is_same_v<typename Problem::QDataType, half_t> &&
                         std::is_same_v<typename Problem::KDataType, half_t> &&
                         std::is_same_v<typename Problem::AccDataType, float>)
            {
                return WarpGemmMfmaF16F16F32M32N32K16SwizzleA{};
            }
            else if constexpr(std::is_same_v<typename Problem::QDataType, bf16_t> &&
                              std::is_same_v<typename Problem::KDataType, bf16_t> &&
                              std::is_same_v<typename Problem::AccDataType, float>)
            {
                return WarpGemmMfmaBf16Bf16F32M32N32K16SwizzleA{};
            }
        }();

        using BlockGemmPolicy =
            BlockGemmASmemBSmemCRegV1CustomPolicy<typename Problem::QDataType,
                                                  typename Problem::KDataType,
                                                  typename Problem::AccDataType,
                                                  typename Problem::BlockFmhaShape::Gemm0BlockWarps,
                                                  decltype(warp_gemm)>;

        return BlockGemmASmemBSmemCRegV1<BlockGemmProblem, BlockGemmPolicy>{};
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto GetPTOGradTBlockGemm()
    {
        using BlockGemmProblem =
            BlockGemmPipelineProblem<typename Problem::GemmDataType,
                                     typename Problem::OGradDataType,
                                     typename Problem::AccDataType,
                                     Problem::kBlockSize,
                                     TileGemmShape<Problem::BlockFmhaShape::kN0,
                                                   Problem::BlockFmhaShape::kVHeaddim,
                                                   Problem::BlockFmhaShape::kK1>>;

        using WarpGemm =
            WarpGemmMfmaDispatcher<typename Problem::GemmDataType,
                                   typename Problem::OGradDataType,
                                   typename Problem::AccDataType,
                                   Problem::BlockFmhaShape::Gemm1WarpTile::at(number<0>{}),
                                   Problem::BlockFmhaShape::Gemm1WarpTile::at(number<1>{}),
                                   Problem::BlockFmhaShape::Gemm1WarpTile::at(number<2>{}),
                                   true>;
        using BlockGemmPolicy =
            BlockGemmARegBSmemCRegV1CustomPolicy<typename Problem::GemmDataType,
                                                 typename Problem::OGradDataType,
                                                 typename Problem::AccDataType,
                                                 typename Problem::BlockFmhaShape::Gemm1BlockWarps,
                                                 WarpGemm>;
        return BlockGemmARegBSmemCRegV1<BlockGemmProblem, BlockGemmPolicy>{};
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto GetOGradVBlockGemm()
    {
        using BlockGemmProblem =
            BlockGemmPipelineProblem<typename Problem::OGradDataType,
                                     typename Problem::VDataType,
                                     typename Problem::AccDataType,
                                     Problem::kBlockSize,
                                     TileGemmShape<Problem::BlockFmhaShape::kM0,
                                                   Problem::BlockFmhaShape::kN0,
                                                   Problem::BlockFmhaShape::kK2>>;

        constexpr auto warp_gemm = []() {
            if constexpr(std::is_same_v<typename Problem::OGradDataType, half_t> &&
                         std::is_same_v<typename Problem::VDataType, half_t> &&
                         std::is_same_v<typename Problem::AccDataType, float>)
            {
                return WarpGemmMfmaF16F16F32M32N32K16SwizzleA{};
            }
            else if constexpr(std::is_same_v<typename Problem::OGradDataType, bf16_t> &&
                              std::is_same_v<typename Problem::VDataType, bf16_t> &&
                              std::is_same_v<typename Problem::AccDataType, float>)
            {
                return WarpGemmMfmaBf16Bf16F32M32N32K16SwizzleA{};
            }
        }();

        using BlockGemmPolicy =
            BlockGemmASmemBRegCRegV1CustomPolicy<typename Problem::OGradDataType,
                                                 typename Problem::VDataType,
                                                 typename Problem::AccDataType,
                                                 typename Problem::BlockFmhaShape::Gemm2BlockWarps,
                                                 decltype(warp_gemm)>;

        return BlockGemmASmemBRegCRegV1<BlockGemmProblem, BlockGemmPolicy>{};
    }

    // template <typename Problem>
    // CK_TILE_HOST_DEVICE static constexpr auto GetOGradVBlockGemm()
    // {
    //     using BlockGemmProblem =
    //         BlockGemmPipelineProblem<typename Problem::OGradDataType,
    //                                  typename Problem::VDataType,
    //                                  typename Problem::AccDataType,
    //                                  Problem::kBlockSize,
    //                                  TileGemmShape<Problem::BlockFmhaShape::kM0,
    //                                                Problem::BlockFmhaShape::kN0,
    //                                                Problem::BlockFmhaShape::kK2>>;
    //     constexpr auto warp_gemm = []() {
    //         if constexpr(std::is_same_v<typename Problem::OGradDataType, half_t> &&
    //                      std::is_same_v<typename Problem::VDataType, half_t> &&
    //                      std::is_same_v<typename Problem::AccDataType, float>)
    //         {
    //             return WarpGemmMfmaF16F16F32M32N32K16SwizzleA{};
    //         }
    //         else if constexpr(std::is_same_v<typename Problem::OGradDataType, bf16_t> &&
    //                           std::is_same_v<typename Problem::VDataType, bf16_t> &&
    //                           std::is_same_v<typename Problem::AccDataType, float>)
    //         {
    //             return WarpGemmMfmaBf16Bf16F32M32N32K16SwizzleA{};
    //         }
    //     }();

    //     using BlockGemmPolicy =
    //         BlockGemmASmemBSmemCRegV1CustomPolicy<typename Problem::OGradDataType,
    //                                               typename Problem::VDataType,
    //                                               typename Problem::AccDataType,
    //                                               typename
    //                                               Problem::BlockFmhaShape::Gemm2BlockWarps,
    //                                               decltype(warp_gemm)>;

    //     return BlockGemmASmemBSmemCRegV1<BlockGemmProblem, BlockGemmPolicy>{};
    // }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto GetSGradTQTBlockGemm()
    {
        using BlockGemmProblem =
            BlockGemmPipelineProblem<typename Problem::GemmDataType,
                                     typename Problem::QDataType,
                                     typename Problem::AccDataType,
                                     Problem::kBlockSize,
                                     TileGemmShape<Problem::BlockFmhaShape::kN0,
                                                   Problem::BlockFmhaShape::kQKHeaddim,
                                                   Problem::BlockFmhaShape::kK3>>;

        using WarpGemm =
            WarpGemmMfmaDispatcher<typename Problem::GemmDataType,
                                   typename Problem::QDataType,
                                   typename Problem::AccDataType,
                                   Problem::BlockFmhaShape::Gemm3WarpTile::at(number<0>{}),
                                   Problem::BlockFmhaShape::Gemm3WarpTile::at(number<1>{}),
                                   Problem::BlockFmhaShape::Gemm3WarpTile::at(number<2>{}),
                                   true>;
        using BlockGemmPolicy =
            BlockGemmARegBSmemCRegV1CustomPolicy<typename Problem::GemmDataType,
                                                 typename Problem::QDataType,
                                                 typename Problem::AccDataType,
                                                 typename Problem::BlockFmhaShape::Gemm3BlockWarps,
                                                 WarpGemm>;
        return BlockGemmARegBSmemCRegV1<BlockGemmProblem, BlockGemmPolicy>{};
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto GetSGradKTBlockGemm()
    {
        using BlockGemmProblem =
            BlockGemmPipelineProblem<typename Problem::GemmDataType,
                                     typename Problem::KDataType,
                                     typename Problem::AccDataType,
                                     Problem::kBlockSize,
                                     TileGemmShape<Problem::BlockFmhaShape::kM0,
                                                   Problem::BlockFmhaShape::kQKHeaddim,
                                                   Problem::BlockFmhaShape::kK4>>;

        using WarpGemm =
            WarpGemmMfmaDispatcher<typename Problem::GemmDataType,
                                   typename Problem::KDataType,
                                   typename Problem::AccDataType,
                                   Problem::BlockFmhaShape::Gemm4WarpTile::at(number<0>{}),
                                   Problem::BlockFmhaShape::Gemm4WarpTile::at(number<1>{}),
                                   Problem::BlockFmhaShape::Gemm4WarpTile::at(number<2>{}),
                                   true>;
        using BlockGemmPolicy =
            BlockGemmASmemBSmemCRegV1CustomPolicy<typename Problem::GemmDataType,
                                                  typename Problem::KDataType,
                                                  typename Problem::AccDataType,
                                                  typename Problem::BlockFmhaShape::Gemm4BlockWarps,
                                                  WarpGemm>;
        return BlockGemmASmemBSmemCRegV1<BlockGemmProblem, BlockGemmPolicy>{};
    }
};

} // namespace ck_tile
