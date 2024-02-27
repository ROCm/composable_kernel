// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2024, Advanced Micro Devices, Inc. All rights reserved.

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
#include "ck/tile_program/block_tile/block_gemm_asmem_breg_creg_v1.hpp"
#include "ck/tile_program/block_tile/block_gemm_asmem_breg_creg_v1_custom_policy.hpp"
#include "ck/tile_program/block_tile/block_gemm_asmem_bsmem_creg_v1.hpp"
#include "ck/tile_program/block_tile/block_gemm_asmem_bsmem_creg_v1_custom_policy.hpp"
#include "ck/tensor_operation/gpu/device/tensor_layout.hpp"

namespace ck {
namespace tile_program {
namespace block {

struct BlockFmhaBwdPipelineDefaultPolicy
{
    template <typename Problem>
    __host__ __device__ static constexpr auto GetSmemKPackQ()
    {
        // TODO: this is for 3d layout
        using QDataType = remove_cvref_t<typename Problem::QDataType>;
        return 16 / sizeof(QDataType);
    }

    template <typename Problem>
    __host__ __device__ static constexpr auto GetSmemKPackK()
    {
        // TODO: this is for 3d layout
        using KDataType = remove_cvref_t<typename Problem::KDataType>;
        return 16 / sizeof(KDataType);
    }

    template <typename Problem>
    __host__ __device__ static constexpr auto GetSmemKPackV()
    {
        // TODO: this is for 3d layout
        using VDataType = remove_cvref_t<typename Problem::VDataType>;
        return 16 / sizeof(VDataType);
    }

    template <typename Problem>
    __host__ __device__ static constexpr auto GetSmemKPackBias()
    {
        // TODO: this is for 3d layout
        using BiasDataType = remove_cvref_t<typename Problem::BiasDataType>;
        return 16 / sizeof(BiasDataType);
    }

    template <typename Problem>
    __host__ __device__ static constexpr auto GetSmemKPackOGrad()
    {
        // TODO: this is for 3d layout
        using OGradDataType = remove_cvref_t<typename Problem::OGradDataType>;
        return 16 / sizeof(OGradDataType);
    }

    template <typename Problem>
    __host__ __device__ static constexpr auto GetSmemKPackSGrad()
    {
        // TODO: this is for 3d layout
        using GemmDataType = remove_cvref_t<typename Problem::GemmDataType>;
        return 16 / sizeof(GemmDataType);
    }

    template <typename Problem>
    __host__ __device__ static constexpr auto GetTransposedVectorloadQ()
    {
        return 4; // TODO: fix me
    }

    template <typename Problem>
    __host__ __device__ static constexpr auto GetTransposedVectorloadK()
    {
        return 4; // TODO: fix me
    }

    template <typename Problem>
    __host__ __device__ static constexpr auto GetTransposedVectorloadOGrad()
    {
        return 4; // TODO: fix me
    }

    template <typename Problem>
    __host__ __device__ static constexpr auto GetTransposedVectorloadBias()
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

    template <typename Problem, typename BlockGemm>
    __host__ __device__ static constexpr auto MakeVDramRegStatTileDistribution()
    {
        constexpr index_t kNPerBlock = Problem::BlockFmhaShape::kN0;
        constexpr index_t kKPerBlock = Problem::BlockFmhaShape::kVHeaddim;

        constexpr auto config = BlockGemm::Policy::template GetWarpGemmMWarpNWarp<Problem>();

        using WG = remove_cvref_t<decltype(config.template At<0>())>;

        constexpr index_t MWarp = config.template At<1>();
        constexpr index_t NWarp = config.template At<2>();

        constexpr index_t NIterPerWarp = kNPerBlock / (NWarp * WG::kN);
        constexpr index_t KIterPerWarp = kKPerBlock / WG::kK;

        constexpr auto v_block_outer_dstr_encoding = StaticTileDistributionEncoding<
            Sequence<MWarp>,
            Tuple<Sequence<NIterPerWarp, NWarp>, Sequence<KIterPerWarp>>,
            Tuple<Sequence<0, 1>>,
            Tuple<Sequence<0, 1>>,
            Sequence<1, 2>,
            Sequence<0, 0>>{};

        constexpr auto v_block_dstr_encode = detail::make_embed_tile_distribution_encoding(
            v_block_outer_dstr_encoding, typename WG::BWarpDstrEncoding{});

        constexpr auto v_block_dstr = make_static_tile_distribution(v_block_dstr_encode);

        return v_block_dstr;
    }

    // 3d + padding
    template <index_t MNPerBlock, index_t KPerBlock, index_t KPack>
    __host__ __device__ static constexpr auto MakeXLdsBlockDescriptor()
    {
        constexpr auto x_lds_block_desc_0 = make_naive_tensor_descriptor(
            make_tuple(Number<KPerBlock / KPack>{}, Number<MNPerBlock>{}, Number<KPack>{}),
            make_tuple(Number<(MNPerBlock + 1) * KPack>{}, Number<KPack>{}, Number<1>{}),
            Number<8>{},
            Number<1>{});

        constexpr auto x_lds_block_desc = transform_tensor_descriptor(
            x_lds_block_desc_0,
            make_tuple(make_pass_through_transform(MNPerBlock),
                       make_merge_transform(make_tuple(KPerBlock / KPack, KPack))),
            make_tuple(Sequence<1>{}, Sequence<0, 2>{}),
            make_tuple(Sequence<0>{}, Sequence<1>{}));

        return x_lds_block_desc;
    }

    // 3d + padding
    template <index_t MNPerBlock, index_t KPerBlock, index_t KPack>
    __host__ __device__ static constexpr auto MakeXLdsBlockDescriptorAsXT()
    {
        constexpr auto x_lds_block_desc_0 = make_naive_tensor_descriptor(
            make_tuple(Number<KPerBlock / KPack>{}, Number<MNPerBlock>{}, Number<KPack>{}),
            make_tuple(Number<(MNPerBlock + 1) * KPack>{}, Number<KPack>{}, Number<1>{}),
            Number<8>{},
            Number<1>{});

        constexpr auto xt_lds_block_desc = transform_tensor_descriptor(
            x_lds_block_desc_0,
            make_tuple(make_pass_through_transform(MNPerBlock),
                       make_merge_transform(make_tuple(KPerBlock / KPack, KPack))),
            make_tuple(Sequence<1>{}, Sequence<0, 2>{}),
            make_tuple(Sequence<1>{}, Sequence<0>{}));

        return xt_lds_block_desc;
    }

    template <index_t MNPerBlock, index_t KPerBlock, index_t KPack, index_t PixelsPerRow>
    __host__ __device__ static constexpr auto MakeXTLdsBlockDescriptor()
    {
        static_assert(PixelsPerRow % KPack == 0);
        constexpr index_t NPerRow = PixelsPerRow / KPack;
        static_assert(MNPerBlock % NPerRow == 0);
        static_assert(KPerBlock % KPack == 0);

        constexpr auto xt_lds_block_desc_0 = make_naive_tensor_descriptor(
            make_tuple(Number<KPerBlock / KPack>{},
                       Number<MNPerBlock / NPerRow>{},
                       Number<NPerRow>{},
                       Number<KPack>{}),
            make_tuple(Number<(MNPerBlock / NPerRow) * (PixelsPerRow + KPack)>{},
                       Number<PixelsPerRow + KPack>{},
                       Number<KPack>{},
                       Number<1>{}),
            Number<KPack>{},
            Number<1>{});

        constexpr auto xt_lds_block_desc = transform_tensor_descriptor(
            xt_lds_block_desc_0,
            make_tuple(
                make_merge_transform(make_tuple(Number<MNPerBlock / NPerRow>{}, Number<NPerRow>{})),
                make_merge_transform(make_tuple(Number<KPerBlock / KPack>{}, Number<KPack>{}))),
            make_tuple(Sequence<1, 2>{}, Sequence<0, 3>{}),
            make_tuple(Sequence<0>{}, Sequence<1>{}));

        return xt_lds_block_desc;
    }

    template <typename Problem>
    __host__ __device__ static constexpr auto MakeQLdsBlockDescriptor()
    {
        constexpr index_t kMPerBlock = Problem::BlockFmhaShape::kM0;
        constexpr index_t kKPerBlock = [&]() {
            if constexpr(Problem::BlockFmhaShape::kQLoadOnce)
                return Problem::BlockFmhaShape::kQKHeaddim;
            else
                return Problem::BlockFmhaShape::kK0;
        }();
        constexpr index_t kKPack = GetSmemKPackQ<Problem>();

        return MakeXLdsBlockDescriptor<kMPerBlock, kKPerBlock, kKPack>();
    }

    template <typename Problem>
    __host__ __device__ static constexpr auto MakeQLdsBlockDescriptorAsQT()
    {
        constexpr index_t kMPerBlock = Problem::BlockFmhaShape::kM0;
        constexpr index_t kKPerBlock = [&]() {
            if constexpr(Problem::BlockFmhaShape::kQLoadOnce)
                return Problem::BlockFmhaShape::kQKHeaddim;
            else
                return Problem::BlockFmhaShape::kK0;
        }();
        constexpr index_t kKPack = GetSmemKPackQ<Problem>();

        return MakeXLdsBlockDescriptorAsXT<kMPerBlock, kKPerBlock, kKPack>();
    }

    template <typename Problem>
    __host__ __device__ static constexpr auto MakeKLdsBlockDescriptor()
    {
        constexpr index_t kNPerBlock = Problem::BlockFmhaShape::kN0;
        constexpr index_t kKPerBlock = [&]() {
            if constexpr(Problem::BlockFmhaShape::kKLoadOnce)
                return Problem::BlockFmhaShape::kQKHeaddim;
            else
                return Problem::BlockFmhaShape::kK0;
        }();
        constexpr index_t kKPack = GetSmemKPackK<Problem>();

        return MakeXLdsBlockDescriptor<kNPerBlock, kKPerBlock, kKPack>();
    }

    template <typename Problem>
    __host__ __device__ static constexpr auto MakeKLdsBlockDescriptorAsKT()
    {
        constexpr index_t kNPerBlock = Problem::BlockFmhaShape::kN0;
        constexpr index_t kKPerBlock = [&]() {
            if constexpr(Problem::BlockFmhaShape::kKLoadOnce)
                return Problem::BlockFmhaShape::kQKHeaddim;
            else
                return Problem::BlockFmhaShape::kK0;
        }();
        constexpr index_t kKPack = GetSmemKPackK<Problem>();

        return MakeXLdsBlockDescriptorAsXT<kNPerBlock, kKPerBlock, kKPack>();
    }

    template <typename Problem>
    __host__ __device__ static constexpr auto MakeVLdsBlockDescriptor()
    {
        constexpr index_t kNPerBlock = Problem::BlockFmhaShape::kN0;
        constexpr index_t kKPerBlock = Problem::BlockFmhaShape::kK2;
        constexpr index_t kKPack     = GetSmemKPackV<Problem>();

        return MakeXLdsBlockDescriptor<kNPerBlock, kKPerBlock, kKPack>();
    }

    template <typename Problem>
    __host__ __device__ static constexpr auto MakeOGradLdsBlockDescriptor()
    {
        constexpr index_t kMPerBlock = Problem::BlockFmhaShape::kM0;
        constexpr index_t kKPerBlock = [&]() {
            if constexpr(Problem::BlockFmhaShape::kOGradLoadOnce)
                return Problem::BlockFmhaShape::kVHeaddim;
            else
                return Problem::BlockFmhaShape::kK2;
        }();
        constexpr index_t kKPack = GetSmemKPackOGrad<Problem>();

        return MakeXLdsBlockDescriptor<kMPerBlock, kKPerBlock, kKPack>();
    }

    template <typename Problem>
    __host__ __device__ static constexpr auto MakeOGradLdsBlockDescriptorAsOGradT()
    {
        constexpr index_t kMPerBlock = Problem::BlockFmhaShape::kM0;
        constexpr index_t kKPerBlock = [&]() {
            if constexpr(Problem::BlockFmhaShape::kOGradLoadOnce)
                return Problem::BlockFmhaShape::kVHeaddim;
            else
                return Problem::BlockFmhaShape::kK2;
        }();
        constexpr index_t kKPack = GetSmemKPackOGrad<Problem>();

        return MakeXLdsBlockDescriptorAsXT<kMPerBlock, kKPerBlock, kKPack>();
    }

    template <typename Problem>
    __host__ __device__ static constexpr auto MakeSGradLdsBlockDescriptor()
    {
        constexpr index_t kMPerBlock = Problem::BlockFmhaShape::kM0;
        constexpr index_t kKPerBlock = Problem::BlockFmhaShape::kN0;
        constexpr index_t kKPack     = GetSmemKPackSGrad<Problem>();

        return MakeXLdsBlockDescriptor<kMPerBlock, kKPerBlock, kKPack>();
    }

    template <typename Problem>
    __host__ __device__ static constexpr auto MakeQTLdsBlockDescriptor()
    {
        using QDataType                = remove_cvref_t<typename Problem::QDataType>;
        constexpr index_t Banks        = 32; // TODO: need change based on arch
        constexpr index_t PixelsPerRow = Banks * 4 / sizeof(QDataType);
        constexpr index_t kKPack       = GetSmemKPackQ<Problem>();
        constexpr index_t kNPerBlock   = Problem::BlockFmhaShape::kQKHeaddim;
        constexpr index_t kKPerBlock   = [&]() {
            if constexpr(Problem::BlockFmhaShape::kQTLoadOnce)
                return Problem::BlockFmhaShape::kM0;
            else
                return Problem::BlockFmhaShape::kK3;
        }();

        return MakeXTLdsBlockDescriptor<kNPerBlock, kKPerBlock, kKPack, PixelsPerRow>();
    }

    template <typename Problem>
    __host__ __device__ static constexpr auto MakeKTLdsBlockDescriptor()
    {
        using KDataType                = remove_cvref_t<typename Problem::KDataType>;
        constexpr index_t Banks        = 32; // TODO: need change based on arch
        constexpr index_t PixelsPerRow = Banks * 4 / sizeof(KDataType);
        constexpr index_t kKPack       = GetSmemKPackK<Problem>();
        constexpr index_t kNPerBlock   = Problem::BlockFmhaShape::kQKHeaddim;
        constexpr index_t kKPerBlock   = [&]() {
            if constexpr(Problem::BlockFmhaShape::kKTLoadOnce)
                return Problem::BlockFmhaShape::kN0;
            else
                return Problem::BlockFmhaShape::kK4;
        }();

        return MakeXTLdsBlockDescriptor<kNPerBlock, kKPerBlock, kKPack, PixelsPerRow>();
    }

    template <typename Problem>
    __host__ __device__ static constexpr auto MakeOGradTLdsBlockDescriptor()
    {
        using QGradDataType            = remove_cvref_t<typename Problem::QGradDataType>;
        constexpr index_t Banks        = 32; // TODO: need change based on arch
        constexpr index_t PixelsPerRow = Banks * 4 / sizeof(QGradDataType);
        constexpr index_t kKPack       = GetSmemKPackOGrad<Problem>();
        constexpr index_t kNPerBlock   = Problem::BlockFmhaShape::kVHeaddim;
        constexpr index_t kKPerBlock   = [&]() {
            if constexpr(Problem::BlockFmhaShape::kOGradTLoadOnce)
                return Problem::BlockFmhaShape::kM0;
            else
                return Problem::BlockFmhaShape::kK1;
        }();

        return MakeXTLdsBlockDescriptor<kNPerBlock, kKPerBlock, kKPack, PixelsPerRow>();
    }

    template <typename Problem>
    __host__ __device__ static constexpr auto MakeBiasTLdsBlockDescriptor()
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
            make_tuple(Number<kMPerBlock / kKPack>{},
                       Number<kNPerBlock / NPerRow>{},
                       Number<NPerRow>{},
                       Number<kKPack>{}),
            make_tuple(Number<(kNPerBlock / NPerRow) * (PixelsPerRow + kKPack)>{},
                       Number<PixelsPerRow + kKPack>{},
                       Number<kKPack>{},
                       Number<1>{}),
            Number<kKPack>{},
            Number<1>{});

        constexpr auto biast_lds_block_desc = transform_tensor_descriptor(
            biast_lds_block_desc_0,
            make_tuple(
                make_merge_transform(make_tuple(Number<kNPerBlock / NPerRow>{}, Number<NPerRow>{})),
                make_merge_transform(make_tuple(Number<kMPerBlock / kKPack>{}, Number<kKPack>{}))),
            make_tuple(Sequence<1, 2>{}, Sequence<0, 3>{}),
            make_tuple(Sequence<1>{}, Sequence<0>{}));

        return biast_lds_block_desc;
    }

    template <typename Problem>
    __host__ __device__ static constexpr ck::index_t GetSmemSizeQ()
    {
        constexpr index_t smem_size_q = sizeof(typename Problem::QDataType) *
                                        MakeQLdsBlockDescriptor<Problem>().GetElementSpaceSize();
        return smem_size_q;
    }

    template <typename Problem>
    __host__ __device__ static constexpr ck::index_t GetSmemSizeQT()
    {
        constexpr index_t smem_size_qt = sizeof(typename Problem::QDataType) *
                                         MakeQTLdsBlockDescriptor<Problem>().GetElementSpaceSize();
        return smem_size_qt;
    }

    template <typename Problem>
    __host__ __device__ static constexpr ck::index_t GetSmemSizeK()
    {
        constexpr index_t smem_size_k = sizeof(typename Problem::KDataType) *
                                        MakeKLdsBlockDescriptor<Problem>().GetElementSpaceSize();
        return smem_size_k;
    }

    template <typename Problem>
    __host__ __device__ static constexpr ck::index_t GetSmemSizeKT()
    {
        constexpr index_t smem_size_kt = sizeof(typename Problem::KDataType) *
                                         MakeKTLdsBlockDescriptor<Problem>().GetElementSpaceSize();
        return smem_size_kt;
    }

    template <typename Problem>
    __host__ __device__ static constexpr ck::index_t GetSmemSizeV()
    {
        constexpr index_t smem_size_v = sizeof(typename Problem::VDataType) *
                                        MakeVLdsBlockDescriptor<Problem>().GetElementSpaceSize();
        return smem_size_v;
    }

    template <typename Problem>
    __host__ __device__ static constexpr ck::index_t GetSmemSizeOGrad()
    {
        constexpr index_t smem_size_do =
            sizeof(typename Problem::OGradDataType) *
            MakeOGradLdsBlockDescriptor<Problem>().GetElementSpaceSize();
        return smem_size_do;
    }

    template <typename Problem>
    __host__ __device__ static constexpr ck::index_t GetSmemSizeOGradT()
    {
        constexpr index_t smem_size_dot =
            sizeof(typename Problem::OGradDataType) *
            MakeOGradTLdsBlockDescriptor<Problem>().GetElementSpaceSize();
        return smem_size_dot;
    }

    template <typename Problem>
    __host__ __device__ static constexpr ck::index_t GetSmemSizeSGrad()
    {
        constexpr index_t smem_size_ds =
            sizeof(typename Problem::GemmDataType) *
            MakeSGradLdsBlockDescriptor<Problem>().GetElementSpaceSize();
        return smem_size_ds;
    }

    template <typename Problem>
    __host__ __device__ static constexpr ck::index_t GetSmemSizeBias()
    {
        constexpr index_t smem_size_bias =
            sizeof(typename Problem::BiasDataType) *
            MakeBiasTLdsBlockDescriptor<Problem>().GetElementSpaceSize();
        return smem_size_bias;
    }

    template <typename Problem>
    __host__ __device__ static constexpr ck::index_t GetSmemSize()
    {
        constexpr index_t smem_size_q  = GetSmemSizeQ<Problem>();
        constexpr index_t smem_size_qt = [&]() {
            if constexpr(Problem::BlockFmhaShape::kQLoadOnce &&
                         !Problem::BlockFmhaShape::kQTLoadOnce)
                return 0;
            else
                return GetSmemSizeQT<Problem>();
        }();

        constexpr index_t smem_size_k  = GetSmemSizeK<Problem>();
        constexpr index_t smem_size_kt = [&]() {
            if constexpr(Problem::BlockFmhaShape::kKLoadOnce &&
                         !Problem::BlockFmhaShape::kKTLoadOnce)
                return 0;
            else
                return GetSmemSizeKT<Problem>();
        }();

        constexpr index_t smem_size_v = [&]() {
            if constexpr(Problem::BlockFmhaShape::kVLoadOnce)
                return 0;
            else
                return GetSmemSizeV<Problem>();
        }();

        constexpr index_t smem_size_do  = GetSmemSizeOGrad<Problem>();
        constexpr index_t smem_size_dot = [&]() {
            if constexpr(Problem::BlockFmhaShape::kOGradLoadOnce &&
                         !Problem::BlockFmhaShape::kOGradTLoadOnce)
                return 0;
            else
                return GetSmemSizeOGradT<Problem>();
        }();

        constexpr index_t smem_size_ds   = GetSmemSizeSGrad<Problem>();
        constexpr index_t smem_size_bias = [&]() {
            if constexpr(Problem::kHasBias)
                return GetSmemSizeBias<Problem>();
            else
                return 0;
        }();
        constexpr index_t smem_size_transpose = math::max(smem_size_ds, smem_size_bias);

        index_t smem_size = 0;

        if constexpr(Problem::BlockFmhaShape::kQLoadOnce && Problem::BlockFmhaShape::kOGradLoadOnce)
            smem_size += smem_size_q + smem_size_qt + smem_size_do + smem_size_dot +
                         smem_size_transpose; // 1~4 & 10
        else if(Problem::BlockFmhaShape::kQLoadOnce && !Problem::BlockFmhaShape::kOGradLoadOnce &&
                !Problem::BlockFmhaShape::kOGradTLoadOnce)
            smem_size += smem_size_q + smem_size_qt +
                         math::max(smem_size_do,
                                   smem_size_dot,
                                   smem_size_transpose); // 5/7/11 TODO: Multiple buffers strategy
        else if(!Problem::BlockFmhaShape::kQLoadOnce && !Problem::BlockFmhaShape::kQTLoadOnce &&
                Problem::BlockFmhaShape::kOGradLoadOnce)
            smem_size += smem_size_do + smem_size_dot +
                         math::max(smem_size_q,
                                   smem_size_qt,
                                   smem_size_transpose); // 6/8/12 TODO: Multiple buffers strategy
        else if(!Problem::BlockFmhaShape::kQLoadOnce && !Problem::BlockFmhaShape::kQTLoadOnce &&
                !Problem::BlockFmhaShape::kOGradLoadOnce &&
                !Problem::BlockFmhaShape::kOGradTLoadOnce)
            smem_size += math::max(smem_size_q,
                                   smem_size_qt,
                                   smem_size_do,
                                   smem_size_dot,
                                   smem_size_transpose); // 9/13 TODO: Multiple buffers strategy

        // 14/15 needs to be adjusted
        if constexpr(Problem::BlockFmhaShape::kKLoadOnce)
            smem_size += (smem_size_k + smem_size_kt); // 1~13
        else
            smem_size = math::max(
                smem_size_k, smem_size_kt, smem_size); // 14/15 TODO: Multiple buffers strategy

        return math::max(smem_size, smem_size_v); // 15
    }

    template <typename Problem, typename BlockGemm>
    __host__ __device__ static constexpr auto MakeLSEDDramTileDistribution()
    {
        constexpr auto config   = BlockGemm::Policy::template GetWarpGemmMWarpNWarp<Problem>();
        using WG                = remove_cvref_t<decltype(config.template At<0>())>;
        constexpr index_t MWarp = config.template At<1>();
        constexpr index_t NWarp = config.template At<2>();

        constexpr index_t kMPerBlock = Problem::BlockFmhaShape::kM0;

        constexpr index_t N1 = WG::WarpGemmAttribute::Impl::kCNLane;
        constexpr index_t N0 = NWarp;

        constexpr index_t M4 = WG::WarpGemmAttribute::Impl::kCM1PerLane * 2;
        constexpr index_t M3 = WG::WarpGemmAttribute::Impl::kCMLane;
        constexpr index_t M2 = WG::WarpGemmAttribute::Impl::kCM0PerLane / 2;
        constexpr index_t M1 = MWarp;
        constexpr index_t M0 = kMPerBlock / (M1 * WG::WarpGemmAttribute::Impl::kM);

        return make_static_tile_distribution(
            StaticTileDistributionEncoding<Sequence<N0, N1>,
                                           Tuple<Sequence<M0, M1, M2, M3, M4>>,
                                           Tuple<Sequence<1, 0>, Sequence<1, 0>>,
                                           Tuple<Sequence<1, 0>, Sequence<3, 1>>,
                                           Sequence<1, 1, 1>,
                                           Sequence<0, 2, 4>>{});
    }

    template <typename Problem>
    __host__ __device__ static constexpr auto MakeVDramRegTempTileDistribution()
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
            StaticTileDistributionEncoding<Sequence<1>,
                                           Tuple<Sequence<N0, N1, N2>, Sequence<K0, K1>>,
                                           Tuple<Sequence<1>, Sequence<1, 2>>,
                                           Tuple<Sequence<1>, Sequence<2, 0>>,
                                           Sequence<1, 2>,
                                           Sequence<0, 1>>{});
    }

    template <typename Problem>
    __host__ __device__ static constexpr auto MakeQDramTileDistribution()
    {
        using QDataType = remove_cvref_t<typename Problem::QDataType>;

        constexpr index_t kBlockSize = Problem::kBlockSize;

        constexpr index_t kMPerBlock = Problem::BlockFmhaShape::kM0;
        constexpr index_t kKPerBlock = [&]() {
            if constexpr(Problem::BlockFmhaShape::kQLoadOnce)
                return Problem::BlockFmhaShape::kQKHeaddim;
            else
                return Problem::BlockFmhaShape::kK0;
        }();

        constexpr index_t K1 = 16 / sizeof(QDataType);
        constexpr index_t K0 = kKPerBlock / K1;
        constexpr index_t M2 = get_warp_size() / K0;
        // coalesce reading for each blocks
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

    template <typename Problem>
    __host__ __device__ static constexpr auto MakeKDramTileDistribution()
    {
        using KDataType = remove_cvref_t<typename Problem::KDataType>;

        constexpr index_t kBlockSize = Problem::kBlockSize;

        constexpr index_t kNPerBlock = Problem::BlockFmhaShape::kN0;
        constexpr index_t kKPerBlock = [&]() {
            if constexpr(Problem::BlockFmhaShape::kKLoadOnce)
                return Problem::BlockFmhaShape::kQKHeaddim;
            else
                return Problem::BlockFmhaShape::kK0;
        }();

        constexpr index_t K1 = 16 / sizeof(KDataType);
        constexpr index_t K0 = kKPerBlock / K1;
        constexpr index_t N2 = get_warp_size() / K0;
        // coalesce reading for each blocks
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

    template <typename Problem>
    __host__ __device__ static constexpr auto MakeOGradDramTileDistribution()
    {
        using OGradDataType = remove_cvref_t<typename Problem::OGradDataType>;

        constexpr index_t kBlockSize = Problem::kBlockSize;

        constexpr index_t kMPerBlock = Problem::BlockFmhaShape::kM0;
        constexpr index_t kKPerBlock = [&]() {
            if constexpr(Problem::BlockFmhaShape::kOGradLoadOnce)
                return Problem::BlockFmhaShape::kVHeaddim;
            else
                return Problem::BlockFmhaShape::kK2;
        }();

        constexpr index_t K1 = 16 / sizeof(OGradDataType);
        constexpr index_t K0 = kKPerBlock / K1;
        constexpr index_t M2 = get_warp_size() / K0;
        // coalesce reading for each blocks
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

    template <typename DataType, index_t BlockSize, index_t KPerBlock>
    __host__ __device__ static constexpr auto MakePreXDramTileDistribution()
    {
        constexpr index_t K1 = 16 / sizeof(DataType);
        constexpr index_t K0 = KPerBlock / K1;
        constexpr index_t M2 = 1;
        constexpr index_t M1 = get_warp_size();
        constexpr index_t M0 = BlockSize / M1;

        return make_static_tile_distribution(
            StaticTileDistributionEncoding<Sequence<1>,
                                           Tuple<Sequence<M0, M1, M2>, Sequence<K0, K1>>,
                                           Tuple<Sequence<1>, Sequence<1>>,
                                           Tuple<Sequence<0>, Sequence<1>>,
                                           Sequence<1, 2, 2>,
                                           Sequence<2, 0, 1>>{});
    }

    template <typename Problem>
    __host__ __device__ static constexpr auto MakePreODramTileDistribution()
    {
        using ODataType = remove_cvref_t<typename Problem::ODataType>;

        constexpr index_t kBlockSize = Problem::kBlockSize;
        constexpr index_t kKPerBlock = Problem::kVHeaddim;

        return MakePreXDramTileDistribution<ODataType, kBlockSize, kKPerBlock>();
    }

    template <typename Problem>
    __host__ __device__ static constexpr auto MakePreOGradDramTileDistribution()
    {
        using OGradDataType = remove_cvref_t<typename Problem::OGradDataType>;

        constexpr index_t kBlockSize = Problem::kBlockSize;
        constexpr index_t kKPerBlock = Problem::kVHeaddim;

        return MakePreXDramTileDistribution<OGradDataType, kBlockSize, kKPerBlock>();
    }

    template <typename Problem>
    __host__ __device__ static constexpr auto MakePreDDramTileDistribution()
    {
        constexpr index_t kBlockSize = Problem::kBlockSize;

        constexpr index_t K0 = 1;
        constexpr index_t M2 = 1;
        constexpr index_t M1 = get_warp_size();
        constexpr index_t M0 = kBlockSize / M1;

        return make_static_tile_distribution(
            StaticTileDistributionEncoding<Sequence<1>,
                                           Tuple<Sequence<M0, M1, M2>, Sequence<K0>>,
                                           Tuple<Sequence<1>, Sequence<1>>,
                                           Tuple<Sequence<0>, Sequence<1>>,
                                           Sequence<1, 2>,
                                           Sequence<2, 0>>{});
    }

    template <typename Problem>
    __device__ static constexpr auto MakeQTDramTileDistribution()
    {
        constexpr index_t kBlockSize = Problem::kBlockSize;
        constexpr index_t kNPerBlock = Problem::BlockFmhaShape::kQKHeaddim;
        constexpr index_t kKPerBlock = [&]() {
            if constexpr(Problem::BlockFmhaShape::kQTLoadOnce)
                return Problem::BlockFmhaShape::kM0;
            else
                return Problem::BlockFmhaShape::kK3;
        }();

        constexpr index_t N1 = GetTransposedVectorloadQ<Problem>();
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
            StaticTileDistributionEncoding<Sequence<1>,
                                           Tuple<Sequence<N0, N1>, Sequence<K0, K1, K2, K3>>,
                                           Tuple<Sequence<2>, Sequence<2, 1, 2>>,
                                           Tuple<Sequence<0>, Sequence<1, 0, 2>>,
                                           Sequence<2, 1>,
                                           Sequence<3, 1>>{});
    }

    template <typename Problem>
    __host__ __device__ static constexpr auto MakeShuffledQTRegBlockDescriptor()
    {
        constexpr index_t kBlockSize = Problem::kBlockSize;
        constexpr index_t kNPerBlock = Problem::BlockFmhaShape::kQKHeaddim;
        constexpr index_t kKPerBlock = [&]() {
            if constexpr(Problem::BlockFmhaShape::kQTLoadOnce)
                return Problem::BlockFmhaShape::kM0;
            else
                return Problem::BlockFmhaShape::kK3;
        }();

        constexpr index_t N1           = GetTransposedVectorloadQ<Problem>();
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
            StaticTileDistributionEncoding<Sequence<1>,
                                           Tuple<Sequence<N0, N1>, Sequence<K0, K1, K2, K3>>,
                                           Tuple<Sequence<2>, Sequence<2, 1, 2>>,
                                           Tuple<Sequence<0>, Sequence<1, 0, 2>>,
                                           Sequence<1, 2>,
                                           Sequence<1, 3>>{});
    }

    template <typename Problem>
    __device__ static constexpr auto MakeKTDramTileDistribution()
    {
        constexpr index_t kBlockSize = Problem::kBlockSize;
        constexpr index_t kNPerBlock = Problem::BlockFmhaShape::kQKHeaddim;
        constexpr index_t kKPerBlock = [&]() {
            if constexpr(Problem::BlockFmhaShape::kKTLoadOnce)
                return Problem::BlockFmhaShape::kN0;
            else
                return Problem::BlockFmhaShape::kK4;
        }();

        constexpr index_t N1 = GetTransposedVectorloadK<Problem>();
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
            StaticTileDistributionEncoding<Sequence<1>,
                                           Tuple<Sequence<N0, N1>, Sequence<K0, K1, K2, K3>>,
                                           Tuple<Sequence<2>, Sequence<2, 1, 2>>,
                                           Tuple<Sequence<0>, Sequence<1, 0, 2>>,
                                           Sequence<2, 1>,
                                           Sequence<3, 1>>{});
    }

    template <typename Problem>
    __host__ __device__ static constexpr auto MakeShuffledKTRegBlockDescriptor()
    {
        constexpr index_t kBlockSize = Problem::kBlockSize;
        constexpr index_t kNPerBlock = Problem::BlockFmhaShape::kQKHeaddim;
        constexpr index_t kKPerBlock = [&]() {
            if constexpr(Problem::BlockFmhaShape::kKTLoadOnce)
                return Problem::BlockFmhaShape::kN0;
            else
                return Problem::BlockFmhaShape::kK4;
        }();

        constexpr index_t N1           = GetTransposedVectorloadK<Problem>();
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
            StaticTileDistributionEncoding<Sequence<1>,
                                           Tuple<Sequence<N0, N1>, Sequence<K0, K1, K2, K3>>,
                                           Tuple<Sequence<2>, Sequence<2, 1, 2>>,
                                           Tuple<Sequence<0>, Sequence<1, 0, 2>>,
                                           Sequence<1, 2>,
                                           Sequence<1, 3>>{});
    }

    template <typename Problem>
    __device__ static constexpr auto MakeOGradTDramTileDistribution()
    {
        constexpr index_t kBlockSize = Problem::kBlockSize;
        constexpr index_t kNPerBlock = Problem::BlockFmhaShape::kVHeaddim;
        constexpr index_t kKPerBlock = [&]() {
            if constexpr(Problem::BlockFmhaShape::kOGradTLoadOnce)
                return Problem::BlockFmhaShape::kM0;
            else
                return Problem::BlockFmhaShape::kK1;
        }();

        constexpr index_t N1 = GetTransposedVectorloadOGrad<Problem>();
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
            StaticTileDistributionEncoding<Sequence<1>,
                                           Tuple<Sequence<N0, N1>, Sequence<K0, K1, K2, K3>>,
                                           Tuple<Sequence<2>, Sequence<2, 1, 2>>,
                                           Tuple<Sequence<0>, Sequence<1, 0, 2>>,
                                           Sequence<2, 1>,
                                           Sequence<3, 1>>{});
    }

    template <typename Problem>
    __host__ __device__ static constexpr auto MakeShuffledOGradTRegBlockDescriptor()
    {
        constexpr index_t kBlockSize = Problem::kBlockSize;
        constexpr index_t kNPerBlock = Problem::BlockFmhaShape::kVHeaddim;
        constexpr index_t kKPerBlock = [&]() {
            if constexpr(Problem::BlockFmhaShape::kOGradTLoadOnce)
                return Problem::BlockFmhaShape::kM0;
            else
                return Problem::BlockFmhaShape::kK1;
        }();

        constexpr index_t N1           = GetTransposedVectorloadOGrad<Problem>();
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
            StaticTileDistributionEncoding<Sequence<1>,
                                           Tuple<Sequence<N0, N1>, Sequence<K0, K1, K2, K3>>,
                                           Tuple<Sequence<2>, Sequence<2, 1, 2>>,
                                           Tuple<Sequence<0>, Sequence<1, 0, 2>>,
                                           Sequence<1, 2>,
                                           Sequence<1, 3>>{});
    }

    template <typename Problem>
    __device__ static constexpr auto MakeBiasTileDistribution()
    {
        constexpr index_t kBlockSize = Problem::kBlockSize;
        constexpr index_t kMPerBlock = Problem::BlockFmhaShape::kM0;
        constexpr index_t kNPerBlock = Problem::BlockFmhaShape::kN0;

        constexpr index_t N1 = GetTransposedVectorloadBias<Problem>();
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
            StaticTileDistributionEncoding<Sequence<1>,
                                           Tuple<Sequence<M0, M1, M2, M3>, Sequence<N0, N1>>,
                                           Tuple<Sequence<1>, Sequence<1, 2, 1>>,
                                           Tuple<Sequence<0>, Sequence<1, 0, 2>>,
                                           Sequence<1, 2>,
                                           Sequence<3, 1>>{});
    }

    template <typename Problem>
    __host__ __device__ static constexpr auto MakeShuffledBiasTileDistribution()
    {
        constexpr index_t kBlockSize = Problem::kBlockSize;
        constexpr index_t kMPerBlock = Problem::BlockFmhaShape::kM0;
        constexpr index_t kNPerBlock = Problem::BlockFmhaShape::kN0;

        constexpr index_t N1           = GetTransposedVectorloadBias<Problem>();
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
            StaticTileDistributionEncoding<Sequence<1>,
                                           Tuple<Sequence<M0, M1, M2, M3>, Sequence<N0, N1>>,
                                           Tuple<Sequence<1>, Sequence<1, 2, 1>>,
                                           Tuple<Sequence<0>, Sequence<1, 0, 2>>,
                                           Sequence<2, 1>,
                                           Sequence<1, 3>>{});
    }

    template <typename Problem, typename BlockGemm>
    __host__ __device__ static constexpr auto MakeBiasTTileDistribution()
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
    __host__ __device__ static constexpr auto GetQKBlockGemm()
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
            if constexpr(is_same_v<typename Problem::QDataType, half_t> &&
                         is_same_v<typename Problem::KDataType, half_t> &&
                         is_same_v<typename Problem::AccDataType, float>)
            {
                return warp::WarpGemmMfmaF16F16F32M32N32K16SwizzleA{};
            }
            else if constexpr(is_same_v<typename Problem::QDataType, bhalf_t> &&
                              is_same_v<typename Problem::KDataType, bhalf_t> &&
                              is_same_v<typename Problem::AccDataType, float>)
            {
                return warp::WarpGemmMfmaBf16Bf16F32M32N32K16SwizzleA{};
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
    __host__ __device__ static constexpr auto GetPTOGradTBlockGemm()
    {
        using BlockGemmProblem =
            BlockGemmPipelineProblem<typename Problem::GemmDataType,
                                     typename Problem::OGradDataType,
                                     typename Problem::AccDataType,
                                     Problem::kBlockSize,
                                     TileGemmShape<Problem::BlockFmhaShape::kN0,
                                                   Problem::BlockFmhaShape::kVHeaddim,
                                                   Problem::BlockFmhaShape::kK1>>;

        using WarpGemm = ck::tile_program::warp::WarpGemmMfmaDispatcher<
            typename Problem::GemmDataType,
            typename Problem::OGradDataType,
            typename Problem::AccDataType,
            Problem::BlockFmhaShape::Gemm1WarpTile::At(Number<0>{}),
            Problem::BlockFmhaShape::Gemm1WarpTile::At(Number<1>{}),
            Problem::BlockFmhaShape::Gemm1WarpTile::At(Number<2>{}),
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
    __host__ __device__ static constexpr auto GetOGradVBlockGemm()
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
            if constexpr(is_same_v<typename Problem::OGradDataType, half_t> &&
                         is_same_v<typename Problem::VDataType, half_t> &&
                         is_same_v<typename Problem::AccDataType, float>)
            {
                return warp::WarpGemmMfmaF16F16F32M32N32K16SwizzleA{};
            }
            else if constexpr(is_same_v<typename Problem::OGradDataType, bhalf_t> &&
                              is_same_v<typename Problem::VDataType, bhalf_t> &&
                              is_same_v<typename Problem::AccDataType, float>)
            {
                return warp::WarpGemmMfmaBf16Bf16F32M32N32K16SwizzleA{};
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
    // __host__ __device__ static constexpr auto GetOGradVBlockGemm()
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
    //         if constexpr(is_same_v<typename Problem::OGradDataType, half_t> &&
    //                      is_same_v<typename Problem::VDataType, half_t> &&
    //                      is_same_v<typename Problem::AccDataType, float>)
    //         {
    //             return warp::WarpGemmMfmaF16F16F32M32N32K16SwizzleA{};
    //         }
    //         else if constexpr(is_same_v<typename Problem::OGradDataType, bhalf_t> &&
    //                           is_same_v<typename Problem::VDataType, bhalf_t> &&
    //                           is_same_v<typename Problem::AccDataType, float>)
    //         {
    //             return warp::WarpGemmMfmaBf16Bf16F32M32N32K16SwizzleA{};
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
    __host__ __device__ static constexpr auto GetSGradTQTBlockGemm()
    {
        using BlockGemmProblem =
            BlockGemmPipelineProblem<typename Problem::GemmDataType,
                                     typename Problem::QDataType,
                                     typename Problem::AccDataType,
                                     Problem::kBlockSize,
                                     TileGemmShape<Problem::BlockFmhaShape::kN0,
                                                   Problem::BlockFmhaShape::kQKHeaddim,
                                                   Problem::BlockFmhaShape::kK3>>;

        using WarpGemm = ck::tile_program::warp::WarpGemmMfmaDispatcher<
            typename Problem::GemmDataType,
            typename Problem::QDataType,
            typename Problem::AccDataType,
            Problem::BlockFmhaShape::Gemm3WarpTile::At(Number<0>{}),
            Problem::BlockFmhaShape::Gemm3WarpTile::At(Number<1>{}),
            Problem::BlockFmhaShape::Gemm3WarpTile::At(Number<2>{}),
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
    __host__ __device__ static constexpr auto GetSGradKTBlockGemm()
    {
        using BlockGemmProblem =
            BlockGemmPipelineProblem<typename Problem::GemmDataType,
                                     typename Problem::KDataType,
                                     typename Problem::AccDataType,
                                     Problem::kBlockSize,
                                     TileGemmShape<Problem::BlockFmhaShape::kM0,
                                                   Problem::BlockFmhaShape::kQKHeaddim,
                                                   Problem::BlockFmhaShape::kK4>>;

        using WarpGemm = ck::tile_program::warp::WarpGemmMfmaDispatcher<
            typename Problem::GemmDataType,
            typename Problem::KDataType,
            typename Problem::AccDataType,
            Problem::BlockFmhaShape::Gemm4WarpTile::At(Number<0>{}),
            Problem::BlockFmhaShape::Gemm4WarpTile::At(Number<1>{}),
            Problem::BlockFmhaShape::Gemm4WarpTile::At(Number<2>{}),
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

} // namespace block
} // namespace tile_program
} // namespace ck
