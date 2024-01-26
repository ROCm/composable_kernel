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

    // 3d + padding
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
            make_tuple(Sequence<1>{}, Sequence<0>{}));

        return q_lds_block_desc;
    }

    // 3d + padding
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

    // 3d + padding
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
            make_tuple(Sequence<1>{}, Sequence<0>{}));

        return k_lds_block_desc;
    }

    // 3d + padding
    template <typename Problem>
    __host__ __device__ static constexpr auto MakeVLdsBlockDescriptor()
    {
        constexpr index_t kNPerBlock = Problem::BlockFmhaShape::kN0;
        constexpr index_t kKPerBlock = Problem::BlockFmhaShape::kK2;
        constexpr index_t kPad       = 1;
        constexpr index_t kKPack     = GetSmemKPackV<Problem>();

        constexpr auto v_lds_block_desc_0 = make_naive_tensor_descriptor(
            make_tuple(Number<kKPerBlock / kKPack>{}, Number<kNPerBlock>{}, Number<kKPack>{}),
            make_tuple(Number<(kNPerBlock + kPad) * kKPack>{}, Number<kKPack>{}, Number<1>{}),
            Number<kKPack>{},
            Number<1>{});

        constexpr auto v_lds_block_desc = transform_tensor_descriptor(
            v_lds_block_desc_0,
            make_tuple(
                make_pass_through_transform(kNPerBlock),
                make_merge_transform(make_tuple(Number<kKPerBlock / kKPack>{}, Number<kKPack>{}))),
            make_tuple(Sequence<1>{}, Sequence<0, 2>{}),
            make_tuple(Sequence<0>{}, Sequence<1>{}));

        return v_lds_block_desc;
    }

    // 3d + padding
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

        constexpr auto do_lds_block_desc_0 = make_naive_tensor_descriptor(
            make_tuple(Number<kKPerBlock / kKPack>{}, Number<kMPerBlock>{}, Number<kKPack>{}),
            make_tuple(Number<(kMPerBlock + 1) * kKPack>{}, Number<kKPack>{}, Number<1>{}),
            Number<8>{},
            Number<1>{});

        constexpr auto do_lds_block_desc = transform_tensor_descriptor(
            do_lds_block_desc_0,
            make_tuple(make_pass_through_transform(kMPerBlock),
                       make_merge_transform(make_tuple(kKPerBlock / kKPack, kKPack))),
            make_tuple(Sequence<1>{}, Sequence<0, 2>{}),
            make_tuple(Sequence<0>{}, Sequence<1>{}));

        return do_lds_block_desc;
    }

    // 3d + padding
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

        constexpr auto do_lds_block_desc_0 = make_naive_tensor_descriptor(
            make_tuple(Number<kKPerBlock / kKPack>{}, Number<kMPerBlock>{}, Number<kKPack>{}),
            make_tuple(Number<(kMPerBlock + 1) * kKPack>{}, Number<kKPack>{}, Number<1>{}),
            Number<8>{},
            Number<1>{});

        constexpr auto do_lds_block_desc = transform_tensor_descriptor(
            do_lds_block_desc_0,
            make_tuple(make_pass_through_transform(kMPerBlock),
                       make_merge_transform(make_tuple(kKPerBlock / kKPack, kKPack))),
            make_tuple(Sequence<1>{}, Sequence<0, 2>{}),
            make_tuple(Sequence<1>{}, Sequence<0>{}));

        return do_lds_block_desc;
    }

    // 3d + padding
    template <typename Problem>
    __host__ __device__ static constexpr auto MakeSGradLdsBlockDescriptor()
    {
        constexpr index_t kMPerBlock = Problem::BlockFmhaShape::kM0;
        constexpr index_t kKPerBlock = Problem::BlockFmhaShape::kN0;
        constexpr index_t kKPack     = GetSmemKPackSGrad<Problem>();

        constexpr auto ds_lds_block_desc_0 = make_naive_tensor_descriptor(
            make_tuple(Number<kKPerBlock / kKPack>{}, Number<kMPerBlock>{}, Number<kKPack>{}),
            make_tuple(Number<(kMPerBlock + 1) * kKPack>{}, Number<kKPack>{}, Number<1>{}),
            Number<8>{},
            Number<1>{});

        constexpr auto ds_lds_block_desc = transform_tensor_descriptor(
            ds_lds_block_desc_0,
            make_tuple(make_pass_through_transform(kMPerBlock),
                       make_merge_transform(make_tuple(kKPerBlock / kKPack, kKPack))),
            make_tuple(Sequence<1>{}, Sequence<0, 2>{}),
            make_tuple(Sequence<0>{}, Sequence<1>{}));

        return ds_lds_block_desc;
    }

    template <typename Problem>
    __host__ __device__ static constexpr auto MakeQTLdsBlockDescriptor()
    {
        using QDataType                = remove_cvref_t<typename Problem::QDataType>;
        constexpr index_t Banks        = 32; // TODO: need change based on arch
        constexpr index_t PixelsPerRow = Banks * 4 / sizeof(QDataType);
        constexpr index_t kKPack       = GetSmemKPackQ<Problem>();
        static_assert(PixelsPerRow % kKPack == 0);
        constexpr index_t NPerRow    = PixelsPerRow / kKPack;
        constexpr index_t kNPerBlock = Problem::BlockFmhaShape::kQKHeaddim;
        constexpr index_t kKPerBlock = [&]() {
            if constexpr(Problem::BlockFmhaShape::kQTLoadOnce)
                return Problem::BlockFmhaShape::kM0;
            else
                return Problem::BlockFmhaShape::kK3;
        }();
        static_assert(kNPerBlock % NPerRow == 0);
        static_assert(kKPerBlock % kKPack == 0);

        constexpr auto qt_lds_block_desc_0 = make_naive_tensor_descriptor(
            make_tuple(Number<kKPerBlock / kKPack>{},
                       Number<kNPerBlock / NPerRow>{},
                       Number<NPerRow>{},
                       Number<kKPack>{}),
            make_tuple(Number<(kNPerBlock / NPerRow) * (PixelsPerRow + kKPack)>{},
                       Number<PixelsPerRow + kKPack>{},
                       Number<kKPack>{},
                       Number<1>{}),
            Number<kKPack>{},
            Number<1>{});

        constexpr auto qt_lds_block_desc = transform_tensor_descriptor(
            qt_lds_block_desc_0,
            make_tuple(
                make_merge_transform(make_tuple(Number<kNPerBlock / NPerRow>{}, Number<NPerRow>{})),
                make_merge_transform(make_tuple(Number<kKPerBlock / kKPack>{}, Number<kKPack>{}))),
            make_tuple(Sequence<1, 2>{}, Sequence<0, 3>{}),
            make_tuple(Sequence<0>{}, Sequence<1>{}));

        return qt_lds_block_desc;
    }

    template <typename Problem>
    __host__ __device__ static constexpr auto MakeKTLdsBlockDescriptor()
    {
        using KDataType                = remove_cvref_t<typename Problem::KDataType>;
        constexpr index_t Banks        = 32; // TODO: need change based on arch
        constexpr index_t PixelsPerRow = Banks * 4 / sizeof(KDataType);
        constexpr index_t kKPack       = GetSmemKPackK<Problem>();
        static_assert(PixelsPerRow % kKPack == 0);
        constexpr index_t NPerRow    = PixelsPerRow / kKPack;
        constexpr index_t kNPerBlock = Problem::BlockFmhaShape::kQKHeaddim;
        constexpr index_t kKPerBlock = [&]() {
            if constexpr(Problem::BlockFmhaShape::kKTLoadOnce)
                return Problem::BlockFmhaShape::kN0;
            else
                return Problem::BlockFmhaShape::kK4;
        }();
        static_assert(kNPerBlock % NPerRow == 0);
        static_assert(kKPerBlock % kKPack == 0);

        constexpr auto kt_lds_block_desc_0 = make_naive_tensor_descriptor(
            make_tuple(Number<kKPerBlock / kKPack>{},
                       Number<kNPerBlock / NPerRow>{},
                       Number<NPerRow>{},
                       Number<kKPack>{}),
            make_tuple(Number<(kNPerBlock / NPerRow) * (PixelsPerRow + kKPack)>{},
                       Number<PixelsPerRow + kKPack>{},
                       Number<kKPack>{},
                       Number<1>{}),
            Number<kKPack>{},
            Number<1>{});

        constexpr auto kt_lds_block_desc = transform_tensor_descriptor(
            kt_lds_block_desc_0,
            make_tuple(
                make_merge_transform(make_tuple(Number<kNPerBlock / NPerRow>{}, Number<NPerRow>{})),
                make_merge_transform(make_tuple(Number<kKPerBlock / kKPack>{}, Number<kKPack>{}))),
            make_tuple(Sequence<1, 2>{}, Sequence<0, 3>{}),
            make_tuple(Sequence<0>{}, Sequence<1>{}));

        return kt_lds_block_desc;
    }

    template <typename Problem>
    __host__ __device__ static constexpr auto MakeOGradTLdsBlockDescriptor()
    {
        using QGradDataType            = remove_cvref_t<typename Problem::QGradDataType>;
        constexpr index_t Banks        = 32; // TODO: need change based on arch
        constexpr index_t PixelsPerRow = Banks * 4 / sizeof(QGradDataType);
        constexpr index_t kKPack       = GetSmemKPackOGrad<Problem>();
        static_assert(PixelsPerRow % kKPack == 0);
        constexpr index_t NPerRow    = PixelsPerRow / kKPack;
        constexpr index_t kNPerBlock = Problem::BlockFmhaShape::kVHeaddim;
        constexpr index_t kKPerBlock = [&]() {
            if constexpr(Problem::BlockFmhaShape::kOGradTLoadOnce)
                return Problem::BlockFmhaShape::kM0;
            else
                return Problem::BlockFmhaShape::kK1;
        }();
        static_assert(kNPerBlock % NPerRow == 0);
        static_assert(kKPerBlock % kKPack == 0);

        constexpr auto dot_lds_block_desc_0 = make_naive_tensor_descriptor(
            make_tuple(Number<kKPerBlock / kKPack>{},
                       Number<kNPerBlock / NPerRow>{},
                       Number<NPerRow>{},
                       Number<kKPack>{}),
            make_tuple(Number<(kNPerBlock / NPerRow) * (PixelsPerRow + kKPack)>{},
                       Number<PixelsPerRow + kKPack>{},
                       Number<kKPack>{},
                       Number<1>{}),
            Number<kKPack>{},
            Number<1>{});

        constexpr auto dot_lds_block_desc = transform_tensor_descriptor(
            dot_lds_block_desc_0,
            make_tuple(
                make_merge_transform(make_tuple(Number<kNPerBlock / NPerRow>{}, Number<NPerRow>{})),
                make_merge_transform(make_tuple(Number<kKPerBlock / kKPack>{}, Number<kKPack>{}))),
            make_tuple(Sequence<1, 2>{}, Sequence<0, 3>{}),
            make_tuple(Sequence<0>{}, Sequence<1>{}));

        return dot_lds_block_desc;
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

        constexpr index_t smem_size_ds = GetSmemSizeSGrad<Problem>();

        index_t smem_size = 0;

        if constexpr(Problem::BlockFmhaShape::kQLoadOnce && Problem::BlockFmhaShape::kOGradLoadOnce)
            smem_size += smem_size_q + smem_size_qt + smem_size_do + smem_size_dot +
                         smem_size_ds; // 1~4 & 10
        else if(Problem::BlockFmhaShape::kQLoadOnce && !Problem::BlockFmhaShape::kOGradLoadOnce &&
                !Problem::BlockFmhaShape::kOGradTLoadOnce)
            smem_size += smem_size_q + smem_size_qt +
                         math::max(smem_size_do,
                                   smem_size_dot,
                                   smem_size_ds); // 5/7/11 TODO: Multiple buffers strategy
        else if(!Problem::BlockFmhaShape::kQLoadOnce && !Problem::BlockFmhaShape::kQTLoadOnce &&
                Problem::BlockFmhaShape::kOGradLoadOnce)
            smem_size += smem_size_do + smem_size_dot +
                         math::max(smem_size_q,
                                   smem_size_qt,
                                   smem_size_ds); // 6/8/12 TODO: Multiple buffers strategy
        else if(!Problem::BlockFmhaShape::kQLoadOnce && !Problem::BlockFmhaShape::kQTLoadOnce &&
                !Problem::BlockFmhaShape::kOGradLoadOnce &&
                !Problem::BlockFmhaShape::kOGradTLoadOnce)
            smem_size += math::max(smem_size_q,
                                   smem_size_qt,
                                   smem_size_do,
                                   smem_size_dot,
                                   smem_size_ds); // 9/13 TODO: Multiple buffers strategy

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

        constexpr index_t N1 = WG::WarpGemmAttribute::Impl::kCNLane; // 32
        constexpr index_t N0 = NWarp;                                // 4

        constexpr index_t M4 = WG::WarpGemmAttribute::Impl::kCM1PerLane * 2;        // 8
        constexpr index_t M3 = WG::WarpGemmAttribute::Impl::kCMLane;                // 2
        constexpr index_t M2 = WG::WarpGemmAttribute::Impl::kCM0PerLane / 2;        // 2
        constexpr index_t M1 = MWarp;                                               // 1
        constexpr index_t M0 = kMPerBlock / (M1 * WG::WarpGemmAttribute::Impl::kM); // 2/4

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

    template <typename Problem>
    __host__ __device__ static constexpr auto MakePreODramTileDistribution()
    {
        using ODataType = remove_cvref_t<typename Problem::ODataType>;

        constexpr index_t kBlockSize = Problem::kBlockSize;
        constexpr index_t kKPerBlock = Problem::kVHeaddim;

        constexpr index_t K1 = 16 / sizeof(ODataType);
        constexpr index_t K0 = kKPerBlock / K1;
        constexpr index_t M2 = 1;
        constexpr index_t M1 = get_warp_size();
        constexpr index_t M0 = kBlockSize / M1;

        return make_static_tile_distribution(
            StaticTileDistributionEncoding<Sequence<1>,
                                           Tuple<Sequence<M0, M1, M2>, Sequence<K0, K1>>,
                                           Tuple<Sequence<1>, Sequence<1>>,
                                           Tuple<Sequence<0>, Sequence<1>>,
                                           Sequence<1, 2, 2>,
                                           Sequence<2, 0, 1>>{});
    }

    template <typename Problem>
    __host__ __device__ static constexpr auto MakePreOGradDramTileDistribution()
    {
        using OGradDataType = remove_cvref_t<typename Problem::OGradDataType>;

        constexpr index_t kBlockSize = Problem::kBlockSize;
        constexpr index_t kKPerBlock = Problem::kVHeaddim;

        constexpr index_t K1 = 16 / sizeof(OGradDataType);
        constexpr index_t K0 = kKPerBlock / K1;
        constexpr index_t M2 = 1;
        constexpr index_t M1 = get_warp_size();
        constexpr index_t M0 = kBlockSize / M1;

        return make_static_tile_distribution(
            StaticTileDistributionEncoding<Sequence<1>,
                                           Tuple<Sequence<M0, M1, M2>, Sequence<K0, K1>>,
                                           Tuple<Sequence<1>, Sequence<1>>,
                                           Tuple<Sequence<0>, Sequence<1>>,
                                           Sequence<1, 2, 2>,
                                           Sequence<2, 0, 1>>{});
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
