// SPDX-License-Identifier: MIT
// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "fmha_bwd.hpp"

template <typename T>
struct FmhaBwdKernelImpl<T, std::enable_if_t<is_fmha_bwd_dot_do_o_traits_v<T>>>
{
    static std::string GetName();
    static void Run(const ck_tile::stream_config& s, fmha_bwd_args a)
    {
        constexpr auto HDim         = T::HDim;
        using DataType              = typename T::DataType;
        constexpr auto kIsGroupMode = T::kIsGroupMode;
        constexpr auto kPadS        = T::kPadS;
        constexpr auto kPadDv       = T::kPadDv;

        using dot_do_o_trait = ck_tile::TileFmhaBwdOGradDotOTraits<kPadS, kPadDv, /* occu */ 2>;
        using fmha_bwd_dot_do_o_pipeline_problem_0 = ck_tile::BlockFmhaBwdOGradDotOPipelineProblem<
            typename FmhaBwdTypeConfig<DataType>::ODataType,
            typename FmhaBwdTypeConfig<DataType>::OGradDataType,
            typename FmhaBwdTypeConfig<DataType>::DDataType,
            /* BlockSize = M0 = */ 64,
            HDim,
            kIsGroupMode,
            dot_do_o_trait>;
        using dot_do_o_block =
            typename ck_tile::BlockFmhaBwdOGradDotO<fmha_bwd_dot_do_o_pipeline_problem_0>;

        using k_                               = ck_tile::FmhaBwdOGradDotOKernel<dot_do_o_block>;
        auto [kargs, grids]                    = fmha_bwd_dot_do_o_create_kargs_and_grids<k_>(a);
        const dim3 blocks                      = k_::BlockSize();
        constexpr ck_tile::index_t kBlockPerCu = k_::kBlockPerCu;
        ck_tile::make_kernel<kBlockPerCu>(k_{}, grids, blocks, 0, kargs)(
            ck_tile::stream_config{s.stream_id_});
    }
};

template <typename T>
struct FmhaBwdKernelImpl<T, std::enable_if_t<is_fmha_bwd_dq_dk_dv_traits_v<T>>>
{
    static std::string GetName();
    static void Run(const ck_tile::stream_config& s, fmha_bwd_args a)
    {
        constexpr auto HDim             = T::HDim;
        constexpr auto kM0              = T::kM0;
        constexpr auto kN0              = T::kN0;
        using DataType                  = typename T::DataType;
        constexpr bool kIsGroupMode     = T::kIsGroupMode;
        using FmhaMask                  = typename T::FmhaMask;
        using FmhaDropout               = typename T::FmhaDropout;
        constexpr auto BiasEnum         = T::BiasEnum;
        constexpr auto kHasBiasGrad     = T::kHasBiasGrad;
        constexpr auto kPadD            = T::kPadD;
        constexpr auto kPadDv           = T::kPadDv;
        constexpr auto kIsDeterministic = T::kIsDeterministic;
        constexpr auto kUseTrLoad       = T::kUseTrLoad;
        constexpr auto MaxSeqLenQ       = T::MaxSeqLenQ;

        using QDataType             = typename FmhaBwdTypeConfig<DataType>::QDataType;
        using KDataType             = typename FmhaBwdTypeConfig<DataType>::KDataType;
        using VDataType             = typename FmhaBwdTypeConfig<DataType>::VDataType;
        using GemmDataType          = typename FmhaBwdTypeConfig<DataType>::GemmDataType;
        using LSEDataType           = typename FmhaBwdTypeConfig<DataType>::LSEDataType;
        using AccDataType           = typename FmhaBwdTypeConfig<DataType>::AccDataType;
        using DDataType             = typename FmhaBwdTypeConfig<DataType>::DDataType;
        using BiasDataType          = typename FmhaBwdTypeConfig<DataType>::BiasDataType;
        using RandValOutputDataType = typename FmhaBwdTypeConfig<DataType>::RandValOutputDataType;
        using ODataType             = typename FmhaBwdTypeConfig<DataType>::ODataType;
        using OGradDataType         = typename FmhaBwdTypeConfig<DataType>::OGradDataType;
        using QGradDataType         = typename FmhaBwdTypeConfig<DataType>::QGradDataType;
        using KGradDataType         = typename FmhaBwdTypeConfig<DataType>::KGradDataType;
        using VGradDataType         = typename FmhaBwdTypeConfig<DataType>::VGradDataType;
        using BiasGradDataType      = typename FmhaBwdTypeConfig<DataType>::BiasGradDataType;

        using dqdkdv_traits = ck_tile::TileFmhaTraits< //
            /* kPadSeqLenQ */ false,
            /* kPadSeqLenK */ false,
            kPadD,
            kPadDv,
            /* kHasLogitsSoftCap */ false,
            BiasEnum,
            kHasBiasGrad,
            /* kStoreLSE */ false,
            /* kHasDropout */ false,
            /* kDoFp8StaticQuant */ false,
            /* kBlockPerCu */ 1>;
        using fmha_bwd_shape =
            decltype(get_fmha_bwd_tile_size<HDim, kM0, kN0, kUseTrLoad, MaxSeqLenQ>());

        using fmha_bwd_pipeline_problem = ck_tile::BlockFmhaBwdPipelineProblem< //
            QDataType,
            KDataType,
            VDataType,
            GemmDataType,
            LSEDataType,
            AccDataType,
            DDataType,
            BiasDataType,
            RandValOutputDataType,
            ODataType,
            OGradDataType,
            QGradDataType,
            KGradDataType,
            VGradDataType,
            BiasGradDataType,
            fmha_bwd_shape,
            kIsGroupMode,
            kIsDeterministic,
            FmhaMask,
            FmhaDropout,
            kUseTrLoad,
            dqdkdv_traits>;

        using dqdkdv_pipeline = ck_tile::BlockFmhaBwdDQDKDVPipeline<fmha_bwd_pipeline_problem>;

        using dk_epilogue = ck_tile::Default2DEpilogue<
            ck_tile::Default2DEpilogueProblem<AccDataType, KGradDataType, false, false>>;
        using dv_epilogue = ck_tile::Default2DEpilogue<
            ck_tile::Default2DEpilogueProblem<AccDataType, VGradDataType, false, false>>;
        using dq_epilogue_ = ck_tile::Default2DEpilogue<
            ck_tile::Default2DEpilogueProblem<AccDataType, QGradDataType, false, false>>;
        using dq_epilogue = std::conditional_t<kUseTrLoad, dq_epilogue_, void>;
        using k_ =
            ck_tile::FmhaBwdDQDKDVKernel<dqdkdv_pipeline, dk_epilogue, dv_epilogue, dq_epilogue>;

        auto [kargs, grids]                    = fmha_bwd_dq_dk_dv_create_kargs_and_grids<k_>(a);
        const dim3 blocks                      = k_::BlockSize();
        constexpr ck_tile::index_t kBlockPerCu = k_::kBlockPerCu;
        ck_tile::make_kernel<kBlockPerCu>(k_{}, grids, blocks, 0, kargs)(
            ck_tile::stream_config{s.stream_id_});
    }
};

template <typename T>
struct FmhaBwdKernelImpl<T, std::enable_if_t<is_fmha_bwd_convert_dq_traits_v<T>>>
{
    static std::string GetName();
    static void Run(const ck_tile::stream_config& s, fmha_bwd_args a)
    {
        constexpr auto HDim             = T::HDim;
        using DataType                  = typename T::DataType;
        constexpr bool kIsGroupMode     = T::kIsGroupMode;
        constexpr auto kPadS            = T::kPadS;
        constexpr auto kPadD            = T::kPadD;
        constexpr bool kIsDeterministic = T::kIsDeterministic;

        using AccDataType   = typename FmhaBwdTypeConfig<DataType>::AccDataType;
        using QGradDataType = typename FmhaBwdTypeConfig<DataType>::QGradDataType;

        using convert_dq_trait = ck_tile::TileFmhaBwdConvertQGradTraits<kPadS, kPadD, 2>;

        using convert_dq_pipeline_problem =
            ck_tile::BlockFmhaBwdConvertQGradPipelineProblem<AccDataType,
                                                             QGradDataType,
                                                             /* BlockSize = */ 256,
                                                             /* M0 */ 64,
                                                             128,
                                                             HDim,
                                                             kIsGroupMode,
                                                             kIsDeterministic,
                                                             convert_dq_trait>;
        using fmha_bwd_convert_dq_0 =
            typename ck_tile::BlockFmhaBwdConvertQGrad<convert_dq_pipeline_problem>;
        using fmha_bwd_convert_dq_kernel_0 =
            ck_tile::FmhaBwdConvertQGradKernel<fmha_bwd_convert_dq_0>;

        using k_                               = fmha_bwd_convert_dq_kernel_0;
        auto [kargs, grids]                    = fmha_bwd_convert_dq_create_kargs_and_grids<k_>(a);
        const dim3 blocks                      = k_::BlockSize();
        constexpr ck_tile::index_t kBlockPerCu = k_::kBlockPerCu;
        ck_tile::make_kernel<kBlockPerCu>(k_{}, grids, blocks, 0, kargs)(
            ck_tile::stream_config{s.stream_id_});
    }
};
