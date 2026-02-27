// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "utils/ckb_conv_test_configs.hpp"
#include "utils/ckb_conv_test_utils.hpp"
#include "utils/conv_algorithm_type_utils.hpp"

namespace {

using namespace ck_tile::builder::test_utils;

// 3D FP32 NGCDHW (channels-first) with Pipeline V1 and FILTER_1X1_PAD0
TEST(FwdConvInstances,
     Create_DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3_Instance_3D_FP32_ChannelsFirst)
{
    using enum ck_tile::builder::ConvDirection;
    using enum ck_tile::builder::DataType;
    using enum ck_tile::builder::TensorLayout;

    constexpr ConvSignature FwdConvSignature{.spatial_dim            = 3,
                                             .direction              = FORWARD,
                                             .data_type              = FP32,
                                             .accumulation_data_type = FP32,
                                             .input  = {.config = {.layout = NGCDHW}},
                                             .weight = {.config = {.layout = GKCZYX}},
                                             .output = {.config = {.layout = NGKDHW}}};

    constexpr auto FwdConvAlgorithm =
        ConvAlgorithm_DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3{}
            .with_thread_block(ThreadBlock_256_256x256x32)
            .with_gemm_config(FwdGemmParams_Xdl_4x4_per_wave)
            .with_transfer(Transfer_4x64x1)
            .with_fwd_specializations(ConvSpecialization::FILTER_1X1_PAD0,
                                      GemmSpecialization::MNKPadding)
            .with_block_gemm(BlockGemmDesc_v1_intrawave)
            .with_num_conv_groups_to_merge(1);

    using Builder = ConvBuilder<FwdConvSignature, FwdConvAlgorithm>;

    const auto expected_transfer_parameters = to_string(FwdConvAlgorithm);
    run_test<Builder>({"DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3",
                       expected_transfer_parameters,
                       "Filter1x1Pad0",
                       "Intrawave",
                       "v1",
                       "NGCDHW,GKCZYX,EmptyTuple,NGKDHW",
                       "PassThrough,PassThrough,PassThrough",
                       "MNKPadding"});
}

// 3D FP32 NGCDHW (channels-first) with Pipeline V1 and FILTER_1X1_PAD0
TEST(
    FwdConvInstances,
    Create_DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3_Instance_3D_FP32_ChannelsFirst_LargeVecSize)
{
    using enum ck_tile::builder::ConvDirection;
    using enum ck_tile::builder::DataType;
    using enum ck_tile::builder::TensorLayout;

    constexpr ConvSignature FwdConvSignature{.spatial_dim            = 3,
                                             .direction              = FORWARD,
                                             .data_type              = FP32,
                                             .accumulation_data_type = FP32,
                                             .input  = {.config = {.layout = NGCDHW}},
                                             .weight = {.config = {.layout = GKCZYX}},
                                             .output = {.config = {.layout = NGKDHW}}};

    constexpr Transfer<> Transfer_4x64x1_Vec16{
        .a =
            {
                .block_transfer               = {.k0 = 2, .m_n = 128, .k1 = 1},
                .lds_transfer                 = {.src_vector_dim            = 2,
                                                 .src_scalar_per_vector     = 16,
                                                 .lds_dst_scalar_per_vector = 4,
                                                 .is_direct_load            = false,
                                                 .lds_padding               = false},
                .thread_cluster_arrange_order = {1, 0, 2},
                .src_access_order             = {1, 0, 2},
            },
        .b =
            {
                .block_transfer               = {.k0 = 4, .m_n = 64, .k1 = 1},
                .lds_transfer                 = {.src_vector_dim            = 2,
                                                 .src_scalar_per_vector     = 4,
                                                 .lds_dst_scalar_per_vector = 4,
                                                 .is_direct_load            = false,
                                                 .lds_padding               = false},
                .thread_cluster_arrange_order = {1, 0, 2},
                .src_access_order             = {1, 0, 2},
            },
        .c =
            {
                .thread_cluster_dims =
                    {.m_block = 1, .m_wave_per_xdl = 32, .n_block = 1, .n_wave_per_xdl = 8},
                .epilogue = {.m_xdl_per_wave_per_shuffle = 1,
                             .n_xdl_per_wave_per_shuffle = 1,
                             .scalar_per_vector          = 4},
            },
    };

    constexpr GridwiseFwdXdlGemm FwdGemmParams{
        .ak1        = 16,
        .bk1        = 8,
        .xdl_params = {.m_per_xdl = 32, .n_per_xdl = 32, .m_xdl_per_wave = 4, .n_xdl_per_wave = 4}};

    constexpr auto FwdConvAlgorithm =
        ConvAlgorithm_DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3{}
            .with_thread_block(ThreadBlock_256_256x256x32)
            .with_gemm_config(FwdGemmParams)
            .with_transfer(Transfer_4x64x1_Vec16)
            .with_fwd_specializations(ConvSpecialization::FILTER_1X1_PAD0,
                                      GemmSpecialization::MNKPadding)
            .with_block_gemm(BlockGemmDesc_v1_intrawave);

    using Builder = ConvBuilder<FwdConvSignature, FwdConvAlgorithm>;

    const auto expected_transfer_parameters = to_string(FwdConvAlgorithm);
    run_test<Builder>({"DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3",
                       expected_transfer_parameters,
                       "Filter1x1Pad0",
                       "Intrawave",
                       "v1",
                       "NGCDHW,GKCZYX,EmptyTuple,NGKDHW",
                       "PassThrough,PassThrough,PassThrough",
                       "MNKPadding"});
}

} // namespace
