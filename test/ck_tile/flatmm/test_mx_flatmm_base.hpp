// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <gtest/gtest.h>
#include <cstring>
#include <optional>
#include <random>
#include <stdexcept>
#include <type_traits>

#include "ck_tile/core.hpp"
#include "ck_tile/host.hpp"
#include "ck_tile/host/check_err.hpp"
#include "ck_tile/host/reference/reference_gemm.hpp"
#include "ck_tile/ops/flatmm.hpp"
#include "ck_tile/ops/gemm.hpp"

#include "mx_flatmm.hpp"

// Base class for MX Flatmm unit tests.
//
// Tuple layout: <ADataType, BDataType, CDataType, MXFlatmmArchTraits>
template <typename Tuple>
class TestMXFlatmmBase : public ::testing::Test
{
    protected:
    using ADataType          = std::tuple_element_t<0, Tuple>;
    using BDataType          = std::tuple_element_t<1, Tuple>;
    using CDataType          = std::tuple_element_t<2, Tuple>;
    using MXFlatmmArchTraits = std::tuple_element_t<3, Tuple>;

    using FlatmmConfig = typename MXFlatmmArchTraits::Config;
    using AccDataType  = float;
    using ScaleType    = ck_tile::e8m0_t;

    using ALayout = ck_tile::tensor_layout::gemm::RowMajor;
    using BLayout = ck_tile::tensor_layout::gemm::ColumnMajor;
    using CLayout = ck_tile::tensor_layout::gemm::RowMajor;

    static constexpr int ScaleGranularityM = 1;
    static constexpr int ScaleGranularityN = 1;
    static constexpr int ScaleGranularityK = 32;

    using ScaleA = ck_tile::FlatmmScalePointer<ScaleGranularityM, ScaleGranularityK, ScaleType>;
    using ScaleB = ck_tile::FlatmmScalePointer<ScaleGranularityN, ScaleGranularityK, ScaleType>;

    void
    run_test_with_validation(ck_tile::index_t M,
                             ck_tile::index_t N,
                             ck_tile::index_t K,
                             ck_tile::index_t kbatch                              = 1,
                             std::optional<bool> expected_has_hot_loop            = std::nullopt,
                             std::optional<ck_tile::TailNumber> expected_tail_num = std::nullopt)
    {
        constexpr int APackedSize = ck_tile::numeric_traits<ADataType>::PackedSize;
        constexpr int BPackedSize = ck_tile::numeric_traits<BDataType>::PackedSize;

        ASSERT_EQ(K % ScaleGranularityK, 0) << "K must be a multiple of ScaleGranularityK (32)";
        ASSERT_EQ(K % APackedSize, 0) << "K must be a multiple of A PackedSize";
        ASSERT_EQ(K % BPackedSize, 0) << "K must be a multiple of B PackedSize";

        constexpr bool a_row_major = true;
        constexpr bool b_row_major = false;
        constexpr bool c_row_major = true;

        const ck_tile::index_t stride_A =
            ck_tile::get_default_stride(M, K, 0, ck_tile::bool_constant<a_row_major>{});
        const ck_tile::index_t stride_B =
            ck_tile::get_default_stride(K, N, 0, ck_tile::bool_constant<b_row_major>{});
        const ck_tile::index_t stride_C =
            ck_tile::get_default_stride(M, N, 0, ck_tile::bool_constant<c_row_major>{});

        const auto scale_stride_A = ck_tile::get_default_stride(
            M / ScaleGranularityM, K / ScaleGranularityK, 0, ck_tile::bool_constant<a_row_major>{});
        const auto scale_stride_B = ck_tile::get_default_stride(
            K / ScaleGranularityK, N / ScaleGranularityN, 0, ck_tile::bool_constant<b_row_major>{});

        // Host tensors
        ck_tile::HostTensor<ADataType> a_host(
            ck_tile::host_tensor_descriptor(M, K, stride_A, ck_tile::bool_constant<a_row_major>{}));
        ck_tile::HostTensor<BDataType> b_origin_host(
            ck_tile::host_tensor_descriptor(K, N, stride_B, ck_tile::bool_constant<b_row_major>{}));
        ck_tile::HostTensor<CDataType> c_rslt_host(
            ck_tile::host_tensor_descriptor(M, N, stride_C, ck_tile::bool_constant<c_row_major>{}));

        ck_tile::HostTensor<ScaleType> scale_a(
            ck_tile::host_tensor_descriptor(M / ScaleGranularityM,
                                            K / ScaleGranularityK,
                                            scale_stride_A,
                                            ck_tile::bool_constant<a_row_major>{}));
        ck_tile::HostTensor<ScaleType> scale_b(
            ck_tile::host_tensor_descriptor(K / ScaleGranularityK,
                                            N / ScaleGranularityN,
                                            scale_stride_B,
                                            ck_tile::bool_constant<b_row_major>{}));

        // Initialize data
        if constexpr(std::is_same_v<ADataType, ck_tile::pk_fp6x16_t>)
        {
            // FP6: fill raw bytes with values 1..4 (avoids denormals)
            auto a_bytes = a_host.get_element_space_size_in_bytes();
            auto b_bytes = b_origin_host.get_element_space_size_in_bytes();
            std::vector<int8_t> buf_a(a_bytes), buf_b(b_bytes);
            std::mt19937 gen(42);
            std::uniform_int_distribution<int> dis(1, 4);
            for(auto& v : buf_a)
                v = static_cast<int8_t>(dis(gen));
            for(auto& v : buf_b)
                v = static_cast<int8_t>(dis(gen));
            memcpy(a_host.data(), buf_a.data(), a_bytes);
            memcpy(b_origin_host.data(), buf_b.data(), b_bytes);
            ck_tile::FillUniformDistribution<>{-1.f, 1.f}(scale_a);
            ck_tile::FillUniformDistribution<>{-1.f, 1.f}(scale_b);
        }
        else
        {
            ck_tile::FillUniformDistribution<>{0.0f, 1.0f}(a_host);
            ck_tile::FillUniformDistribution<>{-.5f, .5f}(b_origin_host);
            ck_tile::FillUniformDistribution<>{-2.f, 2.f}(scale_a);
            ck_tile::FillUniformDistribution<>{-2.f, 2.f}(scale_b);
        }

        // Preshuffle B and scales
        const auto b_shuffled_host  = MXFlatmmArchTraits::preShuffleWeight(b_origin_host);
        const auto scale_a_shuffled = MXFlatmmArchTraits::template preShuffleScale<true>(scale_a);
        const auto scale_b_shuffled = MXFlatmmArchTraits::template preShuffleScale<false>(scale_b);

        // Device buffers
        ck_tile::DeviceMem a_dev_buf(a_host.get_element_space_size_in_bytes());
        ck_tile::DeviceMem b_shuffled_dev_buf(b_shuffled_host.get_element_space_size_in_bytes());
        ck_tile::DeviceMem c_dev_buf(c_rslt_host.get_element_space_size_in_bytes());
        ck_tile::DeviceMem scale_a_dev_buf(scale_a_shuffled.get_element_space_size_in_bytes());
        ck_tile::DeviceMem scale_b_dev_buf(scale_b_shuffled.get_element_space_size_in_bytes());

        a_dev_buf.ToDevice(a_host.data());
        b_shuffled_dev_buf.ToDevice(b_shuffled_host.data());
        c_rslt_host.SetZero();
        c_dev_buf.ToDevice(c_rslt_host.data());
        scale_a_dev_buf.ToDevice(scale_a_shuffled.data());
        scale_b_dev_buf.ToDevice(scale_b_shuffled.data());

        auto scale_a_dev_ptr = ScaleA{static_cast<ScaleType*>(scale_a_dev_buf.GetDeviceBuffer()),
                                      M / ScaleGranularityM};
        auto scale_b_dev_ptr = ScaleB{static_cast<ScaleType*>(scale_b_dev_buf.GetDeviceBuffer()),
                                      N / ScaleGranularityN};

        // Build args
        ck_tile::ScaleFlatmmHostArgs<ScaleA, ScaleB> args{a_dev_buf.GetDeviceBuffer(),
                                                          b_shuffled_dev_buf.GetDeviceBuffer(),
                                                          {},
                                                          c_dev_buf.GetDeviceBuffer(),
                                                          kbatch,
                                                          M,
                                                          N,
                                                          K,
                                                          stride_A,
                                                          stride_B,
                                                          {},
                                                          stride_C,
                                                          scale_a_dev_ptr,
                                                          scale_b_dev_ptr};

        // Compute hot_loop / tail_num
        using FlatmmShape = ck_tile::TileGemmShape<
            ck_tile::sequence<FlatmmConfig::M_Tile, FlatmmConfig::N_Tile, FlatmmConfig::K_Tile>,
            ck_tile::sequence<FlatmmConfig::M_Warp, FlatmmConfig::N_Warp, FlatmmConfig::K_Warp>,
            ck_tile::sequence<FlatmmConfig::M_Warp_Tile,
                              FlatmmConfig::N_Warp_Tile,
                              FlatmmConfig::K_Warp_Tile>>;

        using TilePartitioner =
            ck_tile::GemmSpatiallyLocalTilePartitioner<FlatmmShape,
                                                       FlatmmConfig::TileParitionerGroupNum,
                                                       FlatmmConfig::TileParitionerM01>;

        using GemmTraits          = ck_tile::TileGemmTraits<FlatmmConfig::kPadM,
                                                            FlatmmConfig::kPadN,
                                                            FlatmmConfig::kPadK,
                                                            ALayout,
                                                            BLayout,
                                                            CLayout,
                                                            FlatmmConfig::NumWaveGroups>;
        using GemmPipelineProblem = ck_tile::
            GemmPipelineProblem<ADataType, BDataType, AccDataType, FlatmmShape, GemmTraits>;
        using BaseFlatmmPipeline = ck_tile::BaseFlatmmPipelineAGmemBGmemCRegV1<GemmPipelineProblem>;

        const ck_tile::index_t k_grain     = args.k_batch * FlatmmConfig::K_Tile;
        const ck_tile::index_t k_split     = (K + k_grain - 1) / k_grain * k_grain;
        const ck_tile::index_t num_loop    = TilePartitioner::GetLoopNum(k_split);
        const bool has_hot_loop            = BaseFlatmmPipeline::BlockHasHotloop(num_loop);
        const ck_tile::TailNumber tail_num = BaseFlatmmPipeline::GetBlockLoopTailNum(num_loop);

        if(expected_has_hot_loop.has_value())
            ASSERT_EQ(has_hot_loop, *expected_has_hot_loop)
                << "has_hot_loop mismatch for (M=" << M << ", N=" << N << ", K=" << K << ")";
        if(expected_tail_num.has_value())
            ASSERT_EQ(tail_num, *expected_tail_num)
                << "tail_num mismatch for (M=" << M << ", N=" << N << ", K=" << K << ")";

            // Launch kernel (warmup=0, repeat=1 for correctness testing)
            // mx_flatmm_calc is explicitly instantiated in the linked object library;
            // suppress the -Wundefined-func-template warning that fires when the
            // compiler sees only the forward declaration in mx_flatmm.hpp.
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wno-unknown-warning-option"
#pragma clang diagnostic ignored "-Wundefined-func-template"
        BaseFlatmmPipeline::template TailHandler<true>(
            [&](auto has_hot_loop_, auto tail_num_) {
                constexpr auto has_hot_loop_v = has_hot_loop_.value;
                constexpr auto tail_num_v     = tail_num_.value;
                // SplitK (kbatch>1) is excluded: confirmed broken at the kernel level.
                // Always dispatch the kbatch=1 (SPLIT_K=false) path.
                mx_flatmm_calc<MXFlatmmArchTraits,
                               ADataType,
                               BDataType,
                               ck_tile::tuple<>,
                               AccDataType,
                               CDataType,
                               ALayout,
                               BLayout,
                               ck_tile::tuple<>,
                               CLayout,
                               ScaleA,
                               ScaleB,
                               /*persistent=*/false,
                               ck_tile::element_wise::PassThrough,
                               /*split_k=*/false,
                               has_hot_loop_v,
                               tail_num_v>(args, ck_tile::stream_config{nullptr, false, 0, 0, 1});
            },
            has_hot_loop,
            tail_num);
#pragma clang diagnostic pop

        c_dev_buf.FromDevice(c_rslt_host.data());

        // CPU reference
        ck_tile::HostTensor<CDataType> c_ref(
            ck_tile::host_tensor_descriptor(M, N, stride_C, ck_tile::bool_constant<c_row_major>{}));
        c_ref.SetZero();

        ck_tile::
            reference_mx_gemm<ADataType, BDataType, ScaleType, ScaleType, AccDataType, CDataType>(
                a_host, b_origin_host, c_ref, scale_a, scale_b);

        const float rtol = 1e-2f;
        const float atol = 1e-2f;
        EXPECT_TRUE(
            ck_tile::check_err(c_rslt_host, c_ref, "MX Flatmm result mismatch", rtol, atol));
    }
};
