// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <gtest/gtest.h>

#include "ck_tile/core.hpp"
#include "ck_tile/host.hpp"
#include "ck_tile/host/check_err.hpp"
#include "ck_tile/host/reference/reference_gemm.hpp"
#include "test_mx_gemm_config.hpp"
#include "test_mx_gemm_instance.hpp"

template <typename Layout>
static constexpr auto is_row_major(Layout)
{
    return ck_tile::bool_constant<
        std::is_same_v<ck_tile::remove_cvref_t<Layout>, ck_tile::tensor_layout::gemm::RowMajor>>{};
}

template <typename ADataType, typename BDataType, typename AccDataType, typename CDataType>
auto calculate_rtol_atol_mx(ck_tile::index_t K, float max_accumulated_value)
{
    using ComputeType =
        std::conditional_t<sizeof(ADataType) < sizeof(BDataType), ADataType, BDataType>;
    const auto rtol = ck_tile::get_relative_threshold<ComputeType, CDataType, AccDataType>(K);
    const auto atol = ck_tile::get_absolute_threshold<ComputeType, CDataType, AccDataType>(
        max_accumulated_value, K);
    return ck_tile::make_tuple(rtol, atol);
}

template <typename Tuple>
class TestMxGemmUtil : public ::testing::Test
{
    protected:
    using ADataType  = std::tuple_element_t<0, Tuple>;
    using BDataType  = std::tuple_element_t<1, Tuple>;
    using GemmConfig = std::tuple_element_t<2, Tuple>;
    using ALayout    = std::tuple_element_t<3, Tuple>;
    using BLayout    = std::tuple_element_t<4, Tuple>;
    using CLayout    = std::tuple_element_t<5, Tuple>;

    using AccDataType = float;
    using CDataType   = ck_tile::fp16_t;
    using ScaleType   = ck_tile::e8m0_t;
    using ScaleM      = ck_tile::MXScalePointer<ScaleType, 1, 32>;
    using ScaleN      = ck_tile::MXScalePointer<ScaleType, 1, 32>;

    // Pack [MN, K/32] e8m0_t scales into [MN/MNPack, K/32/KPack] int32_t
    // Each int32_t contains MNPack * KPack e8m0_t values with byte layout matching
    // the GPU tile distribution: values are XdlMNThread apart in M and XdlKThread apart in K.
    //   byte[ik * MNPack + imn] = e8m0 at strided (mn, k) position
    // kLast=true for A scales (layout [M, K/32]), kLast=false for B scales (layout [K/32, N])
    template <ck_tile::index_t MNPack      = 2,
              ck_tile::index_t KPack       = 2,
              ck_tile::index_t XdlMNThread = 16,
              ck_tile::index_t XdlKThread  = 4>
    static auto packScalesMNxK(const ck_tile::HostTensor<ck_tile::e8m0_t>& src, bool kLast)
    {
        auto src_lengths                    = src.get_lengths();
        const ck_tile::index_t MN           = kLast ? src_lengths[0] : src_lengths[1];
        const ck_tile::index_t K_scale      = kLast ? src_lengths[1] : src_lengths[0];
        const ck_tile::index_t MN_packed    = MN / MNPack;
        const ck_tile::index_t K_packed     = K_scale / KPack;
        const ck_tile::index_t total_packed = MN_packed * K_packed;

        std::vector<int32_t> packed(total_packed);

        for(ck_tile::index_t packed_mn = 0; packed_mn < MN_packed; packed_mn++)
        {
            for(ck_tile::index_t packed_k = 0; packed_k < K_packed; packed_k++)
            {
                int32_t val               = 0;
                ck_tile::index_t mn_lane  = packed_mn % XdlMNThread;
                ck_tile::index_t mn_group = packed_mn / XdlMNThread;
                ck_tile::index_t k_lane   = packed_k % XdlKThread;
                ck_tile::index_t k_group  = packed_k / XdlKThread;
                for(ck_tile::index_t ik = 0; ik < KPack; ik++)
                {
                    for(ck_tile::index_t imn = 0; imn < MNPack; imn++)
                    {
                        ck_tile::index_t byteIdx = ik * MNPack + imn;
                        ck_tile::index_t orig_mn =
                            mn_group * XdlMNThread * MNPack + imn * XdlMNThread + mn_lane;
                        ck_tile::index_t orig_k =
                            k_group * XdlKThread * KPack + ik * XdlKThread + k_lane;

                        ck_tile::e8m0_t v = kLast ? src(orig_mn, orig_k) : src(orig_k, orig_mn);
                        val |= (static_cast<int32_t>(v.get()) << (byteIdx * 8));
                    }
                }
                packed[packed_mn * K_packed + packed_k] = val;
            }
        }
        return packed;
    }

    void Run(ck_tile::index_t M, ck_tile::index_t N, ck_tile::index_t K)
    {
        const ck_tile::index_t scale_k_size = K / 32;
        const ck_tile::index_t stride_A =
            ck_tile::get_default_stride(M, K, 0, is_row_major(ALayout{}));
        const ck_tile::index_t stride_B =
            ck_tile::get_default_stride(K, N, 0, is_row_major(BLayout{}));
        const ck_tile::index_t stride_C =
            ck_tile::get_default_stride(M, N, 0, is_row_major(CLayout{}));
        const ck_tile::index_t stride_scale_a =
            ck_tile::get_default_stride(M, scale_k_size, 0, is_row_major(ALayout{}));
        const ck_tile::index_t stride_scale_b =
            ck_tile::get_default_stride(scale_k_size, N, 0, is_row_major(BLayout{}));

        ck_tile::HostTensor<ADataType> a_host(
            ck_tile::host_tensor_descriptor(M, K, stride_A, is_row_major(ALayout{})));
        ck_tile::HostTensor<BDataType> b_host(
            ck_tile::host_tensor_descriptor(K, N, stride_B, is_row_major(BLayout{})));
        ck_tile::HostTensor<CDataType> c_host(
            ck_tile::host_tensor_descriptor(M, N, stride_C, is_row_major(CLayout{})));
        ck_tile::HostTensor<ScaleType> scale_a_host(ck_tile::host_tensor_descriptor(
            M, scale_k_size, stride_scale_a, is_row_major(ALayout{})));
        ck_tile::HostTensor<ScaleType> scale_b_host(ck_tile::host_tensor_descriptor(
            scale_k_size, N, stride_scale_b, is_row_major(BLayout{})));

        std::mt19937 gen(42);
        std::uniform_int_distribution<std::uint32_t> fill_seed(0, 500);

        auto gen_scales = [&](auto& scales, float range_min, float range_max) {
            // e8m0_t is basically an exponent of float32
            ck_tile::HostTensor<float> pow2(scales.get_lengths());
            ck_tile::FillUniformDistributionIntegerValue<float>{
                range_min, range_max, fill_seed(gen)}(pow2);
            scales.ForEach([&](auto& self, const auto& i) {
                self(i) = static_cast<ScaleType>(std::exp2(pow2(i)));
            });
        };

        ck_tile::FillUniformDistribution<ADataType>{-2.f, 2.f, fill_seed(gen)}(a_host);
        ck_tile::FillUniformDistribution<BDataType>{-2.f, 2.f, fill_seed(gen)}(b_host);
        gen_scales(scale_a_host, -2, 2);
        gen_scales(scale_b_host, -2, 2);

        // Compute effective XdlPack sizes based on GemmConfig tile dimensions
        constexpr ck_tile::index_t MPerXdl = GemmConfig::M_Warp_Tile;
        constexpr ck_tile::index_t NPerXdl = GemmConfig::N_Warp_Tile;
        constexpr ck_tile::index_t KPerXdl = GemmConfig::K_Warp_Tile;
        constexpr ck_tile::index_t MIterPerWarp =
            GemmConfig::M_Tile / (GemmConfig::M_Warp * MPerXdl);
        constexpr ck_tile::index_t NIterPerWarp =
            GemmConfig::N_Tile / (GemmConfig::N_Warp * NPerXdl);
        constexpr ck_tile::index_t KIterPerWarp = GemmConfig::K_Tile / KPerXdl;

        constexpr ck_tile::index_t MXdlPackEff =
            (MIterPerWarp >= 2 && MIterPerWarp % 2 == 0) ? 2 : 1;
        constexpr ck_tile::index_t NXdlPackEff =
            (NIterPerWarp >= 2 && NIterPerWarp % 2 == 0) ? 2 : 1;
        constexpr ck_tile::index_t KXdlPackEff =
            (KIterPerWarp >= 2 && KIterPerWarp % 2 == 0) ? 2 : 1;

        constexpr ck_tile::index_t XdlMNThread = GemmConfig::M_Warp_Tile;
        constexpr ck_tile::index_t XdlKThread  = 64 / XdlMNThread;

        // Pack scales into int32_t for GPU consumption
        auto scale_a_packed =
            packScalesMNxK<MXdlPackEff, KXdlPackEff, XdlMNThread, XdlKThread>(scale_a_host, true);
        auto scale_b_packed =
            packScalesMNxK<NXdlPackEff, KXdlPackEff, XdlMNThread, XdlKThread>(scale_b_host, false);

        ck_tile::DeviceMem a_dev_buf(a_host.get_element_space_size_in_bytes());
        ck_tile::DeviceMem b_dev_buf(b_host.get_element_space_size_in_bytes());
        ck_tile::DeviceMem c_dev_buf(c_host.get_element_space_size_in_bytes());
        ck_tile::DeviceMem scale_a_dev_buf(scale_a_packed.size() * sizeof(int32_t));
        ck_tile::DeviceMem scale_b_dev_buf(scale_b_packed.size() * sizeof(int32_t));

        a_dev_buf.ToDevice(a_host.data());
        b_dev_buf.ToDevice(b_host.data());
        c_dev_buf.SetZero();
        scale_a_dev_buf.ToDevice(scale_a_packed.data());
        scale_b_dev_buf.ToDevice(scale_b_packed.data());

        ScaleM scale_m(reinterpret_cast<ScaleType*>(scale_a_dev_buf.GetDeviceBuffer()));
        ScaleN scale_n(reinterpret_cast<ScaleType*>(scale_b_dev_buf.GetDeviceBuffer()));

        MXGemmHostArgs<ScaleM, ScaleN> args(a_dev_buf.GetDeviceBuffer(),
                                            b_dev_buf.GetDeviceBuffer(),
                                            c_dev_buf.GetDeviceBuffer(),
                                            1,
                                            M,
                                            N,
                                            K,
                                            stride_A,
                                            stride_B,
                                            stride_C,
                                            scale_m,
                                            scale_n);

        mx_gemm_calc<GemmConfig,
                     ADataType,
                     BDataType,
                     AccDataType,
                     CDataType,
                     ALayout,
                     BLayout,
                     CLayout,
                     ScaleM,
                     ScaleN,
                     true,
                     false>(args, ck_tile::stream_config{nullptr, true, 1, 0, 1, true, true, 50});

        c_dev_buf.FromDevice(c_host.data());

        ck_tile::HostTensor<CDataType> c_ref(
            ck_tile::host_tensor_descriptor(M, N, stride_C, is_row_major(CLayout{})));
        c_ref.SetZero();
        ck_tile::
            reference_mx_gemm<ADataType, BDataType, ScaleType, ScaleType, AccDataType, CDataType>(
                a_host, b_host, c_ref, scale_a_host, scale_b_host);

        const float max_accumulated_value = ck_tile::type_convert<float>(c_ref.max());
        const auto rtol_atol = calculate_rtol_atol_mx<ADataType, BDataType, AccDataType, CDataType>(
            K, max_accumulated_value);
        const double rtol = rtol_atol.at(ck_tile::number<0>{});
        const double atol = rtol_atol.at(ck_tile::number<1>{});

        bool pass = ck_tile::check_err(c_host, c_ref, "MX GEMM: Incorrect results!", rtol, atol);

        EXPECT_TRUE(pass);
    }
};
