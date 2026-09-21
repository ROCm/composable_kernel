// SPDX-License-Identifier: MIT
// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.

#include <gtest/gtest.h>
#include "ck_tile/host.hpp"
#include "ck_tile/host/kernel_launch.hpp"
#include "ck_tile/core.hpp"
#include "ck_tile/ops/gemm/warp/warp_gemm_dispatcher.hpp"

using namespace ck_tile;

template <index_t NumScales>
CK_TILE_DEVICE static constexpr auto MakeScaleDistribution()
{
    return make_static_tile_distribution(
        tile_distribution_encoding<sequence<2>,
                                   tuple<sequence<16>, sequence<NumScales>>,
                                   tuple<sequence<0, 1>>,
                                   tuple<sequence<0, 0>>,
                                   sequence<2>,
                                   sequence<0>>{});
}

// Scale16 kernel using the WarpGemm wrapper (Layer 4).
// WarpGemmWmma_f32_16x16x128_f8f6f4_scale16 produces AWarpDstr with sequence<4,2,16>,
// matching the hardware's 1-scale-byte-per-16-K-elements register layout.
template <typename AType,
          typename BType,
          typename CType,
          index_t M,
          index_t N,
          index_t K,
          bool TransposeC>
struct WarpGemmScale16Kernel
{
    static constexpr int kBlockSize      = 32;
    static constexpr index_t ScaleBlockK = 16;
    static constexpr index_t NumScales   = K / ScaleBlockK;

    __device__ void operator()(void* A, void* B, void* C, void* ScaleA, void* ScaleB) const
    {
        const auto a_view =
            make_naive_tensor_view<address_space_enum::global>(static_cast<AType*>(A),
                                                               make_tuple(M, K),
                                                               make_tuple(K, number<1>{}),
                                                               number<K>{},
                                                               number<1>{});
        const auto b_view =
            make_naive_tensor_view<address_space_enum::global>(static_cast<BType*>(B),
                                                               make_tuple(N, K),
                                                               make_tuple(K, number<1>{}),
                                                               number<K>{},
                                                               number<1>{});
        const auto c_view =
            make_naive_tensor_view<address_space_enum::global>(static_cast<CType*>(C),
                                                               make_tuple(M, N),
                                                               make_tuple(N, number<1>{}),
                                                               number<N>{},
                                                               number<1>{});
        const auto sa_view =
            make_naive_tensor_view<address_space_enum::global>(static_cast<e8m0_t*>(ScaleA),
                                                               make_tuple(M, NumScales),
                                                               make_tuple(NumScales, number<1>{}),
                                                               number<NumScales>{},
                                                               number<1>{});
        const auto sb_view =
            make_naive_tensor_view<address_space_enum::global>(static_cast<e8m0_t*>(ScaleB),
                                                               make_tuple(N, NumScales),
                                                               make_tuple(NumScales, number<1>{}),
                                                               number<NumScales>{},
                                                               number<1>{});

        using WarpGemm            = WarpGemmDispatcher<AType,
                                                       BType,
                                                       float,
                                                       M,
                                                       N,
                                                       K,
                                                       TransposeC,
                                                       false,
                                                       false,
                                                       WGAttrNumAccessEnum::Default,
                                                       WGAttrNumAccessEnum::Default,
                                                       true>;
        constexpr auto a_dstr     = typename WarpGemm::AWarpDstr{};
        constexpr auto b_dstr     = typename WarpGemm::BWarpDstr{};
        constexpr auto c_dstr     = typename WarpGemm::CWarpDstr{};
        constexpr auto scale_dstr = MakeScaleDistribution<NumScales>();

        auto a_win = make_tile_window(
            a_view, make_tuple(number<M>{}, number<K>{}), make_multi_index(0, 0), a_dstr);
        auto b_win = make_tile_window(
            b_view, make_tuple(number<N>{}, number<K>{}), make_multi_index(0, 0), b_dstr);
        auto c_win = make_tile_window(
            c_view, make_tuple(number<M>{}, number<N>{}), make_multi_index(0, 0), c_dstr);
        auto sa_win = make_tile_window(sa_view,
                                       make_tuple(number<M>{}, number<NumScales>{}),
                                       make_multi_index(0, 0),
                                       scale_dstr);
        auto sb_win = make_tile_window(sb_view,
                                       make_tuple(number<N>{}, number<NumScales>{}),
                                       make_multi_index(0, 0),
                                       scale_dstr);

        auto a_tile  = load_tile(a_win);
        auto b_tile  = load_tile(b_win);
        auto sa_tile = load_tile(sa_win);
        auto sb_tile = load_tile(sb_win);

        int64_t scale_a =
            bit_cast<int64_t>(sa_tile.get_thread_buffer()
                                  .template get_as<ext_vector_t<e8m0_t, NumScales>>()[number<0>{}]);
        int64_t scale_b =
            bit_cast<int64_t>(sb_tile.get_thread_buffer()
                                  .template get_as<ext_vector_t<e8m0_t, NumScales>>()[number<0>{}]);

        auto c_tile = WarpGemm{}(a_tile, b_tile, scale_a, scale_b);
        store_tile(c_win, c_tile);
    }
};

template <typename A, typename B, typename Acc, index_t M, index_t N, index_t K>
struct WGDispCase
{
    using AType                       = A;
    using BType                       = B;
    using AccType                     = Acc;
    static constexpr index_t MPerWave = M;
    static constexpr index_t NPerWave = N;
    static constexpr index_t KPerWave = K;
};

template <typename Case, bool TransposeC>
static void RunTest(const HostTensor<typename Case::AType>& A,
                    const HostTensor<typename Case::BType>& B,
                    const HostTensor<e8m0_t>& ScaleA,
                    const HostTensor<e8m0_t>& ScaleB,
                    HostTensor<typename Case::AccType>& C)
{
    DeviceMem Ad(A), Bd(B), Cd(C), SAd(ScaleA), SBd(ScaleB);
    dim3 grid(1), block{32};

    using K = WarpGemmScale16Kernel<typename Case::AType,
                                    typename Case::BType,
                                    typename Case::AccType,
                                    Case::MPerWave,
                                    Case::NPerWave,
                                    Case::KPerWave,
                                    TransposeC>;

    (void)launch_kernel(stream_config{nullptr, true, 0, 0, 1},
                        make_kernel(K{},
                                    grid,
                                    block,
                                    0,
                                    Ad.GetDeviceBuffer(),
                                    Bd.GetDeviceBuffer(),
                                    Cd.GetDeviceBuffer(),
                                    SAd.GetDeviceBuffer(),
                                    SBd.GetDeviceBuffer()));

    Cd.FromDevice(C.mData.data());
}

using WGDispatcherTypesList = ::testing::Types<WGDispCase<fp8_t, fp8_t, float, 16, 16, 128>,
                                               WGDispCase<bf8_t, bf8_t, float, 16, 16, 128>,
                                               WGDispCase<fp8_t, bf8_t, float, 16, 16, 128>,
                                               WGDispCase<bf8_t, fp8_t, float, 16, 16, 128>>;

template <typename Case>
class WGScale16Test : public ::testing::Test
{
};

TYPED_TEST_SUITE(WGScale16Test, WGDispatcherTypesList);

TYPED_TEST(WGScale16Test, Scale16_16x16x128_UniformScale)
{
    using Case  = TypeParam;
    using AType = typename Case::AType;
    using BType = typename Case::BType;
    using CType = typename Case::AccType;

    constexpr index_t M         = Case::MPerWave;
    constexpr index_t N         = Case::NPerWave;
    constexpr index_t K         = Case::KPerWave;
    constexpr index_t NumScales = K / 16;

    HostTensor<AType> A({M, K});
    HostTensor<BType> B({N, K});
    HostTensor<CType> C({M, N});
    HostTensor<e8m0_t> sA({M, NumScales});
    HostTensor<e8m0_t> sB({N, NumScales});

    FillUniformDistribution<AType>{-5.f, 5.f}(A);
    FillUniformDistribution<BType>{-5.f, 5.f}(B);
    C.SetZero();
    FillConstant<e8m0_t>{e8m0_t{2.f}}(sA);
    FillConstant<e8m0_t>{e8m0_t{4.f}}(sB);

    RunTest<Case, false>(A, B, sA, sB, C);

    HostTensor<CType> C_ref({M, N});
    C_ref.SetZero();
    reference_mx_gemm<AType, BType, e8m0_t, e8m0_t, CType, CType>(
        A, B.transpose(), C_ref, sA, sB.transpose());

    EXPECT_TRUE(check_err(C, C_ref, "Scale16 uniform scale error."));
}

TYPED_TEST(WGScale16Test, Scale16_16x16x128_RandomDataRandomScales)
{
    using Case  = TypeParam;
    using AType = typename Case::AType;
    using BType = typename Case::BType;
    using CType = typename Case::AccType;

    constexpr index_t M         = Case::MPerWave;
    constexpr index_t N         = Case::NPerWave;
    constexpr index_t K         = Case::KPerWave;
    constexpr index_t NumScales = K / 16;

    HostTensor<AType> A({M, K});
    HostTensor<BType> B({N, K});
    HostTensor<CType> C({M, N});
    HostTensor<e8m0_t> sA({M, NumScales});
    HostTensor<e8m0_t> sB({N, NumScales});

    FillUniformDistribution<AType>{-5.f, 5.f, 42}(A);
    FillUniformDistribution<BType>{-5.f, 5.f, 137}(B);
    C.SetZero();

    {
        constexpr int bias = ck_tile::numeric_traits<e8m0_t>::bias;
        std::mt19937 gen(9999);
        std::uniform_int_distribution<int> dist(bias - 4, bias + 2);
        for(auto& s : sA.mData)
            s = e8m0_t(static_cast<typename e8m0_t::type>(dist(gen)));
        for(auto& s : sB.mData)
            s = e8m0_t(static_cast<typename e8m0_t::type>(dist(gen)));
    }

    RunTest<Case, false>(A, B, sA, sB, C);

    HostTensor<CType> C_ref({M, N});
    C_ref.SetZero();
    reference_mx_gemm<AType, BType, e8m0_t, e8m0_t, CType, CType>(
        A, B.transpose(), C_ref, sA, sB.transpose());

    const float max_acc = *std::max_element(C_ref.mData.begin(), C_ref.mData.end());
    const auto rtol     = ck_tile::get_relative_threshold<AType, CType, CType>(K);
    const auto atol     = ck_tile::get_absolute_threshold<AType, CType, CType>(max_acc, K);
    EXPECT_TRUE(check_err(C, C_ref, "Scale16 random data + random scales error.", rtol, atol));
}

TYPED_TEST(WGScale16Test, Scale16_16x16x128_TransposeC)
{
    using Case  = TypeParam;
    using AType = typename Case::AType;
    using BType = typename Case::BType;
    using CType = typename Case::AccType;

    constexpr index_t M         = Case::MPerWave;
    constexpr index_t N         = Case::NPerWave;
    constexpr index_t K         = Case::KPerWave;
    constexpr index_t NumScales = K / 16;

    HostTensor<AType> A({M, K});
    HostTensor<BType> B({N, K});
    HostTensor<CType> C({M, N});
    HostTensor<e8m0_t> sA({M, NumScales});
    HostTensor<e8m0_t> sB({N, NumScales});

    FillUniformDistribution<AType>{-5.f, 5.f, 77}(A);
    FillUniformDistribution<BType>{-5.f, 5.f, 88}(B);
    C.SetZero();

    {
        constexpr int bias = ck_tile::numeric_traits<e8m0_t>::bias;
        std::mt19937 gen(5555);
        std::uniform_int_distribution<int> dist(bias - 4, bias + 2);
        for(auto& s : sA.mData)
            s = e8m0_t(static_cast<typename e8m0_t::type>(dist(gen)));
        for(auto& s : sB.mData)
            s = e8m0_t(static_cast<typename e8m0_t::type>(dist(gen)));
    }

    RunTest<Case, true>(A, B, sA, sB, C);

    HostTensor<CType> C_ref({M, N});
    C_ref.SetZero();
    reference_mx_gemm<AType, BType, e8m0_t, e8m0_t, CType, CType>(
        A, B.transpose(), C_ref, sA, sB.transpose());

    const float max_acc = *std::max_element(C_ref.mData.begin(), C_ref.mData.end());
    const auto rtol     = ck_tile::get_relative_threshold<AType, CType, CType>(K);
    const auto atol     = ck_tile::get_absolute_threshold<AType, CType, CType>(max_acc, K);
    EXPECT_TRUE(check_err(C, C_ref, "Scale16 TransposeC error.", rtol, atol));
}
