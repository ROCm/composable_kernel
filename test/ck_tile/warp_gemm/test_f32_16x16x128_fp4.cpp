// SPDX-License-Identifier: MIT
// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.

#include <gtest/gtest.h>
#include "ck_tile/host.hpp"
#include "ck_tile/host/kernel_launch.hpp"
#include "ck_tile/ops/gemm/warp/warp_gemm_dispatcher.hpp"

using namespace ck_tile;

template <typename A,
          typename B,
          typename Acc,
          index_t M,
          index_t N,
          index_t K,
          bool TransposeC,
          bool SwizzleA              = false,
          bool UseStructuredSparsity = false,
          WGAttrNumAccessEnum NA     = WGAttrNumAccessEnum::Single>
struct WGDispCase
{
    using AType                              = A;
    using BType                              = B;
    using AccType                            = Acc;
    static constexpr index_t MPerWave        = M;
    static constexpr index_t NPerWave        = N;
    static constexpr index_t KPerWave        = K;
    static constexpr bool kTransposeC        = TransposeC;
    static constexpr bool kSwizzleA          = SwizzleA;
    static constexpr bool kUSS               = UseStructuredSparsity;
    static constexpr WGAttrNumAccessEnum kNA = NA;
};

using WGDispatcherTypesList =
    ::testing::Types<WGDispCase<ck_tile::pk_fp4_t, ck_tile::pk_fp4_t, float, 16, 16, 128, false>>;

template <typename AType,
          typename BType,
          typename CType,
          index_t M,
          index_t N,
          index_t K,
          bool TransposeC,
          bool SwizzleA,
          bool UseStructuredSparsity,
          WGAttrNumAccessEnum NumAccess>
struct WarpGemmKernel
{
    static constexpr int kBlockSize = 64;
    __device__ void operator()(const AType* A,
                               const BType* B,
                               CType* C,
                               const ck_tile::e8m0_t ScaleA,
                               const ck_tile::e8m0_t ScaleB) const
    {
        using WarpGemm = ck_tile::WarpGemmDispatcher<AType,
                                                     BType,
                                                     CType,
                                                     M,
                                                     N,
                                                     K,
                                                     TransposeC,
                                                     SwizzleA,
                                                     UseStructuredSparsity,
                                                     NumAccess>;

        // A: [M,K] row-major (packed)
        const auto a_view = ck_tile::make_naive_tensor_view<ck_tile::address_space_enum::global>(
            const_cast<AType*>(A),
            ck_tile::make_tuple(M, K),
            ck_tile::make_tuple(K, ck_tile::number<1>{}),
            ck_tile::number<K>{},
            ck_tile::number<1>{});
        // B: expose as logical [N,K] with strides (1, N) over the original row-major [K,N] buffer
        const auto b_view = ck_tile::make_naive_tensor_view<ck_tile::address_space_enum::global>(
            const_cast<BType*>(B),
            ck_tile::make_tuple(N, K),
            ck_tile::make_tuple(ck_tile::number<1>{}, N),
            ck_tile::number<N>{},
            ck_tile::number<1>{});
        // C: [M,N] row-major (packed)
        const auto c_view = ck_tile::make_naive_tensor_view<ck_tile::address_space_enum::global>(
            const_cast<CType*>(C),
            ck_tile::make_tuple(M, N),
            ck_tile::make_tuple(N, ck_tile::number<1>{}),
            ck_tile::number<N>{},
            ck_tile::number<1>{});

        using AWarpTensor = typename WarpGemm::AWarpTensor;
        using BWarpTensor = typename WarpGemm::BWarpTensor;
        using CWarpTensor = typename WarpGemm::CWarpTensor;

        constexpr auto a_len = AWarpTensor::get_tile_distribution().get_lengths();
        constexpr auto b_len = BWarpTensor::get_tile_distribution().get_lengths();
        constexpr auto c_len = CWarpTensor::get_tile_distribution().get_lengths();

        auto a_win = ck_tile::make_tile_window(
            a_view, a_len, ck_tile::make_multi_index(0, 0), AWarpTensor::get_tile_distribution());
        auto b_win = ck_tile::make_tile_window(
            b_view, b_len, ck_tile::make_multi_index(0, 0), BWarpTensor::get_tile_distribution());
        auto c_win = ck_tile::make_tile_window(
            c_view, c_len, ck_tile::make_multi_index(0, 0), CWarpTensor::get_tile_distribution());

        AWarpTensor a_tile;
        BWarpTensor b_tile;
        ck_tile::load_tile(a_tile, a_win);
        ck_tile::load_tile(b_tile, b_win);

        CWarpTensor c_tile;

        // TODO: integrate opsel and multiple e8m0 into tool to reduce code complexity.
        auto scale_a = static_cast<int32_t>(ScaleA.get());
        auto scale_b = static_cast<int32_t>(ScaleB.get());
        WarpGemm{}.template operator()<0, 0>(c_tile, a_tile, b_tile, scale_a, scale_b);

        ck_tile::store_tile(c_win, c_tile);
    }
};

template <typename Case>
static void RunWarpGemmCase(const ck_tile::HostTensor<typename Case::AType>& A,
                            const ck_tile::HostTensor<typename Case::BType>& B,
                            const ck_tile::e8m0_t ScaleA,
                            const ck_tile::e8m0_t ScaleB,
                            ck_tile::HostTensor<typename Case::AccType>& C)
{
    using AType = typename Case::AType;
    using BType = typename Case::BType;
    using CType = typename Case::AccType; // CDataType equals Acc for these tests

    ck_tile::DeviceMem Ad(A.get_element_space_size_in_bytes());
    ck_tile::DeviceMem Bd(B.get_element_space_size_in_bytes());
    ck_tile::DeviceMem Cd(C.get_element_space_size_in_bytes());

    Ad.ToDevice(A.data());
    Bd.ToDevice(B.data());
    Cd.SetZero();

    dim3 grid(1);
    dim3 block{64};

    using Kernel = WarpGemmKernel<AType,
                                  BType,
                                  CType,
                                  Case::MPerWave,
                                  Case::NPerWave,
                                  Case::KPerWave,
                                  Case::kTransposeC,
                                  Case::kSwizzleA,
                                  Case::kUSS,
                                  Case::kNA>;

    (void)ck_tile::launch_kernel(
        ck_tile::stream_config{nullptr, true, 0, 0, 1},
        ck_tile::make_kernel(Kernel{},
                             grid,
                             block,
                             0,
                             static_cast<const AType*>(Ad.GetDeviceBuffer()),
                             static_cast<const BType*>(Bd.GetDeviceBuffer()),
                             static_cast<CType*>(Cd.GetDeviceBuffer()),
                             ScaleA,
                             ScaleB));

    Cd.FromDevice(C.mData.data());
}

template <typename Case>
class WGRuntimeTest : public ::testing::Test
{
};

TYPED_TEST_SUITE(WGRuntimeTest, WGDispatcherTypesList);

TYPED_TEST(WGRuntimeTest, Compare_Dispatcher_MakeWG)
{
    using Case = TypeParam;

    using AType = typename Case::AType;
    using BType = typename Case::BType;
    using CType = typename Case::AccType;
    using ck_tile::e8m0_t;

    constexpr index_t M = Case::MPerWave;
    constexpr index_t N = Case::NPerWave;
    constexpr index_t K = Case::KPerWave;

    ck_tile::HostTensor<AType> A({M, K});
    ck_tile::HostTensor<BType> B({K, N});
    ck_tile::HostTensor<CType> C({M, N});

    ck_tile::FillConstant<AType>{AType(1.f)}(A);
    ck_tile::FillConstant<AType>{AType(1.f)}(B);

    auto ScaleA = ck_tile::e8m0_t{1.0f};
    auto ScaleB = ck_tile::e8m0_t{1.0f};

    C.SetZero();
    RunWarpGemmCase<Case>(A, B, ScaleA, ScaleB, C);

    ck_tile::HostTensor<e8m0_t> sA({M, 1}); // Just 1 scale for all K
    ck_tile::HostTensor<e8m0_t> sB({1, N});
    ck_tile::HostTensor<CType> C_ref({M, N});
    C_ref.SetZero();

    ck_tile::FillConstant<e8m0_t>{e8m0_t(1.f)}(sA);
    ck_tile::FillConstant<e8m0_t>{e8m0_t(1.f)}(sB);

    ck_tile::reference_mx_gemm<AType, BType, e8m0_t, CType, CType>(A, B, C_ref, sA, sB);

    EXPECT_TRUE(ck_tile::check_err(C, C_ref, "Warp gemm result error."));
}
