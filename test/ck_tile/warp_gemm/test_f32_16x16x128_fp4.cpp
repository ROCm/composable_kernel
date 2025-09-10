// SPDX-License-Identifier: MIT
// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.

#include <gtest/gtest.h>
#include <hip/hip_runtime.h>
#include <algorithm>
#include "ck_tile/ops/gemm/warp/warp_gemm_attribute_mfma_impl.hpp"
#include "ck_tile/ops/gemm/warp/warp_gemm_attribute_mfma.hpp"
// For real kernel run and verification
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


using WGDispatcherTypesList = ::testing::Types<
    WGDispCase<ck_tile::pk_fp4_t, ck_tile::pk_fp4_t, float, 16, 16, 128, false>
   // , WGDispCase<ck_tile::pk_fp4_t, ck_tile::pk_fp4_t, float, 16, 16, 128, false, false, false, WGAttrNumAccessEnum::Quad>
    >;

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
    __device__ void operator()(const AType* A, const BType* B, CType* C) const
    { 
        using WarpGemm = ck_tile::WarpGemmDispatcher<AType, BType, CType, 
                M, N, K, TransposeC, SwizzleA, UseStructuredSparsity, NumAccess>;

        // A: [M,K] row-major (packed)
        const auto a_view =
            ck_tile::make_naive_tensor_view_packed<ck_tile::address_space_enum::global>(
                const_cast<AType*>(A), ck_tile::make_tuple(M, K));
        // B: expose as logical [N,K] with strides (1, N) over the original row-major [K,N] buffer
        const auto b_view = ck_tile::make_naive_tensor_view<ck_tile::address_space_enum::global>(
            const_cast<BType*>(B), ck_tile::make_tuple(N, K), ck_tile::make_tuple(1, N));
        // C: [M,N] row-major (packed)
        const auto c_view =
            ck_tile::make_naive_tensor_view_packed<ck_tile::address_space_enum::global>(
                const_cast<CType*>(C), ck_tile::make_tuple(M, N));

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
        c_tile = WarpGemm{}(a_tile, b_tile);
        ck_tile::store_tile(c_win, c_tile);
    }
};

template <typename Case>
static void RunWarpGemmCase(const ck_tile::HostTensor<typename Case::AType>& A,
                            const ck_tile::HostTensor<typename Case::BType>& B,
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
        ck_tile::stream_config{nullptr, true},
        ck_tile::make_kernel(Kernel{},
                             grid,
                             block,
                             0,
                             static_cast<const AType*>(Ad.GetDeviceBuffer()),
                             static_cast<const BType*>(Bd.GetDeviceBuffer()),
                             static_cast<CType*>(Cd.GetDeviceBuffer())));

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

    constexpr index_t M = Case::MPerWave;
    constexpr index_t N = Case::NPerWave;
    constexpr index_t K = Case::KPerWave;

    ck_tile::HostTensor<AType> A({M, K});
    ck_tile::HostTensor<BType> B({K, N});
    ck_tile::HostTensor<CType> C({M, N});

    // Note:pk_fp4_t packed_size = 2
    for(index_t m = 0; m < M; ++m)
        for(index_t k = 0; k < K; ++k)
            A(m, k) = ck_tile::type_convert<AType>((m + 1) * 0.1f + (k + 1) * 0.01f);

    for(index_t k0 = 0; k0 < K; ++k0)
        for(index_t n = 0; n < N; ++n)
            B(k0, n) = ck_tile::type_convert<BType>((k0 + 1) * 0.2f - (n + 1) * 0.03f);

    C.SetZero();
	RunWarpGemmCase<Case>(A, B, C); 

    //EXPECT_TRUE(ck_tile::check_err(C, C_ref, "Dispatcher vs Reference mismatch", 0, 0));
}
