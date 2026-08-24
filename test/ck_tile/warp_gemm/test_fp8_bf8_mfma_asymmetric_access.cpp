// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// Tests for the warp_gemm_dispatcher specializations that were added to fix a
// compile error on gfx950: missing Dispatcher entries for asymmetric
// AttrNumAccess combinations (Single/Double and Double/Single) on FP8/BF8
// MFMA warp gemm shapes, and the missing bf8xbf8 EDouble entry.
//
// Shapes covered:
//   fp8xfp8 16x16x64, bf8xbf8 16x16x64 -- EDouble, Single/Double, Double/Single
//   fp8xfp8 32x32x32, bf8xbf8 32x32x32 -- Single/Double, Double/Single  (gfx950 only)
//
// Each case: random input, compare device result to CPU reference_gemm.

#include <gtest/gtest.h>

#include "ck_tile/host.hpp"
#include "ck_tile/host/kernel_launch.hpp"
#include "ck_tile/ops/gemm/warp/warp_gemm_dispatcher.hpp"

using namespace ck_tile;

// ---------------------------------------------------------------------------
// Device kernel: one warp, one warp-gemm call, stores result to global memory
// ---------------------------------------------------------------------------
template <typename AType,
          typename BType,
          index_t M,
          index_t N,
          index_t K,
          WGAttrNumAccessEnum NAA,
          WGAttrNumAccessEnum NAB>
struct WarpGemmAsymKernel
{
    static constexpr int kBlockSize = 64;

    __device__ void operator()(void* A, void* B, void* C) const
    {
        static constexpr bool UsePackedNumAccess = (NAA != NAB);
        using WarpGemm                           = WarpGemmDispatcher<AType,
                                                                      BType,
                                                                      float,
                                                                      M,
                                                                      N,
                                                                      K,
                                                                      /*TransposeC=*/false,
                                                                      /*SwizzleA=*/false,
                                                                      /*USS=*/false,
                                                                      NAA,
                                                                      NAB,
                                                                      /*IsScale16=*/false,
                                                                      UsePackedNumAccess>;

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
            make_naive_tensor_view<address_space_enum::global>(static_cast<float*>(C),
                                                               make_tuple(M, N),
                                                               make_tuple(N, number<1>{}),
                                                               number<N>{},
                                                               number<1>{});

        using AWarpTensor = typename WarpGemm::AWarpTensor;
        using BWarpTensor = typename WarpGemm::BWarpTensor;
        using CWarpTensor = typename WarpGemm::CWarpTensor;

        auto a_win = make_tile_window(a_view,
                                      AWarpTensor::get_tile_distribution().get_lengths(),
                                      make_multi_index(0, 0),
                                      AWarpTensor::get_tile_distribution());
        auto b_win = make_tile_window(b_view,
                                      BWarpTensor::get_tile_distribution().get_lengths(),
                                      make_multi_index(0, 0),
                                      BWarpTensor::get_tile_distribution());
        auto c_win = make_tile_window(c_view,
                                      CWarpTensor::get_tile_distribution().get_lengths(),
                                      make_multi_index(0, 0),
                                      CWarpTensor::get_tile_distribution());

        AWarpTensor a_tile;
        BWarpTensor b_tile;
        load_tile(a_tile, a_win);
        load_tile(b_tile, b_win);

        auto c_tile = WarpGemm{}(a_tile, b_tile);
        store_tile(c_win, c_tile);
    }
};

template <typename AType,
          typename BType,
          index_t M,
          index_t N,
          index_t K,
          WGAttrNumAccessEnum NAA,
          WGAttrNumAccessEnum NAB>
void RunWarpGemmAsym(const HostTensor<AType>& A, const HostTensor<BType>& B, HostTensor<float>& C)
{
    using Kern = WarpGemmAsymKernel<AType, BType, M, N, K, NAA, NAB>;
    DeviceMem Ad(A), Bd(B), Cd(C);
    dim3 grid(1), block{Kern::kBlockSize};
    (void)launch_kernel(stream_config{nullptr, false, 0, 0, 1},
                        make_kernel(Kern{},
                                    grid,
                                    block,
                                    0,
                                    Ad.GetDeviceBuffer(),
                                    Bd.GetDeviceBuffer(),
                                    Cd.GetDeviceBuffer()));
    Cd.FromDevice(C.mData.data());
}

// ---------------------------------------------------------------------------
// Typed test infrastructure
// ---------------------------------------------------------------------------
template <typename AType,
          typename BType,
          index_t M_,
          index_t N_,
          index_t K_,
          WGAttrNumAccessEnum NAA_,
          WGAttrNumAccessEnum NAB_>
struct AsymCase
{
    using A                                  = AType;
    using B                                  = BType;
    static constexpr index_t M               = M_;
    static constexpr index_t N               = N_;
    static constexpr index_t K               = K_;
    static constexpr WGAttrNumAccessEnum NAA = NAA_;
    static constexpr WGAttrNumAccessEnum NAB = NAB_;
};

template <typename Case>
class WarpGemmAsymAccessTest : public ::testing::Test
{
};

// clang-format off
using AsymCaseList = ::testing::Types<
    // --- bf8xbf8 16x16x64 EDouble (both same, was missing before fix) ---
    AsymCase<bf8_t, bf8_t, 16, 16, 64, WGAttrNumAccessEnum::Double, WGAttrNumAccessEnum::Double>,
    // --- fp8xfp8 16x16x64 asymmetric (gfx950 new specializations) ---
    AsymCase<fp8_t, fp8_t, 16, 16, 64, WGAttrNumAccessEnum::Single, WGAttrNumAccessEnum::Double>,
    AsymCase<fp8_t, fp8_t, 16, 16, 64, WGAttrNumAccessEnum::Double, WGAttrNumAccessEnum::Single>,
    // --- bf8xbf8 16x16x64 asymmetric ---
    AsymCase<bf8_t, bf8_t, 16, 16, 64, WGAttrNumAccessEnum::Single, WGAttrNumAccessEnum::Double>,
    AsymCase<bf8_t, bf8_t, 16, 16, 64, WGAttrNumAccessEnum::Double, WGAttrNumAccessEnum::Single>
>;
// clang-format on

TYPED_TEST_SUITE(WarpGemmAsymAccessTest, AsymCaseList);

TYPED_TEST(WarpGemmAsymAccessTest, CorrectVsReference)
{
    using Case  = TypeParam;
    using AType = typename Case::A;
    using BType = typename Case::B;
    using CType = float;

    constexpr index_t M = Case::M;
    constexpr index_t N = Case::N;
    constexpr index_t K = Case::K;

    HostTensor<AType> A({M, K});
    HostTensor<BType> B({N, K});
    HostTensor<CType> C({M, N});
    HostTensor<CType> C_ref({M, N});

    FillUniformDistribution<AType>{-1.f, 1.f}(A);
    FillUniformDistribution<BType>{-1.f, 1.f}(B);
    C.SetZero();
    C_ref.SetZero();

    RunWarpGemmAsym<AType, BType, M, N, K, Case::NAA, Case::NAB>(A, B, C);

    // Reference: A(MxK) * B^T(KxN) -- B is stored (N,K) so B.transpose() is (K,N)
    reference_gemm<AType, BType, CType, CType>(A, B.transpose(), C_ref);

    const float max_acc = *std::max_element(C_ref.mData.begin(), C_ref.mData.end());
    const auto rtol     = get_relative_threshold<AType, CType, CType>(K);
    const auto atol     = get_absolute_threshold<AType, CType, CType>(max_acc, K);

    EXPECT_TRUE(check_err(C, C_ref, "WarpGemm asymmetric access mismatch.", rtol, atol));
}

// ---------------------------------------------------------------------------
// gfx950-only: 32x32x32 asymmetric access cases
// ---------------------------------------------------------------------------
#if defined(__gfx950__)

template <typename AType,
          typename BType,
          index_t M_,
          index_t N_,
          index_t K_,
          WGAttrNumAccessEnum NAA_,
          WGAttrNumAccessEnum NAB_>
struct Asym32Case
{
    using A                                  = AType;
    using B                                  = BType;
    static constexpr index_t M               = M_;
    static constexpr index_t N               = N_;
    static constexpr index_t K               = K_;
    static constexpr WGAttrNumAccessEnum NAA = NAA_;
    static constexpr WGAttrNumAccessEnum NAB = NAB_;
};

template <typename Case>
class WarpGemmAsym32Test : public ::testing::Test
{
};

using Asym32CaseList = ::testing::Types<
    Asym32Case<fp8_t, fp8_t, 32, 32, 32, WGAttrNumAccessEnum::Single, WGAttrNumAccessEnum::Double>,
    Asym32Case<fp8_t, fp8_t, 32, 32, 32, WGAttrNumAccessEnum::Double, WGAttrNumAccessEnum::Single>,
    Asym32Case<bf8_t, bf8_t, 32, 32, 32, WGAttrNumAccessEnum::Single, WGAttrNumAccessEnum::Double>,
    Asym32Case<bf8_t, bf8_t, 32, 32, 32, WGAttrNumAccessEnum::Double, WGAttrNumAccessEnum::Single>>;

TYPED_TEST_SUITE(WarpGemmAsym32Test, Asym32CaseList);

TYPED_TEST(WarpGemmAsym32Test, CorrectVsReference)
{
    using Case  = TypeParam;
    using AType = typename Case::A;
    using BType = typename Case::B;
    using CType = float;

    constexpr index_t M = Case::M;
    constexpr index_t N = Case::N;
    constexpr index_t K = Case::K;

    HostTensor<AType> A({M, K});
    HostTensor<BType> B({N, K});
    HostTensor<CType> C({M, N});
    HostTensor<CType> C_ref({M, N});

    FillUniformDistribution<AType>{-1.f, 1.f}(A);
    FillUniformDistribution<BType>{-1.f, 1.f}(B);
    C.SetZero();
    C_ref.SetZero();

    RunWarpGemmAsym<AType, BType, M, N, K, Case::NAA, Case::NAB>(A, B, C);

    reference_gemm<AType, BType, CType, CType>(A, B.transpose(), C_ref);

    const float max_acc = *std::max_element(C_ref.mData.begin(), C_ref.mData.end());
    const auto rtol     = get_relative_threshold<AType, CType, CType>(K);
    const auto atol     = get_absolute_threshold<AType, CType, CType>(max_acc, K);

    EXPECT_TRUE(check_err(C, C_ref, "WarpGemm 32x32x32 asymmetric access mismatch.", rtol, atol));
}

#endif // defined(__gfx950__)
