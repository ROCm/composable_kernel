// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <gtest/gtest.h>
#include "ck_tile/host.hpp"
#include "ck_tile/host/kernel_launch.hpp"
#include "ck_tile/ops/gemm/warp/warp_gemm_dispatcher.hpp"

using namespace ck_tile;

template <typename A, typename B, typename Acc, index_t M, index_t N, index_t K, bool TransposeC>
struct WGDispCase
{
    using AType                       = A;
    using BType                       = B;
    using AccType                     = Acc;
    static constexpr index_t MPerWave = M;
    static constexpr index_t NPerWave = N;
    static constexpr index_t KPerWave = K;
    static constexpr bool kTransposeC = TransposeC;
};

using WGDispatcherTypesList =
    ::testing::Types<WGDispCase<bf16_t, bf16_t, bf16_t, 16, 16, 32, false>,
                     WGDispCase<bf16_t, bf16_t, float, 16, 16, 32, false>>;

template <typename AType,
          typename BType,
          typename CType,
          index_t M,
          index_t N,
          index_t K,
          bool TransposeC>
struct WarpGemmKernel
{
    static constexpr int kBlockSize = 32;
    __device__ void operator()(void* A, void* B, void* C) const
    {
        using WarpGemm = WarpGemmDispatcher<AType, BType, CType, M, N, K, TransposeC>;

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

        using AWarpTensor = typename WarpGemm::AWarpTensor;
        using BWarpTensor = typename WarpGemm::BWarpTensor;
        using CWarpTensor = typename WarpGemm::CWarpTensor;

        constexpr auto a_len = AWarpTensor::get_tile_distribution().get_lengths();
        constexpr auto b_len = BWarpTensor::get_tile_distribution().get_lengths();
        constexpr auto c_len = CWarpTensor::get_tile_distribution().get_lengths();

        auto a_win = make_tile_window(
            a_view, a_len, make_multi_index(0, 0), AWarpTensor::get_tile_distribution());
        auto b_win = make_tile_window(
            b_view, b_len, make_multi_index(0, 0), BWarpTensor::get_tile_distribution());
        auto c_win = make_tile_window(
            c_view, c_len, make_multi_index(0, 0), CWarpTensor::get_tile_distribution());

        AWarpTensor a_tile;
        BWarpTensor b_tile;
        load_tile(a_tile, a_win);
        load_tile(b_tile, b_win);

        auto c_tile = WarpGemm{}(a_tile, b_tile);

        store_tile(c_win, c_tile);
    }
};

template <typename Case>
static void RunWarpGemmCase(const HostTensor<typename Case::AType>& A,
                            const HostTensor<typename Case::BType>& B,
                            HostTensor<typename Case::AccType>& C)
{
    DeviceMem Ad(A), Bd(B), Cd(C);

    using Kernel = WarpGemmKernel<typename Case::AType,
                                  typename Case::BType,
                                  typename Case::AccType,
                                  Case::MPerWave,
                                  Case::NPerWave,
                                  Case::KPerWave,
                                  Case::kTransposeC>;
    dim3 grid(1), block{Kernel::kBlockSize};

    (void)launch_kernel(stream_config{nullptr, true, 0, 0, 1},
                        make_kernel(Kernel{},
                                    grid,
                                    block,
                                    0,
                                    Ad.GetDeviceBuffer(),
                                    Bd.GetDeviceBuffer(),
                                    Cd.GetDeviceBuffer()));

    Cd.FromDevice(C.mData.data());
}

template <typename Case>
class WGRuntimeTest : public ::testing::Test
{
};

TYPED_TEST_SUITE(WGRuntimeTest, WGDispatcherTypesList);

TYPED_TEST(WGRuntimeTest, Compare_Dispatcher_ReferenceGemm)
{
    using Case = TypeParam;

    using AType = typename Case::AType;
    using BType = typename Case::BType;
    using CType = typename Case::AccType;

    constexpr index_t M = Case::MPerWave;
    constexpr index_t N = Case::NPerWave;
    constexpr index_t K = Case::KPerWave;

    HostTensor<AType> A({M, K});
    HostTensor<BType> B({N, K});
    HostTensor<CType> C({M, N});

    FillUniformDistribution<AType>{-1.f, 1.f, 11939}(A);
    FillUniformDistribution<BType>{-1.f, 1.f, 11940}(B);
    C.SetZero();

    RunWarpGemmCase<Case>(A, B, C);

    HostTensor<CType> C_ref({M, N});
    C_ref.SetZero();
    reference_gemm<AType, BType, float, CType>(A, B.transpose(), C_ref);

    EXPECT_TRUE(check_err(C, C_ref, "Warp gemm bf16 result error."));
}
