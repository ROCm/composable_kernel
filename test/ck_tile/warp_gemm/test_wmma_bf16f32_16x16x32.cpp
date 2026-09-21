// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <cmath>
#include <gtest/gtest.h>
#include "ck_tile/host.hpp"
#include "ck_tile/host/kernel_launch.hpp"
#include "ck_tile/ops/gemm/warp/warp_gemm_dispatcher.hpp"

using namespace ck_tile;

static constexpr index_t kM = 16;
static constexpr index_t kN = 16;
static constexpr index_t kK = 32;

// Y(bf16) = X(bf16) * W(bf16) + R(fp32), via wmma_bf16f32 (fp32 accumulate -> bf16 out).
// gfx125-only: the WMMA dispatcher entries are guarded by __gfx125__, so the body is
// compiled only in the device pass (host pass would resolve to MFMA, which has no
// mac_downconvert).
struct Bf16f32ResidualKernel
{
    static constexpr int kBlockSize = 32;
    __device__ void operator()(void* X, void* W, void* R, void* Y) const
    {
#if defined(__gfx125__)
        using WarpGemm = WarpGemmDispatcher<bf16_t, bf16_t, float, kM, kN, kK, false>;

        const auto x_view =
            make_naive_tensor_view<address_space_enum::global>(static_cast<bf16_t*>(X),
                                                               make_tuple(kM, kK),
                                                               make_tuple(kK, number<1>{}),
                                                               number<kK>{},
                                                               number<1>{});
        const auto w_view =
            make_naive_tensor_view<address_space_enum::global>(static_cast<bf16_t*>(W),
                                                               make_tuple(kN, kK),
                                                               make_tuple(kK, number<1>{}),
                                                               number<kK>{},
                                                               number<1>{});
        const auto r_view =
            make_naive_tensor_view<address_space_enum::global>(static_cast<float*>(R),
                                                               make_tuple(kM, kN),
                                                               make_tuple(kN, number<1>{}),
                                                               number<kN>{},
                                                               number<1>{});
        const auto y_view =
            make_naive_tensor_view<address_space_enum::global>(static_cast<bf16_t*>(Y),
                                                               make_tuple(kM, kN),
                                                               make_tuple(kN, number<1>{}),
                                                               number<kN>{},
                                                               number<1>{});

        using AWarpTensor = typename WarpGemm::AWarpTensor;
        using BWarpTensor = typename WarpGemm::BWarpTensor;
        using CWarpTensor = typename WarpGemm::CWarpTensor; // fp32 accumulator tile

        constexpr auto a_len = AWarpTensor::get_tile_distribution().get_lengths();
        constexpr auto b_len = BWarpTensor::get_tile_distribution().get_lengths();
        constexpr auto c_len = CWarpTensor::get_tile_distribution().get_lengths();

        auto x_win = make_tile_window(
            x_view, a_len, make_multi_index(0, 0), AWarpTensor::get_tile_distribution());
        auto w_win = make_tile_window(
            w_view, b_len, make_multi_index(0, 0), BWarpTensor::get_tile_distribution());
        auto r_win = make_tile_window(
            r_view, c_len, make_multi_index(0, 0), CWarpTensor::get_tile_distribution());
        auto y_win = make_tile_window(
            y_view, c_len, make_multi_index(0, 0), CWarpTensor::get_tile_distribution());

        AWarpTensor x_tile;
        BWarpTensor w_tile;
        CWarpTensor r_tile;
        load_tile(x_tile, x_win);
        load_tile(w_tile, w_win);
        load_tile(r_tile, r_win);

        auto y_tile = WarpGemm{}.mac_downconvert(r_tile, x_tile, w_tile); // bf16 C-output tile

        store_tile(y_win, y_tile);
#else
        ck_tile::ignore = X;
        ck_tile::ignore = W;
        ck_tile::ignore = R;
        ck_tile::ignore = Y;
#endif
    }
};

// Reference path: bf16 accumulate (R rounded to bf16), via wmma_bf16_16x16x32_bf16.
struct Bf16AccumResidualKernel
{
    static constexpr int kBlockSize = 32;
    __device__ void operator()(void* X, void* W, void* Rbf16, void* Y) const
    {
#if defined(__gfx125__)
        using WarpGemm = WarpGemmDispatcher<bf16_t, bf16_t, bf16_t, kM, kN, kK, false>;

        const auto x_view =
            make_naive_tensor_view<address_space_enum::global>(static_cast<bf16_t*>(X),
                                                               make_tuple(kM, kK),
                                                               make_tuple(kK, number<1>{}),
                                                               number<kK>{},
                                                               number<1>{});
        const auto w_view =
            make_naive_tensor_view<address_space_enum::global>(static_cast<bf16_t*>(W),
                                                               make_tuple(kN, kK),
                                                               make_tuple(kK, number<1>{}),
                                                               number<kK>{},
                                                               number<1>{});
        const auto r_view =
            make_naive_tensor_view<address_space_enum::global>(static_cast<bf16_t*>(Rbf16),
                                                               make_tuple(kM, kN),
                                                               make_tuple(kN, number<1>{}),
                                                               number<kN>{},
                                                               number<1>{});
        const auto y_view =
            make_naive_tensor_view<address_space_enum::global>(static_cast<bf16_t*>(Y),
                                                               make_tuple(kM, kN),
                                                               make_tuple(kN, number<1>{}),
                                                               number<kN>{},
                                                               number<1>{});

        using AWarpTensor = typename WarpGemm::AWarpTensor;
        using BWarpTensor = typename WarpGemm::BWarpTensor;
        using CWarpTensor = typename WarpGemm::CWarpTensor; // bf16 accumulator tile

        constexpr auto a_len = AWarpTensor::get_tile_distribution().get_lengths();
        constexpr auto b_len = BWarpTensor::get_tile_distribution().get_lengths();
        constexpr auto c_len = CWarpTensor::get_tile_distribution().get_lengths();

        auto x_win = make_tile_window(
            x_view, a_len, make_multi_index(0, 0), AWarpTensor::get_tile_distribution());
        auto w_win = make_tile_window(
            w_view, b_len, make_multi_index(0, 0), BWarpTensor::get_tile_distribution());
        auto r_win = make_tile_window(
            r_view, c_len, make_multi_index(0, 0), CWarpTensor::get_tile_distribution());
        auto y_win = make_tile_window(
            y_view, c_len, make_multi_index(0, 0), CWarpTensor::get_tile_distribution());

        AWarpTensor x_tile;
        BWarpTensor w_tile;
        CWarpTensor r_tile;
        load_tile(x_tile, x_win);
        load_tile(w_tile, w_win);
        load_tile(r_tile, r_win); // preload accumulator with R (bf16)

        WarpGemm{}(r_tile, x_tile, w_tile); // r += x * w (bf16 accumulate)

        store_tile(y_win, r_tile);
#else
        ck_tile::ignore = X;
        ck_tile::ignore = W;
        ck_tile::ignore = Rbf16;
        ck_tile::ignore = Y;
#endif
    }
};

static double max_abs_err(const HostTensor<bf16_t>& got, const HostTensor<float>& ref)
{
    double e = 0.0;
    for(std::size_t i = 0; i < ref.mData.size(); ++i)
    {
        const double g = static_cast<double>(type_convert<float>(got.mData[i]));
        e              = std::max(e, std::abs(g - static_cast<double>(ref.mData[i])));
    }
    return e;
}

TEST(WmmaBf16f32, ResidualPrecisionContrast)
{
    HostTensor<bf16_t> X({kM, kK}), W({kN, kK});
    HostTensor<float> R({kM, kN});

    // Small products, large fp32 residual: bf16-rounding of R is the visible differentiator.
    FillUniformDistribution<bf16_t>{-0.05f, 0.05f, 11939}(X);
    FillUniformDistribution<bf16_t>{-0.05f, 0.05f, 11940}(W);
    FillUniformDistribution<float>{-8.f, 8.f, 11941}(R);

    // fp32 reference: Y_ref = X.f32 * W.f32 + R
    HostTensor<float> Y_ref({kM, kN});
    Y_ref.SetZero();
    reference_gemm<bf16_t, bf16_t, float, float>(X, W.transpose(), Y_ref);
    for(std::size_t i = 0; i < Y_ref.mData.size(); ++i)
        Y_ref.mData[i] += R.mData[i];

    // (1) bf16f32 path: fp32 accumulate + fp32 residual, bf16 out
    HostTensor<bf16_t> Y_bf16f32({kM, kN});
    {
        DeviceMem Xd(X), Wd(W), Rd(R), Yd(Y_bf16f32);
        (void)launch_kernel(stream_config{nullptr, true, 0, 0, 1},
                            make_kernel(Bf16f32ResidualKernel{},
                                        dim3(1),
                                        dim3(Bf16f32ResidualKernel::kBlockSize),
                                        0,
                                        Xd.GetDeviceBuffer(),
                                        Wd.GetDeviceBuffer(),
                                        Rd.GetDeviceBuffer(),
                                        Yd.GetDeviceBuffer()));
        Yd.FromDevice(Y_bf16f32.mData.data());
    }

    // (2) bf16-accumulate path: R rounded to bf16
    HostTensor<bf16_t> Rb({kM, kN});
    for(std::size_t i = 0; i < R.mData.size(); ++i)
        Rb.mData[i] = type_convert<bf16_t>(R.mData[i]);
    HostTensor<bf16_t> Y_bf16({kM, kN});
    {
        DeviceMem Xd(X), Wd(W), Rd(Rb), Yd(Y_bf16);
        (void)launch_kernel(stream_config{nullptr, true, 0, 0, 1},
                            make_kernel(Bf16AccumResidualKernel{},
                                        dim3(1),
                                        dim3(Bf16AccumResidualKernel::kBlockSize),
                                        0,
                                        Xd.GetDeviceBuffer(),
                                        Wd.GetDeviceBuffer(),
                                        Rd.GetDeviceBuffer(),
                                        Yd.GetDeviceBuffer()));
        Yd.FromDevice(Y_bf16.mData.data());
    }

    const double err_bf16f32 = max_abs_err(Y_bf16f32, Y_ref);
    const double err_bf16    = max_abs_err(Y_bf16, Y_ref);

    // bf16f32 is within bf16 tolerance of the fp32 reference ...
    EXPECT_LT(err_bf16f32, 1e-1);
    // ... and is at least as accurate as the bf16-accumulate path (the benefit of fp32 C).
    EXPECT_LE(err_bf16f32, err_bf16);
}
