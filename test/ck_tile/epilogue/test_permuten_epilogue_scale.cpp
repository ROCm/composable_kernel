// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <algorithm>
#include <cmath>
#include <cstddef>

#include <gtest/gtest.h>

#include "ck_tile/core.hpp"
#include "ck_tile/host.hpp"
#include "ck_tile/host/hip_check_error.hpp"
#include "ck_tile/ops/elementwise.hpp"
#include "ck_tile/ops/epilogue/permuten_epilogue.hpp"
#include "ck_tile/ops/gemm/warp/warp_gemm_dispatcher.hpp"

namespace {

constexpr ck_tile::index_t M = 128;
constexpr ck_tile::index_t N = 256;

using Problem = ck_tile::PermuteNEpilogueProblem<ck_tile::fp8_t,
                                                 ck_tile::fp8_t,
                                                 ck_tile::tuple<>,
                                                 float,
                                                 ck_tile::half_t,
                                                 ck_tile::tuple<>,
                                                 ck_tile::tensor_layout::gemm::RowMajor,
                                                 ck_tile::element_wise::PassThrough,
                                                 M,
                                                 N,
                                                 1,
                                                 4,
                                                 16,
                                                 16,
                                                 128,
                                                 false>;

template <bool ApplyScales>
__global__ void permuten_epilogue_scale_kernel(const float* input,
                                               ck_tile::half_t* output,
                                               float* scale_m,
                                               float* scale_n)
{
    using Epilogue = ck_tile::PermuteNEpilogue<Problem>;
    using WG       = ck_tile::WarpGemmDispatcher<typename Epilogue::ATypeToUse,
                                                 typename Epilogue::BTypeToUse,
                                                 typename Problem::AccDataType,
                                                 Problem::MPerXdl,
                                                 Problem::NPerXdl,
                                                 Problem::KPerXdl,
                                                 Problem::isCTransposed>;

    constexpr ck_tile::index_t MIterPerWarp = M / (Problem::MWave * Problem::MPerXdl);
    constexpr ck_tile::index_t NIterPerWarp = N / (Problem::NWave * Problem::NPerXdl);

    constexpr auto outer_distribution = ck_tile::tile_distribution_encoding<
        ck_tile::sequence<>,
        ck_tile::tuple<ck_tile::sequence<MIterPerWarp, Problem::MWave>,
                       ck_tile::sequence<NIterPerWarp, Problem::NWave>>,
        ck_tile::tuple<ck_tile::sequence<1, 2>>,
        ck_tile::tuple<ck_tile::sequence<1, 1>>,
        ck_tile::sequence<1, 2>,
        ck_tile::sequence<0, 0>>{};

    constexpr auto acc_distribution_encoding =
        ck_tile::detail::make_embed_tile_distribution_encoding(outer_distribution,
                                                               typename WG::CWarpDstrEncoding{});
    constexpr auto acc_distribution =
        ck_tile::make_static_tile_distribution(acc_distribution_encoding);
    auto acc_tile = ck_tile::make_static_distributed_tensor<float>(acc_distribution);

    auto input_view = ck_tile::make_naive_tensor_view<ck_tile::address_space_enum::global>(
        const_cast<float*>(input),
        ck_tile::make_tuple(M, N),
        ck_tile::make_tuple(N, 1),
        ck_tile::number<1>{},
        ck_tile::number<1>{});
    auto input_window =
        ck_tile::make_tile_window(input_view,
                                  ck_tile::make_tuple(ck_tile::number<M>{}, ck_tile::number<N>{}),
                                  {0, 0},
                                  acc_distribution);
    ck_tile::load_tile(acc_tile, input_window);

    auto output_view = ck_tile::make_naive_tensor_view<ck_tile::address_space_enum::global>(
        output,
        ck_tile::make_tuple(M, N),
        ck_tile::make_tuple(N, 1),
        ck_tile::number<Epilogue::GetVectorSizeC()>{},
        ck_tile::number<1>{});
    auto output_window = ck_tile::make_tile_window(
        output_view, ck_tile::make_tuple(ck_tile::number<M>{}, ck_tile::number<N>{}), {0, 0});

    auto empty_ds = ck_tile::make_tuple();
    if constexpr(ApplyScales)
    {
        auto scale_m_view = ck_tile::make_naive_tensor_view<ck_tile::address_space_enum::global>(
            scale_m,
            ck_tile::make_tuple(M, N),
            ck_tile::make_tuple(1, 0),
            ck_tile::number<1>{},
            ck_tile::number<1>{});
        auto scale_n_view = ck_tile::make_naive_tensor_view<ck_tile::address_space_enum::global>(
            scale_n,
            ck_tile::make_tuple(M, N),
            ck_tile::make_tuple(0, 1),
            ck_tile::number<1>{},
            ck_tile::number<1>{});
        auto scale_m_window = ck_tile::make_tile_window(
            scale_m_view, ck_tile::make_tuple(ck_tile::number<M>{}, ck_tile::number<N>{}), {0, 0});
        auto scale_n_window = ck_tile::make_tile_window(
            scale_n_view, ck_tile::make_tuple(ck_tile::number<M>{}, ck_tile::number<N>{}), {0, 0});

        Epilogue{}(output_window, acc_tile, empty_ds, nullptr, scale_m_window, scale_n_window);
    }
    else
    {
        Epilogue{}(output_window, acc_tile, empty_ds, nullptr);
    }
}

TEST(PermuteNEpilogueScale, AppliesPerTokenAndPerChannelScalesAtCanonicalCoordinates)
{
    ck_tile::HostTensor<float> input({M, N});
    ck_tile::HostTensor<float> scale_m({M});
    ck_tile::HostTensor<float> scale_n({N});

    for(ck_tile::index_t m = 0; m < M; ++m)
    {
        scale_m.mData[m] = 0.5F + static_cast<float>(m % 31) * 0.015625F;
        for(ck_tile::index_t n = 0; n < N; ++n)
        {
            input(m, n) = 0.5F + static_cast<float>((m * 5 + n * 3) % 16) * 0.0625F;
        }
    }
    for(ck_tile::index_t n = 0; n < N; ++n)
    {
        scale_n.mData[n] = 0.75F + static_cast<float>(n % 29) * 0.015625F;
    }

    ck_tile::DeviceMem input_device(input.get_element_space_size_in_bytes());
    ck_tile::DeviceMem scale_m_device(scale_m.get_element_space_size_in_bytes());
    ck_tile::DeviceMem scale_n_device(scale_n.get_element_space_size_in_bytes());
    ck_tile::DeviceMem unscaled_device(M * N * sizeof(ck_tile::half_t));
    ck_tile::DeviceMem scaled_device(M * N * sizeof(ck_tile::half_t));
    input_device.ToDevice(input.data());
    scale_m_device.ToDevice(scale_m.data());
    scale_n_device.ToDevice(scale_n.data());
    unscaled_device.SetZero();
    scaled_device.SetZero();

    const dim3 grid(1);
    const dim3 block(Problem::kBlockSize);
    permuten_epilogue_scale_kernel<false>
        <<<grid, block>>>(static_cast<const float*>(input_device.GetDeviceBuffer()),
                          static_cast<ck_tile::half_t*>(unscaled_device.GetDeviceBuffer()),
                          nullptr,
                          nullptr);
    permuten_epilogue_scale_kernel<true>
        <<<grid, block>>>(static_cast<const float*>(input_device.GetDeviceBuffer()),
                          static_cast<ck_tile::half_t*>(scaled_device.GetDeviceBuffer()),
                          static_cast<float*>(scale_m_device.GetDeviceBuffer()),
                          static_cast<float*>(scale_n_device.GetDeviceBuffer()));
    HIP_CHECK_ERROR(hipGetLastError());
    HIP_CHECK_ERROR(hipDeviceSynchronize());

    ck_tile::HostTensor<ck_tile::half_t> unscaled({M, N});
    ck_tile::HostTensor<ck_tile::half_t> scaled({M, N});
    unscaled_device.FromDevice(unscaled.data());
    scaled_device.FromDevice(scaled.data());

    constexpr float tolerance = 0.004F;
    float max_error           = 0.0F;
    std::size_t error_count   = 0;
    for(ck_tile::index_t m = 0; m < M; ++m)
    {
        for(ck_tile::index_t n = 0; n < N; ++n)
        {
            const float expected =
                ck_tile::type_convert<float>(unscaled(m, n)) * scale_m.mData[m] * scale_n.mData[n];
            const float actual = ck_tile::type_convert<float>(scaled(m, n));
            const float error  = std::abs(actual - expected);
            max_error          = std::max(max_error, error);
            error_count += error > tolerance;
        }
    }

    EXPECT_EQ(error_count, 0) << "max error: " << max_error;
}

} // namespace
