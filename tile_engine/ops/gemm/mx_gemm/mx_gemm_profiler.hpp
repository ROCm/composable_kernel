// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <vector>

#include "ck_tile/host/device_prop.hpp"
#include "gemm/gemm_profiler.hpp"
#include "mx_gemm_benchmark.hpp"

class MXGemmProfiler : public GemmProfiler<MXGemmProfiler, GemmProblem, MxGemmHostArgs>
{
    public:
    using BaseGemm = GemmProfiler<MXGemmProfiler, GemmProblem, MxGemmHostArgs>;
    using BaseGemm::benchmark;

    MXGemmProfiler(Settings setting) : BaseGemm(setting) {}

    void benchmark(GemmProblem& gemm_problem,
                   std::vector<std::function<std::tuple<std::string, float>(
                       MxGemmHostArgs&, const ck_tile::stream_config&)>>& callables) override
    {
        const ALayout layout_a = ALayout{};
        const BLayout layout_b = BLayout{};
        const CLayout layout_c = CLayout{};

        if(gemm_problem.k_ % 32 != 0)
        {
            throw std::runtime_error("MX GEMM requires K to be a multiple of 32");
        }

        gemm_problem.stride_a_ = ck_tile::get_default_stride(
            gemm_problem.m_, gemm_problem.k_, gemm_problem.stride_a_, is_row_major(layout_a));
        gemm_problem.stride_b_ = ck_tile::get_default_stride(
            gemm_problem.k_, gemm_problem.n_, gemm_problem.stride_b_, is_row_major(layout_b));
        gemm_problem.stride_c_ = ck_tile::get_default_stride(
            gemm_problem.m_, gemm_problem.n_, gemm_problem.stride_c_, is_row_major(layout_c));

        ck_tile::HostTensor<ADataType> a_m_k(ck_tile::host_tensor_descriptor(
            gemm_problem.m_, gemm_problem.k_, gemm_problem.stride_a_, is_row_major(layout_a)));
        ck_tile::HostTensor<BDataType> b_k_n(ck_tile::host_tensor_descriptor(
            gemm_problem.k_, gemm_problem.n_, gemm_problem.stride_b_, is_row_major(layout_b)));
        ck_tile::HostTensor<CDataType> c_m_n_dev_result(ck_tile::host_tensor_descriptor(
            gemm_problem.m_, gemm_problem.n_, gemm_problem.stride_c_, is_row_major(layout_c)));

        const ck_tile::index_t scale_k_size = gemm_problem.k_ / 32;

        ck_tile::HostTensor<ScaleType> scale_a_host(
            {static_cast<std::size_t>(gemm_problem.m_), static_cast<std::size_t>(scale_k_size)},
            {static_cast<std::size_t>(scale_k_size), static_cast<std::size_t>(1)});
        ck_tile::HostTensor<ScaleType> scale_b_host(
            {static_cast<std::size_t>(gemm_problem.n_), static_cast<std::size_t>(scale_k_size)},
            {static_cast<std::size_t>(scale_k_size), static_cast<std::size_t>(1)});

        if(setting_.init_method == 0)
        {
            int seed = 1234;
            ck_tile::FillUniformDistribution<ADataType>{-2.f, 2.f, seed++}(a_m_k);
            ck_tile::FillUniformDistribution<BDataType>{-2.f, 2.f, seed++}(b_k_n);
            ck_tile::FillUniformDistribution<ScaleType>{0.001f, 10.f, seed++}(scale_a_host);
            ck_tile::FillUniformDistribution<ScaleType>{0.001f, 10.f, seed++}(scale_b_host);
        }
        else if(setting_.init_method == 1)
        {
            ck_tile::FillConstant<ADataType>{ADataType(1.f)}(a_m_k);
            ck_tile::FillConstant<BDataType>{BDataType(1.f)}(b_k_n);
            ck_tile::FillConstant<ScaleType>{ScaleType(1.f)}(scale_a_host);
            ck_tile::FillConstant<ScaleType>{ScaleType(1.f)}(scale_b_host);
        }
        else if(setting_.init_method == 2)
        {
            int seed = 1234;
            ck_tile::FillUniformDistribution<ADataType>{-2.f, 2.f, seed++}(a_m_k);
            ck_tile::FillUniformDistribution<BDataType>{-2.f, 2.f, seed++}(b_k_n);
            ck_tile::FillConstant<ScaleType>{ScaleType(0.1f)}(scale_a_host);
            ck_tile::FillConstant<ScaleType>{ScaleType(0.1f)}(scale_b_host);
        }
        else
        {
            a_m_k.SetZero();
            b_k_n.SetZero();
            scale_a_host.SetZero();
            scale_b_host.SetZero();
        }

        constexpr ck_tile::index_t m_per_xdl = SelectedKernel::WarpTileM;
        constexpr ck_tile::index_t n_per_xdl = SelectedKernel::WarpTileN;
        constexpr ck_tile::index_t k_per_xdl = SelectedKernel::WarpTileK;
        constexpr ck_tile::index_t m_iter_per_warp =
            SelectedKernel::TileM / (SelectedKernel::WarpPerBlock_M * m_per_xdl);
        constexpr ck_tile::index_t n_iter_per_warp =
            SelectedKernel::TileN / (SelectedKernel::WarpPerBlock_N * n_per_xdl);
        constexpr ck_tile::index_t k_iter_per_warp = SelectedKernel::TileK / k_per_xdl;

        constexpr ck_tile::index_t m_xdl_pack =
            (m_iter_per_warp >= 2 && m_iter_per_warp % 2 == 0) ? 2 : 1;
        constexpr ck_tile::index_t n_xdl_pack =
            (n_iter_per_warp >= 2 && n_iter_per_warp % 2 == 0) ? 2 : 1;
        constexpr ck_tile::index_t k_xdl_pack =
            (k_iter_per_warp >= 2 && k_iter_per_warp % 2 == 0) ? 2 : 1;

        constexpr ck_tile::index_t xdl_mn_thread = SelectedKernel::WarpTileM;
        constexpr ck_tile::index_t xdl_k_thread  = 64 / xdl_mn_thread;

        ck_tile::HostTensor<ScaleType> scale_a_shuffled(
            {static_cast<std::size_t>(gemm_problem.m_ / m_xdl_pack * 2),
             static_cast<std::size_t>(scale_k_size / k_xdl_pack * 2)},
            {static_cast<std::size_t>(scale_k_size / k_xdl_pack * 2), static_cast<std::size_t>(1)});

        ck_tile::HostTensor<ScaleType> scale_b_shuffled(
            {static_cast<std::size_t>(gemm_problem.n_ / n_xdl_pack * 2),
             static_cast<std::size_t>(scale_k_size / k_xdl_pack * 2)},
            {static_cast<std::size_t>(scale_k_size / k_xdl_pack * 2), static_cast<std::size_t>(1)});

        ck_tile::preShuffleScaleBuffer_gfx950<m_xdl_pack, k_xdl_pack, xdl_mn_thread, xdl_k_thread>(
            scale_a_host.mData.data(),
            scale_a_shuffled.mData.data(),
            gemm_problem.m_,
            scale_k_size,
            true);

        ck_tile::preShuffleScaleBuffer_gfx950<n_xdl_pack, k_xdl_pack, xdl_mn_thread, xdl_k_thread>(
            scale_b_host.mData.data(),
            scale_b_shuffled.mData.data(),
            gemm_problem.n_,
            scale_k_size,
            true);

        ck_tile::DeviceMem a_m_k_dev_buf(a_m_k.get_element_space_size_in_bytes());
        ck_tile::DeviceMem b_k_n_dev_buf(b_k_n.get_element_space_size_in_bytes());
        ck_tile::DeviceMem c_m_n_dev_buf(c_m_n_dev_result.get_element_space_size_in_bytes());
        ck_tile::DeviceMem scale_a_dev_buf(scale_a_shuffled.get_element_space_size_in_bytes());
        ck_tile::DeviceMem scale_b_dev_buf(scale_b_shuffled.get_element_space_size_in_bytes());

        a_m_k_dev_buf.ToDevice(a_m_k.data());
        b_k_n_dev_buf.ToDevice(b_k_n.data());
        c_m_n_dev_buf.SetZero();
        c_m_n_dev_result.SetZero();
        scale_a_dev_buf.ToDevice(scale_a_shuffled.data());
        scale_b_dev_buf.ToDevice(scale_b_shuffled.data());

        MxGemmHostArgs gemm_args({a_m_k_dev_buf.GetDeviceBuffer()},
                                 {scale_a_dev_buf.GetDeviceBuffer()},
                                 {b_k_n_dev_buf.GetDeviceBuffer()},
                                 {scale_b_dev_buf.GetDeviceBuffer()},
                                 {},
                                 c_m_n_dev_buf.GetDeviceBuffer(),
                                 gemm_problem.split_k_,
                                 gemm_problem.m_,
                                 gemm_problem.n_,
                                 gemm_problem.k_,
                                 {gemm_problem.stride_a_},
                                 {gemm_problem.stride_b_},
                                 {},
                                 gemm_problem.stride_c_);

        ck_tile::HostTensor<CDataType> c_m_n_host_result(ck_tile::host_tensor_descriptor(
            gemm_problem.m_, gemm_problem.n_, gemm_problem.stride_c_, is_row_major(layout_c)));

        if(setting_.verify)
        {
            // Host reference computation using reference_mx_gemm
            // reference_mx_gemm expects scale_a(M, K/ScaleBlockSize) and scale_b(K/ScaleBlockSize,
            // N) We need to create scale_b in (K/ScaleBlockSize, N) format for the reference
            ck_tile::HostTensor<ScaleType> scale_b_ref(
                {static_cast<std::size_t>(scale_k_size), static_cast<std::size_t>(gemm_problem.n_)},
                {static_cast<std::size_t>(1), static_cast<std::size_t>(scale_k_size)});
            // Copy scale_b data (our scale_b is (N, scale_k_size) row-major,
            // reference expects (scale_k_size, N) col-major, which is the same memory layout)
            std::copy(
                scale_b_host.mData.begin(), scale_b_host.mData.end(), scale_b_ref.mData.begin());

            mx_gemm_host_reference(
                setting_.verify, a_m_k, b_k_n, c_m_n_host_result, scale_a_host, scale_b_ref);
        }

        for(auto& callable : callables)
        {
            auto kernel_run_result = callable(gemm_args,
                                              ck_tile::stream_config{nullptr,
                                                                     true,
                                                                     setting_.log,
                                                                     setting_.n_warmup,
                                                                     setting_.n_repeat,
                                                                     setting_.is_gpu_timer,
                                                                     setting_.flush_cache,
                                                                     setting_.rotating_count});
            process_result(gemm_problem,
                           c_m_n_dev_buf,
                           c_m_n_host_result,
                           c_m_n_dev_result,
                           kernel_run_result);
        }
    }
};
