// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <iostream>
#include <fstream>
#include <iomanip>
#include <vector>
#include <algorithm>
#include <functional>
#include <tuple>

#include "ck_tile/host/device_prop.hpp"
#include "ck_tile/ops/gemm.hpp"
#include "gemm/gemm_profiler.hpp"
#include "common/utils.hpp"
#include "gemm_multi_d_benchmark.hpp"

class GemmMultiDProfiler : public GemmProfiler<GemmMultiDProfiler,
                                               GemmMultiDProblem,
                                               ck_tile::GemmMultiDHostArgs<DsDataType::size()>>
{
    public:
    using BaseGemm = GemmProfiler<GemmMultiDProfiler,
                                  GemmMultiDProblem,
                                  ck_tile::GemmMultiDHostArgs<DsDataType::size()>>;
    using BaseGemm::benchmark;

    GemmMultiDProfiler(Settings setting)
        : GemmProfiler<GemmMultiDProfiler,
                       GemmMultiDProblem,
                       ck_tile::GemmMultiDHostArgs<DsDataType::size()>>(setting)
    {
    }

    void benchmark(
        GemmMultiDProblem& gemm_multi_d_problem,
        std::vector<std::function<std::tuple<std::string, float>(
            ck_tile::GemmMultiDHostArgs<DsDataType::size()>&, const ck_tile::stream_config&)>>&
            callables) override
    {
        const ALayout layout_a   = ALayout{};
        const BLayout layout_b   = BLayout{};
        const D0Layout layout_d0 = D0Layout{};
        const D1Layout layout_d1 = D1Layout{};
        const CLayout layout_c   = CLayout{};

        gemm_multi_d_problem.stride_a_ = ck_tile::get_default_stride(gemm_multi_d_problem.m_,
                                                                     gemm_multi_d_problem.k_,
                                                                     gemm_multi_d_problem.stride_a_,
                                                                     is_row_major(layout_a));
        gemm_multi_d_problem.stride_b_ = ck_tile::get_default_stride(gemm_multi_d_problem.k_,
                                                                     gemm_multi_d_problem.n_,
                                                                     gemm_multi_d_problem.stride_b_,
                                                                     is_row_major(layout_b));
        gemm_multi_d_problem.stride_d0_ =
            ck_tile::get_default_stride(gemm_multi_d_problem.m_,
                                        gemm_multi_d_problem.n_,
                                        gemm_multi_d_problem.stride_d0_,
                                        is_row_major(layout_d0));
        gemm_multi_d_problem.stride_d1_ =
            ck_tile::get_default_stride(gemm_multi_d_problem.m_,
                                        gemm_multi_d_problem.n_,
                                        gemm_multi_d_problem.stride_d1_,
                                        is_row_major(layout_d1));
        gemm_multi_d_problem.stride_c_ = ck_tile::get_default_stride(gemm_multi_d_problem.m_,
                                                                     gemm_multi_d_problem.n_,
                                                                     gemm_multi_d_problem.stride_c_,
                                                                     is_row_major(layout_c));

        ck_tile::HostTensor<ADataType> a_m_k(
            ck_tile::host_tensor_descriptor(gemm_multi_d_problem.m_,
                                            gemm_multi_d_problem.k_,
                                            gemm_multi_d_problem.stride_a_,
                                            is_row_major(layout_a)));
        ck_tile::HostTensor<BDataType> b_k_n(
            ck_tile::host_tensor_descriptor(gemm_multi_d_problem.k_,
                                            gemm_multi_d_problem.n_,
                                            gemm_multi_d_problem.stride_b_,
                                            is_row_major(layout_b)));
        ck_tile::HostTensor<D0DataType> d0_m_n(
            ck_tile::host_tensor_descriptor(gemm_multi_d_problem.m_,
                                            gemm_multi_d_problem.n_,
                                            gemm_multi_d_problem.stride_d0_,
                                            is_row_major(layout_d0)));
        ck_tile::HostTensor<D1DataType> d1_m_n(
            ck_tile::host_tensor_descriptor(gemm_multi_d_problem.m_,
                                            gemm_multi_d_problem.n_,
                                            gemm_multi_d_problem.stride_d1_,
                                            is_row_major(layout_d1)));
        ck_tile::HostTensor<CDataType> c_m_n_dev_result(
            ck_tile::host_tensor_descriptor(gemm_multi_d_problem.m_,
                                            gemm_multi_d_problem.n_,
                                            gemm_multi_d_problem.stride_c_,
                                            is_row_major(layout_c)));

        ck_tile::FillUniformDistribution<ADataType>{-5.f, 5.f}(a_m_k);
        ck_tile::FillUniformDistribution<BDataType>{-5.f, 5.f}(b_k_n);
        ck_tile::FillUniformDistribution<D0DataType>{-1.f, 1.f}(d0_m_n);
        ck_tile::FillUniformDistribution<D1DataType>{-1.f, 1.f}(d1_m_n);

        ck_tile::DeviceMem a_m_k_dev_buf(a_m_k.get_element_space_size_in_bytes());
        ck_tile::DeviceMem b_k_n_dev_buf(b_k_n.get_element_space_size_in_bytes());
        ck_tile::DeviceMem d0_m_n_dev_buf(d0_m_n.get_element_space_size_in_bytes());
        ck_tile::DeviceMem d1_m_n_dev_buf(d1_m_n.get_element_space_size_in_bytes());
        ck_tile::DeviceMem c_m_n_dev_buf(c_m_n_dev_result.get_element_space_size_in_bytes());

        a_m_k_dev_buf.ToDevice(a_m_k.mData.data());
        b_k_n_dev_buf.ToDevice(b_k_n.mData.data());
        d0_m_n_dev_buf.ToDevice(d0_m_n.mData.data());
        d1_m_n_dev_buf.ToDevice(d1_m_n.mData.data());

        c_m_n_dev_buf.SetZero();
        c_m_n_dev_result.SetZero();

        std::array<const void*, DsDataType::size()> ds_ptr_buf = {d0_m_n_dev_buf.GetDeviceBuffer(),
                                                                  d1_m_n_dev_buf.GetDeviceBuffer()};

        std::array<ck_tile::index_t, DsDataType::size()> stridesDs = {
            gemm_multi_d_problem.stride_d0_, gemm_multi_d_problem.stride_d1_};

        ck_tile::GemmMultiDHostArgs<DsDataType::size()> gemm_multi_d_args = {
            a_m_k_dev_buf.GetDeviceBuffer(),
            b_k_n_dev_buf.GetDeviceBuffer(),
            ds_ptr_buf,
            c_m_n_dev_buf.GetDeviceBuffer(),
            gemm_multi_d_problem.split_k_,
            gemm_multi_d_problem.m_,
            gemm_multi_d_problem.n_,
            gemm_multi_d_problem.k_,
            gemm_multi_d_problem.stride_a_,
            gemm_multi_d_problem.stride_b_,
            stridesDs,
            gemm_multi_d_problem.stride_c_,
        };

        ck_tile::HostTensor<CDataType> c_m_n_host_result(
            ck_tile::host_tensor_descriptor(gemm_multi_d_problem.m_,
                                            gemm_multi_d_problem.n_,
                                            gemm_multi_d_problem.stride_c_,
                                            is_row_major(layout_c)));

        if(setting_.verify)
        {
            gemm_multi_d_host_reference(
                setting_.verify, a_m_k, b_k_n, d0_m_n, d1_m_n, c_m_n_host_result);
        }

        for(auto& callable : callables)
        {
            auto kernel_run_result = callable(gemm_multi_d_args,
                                              ck_tile::stream_config{nullptr,
                                                                     true,
                                                                     setting_.log,
                                                                     setting_.n_warmup,
                                                                     setting_.n_repeat,
                                                                     setting_.is_gpu_timer,
                                                                     setting_.flush_cache,
                                                                     setting_.rotating_count});
            process_result(gemm_multi_d_problem,
                           c_m_n_dev_buf,
                           c_m_n_host_result,
                           c_m_n_dev_result,
                           kernel_run_result);
        }
    }
};
