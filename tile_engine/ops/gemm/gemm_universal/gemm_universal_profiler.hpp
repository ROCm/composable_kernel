// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <iostream>
#include <fstream>
#include <iomanip>

#include "ck_tile/host/device_prop.hpp"
#include "ck_tile/ops/gemm.hpp"
#include "gemm/gemm_benchmark.hpp"
#include "gemm/gemm_profiler.hpp"
#include "gemm_universal_benchmark.hpp"

class UniversalGemmProfiler
    : public GemmProfiler<UniversalGemmProfiler, GemmProblem, ck_tile::GemmHostArgs>
{
    public:
    using BaseGemm = GemmProfiler<UniversalGemmProfiler, GemmProblem, ck_tile::GemmHostArgs>;
    using BaseGemm::benchmark;

    UniversalGemmProfiler(Settings setting)
        : GemmProfiler<UniversalGemmProfiler, GemmProblem, ck_tile::GemmHostArgs>(setting)
    {
    }

    /// Exact verification (-verify_exact 1): rtol = atol = 0 and a mismatch count, on top
    /// of the tolerance check. Valid only with integer inputs (-init 4), whose fp32
    /// accumulation is exact.
    void set_verify_exact(bool verify_exact) { verify_exact_ = verify_exact; }

    /// True when an exact verification ran and found mismatches.
    bool exact_verification_failed() const { return exact_failed_; }

    void benchmark(GemmProblem& gemm_problem,
                   std::vector<std::function<std::tuple<std::string, float>(
                       ck_tile::GemmHostArgs&, const ck_tile::stream_config&)>>& callables) override
    {
        const ALayout layout_a = ALayout{};
        const BLayout layout_b = BLayout{};
        const CLayout layout_c = CLayout{};

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

        if(setting_.init_method == 0)
        {
            ck_tile::FillUniformDistribution<ADataType>{-1.f, 1.f}(a_m_k);
            ck_tile::FillUniformDistribution<BDataType>{-1.f, 1.f}(b_k_n);
        }
        else if(setting_.init_method == 1)
        {
            ck_tile::FillMonotonicSeq<ADataType>{}(a_m_k);
            ck_tile::FillMonotonicSeq<BDataType>{}(b_k_n);
        }
        else if(setting_.init_method == 2)
        {
            ck_tile::FillConstant<ADataType>{static_cast<ADataType>(1)}(a_m_k);
            ck_tile::FillConstant<BDataType>{static_cast<BDataType>(1)}(b_k_n);
        }
        else if(setting_.init_method == 3)
        {
            // normal(mean 0, sigma 0.1); the second field is the variance
            ck_tile::FillNormalDistribution<ADataType>{0.f, 0.01f, 11939}(a_m_k);
            ck_tile::FillNormalDistribution<BDataType>{0.f, 0.01f, 11940}(b_k_n);
        }
        else if(setting_.init_method == 4)
        {
            // integers in [-5, 5]: exact in every input type, and the fp32 accumulation stays
            // exact while 25 * K < 2^24, so the output can be checked bit for bit
            ck_tile::FillUniformDistributionIntegerValue<ADataType>{-5.f, 5.f, 11939}(a_m_k);
            ck_tile::FillUniformDistributionIntegerValue<BDataType>{-5.f, 5.f, 11940}(b_k_n);
        }
        else
        {
            a_m_k.SetZero();
            b_k_n.SetZero();
        }

        if(gemm_problem.structured_sparsity_)
        {
            ck_tile::AdjustToStructuredSparsity<ADataType>{}(a_m_k);
        }

        ck_tile::DeviceMem a_m_k_dev_buf(a_m_k.get_element_space_size_in_bytes());
        ck_tile::DeviceMem b_k_n_dev_buf(b_k_n.get_element_space_size_in_bytes());
        ck_tile::DeviceMem c_m_n_dev_buf(c_m_n_dev_result.get_element_space_size_in_bytes());

        if constexpr(std::is_same_v<BDataType, ck_tile::pk_int4_t>)
        {
            // Permute vector pk_i4x4 data for device implementation
            ck_tile::HostTensor<BDataType> b_k_n_dev = b_k_n;
            // permute_tensor_b<decltype(b_k_n_dev)>(b_k_n_dev);
            ck_tile::permute_vectors_i4x4_b(b_k_n_dev);
            b_k_n_dev_buf.ToDevice(b_k_n_dev.data());
        }
        else
        {
            b_k_n_dev_buf.ToDevice(b_k_n.data());
        }

        a_m_k_dev_buf.ToDevice(a_m_k.data());
        c_m_n_dev_buf.SetZero();
        c_m_n_dev_result.SetZero();

        ck_tile::GemmHostArgs gemm_args = {
            a_m_k_dev_buf.GetDeviceBuffer(),
            b_k_n_dev_buf.GetDeviceBuffer(),
            c_m_n_dev_buf.GetDeviceBuffer(),
            gemm_problem.split_k_,
            gemm_problem.m_,
            gemm_problem.n_,
            gemm_problem.k_,
            gemm_problem.stride_a_,
            gemm_problem.stride_b_,
            gemm_problem.stride_c_,
        };

        ck_tile::HostTensor<CDataType> c_m_n_host_result(ck_tile::host_tensor_descriptor(
            gemm_problem.m_, gemm_problem.n_, gemm_problem.stride_c_, is_row_major(layout_c)));

        if(setting_.verify)
        {
            gemm_host_reference(setting_.verify,
                                a_m_k,
                                b_k_n,
                                c_m_n_host_result,
                                a_m_k_dev_buf,
                                b_k_n_dev_buf,
                                gemm_problem.m_,
                                gemm_problem.n_,
                                gemm_problem.k_,
                                gemm_problem.stride_a_,
                                gemm_problem.stride_b_,
                                gemm_problem.stride_c_);
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
            if(verify_exact_ && setting_.verify)
            {
                c_m_n_dev_buf.FromDevice(c_m_n_dev_result.data());
                if(count_exact_mismatches(
                       std::get<0>(kernel_run_result), c_m_n_dev_result, c_m_n_host_result) != 0)
                {
                    exact_failed_ = true;
                    std::cout << "Verification failed, skip kernel: "
                              << std::get<0>(kernel_run_result) << std::endl;
                    c_m_n_dev_buf.SetZero();
                    c_m_n_dev_result.SetZero();
                    continue;
                }
            }
            process_result(gemm_problem,
                           c_m_n_dev_buf,
                           c_m_n_host_result,
                           c_m_n_dev_result,
                           kernel_run_result);
        }
    }

    private:
    bool verify_exact_ = false;
    bool exact_failed_ = false;
};
