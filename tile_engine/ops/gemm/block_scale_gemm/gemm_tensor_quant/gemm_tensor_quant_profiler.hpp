// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <functional>
#include <tuple>

#include "ck_tile/ops/gemm_quant.hpp"
#include "gemm/gemm_profiler.hpp"
#include "gemm_tensor_quant_benchmark.hpp"

class GemmTensorQuantProfiler : public GemmProfiler<GemmTensorQuantProfiler,
                                                    GemmTensorQuantProblem,
                                                    ck_tile::QuantGemmHostArgs>
{
    public:
    using BaseGemm =
        GemmProfiler<GemmTensorQuantProfiler, GemmTensorQuantProblem, ck_tile::QuantGemmHostArgs>;
    using BaseGemm::benchmark;

    GemmTensorQuantProfiler(Settings setting)
        : GemmProfiler<GemmTensorQuantProfiler, GemmTensorQuantProblem, ck_tile::QuantGemmHostArgs>(
              setting)
    {
    }

    void
    benchmark(GemmTensorQuantProblem& gemm_problem,
              std::vector<std::function<std::tuple<std::string, float>(
                  ck_tile::QuantGemmHostArgs&, const ck_tile::stream_config&)>>& callables) override
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

        constexpr ck_tile::index_t single_tensor_scale = 1;
        constexpr ck_tile::index_t scale_stride        = 1;

        ck_tile::HostTensor<ADataType> a_m_k(ck_tile::host_tensor_descriptor(
            gemm_problem.m_, gemm_problem.k_, gemm_problem.stride_a_, is_row_major(layout_a)));
        ck_tile::HostTensor<BDataType> b_k_n(ck_tile::host_tensor_descriptor(
            gemm_problem.k_, gemm_problem.n_, gemm_problem.stride_b_, is_row_major(layout_b)));
        ck_tile::HostTensor<CDataType> c_m_n_dev_result(ck_tile::host_tensor_descriptor(
            gemm_problem.m_, gemm_problem.n_, gemm_problem.stride_c_, is_row_major(layout_c)));

        ck_tile::HostTensor<AQDataType> aq_tensor(
            ck_tile::host_tensor_descriptor(single_tensor_scale,
                                            single_tensor_scale,
                                            scale_stride,
                                            ck_tile::bool_constant<true>{}));
        ck_tile::HostTensor<BQDataType> bq_tensor(
            ck_tile::host_tensor_descriptor(single_tensor_scale,
                                            single_tensor_scale,
                                            scale_stride,
                                            ck_tile::bool_constant<true>{}));

        if(setting_.init_method == 0)
        {
            ck_tile::FillUniformDistribution<ADataType>{-1.f, 1.f}(a_m_k);
            ck_tile::FillUniformDistribution<BDataType>{-1.f, 1.f}(b_k_n);
            ck_tile::FillUniformDistribution<AQDataType>{0.5f, 2.0f}(aq_tensor);
            ck_tile::FillUniformDistribution<BQDataType>{0.5f, 2.0f}(bq_tensor);
        }
        else if(setting_.init_method == 1)
        {
            ck_tile::FillMonotonicSeq<ADataType>{}(a_m_k);
            ck_tile::FillMonotonicSeq<BDataType>{}(b_k_n);
            ck_tile::FillMonotonicSeq<AQDataType>{}(aq_tensor);
            ck_tile::FillMonotonicSeq<BQDataType>{}(bq_tensor);
        }
        else if(setting_.init_method == 2)
        {
            ck_tile::FillConstant<ADataType>{static_cast<ADataType>(1)}(a_m_k);
            ck_tile::FillConstant<BDataType>{static_cast<BDataType>(1)}(b_k_n);
            ck_tile::FillConstant<AQDataType>{static_cast<AQDataType>(1)}(aq_tensor);
            ck_tile::FillConstant<BQDataType>{static_cast<BQDataType>(1)}(bq_tensor);
        }
        else
        {
            a_m_k.SetZero();
            b_k_n.SetZero();
            aq_tensor.SetZero();
            bq_tensor.SetZero();
        }

        ck_tile::DeviceMem a_m_k_dev_buf(a_m_k.get_element_space_size_in_bytes());
        ck_tile::DeviceMem b_k_n_dev_buf(b_k_n.get_element_space_size_in_bytes());
        ck_tile::DeviceMem c_m_n_dev_buf(c_m_n_dev_result.get_element_space_size_in_bytes());

        ck_tile::DeviceMem aq_dev_buf(aq_tensor.get_element_space_size_in_bytes());
        ck_tile::DeviceMem bq_dev_buf(bq_tensor.get_element_space_size_in_bytes());

        a_m_k_dev_buf.ToDevice(a_m_k.data());
        b_k_n_dev_buf.ToDevice(b_k_n.data());
        aq_dev_buf.ToDevice(aq_tensor.data());
        bq_dev_buf.ToDevice(bq_tensor.data());
        c_m_n_dev_buf.SetZero();
        c_m_n_dev_result.SetZero();

        ck_tile::QuantGemmHostArgs gemm_args = {a_m_k_dev_buf.GetDeviceBuffer(),
                                                b_k_n_dev_buf.GetDeviceBuffer(),
                                                c_m_n_dev_buf.GetDeviceBuffer(),
                                                aq_dev_buf.GetDeviceBuffer(),
                                                bq_dev_buf.GetDeviceBuffer(),
                                                gemm_problem.split_k_,
                                                gemm_problem.m_,
                                                gemm_problem.n_,
                                                gemm_problem.k_,
                                                single_tensor_scale,
                                                single_tensor_scale,
                                                gemm_problem.stride_a_,
                                                gemm_problem.stride_b_,
                                                gemm_problem.stride_c_,
                                                scale_stride,
                                                scale_stride};

        ck_tile::HostTensor<CDataType> c_m_n_host_result(ck_tile::host_tensor_descriptor(
            gemm_problem.m_, gemm_problem.n_, gemm_problem.stride_c_, is_row_major(layout_c)));

        if(setting_.verify)
        {
            gemm_tensor_quant_host_reference(
                setting_.verify, a_m_k, aq_tensor, b_k_n, bq_tensor, c_m_n_host_result);
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

    protected:
    std::size_t get_byte_count(const GemmTensorQuantProblem& problem) const override
    {
        std::size_t num_byte = BaseGemm::get_byte_count(problem);

        num_byte += sizeof(AQDataType);
        num_byte += sizeof(BQDataType);

        return num_byte;
    }
};
