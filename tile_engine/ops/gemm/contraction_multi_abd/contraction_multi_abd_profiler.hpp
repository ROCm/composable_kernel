// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <algorithm>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <functional>
#include <tuple>

#include "ck_tile/host/device_prop.hpp"
#include "ck_tile/ops/batched_contraction_multi_abd.hpp"
#include "ck_tile/host/reference/reference_batched_contraction.hpp"
#include "contraction_multi_abd_benchmark.hpp"

class ContractionMultiABDProfiler
{
    public:
    using ContractionInstance = KernelInstance<ContractionMultiABDProblem>;

    static ContractionMultiABDProfiler& instance(Settings setting)
    {
        static ContractionMultiABDProfiler inst{setting};
        return inst;
    }

    void
    benchmark(ContractionMultiABDProblem& problem,
              std::function<float(const ck_tile::BatchedContractionMultiABDHostArgs<NumDimG,
                                                                                    NumDimM,
                                                                                    NumDimN,
                                                                                    NumDimK,
                                                                                    NumATensor,
                                                                                    NumBTensor,
                                                                                    NumDTensor>&,
                                  const ck_tile::stream_config&)> kernel_func)
    {
        const auto g_dims = problem.g_dims_;
        const auto m_dims = problem.m_dims_;
        const auto n_dims = problem.n_dims_;
        const auto k_dims = problem.k_dims_;

        const auto g_total = problem.g_total_;
        const auto m_total = problem.m_total_;
        const auto n_total = problem.n_total_;
        const auto k_total = problem.k_total_;

        // A dims: [G..., M..., K...]; strides honor ALayout
        auto a_dims    = concatenate_dims({g_dims, m_dims, k_dims});
        auto a_strides = compute_strides_for_layout<ALayout>(g_dims, m_dims, k_dims);

        // B dims: [G..., N..., K...]; strides honor BLayout
        auto b_dims    = concatenate_dims({g_dims, n_dims, k_dims});
        auto b_strides = compute_strides_for_layout<BLayout>(g_dims, n_dims, k_dims);

        // D/E dims: [G..., M..., N...]; strides honor ELayout (D layouts equal ELayout today)
        auto e_dims    = concatenate_dims({g_dims, m_dims, n_dims});
        auto e_strides = compute_strides_for_layout<ELayout>(g_dims, m_dims, n_dims);

        const auto total_a = g_total * m_total * k_total;
        const auto total_b = g_total * n_total * k_total;
        const auto total_e = g_total * m_total * n_total;

        const auto a_desc = ck_tile::HostTensorDescriptor(
            std::vector<std::size_t>(a_dims.begin(), a_dims.end()),
            std::vector<std::size_t>(a_strides.begin(), a_strides.end()));
        const auto b_desc = ck_tile::HostTensorDescriptor(
            std::vector<std::size_t>(b_dims.begin(), b_dims.end()),
            std::vector<std::size_t>(b_strides.begin(), b_strides.end()));
        const auto e_desc = ck_tile::HostTensorDescriptor(
            std::vector<std::size_t>(e_dims.begin(), e_dims.end()),
            std::vector<std::size_t>(e_strides.begin(), e_strides.end()));

        // Per-tensor element types pulled from the instance tuples; today A_i / B_i / D_i
        // share a single type each, but we still source them through the tuples so the profiler
        // keeps working if the generator is taught to emit divergent types in the future.
        using A0Type = ck_tile::remove_cvref_t<std::tuple_element_t<0, AsDataType>>;
        using A1Type = ck_tile::remove_cvref_t<std::tuple_element_t<1, AsDataType>>;
        using B0Type = ck_tile::remove_cvref_t<std::tuple_element_t<0, BsDataType>>;
        using D0Type = ck_tile::remove_cvref_t<std::tuple_element_t<0, DsDataType>>;

        ck_tile::HostTensor<A0Type> a0_host(a_desc);
        ck_tile::HostTensor<A1Type> a1_host(a_desc);
        ck_tile::HostTensor<B0Type> b0_host(b_desc);
        ck_tile::HostTensor<D0Type> d0_host(e_desc);
        ck_tile::HostTensor<EDataType> e_host_dev_result(e_desc);

        ck_tile::FillUniformDistribution<A0Type>{-5.f, 5.f}(a0_host);
        ck_tile::FillUniformDistribution<A1Type>{-5.f, 5.f}(a1_host);
        ck_tile::FillUniformDistribution<B0Type>{-5.f, 5.f}(b0_host);
        ck_tile::FillUniformDistribution<D0Type>{-1.f, 1.f}(d0_host);

        ck_tile::DeviceMem a0_dev(a0_host.get_element_space_size_in_bytes());
        ck_tile::DeviceMem a1_dev(a1_host.get_element_space_size_in_bytes());
        ck_tile::DeviceMem b0_dev(b0_host.get_element_space_size_in_bytes());
        ck_tile::DeviceMem d0_dev(d0_host.get_element_space_size_in_bytes());
        ck_tile::DeviceMem e_dev(e_host_dev_result.get_element_space_size_in_bytes());

        a0_dev.ToDevice(a0_host.mData.data());
        a1_dev.ToDevice(a1_host.mData.data());
        b0_dev.ToDevice(b0_host.mData.data());
        d0_dev.ToDevice(d0_host.mData.data());
        e_dev.SetZero();
        e_host_dev_result.SetZero();

        using HostArgs = ck_tile::BatchedContractionMultiABDHostArgs<NumDimG,
                                                                     NumDimM,
                                                                     NumDimN,
                                                                     NumDimK,
                                                                     NumATensor,
                                                                     NumBTensor,
                                                                     NumDTensor>;
        using ADims    = typename HostArgs::ADims;
        using BDims    = typename HostArgs::BDims;
        using EDims    = typename HostArgs::EDims;

        std::array<const void*, NumATensor> as_ptr = {a0_dev.GetDeviceBuffer(),
                                                      a1_dev.GetDeviceBuffer()};
        std::array<const void*, NumBTensor> bs_ptr = {b0_dev.GetDeviceBuffer()};
        std::array<const void*, NumDTensor> ds_ptr = {d0_dev.GetDeviceBuffer()};
        std::array<ADims, NumATensor> As_dims{};
        std::array<BDims, NumBTensor> Bs_dims{};
        std::array<EDims, NumDTensor> Ds_dims{};
        std::array<ADims, NumATensor> As_strides{};
        std::array<BDims, NumBTensor> Bs_strides{};
        std::array<EDims, NumDTensor> Ds_strides{};
        for(ck_tile::index_t i = 0; i < NumATensor; ++i)
        {
            As_dims[i]    = to_fixed_dims<NumDimG + NumDimM + NumDimK>(a_dims);
            As_strides[i] = to_fixed_dims<NumDimG + NumDimM + NumDimK>(a_strides);
        }
        for(ck_tile::index_t i = 0; i < NumBTensor; ++i)
        {
            Bs_dims[i]    = to_fixed_dims<NumDimG + NumDimN + NumDimK>(b_dims);
            Bs_strides[i] = to_fixed_dims<NumDimG + NumDimN + NumDimK>(b_strides);
        }
        for(ck_tile::index_t i = 0; i < NumDTensor; ++i)
        {
            Ds_dims[i]    = to_fixed_dims<NumDimG + NumDimM + NumDimN>(e_dims);
            Ds_strides[i] = to_fixed_dims<NumDimG + NumDimM + NumDimN>(e_strides);
        }
        HostArgs host_args{as_ptr,
                           bs_ptr,
                           ds_ptr,
                           e_dev.GetDeviceBuffer(),
                           As_dims,
                           Bs_dims,
                           Ds_dims,
                           to_fixed_dims<NumDimG + NumDimM + NumDimN>(e_dims),
                           As_strides,
                           Bs_strides,
                           Ds_strides,
                           to_fixed_dims<NumDimG + NumDimM + NumDimN>(e_strides)};

        float avg_time =
            kernel_func(host_args,
                        ck_tile::stream_config{
                            nullptr, true, setting_.log, setting_.n_warmup, setting_.n_repeat});

        ContractionInstance ki{std::string(KERNEL_NAME), problem, {-1.0, -1.0, -1.0}};

        std::size_t flop     = std::size_t(2) * g_total * m_total * n_total * k_total;
        std::size_t num_byte = sizeof(A0Type) * NumATensor * total_a +
                               sizeof(B0Type) * NumBTensor * total_b +
                               sizeof(D0Type) * NumDTensor * total_e + sizeof(EDataType) * total_e;

        ki.perf_result_.latency_   = avg_time;
        ki.perf_result_.tflops_    = static_cast<float>(flop) / 1.E9 / avg_time;
        ki.perf_result_.bandwidth_ = num_byte / 1.E6 / avg_time;

        if(setting_.log && !setting_.json_output)
        {
            std::cout << ki << std::endl;
        }

        // Verify if requested
        if(setting_.verify)
        {
            e_dev.FromDevice(e_host_dev_result.mData.data());

            ck_tile::HostTensor<EDataType> e_ref_host(e_desc);
            e_ref_host.SetZero();

            ck_tile::compute_reference_batched_contraction_multi_abd<AsDataType,
                                                                     BsDataType,
                                                                     DsDataType,
                                                                     AccDataType,
                                                                     EDataType,
                                                                     AElementWise,
                                                                     BElementWise,
                                                                     CDEElementWise>(
                {a0_host, a1_host},
                {b0_host},
                {d0_host},
                e_ref_host,
                g_total,
                m_total,
                n_total,
                k_total,
                AElementWise{},
                BElementWise{},
                CDEElementWise{},
                g_dims,
                m_dims,
                n_dims,
                k_dims);

            using ADataType = ck_tile::remove_cvref_t<std::tuple_element_t<0, AsDataType>>;
            using BDataType = ck_tile::remove_cvref_t<std::tuple_element_t<0, BsDataType>>;
            const float max_accumulated_value =
                *std::max_element(e_ref_host.mData.begin(), e_ref_host.mData.end());
            const auto rtol_atol =
                calculate_rtol_atol<ADataType, BDataType, AccDataType, EDataType>(
                    k_total, 1, max_accumulated_value);

            const bool pass = ck_tile::check_err(e_host_dev_result,
                                                 e_ref_host,
                                                 "Contraction multi-ABD: incorrect results!",
                                                 rtol_atol.at(ck_tile::number<0>{}),
                                                 rtol_atol.at(ck_tile::number<1>{}));

            if(!pass)
            {
                std::cout << "Verification failed for kernel: " << KERNEL_NAME << std::endl;
            }
            else
            {
                kernel_instances_.emplace_back(ki);
            }
        }
        else
        {
            kernel_instances_.emplace_back(ki);
        }
    }

    ContractionInstance select_best_instance(Metric metric)
    {
        if(kernel_instances_.empty())
            throw std::runtime_error("Empty instances");

        auto ki = *std::max_element(kernel_instances_.begin(),
                                    kernel_instances_.end(),
                                    [metric](const auto& a, const auto& b) {
                                        return PerformanceResult::compare(
                                            b.perf_result_, a.perf_result_, metric);
                                    });

        if(setting_.json_output)
        {
            std::cout << ki << std::endl;
        }
        else
        {
            std::cout << "**********************************" << std::endl;
            std::cout << "According to given metrics: " << get_metric_name(metric) << "\n"
                      << "Current kernel performance is: " << ki << std::endl;
            std::cout << "**********************************" << std::endl;
        }

        if(!setting_.csv_filename.empty())
        {
            std::ofstream file(setting_.csv_filename + ".csv", std::ios::app);
            if(file.is_open())
            {
                if(file.tellp() == 0)
                {
                    file << "rocm_version,device_name," << "g_total,m_total,n_total,k_total,"
                         << "name,latency(ms),tflops(TFlops),bandwidth(GB/s),metric\n";
                }

                const auto& p    = ki.problem_;
                const auto& perf = ki.perf_result_;

                file << get_rocm_version() << "," << ck_tile::get_device_name() << "," << p.g_total_
                     << "," << p.m_total_ << "," << p.n_total_ << "," << p.k_total_ << ","
                     << ki.name_ << "," << std::fixed << std::setprecision(4) << perf.latency_
                     << "," << perf.tflops_ << "," << perf.bandwidth_ << ","
                     << get_metric_name(metric) << "\n";
            }
        }

        return ki;
    }

    ContractionMultiABDProfiler(const ContractionMultiABDProfiler&)            = delete;
    ContractionMultiABDProfiler& operator=(const ContractionMultiABDProfiler&) = delete;

    private:
    ~ContractionMultiABDProfiler() { kernel_instances_.clear(); }
    ContractionMultiABDProfiler(Settings setting) : setting_(setting) {}

    Settings setting_;
    std::vector<ContractionInstance> kernel_instances_;
};
