// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <iostream>
#include <fstream>
#include <iomanip>
#include <numeric>
#include <array>

#include "ck_tile/host/device_prop.hpp"
#include "ck_tile/ops/batched_contraction.hpp"
#include "ck_tile/host/reference/reference_batched_contraction.hpp"
#include "batched_contraction_benchmark.hpp"

class BatchedContractionProfiler
{
    public:
    static BatchedContractionProfiler& instance(Settings setting)
    {
        static BatchedContractionProfiler instance{setting};
        return instance;
    }

    // Overload for single kernel benchmarking
    void benchmark(BatchedContractionProblem& problem,
                   std::function<float(const ck_tile::BatchedContractionHostArgs<NUM_D_TENSORS>&,
                                       const ck_tile::stream_config&)> kernel_func)
    {
        std::vector<std::function<std::tuple<std::string, float>(
            ck_tile::BatchedContractionHostArgs<NUM_D_TENSORS>&, const ck_tile::stream_config&)>>
            callables;

        callables.push_back([kernel_func](ck_tile::BatchedContractionHostArgs<NUM_D_TENSORS>& args,
                                          const ck_tile::stream_config& stream) {
            float time = kernel_func(args, stream);
            return std::make_tuple(std::string(KERNEL_NAME), time);
        });

        benchmark(problem, callables);
    }

    void benchmark(
        BatchedContractionProblem& problem,
        std::vector<std::function<std::tuple<std::string, float>(
            ck_tile::BatchedContractionHostArgs<NUM_D_TENSORS>&, const ck_tile::stream_config&)>>&
            callables)
    {
        // Parse dimensions
        const auto& G_dims = problem.g_dims_;
        const auto& M_dims = problem.m_dims_;
        const auto& N_dims = problem.n_dims_;
        const auto& K_dims = problem.k_dims_;

        const auto G_total = problem.G_total();
        const auto M_total = problem.M_total();
        const auto N_total = problem.N_total();
        const auto K_total = problem.K_total();

        // Construct full dimension vectors for each tensor
        // A: [G0,G1,..., M0,M1,..., K0,K1,...]
        // B: [G0,G1,..., N0,N1,..., K0,K1,...]
        // E: [G0,G1,..., M0,M1,..., N0,N1,...]
        // D: [G0,G1,..., M0,M1,..., N0,N1,...] (same as E)
        std::vector<ck_tile::index_t> A_dims = concatenate_dims({G_dims, M_dims, K_dims});
        std::vector<ck_tile::index_t> B_dims = concatenate_dims({G_dims, N_dims, K_dims});
        std::vector<ck_tile::index_t> E_dims = concatenate_dims({G_dims, M_dims, N_dims});

        // Create host tensor descriptors (row-major contiguous by default)
        ck_tile::HostTensorDescriptor a_desc(A_dims);
        ck_tile::HostTensorDescriptor b_desc(B_dims);
        ck_tile::HostTensorDescriptor e_desc(E_dims);

        // Create host tensors
        ck_tile::HostTensor<ADataType> a_tensor(a_desc);
        ck_tile::HostTensor<BDataType> b_tensor(b_desc);
        ck_tile::HostTensor<EDataType> e_dev_result(e_desc);

        // Create D tensor descriptors and tensors
        std::array<ck_tile::HostTensorDescriptor, NUM_D_TENSORS> ds_descs;
        for(ck_tile::index_t d = 0; d < NUM_D_TENSORS; ++d)
        {
            ds_descs[d] = ck_tile::HostTensorDescriptor(E_dims);
        }

        auto ds_tensors = make_ds_host_tensors<DBaseDataType, NUM_D_TENSORS>(ds_descs);

        // Fill tensors with random data
        ck_tile::FillUniformDistribution<ADataType>{-5.f, 5.f}(a_tensor);
        ck_tile::FillUniformDistribution<BDataType>{-5.f, 5.f}(b_tensor);
        for(ck_tile::index_t d = 0; d < NUM_D_TENSORS; ++d)
        {
            ck_tile::FillUniformDistribution<DBaseDataType>{-1.f, 1.f}(ds_tensors[d]);
        }

        // Allocate device memory
        ck_tile::DeviceMem a_dev_buf(a_tensor.get_element_space_size_in_bytes());
        ck_tile::DeviceMem b_dev_buf(b_tensor.get_element_space_size_in_bytes());
        ck_tile::DeviceMem e_dev_buf(e_dev_result.get_element_space_size_in_bytes());

        std::array<ck_tile::DeviceMem, NUM_D_TENSORS> ds_dev_bufs;
        for(ck_tile::index_t d = 0; d < NUM_D_TENSORS; ++d)
        {
            ds_dev_bufs[d].Realloc(ds_tensors[d].get_element_space_size_in_bytes());
        }

        // Copy to device
        a_dev_buf.ToDevice(a_tensor.mData.data());
        b_dev_buf.ToDevice(b_tensor.mData.data());
        for(ck_tile::index_t d = 0; d < NUM_D_TENSORS; ++d)
        {
            ds_dev_bufs[d].ToDevice(ds_tensors[d].mData.data());
        }

        e_dev_buf.SetZero();
        e_dev_result.SetZero();

        // Prepare D tensor pointers
        std::array<const void*, NUM_D_TENSORS> ds_ptr_buf;
        for(ck_tile::index_t d = 0; d < NUM_D_TENSORS; ++d)
        {
            ds_ptr_buf[d] = ds_dev_bufs[d].GetDeviceBuffer();
        }

        // Strides from host tensor descriptors
        auto convert_strides = [](const std::vector<std::size_t>& strides) {
            std::vector<ck_tile::index_t> converted(strides.size());
            std::copy(strides.begin(), strides.end(), converted.begin());
            return converted;
        };

        std::vector<ck_tile::index_t> A_strides = convert_strides(a_tensor.get_strides());
        std::vector<ck_tile::index_t> B_strides = convert_strides(b_tensor.get_strides());
        std::vector<ck_tile::index_t> E_strides = convert_strides(e_dev_result.get_strides());

        std::array<std::vector<ck_tile::index_t>, NUM_D_TENSORS> Ds_dims;
        std::array<std::vector<ck_tile::index_t>, NUM_D_TENSORS> Ds_strides;
        for(ck_tile::index_t d = 0; d < NUM_D_TENSORS; ++d)
        {
            Ds_dims[d]    = E_dims;
            Ds_strides[d] = convert_strides(ds_tensors[d].get_strides());
        }

        // Create BatchedContractionHostArgs
        ck_tile::BatchedContractionHostArgs<NUM_D_TENSORS> contraction_args(
            a_dev_buf.GetDeviceBuffer(),
            b_dev_buf.GetDeviceBuffer(),
            ds_ptr_buf,
            e_dev_buf.GetDeviceBuffer(),
            problem.split_k_,
            A_dims,
            B_dims,
            Ds_dims,
            E_dims,
            A_strides,
            B_strides,
            Ds_strides,
            E_strides);

        // Compute host reference if verification is requested
        ck_tile::HostTensor<EDataType> e_host_result(e_desc);
        if(setting_.verify)
        {
            batched_contraction_host_reference<ADataType,
                                               BDataType,
                                               DBaseDataType,
                                               AccDataType,
                                               EDataType,
                                               CDEElementWise,
                                               NUM_D_TENSORS>(setting_.verify,
                                                              a_tensor,
                                                              b_tensor,
                                                              ds_tensors,
                                                              e_host_result,
                                                              G_total,
                                                              M_total,
                                                              N_total,
                                                              K_total,
                                                              CDEElementWise{},
                                                              G_dims,
                                                              M_dims,
                                                              N_dims,
                                                              K_dims);
        }

        for(auto& callable : callables)
        {
            auto kernel_run_result = callable(contraction_args,
                                              ck_tile::stream_config{nullptr,
                                                                     true,
                                                                     setting_.log,
                                                                     setting_.n_warmup,
                                                                     setting_.n_repeat,
                                                                     setting_.is_gpu_timer,
                                                                     setting_.flush_cache,
                                                                     setting_.rotating_count});
            process_result(problem, e_dev_buf, e_host_result, e_dev_result, kernel_run_result);
        }
    }

    void process_result(const BatchedContractionProblem& problem,
                        ck_tile::DeviceMem& e_dev_buf,
                        ck_tile::HostTensor<EDataType>& e_host_result,
                        ck_tile::HostTensor<EDataType>& e_dev_result,
                        const std::tuple<std::string, float>& kernel_run_result)
    {
        auto [name, avg_time] = kernel_run_result;

        KernelInstance<BatchedContractionProblem> kernel_instance{
            name, problem, {-1.0f, -1.0f, -1.0f}};

        const auto M_total = problem.M_total();
        const auto N_total = problem.N_total();
        const auto K_total = problem.K_total();
        const auto G_total = problem.G_total();

        // compute performance metric
        std::size_t flop = std::size_t(2) * G_total * M_total * N_total * K_total;
        std::size_t num_byte =
            G_total *
            (sizeof(ADataType) * M_total * K_total + sizeof(BDataType) * N_total * K_total +
             sizeof(EDataType) * M_total * N_total +
             NUM_D_TENSORS * sizeof(DBaseDataType) * M_total * N_total);

        // update
        kernel_instance.perf_result_.latency_   = avg_time;
        kernel_instance.perf_result_.tflops_    = static_cast<float>(flop) / 1.E9 / avg_time;
        kernel_instance.perf_result_.bandwidth_ = num_byte / 1.E6 / avg_time;

        if(setting_.log > 0 && !setting_.json_output)
        {
            std::cout << kernel_instance << std::endl;
        }

        // verify result
        e_dev_buf.FromDevice(e_dev_result.data());
        bool verified_correct = !setting_.verify;

        if(setting_.verify)
        {
            if constexpr(NUM_D_TENSORS > 0)
            {
                verified_correct =
                    compare<ADataType, BDataType, AccDataType, EDataType, DBaseDataType>(
                        name, K_total, problem.split_k_, e_dev_result, e_host_result);
            }
            else
            {
                verified_correct = compare<ADataType, BDataType, AccDataType, EDataType>(
                    name, K_total, problem.split_k_, e_dev_result, e_host_result);
            }
        }

        if(verified_correct)
        {
            kernel_instances_.emplace_back(kernel_instance);
        }
        else
        {
            std::cout << "Verification failed, skip kernel: " << name << std::endl;
        }

        // clear tensor
        e_dev_buf.SetZero();
        e_dev_result.SetZero();
    }

    KernelInstance<BatchedContractionProblem> select_best_instance(Metric metric)
    {
        if(kernel_instances_.empty())
            throw std::runtime_error("Empty instances");

        auto kernel_instance = *std::max_element(kernel_instances_.begin(),
                                                 kernel_instances_.end(),
                                                 [metric](const auto& a, const auto& b) {
                                                     return PerformanceResult::compare(
                                                         b.perf_result_, a.perf_result_, metric);
                                                 });

        if(setting_.json_output)
        {
            // Output clean JSON only
            std::cout << kernel_instance << std::endl;
        }
        else
        {
            std::cout << "**********************************" << std::endl;
            std::cout << "According to given metrics: " << get_metric_name(metric) << "\n"
                      << "Current kernel performance is: " << kernel_instance << std::endl;
            std::cout << "**********************************" << std::endl;
        }

        if(!setting_.csv_filename.empty())
        {
            std::ofstream file(setting_.csv_filename + ".csv", std::ios::app);

            if(!file.is_open())
            {
                std::cerr << "Warning: Failed to open CSV file for writing." << std::endl;
            }
            else
            {
                if(file.tellp() == 0)
                {
                    file << "rocm_version,device_name," << "split_k,g_dims,m_dims,n_dims,k_dims,"
                         << "G_total,M_total,N_total,K_total," << "num_d_tensors,"
                         << "dtype_a,dtype_b,dtype_acc,dtype_e," << "layout_a,layout_b,layout_e,"
                         << "name," << "latency(ms),tflops(TFlops),bandwidth(GB/s),metric\n";
                }

                const auto& prob = kernel_instance.problem_;
                const auto& perf = kernel_instance.perf_result_;

                file << get_rocm_version() << "," << ck_tile::get_device_name() << ","
                     << prob.split_k_ << "," << dims_to_string(prob.g_dims_) << ","
                     << dims_to_string(prob.m_dims_) << "," << dims_to_string(prob.n_dims_) << ","
                     << dims_to_string(prob.k_dims_) << "," << prob.G_total() << ","
                     << prob.M_total() << "," << prob.N_total() << "," << prob.K_total() << ","
                     << prob.num_d_tensors_ << "," << prob.dtype_a_ << "," << prob.dtype_b_ << ","
                     << prob.dtype_acc_ << "," << prob.dtype_e_ << "," << prob.layout_a_ << ","
                     << prob.layout_b_ << "," << prob.layout_e_ << "," << kernel_instance.name_
                     << "," << std::fixed << std::setprecision(4) << perf.latency_ << ","
                     << std::fixed << std::setprecision(4) << perf.tflops_ << "," << std::fixed
                     << std::setprecision(4) << perf.bandwidth_ << "," << get_metric_name(metric)
                     << "\n";

                if(!file)
                {
                    std::cerr << "Warning: Error occurred while writing to CSV file." << std::endl;
                }
            }
        }

        return kernel_instance;
    }

    BatchedContractionProfiler(const BatchedContractionProfiler&)            = delete;
    BatchedContractionProfiler& operator=(const BatchedContractionProfiler&) = delete;

    private:
    ~BatchedContractionProfiler() { kernel_instances_.clear(); }
    BatchedContractionProfiler(Settings setting) : setting_(setting) {}

    // Helper to create array of HostTensors from descriptors
    template <typename DType, std::size_t N, std::size_t... Is>
    static std::array<ck_tile::HostTensor<DType>, N>
    make_ds_host_tensors_impl(const std::array<ck_tile::HostTensorDescriptor, N>& descs,
                              std::index_sequence<Is...>)
    {
        return {ck_tile::HostTensor<DType>(descs[Is])...};
    }

    template <typename DType, std::size_t N>
    static std::array<ck_tile::HostTensor<DType>, N>
    make_ds_host_tensors(const std::array<ck_tile::HostTensorDescriptor, N>& descs)
    {
        return make_ds_host_tensors_impl<DType, N>(descs, std::make_index_sequence<N>{});
    }

    Settings setting_;
    std::vector<KernelInstance<BatchedContractionProblem>> kernel_instances_;
};
