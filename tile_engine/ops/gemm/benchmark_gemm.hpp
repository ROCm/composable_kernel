// SPDX-License-Identifier: MIT
// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include <iostream>
#include <vector>
#include <filesystem>
#include <memory>

#include "ck/host_utility/device_prop.hpp"
#include "profile_cache.hpp"

class Executor
{
    public:
    ~Executor() { kernel_instances_.clear(); }

    static Executor& instance(bool enable_profile_cache = true, bool flush_profile_cache = false)
    {
        static Executor instance{enable_profile_cache, flush_profile_cache};
        return instance;
    }

    static std::string get_rocm_version()
    {
        std::ifstream version_file("/opt/rocm/.info/version");
        if(version_file.is_open())
        {
            std::string version;
            std::getline(version_file, version);
            return version;
        }
        return "Unknown";
    }

    template <typename Kernel>
    void benchmark_kernel(ck_tile::DeviceMem& c_m_n_dev_buf,
                          ck_tile::HostTensor<CDataType>& c_m_n_host_result,
                          ck_tile::HostTensor<CDataType>& c_m_n_dev_result,
                          int verify,
                          ck_tile::GemmHostArgs& args,
                          const ck_tile::stream_config& s)
    {
        std::string description = Kernel::get_name();

        GemmProblem problem{args.k_batch,
                            args.M,
                            args.N,
                            args.K,
                            args.stride_A,
                            args.stride_B,
                            args.stride_C,
                            DataTypeTraits<ADataType>::name,
                            DataTypeTraits<BDataType>::name,
                            DataTypeTraits<AccDataType>::name,
                            DataTypeTraits<CDataType>::name,
                            ALayout::name,
                            BLayout::name,
                            CLayout::name};

        KernelInstance kernel_instance{environment_, description, problem, {-1.0f, -1.0f, -1.0f}};

        auto launch_kernel = [&] {
            float avg_time = Kernel::launch(args, s);
            c_m_n_dev_buf.FromDevice(c_m_n_dev_result.data());

            std::size_t flop     = std::size_t(2) * args.M * args.N * args.K;
            std::size_t num_byte = sizeof(ADataType) * args.M * args.K +
                                   sizeof(BDataType) * args.N * args.K +
                                   sizeof(CDataType) * args.M * args.N;
            float tflops     = static_cast<float>(flop) / 1.E9 / avg_time;
            float gb_per_sec = num_byte / 1.E6 / avg_time;

            kernel_instance.perf_result.latency   = avg_time;
            kernel_instance.perf_result.tflops    = tflops;
            kernel_instance.perf_result.bandwidth = gb_per_sec;

            std::cout << kernel_instance << std::endl;

            bool verified_correct =
                !verify || compare(args.K, args.k_batch, c_m_n_dev_result, c_m_n_host_result);

            if(verified_correct)
            {
                kernel_instances_.emplace_back(kernel_instance);
            }
            else
            {
                std::cout << "Verification failed, skip kernel: " << description << std::endl;
            }

            c_m_n_dev_buf.SetZero();
            c_m_n_dev_result.SetZero();
        };

        if(enable_profile_cache_)
        {
            if(!cache_db_->check_if_record(kernel_instance))
            {

                launch_kernel();
                cache_db_->insert_batch({kernel_instance});
            }
            else
            {
                auto perf_result = cache_db_->query_performance_result(kernel_instance);
                kernel_instance.perf_result.latency   = perf_result.latency;
                kernel_instance.perf_result.tflops    = perf_result.tflops;
                kernel_instance.perf_result.bandwidth = perf_result.bandwidth;
                std::cout << "Skip this kernel for " << description
                          << ", Because it has already been recorded in the cache database"
                          << std::endl;
                kernel_instances_.emplace_back(kernel_instance);
            }
        }
        else
        {
            launch_kernel();
        }
    }

    void export_perf_to_csv(const std::vector<KernelInstance>& instances,
                            const std::string& filename)
    {

        std::ostringstream buffer;

        buffer << "ROCmVersion,CommitID,DeviceName,KernelName,SplitK,M,N,K,"
               << "StrideA,StrideB,StrideC,ADataType,BDataType,AccDataType,"
               << "CDataType,ALayout,BLayout,CLayout,Latency(ms),TFLOPS,Bandwidth\n";

        for(const auto& instance : instances)
        {
            const auto& env  = instance.env;
            const auto& p    = instance.problem;
            const auto& perf = instance.perf_result;

            std::string sanitized_name = instance.name;
            std::replace(sanitized_name.begin(), sanitized_name.end(), '\"', '\'');

            buffer << env.rocm_version << "," << env.commit_id << "," << env.device_name << ","
                   << "\"" << sanitized_name << "\"," << p.split_k << "," << p.m << "," << p.n
                   << "," << p.k << "," << p.stride_a << "," << p.stride_b << "," << p.stride_c
                   << "," << p.dtype_a << "," << p.dtype_b << "," << p.dtype_acc << "," << p.dtype_c
                   << "," << p.layout_a << "," << p.layout_b << "," << p.layout_c << ","
                   << std::fixed << std::setprecision(6) << perf.latency << "," << std::scientific
                   << perf.tflops << "," << std::fixed << perf.bandwidth << "\n";
        }

        std::ofstream csv_file(filename, std::ios::trunc);
        if(!csv_file)
        {
            throw std::runtime_error("Failed to open CSV file: " + filename);
        }
        csv_file << buffer.str();
        csv_file.close();

        if(csv_file.fail())
        {
            throw std::runtime_error("Incomplete write to CSV file: " + filename);
        }
    }

    KernelInstance select_best_instance(Metric metric, const std::string& csv_path = "")
    {
        if(kernel_instances_.empty())
            throw std::runtime_error("Empty instances");

        auto kernel_instance = *std::max_element(kernel_instances_.begin(),
                                                 kernel_instances_.end(),
                                                 [metric](const auto& a, const auto& b) {
                                                     return PerformanceResult::compare(
                                                         b.perf_result, a.perf_result, metric);
                                                 });

        std::cout << "According to given metrics: " << get_metric_name(metric) << "\n"
                  << "The best kernel instance is: " << kernel_instance << std::endl;

        if(!csv_path.empty())
        {
            try
            {
                export_perf_to_csv(kernel_instances_, csv_path);
            }
            catch(const std::exception& e)
            {
                std::cerr << "CSV export failed: " << e.what() << std::endl;
            }
        }

        return kernel_instance;
    }

    private:
    Executor(bool enable_profile_cache = true, bool flush_profile_cache = false)
        : enable_profile_cache_(enable_profile_cache), flush_profile_cache_(flush_profile_cache)
    {
        environment_ = Environment{
            get_rocm_version(),
            "89f",
            ck::get_device_name(),
        };
        std::cout << "Init gemm bechmark on device: " << environment_.device_name << std::endl;

        initialize_profile_cache();
    }

    void initialize_profile_cache()
    {
        // Init cache if enable profile cache
        if(enable_profile_cache_)
        {
            // get profile cache path
            std::filesystem::path cache_db_prefix_path =
                std::filesystem::current_path() / ".tile_engine";
            if(!create_cache_directory(cache_db_prefix_path))
            {
                std::cerr << "Error: Failed to create cache directory" << std::endl;
                return;
            }
            std::filesystem::path cache_db_path =
                cache_db_prefix_path / ("tile_engine_" + environment_.device_name + ".db");

            // remove cache if flush_profile_cache
            handle_cache_flush(cache_db_path);

            // load profile cache
            initialize_cache_db(cache_db_path);
        }
        else
        {
            std::cout << "Executor disable profile cache! " << std::endl;
        }
    }

    bool create_cache_directory(const std::filesystem::path& cache_db_prefix_path)
    {
        std::error_code ec;
        bool created = std::filesystem::create_directories(cache_db_prefix_path, ec);

        if(ec)
        {
            std::cerr << "Error creating directory " << cache_db_prefix_path << ": " << ec.message()
                      << std::endl;
            return false;
        }

        if(created)
        {
            std::cout << "Created cache directory: " << cache_db_prefix_path << std::endl;
        }
        else
        {
            std::cout << "Using existing cache directory: " << cache_db_prefix_path << std::endl;
        }
        return true;
    }

    void handle_cache_flush(const std::filesystem::path& cache_db_path) const
    {
        if(flush_profile_cache_ && std::filesystem::exists(cache_db_path))
        {
            std::error_code ec;
            if(std::filesystem::remove(cache_db_path, ec))
            {
                std::cout << "Successfully flushed cache: " << cache_db_path << std::endl;
            }
            else
            {
                std::cerr << "Error flushing cache: " << ec.message() << std::endl;
            }
        }
    }

    void initialize_cache_db(const std::filesystem::path& path)
    {
        try
        {
            cache_db_ = std::make_unique<ProfileCacheDB>(path);
            std::cout << "Loaded profile cache from " << path << std::endl;
        }
        catch(const std::exception& e)
        {
            std::cerr << "Failed to initialize profile cache: " << e.what() << std::endl;
        }
    }

    Executor(const Executor&)            = delete;
    Executor& operator=(const Executor&) = delete;

    Environment environment_;

    bool enable_profile_cache_;
    bool flush_profile_cache_;

    std::unique_ptr<ProfileCacheDB> cache_db_;
    std::vector<KernelInstance> kernel_instances_;
};
