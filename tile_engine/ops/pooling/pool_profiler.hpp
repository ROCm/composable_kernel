// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <iostream>
#include <fstream>
#include <iomanip>

#include "ck_tile/host/device_prop.hpp"
#include "ck_tile/ops/pooling.hpp"
#include "ck_tile/host/reference/reference_pool.hpp"
#include "pool_benchmark.hpp"

class PoolProfiler
{
    public:
    static PoolProfiler& instance(Setting setting)
    {
        static PoolProfiler instance{setting};
        return instance;
    }

    // Overload for single kernel benchmarking
    template <typename TensorShape, typename WindowShape>
    void benchmark(PoolProblem& pool_problem,
                   std::function<float(const ck_tile::PoolHostArgs<TensorShape, WindowShape>&,
                                       const ck_tile::stream_config&)> kernel_func)
    {
        // Create a vector with a single callable that returns both name and time
        std::vector<std::function<std::tuple<std::string, float>(
            ck_tile::PoolHostArgs<TensorShape, WindowShape>&, const ck_tile::stream_config&)>>
            callables;

        callables.push_back([kernel_func](ck_tile::PoolHostArgs<TensorShape, WindowShape>& args,
                                          const ck_tile::stream_config& stream) {
            float time = kernel_func(args, stream);
            return std::make_tuple(std::string(KERNEL_NAME), time);
        });

        benchmark(pool_problem, callables);
    }

    template <typename TensorShape, typename WindowShape>
    void benchmark(
        PoolProblem& pool_problem,
        std::vector<std::function<std::tuple<std::string, float>(
            ck_tile::PoolHostArgs<TensorShape, WindowShape>&, const ck_tile::stream_config&)>>&
            callables)
    {
        // Calculate output dimensions based on pool dimension
        const ck_tile::index_t N = pool_problem.N;
        const ck_tile::index_t D = pool_problem.D;
        const ck_tile::index_t H = pool_problem.H;
        const ck_tile::index_t W = pool_problem.W;
        const ck_tile::index_t C = pool_problem.C;

        const ck_tile::index_t Z = pool_problem.windowZ;
        const ck_tile::index_t Y = pool_problem.windowY;
        const ck_tile::index_t X = pool_problem.windowX;

        const ck_tile::index_t Sz = pool_problem.strideZ;
        const ck_tile::index_t Sy = pool_problem.strideY;
        const ck_tile::index_t Sx = pool_problem.strideX;

        const ck_tile::index_t Dz = pool_problem.dilationZ;
        const ck_tile::index_t Dy = pool_problem.dilationY;
        const ck_tile::index_t Dx = pool_problem.dilationX;

        const ck_tile::index_t LeftPz  = pool_problem.leftPadZ;
        const ck_tile::index_t LeftPy  = pool_problem.leftPadY;
        const ck_tile::index_t LeftPx  = pool_problem.leftPadX;
        const ck_tile::index_t RightPz = pool_problem.rightPadZ;
        const ck_tile::index_t RightPy = pool_problem.rightPadY;
        const ck_tile::index_t RightPx = pool_problem.rightPadX;

        // Calculate effective window sizes
        const ck_tile::index_t Zs = (Z - 1) * Dz + 1;
        const ck_tile::index_t Ys = (Y - 1) * Dy + 1;
        const ck_tile::index_t Xs = (X - 1) * Dx + 1;

        // Calculate output dimensions
        const ck_tile::index_t Do = (D + LeftPz + RightPz - Zs) / Sz + 1;
        const ck_tile::index_t Ho = (H + LeftPy + RightPy - Ys) / Sy + 1;
        const ck_tile::index_t Wo = (W + LeftPx + RightPx - Xs) / Sx + 1;

        // Create input/output tensors based on pool dimension (3D: NDHWC, 2D: NHWC)
        ck_tile::HostTensor<InDataType> in_tensor(
            pool_problem.poolDim == 3
                ? std::vector<std::size_t>{static_cast<std::size_t>(N),
                                           static_cast<std::size_t>(D),
                                           static_cast<std::size_t>(H),
                                           static_cast<std::size_t>(W),
                                           static_cast<std::size_t>(C)}
                : std::vector<std::size_t>{static_cast<std::size_t>(N),
                                           static_cast<std::size_t>(H),
                                           static_cast<std::size_t>(W),
                                           static_cast<std::size_t>(C)});

        ck_tile::HostTensor<OutDataType> out_tensor(
            pool_problem.poolDim == 3
                ? std::vector<std::size_t>{static_cast<std::size_t>(N),
                                           static_cast<std::size_t>(Do),
                                           static_cast<std::size_t>(Ho),
                                           static_cast<std::size_t>(Wo),
                                           static_cast<std::size_t>(C)}
                : std::vector<std::size_t>{static_cast<std::size_t>(N),
                                           static_cast<std::size_t>(Ho),
                                           static_cast<std::size_t>(Wo),
                                           static_cast<std::size_t>(C)});

        ck_tile::HostTensor<OutDataType> out_host_result(
            pool_problem.poolDim == 3
                ? std::vector<std::size_t>{static_cast<std::size_t>(N),
                                           static_cast<std::size_t>(Do),
                                           static_cast<std::size_t>(Ho),
                                           static_cast<std::size_t>(Wo),
                                           static_cast<std::size_t>(C)}
                : std::vector<std::size_t>{static_cast<std::size_t>(N),
                                           static_cast<std::size_t>(Ho),
                                           static_cast<std::size_t>(Wo),
                                           static_cast<std::size_t>(C)});

        ck_tile::HostTensor<IndexDataType> out_index_tensor(
            pool_problem.outputIndex
                ? (pool_problem.poolDim == 3
                       ? std::vector<std::size_t>{static_cast<std::size_t>(N),
                                                  static_cast<std::size_t>(Do),
                                                  static_cast<std::size_t>(Ho),
                                                  static_cast<std::size_t>(Wo),
                                                  static_cast<std::size_t>(C)}
                       : std::vector<std::size_t>{static_cast<std::size_t>(N),
                                                  static_cast<std::size_t>(Ho),
                                                  static_cast<std::size_t>(Wo),
                                                  static_cast<std::size_t>(C)})
                : std::vector<std::size_t>{1});

        ck_tile::HostTensor<IndexDataType> out_index_host_result(
            pool_problem.outputIndex
                ? (pool_problem.poolDim == 3
                       ? std::vector<std::size_t>{static_cast<std::size_t>(N),
                                                  static_cast<std::size_t>(Do),
                                                  static_cast<std::size_t>(Ho),
                                                  static_cast<std::size_t>(Wo),
                                                  static_cast<std::size_t>(C)}
                       : std::vector<std::size_t>{static_cast<std::size_t>(N),
                                                  static_cast<std::size_t>(Ho),
                                                  static_cast<std::size_t>(Wo),
                                                  static_cast<std::size_t>(C)})
                : std::vector<std::size_t>{1});

        // Initialize input tensor
        if(setting_.init_method_ == 0)
        {
            ck_tile::FillUniformDistribution<InDataType>{-5.f, 5.f}(in_tensor);
        }
        else if(setting_.init_method_ == 1)
        {
            ck_tile::FillMonotonicSeq<InDataType>{}(in_tensor);
        }
        else if(setting_.init_method_ == 2)
        {
            ck_tile::FillConstant<InDataType>{static_cast<InDataType>(1)}(in_tensor);
        }
        else
        {
            in_tensor.SetZero();
        }

        // Allocate device memory
        ck_tile::DeviceMem in_dev_buf(in_tensor.get_element_space_size_in_bytes());
        ck_tile::DeviceMem out_dev_buf(out_tensor.get_element_space_size_in_bytes());
        ck_tile::DeviceMem out_index_dev_buf(
            pool_problem.outputIndex ? out_index_tensor.get_element_space_size_in_bytes() : 0);

        in_dev_buf.ToDevice(in_tensor.data());
        out_dev_buf.SetZero();
        if(pool_problem.outputIndex)
        {
            out_index_dev_buf.SetZero();
        }

        // Create shapes for host args
        TensorShape input_shape, output_shape, input_strides, output_strides;
        WindowShape window_lengths, window_strides, window_dilations, input_left_pads,
            input_right_pads;

        // Create host arguments
        ck_tile::PoolHostArgs<TensorShape, WindowShape> pool_args{
            in_dev_buf.GetDeviceBuffer(),
            out_dev_buf.GetDeviceBuffer(),
            pool_problem.outputIndex ? out_index_dev_buf.GetDeviceBuffer() : nullptr,
            input_shape,
            output_shape,
            input_strides,
            output_strides,
            window_lengths,
            window_strides,
            window_dilations,
            input_left_pads,
            input_right_pads};

        // Run reference if verification is enabled
        // (Reference computation would be added here based on pool dimension)

        for(auto& callable : callables)
        {
            auto kernel_run_result = callable(pool_args,
                                              ck_tile::stream_config{nullptr,
                                                                     true,
                                                                     setting_.log_,
                                                                     setting_.n_warmup_,
                                                                     setting_.n_repeat_,
                                                                     setting_.is_gpu_timer_,
                                                                     setting_.flush_cache_,
                                                                     setting_.rotating_count_});
            process_result(pool_problem,
                           out_dev_buf,
                           out_host_result,
                           out_tensor,
                           out_index_dev_buf,
                           out_index_host_result,
                           out_index_tensor,
                           kernel_run_result);
        }
    }

    void process_result(const PoolProblem& pool_problem,
                        ck_tile::DeviceMem& out_dev_buf,
                        ck_tile::HostTensor<OutDataType>& out_host_result,
                        ck_tile::HostTensor<OutDataType>& out_dev_result,
                        ck_tile::DeviceMem& out_index_dev_buf,
                        ck_tile::HostTensor<IndexDataType>& out_index_host_result,
                        ck_tile::HostTensor<IndexDataType>& out_index_dev_result,
                        const std::tuple<std::string, float>& kernel_run_result)
    {
        auto [name, avg_time] = kernel_run_result;

        KernelInstance kernel_instance{name, pool_problem, {-1.0f, -1.0f, -1.0f}};

        // Compute performance metrics
        const ck_tile::index_t N  = pool_problem.N;
        const ck_tile::index_t D  = pool_problem.D;
        const ck_tile::index_t H  = pool_problem.H;
        const ck_tile::index_t W  = pool_problem.W;
        const ck_tile::index_t C  = pool_problem.C;
        const ck_tile::index_t Z  = pool_problem.windowZ;
        const ck_tile::index_t Y  = pool_problem.windowY;
        const ck_tile::index_t X  = pool_problem.windowX;
        const ck_tile::index_t Sz = pool_problem.strideZ;
        const ck_tile::index_t Sy = pool_problem.strideY;
        const ck_tile::index_t Sx = pool_problem.strideX;
        const ck_tile::index_t Dz = pool_problem.dilationZ;
        const ck_tile::index_t Dy = pool_problem.dilationY;
        const ck_tile::index_t Dx = pool_problem.dilationX;

        const ck_tile::index_t Zs = (Z - 1) * Dz + 1;
        const ck_tile::index_t Ys = (Y - 1) * Dy + 1;
        const ck_tile::index_t Xs = (X - 1) * Dx + 1;

        const ck_tile::index_t Do =
            (D + pool_problem.leftPadZ + pool_problem.rightPadZ - Zs) / Sz + 1;
        const ck_tile::index_t Ho =
            (H + pool_problem.leftPadY + pool_problem.rightPadY - Ys) / Sy + 1;
        const ck_tile::index_t Wo =
            (W + pool_problem.leftPadX + pool_problem.rightPadX - Xs) / Sx + 1;

        // Calculate FLOPs: for pooling, we count one compare/add per window element per output
        // element
        std::size_t window_size =
            static_cast<std::size_t>(Z) * static_cast<std::size_t>(Y) * static_cast<std::size_t>(X);
        std::size_t output_elements = static_cast<std::size_t>(N) * static_cast<std::size_t>(Do) *
                                      static_cast<std::size_t>(Ho) * static_cast<std::size_t>(Wo) *
                                      static_cast<std::size_t>(C);
        std::size_t flop = output_elements * window_size;

        // Calculate memory bandwidth
        std::size_t num_byte = sizeof(InDataType) * N * D * H * W * C +
                               sizeof(OutDataType) * N * Do * Ho * Wo * C;

        // Update performance results
        kernel_instance.perf_result_.latency_   = avg_time;
        kernel_instance.perf_result_.tflops_    = static_cast<float>(flop) / 1.E9 / avg_time;
        kernel_instance.perf_result_.bandwidth_ = num_byte / 1.E6 / avg_time;

        if(setting_.log_ > 0 && !setting_.json_output_)
        {
            std::cout << kernel_instance << std::endl;
        }

        // Verify result
        out_dev_buf.FromDevice(out_dev_result.data());

        bool verified_correct = true;
        if(setting_.verify_)
        {
            verified_correct = compare_pool_results(name, out_dev_result, out_host_result);
            if(pool_problem.outputIndex)
            {
                out_index_dev_buf.FromDevice(out_index_dev_result.data());
                verified_correct =
                    verified_correct &&
                    compare_pool_index_results(name, out_index_dev_result, out_index_host_result);
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

        // Clear tensors
        out_dev_buf.SetZero();
        out_dev_result.SetZero();
    }

    KernelInstance select_best_instance(Metric metric)
    {
        if(kernel_instances_.empty())
            throw std::runtime_error("Empty instances");

        auto kernel_instance = *std::max_element(kernel_instances_.begin(),
                                                 kernel_instances_.end(),
                                                 [metric](const auto& a, const auto& b) {
                                                     return PerformanceResult::compare(
                                                         b.perf_result_, a.perf_result_, metric);
                                                 });

        if(setting_.json_output_)
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

        if(!setting_.csv_filename_.empty())
        {
            std::ofstream file(setting_.csv_filename_ + ".csv", std::ios::app);

            if(!file.is_open())
            {
                std::cerr << "Warning: Failed to open CSV file for writing." << std::endl;
            }
            else
            {
                if(file.tellp() == 0)
                {
                    file << "rocm_version,device_name,"
                         << "in_dtype,out_dtype,compute_dtype,index_dtype,"
                         << "block_shape,reduce_op,pool_dim,"
                         << "N,D,H,W,C,"
                         << "window_z,window_y,window_x,"
                         << "stride_z,stride_y,stride_x,"
                         << "dilation_z,dilation_y,dilation_x,"
                         << "left_pad_z,left_pad_y,left_pad_x,"
                         << "right_pad_z,right_pad_y,right_pad_x,"
                         << "output_index,propagate_nan," << "name,"
                         << "latency(ms),tflops(TFlops),bandwidth(GB/s),metric\n";
                }

                const auto& problem = kernel_instance.problem_;
                const auto& name    = kernel_instance.name_;
                const auto& perf    = kernel_instance.perf_result_;

                file << get_rocm_version() << "," << ck_tile::get_device_name() << ","
                     << problem.inDType << "," << problem.outDType << "," << problem.computeDType
                     << "," << problem.indexDType << "," << problem.blockShape << ","
                     << problem.reduceOp << "," << problem.poolDim << "," << problem.N << ","
                     << problem.D << "," << problem.H << "," << problem.W << "," << problem.C << ","
                     << problem.windowZ << "," << problem.windowY << "," << problem.windowX << ","
                     << problem.strideZ << "," << problem.strideY << "," << problem.strideX << ","
                     << problem.dilationZ << "," << problem.dilationY << "," << problem.dilationX
                     << "," << problem.leftPadZ << "," << problem.leftPadY << ","
                     << problem.leftPadX << "," << problem.rightPadZ << "," << problem.rightPadY
                     << "," << problem.rightPadX << "," << problem.outputIndex << ","
                     << problem.propagateNan << "," << name << "," << std::fixed
                     << std::setprecision(4) << perf.latency_ << "," << std::fixed
                     << std::setprecision(4) << perf.tflops_ << "," << std::fixed
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

    PoolProfiler(const PoolProfiler&)            = delete;
    PoolProfiler& operator=(const PoolProfiler&) = delete;

    private:
    ~PoolProfiler() { kernel_instances_.clear(); }
    PoolProfiler(Setting setting) : setting_(setting) {}

    Setting setting_;

    std::vector<KernelInstance> kernel_instances_;
};
