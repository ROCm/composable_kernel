// SPDX-License-Identifier: MIT
// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.

#include <iostream>
#include <iomanip>
#include <string>
#include <vector>
#include <random>
#include <limits>
#include <cmath>
#include <fstream>
#include <sstream>
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <hip/hip_bfloat16.h>

#include "ck_tile/core.hpp"
#include "ck_tile/host.hpp"
#include "gemm_common.hpp"

// The kernel header is included via the compile command line with -include flag
// It defines SelectedKernel struct and KERNEL_NAME

// Helper function to determine if a layout is row-major
template <typename Layout>
constexpr auto is_row_major(Layout)
{
    return ck_tile::bool_constant<std::is_same_v<Layout, ck_tile::tensor_layout::gemm::RowMajor>>{};
}

#define HIP_CHECK(cmd)                                                                          \
    do                                                                                          \
    {                                                                                           \
        hipError_t error = (cmd);                                                               \
        if(error != hipSuccess)                                                                 \
        {                                                                                       \
            std::cerr << "HIP error: " << hipGetErrorString(error) << " at " << __FILE__ << ":" \
                      << __LINE__ << std::endl;                                                 \
            exit(EXIT_FAILURE);                                                                 \
        }                                                                                       \
    } while(0)

// DataTypeTraits for all supported types
template <typename T>
struct DataTypeTraits;

template <>
struct DataTypeTraits<float>
{
    static constexpr const char* name = "fp32";
};

template <>
struct DataTypeTraits<double>
{
    static constexpr const char* name = "fp64";
};

template <>
struct DataTypeTraits<ck_tile::half_t>
{
    static constexpr const char* name = "fp16";
};

template <>
struct DataTypeTraits<ck_tile::bf16_t>
{
    static constexpr const char* name = "bf16";
};

template <>
struct DataTypeTraits<ck_tile::fp8_t>
{
    static constexpr const char* name = "fp8";
};

template <>
struct DataTypeTraits<ck_tile::bf8_t>
{
    static constexpr const char* name = "bf8";
};

template <>
struct DataTypeTraits<ck_tile::int8_t>
{
    static constexpr const char* name = "int8";
};

template <>
struct DataTypeTraits<ck_tile::int32_t>
{
    static constexpr const char* name = "int32";
};

template <>
struct DataTypeTraits<ck_tile::pk_int4_t>
{
    static constexpr const char* name = "pk_int4_t";
};

// Permutation function for pk_int4_t
template <typename Tensor>
void permute_vectors_i4x4_b(Tensor& tensor)
{
    const ck_tile::index_t K = tensor.get_length(0);
    const ck_tile::index_t N = tensor.get_length(1);
    // vector pk_i4x4 permute
    for(int i = 0; i < N; i++)
    {
        for(int j = 0; j < K; j += 8)
        {
            int8_t input[8];

            for(int k = 0; k < 4; k++)
            {
                int8_t i4x2      = tensor(j + k * 2, i).data;
                input[k * 2 + 0] = (i4x2 >> 4) & 0xf;
                input[k * 2 + 1] = (i4x2 >> 0) & 0xf;
            }

            // permute 01234567->20643175
            {
                int8_t hi        = input[2];
                int8_t lo        = input[0];
                int8_t i4x2      = (hi << 4) | lo;
                tensor(j + 0, i) = i4x2;
            }

            {
                int8_t hi        = input[6];
                int8_t lo        = input[4];
                int8_t i4x2      = (hi << 4) | lo;
                tensor(j + 2, i) = i4x2;
            }

            {
                int8_t hi        = input[3];
                int8_t lo        = input[1];
                int8_t i4x2      = (hi << 4) | lo;
                tensor(j + 4, i) = i4x2;
            }

            {
                int8_t hi        = input[7];
                int8_t lo        = input[5];
                int8_t i4x2      = (hi << 4) | lo;
                tensor(j + 6, i) = i4x2;
            }
        }
    }
}

// Initialize tensor based on method
template <typename T>
void initialize_tensor(std::vector<T>& tensor, int init_method, int size)
{
    if(init_method == 0) // random
    {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<float> dis(-1.0f, 1.0f);

        for(int i = 0; i < size; i++)
        {
            tensor[i] = static_cast<T>(dis(gen));
        }
    }
    else if(init_method == 1) // linear
    {
        for(int i = 0; i < size; i++)
        {
            tensor[i] = static_cast<T>(i % 17);
        }
    }
    else if(init_method == 2) // constant
    {
        for(int i = 0; i < size; i++)
        {
            tensor[i] = static_cast<T>(1.0f);
        }
    }
}

// Reference implementation wrapper
template <typename AType, typename BType, typename CType>
void gemm_host_reference(int verify,
                         const std::vector<AType>& h_a,
                         const std::vector<BType>& h_b,
                         std::vector<CType>& h_c_ref,
                         void* d_a,
                         void* d_b,
                         int m,
                         int n,
                         int k,
                         int stride_a,
                         int stride_b,
                         int stride_c)
{
    if(verify == 1)
    { // CPU verification
        // Create HostTensor objects with proper layout descriptors
        // Use host_tensor_descriptor to handle layout-specific strides
        ck_tile::HostTensor<AType> a_m_k(
            ck_tile::host_tensor_descriptor(m, k, stride_a, is_row_major(ALayout{})));
        ck_tile::HostTensor<BType> b_k_n(
            ck_tile::host_tensor_descriptor(k, n, stride_b, is_row_major(BLayout{})));
        ck_tile::HostTensor<CType> c_m_n(
            ck_tile::host_tensor_descriptor(m, n, stride_c, is_row_major(CLayout{})));

        // Copy data to tensors
        std::copy(h_a.begin(), h_a.end(), a_m_k.mData.begin());
        std::copy(h_b.begin(), h_b.end(), b_k_n.mData.begin());

        // Use ck_tile reference implementation
        c_m_n.SetZero();
        ck_tile::reference_gemm<AType, BType, float, CType>(a_m_k, b_k_n, c_m_n);

        // Copy result back
        std::copy(c_m_n.mData.begin(), c_m_n.mData.end(), h_c_ref.begin());
    }
    else if(verify == 2)
    { // GPU verification
        void* d_c_ref;
        size_t size_c = m * stride_c * sizeof(CType);
        HIP_CHECK(hipMalloc(&d_c_ref, size_c));
        HIP_CHECK(hipMemset(d_c_ref, 0, size_c));

        // Use ck_tile GPU reference with correct layouts matching the kernel
        // The kernel uses Row-Column-Row (RCR) layout
        ck_tile::reference_gemm_gpu<AType, BType, float, CType, ALayout, BLayout, CLayout>(
            static_cast<AType*>(d_a),
            static_cast<BType*>(d_b),
            static_cast<CType*>(d_c_ref),
            m,
            n,
            k,
            stride_a,
            stride_b,
            stride_c);

        HIP_CHECK(hipMemcpy(h_c_ref.data(), d_c_ref, size_c, hipMemcpyDeviceToHost));
        HIP_CHECK(hipFree(d_c_ref));
    }
}

// Calculate relative and absolute tolerances for verification
template <typename AType, typename BType, typename AccType, typename CType>
auto calculate_rtol_atol(const int K, const int kbatch, const float max_accumulated_value)
{
    using ComputeType = std::conditional_t<sizeof(AType) < sizeof(BType), AType, BType>;

    // Calculate thresholds
    const auto rtol = ck_tile::get_relative_threshold<ComputeType, CType, AccType>(
        ck_tile::integer_divide_ceil(K, kbatch));
    const auto atol = ck_tile::get_absolute_threshold<ComputeType, CType, AccType>(
        max_accumulated_value / kbatch, ck_tile::integer_divide_ceil(K, kbatch));

    // Calculate error due to split_k accumulation
    const auto rtol_split_k = ck_tile::get_relative_threshold<CType, CType, CType>(kbatch);
    const auto atol_split_k =
        ck_tile::get_absolute_threshold<CType, CType, CType>(max_accumulated_value, kbatch);

    // Use higher threshold
    return std::make_pair(std::max(rtol, rtol_split_k), std::max(atol, atol_split_k));
}

// Verification function using ck_tile utilities
template <typename T>
bool verify_results(const std::vector<T>& device_result,
                    const std::vector<T>& host_result,
                    int m,
                    int n,
                    int k,
                    int split_k,
                    const std::string& kernel_name)
{
    // Create HostTensor objects and copy data
    ck_tile::HostTensor<T> device_tensor({m, n});
    ck_tile::HostTensor<T> host_tensor({m, n});

    // Copy data to tensors
    std::copy(device_result.begin(), device_result.end(), device_tensor.mData.begin());
    std::copy(host_result.begin(), host_result.end(), host_tensor.mData.begin());

    // Find max accumulated value
    float max_accumulated_value = 0.0f;
    for(const auto& val : host_result)
    {
        max_accumulated_value = std::max(max_accumulated_value, std::abs(static_cast<float>(val)));
    }

    // Calculate tolerances
    auto [rtol, atol] = calculate_rtol_atol<ck_tile::half_t, ck_tile::half_t, float, T>(
        k, split_k, max_accumulated_value);

    // Use ck_tile verification
    bool pass =
        ck_tile::check_err(device_tensor, host_tensor, "Error: Incorrect results!", rtol, atol);

    std::cout << "For " << kernel_name << " Relative error threshold is " << rtol
              << " Absolute error threshold is " << atol << std::endl;
    std::cout << "The verification result is: " << (pass ? "correct" : "fail") << std::endl;

    return pass;
}

// Flush cache function
void flush_cache()
{
    const size_t cache_size = 128 * 1024 * 1024; // 128MB
    std::vector<char> flush_buffer(cache_size);

    // Touch all cache lines
    for(size_t i = 0; i < cache_size; i += 64)
    {
        flush_buffer[i] = static_cast<char>(i);
    }
}

int main(int argc, char* argv[])
{
    // Create comprehensive ArgParser with all parameters
    ck_tile::ArgParser arg_parser;
    arg_parser.insert("m", "3840", "The value for m dimension. Default is 3840.")
        .insert("n", "4096", "The value for n dimension. Default is 4096.")
        .insert("k", "2048", "The value for k dimension. Default is 2048.")
        .insert("stride_a", "0", "The stride value for tensor A. Default is 0.")
        .insert("stride_b", "0", "The stride value for tensor B. Default is 0.")
        .insert("stride_c", "0", "The stride value for tensor C. Default is 0.")
        .insert("split_k", "1", "The split value for k dimension. Default is 1.")
        .insert("verify",
                "0",
                "The type of validation. Set to 0 for no validation, 1 for validation on CPU, or 2 "
                "for validation on GPU. Default is 0, no validation.")
        .insert("log",
                "false",
                "Whether output kernel instance information or not. Possible values are true or "
                "false. Default is false")
        .insert(
            "warmup", "50", "The number of iterations before benchmark the kernel. Default is 50.")
        .insert(
            "repeat", "100", "The number of iterations to benchmark the kernel. Default is 100.")
        .insert("timer",
                "true",
                "Whether if the timer is gpu timer or not. Possible values are false or true. "
                "Default is true.")
        .insert("init",
                "0",
                "The method of tensor initialization. Set to 0 for random, to 1 for linear, or 2 "
                "for constant(1). Default is 0, random.")
        .insert("flush_cache",
                "false",
                "To flush cache, possible values are true or false. "
                "Default is false.")
        .insert("rotating_count", "5", "number of iterations to rotate the cache. default is 5.")
        .insert("metric",
                "0",
                "Metric with which to measure kernel performance. Set to 0 for latency, 1 for "
                "tflops, or 2 for bandwidth. Default is 0, latency.")
        .insert("csv_filename",
                "",
                "The filename of benchmark result. Default is empty (no CSV output).")
        .insert("csv_format",
                "comprehensive",
                "CSV format: 'simple' or 'comprehensive'. Default is comprehensive.")
        .insert(
            "json_output", "false", "Output results in JSON format for parsing. Default is false.")
        .insert("structured_sparsity",
                "false",
                "Whether use sparsity kernel or not. Possible values are true or false. Default is "
                "false")
        .insert(
            "pipeline",
            "compv3",
            "The type of pipeline. Possible values are compv3, compv4 or mem. Default is compv3.")
        .insert("scheduler",
                "intrawave",
                "The type of scheduler. Possible values are intrawave or interwave. Default is "
                "intrawave.")
        .insert(
            "epilogue",
            "cshuffle",
            "The type of epilogue. Possible values are cshuffle or default. Default is cshuffle.")
        .insert("pad_m",
                "false",
                "Whether pad or not in m direction. Possible values are true or false. Default is "
                "false.")
        .insert("pad_n",
                "false",
                "Whether pad or not in n direction. Possible values are true or false. Default is "
                "false.")
        .insert("pad_k",
                "false",
                "Whether pad or not in k direction. Possible values are true or false. Default is "
                "false.")
        .insert("persistent", "false", "Whether to use persistent kernel. Default is false.");

    if(!arg_parser.parse(argc, argv))
    {
        return EXIT_FAILURE;
    }

    // Get all parameters
    int m                    = arg_parser.get_int("m");
    int n                    = arg_parser.get_int("n");
    int k                    = arg_parser.get_int("k");
    int stride_a_arg         = arg_parser.get_int("stride_a");
    int stride_b_arg         = arg_parser.get_int("stride_b");
    int stride_c_arg         = arg_parser.get_int("stride_c");
    int split_k              = arg_parser.get_int("split_k");
    int verify               = arg_parser.get_int("verify");
    bool log_info            = arg_parser.get_bool("log");
    int warmup               = arg_parser.get_int("warmup");
    int repeat               = arg_parser.get_int("repeat");
    bool use_gpu_timer       = arg_parser.get_bool("timer");
    int init_method          = arg_parser.get_int("init");
    bool flush_cache_flag    = arg_parser.get_bool("flush_cache");
    int rotating_count       = arg_parser.get_int("rotating_count");
    int metric               = arg_parser.get_int("metric");
    std::string csv_filename = arg_parser.get_str("csv_filename");
    std::string csv_format   = arg_parser.get_str("csv_format");
    bool json_output         = arg_parser.get_bool("json_output");

    // Calculate strides (0 means use default)
    int stride_a = (stride_a_arg == 0) ? k : stride_a_arg;
    int stride_b = (stride_b_arg == 0) ? n : stride_b_arg;
    int stride_c = (stride_c_arg == 0) ? n : stride_c_arg;

    // Allocate host memory
    size_t size_a = m * stride_a * sizeof(ck_tile::half_t);
    size_t size_b = k * stride_b * sizeof(ck_tile::half_t);
    size_t size_c = m * stride_c * sizeof(ck_tile::half_t);

    std::vector<ck_tile::half_t> h_a(m * stride_a);
    std::vector<ck_tile::half_t> h_b(k * stride_b);
    std::vector<ck_tile::half_t> h_c(m * stride_c);
    std::vector<ck_tile::half_t> h_c_ref(m * stride_c);

    // Initialize tensors
    initialize_tensor(h_a, init_method, m * stride_a);
    initialize_tensor(h_b, init_method, k * stride_b);
    std::fill(h_c.begin(), h_c.end(), ck_tile::half_t(0));

    // Allocate device memory
    void* d_a;
    void* d_b;
    void* d_c;

    HIP_CHECK(hipMalloc(&d_a, size_a));
    HIP_CHECK(hipMalloc(&d_b, size_b));
    HIP_CHECK(hipMalloc(&d_c, size_c));

    // Copy data to device
    HIP_CHECK(hipMemcpy(d_a, h_a.data(), size_a, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_b, h_b.data(), size_b, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemset(d_c, 0, size_c));

    // Create GemmHostArgs
    ck_tile::GemmHostArgs args = {
        d_a,      // a_ptr
        d_b,      // b_ptr
        d_c,      // c_ptr
        split_k,  // k_batch (split_k)
        m,        // M
        n,        // N
        k,        // K
        stride_a, // stride_A
        stride_b, // stride_B
        stride_c  // stride_C
    };

    // Create stream config
    ck_tile::stream_config stream{
        nullptr,          // stream
        true,             // time_kernel
        log_info ? 1 : 0, // log_level
        warmup,           // n_warmup
        repeat,           // n_repeat
        use_gpu_timer,    // use_gpu_timer
        flush_cache_flag, // flush_cache
        rotating_count    // rotating_count
    };

    float avg_time           = 0.0f;
    bool verification_passed = true;

    try
    {
        // Call the kernel's launch function directly
        avg_time = SelectedKernel::launch(args, stream);

        // Copy result back for verification if needed
        if(verify > 0)
        {
            HIP_CHECK(hipMemcpy(h_c.data(), d_c, size_c, hipMemcpyDeviceToHost));

            // Use unified reference implementation
            gemm_host_reference(
                verify, h_a, h_b, h_c_ref, d_a, d_b, m, n, k, stride_a, stride_b, stride_c);
            verification_passed = verify_results(h_c, h_c_ref, m, n, k, split_k, KERNEL_NAME);
        }

        // Calculate performance metrics
        size_t flop     = size_t(2) * m * n * k;
        size_t num_byte = sizeof(ck_tile::half_t) * (m * k + k * n + m * n);

        float tflops    = static_cast<float>(flop) / 1.E9 / avg_time;
        float bandwidth = num_byte / 1.E6 / avg_time;

        // Output results
        if(json_output)
        {
            // JSON format for Python parsing
            std::cout << "{" << std::endl;
            std::cout << "  \"kernel_name\": \"" << KERNEL_NAME << "\"," << std::endl;
            std::cout << "  \"m\": " << m << "," << std::endl;
            std::cout << "  \"n\": " << n << "," << std::endl;
            std::cout << "  \"k\": " << k << "," << std::endl;
            std::cout << "  \"split_k\": " << split_k << "," << std::endl;
            std::cout << "  \"time_ms\": " << avg_time << "," << std::endl;
            std::cout << "  \"tflops\": " << tflops << "," << std::endl;
            std::cout << "  \"bandwidth_gb_s\": " << bandwidth << "," << std::endl;
            std::cout << "  \"verification_passed\": " << (verification_passed ? "true" : "false")
                      << std::endl;
            std::cout << "}" << std::endl;
        }
        else
        {
            // Human-readable format
            std::cout << "Running kernel: " << KERNEL_NAME << std::endl;
            std::cout << "Problem size: M=" << m << ", N=" << n << ", K=" << k << std::endl;
            std::cout << "Split-K: " << split_k << std::endl;
            std::cout << "Strides: A=" << stride_a << ", B=" << stride_b << ", C=" << stride_c
                      << std::endl;

            if(metric == 0)
            { // latency
                std::cout << "Time: " << avg_time << " ms" << std::endl;
            }
            else if(metric == 1)
            { // tflops
                std::cout << "Performance: " << tflops << " TFLOPS" << std::endl;
            }
            else if(metric == 2)
            { // bandwidth
                std::cout << "Bandwidth: " << bandwidth << " GB/s" << std::endl;
            }

            if(verify > 0)
            {
                std::cout << "Verification: " << (verification_passed ? "PASSED" : "FAILED")
                          << std::endl;
            }
        }

        // CSV output if requested
        if(!csv_filename.empty())
        {
            std::ofstream csv_file(csv_filename, std::ios::app);
            if(csv_file.is_open())
            {
                // Write header if file is empty
                csv_file.seekp(0, std::ios::end);
                if(csv_file.tellp() == 0)
                {
                    if(csv_format == "comprehensive")
                    {
                        csv_file << "rocm_version,device_name,"
                                 << "split_k,m,n,k,stride_a,stride_b,stride_c,"
                                 << "dtype_a,dtype_b,dtype_acc,dtype_c,"
                                 << "layout_a,layout_b,layout_c," << "structured_sparsity,"
                                 << "name," << "latency(ms),tflops(TFlops),bandwidth(GB/s),metric,"
                                 << "warmup,repeat,flush_cache,rotating_count,"
                                 << "verification_mode,verification_passed\n";
                    }
                    else
                    {
                        // Simple format
                        csv_file << "kernel_name,m,n,k,split_k,stride_a,stride_b,stride_c,"
                                 << "time_ms,tflops,bandwidth_gb_s,verification_passed\n";
                    }
                }

                // Get ROCm version
                std::string rocm_version = "unknown";
#ifdef __HIP_PLATFORM_AMD__
                int major    = HIP_VERSION_MAJOR;
                int minor    = HIP_VERSION_MINOR;
                int patch    = HIP_VERSION_PATCH;
                rocm_version = std::to_string(major) + "." + std::to_string(minor) + "." +
                               std::to_string(patch);
#endif

                // Get device name
                hipDeviceProp_t props;
                HIP_CHECK(hipGetDeviceProperties(&props, 0));
                std::string device_name = props.name;

                // Get metric name
                std::string metric_name;
                switch(metric)
                {
                case 0: metric_name = "latency"; break;
                case 1: metric_name = "tflops"; break;
                case 2: metric_name = "bandwidth"; break;
                default: metric_name = "unknown"; break;
                }

                // Extract kernel traits from name
                KernelTraits traits = extract_traits_from_name(KERNEL_NAME);

                // Get structured sparsity from arg parser
                bool structured_sparsity = arg_parser.get_bool("structured_sparsity");

                if(csv_format == "comprehensive")
                {
                    // For this single kernel benchmark, we assume fp16 and rcr layout
                    // In a real implementation, these would be template parameters
                    std::string dtype_str  = "fp16";
                    std::string layout_str = "rcr";

                    csv_file << rocm_version << "," << device_name << "," << split_k << "," << m
                             << "," << n << "," << k << "," << stride_a << "," << stride_b << ","
                             << stride_c << "," << dtype_str << "," << dtype_str << "," << "fp32"
                             << "," << dtype_str << "," << "row" << "," << "col" << "," << "row"
                             << "," << (structured_sparsity ? "true" : "false") << ","
                             << KERNEL_NAME << "," << std::fixed << std::setprecision(4) << avg_time
                             << "," << std::fixed << std::setprecision(4) << tflops << ","
                             << std::fixed << std::setprecision(4) << bandwidth << ","
                             << metric_name << "," << warmup << "," << repeat << ","
                             << (flush_cache_flag ? "true" : "false") << "," << rotating_count
                             << "," << verify << "," << (verification_passed ? "true" : "false")
                             << "\n";
                }
                else
                {
                    // Simple format
                    csv_file << KERNEL_NAME << "," << m << "," << n << "," << k << "," << split_k
                             << "," << stride_a << "," << stride_b << "," << stride_c << ","
                             << std::fixed << std::setprecision(6) << avg_time << "," << std::fixed
                             << std::setprecision(2) << tflops << "," << std::fixed
                             << std::setprecision(2) << bandwidth << ","
                             << (verification_passed ? "true" : "false") << "\n";
                }

                if(!csv_file)
                {
                    std::cerr << "Warning: Error occurred while writing to CSV file." << std::endl;
                }
                csv_file.close();
            }
            else
            {
                std::cerr << "Warning: Failed to open CSV file for writing." << std::endl;
            }
        }
    }
    catch(const std::exception& e)
    {
        std::cerr << "Error: " << e.what() << std::endl;
        HIP_CHECK(hipFree(d_a));
        HIP_CHECK(hipFree(d_b));
        HIP_CHECK(hipFree(d_c));
        return EXIT_FAILURE;
    }

    // Cleanup
    HIP_CHECK(hipFree(d_a));
    HIP_CHECK(hipFree(d_b));
    HIP_CHECK(hipFree(d_c));

    return verification_passed ? 0 : 1;
}
