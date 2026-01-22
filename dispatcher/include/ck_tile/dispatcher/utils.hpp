// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

/**
 * @file utils.hpp
 * @brief Common utilities for CK Tile Dispatcher
 *
 * This header provides reusable utilities for:
 * - GPU memory management (GpuBuffer)
 * - Performance measurement (Timer, GpuTimer, BenchmarkStats)
 * - Validation (ValidationResult, validate_result)
 * - Kernel registration helpers
 * - Data generation (fill_random, etc.)
 *
 * Usage:
 *   #include "ck_tile/dispatcher/utils.hpp"
 *   using namespace ck_tile::dispatcher::utils;
 *
 *   // GPU memory
 *   GpuBuffer<half_t> buffer(1024);
 *
 *   // Timing
 *   GpuTimer timer;
 *   timer.start();
 *   // ... kernel ...
 *   timer.stop();
 *   float ms = timer.elapsed_ms();
 *
 *   // Validation
 *   auto result = validate_result(gpu_data, ref_data, size);
 */

#pragma once

#include <hip/hip_runtime.h>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <random>
#include <sstream>
#include <string>
#include <vector>
#include <algorithm>

#include "ck_tile/dispatcher/dispatcher.hpp"
#include "ck_tile/dispatcher/registry.hpp"
#include "ck_tile/dispatcher/backends/generated_tile_backend.hpp"

namespace ck_tile {
namespace dispatcher {
namespace utils {

// =============================================================================
// HIP Error Handling
// =============================================================================

#define CK_HIP_CHECK(call)                                                      \
    do                                                                          \
    {                                                                           \
        hipError_t err = call;                                                  \
        if(err != hipSuccess)                                                   \
        {                                                                       \
            std::cerr << "HIP error at " << __FILE__ << ":" << __LINE__ << ": " \
                      << hipGetErrorString(err) << std::endl;                   \
            return false;                                                       \
        }                                                                       \
    } while(0)

#define CK_HIP_CHECK_THROW(call)                                                           \
    do                                                                                     \
    {                                                                                      \
        hipError_t err = call;                                                             \
        if(err != hipSuccess)                                                              \
        {                                                                                  \
            throw std::runtime_error(std::string("HIP error: ") + hipGetErrorString(err)); \
        }                                                                                  \
    } while(0)

// =============================================================================
// Timing Utilities
// =============================================================================

/**
 * @brief High-resolution timer for CPU timing
 */
class Timer
{
    public:
    void start() { start_ = std::chrono::high_resolution_clock::now(); }

    double elapsed_ms() const
    {
        auto end = std::chrono::high_resolution_clock::now();
        return std::chrono::duration<double, std::milli>(end - start_).count();
    }

    private:
    std::chrono::high_resolution_clock::time_point start_;
};

/**
 * @brief GPU timing using HIP events
 *
 * Times kernel execution on a specific HIP stream. Events are recorded
 * on the provided stream to accurately measure kernel execution time.
 *
 * Usage:
 *   hipStream_t stream;
 *   hipStreamCreate(&stream);
 *   GpuTimer timer(stream);  // or timer.set_stream(stream)
 *   timer.start();
 *   kernel<<<grid, block, 0, stream>>>(...);
 *   timer.stop();
 *   float ms = timer.elapsed_ms();
 */
class GpuTimer
{
    public:
    /**
     * @brief Construct timer with optional stream
     * @param stream HIP stream to record events on (default: null stream)
     */
    explicit GpuTimer(hipStream_t stream = nullptr) : stream_(stream)
    {
        (void)hipEventCreate(&start_);
        (void)hipEventCreate(&stop_);
    }

    ~GpuTimer()
    {
        (void)hipEventDestroy(start_);
        (void)hipEventDestroy(stop_);
    }

    // Non-copyable
    GpuTimer(const GpuTimer&)            = delete;
    GpuTimer& operator=(const GpuTimer&) = delete;

    // Movable
    GpuTimer(GpuTimer&& other) noexcept
        : start_(other.start_), stop_(other.stop_), stream_(other.stream_)
    {
        other.start_  = nullptr;
        other.stop_   = nullptr;
        other.stream_ = nullptr;
    }

    GpuTimer& operator=(GpuTimer&& other) noexcept
    {
        if(this != &other)
        {
            if(start_)
                (void)hipEventDestroy(start_);
            if(stop_)
                (void)hipEventDestroy(stop_);
            start_        = other.start_;
            stop_         = other.stop_;
            stream_       = other.stream_;
            other.start_  = nullptr;
            other.stop_   = nullptr;
            other.stream_ = nullptr;
        }
        return *this;
    }

    /**
     * @brief Set the stream to record events on
     * @param stream HIP stream (pass nullptr for default stream)
     */
    void set_stream(hipStream_t stream) { stream_ = stream; }

    /**
     * @brief Get the current stream
     */
    hipStream_t get_stream() const { return stream_; }

    /**
     * @brief Record start event on the stream
     */
    void start() { (void)hipEventRecord(start_, stream_); }

    /**
     * @brief Record stop event on the stream
     */
    void stop() { (void)hipEventRecord(stop_, stream_); }

    /**
     * @brief Get elapsed time in milliseconds
     *
     * Synchronizes on the stop event before calculating time.
     * @return Elapsed time between start and stop in milliseconds
     */
    float elapsed_ms()
    {
        (void)hipEventSynchronize(stop_);
        float ms = 0;
        (void)hipEventElapsedTime(&ms, start_, stop_);
        return ms;
    }

    private:
    hipEvent_t start_   = nullptr;
    hipEvent_t stop_    = nullptr;
    hipStream_t stream_ = nullptr;
};

// =============================================================================
// Performance Metrics
// =============================================================================

/**
 * @brief Calculate TFLOPS for GEMM
 */
inline double calculate_tflops(int64_t M, int64_t N, int64_t K, double time_ms)
{
    double flops = 2.0 * M * N * K;
    return (flops / (time_ms * 1e-3)) / 1e12;
}

/**
 * @brief Calculate memory bandwidth in GB/s
 */
template <typename AType, typename BType, typename CType>
inline double calculate_bandwidth_gbs(int64_t M, int64_t N, int64_t K, double time_ms)
{
    double bytes = M * K * sizeof(AType) + K * N * sizeof(BType) + M * N * sizeof(CType);
    return (bytes / (time_ms * 1e-3)) / 1e9;
}

/**
 * @brief Benchmark statistics
 */
struct BenchmarkStats
{
    double min_ms        = 0;
    double avg_ms        = 0;
    double max_ms        = 0;
    double median_ms     = 0;
    double tflops        = 0;
    double bandwidth_gbs = 0;
    int iterations       = 0;

    void print(std::ostream& os = std::cout) const
    {
        os << std::fixed << std::setprecision(4);
        os << "  Min: " << min_ms << " ms\n";
        os << "  Avg: " << avg_ms << " ms\n";
        os << "  Max: " << max_ms << " ms\n";
        os << "  Median: " << median_ms << " ms\n";
        os << "  TFLOPS: " << std::setprecision(2) << tflops << "\n";
        os << "  Bandwidth: " << bandwidth_gbs << " GB/s\n";
    }
};

/**
 * @brief Run benchmark and compute statistics
 */
template <typename Func>
BenchmarkStats run_benchmark(Func&& func, int warmup = 2, int iterations = 10)
{
    std::vector<double> times;
    times.reserve(iterations);

    for(int i = 0; i < warmup; ++i)
        func();

    for(int i = 0; i < iterations; ++i)
        times.push_back(func());

    std::sort(times.begin(), times.end());

    BenchmarkStats stats;
    stats.iterations = iterations;
    stats.min_ms     = times.front();
    stats.max_ms     = times.back();
    stats.median_ms  = times[iterations / 2];

    double sum = 0;
    for(double t : times)
        sum += t;
    stats.avg_ms = sum / iterations;

    return stats;
}

// =============================================================================
// Validation Utilities
// =============================================================================

/**
 * @brief Validation result
 */
struct ValidationResult
{
    bool correct     = false;
    double max_diff  = 0;
    double mean_diff = 0;
    double accuracy  = 0;
    int64_t matches  = 0;
    int64_t total    = 0;

    void print(std::ostream& os = std::cout) const
    {
        os << "  Correct: " << (correct ? "YES" : "NO") << "\n";
        os << "  Max diff: " << max_diff << "\n";
        os << "  Mean diff: " << mean_diff << "\n";
        os << "  Accuracy: " << accuracy << "%\n";
        os << "  Matches: " << matches << "/" << total << "\n";
    }
};

/**
 * @brief Validate GEMM result against reference
 */
template <typename T>
ValidationResult validate_result(
    const T* result, const T* reference, int64_t size, double rtol = 1e-3, double atol = 1e-2)
{
    ValidationResult v;
    v.total    = size;
    v.max_diff = 0;
    v.matches  = 0;

    double sum_diff = 0;

    for(int64_t i = 0; i < size; ++i)
    {
        double r    = static_cast<double>(result[i]);
        double ref  = static_cast<double>(reference[i]);
        double diff = std::abs(r - ref);

        v.max_diff = std::max(v.max_diff, diff);
        sum_diff += diff;

        double threshold = atol + rtol * std::abs(ref);
        if(diff <= threshold)
            ++v.matches;
    }

    v.mean_diff = sum_diff / size;
    v.accuracy  = 100.0 * v.matches / v.total;
    v.correct   = (v.matches == v.total) || (v.accuracy >= 99.9);

    return v;
}

/**
 * @brief Compute reference GEMM on CPU
 */
template <typename AType, typename BType, typename CType>
void compute_reference_gemm(
    const AType* A, const BType* B, CType* C, int64_t M, int64_t N, int64_t K)
{
    for(int64_t m = 0; m < M; ++m)
    {
        for(int64_t n = 0; n < N; ++n)
        {
            double acc = 0;
            for(int64_t k = 0; k < K; ++k)
                acc += static_cast<double>(A[m * K + k]) * static_cast<double>(B[k * N + n]);
            C[m * N + n] = static_cast<CType>(acc);
        }
    }
}

// =============================================================================
// Data Generation
// =============================================================================

template <typename T>
void fill_random(T* data, int64_t size, T min_val = T(-1), T max_val = T(1))
{
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(static_cast<float>(min_val),
                                               static_cast<float>(max_val));
    for(int64_t i = 0; i < size; ++i)
        data[i] = static_cast<T>(dist(gen));
}

template <typename T>
void fill_zeros(T* data, int64_t size)
{
    std::fill(data, data + size, T(0));
}

template <typename T>
void fill_ones(T* data, int64_t size)
{
    std::fill(data, data + size, T(1));
}

template <typename T>
void fill_identity(T* data, int64_t rows, int64_t cols)
{
    fill_zeros(data, rows * cols);
    int64_t min_dim = std::min(rows, cols);
    for(int64_t i = 0; i < min_dim; ++i)
        data[i * cols + i] = T(1);
}

// =============================================================================
// GPU Memory Management
// =============================================================================

/**
 * @brief RAII wrapper for GPU memory
 */
template <typename T>
class GpuBuffer
{
    public:
    GpuBuffer() : data_(nullptr), size_(0) {}

    explicit GpuBuffer(int64_t count) : size_(count * sizeof(T))
    {
        CK_HIP_CHECK_THROW(hipMalloc(&data_, size_));
    }

    ~GpuBuffer()
    {
        if(data_)
            (void)hipFree(data_);
    }

    // Non-copyable
    GpuBuffer(const GpuBuffer&)            = delete;
    GpuBuffer& operator=(const GpuBuffer&) = delete;

    // Movable
    GpuBuffer(GpuBuffer&& other) noexcept : data_(other.data_), size_(other.size_)
    {
        other.data_ = nullptr;
        other.size_ = 0;
    }

    GpuBuffer& operator=(GpuBuffer&& other) noexcept
    {
        if(this != &other)
        {
            if(data_)
                (void)hipFree(data_);
            data_       = other.data_;
            size_       = other.size_;
            other.data_ = nullptr;
            other.size_ = 0;
        }
        return *this;
    }

    T* get() { return data_; }
    const T* get() const { return data_; }
    int64_t size_bytes() const { return size_; }
    int64_t count() const { return size_ / sizeof(T); }

    void copy_from_host(const T* host_data)
    {
        CK_HIP_CHECK_THROW(hipMemcpy(data_, host_data, size_, hipMemcpyHostToDevice));
    }

    void copy_to_host(T* host_data) const
    {
        CK_HIP_CHECK_THROW(hipMemcpy(host_data, data_, size_, hipMemcpyDeviceToHost));
    }

    void zero() { CK_HIP_CHECK_THROW(hipMemset(data_, 0, size_)); }

    private:
    T* data_;
    int64_t size_;
};

// =============================================================================
// Printing Utilities
// =============================================================================

inline void print_separator(char c = '=', int width = 70)
{
    std::cout << std::string(width, c) << "\n";
}

inline void print_header(const std::string& title)
{
    print_separator();
    std::cout << title << "\n";
    print_separator();
}

inline std::string format_size(int64_t M, int64_t N, int64_t K)
{
    std::ostringstream oss;
    oss << M << "x" << N << "x" << K;
    return oss.str();
}

inline std::string format_number(int64_t n)
{
    std::string s = std::to_string(n);
    int pos       = static_cast<int>(s.length()) - 3;
    while(pos > 0)
    {
        s.insert(pos, ",");
        pos -= 3;
    }
    return s;
}

/**
 * @brief Print all registered kernels in a registry
 *
 * @param registry The registry to list kernels from
 * @param os Output stream (default: std::cout)
 * @param verbose If true, show full kernel config details
 */
inline void print_registered_kernels(const Registry& registry,
                                     std::ostream& os = std::cout,
                                     bool verbose     = false)
{
    const auto& kernels = registry.get_all();
    os << "Registered Kernels (" << kernels.size() << "):\n";
    os << std::string(70, '-') << "\n";

    int idx = 1;
    for(const auto& kernel : kernels)
    {
        const auto& key = kernel->get_key();

        os << "  " << idx++ << ". " << kernel->get_name() << "\n";

        if(verbose)
        {
            os << "     Tile:      " << key.algorithm.tile_shape.m << "x"
               << key.algorithm.tile_shape.n << "x" << key.algorithm.tile_shape.k << "\n";
            os << "     Wave:      " << static_cast<int>(key.algorithm.wave_shape.m) << "x"
               << static_cast<int>(key.algorithm.wave_shape.n) << "x"
               << static_cast<int>(key.algorithm.wave_shape.k) << "\n";
            os << "     WarpTile:  " << static_cast<int>(key.algorithm.warp_tile_shape.m) << "x"
               << static_cast<int>(key.algorithm.warp_tile_shape.n) << "x"
               << static_cast<int>(key.algorithm.warp_tile_shape.k) << "\n";
            os << "     Pipeline:  " << to_string(key.algorithm.pipeline) << "\n";
            os << "     Scheduler: " << to_string(key.algorithm.scheduler) << "\n";
            os << "     Arch:      " << key.gfx_arch << "\n";
            os << "\n";
        }
    }

    if(!verbose && !kernels.empty())
    {
        os << "\n  Use --list-verbose for full details\n";
    }
    os << std::string(70, '-') << "\n";
}

/**
 * @brief Print a single kernel's configuration
 */
inline void print_kernel_info(const KernelInstance& kernel, std::ostream& os = std::cout)
{
    const auto& key = kernel.get_key();

    os << "Kernel: " << kernel.get_name() << "\n";
    os << "  Signature:\n";
    os << "    dtype:  " << to_string(key.signature.dtype_a) << "/"
       << to_string(key.signature.dtype_b) << "/" << to_string(key.signature.dtype_c) << "\n";
    os << "    layout: " << to_string(key.signature.layout_a) << to_string(key.signature.layout_b)
       << to_string(key.signature.layout_c) << "\n";

    os << "  Algorithm:\n";
    os << "    tile:      " << key.algorithm.tile_shape.m << "x" << key.algorithm.tile_shape.n
       << "x" << key.algorithm.tile_shape.k << "\n";
    os << "    wave:      " << static_cast<int>(key.algorithm.wave_shape.m) << "x"
       << static_cast<int>(key.algorithm.wave_shape.n) << "x"
       << static_cast<int>(key.algorithm.wave_shape.k) << "\n";
    os << "    warp_tile: " << static_cast<int>(key.algorithm.warp_tile_shape.m) << "x"
       << static_cast<int>(key.algorithm.warp_tile_shape.n) << "x"
       << static_cast<int>(key.algorithm.warp_tile_shape.k) << "\n";
    os << "    pipeline:  " << to_string(key.algorithm.pipeline) << "\n";
    os << "    scheduler: " << to_string(key.algorithm.scheduler) << "\n";
    os << "    epilogue:  " << to_string(key.algorithm.epilogue) << "\n";

    os << "  Target: " << key.gfx_arch << "\n";
}

// =============================================================================
// Kernel Key Builders
// =============================================================================

/**
 * @brief Build a KernelKey for FP16 Row-Col-Row layout GEMM
 *
 * This is the most common configuration. Customize parameters as needed.
 */
struct KernelKeyBuilder
{
    // Tile shape
    int tile_m = 128;
    int tile_n = 128;
    int tile_k = 32;

    // Wave shape (warps per block)
    int wave_m = 2;
    int wave_n = 2;
    int wave_k = 1;

    // Warp tile shape
    int warp_m = 32;
    int warp_n = 32;
    int warp_k = 16;

    // Block size
    int block_size = 256;

    // Data types
    DataType dtype_a   = DataType::FP16;
    DataType dtype_b   = DataType::FP16;
    DataType dtype_c   = DataType::FP16;
    DataType dtype_acc = DataType::FP32;

    // Layouts
    LayoutTag layout_a = LayoutTag::RowMajor;
    LayoutTag layout_b = LayoutTag::ColMajor;
    LayoutTag layout_c = LayoutTag::RowMajor;

    // Pipeline/scheduler
    Pipeline pipeline   = Pipeline::CompV4;
    Scheduler scheduler = Scheduler::Intrawave;
    Epilogue epilogue   = Epilogue::CShuffle;

    // Features
    bool preshuffle            = false;
    int num_d_tensors          = 0; // Multi-D: number of additional input tensors
    std::string elementwise_op = "PassThrough";

    // Target GPU
    std::string gfx_arch = "gfx942";

    /**
     * @brief Build the KernelKey
     */
    KernelKey build() const
    {
        KernelKey key;

        // Signature
        key.signature.dtype_a             = dtype_a;
        key.signature.dtype_b             = dtype_b;
        key.signature.dtype_c             = dtype_c;
        key.signature.dtype_acc           = dtype_acc;
        key.signature.layout_a            = layout_a;
        key.signature.layout_b            = layout_b;
        key.signature.layout_c            = layout_c;
        key.signature.transpose_a         = false;
        key.signature.transpose_b         = false;
        key.signature.grouped             = false;
        key.signature.split_k             = 1;
        key.signature.elementwise_op      = elementwise_op;
        key.signature.num_d_tensors       = num_d_tensors;
        key.signature.structured_sparsity = false;

        // Algorithm
        key.algorithm.tile_shape      = {static_cast<std::uint16_t>(tile_m),
                                         static_cast<std::uint16_t>(tile_n),
                                         static_cast<std::uint16_t>(tile_k)};
        key.algorithm.wave_shape      = {static_cast<std::uint8_t>(wave_m),
                                         static_cast<std::uint8_t>(wave_n),
                                         static_cast<std::uint8_t>(wave_k)};
        key.algorithm.warp_tile_shape = {static_cast<std::uint8_t>(warp_m),
                                         static_cast<std::uint8_t>(warp_n),
                                         static_cast<std::uint8_t>(warp_k)};
        key.algorithm.pipeline        = pipeline;
        key.algorithm.scheduler       = scheduler;
        key.algorithm.epilogue        = epilogue;
        key.algorithm.block_size      = block_size;
        key.algorithm.double_buffer   = true;
        key.algorithm.persistent      = false;
        key.algorithm.preshuffle      = preshuffle;
        key.algorithm.transpose_c     = false;
        key.algorithm.num_wave_groups = 1;

        key.gfx_arch = gfx_arch;

        return key;
    }

    // Convenience preset methods
    static KernelKeyBuilder fp16_rcr() { return KernelKeyBuilder{}; }

    static KernelKeyBuilder fp16_rrr()
    {
        auto b     = KernelKeyBuilder{};
        b.layout_b = LayoutTag::RowMajor;
        return b;
    }

    static KernelKeyBuilder preshuffle_v1()
    {
        auto b       = KernelKeyBuilder{};
        b.pipeline   = Pipeline::PreShuffleV1;
        b.preshuffle = true;
        return b;
    }

    static KernelKeyBuilder preshuffle_v2()
    {
        auto b       = KernelKeyBuilder{};
        b.pipeline   = Pipeline::PreShuffleV2;
        b.preshuffle = true;
        return b;
    }

    static KernelKeyBuilder multi_d(int num_d, const std::string& op = "MultiDAdd")
    {
        auto b           = KernelKeyBuilder{};
        b.num_d_tensors  = num_d;
        b.elementwise_op = op;
        return b;
    }
};

} // namespace utils
} // namespace dispatcher
} // namespace ck_tile
