// SPDX-License-Identifier: MIT
// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.

#include <hip/hip_runtime.h>
#include <cstring>
#include <iostream>
#include <ostream>
#include <string>
#include <tuple>
#include <memory>
#include <vector>
#include <algorithm>
#include <thread>
#include <chrono>

#include "ck_tile/core.hpp"
#include "ck_tile/ops/epilogue.hpp"
#include "ck_tile/ops/gemm.hpp"
#include "ck_tile/host.hpp"
#include "ck_tile/ops/gemm/kernel/persistent_async_utils.hpp"
#include "persistent_async_scheduler.hpp"
#include "grouped_gemm.hpp"
#include "grouped_gemm_persistent_async.hpp"

/**
 * @brief Helper to allocate and initialize chunk signals
 *
 * @param num_chunks Number of chunks to allocate signals for
 * @param stream HIP stream for async operations
 * @return Device pointer to chunk signals array
 */
[[maybe_unused]] static uint32_t* allocate_chunk_signals(int num_chunks, hipStream_t stream)
{
    uint32_t* signals_device = nullptr;

    // Allocate device memory for signals
    ck_tile::hip_check_error(hipMalloc(&signals_device, num_chunks * sizeof(uint32_t)));

    // Initialize all signals to 0 (not ready)
    ck_tile::hip_check_error(
        hipMemsetAsync(signals_device, 0, num_chunks * sizeof(uint32_t), stream));

    return signals_device;
}

/**
 * @brief Helper to signal chunk readiness
 *
 * @param signals Device pointer to signals array
 * @param chunk_idx Index of chunk to signal
 * @param stream HIP stream for async operations
 */
[[maybe_unused]] static void
signal_chunk_ready(uint32_t* signals, int chunk_idx, hipStream_t stream)
{
    uint32_t ready = 1;
    ck_tile::hip_check_error(hipMemcpyAsync(
        &signals[chunk_idx], &ready, sizeof(uint32_t), hipMemcpyHostToDevice, stream));
}

template <typename GemmConfig,
          typename ADataType,
          typename BDataType,
          typename CDataType,
          typename AccDataType,
          typename ALayout,
          typename BLayout,
          typename CLayout>
int run_grouped_gemm_persistent_async_example(ck_tile::ArgParser& arg_parser,
                                              const ALayout a_layout,
                                              const BLayout b_layout,
                                              const CLayout c_layout,
                                              ck_tile::PersistentAsyncArgs async_args)
{
    const int group_count = arg_parser.get_int("group_count");
    const int repeat      = arg_parser.get_int("repeat");
    const int warmup      = arg_parser.get_int("warmup");
    const int kbatch      = arg_parser.get_int("kbatch");
    const bool validate   = arg_parser.get_bool("validate");

    // Get problem dimensions (use defaults if not provided)
    std::vector<ck_tile::index_t> Ms = arg_parser.get_int_vec("Ms");
    std::vector<ck_tile::index_t> Ns = arg_parser.get_int_vec("Ns");
    std::vector<ck_tile::index_t> Ks = arg_parser.get_int_vec("Ks");

    if(Ms.empty() || Ns.empty() || Ks.empty())
    {
        std::cout << "Using default problem sizes..." << std::endl;
        Ms.clear();
        Ns.clear();
        Ks.clear();
        for(int i = 0; i < group_count; i++)
        {
            Ms.push_back(256 + 256 * i);
            Ns.push_back(256 + 512 * i);
            Ks.push_back(512 + 384 * i);
        }
    }

    // Calculate number of chunks needed for async mode
    int num_chunks                 = 0;
    uint32_t* chunk_signals_device = nullptr;

    if(async_args.enable_async && async_args.tiles_per_chunk_m > 0)
    {
        // Calculate total number of M tiles across all groups
        for(int i = 0; i < group_count; ++i)
        {
            const int m_tiles = (Ms[i] + GemmConfig::M_Tile - 1) / GemmConfig::M_Tile;
            const int chunks_for_group =
                (m_tiles + async_args.tiles_per_chunk_m - 1) / async_args.tiles_per_chunk_m;
            num_chunks += chunks_for_group;
        }

        if(num_chunks > 0)
        {
            std::cout << "  Allocating " << num_chunks << " chunk signals for async mode..."
                      << std::endl;
            chunk_signals_device     = allocate_chunk_signals(num_chunks, nullptr);
            async_args.chunk_signals = chunk_signals_device;
        }
    }

    std::cout << "\nPersistent Async GEMM Configuration:" << std::endl;
    std::cout << "  Group Count: " << group_count << std::endl;
    std::cout << "  Warmup: " << warmup << ", Repeat: " << repeat << std::endl;
    std::cout << "  K-Batch: " << kbatch << std::endl;
    std::cout << "  Validation: " << (validate ? "Enabled" : "Disabled") << std::endl;
    std::cout << "  Async Mode: " << (async_args.enable_async ? "ENABLED" : "Disabled")
              << std::endl;

    if(async_args.enable_async)
    {
        std::cout << "  Tiles per chunk (M): " << async_args.tiles_per_chunk_m << std::endl;
        std::cout << "  Tile pivot (M): " << async_args.tile_idx_pivot_m << std::endl;
    }

    // Allocate and initialize host/device tensors
    std::vector<ck_tile::HostTensor<ADataType>> a_tensors;
    std::vector<ck_tile::HostTensor<BDataType>> b_tensors;
    std::vector<ck_tile::HostTensor<CDataType>> c_tensors;
    std::vector<std::unique_ptr<ck_tile::DeviceMem>> a_dev_bufs;
    std::vector<std::unique_ptr<ck_tile::DeviceMem>> b_dev_bufs;
    std::vector<std::unique_ptr<ck_tile::DeviceMem>> c_dev_bufs;
    std::vector<grouped_gemm_kargs> gemm_descs;

    for(int i = 0; i < group_count; ++i)
    {
        const auto M = Ms[i];
        const auto N = Ns[i];
        const auto K = Ks[i];

        // Create host tensors
        a_tensors.emplace_back(ck_tile::HostTensor<ADataType>({M, K}));
        b_tensors.emplace_back(ck_tile::HostTensor<BDataType>({K, N}));
        c_tensors.emplace_back(ck_tile::HostTensor<CDataType>({M, N}));

        // Initialize with random data
        ck_tile::FillUniformDistribution<ADataType>{-5.f, 5.f}(a_tensors[i]);
        ck_tile::FillUniformDistribution<BDataType>{-5.f, 5.f}(b_tensors[i]);

        // Allocate device memory
        a_dev_bufs.emplace_back(
            std::make_unique<ck_tile::DeviceMem>(a_tensors[i].get_element_space_size_in_bytes()));
        b_dev_bufs.emplace_back(
            std::make_unique<ck_tile::DeviceMem>(b_tensors[i].get_element_space_size_in_bytes()));
        c_dev_bufs.emplace_back(
            std::make_unique<ck_tile::DeviceMem>(c_tensors[i].get_element_space_size_in_bytes()));

        // Copy to device
        a_dev_bufs[i]->ToDevice(a_tensors[i].data());
        b_dev_bufs[i]->ToDevice(b_tensors[i].data());

        // Calculate strides
        const auto get_stride = [](auto layout, auto m, auto n) {
            if constexpr(std::is_same_v<decltype(layout), ck_tile::tensor_layout::gemm::RowMajor>)
                return n;
            else
                return m;
        };

        gemm_descs.push_back({a_dev_bufs[i]->GetDeviceBuffer(),
                              b_dev_bufs[i]->GetDeviceBuffer(),
                              {/*ds_ptr*/},
                              c_dev_bufs[i]->GetDeviceBuffer(),
                              kbatch,
                              M,
                              N,
                              K,
                              get_stride(a_layout, M, K),
                              get_stride(b_layout, K, N),
                              {/*stride_Ds*/},
                              get_stride(c_layout, M, N)});
    }

    // Allocate workspace for kernel arguments
    ck_tile::DeviceMem gemm_workspace;
    gemm_workspace.Realloc(get_workspace_size(gemm_descs));

    // Prepare kernel arguments
    std::vector<ck_tile::GemmTransKernelArg<>> kargs;
    int cumulative_chunks = 0;

    for(const auto& desc : gemm_descs)
    {
        // Calculate chunk offset for this group (if async mode enabled)
        // int chunk_offset = cumulative_chunks;
        if(async_args.enable_async && async_args.tiles_per_chunk_m > 0)
        {
            const int m_tiles = (desc.M + GemmConfig::M_Tile - 1) / GemmConfig::M_Tile;
            const int chunks_for_group =
                (m_tiles + async_args.tiles_per_chunk_m - 1) / async_args.tiles_per_chunk_m;
            cumulative_chunks += chunks_for_group;
        }

        kargs.emplace_back(ck_tile::UniversalGemmKernelArgs<>{{desc.a_ptr},
                                                              {desc.b_ptr},
                                                              {},
                                                              desc.e_ptr,
                                                              desc.M,
                                                              desc.N,
                                                              desc.K,
                                                              {desc.stride_A},
                                                              {desc.stride_B},
                                                              {},
                                                              desc.stride_E,
                                                              desc.k_batch});
    }

    ck_tile::ignore = cumulative_chunks;

    // Copy kernel args to device
    const auto stream = ck_tile::stream_config{nullptr, true, 1, warmup, repeat};
    void* kargs_ptr   = gemm_workspace.GetDeviceBuffer();
    HIP_CHECK_ERROR(hipMemcpyWithStream(kargs_ptr,
                                        kargs.data(),
                                        kargs.size() * sizeof(ck_tile::GemmTransKernelArg<>),
                                        hipMemcpyHostToDevice,
                                        stream.stream_id_));

    // If async mode is enabled, signal all chunks as ready
    // In a real async scenario, these would be signaled incrementally as data arrives
    if(async_args.enable_async && async_args.chunk_signals != nullptr)
    {
        std::cout << "  [ASYNC MODE] Signaling " << num_chunks << " chunks as ready..."
                  << std::endl;
        std::cout << "  [ASYNC MODE] Chunk signals allocated at device address: "
                  << async_args.chunk_signals << std::endl;

        // For demonstration, signal all chunks immediately
        // In production, this would happen asynchronously as data becomes available
        std::vector<uint32_t> ready_signals(num_chunks, 1);
        HIP_CHECK_ERROR(hipMemcpyAsync(async_args.chunk_signals,
                                       ready_signals.data(),
                                       num_chunks * sizeof(uint32_t),
                                       hipMemcpyHostToDevice,
                                       stream.stream_id_));

        // Synchronize to ensure signals are visible before kernel launch
        HIP_CHECK_ERROR(hipStreamSynchronize(stream.stream_id_));
        std::cout << "  [ASYNC MODE] All chunks signaled and ready for processing" << std::endl;

        // NOTE: In a real async scenario, you could signal chunks incrementally like this:
        // std::thread producer([&]() {
        //     for(int chunk = 0; chunk < num_chunks; ++chunk) {
        //         // Simulate data preparation delay
        //         std::this_thread::sleep_for(std::chrono::microseconds(100));
        //         signal_chunk_ready(async_args.chunk_signals, chunk, stream.stream_id_);
        //         std::cout << "  [PRODUCER] Chunk " << chunk << " ready" << std::endl;
        //     }
        // });
        // producer.join();
    }
    else
    {
        std::cout << "  [NON-ASYNC MODE] Running without chunk signaling" << std::endl;
    }

    // Launch persistent async kernel
    std::cout << "\nLaunching persistent async GEMM kernel..." << std::endl;

    float ave_time = invoke_grouped_gemm_persistent<GemmConfig,
                                                    ADataType,
                                                    BDataType,
                                                    AccDataType,
                                                    ck_tile::tuple<>,
                                                    CDataType,
                                                    ck_tile::tuple<>,
                                                    ALayout,
                                                    BLayout,
                                                    CLayout>(stream, group_count, kargs_ptr);

    std::size_t total_flops = 0;
    std::size_t total_bytes = 0;

    for(int i = 0; i < group_count; ++i)
    {
        const auto M = Ms[i];
        const auto N = Ns[i];
        const auto K = Ks[i];

        total_flops += 2ull * M * N * K * kbatch;
        total_bytes += (M * K + K * N + M * N * kbatch) * sizeof(ADataType);
    }

    float tflops    = static_cast<float>(total_flops) / 1.e12 / ave_time;
    float bandwidth = static_cast<float>(total_bytes) / 1.e9 / ave_time;

    std::cout << "\nPerformance Results:" << std::endl;
    std::cout << "  Average Time: " << ave_time << " ms" << std::endl;
    std::cout << "  Performance: " << tflops << " TFlops" << std::endl;
    std::cout << "  Bandwidth: " << bandwidth << " GB/s" << std::endl;

    // Validation (if enabled)
    bool pass = true;
    if(validate)
    {
        std::cout << "\nValidating results..." << std::endl;

        for(int i = 0; i < group_count; ++i)
        {
            // Copy result back from device
            c_dev_bufs[i]->FromDevice(c_tensors[i].data());

            // Compute reference on CPU
            ck_tile::HostTensor<CDataType> c_ref({Ms[i], Ns[i]});
            c_ref.SetZero();
            ck_tile::reference_gemm<ADataType, BDataType, AccDataType, CDataType>(
                a_tensors[i], b_tensors[i], c_ref);

            // Calculate thresholds based on accumulation
            const float max_accumulated_value =
                *std::max_element(c_ref.mData.begin(), c_ref.mData.end());

            // Use proper error calculation
            using ComputeType =
                std::conditional_t<sizeof(ADataType) < sizeof(BDataType), ADataType, BDataType>;
            const auto rtol = ck_tile::get_relative_threshold<ComputeType, CDataType, AccDataType>(
                ck_tile::integer_divide_ceil(Ks[i], kbatch));
            const auto atol = ck_tile::get_absolute_threshold<ComputeType, CDataType, AccDataType>(
                max_accumulated_value / kbatch, ck_tile::integer_divide_ceil(Ks[i], kbatch));

            // Compare
            bool group_pass =
                ck_tile::check_err(c_tensors[i],
                                   c_ref,
                                   "Error: Incorrect results in group " + std::to_string(i),
                                   rtol,
                                   atol);

            std::cout << "Group[" << i << "] M=" << Ms[i] << " N=" << Ns[i] << " K=" << Ks[i]
                      << " - rtol=" << rtol << " atol=" << atol << " - "
                      << (group_pass ? "PASS" : "FAIL") << std::endl;

            pass &= group_pass;
        }

        std::cout << "\nOverall validation: " << (pass ? "PASSED" : "FAILED") << std::endl;
    }

    // Cleanup chunk signals if allocated
    if(async_args.chunk_signals != nullptr)
    {
        std::cout << "\n[ASYNC MODE] Freeing " << num_chunks << " chunk signals..." << std::endl;
        HIP_CHECK_ERROR(hipFree(async_args.chunk_signals));
        async_args.chunk_signals = nullptr;
    }

    if(!pass && validate)
    {
        return -1;
    }

    return 0;
}

template <typename GemmConfig, typename PrecType>
int run_gemm_example_prec_type(std::string a_layout,
                               std::string b_layout,
                               ck_tile::ArgParser& arg_parser)
{
    using Row   = ck_tile::tensor_layout::gemm::RowMajor;
    using Col   = ck_tile::tensor_layout::gemm::ColumnMajor;
    using Types = GemmTypeConfig<PrecType>;

    // Specific type aliases for easy access
    using ADataType   = typename Types::ADataType;
    using BDataType   = typename Types::BDataType;
    using AccDataType = typename Types::AccDataType;
    using CDataType   = typename Types::CDataType;

    // Parse async-specific arguments
    const bool enable_async                  = arg_parser.get_int("enable_async") != 0;
    const ck_tile::index_t tiles_per_chunk_m = arg_parser.get_int("tiles_per_chunk_m");
    const ck_tile::index_t tile_idx_pivot_m  = arg_parser.get_int("tile_idx_pivot_m");

    std::cout << "\n=== Async Parameters ===" << std::endl;
    std::cout << "  enable_async: " << (enable_async ? "YES (will allocate chunk signals)" : "NO")
              << std::endl;
    std::cout << "  tiles_per_chunk_m: " << tiles_per_chunk_m << std::endl;
    std::cout << "  tile_idx_pivot_m: " << tile_idx_pivot_m << std::endl;

    // Create async args (chunk signals will be allocated in the example function)
    ck_tile::PersistentAsyncArgs async_args(
        tiles_per_chunk_m, nullptr, tile_idx_pivot_m, enable_async);

    if(a_layout == "R" && b_layout == "C")
    {
        return run_grouped_gemm_persistent_async_example<GemmConfig,
                                                         ADataType,
                                                         BDataType,
                                                         CDataType,
                                                         AccDataType>(
            arg_parser, Row{}, Col{}, Row{}, async_args);
    }
    else if(a_layout == "R" && b_layout == "R")
    {
        return run_grouped_gemm_persistent_async_example<GemmConfig,
                                                         ADataType,
                                                         BDataType,
                                                         CDataType,
                                                         AccDataType>(
            arg_parser, Row{}, Row{}, Row{}, async_args);
    }
    else if(a_layout == "C" && b_layout == "R")
    {
        return run_grouped_gemm_persistent_async_example<GemmConfig,
                                                         ADataType,
                                                         BDataType,
                                                         CDataType,
                                                         AccDataType>(
            arg_parser, Col{}, Row{}, Row{}, async_args);
    }
    else if(a_layout == "C" && b_layout == "C")
    {
        return run_grouped_gemm_persistent_async_example<GemmConfig,
                                                         ADataType,
                                                         BDataType,
                                                         CDataType,
                                                         AccDataType>(
            arg_parser, Col{}, Col{}, Row{}, async_args);
    }
    else
    {
        throw std::runtime_error("Unsupported data layout configuration for A and B tensors!");
    }
}

template <template <typename PrecType> typename GemmConfig>
int run_grouped_gemm_example(ck_tile::ArgParser& arg_parser)
{
    const std::string a_layout  = arg_parser.get_str("a_layout");
    const std::string b_layout  = arg_parser.get_str("b_layout");
    const std::string data_type = arg_parser.get_str("prec");

    if(data_type == "fp16")
    {
        return run_gemm_example_prec_type<GemmConfig<ck_tile::half_t>, ck_tile::half_t>(
            a_layout, b_layout, arg_parser);
    }
    else if(data_type == "bf16")
    {
        return run_gemm_example_prec_type<GemmConfig<ck_tile::bf16_t>, ck_tile::bf16_t>(
            a_layout, b_layout, arg_parser);
    }
    else if(data_type == "fp8")
    {
        return run_gemm_example_prec_type<GemmConfig<ck_tile::fp8_t>, ck_tile::fp8_t>(
            a_layout, b_layout, arg_parser);
    }
    else
    {
        throw std::runtime_error("Unsupported data type configuration.");
    }
}

int main(int argc, char* argv[])
{
    auto [result, arg_parser] = create_args(argc, argv);

    // Add async-specific arguments
    arg_parser.insert(
        "tiles_per_chunk_m", "1", "Number of M tiles per chunk (granularity of async readiness)");
    arg_parser.insert(
        "tile_idx_pivot_m", "0", "Pivot offset for M dimension (for hotspot spreading)");
    arg_parser.insert("enable_async", "1", "Enable async input signaling (0=disabled, 1=enabled)");

    if(!result)
        return -1;

    std::cout << "=== Grouped GEMM Persistent Async Test ===" << std::endl;

    try
    {
        int ret = run_grouped_gemm_example<GemmConfigComputeV3_2>(arg_parser);
        return ret;
    }
    catch(const std::exception& e)
    {
        std::cerr << "Error: " << e.what() << std::endl;
        return -1;
    }
}
