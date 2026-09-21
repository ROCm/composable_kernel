// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

/**
 * Minimal standalone stream-K GEMM driver (dispatcher way).
 *
 * Stream-K is a SINGLE GEMM that splits the K dimension across CUs and reduces
 * the partial results through a device workspace. Like grouped GEMM it cannot
 * ride the standard dispatcher.run(A,B,C,problem) path, so this driver includes
 * a single generated stream-K kernel header (CK_TILE_SINGLE_KERNEL_INCLUDE) and
 * calls SelectedKernel::launch(args, stream) directly with a single
 * StreamKHostArgs -- the same 2-arg signature the dispatcher generates (the
 * workspace is allocated INSIDE launch() via DeviceMem). It builds one A/B/C
 * tensor, runs, and verifies against ck_tile::reference_gemm.
 *
 * Build (single-kernel include style):
 *   hipcc -std=c++17 --offload-arch=gfx942 -O3 \
 *     -DCK_TILE_SINGLE_KERNEL_INCLUDE \
 *     -I <ck>/include -I <generated_dir> \
 *     -include <generated_dir>/<one>_streamk.hpp \
 *     03_streamk_gemm_driver.cpp -o streamk_gemm_driver
 */

#include <hip/hip_runtime.h>

#include <algorithm>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <string>

#include "ck_tile/core.hpp"
#include "ck_tile/host.hpp"
#include "ck_tile/ops/gemm.hpp"

#include "streamk_driver_common.hpp"

// The generated stream-K kernel header is injected on the command line with
// -include and -DCK_TILE_SINGLE_KERNEL_INCLUDE. It exports into the global
// namespace: SelectedKernel, ADataType, BDataType, CDataType, AccDataType,
// ALayout, BLayout, CLayout, and KERNEL_NAME.

int main(int argc, char** argv)
{
    const ck_tile::index_t M = std::stoll(get_opt(argc, argv, "--m", "3840"));
    const ck_tile::index_t N = std::stoll(get_opt(argc, argv, "--n", "4096"));
    const ck_tile::index_t K = std::stoll(get_opt(argc, argv, "--k", "2048"));
    int warmup               = std::stoi(get_opt(argc, argv, "--warmup", "50"));
    int repeat               = std::stoi(get_opt(argc, argv, "--repeat", "100"));
    const bool validate      = get_opt(argc, argv, "--validate", "1") != "0";

    // Apple-to-apple with tile_engine: time the kernel with the SAME methodology the
    // tile_engine benchmark uses (gemm_streamk_profiler.hpp) -- gpu timer and a
    // cold-cache measurement that flushes the cache and rotates input buffers each
    // iteration. tile_engine defaults: timer=true, flush_cache=true, rotating_count=1000.
    // Without these the driver measured a warm-cache best case and over-reported TFlops,
    // which is the entire source of the dispatcher-vs-TE "performance gap".
    const bool gpu_timer = get_opt(argc, argv, "--timer", "1") != "0";
    bool flush_cache     = get_opt(argc, argv, "--flush_cache", "1") != "0";
    int rotating_count   = std::stoi(get_opt(argc, argv, "--rotating_count", "1000"));

    // Verification reads C back and compares against the reference for the known A/B.
    // Rotating buffers and multi-repeat rotate/accumulate the output, so the C left on
    // the device would not correspond to the reference inputs. tile_engine handles this
    // with repeat_once_if_verify(); we mirror it -- a validating run times a single cold
    // shot. Run a separate --validate 0 pass to collect apple-to-apple perf numbers.
    if(validate)
    {
        warmup         = 0;
        repeat         = 1;
        flush_cache    = false;
        rotating_count = 1;
    }

    std::cout << "Kernel: " << KERNEL_NAME << "\n";
    std::cout << "M=" << M << " N=" << N << " K=" << K << "\n";

    const ck_tile::index_t sA = ck_tile::get_default_stride(M, K, 0, is_row_major(ALayout{}));
    const ck_tile::index_t sB = ck_tile::get_default_stride(K, N, 0, is_row_major(BLayout{}));
    const ck_tile::index_t sC = ck_tile::get_default_stride(M, N, 0, is_row_major(CLayout{}));

    ck_tile::HostTensor<ADataType> a_host(
        ck_tile::host_tensor_descriptor(M, K, sA, is_row_major(ALayout{})));
    ck_tile::HostTensor<BDataType> b_host(
        ck_tile::host_tensor_descriptor(K, N, sB, is_row_major(BLayout{})));
    ck_tile::HostTensor<CDataType> c_host(
        ck_tile::host_tensor_descriptor(M, N, sC, is_row_major(CLayout{})));

    ck_tile::FillUniformDistribution<ADataType>{-1.f, 1.f}(a_host);
    ck_tile::FillUniformDistribution<BDataType>{-1.f, 1.f}(b_host);
    c_host.SetZero();

    ck_tile::DeviceMem a_dev(a_host);
    ck_tile::DeviceMem b_dev(b_host);
    ck_tile::DeviceMem c_dev(c_host);
    c_dev.SetZero();

    ck_tile::StreamKHostArgs args{a_dev.GetDeviceBuffer(),
                                  b_dev.GetDeviceBuffer(),
                                  c_dev.GetDeviceBuffer(),
                                  M,
                                  N,
                                  K,
                                  sA,
                                  sB,
                                  sC};

    const ck_tile::stream_config s{
        nullptr, true, /*log=*/0, warmup, repeat, gpu_timer, flush_cache, rotating_count};
    float ave_time = SelectedKernel::launch(args, s);

    const std::size_t flop = std::size_t(2) * M * N * K;
    const std::size_t bytes =
        sizeof(ADataType) * M * K + sizeof(BDataType) * K * N + sizeof(CDataType) * M * N;
    const float tflops = static_cast<float>(flop) / 1.E9 / ave_time;
    const float gbps   = static_cast<float>(bytes) / 1.E6 / ave_time;
    std::cout << "Perf: " << std::setw(10) << ave_time << " ms, " << tflops << " TFlops, " << gbps
              << " GB/s\n";

    c_dev.FromDevice(c_host.data());

    bool pass = true;
    if(validate)
    {
        ck_tile::HostTensor<CDataType> ref(
            ck_tile::host_tensor_descriptor(M, N, sC, is_row_major(CLayout{})));
        ref.SetZero();
        ck_tile::reference_gemm<ADataType, BDataType, AccDataType, CDataType>(a_host, b_host, ref);
        const float maxv = *std::max_element(ref.mData.begin(), ref.mData.end());

        // num_wgs_per_tile is the number of workgroups reducing into a single
        // output tile (Stream-K has no fixed split-k), taken from the kernel's
        // own tile partitioner so the driver and tile_engine agree on the split
        // factor. streamk_tolerance() then widens the verify tolerance for the
        // split-K accumulation error (see streamk_driver_common.hpp).
        using ComputeType =
            std::conditional_t<sizeof(ADataType) < sizeof(BDataType), ADataType, BDataType>;
        auto kargs = SelectedKernel::StreamKGemmKernel::MakeKernelArgs(args);
        const ck_tile::index_t num_wgs_per_tile =
            std::max<ck_tile::index_t>(1, kargs.tile_partitioner.estimate_num_wgs_per_tile());
        const auto tol =
            streamk_tolerance<ComputeType, CDataType, AccDataType>(K, num_wgs_per_tile, maxv);
        pass = ck_tile::check_err(c_host, ref, "streamk", tol.rtol, tol.atol);
        std::cout << "Verification: " << (pass ? "PASS" : "FAIL") << "\n";
    }

    return pass ? 0 : 1;
}
