// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

/**
 * Stream-K GEMM driver through the Registry + Dispatcher (deep-core path).
 *
 * Unlike 03_streamk_gemm_driver.cpp (which calls SelectedKernel::launch()
 * DIRECTLY, bypassing the dispatcher), this driver proves the full deep-core
 * path that PR-A..PR-C built:
 *
 *     Registry::register_kernel(GeneratedStreamKKernelInstance)
 *         -> Dispatcher::run(Problem.stream_k(Atomic))
 *         -> Dispatcher::select_first_fit -> SK instance.supports()
 *         -> GeneratedStreamKKernelInstance::run -> SelectedKernel::launch()
 *
 * It registers ONE generated Stream-K kernel (force-included via
 * -include / -DCK_TILE_SINGLE_KERNEL_INCLUDE), selects it through the registry
 * by Problem::reduction_strategy, runs it, and verifies vs reference_gemm.
 *
 * Build (single-kernel include style):
 *   hipcc -std=c++17 --offload-arch=gfx942 -O3 \
 *     -DCK_TILE_SINGLE_KERNEL_INCLUDE \
 *     -I <ck>/include -I <ck>/dispatcher/include -I <generated_dir> \
 *     -include <generated_dir>/<one>_streamk.hpp \
 *     04_streamk_registry_driver.cpp -o streamk_registry_driver
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

#include "ck_tile/dispatcher/dispatcher.hpp"
#include "ck_tile/dispatcher/registry.hpp"
#include "ck_tile/dispatcher/backends/generated_tile_backend_streamk.hpp"

#include "streamk_driver_common.hpp"

// The generated stream-K kernel header is injected on the command line with
// -include and -DCK_TILE_SINGLE_KERNEL_INCLUDE. It exports into the global
// namespace: SelectedKernel, ADataType, BDataType, CDataType, AccDataType,
// ALayout, BLayout, CLayout, and KERNEL_NAME.

#ifndef GFX_ARCH
#define GFX_ARCH "gfx942"
#endif

using namespace ck_tile::dispatcher;
using namespace ck_tile::dispatcher::backends;
using Priority = ck_tile::dispatcher::Registry::Priority;

// CLI parsing, layout/dtype tags, and the Stream-K verification tolerance are
// shared with the standalone 03 driver via streamk_driver_common.hpp
// (is_row_major, get_opt, dtype_enum_of, layout_tag_of, streamk_tolerance).

// Build the KernelKey for the force-included Stream-K kernel. Only the Stream-K
// axis (streamk + reduction_strategy) governs selection; the remaining fields
// are populated for a faithful encode_identifier()/registry entry.
static KernelKey make_streamk_key(ReductionStrategy strategy)
{
    KernelKey key;
    key.signature.dtype_a     = dtype_enum_of<ADataType>();
    key.signature.dtype_b     = dtype_enum_of<BDataType>();
    key.signature.dtype_c     = dtype_enum_of<CDataType>();
    key.signature.dtype_acc   = dtype_enum_of<AccDataType>();
    key.signature.layout_a    = layout_tag_of<ALayout>();
    key.signature.layout_b    = layout_tag_of<BLayout>();
    key.signature.layout_c    = layout_tag_of<CLayout>();
    key.signature.transpose_a = false;
    key.signature.transpose_b = false;
    key.signature.grouped     = false;
    // Stream-K performs its own K-dimension partitioning through the tile
    // partitioner, so classic split-k is always 1 here. A value > 1 would
    // describe a two-level K split the Stream-K kernel does not implement.
    key.signature.split_k             = 1;
    key.signature.elementwise_op      = "PassThrough";
    key.signature.num_d_tensors       = 0;
    key.signature.structured_sparsity = false;

    // Derive algorithm metadata from the generated kernel's own static traits so
    // the registry identifier describes the kernel that was actually built,
    // instead of assuming one tile/wave config. (Selection keys only on the
    // Stream-K axis below, but a faithful identifier matters for logging and any
    // future key-based lookup.)
    key.algorithm.tile_shape = {
        SelectedKernel::TileM, SelectedKernel::TileN, SelectedKernel::TileK};
    key.algorithm.warp_tile_shape = {static_cast<std::uint8_t>(SelectedKernel::WarpTileM),
                                     static_cast<std::uint8_t>(SelectedKernel::WarpTileN),
                                     static_cast<std::uint8_t>(SelectedKernel::WarpTileK)};
    key.algorithm.wave_shape      = {static_cast<std::uint8_t>(SelectedKernel::WarpPerBlock_M),
                                     static_cast<std::uint8_t>(SelectedKernel::WarpPerBlock_N),
                                     static_cast<std::uint8_t>(SelectedKernel::WarpPerBlock_K)};
    // Pipeline (CompV3) and scheduler (Intrawave) are baked into the generated
    // kernel's type, not exposed as standalone enum values, and are not part of
    // the Stream-K selection axis -- they stay at the codegen defaults.
    key.algorithm.pipeline        = Pipeline::CompV3;
    key.algorithm.scheduler       = Scheduler::Intrawave;
    key.algorithm.epilogue        = Epilogue::CShuffle;
    key.algorithm.block_size      = SelectedKernel::BlockSize;
    key.algorithm.double_buffer   = SelectedKernel::DoubleSmemBuffer;
    key.algorithm.persistent      = SelectedKernel::UsePersistentKernel;
    key.algorithm.preshuffle      = SelectedKernel::Preshuffle;
    key.algorithm.transpose_c     = SelectedKernel::TransposeC;
    key.algorithm.num_wave_groups = SelectedKernel::NumWaveGroups;
    key.algorithm.pad_m           = SelectedKernel::kPadM;
    key.algorithm.pad_n           = SelectedKernel::kPadN;
    key.algorithm.pad_k           = SelectedKernel::kPadK;

    // The Stream-K selection axis (the whole point of this path).
    key.algorithm.streamk            = true;
    key.algorithm.reduction_strategy = strategy;
    key.algorithm.workspace          = (strategy != ReductionStrategy::Atomic);

    key.gfx_arch = GFX_ARCH;
    return key;
}

static ReductionStrategy parse_strategy(const std::string& s)
{
    if(s == "linear")
        return ReductionStrategy::Linear;
    if(s == "tree")
        return ReductionStrategy::Tree;
    return ReductionStrategy::Atomic;
}

int main(int argc, char** argv)
{
    const ck_tile::index_t M         = std::stoll(get_opt(argc, argv, "--m", "3840"));
    const ck_tile::index_t N         = std::stoll(get_opt(argc, argv, "--n", "4096"));
    const ck_tile::index_t K         = std::stoll(get_opt(argc, argv, "--k", "2048"));
    const bool validate              = get_opt(argc, argv, "--validate", "1") != "0";
    const ReductionStrategy strategy = parse_strategy(get_opt(argc, argv, "--strategy", "atomic"));

    std::cout << "Kernel: " << KERNEL_NAME << "\n";
    std::cout << "M=" << M << " N=" << N << " K=" << K << " strategy=" << to_string(strategy)
              << "\n";

    // --- Register the kernel into the global registry ---------------------------
    KernelKey key = make_streamk_key(strategy);
    auto kernel   = create_generated_streamk_kernel<SelectedKernel,
                                                    ADataType,
                                                    BDataType,
                                                    CDataType,
                                                    AccDataType>(key, KERNEL_NAME);
    Registry::instance().clear();
    Registry::instance().register_kernel(kernel, Priority::High);
    std::cout << "Registered kernels: " << Registry::instance().size()
              << "  identifier=" << key.encode_identifier() << "\n";

    // --- Build the problem requesting THIS Stream-K strategy --------------------
    Problem problem(M, N, K);
    problem.streamk            = true;
    problem.reduction_strategy = strategy;

    Dispatcher dispatcher;
    auto selected = dispatcher.select_kernel(problem);
    if(!selected)
    {
        std::cout << "Dispatcher selected NO kernel for the Stream-K problem -> FAIL\n";
        return 1;
    }
    std::cout << "Dispatcher selected: " << selected->get_name() << "\n";

    // --- Tensors (rcr) ---------------------------------------------------------
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

    // --- Run through the dispatcher (registry -> Dispatcher::run -> SK backend) -
    float ave_time = 0.f;
    try
    {
        ave_time = dispatcher.run(
            a_dev.GetDeviceBuffer(), b_dev.GetDeviceBuffer(), c_dev.GetDeviceBuffer(), problem);
    }
    catch(const std::exception& e)
    {
        std::cout << "Dispatcher::run threw: " << e.what() << " -> FAIL\n";
        return 1;
    }

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
        // own tile partitioner so this driver and tile_engine agree on the split
        // factor. streamk_tolerance() then widens the verify tolerance for the
        // split-K accumulation error (see streamk_driver_common.hpp).
        ck_tile::StreamKHostArgs sk_args{a_dev.GetDeviceBuffer(),
                                         b_dev.GetDeviceBuffer(),
                                         c_dev.GetDeviceBuffer(),
                                         M,
                                         N,
                                         K,
                                         sA,
                                         sB,
                                         sC};
        using ComputeType =
            std::conditional_t<sizeof(ADataType) < sizeof(BDataType), ADataType, BDataType>;
        auto kargs = SelectedKernel::StreamKGemmKernel::MakeKernelArgs(sk_args);
        const ck_tile::index_t num_wgs_per_tile =
            std::max<ck_tile::index_t>(1, kargs.tile_partitioner.estimate_num_wgs_per_tile());
        const auto tol =
            streamk_tolerance<ComputeType, CDataType, AccDataType>(K, num_wgs_per_tile, maxv);
        pass = ck_tile::check_err(c_host, ref, "streamk_registry", tol.rtol, tol.atol);
        std::cout << "Verification: " << (pass ? "PASS" : "FAIL") << "\n";
    }

    return pass ? 0 : 1;
}
