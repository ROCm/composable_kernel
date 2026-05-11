// SPDX-License-Identifier: MIT
// Copyright (c) 2024, Advanced Micro Devices, Inc. All rights reserved.

/*
 * Tutorial 16: Row Reduction — Warp Reduce vs Block Reduce
 *
 * Computes Y[M] = reduce(X[M, N], axis=1)  (per-row reduction)
 * using four approaches across ck_tile reduce abstraction levels:
 *
 *   Approach 1 - Warp-only reduce (row-max)
 *     Custom device kernel. Uses BlockReduce2d (in-thread accumulation) +
 *     BlockReduce2dSync (cross-lane warp shuffle). No cross-warp sync,
 *     no LDS. Shape: all warps along M, 1 warp along N.
 *
 *   Approach 2 - Warp + cross-warp reduce (row-max)
 *     Custom device kernel. Uses BlockReduce2d (in-thread) +
 *     BlockReduce2dSync (warp shuffle) + BlockReduce2dCrossWarpSync
 *     (cross-warp via LDS). Shape: warps along both M and N.
 *
 *   Approach 3 - ReduceKernel host-only (row-max)
 *     Host-only. Uses the production ReduceKernel which wraps all three
 *     stages automatically. No custom device code needed.
 *
 *   Approach 4 - Row-sum with ReduceOp::Add
 *     Same 3-stage hierarchy as approach 2, but with addition.
 *     Shows that switching reduce operations is trivial.
 *
 * Reduce hierarchy (same as unified attention / production reduce):
 *   Stage 1: In-thread     — each thread reduces its local elements
 *   Stage 2: Warp shuffle  — cross-lane XOR butterfly reduce
 *   Stage 3: Cross-warp    — LDS store → sync → tree reduce across warps
 *
 * Data: float32 throughout.  Default: M=1024, N=2048
 */

#include <iostream>
#include <vector>
#include <cstdlib>
#include <cmath>
#include <limits>

#include "ck_tile/core.hpp"
#include "ck_tile/host.hpp"
#include "ck_tile/ops/reduce.hpp"

using namespace ck_tile;

// ======================= Tile shapes ========================================
//
// Reduce2dShape<BlockWarps, BlockTile, WarpTile, ThreadTile>
//   BlockWarps  — number of warps along seq<M, N>
//   BlockTile   — tile size processed by one thread-block, seq<M, N>
//   WarpTile    — tile size per warp, seq<M, N>
//   ThreadTile  — contiguous elements per thread (vector load), seq<M, N>

// Shape 1: 4 warps along M, 1 along N → warp-only reduce (no cross-warp)
using Shape1 = Reduce2dShape<
    sequence<4, 1>,
    sequence<64, 256>,
    sequence<16, 256>,
    sequence<1, 4>>;

// Shape 2: 2 warps along M, 2 along N → cross-warp reduce needed
// WarpTile_M must be 1 so all 64 lanes in a warp handle the same row
// (BlockReduce2dCrossWarpSync communicates via lane-0 only).
// Repeat_M = Block_M / WarpPerBlock_M = 32 covers all 64 rows per block.
using Shape2 = Reduce2dShape<
    sequence<2, 2>,
    sequence<64, 256>,
    sequence<1, 128>,
    sequence<1, 2>>;

// ============================================================================
//  Approach 1 — Warp-only row-max
//  Stage 1 (in-thread) + Stage 2 (warp shuffle). No LDS.
// ============================================================================
struct Approach1_WarpMax
{
    static constexpr index_t kBlockSize = Shape1::BlockSize;
    static constexpr index_t kBlockM   = Shape1::Block_M;

    using MyReduceOp = ReduceOp::Max;
    using ReduceProb = BlockReduce2dProblem<float, float, Shape1, false>;
    using FullProb   = Reduce2dProblem<float, float, float, Shape1, MyReduceOp,
                                       sequence<0>, sequence<1>, 2>;

    CK_TILE_DEVICE void operator()(const float* p_x, float* p_y,
                                   index_t M, index_t N) const
    {
        using S = Shape1;
        const index_t iM = get_block_id() * S::Block_M;

        MyReduceOp reduce_op{};
        const float identity = reduce_op.GetIdentityValue<float>();

        // 2D input [M, N] — out-of-bounds reads return the identity value
        auto x_desc = make_naive_tensor_descriptor(
            make_tuple(M, N), make_tuple(N, 1), number<1>{}, number<S::ThreadTile_N>{});
        auto x_buf = make_buffer_view<address_space_enum::global>(
            p_x, x_desc.get_element_space_size(), identity);
        auto x_view = tensor_view<decltype(x_buf), decltype(x_desc)>{x_buf, x_desc};
        auto x_pad  = pad_tensor_view(
            x_view,
            make_tuple(number<S::Block_M>{}, number<S::Block_N>{}),
            sequence<1, 1>{});

        // 1D output [M]
        auto y_view = make_naive_tensor_view_packed<address_space_enum::global>(
            p_y, make_tuple(M));

        // Tile windows with 2D distribution
        auto x_window = make_tile_window(
            x_pad,
            make_tuple(number<S::Block_M>{}, number<S::Block_N>{}),
            {iM, 0},
            Reduce2dDefaultPolicy::MakeXBlockTileDistribution<FullProb>());

        auto y_window = make_tile_window(
            y_view, make_tuple(number<S::Block_M>{}), {iM});

        // Reduce operators (stages 1 & 2 only)
        auto block_reduce2d      = BlockReduce2d<ReduceProb>{};
        auto block_reduce2d_sync = BlockReduce2dSync<ReduceProb>{};

        // Per-row accumulator, initialized to identity
        using XTileType = decltype(load_tile(x_window));
        auto y_acc = block_reduce2d.template MakeYBlockTile<XTileType>();
        set_tile(y_acc, identity);

        // N-tile loop: load 2D tile, accumulate per-row reduce
        const index_t num_n_iters = (N + S::Block_N - 1) / S::Block_N;
        for(index_t iN = 0; iN < num_n_iters; ++iN)
        {
            const auto x_tile = load_tile(x_window);
            block_reduce2d(x_tile, y_acc, reduce_op);   // stage 1: in-thread
            move_tile_window(x_window, {0, S::Block_N});
        }

        block_reduce2d_sync(y_acc, reduce_op);           // stage 2: warp shuffle

        store_tile(y_window, y_acc);
    }
};

// ============================================================================
//  Approach 2 — Warp + cross-warp row-max (full 3-stage with LDS)
// ============================================================================
struct Approach2_BlockMax
{
    static constexpr index_t kBlockSize = Shape2::BlockSize;
    static constexpr index_t kBlockM   = Shape2::Block_M;

    using MyReduceOp = ReduceOp::Max;
    using ReduceProb = BlockReduce2dProblem<float, float, Shape2, false>;
    using FullProb   = Reduce2dProblem<float, float, float, Shape2, MyReduceOp,
                                       sequence<0>, sequence<1>, 2>;

    static constexpr index_t kSmemSize =
        Reduce2dDefaultPolicy::GetSmemSize<FullProb>();

    CK_TILE_DEVICE void operator()(const float* p_x, float* p_y,
                                   index_t M, index_t N) const
    {
        using S = Shape2;
        __shared__ char smem[kSmemSize];

        const index_t iM = get_block_id() * S::Block_M;

        MyReduceOp reduce_op{};
        const float identity = reduce_op.GetIdentityValue<float>();

        auto x_desc = make_naive_tensor_descriptor(
            make_tuple(M, N), make_tuple(N, 1), number<S::ThreadTile_N>{});
        auto x_buf = make_buffer_view<address_space_enum::global>(
            p_x, x_desc.get_element_space_size(), identity);
        auto x_view = tensor_view<decltype(x_buf), decltype(x_desc)>{x_buf, x_desc};
        auto x_pad  = pad_tensor_view(
            x_view,
            make_tuple(number<S::Block_M>{}, number<S::Block_N>{}),
            sequence<1, 1>{});

        auto y_view = make_naive_tensor_view_packed<address_space_enum::global>(
            p_y, make_tuple(M));

        auto x_window = make_tile_window(
            x_pad,
            make_tuple(number<S::Block_M>{}, number<S::Block_N>{}),
            {iM, 0},
            Reduce2dDefaultPolicy::MakeXBlockTileDistribution<FullProb>());

        auto y_window = make_tile_window(
            y_view, make_tuple(number<S::Block_M>{}), {iM});

        // All 3 stages
        auto block_reduce2d            = BlockReduce2d<ReduceProb>{};
        auto block_reduce2d_sync       = BlockReduce2dSync<ReduceProb>{};
        auto block_reduce2d_cross_warp = BlockReduce2dCrossWarpSync<ReduceProb>{};

        using XTileType = decltype(load_tile(x_window));
        auto y_acc = block_reduce2d.template MakeYBlockTile<XTileType>();
        set_tile(y_acc, identity);

        const index_t num_n_iters = (N + S::Block_N - 1) / S::Block_N;
        for(index_t iN = 0; iN < num_n_iters; ++iN)
        {
            const auto x_tile = load_tile(x_window);
            block_reduce2d(x_tile, y_acc, reduce_op);   // stage 1: in-thread
            move_tile_window(x_window, {0, S::Block_N});
        }

        block_reduce2d_sync(y_acc, reduce_op);           // stage 2: warp shuffle
        block_reduce2d_cross_warp(y_acc, smem, reduce_op); // stage 3: cross-warp LDS

        store_tile(y_window, y_acc);
    }
};

// ============================================================================
//  Approach 3 — ReduceKernel (host-only, production-style row-max)
// ============================================================================
namespace approach3 {
using MyShape    = Shape2;
using MyReduceOp = ReduceOp::Max;
using Problem    = Reduce2dProblem<float, float, float, MyShape, MyReduceOp,
                                   sequence<0>, sequence<1>, 2>;
using Kernel     = ReduceKernel<Problem>;
} // namespace approach3

// ============================================================================
//  Approach 4 — Cross-warp row-sum (ReduceOp::Add)
//  Same 3-stage hierarchy as approach 2, only the reduce op changes.
// ============================================================================
struct Approach4_BlockSum
{
    static constexpr index_t kBlockSize = Shape2::BlockSize;
    static constexpr index_t kBlockM   = Shape2::Block_M;

    using MyReduceOp = ReduceOp::Add;
    using ReduceProb = BlockReduce2dProblem<float, float, Shape2, false>;
    using FullProb   = Reduce2dProblem<float, float, float, Shape2, MyReduceOp,
                                       sequence<0>, sequence<1>, 2>;

    static constexpr index_t kSmemSize =
        Reduce2dDefaultPolicy::GetSmemSize<FullProb>();

    CK_TILE_DEVICE void operator()(const float* p_x, float* p_y,
                                   index_t M, index_t N) const
    {
        using S = Shape2;
        __shared__ char smem[kSmemSize];

        const index_t iM = get_block_id() * S::Block_M;

        MyReduceOp reduce_op{};
        const float identity = reduce_op.GetIdentityValue<float>();

        auto x_desc = make_naive_tensor_descriptor(
            make_tuple(M, N), make_tuple(N, 1), number<S::ThreadTile_N>{});
        auto x_buf = make_buffer_view<address_space_enum::global>(
            p_x, x_desc.get_element_space_size(), identity);
        auto x_view = tensor_view<decltype(x_buf), decltype(x_desc)>{x_buf, x_desc};
        auto x_pad  = pad_tensor_view(
            x_view,
            make_tuple(number<S::Block_M>{}, number<S::Block_N>{}),
            sequence<1, 1>{});

        auto y_view = make_naive_tensor_view_packed<address_space_enum::global>(
            p_y, make_tuple(M));

        auto x_window = make_tile_window(
            x_pad,
            make_tuple(number<S::Block_M>{}, number<S::Block_N>{}),
            {iM, 0},
            Reduce2dDefaultPolicy::MakeXBlockTileDistribution<FullProb>());

        auto y_window = make_tile_window(
            y_view, make_tuple(number<S::Block_M>{}), {iM});

        auto block_reduce2d            = BlockReduce2d<ReduceProb>{};
        auto block_reduce2d_sync       = BlockReduce2dSync<ReduceProb>{};
        auto block_reduce2d_cross_warp = BlockReduce2dCrossWarpSync<ReduceProb>{};

        using XTileType = decltype(load_tile(x_window));
        auto y_acc = block_reduce2d.template MakeYBlockTile<XTileType>();
        set_tile(y_acc, identity);

        const index_t num_n_iters = (N + S::Block_N - 1) / S::Block_N;
        for(index_t iN = 0; iN < num_n_iters; ++iN)
        {
            const auto x_tile = load_tile(x_window);
            block_reduce2d(x_tile, y_acc, reduce_op);
            move_tile_window(x_window, {0, S::Block_N});
        }

        block_reduce2d_sync(y_acc, reduce_op);
        block_reduce2d_cross_warp(y_acc, smem, reduce_op);

        store_tile(y_window, y_acc);
    }
};

// ============================================================================
//  CPU reference
// ============================================================================
static void cpu_row_max(const std::vector<float>& x, std::vector<float>& y,
                        index_t M, index_t N)
{
    for(index_t m = 0; m < M; ++m)
    {
        float val = -std::numeric_limits<float>::max();
        for(index_t n = 0; n < N; ++n)
            val = std::max(val, x[m * N + n]);
        y[m] = val;
    }
}

static void cpu_row_sum(const std::vector<float>& x, std::vector<float>& y,
                        index_t M, index_t N)
{
    for(index_t m = 0; m < M; ++m)
    {
        float val = 0.0f;
        for(index_t n = 0; n < N; ++n)
            val += x[m * N + n];
        y[m] = val;
    }
}

// ============================================================================
//  Utilities
// ============================================================================
[[maybe_unused]] static void fill_random(std::vector<float>& v, float lo = -5.f, float hi = 5.f)
{
    for(auto& x : v)
        x = lo + (hi - lo) * static_cast<float>(rand()) / RAND_MAX;
}

// Fills v with 1, 2, 3, ..., v.size() in row-major order.
// Useful for debugging: with M=4, N=8 the input tile prints as
//   row 0:  1  2  3  4  5  6  7  8
//   row 1:  9 10 11 12 13 14 15 16
//   ...
// so per-row max = (m+1)*N  and  per-row sum = N*(2*m*N + N + 1)/2,
// both easy to verify by eye.
static void fill_sequential(std::vector<float>& v, float start = 1.0f)
{
    for(size_t i = 0; i < v.size(); ++i)
        v[i] = start + static_cast<float>(i);
}

static bool verify(const std::vector<float>& ref,
                   const std::vector<float>& test,
                   float tol = 1e-3f)
{
    index_t errs = 0;
    float max_err = 0;
    for(size_t i = 0; i < ref.size(); ++i)
    {
        float e = std::abs(ref[i] - test[i]);
        max_err = std::max(max_err, e);
        if(e > tol)
        {
            if(errs < 3)
                std::cout << "  mismatch [" << i << "]: " << test[i]
                          << " vs ref " << ref[i] << "  (err " << e << ")\n";
            ++errs;
        }
    }
    std::cout << "  max error: " << max_err;
    if(errs)
        std::cout << "  (" << errs << " errors)";
    std::cout << "\n";
    return errs == 0;
}

// Launch + time a custom reduce kernel
template <typename Functor>
static void run_custom_reduce(const char* name,
                              Functor func,
                              const float* x_dev, float* y_dev,
                              const std::vector<float>& y_ref,
                              std::vector<float>& y_host,
                              index_t M, index_t N,
                              float tol = 1e-3f)
{
    std::cout << name << "\n";

    const index_t grid_size  = (M + Functor::kBlockM - 1) / Functor::kBlockM;
    const index_t block_size = Functor::kBlockSize;

    stream_config stream;
    constexpr int kWarmup = 5;
    constexpr int kRepeat = 20;

    auto run = [&]() {
        launch_kernel(stream,
                      make_kernel<1>(func, dim3(grid_size), dim3(block_size), 0,
                                     x_dev, y_dev, M, N));
    };

    for(int i = 0; i < kWarmup; ++i) run();
    hip_check_error(hipDeviceSynchronize());

    hipEvent_t ev0, ev1;
    hip_check_error(hipEventCreate(&ev0));
    hip_check_error(hipEventCreate(&ev1));
    hip_check_error(hipEventRecord(ev0));
    for(int i = 0; i < kRepeat; ++i) run();
    hip_check_error(hipEventRecord(ev1));
    hip_check_error(hipEventSynchronize(ev1));

    float ms = 0;
    hip_check_error(hipEventElapsedTime(&ms, ev0, ev1));
    double avg = ms / kRepeat;
    hip_check_error(hipEventDestroy(ev0));
    hip_check_error(hipEventDestroy(ev1));

    double bytes = static_cast<double>(M) * N * sizeof(float)
                 + static_cast<double>(M) * sizeof(float);
    double gbps  = bytes / 1e9 / (avg / 1e3);

    hip_check_error(hipMemcpy(y_host.data(), y_dev,
                              M * sizeof(float), hipMemcpyDeviceToHost));
    bool ok = verify(y_ref, y_host, tol);
    std::cout << "  " << (ok ? "PASSED" : "FAILED")
              << "  |  " << avg << " ms  |  " << gbps << " GB/s\n\n";
}

// ============================================================================
//  main
// ============================================================================
int main(int argc, char* argv[])
{
    index_t M = 1024, N = 2048;
    if(argc >= 3)
    {
        M = std::atoi(argv[1]);
        N = std::atoi(argv[2]);
    }

    std::cout << "\n============================================\n"
              << "Tutorial 16: Row Reduction — Warp vs Block\n"
              << "============================================\n"
              << "Problem: X[" << M << ", " << N << "] -> Y[" << M << "]\n"
              << "Shape 1 (warp-only):  Block=" << Shape1::Block_M << "x"
              << Shape1::Block_N << ", Warps=" << Shape1::WarpPerBlock_M
              << "x" << Shape1::WarpPerBlock_N
              << " (" << Shape1::BlockSize << " threads)\n"
              << "Shape 2 (cross-warp): Block=" << Shape2::Block_M << "x"
              << Shape2::Block_N << ", Warps=" << Shape2::WarpPerBlock_M
              << "x" << Shape2::WarpPerBlock_N
              << " (" << Shape2::BlockSize << " threads)\n\n";

    std::vector<float> h_x(M * N);
    std::vector<float> h_y_max_ref(M);
    std::vector<float> h_y_sum_ref(M);
    std::vector<float> h_y(M);

    fill_sequential(h_x);

    std::cout << "Running CPU reference ...";
    std::cout.flush();
    cpu_row_max(h_x, h_y_max_ref, M, N);
    cpu_row_sum(h_x, h_y_sum_ref, M, N);
    std::cout << " done.\n\n";

    DeviceMem d_x(M * N * sizeof(float));
    DeviceMem d_y(M * sizeof(float));
    d_x.ToDevice(h_x.data(), M * N * sizeof(float));

    const auto* x_dev = static_cast<const float*>(d_x.GetDeviceBuffer());
    auto* y_dev       = static_cast<float*>(d_y.GetDeviceBuffer());

    // ---- Approach 1: Warp-only row-max ----
    run_custom_reduce(
        "Approach 1: Warp-only row-max (BlockReduce2d + BlockReduce2dSync)",
        Approach1_WarpMax{}, x_dev, y_dev, h_y_max_ref, h_y, M, N, 1e-5f);

    // ---- Approach 2: Cross-warp row-max ----
    run_custom_reduce(
        "Approach 2: Cross-warp row-max (+ BlockReduce2dCrossWarpSync via LDS)",
        Approach2_BlockMax{}, x_dev, y_dev, h_y_max_ref, h_y, M, N, 1e-5f);

    // ---- Approach 3: ReduceKernel (host-only) ----
    {
        std::cout << "Approach 3: ReduceKernel host-only (row-max)\n";

        using Kernel = approach3::Kernel;
        const index_t block_size = Kernel::BlockSize();
        const index_t grid_size  = (M + Shape2::Block_M - 1) / Shape2::Block_M;

        auto input_shape   = make_tuple(M, N);
        auto input_strides = make_tuple(N, static_cast<index_t>(1));

        stream_config stream;
        constexpr int kWarmup = 5;
        constexpr int kRepeat = 20;

        auto run = [&]() {
            launch_kernel(
                stream,
                make_kernel<1>(Kernel{}, dim3(grid_size), dim3(block_size), 0,
                               x_dev, y_dev, input_shape, input_strides));
        };

        for(int i = 0; i < kWarmup; ++i) run();
        hip_check_error(hipDeviceSynchronize());

        hipEvent_t ev0, ev1;
        hip_check_error(hipEventCreate(&ev0));
        hip_check_error(hipEventCreate(&ev1));
        hip_check_error(hipEventRecord(ev0));
        for(int i = 0; i < kRepeat; ++i) run();
        hip_check_error(hipEventRecord(ev1));
        hip_check_error(hipEventSynchronize(ev1));

        float ms = 0;
        hip_check_error(hipEventElapsedTime(&ms, ev0, ev1));
        double avg = ms / kRepeat;
        hip_check_error(hipEventDestroy(ev0));
        hip_check_error(hipEventDestroy(ev1));

        double bytes = static_cast<double>(M) * N * sizeof(float)
                     + static_cast<double>(M) * sizeof(float);
        double gbps  = bytes / 1e9 / (avg / 1e3);

        hip_check_error(hipMemcpy(h_y.data(), y_dev,
                                  M * sizeof(float), hipMemcpyDeviceToHost));
        bool ok = verify(h_y_max_ref, h_y, 1e-5f);
        std::cout << "  " << (ok ? "PASSED" : "FAILED")
                  << "  |  " << avg << " ms  |  " << gbps << " GB/s\n\n";
    }

    // ---- Approach 4: Cross-warp row-sum ----
    run_custom_reduce(
        "Approach 4: Cross-warp row-sum (ReduceOp::Add, same 3-stage hierarchy)",
        Approach4_BlockSum{}, x_dev, y_dev, h_y_sum_ref, h_y, M, N, 0.05f);

    std::cout << "============================================\n"
              << "Summary:\n"
              << "  1. Warp-only max     — BlockReduce2d + BlockReduce2dSync\n"
              << "  2. Cross-warp max    — + BlockReduce2dCrossWarpSync (LDS)\n"
              << "  3. ReduceKernel max  — host-only, wraps all stages\n"
              << "  4. Cross-warp sum    — same as 2 with ReduceOp::Add\n"
              << "============================================\n\n";

    return 0;
}
