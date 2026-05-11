// SPDX-License-Identifier: MIT
// Copyright (c) 2024, Advanced Micro Devices, Inc. All rights reserved.

/*
 * Tutorial 14.20: static_distributed_tensor — what it is + how you use it
 *
 * One-liner:
 *   static_distributed_tensor<DataType, StaticTileDistribution> is a
 *   per-thread register tile. Hardware storage is a thread_buffer<T, N>
 *   (same VGPR pool as 14.15 part B). The "distribution" half is the
 *   compile-time map from logical (M, N) tile coordinates to (thread_id,
 *   register_slot). Together: a register file that knows where each
 *   element lives in the warp.
 *
 *   tile_distribution_encoding<...>      // compile-time mapping
 *      |
 *   make_static_tile_distribution(enc)   // -> StaticTileDistribution
 *      |
 *   make_static_distributed_tensor<T>(d) // -> a per-thread register tile
 *
 * Same tiny distribution as 14.12 so you can compare side-by-side:
 *   tile = M=8 rows x N=64 cols of float
 *   1 warp = 64 lanes (Lane_M=4, Lane_N=16)
 *   per-thread Y = (Repeat_M=2, Vector_M=1, Repeat_N=1, Vector_N=4)
 *                = 8 floats per thread => 8 VGPRs per thread for this tile
 *   spans: span0 (M).Impl = sequence<2,1>   (R_M, V_M)
 *          span1 (N).Impl = sequence<1,4>   (R_N, V_N)
 *
 * Sections (each prints "[k]" so you can grep them):
 *   [1] Static accounting   : NDimX, lengths, spans, per-thread buffer size.
 *   [2] Construction        : declare DTens t; vs make_static_distributed_tensor.
 *                             Show that get_thread_buffer() == thread_buffer.
 *   [3] Element access      : t(idx) writer, t[idx] reader inside sweep_tile_span.
 *                             Map idx -> tile-space (M,N) coord.
 *   [4] Bulk init helpers   : clear_tile, set_tile, t.initialize, raw thread_buf_.
 *   [5] tile_elementwise_*  : in-place inout vs in-producing-out.
 *   [6] set_tile_if         : value <- v iff predicate(x_indices) holds.
 *   [7] Y-slice get/set     : work on a contiguous chunk of the per-thread buffer
 *                             without touching the rest.
 *   [8] block_tile_reduce   : produce a NEW static_distributed_tensor whose
 *                             reduced X-dim is collapsed (in-thread + warp shuffle).
 *   [9] load_tile           : populate the register file from a real HBM buffer
 *                             of "increasing numbers" so you can see exactly
 *                             which 8 floats end up in which lane's slots.
 *
 * Build:
 *   target is aa_tutorial_14_20_static_distributed_tensor_intro
 *
 * NOTE: this single-kernel tutorial uses a lot of templates simultaneously,
 * so under build-debug (-O0 -fno-inline -mcmodel=large) the kernel saturates
 * gfx950's SGPR cap and exceeds the per-wave scratch budget; the runtime
 * will refuse to launch it. Build it from the regular `build/` (optimized)
 * tree to see the output. For a tiny load_tile + distribution example that
 * runs cleanly under build-debug (and is meant for debugger inspection of
 * the per-lane register slots), see tutorial 14.21.
 */

#include "ck_tile/core.hpp"
#include "ck_tile/host.hpp"
#include "ck_tile/ops/reduce.hpp"

#include <cstdio>
#include <cfloat>
#include <vector>
#include <hip/hip_runtime.h>

using namespace ck_tile;

// --------------------------------------------------------------------------
// Distribution: same shape as 14.12 — easy side-by-side reading.
// --------------------------------------------------------------------------
CK_TILE_HOST_DEVICE constexpr auto make_tiny_distribution()
{
    constexpr index_t R_M = 2, W_M = 1, T_M = 4,  V_M = 1;
    constexpr index_t R_N = 1, W_N = 1, T_N = 16, V_N = 4;
    static_assert(T_M * T_N == 64, "wave64");
    static_assert(W_M * W_N == 1, "1-warp toy block");

    return make_static_tile_distribution(
        tile_distribution_encoding<
            sequence<>,
            tuple<sequence<R_M, W_M, T_M, V_M>,
                  sequence<R_N, W_N, T_N, V_N>>,
            tuple<sequence<1, 2>, sequence<1, 2>>,
            tuple<sequence<1, 1>, sequence<2, 2>>,
            sequence<1, 1, 2, 2>,
            sequence<0, 3, 0, 3>>{});
}

using Dist  = decltype(make_tiny_distribution());
using DTens = static_distributed_tensor<float, Dist>;

// 8 floats per thread; small fixed-size staging arrays for runtime printing
// outside template-heavy sweeps (same pattern as 14.12).
static constexpr index_t kPerThread = 8;

struct Row
{
    int   di0a, di0b;   // X0 (M) distributed index components (R_M, V_M)
    int   di1a, di1b;   // X1 (N) distributed index components (R_N, V_N)
    int   xm,  xn;      // tile-space (M, N) coord
    float v;            // value held by this thread for that idx
};

// Helper: sweep_tile and dump everything into rows[0..kPerThread).
//
// idx is a tuple<tile_distributed_index<R_M, V_M>, tile_distributed_index<R_N, V_N>>.
// Real pipelines (fmha / softmax / rmsnorm) NEVER look inside it -- they just
// forward the index back into the tensor: `tensor(make_tuple(idx0, idx1))`.
// We only crack it open here because we want to PRINT each step.
//
// tile_distributed_index<I...> stores its values in two equivalent places:
//   using Impl              = sequence<I...>;   // the type
//   static constexpr auto impl_ = Impl{};       // a constexpr instance of it
// Going through `impl_` is cleaner than `typename decltype(...)::Impl` -- no
// decltype, no typename, no extra `{}`. sequence<...>::operator[](number<i>{})
// returns a number<X> that implicitly converts to int.
template <typename T>
CK_TILE_DEVICE void snapshot(const T& t, Row (&rows)[kPerThread])
{
    int seq = 0;
    sweep_tile(t, [&](auto idx) {
        constexpr auto i0 = idx[number<0>{}].impl_;   // sequence<R_M_step, V_M_step>
        constexpr auto i1 = idx[number<1>{}].impl_;   // sequence<R_N_step, V_N_step>
        const auto     x  = get_x_indices_from_distributed_indices(Dist{}, idx);
        rows[seq] = Row{
            static_cast<int>(i0[number<0>{}]),
            static_cast<int>(i0[number<1>{}]),
            static_cast<int>(i1[number<0>{}]),
            static_cast<int>(i1[number<1>{}]),
            static_cast<int>(x[number<0>{}]),
            static_cast<int>(x[number<1>{}]),
            t[idx],
        };
        ++seq;
    });
}

CK_TILE_DEVICE void print_rows(const char* tag, const Row (&rows)[kPerThread])
{
    printf("    %s\n", tag);
    for(int i = 0; i < kPerThread; ++i)
    {
        const auto& r = rows[i];
        printf("      seq=%d  idx=(di<%d,%d>, di<%d,%d>)  (m,n)=(%d,%d)  v=%g\n",
               i, r.di0a, r.di0b, r.di1a, r.di1b, r.xm, r.xn,
               static_cast<double>(r.v));
    }
}

// Tile shape used in section [9]: an [M=8, N=64] flat buffer in HBM whose
// element at (m, n) is just (m*64 + n). Lets us read each per-thread
// register slot and recognize where it came from in the source tile.
static constexpr index_t kTileM = 8;
static constexpr index_t kTileN = 64;

// --------------------------------------------------------------------------
// kernel: 1 block * 64 threads = 1 warp. Lane 0 prints; all lanes execute
// the sweeps so partition_index() (lane id) is well-defined.
// --------------------------------------------------------------------------
__global__ void sdt_intro_kernel(const float* __restrict__ src)
{
    const bool dbg = (threadIdx.x == 0);

    if(dbg) printf("=== Tutorial 14.20: static_distributed_tensor intro ===\n\n");

    // --------------------------------------------------------------- [1]
    if(dbg)
    {
        constexpr auto lengths = DTens::get_lengths();
        constexpr auto spans   = DTens::get_distributed_spans();
        // tile_distributed_span has the same Impl/impl_ pair as
        // tile_distributed_index. Same .impl_ trick avoids the
        // typename + decltype dance.
        constexpr auto s0 = spans[number<0>{}].impl_;   // sequence<R_M, V_M>
        constexpr auto s1 = spans[number<1>{}].impl_;   // sequence<R_N, V_N>
        printf("[1] static accounting (compile time)\n");
        printf("    NDimX                       = %d\n",
               static_cast<int>(DTens::get_num_of_dimension()));
        printf("    get_lengths()               = (%d, %d)   // (M, N)\n",
               static_cast<int>(lengths[number<0>{}]),
               static_cast<int>(lengths[number<1>{}]));
        printf("    span0.impl_ (M = R_M,V_M)   = sequence<%d,%d>\n",
               static_cast<int>(s0[number<0>{}]),
               static_cast<int>(s0[number<1>{}]));
        printf("    span1.impl_ (N = R_N,V_N)   = sequence<%d,%d>\n",
               static_cast<int>(s1[number<0>{}]),
               static_cast<int>(s1[number<1>{}]));
        printf("    get_thread_buffer_size()    = %d   // VGPR slots per thread\n",
               static_cast<int>(DTens::get_thread_buffer_size()));
        printf("    sizeof(DTens)               = %zu B  // == thread_buffer<float,8>\n\n",
               sizeof(DTens));
    }

    // --------------------------------------------------------------- [2]
    // Two equivalent constructors. The factory form is more common because
    // you usually have the distribution object on hand from a Policy::...
    DTens t1;                                                           // default ctor
    auto  t2 = make_static_distributed_tensor<float>(make_tiny_distribution());
    static_assert(std::is_same_v<decltype(t2), DTens>,
                  "factory and direct types must match");

    // The underlying register file is a thread_buffer (see 14.15 part B).
    using BufT = std::remove_reference_t<decltype(t1.get_thread_buffer())>;
    static_assert(BufT::size() == kPerThread, "8 VGPR slots per thread");

    if(dbg)
    {
        printf("[2] construction\n");
        printf("    DTens t1;                            // default-constructed register file\n");
        printf("    auto  t2 = make_static_distributed_tensor<float>(dist);\n");
        printf("    decltype(t1)::DataType       == float\n");
        printf("    t1.get_thread_buffer().size()= %d   // == get_thread_buffer_size()\n\n",
               static_cast<int>(BufT::size()));
    }

    // --------------------------------------------------------------- [3]
    // Writer / reader through distributed indices.
    sweep_tile(t1, [&](auto idx) {
        const auto x = get_x_indices_from_distributed_indices(Dist{}, idx);
        // Encode (m, n) so we can recognize each cell when we print it.
        t1(idx) = static_cast<float>(100 * x[number<0>{}] + x[number<1>{}]);
    });

    Row rows3[kPerThread] = {};
    snapshot(t1, rows3);

    if(dbg)
    {
        printf("[3] element access via t(idx) / t[idx]   (lane 0's view)\n");
        printf("    (filled with 100*m + n; lane 0 owns m=0 and m=4 in this dist)\n");
        print_rows("after t(idx) = 100*m + n:", rows3);
        printf("\n");
    }

    // --------------------------------------------------------------- [4]
    // Three bulk-init knobs + the raw thread-buffer escape hatch.
    //
    // NOTE: static_distributed_tensor::initialize(x) exists in the header
    // but forwards to thread_buf_.initialize(x), which doesn't exist on
    // thread_buffer<T, N> -- that overload is dead code. Use either
    //   set_tile(t, v)                               // documented path, or
    //   t.get_thread_buffer() = thread_buffer<...>{v}; // broadcast ctor.
    DTens a;
    clear_tile(a);                                       // all zeros
    DTens b;
    set_tile(b, 7.0f);                                   // set every slot to 7
    DTens c;
    c.get_thread_buffer() = thread_buffer<float, kPerThread>{3.5f};  // broadcast ctor
    DTens d;                                             // poke storage directly
    d.get_thread_buffer()(number<0>{}) = 99.0f;
    d.get_thread_buffer()(number<1>{}) = 0.0f;

    if(dbg)
    {
        printf("[4] bulk init helpers (lane 0 prints first 3 thread_buf_ slots)\n");
        printf("    clear_tile(a)                                -> %g %g %g\n",
               static_cast<double>(a.get_thread_buffer()[number<0>{}]),
               static_cast<double>(a.get_thread_buffer()[number<1>{}]),
               static_cast<double>(a.get_thread_buffer()[number<2>{}]));
        printf("    set_tile(b, 7.0f)                            -> %g %g %g\n",
               static_cast<double>(b.get_thread_buffer()[number<0>{}]),
               static_cast<double>(b.get_thread_buffer()[number<1>{}]),
               static_cast<double>(b.get_thread_buffer()[number<2>{}]));
        printf("    c.get_thread_buffer() = thread_buffer<.>{3.5}-> %g %g %g\n",
               static_cast<double>(c.get_thread_buffer()[number<0>{}]),
               static_cast<double>(c.get_thread_buffer()[number<1>{}]),
               static_cast<double>(c.get_thread_buffer()[number<2>{}]));
        printf("    d.get_thread_buffer()(number<0>{}) = 99      -> %g %g %g\n\n",
               static_cast<double>(d.get_thread_buffer()[number<0>{}]),
               static_cast<double>(d.get_thread_buffer()[number<1>{}]),
               static_cast<double>(d.get_thread_buffer()[number<2>{}]));
    }

    // --------------------------------------------------------------- [5]
    // Element-wise: `inout` mutates in place; `in` builds a new tensor.
    DTens e = t1;                                         // copy
    tile_elementwise_inout([](auto& x) { x *= 2.0f; }, e);

    auto f = tile_elementwise_in([](auto x) { return x + 0.5f; }, t1);
    static_assert(std::is_same_v<decltype(f), DTens>,
                  "tile_elementwise_in returns a tile of the same shape/dist");

    if(dbg)
    {
        printf("[5] tile_elementwise_inout / tile_elementwise_in\n");
        printf("    e = t1; tile_elementwise_inout(*=2, e):\n");
        printf("      e.thread_buf_[0..2] = %g %g %g  (== 2 * t1[0..2] which was %g %g %g)\n",
               static_cast<double>(e.get_thread_buffer()[number<0>{}]),
               static_cast<double>(e.get_thread_buffer()[number<1>{}]),
               static_cast<double>(e.get_thread_buffer()[number<2>{}]),
               static_cast<double>(t1.get_thread_buffer()[number<0>{}]),
               static_cast<double>(t1.get_thread_buffer()[number<1>{}]),
               static_cast<double>(t1.get_thread_buffer()[number<2>{}]));
        printf("    auto f = tile_elementwise_in(+0.5, t1):\n");
        printf("      f.thread_buf_[0..2] = %g %g %g\n\n",
               static_cast<double>(f.get_thread_buffer()[number<0>{}]),
               static_cast<double>(f.get_thread_buffer()[number<1>{}]),
               static_cast<double>(f.get_thread_buffer()[number<2>{}]));
    }

    // --------------------------------------------------------------- [6]
    // set_tile_if: predicate runs on the (M, N) coordinate, NOT on the
    // distributed index. Used in fmha for masking edge tiles.
    DTens g;
    clear_tile(g);
    set_tile_if(g, 9.0f, [](auto xidx) {
        return xidx[number<0>{}] == 0 && xidx[number<1>{}] < 8;   // top-left 1x8 block
    });

    Row rows6[kPerThread] = {};
    snapshot(g, rows6);
    if(dbg)
    {
        printf("[6] set_tile_if: g <- 9 if m==0 && n<8, else 0\n");
        print_rows("after set_tile_if:", rows6);
        printf("\n");
    }

    // --------------------------------------------------------------- [7]
    // get_y_sliced_thread_data / set_y_sliced_thread_data
    //
    // Per-thread Y block lengths = (R_M, V_M, R_N, V_N) = (2, 1, 1, 4) = 8.
    // Take the first (1, 1, 1, 4) chunk = 4 floats starting at Y origin
    // (0,0,0,0). For lane 0 those are the 4 N-vector lanes of M-row 0.
    DTens h;
    sweep_tile(h, [&](auto idx) {
        const auto x = get_x_indices_from_distributed_indices(Dist{}, idx);
        h(idx) = static_cast<float>(1000 + 10 * x[number<0>{}] + x[number<1>{}]);
    });

    constexpr auto y_origin  = sequence<0, 0, 0, 0>{};
    constexpr auto y_lengths = sequence<1, 1, 1, 4>{};
    auto slice = h.get_y_sliced_thread_data(y_origin, y_lengths);  // thread_buffer<float,4>

    // Mutate the slice and write it back. Only the first 4 of 8 slots change.
    static_for<0, 4, 1>{}([&](auto i) { slice(i) = -static_cast<float>(i + 1); });
    h.set_y_sliced_thread_data(y_origin, y_lengths, slice);

    if(dbg)
    {
        printf("[7] get_y_sliced_thread_data / set_y_sliced_thread_data\n");
        printf("    slice (first 4 of 8 slots) was filled with 1000+10m+n; replaced with -1..-4\n");
        printf("    h.thread_buf_[0..7] = %g %g %g %g | %g %g %g %g\n",
               static_cast<double>(h.get_thread_buffer()[number<0>{}]),
               static_cast<double>(h.get_thread_buffer()[number<1>{}]),
               static_cast<double>(h.get_thread_buffer()[number<2>{}]),
               static_cast<double>(h.get_thread_buffer()[number<3>{}]),
               static_cast<double>(h.get_thread_buffer()[number<4>{}]),
               static_cast<double>(h.get_thread_buffer()[number<5>{}]),
               static_cast<double>(h.get_thread_buffer()[number<6>{}]),
               static_cast<double>(h.get_thread_buffer()[number<7>{}]));
        printf("    (the second half is untouched, still 1000+10m+n for that slot)\n\n");
    }

    // --------------------------------------------------------------- [8]
    // block_tile_reduce: row-max along X1 (N). Returns a NEW tensor whose
    // type is "the same distribution, with X1 replicated", i.e. only the M
    // axis remains. The result is broadcast across the lanes that used to
    // own different N columns, so every lane in a row reads the same value.
    //
    // Step 1: in-thread reduction (each thread reduces its 4 N values).
    // Step 2: cross-lane warp shuffle to combine across the 16 N-lanes.
    DTens t_full;
    sweep_tile(t_full, [&](auto idx) {
        const auto x = get_x_indices_from_distributed_indices(Dist{}, idx);
        t_full(idx) = static_cast<float>(100 * x[number<0>{}] + x[number<1>{}]);
    });

    auto f_max  = [](auto a_, auto b_) { return max(a_, b_); };
    auto rowmax = block_tile_reduce<float>(t_full, sequence<1>{}, f_max, -FLT_MAX);
    block_tile_reduce_sync(rowmax, f_max);  // warp shuffle across N-lanes

    if(dbg)
    {
        using RT = decltype(rowmax);
        printf("[8] block_tile_reduce<float>(t_full, sequence<1>{}, max, -FLT_MAX)\n");
        printf("    decltype(rowmax)::get_thread_buffer_size() = %d  (was 8 before reduce)\n",
               static_cast<int>(RT::get_thread_buffer_size()));
        printf("    Each row's max should be 100*m + 63.\n");
        printf("    lane 0 owns m=0 and m=4 -> rowmax[0]=%g  rowmax[1]=%g  (expect 63 and 463)\n\n",
               static_cast<double>(rowmax.get_thread_buffer()[number<0>{}]),
               static_cast<double>(rowmax.get_thread_buffer()[number<1>{}]));
    }

    // --------------------------------------------------------------- [9]
    // load_tile from a real global buffer of "increasing numbers".
    //
    // src is a flat [M=8, N=64] row-major float array with src[m, n] = m*64 + n,
    // so values run 0, 1, 2, ..., 511 in memory order. After load_tile, each
    // thread holds 8 of those 512 floats. The mapping (which slot <- which
    // tensor coord) is fixed by the distribution:
    //
    //   slot k=0..3:  m = lane_M,      n = 4*lane_N + k       (R_M=0 half)
    //   slot k=4..7:  m = lane_M + 4,  n = 4*lane_N + (k-4)   (R_M=1 half)
    //
    //   lane_M = lid / 16,  lane_N = lid % 16
    //
    // Since the source is row-major increasing, value(m, n) = m*64 + n, so:
    //   slot k=0..3:  64*lane_M       + 4*lane_N + k
    //   slot k=4..7:  64*(lane_M + 4) + 4*lane_N + (k - 4)
    //                       = 64*lane_M + 4*lane_N + (k - 4) + 256
    //
    // Therefore inside one thread:
    //   slots[0..3] are 4 contiguous floats (== one buffer_load_dwordx4)
    //   slots[4..7] are 4 contiguous floats 256 elements later (one more dwordx4)
    auto src_view = make_naive_tensor_view<address_space_enum::global>(
        src,
        make_tuple(kTileM, kTileN),
        make_tuple(kTileN, 1),
        number<1>{},   // inner vector dim = N
        number<4>{});  // V_N = 4 -> dwordx4 friendly

    constexpr auto dist = make_tiny_distribution();
    auto src_window     = make_tile_window(
        src_view,
        make_tuple(number<kTileM>{}, number<kTileN>{}),
        multi_index<2>{0, 0},
        dist);

    auto loaded = load_tile(src_window);  // <- this is the line that fills 8 VGPRs / lane

    // Stage all 8 slots for several interesting lanes into shared memory so
    // we can print them outside any compile-time expansion.
    __shared__ float lane_slots[7][kPerThread];
    static constexpr int probe_lanes[7] = {0, 1, 15, 16, 32, 48, 63};
    for(int p = 0; p < 7; ++p)
    {
        if(static_cast<int>(threadIdx.x) == probe_lanes[p])
        {
            static_for<0, kPerThread, 1>{}([&](auto i) {
                lane_slots[p][i] = loaded.get_thread_buffer()[number<i>{}];
            });
        }
    }
    __syncthreads();

    if(dbg)
    {
        printf("[9] load_tile from src[m, n] = m*64 + n  (values 0..511 in memory)\n");
        printf("    Per-thread slot mapping for the toy distribution:\n");
        printf("      slot k=0..3:  m = lane_M,     n = 4*lane_N + k\n");
        printf("      slot k=4..7:  m = lane_M + 4, n = 4*lane_N + (k-4)\n");
        printf("    => slots 0..3 are 4 contiguous floats; slots 4..7 are\n");
        printf("       4 more contiguous floats 256 elements later.\n");
        printf("    expected vs. actual register file (one row per probe lane):\n");
        printf("      lane | (lane_M, lane_N) | slots[0..7]\n");
        for(int p = 0; p < 7; ++p)
        {
            const int lid    = probe_lanes[p];
            const int lane_M = lid / 16;
            const int lane_N = lid % 16;
            printf("      %4d |   (%d, %2d)        | "
                   "%6.0f %6.0f %6.0f %6.0f  %6.0f %6.0f %6.0f %6.0f\n",
                   lid, lane_M, lane_N,
                   static_cast<double>(lane_slots[p][0]),
                   static_cast<double>(lane_slots[p][1]),
                   static_cast<double>(lane_slots[p][2]),
                   static_cast<double>(lane_slots[p][3]),
                   static_cast<double>(lane_slots[p][4]),
                   static_cast<double>(lane_slots[p][5]),
                   static_cast<double>(lane_slots[p][6]),
                   static_cast<double>(lane_slots[p][7]));
        }
    }
}

int main()
{
    printf("=== Tutorial 14.20: launching 1 warp (64 lanes), lane 0 prints ===\n\n");

    // Source for section [9]: M=8, N=64 floats with value(m, n) = m*64 + n.
    std::vector<float> h_src(kTileM * kTileN);
    for(index_t m = 0; m < kTileM; ++m)
        for(index_t n = 0; n < kTileN; ++n)
            h_src[m * kTileN + n] = static_cast<float>(m * kTileN + n);
    DeviceMem d_src(kTileM * kTileN * sizeof(float));
    d_src.ToDevice(h_src.data(), kTileM * kTileN * sizeof(float));

    hipLaunchKernelGGL(sdt_intro_kernel, dim3(1), dim3(64), 0, nullptr,
                       static_cast<const float*>(d_src.GetDeviceBuffer()));
    auto err = hipDeviceSynchronize();
    if(err != hipSuccess)
    {
        fprintf(stderr, "hipDeviceSynchronize failed: %s\n", hipGetErrorString(err));
        return 1;
    }
    return 0;
}
