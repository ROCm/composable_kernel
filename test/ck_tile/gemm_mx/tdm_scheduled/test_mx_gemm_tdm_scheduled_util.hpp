// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
#pragma once

// Shared by the comp_tdm_scheduled GPU tests: instance aliases, the sparse check inputs, a
// device run and the bitwise check against reference_mx_gemm_tdm_scheduled.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <set>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

#include <gtest/gtest.h>

#include "ck_tile/core.hpp"
#include "ck_tile/host.hpp"
#include "ck_tile/host/reference/reference_mx_gemm_tdm_scheduled.hpp"
#include "ck_tile/ops/epilogue.hpp"
#include "ck_tile/ops/gemm.hpp"

namespace mx_tdm_scheduled_test {

using ck_tile::index_t;
using ck_tile::long_index_t;
namespace sched = ck_tile::mx_tdm_scheduled;

/// A comp_tdm_scheduled pipeline: A, B and E types, tile, wave group, WMMA atom (32 or 16 rows),
/// tuning (a Tu) and the shape traits (edge M/N, wide A/B/E, tail steps).
template <typename A,
          typename B,
          typename O,
          index_t TileM,
          index_t TileN,
          index_t TileK,
          index_t WarpM,
          index_t WarpN,
          index_t AtomM,
          typename Tuning,
          typename Traits>
using Pipeline = ck_tile::MxGemmPipelineAgBgCrCompTDMScheduled<
    ck_tile::MxGemmTdmScheduledProblem<
        A,
        B,
        O,
        ck_tile::ClusterTileGemmShape<typename Tuning::Cluster,
                                      ck_tile::sequence<TileM, TileN, TileK>,
                                      ck_tile::sequence<WarpM, WarpN, 1>,
                                      ck_tile::sequence<AtomM, 16, 128>>,
        Traits>,
    typename Tuning::Policy>;

/// The kernel of a comp_tdm_scheduled pipeline: its N-major partitioner and TdmEpilogue (E
/// through LDS).
template <typename P>
using KernelOf =
    ck_tile::MxGemmKernel<ck_tile::GemmTile2DNMajorPartitioner<typename P::BlockGemmShape>,
                          P,
                          ck_tile::MxTdmStoreEpilogue<P>>;

/// Kernel with its pipeline's Problem::TailNum T. TdmEpilogue does not depend on the pipeline's
/// TailNum.
template <typename Kernel, ck_tile::TailNumber T>
struct WithTailNum;
template <typename Partitioner, typename P, typename Epilogue, ck_tile::TailNumber T>
struct WithTailNum<ck_tile::MxGemmKernel<Partitioner, P, Epilogue>, T>
{
    using type = ck_tile::MxGemmKernel<Partitioner, typename P::template WithTailNum<T>, Epilogue>;
};

/// Calls f(WithTailNum<Kernel, T>{}) with T the TailNum of K: the pipeline's TailHandler picks
/// TailNumber::One or Two from the whole-tile count on the host.
template <typename Kernel, typename F>
auto for_tail_num(index_t K, F&& f)
{
    using P                  = typename Kernel::GemmPipeline;
    const index_t full_tiles = K / P::kTileK;
    return P::TailHandler(
        [&](auto, auto tail_num) { return f(WithTailNum<Kernel, decltype(tail_num)::value>{}); },
        P::BlockHasHotloop(full_tiles),
        P::GetBlockLoopTailNum(full_tiles));
}

/// A comp_tdm_scheduled instance, named by its TailNumber::Full kernel: check runs the kernel of
/// each problem's TailNum (for_tail_num).
template <typename A,
          typename B,
          typename O,
          index_t TileM,
          index_t TileN,
          index_t TileK,
          index_t WarpM,
          index_t WarpN,
          index_t AtomM,
          typename Tuning,
          typename Traits>
using Kernel =
    KernelOf<Pipeline<A, B, O, TileM, TileN, TileK, WarpM, WarpN, AtomM, Tuning, Traits>>;

/// A tuning: the cluster (CX N tiles by CY M tiles), then the policy's knobs with its defaults:
/// GL2 prefetch distance, PrefetchLdsRead, SplitBLdsRead, InterleaveLdsRegions.
template <index_t CX,
          index_t CY,
          index_t Gl2      = 4,
          bool LdsPrefetch = true,
          bool SplitB      = true,
          bool Interleave  = true>
struct Tu
{
    using Cluster = ck_tile::sequence<CY, CX, 1>;
    using Policy =
        ck_tile::MxGemmPipelineAgBgCrCompTDMScheduledPolicy<Gl2, LdsPrefetch, SplitB, Interleave>;
};

template <bool EdgeM, bool EdgeN, index_t TailSteps>
using Shape = ck_tile::MxTdmScheduledGemmTraits<EdgeM, EdgeN, false, false, false, TailSteps>;

/// The sparse check: A and B nonzero only in a few 128-K steps
/// (the first and last, both sides of the first and last tile boundaries, then random ones up to
/// 12), FP4 all 16 codes with E8M0 2^-2..2^1, 8-bit +-(1 + m / 2^bits) * 2^e, e in {-1, 0}, with
/// E8M0 2^-1..2^0. An alpha of 2^-6 is folded into the scales, 2^-3 on each operand
/// (kSparseScaleShift), which keeps E in FP8 range. Every FP32 partial sum is exact, so E is
/// checked bit for bit at any K, and at any scale shift that keeps the sums finite and normal.
struct Inputs
{
    std::vector<uint8_t> a;       // M rows of row_bytes<A>(K)
    std::vector<uint8_t> b;       // N rows
    std::vector<uint8_t> a_scale; // K-major
    std::vector<uint8_t> b_scale; // K-major
};

/// The E8M0 exponent offset of each operand's scales in the sparse check: alpha 2^-6.
inline constexpr int kSparseScaleShift = -3;

inline std::vector<long_index_t> sparse_steps(long_index_t K, index_t tile_k, std::mt19937& rng)
{
    const long_index_t steps     = (K + 127) / 128;
    const long_index_t per_tile  = tile_k / 128;
    const long_index_t last_tile = (steps - 1) / per_tile * per_tile;
    std::set<long_index_t> chosen;
    for(long_index_t s :
        {long_index_t{0}, steps - 1, per_tile - 1, per_tile, last_tile - 1, last_tile})
        if(s >= 0 && s < steps)
            chosen.insert(s);
    std::vector<long_index_t> rest;
    for(long_index_t s = 0; s < steps; ++s)
        if(!chosen.count(s))
            rest.push_back(s);
    std::shuffle(rest.begin(), rest.end(), rng);
    for(std::size_t i = 0; chosen.size() < 12 && i < rest.size(); ++i)
        chosen.insert(rest[i]);
    return {chosen.begin(), chosen.end()};
}

template <typename T>
uint8_t sparse_code(std::mt19937& rng)
{
    if constexpr(std::is_same_v<T, ck_tile::pk_fp4_t>)
        return static_cast<uint8_t>(rng() % 16);
    const int bits     = std::is_same_v<T, ck_tile::fp8_t> ? 3 : 2;
    const int bias     = std::is_same_v<T, ck_tile::fp8_t> ? 7 : 15;
    const int mantissa = static_cast<int>(rng() % (1u << bits));
    const int exponent = bias - static_cast<int>(rng() % 2); // 2^0 or 2^-1
    const int sign     = static_cast<int>(rng() % 2);
    return static_cast<uint8_t>((sign << 7) | (exponent << bits) | mantissa);
}

/// The sparse inputs of A and B types A and B (pk_fp4_t, fp8_t or bf8_t).
template <typename A, typename B>
Inputs make_inputs(index_t M,
                   index_t N,
                   long_index_t K,
                   index_t tile_k,
                   int scale_shift = kSparseScaleShift,
                   uint32_t seed   = 0x5BA6)
{
    std::mt19937 rng(seed);
    const auto steps = sparse_steps(K, tile_k, rng);
    Inputs in;
    auto operand = [&](auto type, index_t rows) {
        using T                      = decltype(type);
        const long_index_t row_bytes = sched::row_bytes<T>(K);
        std::vector<uint8_t> data(static_cast<std::size_t>(rows) * row_bytes, 0);
        for(index_t r = 0; r < rows; ++r)
            for(long_index_t s : steps)
                for(long_index_t k = s * 128; k < std::min(K, (s + 1) * 128); ++k)
                {
                    const uint8_t code = sparse_code<T>(rng);
                    uint8_t* row       = data.data() + r * row_bytes;
                    if constexpr(std::is_same_v<T, ck_tile::pk_fp4_t>)
                        row[k / 2] |= static_cast<uint8_t>(k % 2 ? code << 4 : code);
                    else
                        row[k] = code;
                }
        return data;
    };
    in.a                      = operand(A{}, M);
    in.b                      = operand(B{}, N);
    const long_index_t groups = (K + 31) / 32;
    const bool fp4            = std::is_same_v<A, ck_tile::pk_fp4_t>;
    auto scales               = [&](index_t rows) {
        std::vector<uint8_t> natural(static_cast<std::size_t>(rows) * groups);
        for(auto& x : natural)
            x = static_cast<uint8_t>(scale_shift + (fp4 ? 125 + rng() % 4 : 126 + rng() % 2));
        return ck_tile::mx_scale_to_kmajor(natural.data(), rows, K, groups);
    };
    in.a_scale = scales(M);
    in.b_scale = scales(N);
    return in;
}

/// The packed kernel arguments of a comp_tdm_scheduled run (k_batch 1, strides K for A and B, N for
/// E).
template <typename Kernel>
typename Kernel::KernelArgs kernel_args(const void* a,
                                        const void* b,
                                        const void* a_scale,
                                        const void* b_scale,
                                        void* e,
                                        index_t M,
                                        index_t N,
                                        index_t K)
{
    return Kernel::MakeKernelArgs(ck_tile::MxGemmHostArgs<>{
        {a}, {a_scale}, {b}, {b_scale}, {}, e, 1, M, N, K, {K}, {K}, {}, N});
}

/// The sparse inputs of a kernel's A and B types and K tile.
template <typename Kernel>
Inputs make_kernel_inputs(index_t M, index_t N, long_index_t K, int scale_shift = kSparseScaleShift)
{
    using Pipeline = typename Kernel::GemmPipeline;
    return make_inputs<typename Pipeline::ADataType, typename Pipeline::BDataType>(
        M, N, K, Pipeline::kTileK, scale_shift);
}

/// Launches Kernel: the grid of GridSize, and with a cluster tuning the cluster of ClusterSize.
template <typename Kernel>
float launch(const ck_tile::stream_config& s, const typename Kernel::KernelArgs& kargs)
{
    using Pipeline = typename Kernel::MxGemmPipeline;
    static_assert(Kernel::ClusterLaunch == Pipeline::kHasCluster &&
                      Kernel::ClusterSize().x == Pipeline::kClusterX &&
                      Kernel::ClusterSize().y == Pipeline::kClusterY &&
                      Kernel::ClusterSize().z == 1,
                  "the launch's cluster is the pipeline's: N tiles along x, M tiles along y");
    if(!Kernel::IsSupportedArgument(kargs))
    {
        const char* why = Kernel::GetUnsupportedReason(kargs);
        throw std::runtime_error(Kernel::GetName() + ": " + (why ? why : "see CK_TILE_LOGGING"));
    }
    const dim3 grids  = Kernel::GridSize(kargs.M, kargs.N, kargs.k_batch);
    const dim3 blocks = Kernel::BlockSize();
    if constexpr(Kernel::ClusterLaunch)
    {
#if CK_TILE_ENABLE_CLUSTER_LAUNCH
        return ck_tile::launch_kernel(
            s, ck_tile::make_kernel<1>(Kernel{}, Kernel::ClusterSize(), grids, blocks, 0, kargs));
#else
        throw std::runtime_error(Kernel::GetName() + ": this build has no cluster launch");
#endif
    }
    else
    {
        return ck_tile::launch_kernel(s,
                                      ck_tile::make_kernel<1>(Kernel{}, grids, blocks, 0, kargs));
    }
}

/// E of Kernel on `in`; nothing in the dry run, which only checks that the instance runs the
/// problem.
template <typename Kernel>
std::vector<uint8_t> run_device(const Inputs& in, index_t M, index_t N, index_t K)
{
    const std::size_t e_bytes = static_cast<std::size_t>(M) * N *
                                sched::out_bytes<typename Kernel::GemmPipeline::ODataType>();
#ifdef MX_TDM_SCHEDULED_DRY_RUN
    // CPU-only build: check that the instance runs the problem, launch nothing.
    ck_tile::ignore  = e_bytes;
    const auto probe = kernel_args<Kernel>(in.a.data(),
                                           in.b.data(),
                                           in.a_scale.data(),
                                           in.b_scale.data(),
                                           const_cast<uint8_t*>(in.a.data()),
                                           M,
                                           N,
                                           K);
    const char* why  = Kernel::GetUnsupportedReason(probe);
    EXPECT_EQ(why, nullptr) << Kernel::GetName() << ": " << why;
    return {};
#else
    ck_tile::DeviceMem a(in.a.size()), b(in.b.size()), as(in.a_scale.size()), bs(in.b_scale.size()),
        e(e_bytes);
    a.ToDevice(in.a.data());
    b.ToDevice(in.b.data());
    as.ToDevice(in.a_scale.data());
    bs.ToDevice(in.b_scale.data());
    e.SetZero();
    const auto args = kernel_args<Kernel>(a.GetDeviceBuffer(),
                                          b.GetDeviceBuffer(),
                                          as.GetDeviceBuffer(),
                                          bs.GetDeviceBuffer(),
                                          e.GetDeviceBuffer(),
                                          M,
                                          N,
                                          K);
    launch<Kernel>(ck_tile::stream_config{nullptr, false}, args);
    HIP_CHECK_ERROR(hipDeviceSynchronize());
    std::vector<uint8_t> out(e_bytes);
    e.FromDevice(out.data());
    return out;
#endif
}

/// E of a kernel's types on `in` from the reference; with `rows` only those rows, compactly.
template <typename Kernel>
std::vector<uint8_t> run_reference(const Inputs& in,
                                   index_t M,
                                   index_t N,
                                   long_index_t K,
                                   const std::vector<index_t>* rows = nullptr)
{
    using A = typename Kernel::GemmPipeline::ADataType;
    using B = typename Kernel::GemmPipeline::BDataType;
    using O = typename Kernel::GemmPipeline::ODataType;
    std::vector<uint8_t> e((rows ? rows->size() : static_cast<std::size_t>(M)) * N *
                           sched::out_bytes<O>());
    ck_tile::reference_mx_gemm_tdm_scheduled<A, B, O>(in.a.data(),
                                                      sched::row_bytes<A>(K),
                                                      in.b.data(),
                                                      sched::row_bytes<B>(K),
                                                      in.a_scale.data(),
                                                      in.b_scale.data(),
                                                      e.data(),
                                                      N,
                                                      M,
                                                      N,
                                                      K,
                                                      0,
                                                      rows ? rows->data() : nullptr,
                                                      rows ? static_cast<index_t>(rows->size())
                                                           : 0);
    return e;
}

/// The first element whose bits differ, or "" if E matches bit for bit (any FP8 NaN code
/// matching any other).
template <typename O>
std::string
first_mismatch(const std::vector<uint8_t>& got, const std::vector<uint8_t>& want, index_t N)
{
    const index_t bytes = sched::out_bytes<O>();
    // The bits as well as the value: NaNs of different bits all print as nan.
    const auto hex = [&](const std::vector<uint8_t>& d, std::size_t i) {
        uint32_t bits = 0;
        std::memcpy(&bits, d.data() + i * bytes, bytes);
        char text[16];
        std::snprintf(text, sizeof(text), " (0x%0*X)", static_cast<int>(bytes * 2), bits);
        return std::string(text);
    };
    // FP8 NaN is 0x7F or 0xFF; the sign of a NaN out of arithmetic is unspecified, so the two
    // codes match.
    const auto fp8_nan = [](uint8_t b) { return (b & 0x7F) == 0x7F; };
    for(std::size_t i = 0; i < got.size() / bytes; ++i)
        if(std::memcmp(got.data() + i * bytes, want.data() + i * bytes, bytes) != 0 &&
           !(std::is_same_v<O, ck_tile::fp8_t> && fp8_nan(got[i]) && fp8_nan(want[i])))
            return "E(" + std::to_string(i / N) + ", " + std::to_string(i % N) + "): got " +
                   std::to_string(sched::load_out<O>(got.data(), i)) + hex(got, i) + ", want " +
                   std::to_string(sched::load_out<O>(want.data(), i)) + hex(want, i);
    return "";
}

/// Runs `Kernel` on the sparse inputs, their scales scale_shift above E8M0's bias, and
/// requires bitwise equality with the reference.
template <typename Kernel>
void check_kernel(index_t M, index_t N, index_t K, int scale_shift)
{
    SCOPED_TRACE(Kernel::GetName() + " M" + std::to_string(M) + " N" + std::to_string(N) + " K" +
                 std::to_string(K) + " scale shift " + std::to_string(scale_shift));
    const Inputs in = make_kernel_inputs<Kernel>(M, N, K, scale_shift);
    const char* why =
        Kernel::GetUnsupportedReason(kernel_args<Kernel>(in.a.data(),
                                                         in.b.data(),
                                                         in.a_scale.data(),
                                                         in.b_scale.data(),
                                                         const_cast<uint8_t*>(in.a.data()),
                                                         M,
                                                         N,
                                                         K));
    ASSERT_EQ(why, nullptr) << why;
    const auto got = run_device<Kernel>(in, M, N, K);
#ifndef MX_TDM_SCHEDULED_DRY_RUN
    const auto want = run_reference<Kernel>(in, M, N, K);
    EXPECT_EQ(first_mismatch<typename Kernel::GemmPipeline::ODataType>(got, want, N), "");
#endif
}

/// check_kernel of the instance's kernel for K's TailNum.
template <typename Kernel>
void check(index_t M, index_t N, index_t K, int scale_shift = kSparseScaleShift)
{
    for_tail_num<Kernel>(K, [&](auto variant) {
        check_kernel<typename decltype(variant)::type>(M, N, K, scale_shift);
    });
}

} // namespace mx_tdm_scheduled_test
