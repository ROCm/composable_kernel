// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// MX GEMM on the comp_tdm_scheduled pipeline (gfx1250): E = A @ B^T with K-major E8M0 scales
// [ceil(K/128)][MN][4]. Named instances carry the policy's tuning knobs; -list prints them,
// -kernel picks one. E is stored through LDS by TDM (MxTdmStoreEpilogue).
//
//   verify  the sparse check inputs, E bit for bit against the host reference (sampled rows
//           when the problem is large)
//   bench   timed on FP4 sin/cos codes (-data=trig) or random finite inputs (-data=random)
//   check   whether the instance runs the problem; no device work

#include <cmath>
#include <cstdio>
#include <functional>
#include <iostream>
#include <map>
#include <random>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

#include "ck_tile/core.hpp"
#include "ck_tile/host.hpp"
#include "ck_tile/host/arg_parser.hpp"
#include "ck_tile/host/reference/reference_mx_gemm_tdm_scheduled.hpp"
#include "ck_tile/ops/epilogue.hpp"
#include "ck_tile/ops/gemm.hpp"

namespace {

using ck_tile::index_t;
using ck_tile::long_index_t;
namespace sched = ck_tile::mx_tdm_scheduled;

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

template <typename P>
using Kernel =
    ck_tile::MxGemmKernel<ck_tile::GemmTile2DNMajorPartitioner<typename P::BlockGemmShape>,
                          P,
                          ck_tile::MxTdmStoreEpilogue<P>>;

/// Launches K: the grid of GridSize, and with a cluster tuning the cluster of ClusterSize.
template <typename K>
float launch(const ck_tile::stream_config& s, const typename K::KernelArgs& kargs)
{
    if(!K::IsSupportedArgument(kargs))
    {
        const char* why = K::GetUnsupportedReason(kargs);
        throw std::runtime_error(K::GetName() + ": " + (why ? why : "see CK_TILE_LOGGING"));
    }
    const dim3 grids  = K::GridSize(kargs.M, kargs.N, kargs.k_batch);
    const dim3 blocks = K::BlockSize();
    if constexpr(K::ClusterLaunch)
    {
#if CK_TILE_ENABLE_CLUSTER_LAUNCH
        return ck_tile::launch_kernel(
            s, ck_tile::make_kernel<1>(K{}, K::ClusterSize(), grids, blocks, 0, kargs));
#else
        throw std::runtime_error(K::GetName() + ": this build has no cluster launch");
#endif
    }
    else
    {
        return ck_tile::launch_kernel(s, ck_tile::make_kernel<1>(K{}, grids, blocks, 0, kargs));
    }
}

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

// Whole M and N tiles, no K tail.
using Whole = ck_tile::MxTdmScheduledGemmTraits<false, false, false, false, false, 0>;
// Any M, N (edges), no K tail.
using Any = ck_tile::MxTdmScheduledGemmTraits<true, true, false, false, false, 0>;

using ck_tile::bf16_t;
using ck_tile::bf8_t;
using ck_tile::fp8_t;
using ck_tile::half_t;
using ck_tile::pk_fp4_t;

struct Options
{
    index_t M, N, K;
    std::string mode, data;
    int warmup, repeat;
    uint32_t seed;
};

/// The sparse check inputs (see test_mx_gemm_tdm_scheduled_util.hpp) or random finite ones: FP4
/// any code, 8-bit codes below 2 in magnitude, E8M0 2^-3..2^3. The sparse scales carry an alpha
/// of 2^-6 as 2^-3 on each operand, which keeps E in FP8 range. With `trig` FP4 A and B are
/// instead the E2M1 code, rounded to nearest even, of sin (A) or cos (B) of the element's linear
/// index, K fastest.
struct Host
{
    std::vector<uint8_t> a, b, as, bs;
};

uint8_t e2m1_rne(float x)
{
    // Midpoints between consecutive E2M1 magnitudes 0, 0.5, 1, 1.5, 2, 3, 4, 6.
    static constexpr float kMidpoint[7] = {0.25f, 0.75f, 1.25f, 1.75f, 2.5f, 3.5f, 5.0f};
    const float a                       = std::fabs(x);
    uint8_t code                        = 7;
    for(uint8_t c = 0; c < 7; ++c)
    {
        if(a < kMidpoint[c])
        {
            code = c;
            break;
        }
        if(!(a > kMidpoint[c]))
        {
            code = static_cast<uint8_t>(c % 2 == 0 ? c : c + 1);
            break;
        }
    }
    return static_cast<uint8_t>((std::signbit(x) ? 8 : 0) | code);
}

template <typename A, typename B>
Host make_inputs(
    index_t M, index_t N, long_index_t K, index_t tile_k, bool sparse, bool trig, uint32_t seed)
{
    std::mt19937 rng(seed);
    std::set<long_index_t> steps;
    const long_index_t n_steps = (K + 127) / 128, per_tile = tile_k / 128;
    const long_index_t last = (n_steps - 1) / per_tile * per_tile;
    for(long_index_t x : {long_index_t{0}, n_steps - 1, per_tile - 1, per_tile, last - 1, last})
        if(x >= 0 && x < n_steps)
            steps.insert(x);
    while(static_cast<long_index_t>(steps.size()) < std::min<long_index_t>(12, n_steps))
        steps.insert(static_cast<long_index_t>(rng() % n_steps));
    auto code = [&](auto type) -> uint8_t {
        using T = decltype(type);
        if(std::is_same_v<T, pk_fp4_t>)
            return static_cast<uint8_t>(rng() % 16);
        if(!sparse)
            return static_cast<uint8_t>(rng() & 0xBF);
        const int bits = std::is_same_v<T, fp8_t> ? 3 : 2, bias = std::is_same_v<T, fp8_t> ? 7 : 15;
        return static_cast<uint8_t>(((rng() % 2) << 7) | ((bias - rng() % 2) << bits) |
                                    (rng() % (1u << bits)));
    };
    auto operand = [&](auto type, index_t rows, bool use_cos) {
        using T               = decltype(type);
        constexpr bool fp4_t  = std::is_same_v<T, pk_fp4_t>;
        const long_index_t rb = sched::row_bytes<T>(K);
        const bool trig_fp4   = trig && !sparse && fp4_t;
        std::vector<uint8_t> data(static_cast<std::size_t>(rows) * rb, 0);
        for(index_t r = 0; r < rows; ++r)
        {
            uint8_t* row = data.data() + r * rb;
            auto put     = [&](long_index_t k) {
                const double index = static_cast<double>(r * K + k);
                const uint8_t x =
                    trig_fp4
                            ? e2m1_rne(static_cast<float>(use_cos ? std::cos(index) : std::sin(index)))
                            : code(type);
                if(fp4_t)
                    row[k / 2] |= static_cast<uint8_t>(k % 2 ? x << 4 : x);
                else
                    row[k] = x;
            };
            if(sparse)
                for(long_index_t st : steps)
                    for(long_index_t k = st * 128; k < std::min(K, (st + 1) * 128); ++k)
                        put(k);
            else
                for(long_index_t k = 0; k < K; ++k)
                    put(k);
        }
        return data;
    };
    Host h;
    h.a                       = operand(A{}, M, false);
    h.b                       = operand(B{}, N, true);
    const long_index_t groups = (K + 31) / 32;
    const bool fp4            = std::is_same_v<A, pk_fp4_t>;
    auto scales               = [&](index_t rows) {
        std::vector<uint8_t> natural(static_cast<std::size_t>(rows) * groups);
        for(auto& x : natural)
            x = static_cast<uint8_t>(!sparse ? 124 + rng() % 7
                                             : (fp4 ? 122 + rng() % 4 : 123 + rng() % 2));
        return ck_tile::mx_scale_to_kmajor(natural.data(), rows, K, groups);
    };
    h.as = scales(M);
    h.bs = scales(N);
    return h;
}

template <typename K>
int run_kernel(const Options& o)
{
    using P = typename K::GemmPipeline;
    using A = typename P::ADataType;
    using B = typename P::BDataType;
    using O = typename P::ODataType;
    {
        uint8_t dummy = 0;
        const ck_tile::MxGemmHostArgs<> probe{{&dummy},
                                              {&dummy},
                                              {&dummy},
                                              {&dummy},
                                              {},
                                              &dummy,
                                              1,
                                              o.M,
                                              o.N,
                                              o.K,
                                              {o.K},
                                              {o.K},
                                              {},
                                              o.N};
        if(const char* why = K::GetUnsupportedReason(K::MakeKernelArgs(probe)))
        {
            std::cerr << K::GetName() << " does not run M " << o.M << " N " << o.N << " K " << o.K
                      << ": " << why << "\n";
            return 2;
        }
    }
    std::cout << K::GetName() << " M " << o.M << " N " << o.N << " K " << o.K << " grid "
              << K::GridSize(o.M, o.N, 1).x << "x" << K::GridSize(o.M, o.N, 1).y << " lds "
              << K::GetSmemSize() << "\n";
    if(o.mode == "check")
        return 0;
    const Host h =
        make_inputs<A, B>(o.M, o.N, o.K, P::kTileK, o.mode == "verify", o.data == "trig", o.seed);

    const std::size_t e_bytes = static_cast<std::size_t>(o.M) * o.N * sched::out_bytes<O>();
    ck_tile::DeviceMem a(h.a.size()), b(h.b.size()), as(h.as.size()), bs(h.bs.size()), e(e_bytes);
    a.ToDevice(h.a.data());
    b.ToDevice(h.b.data());
    as.ToDevice(h.as.data());
    bs.ToDevice(h.bs.data());
    e.SetZero();
    const ck_tile::MxGemmHostArgs<> host_args{{a.GetDeviceBuffer()},
                                              {as.GetDeviceBuffer()},
                                              {b.GetDeviceBuffer()},
                                              {bs.GetDeviceBuffer()},
                                              {},
                                              e.GetDeviceBuffer(),
                                              1,
                                              o.M,
                                              o.N,
                                              o.K,
                                              {o.K},
                                              {o.K},
                                              {},
                                              o.N};
    const auto args = K::MakeKernelArgs(host_args);

    if(o.mode == "bench")
    {
        const float ms =
            launch<K>(ck_tile::stream_config{nullptr, true, 0, o.warmup, o.repeat}, args);
        const double flops = 2.0 * o.M * o.N * static_cast<double>(o.K);
        std::printf("%.3f us, %.1f TFLOPS\n", ms * 1e3, flops / (ms * 1e-3) / 1e12);
        return 0;
    }

    launch<K>(ck_tile::stream_config{nullptr, false}, args);
    HIP_CHECK_ERROR(hipDeviceSynchronize());
    std::vector<uint8_t> got(e_bytes);
    e.FromDevice(got.data());
    // Full check up to 2^32 multiply-adds, else the first and last tile of rows and 64 random.
    std::vector<index_t> rows;
    if(static_cast<double>(o.M) * o.N * o.K <= 4294967296.0)
        for(index_t r = 0; r < o.M; ++r)
            rows.push_back(r);
    else
    {
        std::set<index_t> pick;
        std::mt19937 rng(0xB10C);
        for(index_t r = 0; r < std::min(o.M, P::kTileM); ++r)
        {
            pick.insert(r);
            pick.insert(o.M - 1 - r);
        }
        for(int i = 0; i < 64; ++i)
            pick.insert(static_cast<index_t>(rng() % o.M));
        rows.assign(pick.begin(), pick.end());
    }
    const index_t ob = sched::out_bytes<O>();
    std::vector<uint8_t> want(rows.size() * o.N * ob);
    ck_tile::reference_mx_gemm_tdm_scheduled<A, B, O>(h.a.data(),
                                                      sched::row_bytes<A>(o.K),
                                                      h.b.data(),
                                                      sched::row_bytes<B>(o.K),
                                                      h.as.data(),
                                                      h.bs.data(),
                                                      want.data(),
                                                      o.N,
                                                      o.M,
                                                      o.N,
                                                      o.K,
                                                      0,
                                                      rows.data(),
                                                      static_cast<index_t>(rows.size()));
    std::size_t bad = 0;
    for(std::size_t i = 0; i < rows.size(); ++i)
        for(index_t n = 0; n < o.N; ++n)
        {
            const std::size_t g = static_cast<std::size_t>(rows[i]) * o.N + n, w = i * o.N + n;
            if(std::memcmp(got.data() + g * ob, want.data() + w * ob, ob) != 0 && bad++ < 8)
                std::printf("E(%d, %d): got %g, want %g\n",
                            rows[i],
                            n,
                            sched::load_out<O>(got.data(), g),
                            sched::load_out<O>(want.data(), w));
        }
    std::printf("%s: %zu of %zu rows checked, %zu mismatches\n",
                bad ? "FAIL" : "PASS",
                rows.size(),
                static_cast<std::size_t>(o.M),
                bad);
    return bad ? 1 : 0;
}

/// Runs pipeline P in the variant of K's whole-tile count.
template <typename P>
int run(const Options& o)
{
    const index_t full_tiles = o.K / P::kTileK;
    return P::TailHandler(
        [&](auto, auto tail_num) {
            using PT = typename P::template WithTailNum<decltype(tail_num)::value>;
            return run_kernel<Kernel<PT>>(o);
        },
        P::BlockHasHotloop(full_tiles),
        P::GetBlockLoopTailNum(full_tiles));
}

/// The named instances: the tuned instances of the maf, medium and mab problems and variants of
/// each knob.
const std::map<std::string, std::pair<const char*, std::function<int(const Options&)>>>& instances()
{
    static const std::map<std::string, std::pair<const char*, std::function<int(const Options&)>>>
        m{
            {"maf",
             {"mxfp4 t256x256x512 c4x4, fp8 out (MAF winner)",
              run<Pipeline<pk_fp4_t, pk_fp4_t, fp8_t, 256, 256, 512, 2, 2, 32, Tu<4, 4>, Whole>>}},
            {"maf_fp16",
             {"maf with fp16 out",
              run<Pipeline<pk_fp4_t, pk_fp4_t, half_t, 256, 256, 512, 2, 2, 32, Tu<4, 4>, Whole>>}},
            {"maf_bf16",
             {"maf with bf16 out",
              run<Pipeline<pk_fp4_t, pk_fp4_t, bf16_t, 256, 256, 512, 2, 2, 32, Tu<4, 4>, Whole>>}},
            {"maf_c2x2",
             {"maf on a 2x2 cluster",
              run<Pipeline<pk_fp4_t, pk_fp4_t, fp8_t, 256, 256, 512, 2, 2, 32, Tu<2, 2>, Whole>>}},
            {"medium_t128",
             {"mxfp4 t128x128x1024 c2x2",
              run<Pipeline<pk_fp4_t, pk_fp4_t, fp8_t, 128, 128, 1024, 2, 2, 32, Tu<2, 2>, Whole>>}},
            {"medium",
             {"mxfp4 t128x128x1024 16x16 atom, c4x4, no SplitBLdsRead (medium winner)",
              run<Pipeline<pk_fp4_t,
                           pk_fp4_t,
                           fp8_t,
                           128,
                           128,
                           1024,
                           2,
                           2,
                           16,
                           Tu<4, 4, 4, true, false>,
                           Whole>>}},
            {"mab",
             {"mxfp4 t128x16x512 wave 4x1, 16x16 atom, c1x4, no PrefetchLdsRead (MAB)",
              run<Pipeline<pk_fp4_t,
                           pk_fp4_t,
                           fp8_t,
                           128,
                           16,
                           512,
                           4,
                           1,
                           16,
                           Tu<1, 4, 0, false, false>,
                           Whole>>}},
            {"mab256",
             {"mab with a t256x16x512 tile (MAB winner)",
              run<Pipeline<pk_fp4_t,
                           pk_fp4_t,
                           fp8_t,
                           256,
                           16,
                           512,
                           4,
                           1,
                           16,
                           Tu<1, 4, 0, false, false>,
                           Whole>>}},
            {"any",
             {"mxfp4 t256x256x512 c1x1 for any M, N and tile count (no K tail)",
              run<Pipeline<pk_fp4_t, pk_fp4_t, fp8_t, 256, 256, 512, 2, 2, 32, Tu<1, 1>, Any>>}},
            {"fp8",
             {"mxfp8 t128x128x256 16x16 atom c2x2",
              run<Pipeline<fp8_t, fp8_t, fp8_t, 128, 128, 256, 2, 2, 16, Tu<2, 2>, Whole>>}},
            {"fp8_bf8_bf16",
             {"fp8 x bf8, bf16 out, t128x128x256 c2x2",
              run<Pipeline<fp8_t, bf8_t, bf16_t, 128, 128, 256, 2, 2, 16, Tu<2, 2>, Whole>>}},
        };
    return m;
}

} // namespace

int main(int argc, char* argv[])
{
    ck_tile::ArgParser arg;
    arg.insert("kernel", "maf", "named instance (-list=1)")
        .insert("list", "0", "1: print the instances and exit")
        .insert("m", "1024", "M")
        .insert("n", "1024", "N")
        .insert("k", "2048", "K (a multiple of 32)")
        .insert("mode", "verify", "verify | bench | check")
        .insert("data", "trig", "bench data: trig (FP4 sin/cos codes) | random")
        .insert("warmup", "50", "bench: untimed launches")
        .insert("repeat", "200", "bench: timed launches")
        .insert("seed", "23206", "input RNG seed");
    if(!arg.parse(argc, argv))
        return 1;
    if(arg.get_int("list"))
    {
        for(const auto& [name, entry] : instances())
            std::printf("%-14s %s\n", name.c_str(), entry.first);
        return 0;
    }
    const auto it = instances().find(arg.get_str("kernel"));
    if(it == instances().end())
    {
        std::cerr << "unknown -kernel " << arg.get_str("kernel") << " (see -list=1)\n";
        return 1;
    }
    const Options o{arg.get_int("m"),
                    arg.get_int("n"),
                    arg.get_int("k"),
                    arg.get_str("mode"),
                    arg.get_str("data"),
                    arg.get_int("warmup"),
                    arg.get_int("repeat"),
                    static_cast<uint32_t>(arg.get_int("seed"))};
    if(o.mode != "verify" && o.mode != "bench" && o.mode != "check")
    {
        std::cerr << "-mode must be verify, bench or check\n";
        return 1;
    }
    if(o.data != "trig" && o.data != "random")
    {
        std::cerr << "-data must be trig or random\n";
        return 1;
    }
    return it->second.second(o);
}
