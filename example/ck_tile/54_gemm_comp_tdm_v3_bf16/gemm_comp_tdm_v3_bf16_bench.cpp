// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// Correctness check and timed benchmark for GemmPipelineAgBgCrCompTDMV3 with bf16 A and B on
// gfx1250, for any M x N x K.
//
// bf16 A and B, fp32 accumulate, bf16 C row-major. Instances, chosen at run time:
//   -layout   rcr | rrr | crr | ccr     A and B layouts (C is row-major)
//   -epilogue tdm | cshuffle            TdmEpilogue writes C with one TDM store that clips to the
//                                       real M x N, so it serves every shape. The CShuffle
//                                       instances are padded (kPadM, kPadN) and store C in vectors
//                                       of -vector_c elements, which must divide N.
//   -vector_c auto | 8 | 2 | 1          CShuffle only; auto picks the widest of 8, 2 and 1 that
//                                       divides N, so CShuffle also serves every shape.
//   -tile     256x256x64 | 128x128x64   16 waves (warps 4x4x1) or 4 waves (warps 2x2x1)
//   -k_batch  N                         split K into N ranges, one workgroup layer each, added
//                                       into C atomically (CShuffle only; C is zeroed before
//                                       every launch, inside the timed loop).
// All use the warp tile 16x16x32, a double LDS buffer and the default policy. TDM clips every A/B
// box to the real extent, so the kernels are unpadded in K and serve any M, N and K.
//
// Modes (-mode=):
//   verify-int   integers in [-5, 5] (exact in bf16). Every partial sum is an integer below 2^24
//                for K <= 671088, so fp32 accumulation is exact in any order; C must equal the
//                exact sum rounded once to bf16 (round to nearest even), bit for bit.
//   verify-sparse  about sqrt(64 / K) of the elements nonzero, each -2, -1, 1 or 2: sum|a*b| stays
//                below 256 at almost every output, so every partial and running sum is exact in
//                bf16 and those outputs must be exact at any -k_batch (the rest use the bound).
//   verify-rand  normal(0, sigma) rounded to bf16. Checked against a double reference with the
//                tolerance |got - ref| <= |ref| * 2^-7 + sum|a*b| * 2^-16.
//   peak         -rounds samples; before each, -gap_ms of idle, -warmup untimed launches, then
//                -repeat launches between two hipEvents. A sample is the mean time per launch.
//   sustained    the same loop with 50 warmup, 100 repeat, no idle gap and one round.
// With -k_batch > 1 every split rounds its partial sum to bf16 and adds it into C atomically, in
// no fixed order, so both modes use |got - ref| <= |ref| * 2^-7 + k_batch * sum|a*b| * 2^-7 and
// the outputs of the launches need not be identical.
// The verify modes launch the kernel -verify_launches times and require every output to be
// byte-identical to the first (k_batch 1). The reference covers every output (-check=full) or a
// sample of -check_per_tile points in every 256x256 output tile plus its four corners
// (-check=sample). -check=auto picks full when M*N*K <= 2^33.
//
// Output: one JSON line per timed sample ("type":"sample") and one summary line
// ("type":"summary"); verify modes print one "type":"verify" line. Exit code: 0 pass, 1 wrong
// result, 2 rejected or bad arguments, 3 error.

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include <hip/hip_runtime.h>

#include "ck_tile/core.hpp"
#include "ck_tile/host/arg_parser.hpp"
#include "ck_tile/host/kernel_launch.hpp"
#include "ck_tile/ops/epilogue.hpp"
#include "ck_tile/ops/gemm.hpp"

namespace {

using ADataType   = ck_tile::bf16_t;
using BDataType   = ck_tile::bf16_t;
using AccDataType = float;
using CDataType   = ck_tile::bf16_t;
using RowMajor    = ck_tile::tensor_layout::gemm::RowMajor;
using ColumnMajor = ck_tile::tensor_layout::gemm::ColumnMajor;
using CLayout     = RowMajor;

constexpr ck_tile::index_t kWarpM = 16;
constexpr ck_tile::index_t kWarpN = 16;
constexpr ck_tile::index_t kWarpK = 32;

template <typename ALayout_,
          typename BLayout_,
          ck_tile::index_t TileM,
          ck_tile::index_t TileN,
          ck_tile::index_t TileK,
          ck_tile::index_t WarpsM,
          ck_tile::index_t WarpsN,
          bool TdmEpi,
          ck_tile::index_t VectorC = 0>
struct Instance
{
    using ALayout = ALayout_;
    using BLayout = BLayout_;
    // The CShuffle epilogue needs kPadM/kPadN for a partial C tile; TDM ignores them for A/B.
    static constexpr bool kPad = !TdmEpi;
    using TileShape            = ck_tile::TileGemmShape<ck_tile::sequence<TileM, TileN, TileK>,
                                                        ck_tile::sequence<WarpsM, WarpsN, 1>,
                                                        ck_tile::sequence<kWarpM, kWarpN, kWarpK>>;
    using Partitioner          = ck_tile::GemmTile1DPartitioner<TileShape>;
    using Traits               = ck_tile::TileGemmUniversalTraits<kPad,
                                                                  kPad,
                                                                  false, // kPadK
                                                                  true,  // DoubleSmemBuffer
                                                                  ALayout,
                                                                  BLayout,
                                                                  CLayout>;
    using Problem =
        ck_tile::UniversalGemmPipelineProblem<ADataType,
                                              BDataType,
                                              AccDataType,
                                              TileShape,
                                              Traits,
                                              ck_tile::GemmPipelineScheduler::Intrawave>;
    using Pipeline        = ck_tile::GemmPipelineAgBgCrCompTDMV3<Problem>;
    using EpilogueProblem = ck_tile::CShuffleEpilogueProblem<ADataType,
                                                             BDataType,
                                                             ck_tile::tuple<>,
                                                             AccDataType,
                                                             CDataType,
                                                             ck_tile::tuple<>,
                                                             CLayout,
                                                             ck_tile::element_wise::PassThrough,
                                                             TileM,
                                                             TileN,
                                                             WarpsM,
                                                             WarpsN,
                                                             kWarpM,
                                                             kWarpN,
                                                             kWarpK,
                                                             false,        // isCTransposed
                                                             1,            // kNumWaveGroups
                                                             VectorC != 0, // FixedVectorSize
                                                             VectorC != 0 ? VectorC : 1,
                                                             1,     // BlockedXDLN_PerWarp
                                                             true>; // DoubleSmemBuffer
    using Epilogue        = std::conditional_t<TdmEpi,
                                               ck_tile::TdmEpilogue<EpilogueProblem>,
                                               ck_tile::CShuffleEpilogue<EpilogueProblem>>;
    using Kernel          = ck_tile::GemmKernel<Partitioner, Pipeline, Epilogue>;
};

#define BENCH_HIP_CHECK(expr)                                                                    \
    do                                                                                           \
    {                                                                                            \
        const hipError_t bench_err_ = (expr);                                                    \
        if(bench_err_ != hipSuccess)                                                             \
        {                                                                                        \
            throw std::runtime_error(std::string("HIP error ") + hipGetErrorString(bench_err_) + \
                                     " at " + __FILE__ + ":" + std::to_string(__LINE__));        \
        }                                                                                        \
    } while(0)

uint16_t bf16_rne(float f)
{
    const uint32_t u = ck_tile::bit_cast<uint32_t>(f);
    return static_cast<uint16_t>((u + 0x7FFFu + ((u >> 16) & 1u)) >> 16);
}

float bf16_to_float(uint16_t b) { return ck_tile::bit_cast<float>(static_cast<uint32_t>(b) << 16); }

uint64_t splitmix64(uint64_t x)
{
    x += 0x9E3779B97F4A7C15ull;
    x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ull;
    x = (x ^ (x >> 27)) * 0x94D049BB133111EBull;
    return x ^ (x >> 31);
}

unsigned num_threads()
{
    const unsigned hc = std::thread::hardware_concurrency();
    return hc == 0 ? 8u : std::min(hc, 64u);
}

template <typename F>
void parallel_for(size_t n, F&& f)
{
    const size_t nt = std::min<size_t>(num_threads(), std::max<size_t>(n, 1));
    std::vector<std::thread> th;
    for(size_t t = 0; t < nt; ++t)
        th.emplace_back([&, t] {
            for(size_t i = n * t / nt; i < n * (t + 1) / nt; ++i)
                f(i);
        });
    for(auto& x : th)
        x.join();
}

enum class Data
{
    Int,
    Rand,
    Sparse
};

// Deterministic for a given seed whatever the thread count: 1 Mi-element chunks, each with its
// own generator seeded from (seed, tensor id, chunk index). Fills the packed storage directly.
// Sparse: each element is nonzero with probability `density`, then one of -2, -1, 1, 2.
void fill(std::vector<uint16_t>& v,
          Data data,
          uint64_t seed,
          uint64_t tensor_id,
          float sigma,
          double density = 1.0)
{
    constexpr size_t kChunk = size_t{1} << 20;
    const size_t chunks     = (v.size() + kChunk - 1) / kChunk;
    parallel_for(chunks, [&](size_t c) {
        std::mt19937_64 g(splitmix64(seed ^ splitmix64(tensor_id * 0x100000000ull + c)));
        std::uniform_int_distribution<int> di(-5, 5);
        std::normal_distribution<float> dn(0.f, sigma);
        std::uniform_real_distribution<double> du(0.0, 1.0);
        std::uniform_int_distribution<int> ds(0, 3);
        const size_t end = std::min(v.size(), (c + 1) * kChunk);
        for(size_t i = c * kChunk; i < end; ++i)
        {
            if(data == Data::Sparse)
            {
                static constexpr float kSparse[4] = {-2.f, -1.f, 1.f, 2.f};
                v[i] = bf16_rne(du(g) < density ? kSparse[ds(g)] : 0.f);
            }
            else
                v[i] = bf16_rne(data == Data::Int ? static_cast<float>(di(g)) : dn(g));
        }
    });
}

struct Point
{
    ck_tile::index_t m, n;
};

std::vector<Point>
check_points(ck_tile::index_t M, ck_tile::index_t N, bool full, int per_tile, uint64_t seed)
{
    constexpr ck_tile::index_t kCheckTile = 256;
    std::vector<Point> pts;
    if(full)
    {
        pts.reserve(static_cast<size_t>(M) * N);
        for(ck_tile::index_t m = 0; m < M; ++m)
            for(ck_tile::index_t n = 0; n < N; ++n)
                pts.push_back({m, n});
        return pts;
    }
    std::mt19937_64 g(splitmix64(seed ^ 0x5A5A5A5Aull));
    for(ck_tile::index_t tm = 0; tm < M; tm += kCheckTile)
        for(ck_tile::index_t tn = 0; tn < N; tn += kCheckTile)
        {
            const ck_tile::index_t m1 = std::min(M, tm + kCheckTile) - 1;
            const ck_tile::index_t n1 = std::min(N, tn + kCheckTile) - 1;
            pts.push_back({tm, tn});
            pts.push_back({tm, n1});
            pts.push_back({m1, tn});
            pts.push_back({m1, n1});
            std::uniform_int_distribution<ck_tile::index_t> dm(tm, m1), dn(tn, n1);
            for(int i = 0; i < per_tile; ++i)
                pts.push_back({dm(g), dn(g)});
        }
    return pts;
}

struct Problem3
{
    ck_tile::index_t M, N, K;
};

struct DeviceBuffers
{
    void* a = nullptr;
    void* b = nullptr;
    void* c = nullptr;
    ~DeviceBuffers()
    {
        for(void* p : {a, b, c})
            if(p != nullptr)
                (void)hipFree(p);
    }
};

double tflops(const Problem3& p, double ms)
{
    return 2.0 * p.M * p.N * static_cast<double>(p.K) / (ms * 1e-3) / 1e12;
}

template <typename Inst>
int run(const ck_tile::ArgParser& arg, const std::string& bench_name)
{
    using Kernel         = typename Inst::Kernel;
    constexpr bool a_row = std::is_same_v<typename Inst::ALayout, RowMajor>;
    constexpr bool b_col = std::is_same_v<typename Inst::BLayout, ColumnMajor>;
    const Problem3 p{arg.get_int("m"), arg.get_int("n"), arg.get_int("k")};
    const std::string mode = arg.get_str("mode");
    const uint64_t seed    = arg.get_uint64("seed");
    const float sigma      = arg.get_float("sigma");
    const bool is_verify = mode == "verify-int" || mode == "verify-rand" || mode == "verify-sparse";
    const bool is_peak   = mode == "peak";
    const int k_batch    = arg.get_int("k_batch");
    if(k_batch < 1)
    {
        std::printf("-k_batch must be at least 1; got %d\n", k_batch);
        return 2;
    }
    if(!is_verify && !is_peak && mode != "sustained")
    {
        std::printf("unknown -mode=%s (verify-int, verify-rand, verify-sparse, peak, sustained)\n",
                    mode.c_str());
        return 2;
    }
    if(p.M <= 0 || p.N <= 0 || p.K <= 0)
    {
        std::printf("M, N and K must be positive; got %d %d %d\n", p.M, p.N, p.K);
        return 2;
    }
    Data data = Data::Rand;
    if(mode == "verify-int" || (!is_verify && arg.get_str("data") == "int"))
        data = Data::Int;
    if(mode == "verify-sparse")
        data = Data::Sparse;
    const char* data_name = data == Data::Int ? "int" : data == Data::Sparse ? "sparse" : "rand";
    // Sparse density: about 64 nonzero products per output, so sum|a*b| stays below 256 at almost
    // every output and every partial and running sum is an integer that bf16 holds exactly.
    const double density = std::min(1.0, std::sqrt(64.0 / static_cast<double>(p.K)));

    int warmup = arg.get_int("warmup"), repeat = arg.get_int("repeat");
    int rounds = arg.get_int("rounds");
    int gap_ms = arg.get_int("gap_ms");
    if(warmup < 0)
        warmup = is_peak ? 1 : 50;
    if(repeat < 0)
        repeat = is_peak ? 3 : 100;
    if(rounds < 0)
        rounds = is_peak ? 15 : 1;
    if(gap_ms < 0)
        gap_ms = is_peak ? 25 : 0;

    int dev = 0;
    BENCH_HIP_CHECK(hipGetDevice(&dev));
    hipDeviceProp_t prop;
    BENCH_HIP_CHECK(hipGetDeviceProperties(&prop, dev));
    std::printf("{\"type\":\"info\",\"bench\":\"%s\",\"kernel\":\"%s\",\"device\":\"%s\","
                "\"arch\":\"%s\",\"cus\":%d,\"hip_version\":%d,\"compiler\":\"%s\"}\n",
                bench_name.c_str(),
                Kernel::GetName().c_str(),
                prop.name,
                prop.gcnArchName,
                prop.multiProcessorCount,
                HIP_VERSION,
                __clang_version__);

    // Packed storage: A is M x K (row-major, stride K) or K x M (column-major, stride M); B is
    // N x K (column-major, stride K) or K x N (row-major, stride N).
    const size_t a_elems  = static_cast<size_t>(p.M) * p.K;
    const size_t b_elems  = static_cast<size_t>(p.N) * p.K;
    const size_t c_elems  = static_cast<size_t>(p.M) * p.N;
    const size_t stride_a = static_cast<size_t>(a_row ? p.K : p.M);
    const size_t stride_b = static_cast<size_t>(b_col ? p.K : p.N);
    std::vector<uint16_t> a(a_elems), b(b_elems);
    const auto t_fill = std::chrono::steady_clock::now();
    fill(a, data, seed, 0, sigma, density);
    fill(b, data, seed, 1, sigma, density);
    std::printf("host fill (%s, seed %llu, sigma %g): %.1f s\n",
                data_name,
                static_cast<unsigned long long>(seed),
                static_cast<double>(sigma),
                std::chrono::duration<double>(std::chrono::steady_clock::now() - t_fill).count());

    DeviceBuffers d;
    BENCH_HIP_CHECK(hipMalloc(&d.a, a_elems * sizeof(ADataType)));
    BENCH_HIP_CHECK(hipMalloc(&d.b, b_elems * sizeof(BDataType)));
    BENCH_HIP_CHECK(hipMalloc(&d.c, c_elems * sizeof(CDataType)));
    BENCH_HIP_CHECK(hipMemcpy(d.a, a.data(), a_elems * sizeof(ADataType), hipMemcpyHostToDevice));
    BENCH_HIP_CHECK(hipMemcpy(d.b, b.data(), b_elems * sizeof(BDataType), hipMemcpyHostToDevice));
    BENCH_HIP_CHECK(hipMemset(d.c, 0, c_elems * sizeof(CDataType)));

    const ck_tile::GemmHostArgs args(d.a,
                                     d.b,
                                     d.c,
                                     k_batch,
                                     p.M,
                                     p.N,
                                     p.K,
                                     static_cast<ck_tile::index_t>(stride_a),
                                     static_cast<ck_tile::index_t>(stride_b),
                                     p.N);
    const auto kargs = Kernel::MakeKernelArgs(args);
    if(!Kernel::IsSupportedArgument(kargs))
    {
        std::printf("{\"type\":\"reject\",\"bench\":\"%s\",\"M\":%d,\"N\":%d,\"K\":%d,"
                    "\"reason\":\"Kernel::IsSupportedArgument (set CK_TILE_LOGGING=1 for "
                    "details)\"}\n",
                    bench_name.c_str(),
                    p.M,
                    p.N,
                    p.K);
        return 2;
    }
    hipStream_t stream = nullptr;
    const dim3 grids   = Kernel::GridSize(p.M, p.N, k_batch);
    const dim3 blocks  = Kernel::BlockSize();
    auto launch        = [&] {
        // The splits add into C, so it starts from zero every launch.
        if(k_batch > 1)
            BENCH_HIP_CHECK(hipMemsetAsync(d.c, 0, c_elems * sizeof(CDataType), stream));
        ck_tile::launch_kernel(ck_tile::stream_config{stream, false},
                               ck_tile::make_kernel<1>(Kernel{}, grids, blocks, 0, kargs));
    };

    if(is_verify)
    {
        const int launches  = std::max(1, arg.get_int("verify_launches"));
        const std::string c = arg.get_str("check");
        const bool full     = c == "full" || (c == "auto" && static_cast<double>(p.M) * p.N * p.K <=
                                                             8589934592.0); // 2^33
        std::vector<uint16_t> out(c_elems), first(c_elems);
        size_t nondeterministic = 0;
        for(int l = 0; l < launches; ++l)
        {
            BENCH_HIP_CHECK(hipMemset(d.c, 0, c_elems * sizeof(CDataType)));
            launch();
            BENCH_HIP_CHECK(hipGetLastError());
            BENCH_HIP_CHECK(hipDeviceSynchronize());
            BENCH_HIP_CHECK(
                hipMemcpy(out.data(), d.c, c_elems * sizeof(CDataType), hipMemcpyDeviceToHost));
            if(l == 0)
                first = out;
            else if(std::memcmp(first.data(), out.data(), c_elems * sizeof(uint16_t)) != 0)
                ++nondeterministic;
        }

        // A(m, k) and B(n, k) in the packed layouts.
        const size_t a_m = a_row ? stride_a : 1, a_k = a_row ? 1 : stride_a;
        const size_t b_n = b_col ? stride_b : 1, b_k = b_col ? 1 : stride_b;
        const auto pts = check_points(p.M, p.N, full, arg.get_int("check_per_tile"), seed);
        std::vector<double> refs(pts.size()), sabs(pts.size());
        const auto t_ref = std::chrono::steady_clock::now();
        parallel_for(pts.size(), [&](size_t i) {
            const uint16_t* pa = &a[static_cast<size_t>(pts[i].m) * a_m];
            const uint16_t* pb = &b[static_cast<size_t>(pts[i].n) * b_n];
            double s = 0, sa = 0;
            for(size_t k = 0; k < static_cast<size_t>(p.K); ++k)
            {
                const double prod =
                    static_cast<double>(bf16_to_float(pa[k * a_k])) * bf16_to_float(pb[k * b_k]);
                s += prod;
                sa += std::fabs(prod);
            }
            refs[i] = s;
            sabs[i] = sa;
        });
        const double ref_s =
            std::chrono::duration<double>(std::chrono::steady_clock::now() - t_ref).count();

        size_t bad          = 0;
        size_t exact_points = 0;
        double max_abs      = 0;
        for(size_t i = 0; i < pts.size(); ++i)
        {
            const uint16_t got = first[static_cast<size_t>(pts[i].m) * p.N + pts[i].n];
            const double gotf  = bf16_to_float(got);
            const double err   = std::fabs(gotf - refs[i]);
            // Sparse: an output with sum|a*b| <= 256 has every partial and running sum exact in
            // bf16, so it must be exact whatever the split count and order.
            const bool exact_sparse = data == Data::Sparse && sabs[i] <= 256.0;
            exact_points += exact_sparse;
            const bool ok =
                exact_sparse  ? got == bf16_rne(static_cast<float>(refs[i]))
                : k_batch > 1 ? err <= std::fabs(refs[i]) * 0x1p-7 + k_batch * sabs[i] * 0x1p-7
                : data == Data::Int ? got == bf16_rne(static_cast<float>(refs[i]))
                                    : err <= std::fabs(refs[i]) * 0x1p-7 + sabs[i] * 0x1p-16;
            max_abs = std::max(max_abs, err);
            if(!ok && bad++ < 5)
                std::printf(
                    "mismatch m=%d n=%d ref=%.9g got=%.9g\n", pts[i].m, pts[i].n, refs[i], gotf);
        }
        const bool pass = bad == 0 && (nondeterministic == 0 || k_batch > 1);
        std::printf("{\"type\":\"verify\",\"bench\":\"%s\",\"mode\":\"%s\",\"M\":%d,\"N\":%d,"
                    "\"K\":%d,\"k_batch\":%d,\"data\":\"%s\",\"seed\":%llu,\"check\":\"%s\","
                    "\"checked\":%zu,\"exact_checked\":%zu,\"bad\":%zu,\"launches\":%d,"
                    "\"nondeterministic_launches\":%zu,"
                    "\"max_abs_err\":%.6g,\"ref_seconds\":%.1f,\"result\":\"%s\"}\n",
                    bench_name.c_str(),
                    mode.c_str(),
                    p.M,
                    p.N,
                    p.K,
                    k_batch,
                    data_name,
                    static_cast<unsigned long long>(seed),
                    full ? "full" : "sample",
                    pts.size(),
                    exact_points,
                    bad,
                    launches,
                    nondeterministic,
                    max_abs,
                    ref_s,
                    pass ? "PASS" : "FAIL");
        return pass ? 0 : 1;
    }

    // Timed modes.
    hipEvent_t start, stop;
    BENCH_HIP_CHECK(hipEventCreate(&start));
    BENCH_HIP_CHECK(hipEventCreate(&stop));
    std::vector<double> samples;
    for(int r = 0; r < rounds; ++r)
    {
        BENCH_HIP_CHECK(hipDeviceSynchronize());
        if(gap_ms > 0)
            std::this_thread::sleep_for(std::chrono::milliseconds(gap_ms));
        for(int w = 0; w < warmup; ++w)
            launch();
        BENCH_HIP_CHECK(hipEventRecord(start, stream));
        for(int i = 0; i < repeat; ++i)
            launch();
        BENCH_HIP_CHECK(hipEventRecord(stop, stream));
        BENCH_HIP_CHECK(hipEventSynchronize(stop));
        BENCH_HIP_CHECK(hipGetLastError());
        float total_ms = 0;
        BENCH_HIP_CHECK(hipEventElapsedTime(&total_ms, start, stop));
        const double ms = static_cast<double>(total_ms) / std::max(repeat, 1);
        samples.push_back(ms);
        std::printf("{\"type\":\"sample\",\"bench\":\"%s\",\"mode\":\"%s\",\"round\":%d,\"M\":%d,"
                    "\"N\":%d,\"K\":%d,\"data\":\"%s\",\"warmup\":%d,\"repeat\":%d,\"gap_ms\":%d,"
                    "\"ms\":%.6f,\"tflops\":%.2f}\n",
                    bench_name.c_str(),
                    mode.c_str(),
                    r,
                    p.M,
                    p.N,
                    p.K,
                    data_name,
                    warmup,
                    repeat,
                    gap_ms,
                    ms,
                    tflops(p, ms));
        std::fflush(stdout);
    }
    BENCH_HIP_CHECK(hipEventDestroy(start));
    BENCH_HIP_CHECK(hipEventDestroy(stop));

    std::vector<double> sorted = samples;
    std::sort(sorted.begin(), sorted.end());
    const size_t n      = sorted.size();
    const double median = n % 2 ? sorted[n / 2] : 0.5 * (sorted[n / 2 - 1] + sorted[n / 2]);
    const double best   = sorted.front();
    std::printf("{\"type\":\"summary\",\"bench\":\"%s\",\"mode\":\"%s\",\"M\":%d,\"N\":%d,"
                "\"K\":%d,\"data\":\"%s\",\"warmup\":%d,\"repeat\":%d,\"rounds\":%d,"
                "\"gap_ms\":%d,\"median_ms\":%.6f,\"best_ms\":%.6f,\"median_tflops\":%.2f,"
                "\"best_tflops\":%.2f,\"best_over_median\":%.4f}\n",
                bench_name.c_str(),
                mode.c_str(),
                p.M,
                p.N,
                p.K,
                data_name,
                warmup,
                repeat,
                rounds,
                gap_ms,
                median,
                best,
                tflops(p, median),
                tflops(p, best),
                median / best);
    return 0;
}

// Run time -> instance. Each (layout, epilogue, tile, C vector) is one compiled kernel.
template <typename ALayout, typename BLayout, bool TdmEpi, ck_tile::index_t VectorC = 0>
int dispatch_tile(const ck_tile::ArgParser& arg, const std::string& name)
{
    const std::string tile = arg.get_str("tile");
    if(tile == "256x256x64")
        return run<Instance<ALayout, BLayout, 256, 256, 64, 4, 4, TdmEpi, VectorC>>(arg, name);
    if(tile == "128x128x64")
        return run<Instance<ALayout, BLayout, 128, 128, 64, 2, 2, TdmEpi, VectorC>>(arg, name);
    std::printf("unknown -tile=%s (256x256x64, 128x128x64)\n", tile.c_str());
    return 2;
}

template <typename ALayout, typename BLayout>
int dispatch_epilogue(const ck_tile::ArgParser& arg, const std::string& name)
{
    const std::string epi = arg.get_str("epilogue");
    if(epi == "tdm")
        return dispatch_tile<ALayout, BLayout, true>(arg, name);
    if(epi == "cshuffle")
    {
        // C vector: the widest of 8, 2 and 1 that divides N (auto), or the one given.
        const std::string vc = arg.get_str("vector_c");
        const int n          = arg.get_int("n");
        const int v = vc == "auto" ? (n % 8 == 0 ? 8 : n % 2 == 0 ? 2 : 1) : std::atoi(vc.c_str());
        const std::string vname = name + "_vc" + std::to_string(v);
        if(v == 8)
            return dispatch_tile<ALayout, BLayout, false>(arg, vname);
        if(v == 2)
            return dispatch_tile<ALayout, BLayout, false, 2>(arg, vname);
        if(v == 1)
            return dispatch_tile<ALayout, BLayout, false, 1>(arg, vname);
        std::printf("unknown -vector_c=%s (auto, 8, 2, 1)\n", vc.c_str());
        return 2;
    }
    std::printf("unknown -epilogue=%s (tdm, cshuffle)\n", epi.c_str());
    return 2;
}

int dispatch(const ck_tile::ArgParser& arg)
{
    const std::string layout = arg.get_str("layout");
    const std::string name   = "ck_comp_tdm_v3_bf16_bf16_" + layout + "_" + arg.get_str("tile") +
                             "_wt16x16x32_" + arg.get_str("epilogue") +
                             (arg.get_int("k_batch") > 1 ? "_kb" + arg.get_str("k_batch") : "");
    if(layout == "rcr")
        return dispatch_epilogue<RowMajor, ColumnMajor>(arg, name);
    if(layout == "rrr")
        return dispatch_epilogue<RowMajor, RowMajor>(arg, name);
    if(layout == "crr")
        return dispatch_epilogue<ColumnMajor, RowMajor>(arg, name);
    if(layout == "ccr")
        return dispatch_epilogue<ColumnMajor, ColumnMajor>(arg, name);
    std::printf("unknown -layout=%s (rcr, rrr, crr, ccr)\n", layout.c_str());
    return 2;
}

} // namespace

int main(int argc, char* argv[])
{
    ck_tile::ArgParser arg;
    arg.insert("m", "4096", "M")
        .insert("n", "4096", "N")
        .insert("k", "4096", "K")
        .insert("layout", "rcr", "A and B layouts, C row-major: rcr | rrr | crr | ccr")
        .insert("epilogue", "tdm", "tdm | cshuffle (padded)")
        .insert("vector_c",
                "auto",
                "cshuffle C vector: auto (8, 2 or 1, widest dividing N) | 8 | 2 | 1")
        .insert("tile", "256x256x64", "256x256x64 (16 waves) | 128x128x64 (4 waves)")
        .insert("k_batch", "1", "split K into this many atomically added ranges (CShuffle only)")
        .insert("mode", "verify-int", "verify-int | verify-rand | peak | sustained")
        .insert("data", "rand", "input data for peak/sustained: rand | int")
        .insert("seed", "12345", "RNG seed for the inputs")
        .insert("sigma", "1.0", "standard deviation of the rand inputs (before bf16 rounding)")
        .insert("warmup", "-1", "untimed launches per sample (-1: peak 1, sustained 50)")
        .insert("repeat", "-1", "timed launches per sample (-1: peak 3, sustained 100)")
        .insert("rounds", "-1", "samples (-1: peak 15, sustained 1)")
        .insert("gap_ms", "-1", "idle before each sample in ms (-1: peak 25, sustained 0)")
        .insert("verify_launches", "3", "verify modes: launches, all outputs must be identical")
        .insert("check", "auto", "verify modes: full | sample | auto (full if M*N*K <= 2^33)")
        .insert("check_per_tile", "64", "verify modes, sampled check: points per 256x256 tile");
    if(!arg.parse(argc, argv))
        return 2;
    try
    {
        return dispatch(arg);
    }
    catch(const std::exception& e)
    {
        std::printf("error: %s\n", e.what());
        return 3;
    }
}
