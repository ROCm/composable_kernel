// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// Timed benchmark and verifier for GemmPipelineAgBgCrCompTDMV3 on gfx1250.
//
// One instance: fp8 e4m3 A and B, fp32 accumulate, bf16 C, A row-major, B column-major, C
// row-major (RCR). Block tile 256x256x256, 16 waves (warps 4x4x1), warp tile 32x32x128, double LDS
// buffer, alternating TDM issuers (the default GemmPipelineAgBgCrCompTDMV3Policy), CShuffle
// epilogue, GemmTile1DPartitioner, one block per CU.
//
// Modes (-mode=):
//   verify-int   integers in [-5, 5] (exact in e4m3). Every partial sum is an integer below 2^24,
//                so fp32 accumulation is exact in any order; C must equal the exact sum rounded
//                once to bf16 (round to nearest even), bit for bit.
//   verify-rand  normal(0, sigma) rounded to e4m3. Checked against a double reference with the
//                tolerance |got - ref| <= |ref| * 2^-7 + sum|a*b| * 2^-16; max errors are reported.
//   peak         -rounds samples; before each, -gap_ms of idle, -warmup untimed launches, then
//                -repeat launches between two hipEvents. A sample is the mean time per launch.
//   sustained    the same loop with 50 warmup, 100 repeat, no idle gap and one round.
// The verify modes launch the kernel -verify_launches times and require every output to be
// byte-identical to the first. The reference covers every output (-check=full) or, for large
// problems, a sample of -check_per_tile points in every 256x256 output tile plus its four corners
// (-check=sample). -check=auto picks full when M*N*K <= 2^33.
//
// Output: one JSON line per timed sample ("type":"sample") and one summary line
// ("type":"summary"); verify modes print one "type":"verify" line. Exit code 0 = pass.

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
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

using ADataType   = ck_tile::fp8_t;
using BDataType   = ck_tile::fp8_t;
using AccDataType = float;
using CDataType   = ck_tile::bf16_t;
using ALayout     = ck_tile::tensor_layout::gemm::RowMajor;
using BLayout     = ck_tile::tensor_layout::gemm::ColumnMajor;
using CLayout     = ck_tile::tensor_layout::gemm::RowMajor;

constexpr ck_tile::index_t kTileM  = 256;
constexpr ck_tile::index_t kTileN  = 256;
constexpr ck_tile::index_t kTileK  = 256;
constexpr ck_tile::index_t kWarpsM = 4;
constexpr ck_tile::index_t kWarpsN = 4;
constexpr ck_tile::index_t kWarpM  = 32;
constexpr ck_tile::index_t kWarpN  = 32;
constexpr ck_tile::index_t kWarpK  = 128;

using TileShape       = ck_tile::TileGemmShape<ck_tile::sequence<kTileM, kTileN, kTileK>,
                                               ck_tile::sequence<kWarpsM, kWarpsN, 1>,
                                               ck_tile::sequence<kWarpM, kWarpN, kWarpK>>;
using Partitioner     = ck_tile::GemmTile1DPartitioner<TileShape>;
using Traits          = ck_tile::TileGemmUniversalTraits<false, // kPadM
                                                         false, // kPadN
                                                         false, // kPadK
                                                         true,  // DoubleSmemBuffer
                                                         ALayout,
                                                         BLayout,
                                                         CLayout>;
using Problem         = ck_tile::UniversalGemmPipelineProblem<ADataType,
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
                                                         kTileM,
                                                         kTileN,
                                                         kWarpsM,
                                                         kWarpsN,
                                                         kWarpM,
                                                         kWarpN,
                                                         kWarpK,
                                                         false, // isCTransposed
                                                         1,     // kNumWaveGroups
                                                         false, // FixedVectorSize
                                                         1,     // VectorSizeC
                                                         1,     // BlockedXDLN_PerWarp
                                                         true>; // DoubleSmemBuffer
using Epilogue        = ck_tile::CShuffleEpilogue<EpilogueProblem>;
using Kernel          = ck_tile::GemmKernel<Partitioner, Pipeline, Epilogue>;

constexpr const char* kBenchName = "ck_comp_tdm_v3_fp8_bf16_rcr_256x256x256_w4x4x1_wt32x32x128";

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

// ---------------------------------------------------------------------------------------------
// fp8 e4m3 (OCP, "fn": no inf, max 448) encode with round to nearest even and saturation.
// Written out here because the per-element host conversion is slow for 2 x 256 MiB of inputs;
// main() checks it against ck_tile::type_convert before use.
uint8_t encode_e4m3(float x)
{
    if(std::isnan(x))
        return 0x7F;
    const uint8_t sign = std::signbit(x) ? 0x80 : 0x00;
    const float a      = std::fabs(x);
    if(a >= 464.f) // halfway between 448 (max) and 480: saturate
        return sign | 0x7E;
    if(a < 0x1p-6f) // subnormal range, step 2^-9; 8 * 2^-9 rounds up into the first normal
    {
        const int q = static_cast<int>(std::nearbyint(a * 512.f));
        return static_cast<uint8_t>(sign | q);
    }
    int e          = 0;
    const float fr = std::frexp(a, &e); // a = fr * 2^e, fr in [0.5, 1)
    int exp2       = e - 1;             // a = (2 * fr) * 2^exp2
    int man        = static_cast<int>(std::nearbyint((2.f * fr - 1.f) * 8.f));
    if(man == 8)
    {
        man = 0;
        ++exp2;
    }
    int code = ((exp2 + 7) << 3) | man;
    if(code > 0x7E)
        code = 0x7E;
    return static_cast<uint8_t>(sign | code);
}

uint16_t bf16_rne(float f)
{
    const uint32_t u = ck_tile::bit_cast<uint32_t>(f);
    return static_cast<uint16_t>((u + 0x7FFFu + ((u >> 16) & 1u)) >> 16);
}

float bf16_to_float(uint16_t b) { return ck_tile::bit_cast<float>(static_cast<uint32_t>(b) << 16); }

std::vector<float> make_decode_table()
{
    std::vector<float> t(256);
    for(int i = 0; i < 256; ++i)
        t[static_cast<size_t>(i)] =
            ck_tile::type_convert<float>(ck_tile::bit_cast<ADataType>(static_cast<uint8_t>(i)));
    return t;
}

// The custom encoder must agree with CK's host conversion bit for bit.
size_t check_encoder(uint64_t seed)
{
    std::mt19937_64 g(seed);
    std::normal_distribution<float> dn(0.f, 1.f);
    std::uniform_real_distribution<float> du(-600.f, 600.f);
    std::vector<float> xs = {0.f,
                             -0.f,
                             1.f,
                             -1.f,
                             448.f,
                             463.9f,
                             464.f,
                             500.f,
                             -1e9f,
                             0x1p-9f,
                             0x1p-10f,
                             0x3p-11f,
                             0x1p-6f,
                             0x0.fp-6f,
                             0x1.0fp0f,
                             0x1.1p0f};
    for(int i = 0; i < 200000; ++i)
        xs.push_back(i % 3 == 0 ? du(g) : dn(g) * (i % 3 == 1 ? 0.1f : 0.002f));
    for(int i = -5; i <= 5; ++i)
        xs.push_back(static_cast<float>(i));
    size_t bad = 0;
    for(float x : xs)
    {
        const uint8_t ref = ck_tile::bit_cast<uint8_t>(ck_tile::type_convert<ADataType>(x));
        if(ref != encode_e4m3(x) && bad++ < 5)
            std::printf("encoder mismatch x=%a ck=0x%02x ours=0x%02x\n",
                        static_cast<double>(x),
                        ref,
                        encode_e4m3(x));
    }
    return bad;
}

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
    Rand
};

// Deterministic for a given seed whatever the thread count: 1 Mi-element chunks, each with its
// own generator seeded from (seed, tensor id, chunk index).
void fill(std::vector<uint8_t>& v, Data data, uint64_t seed, uint64_t tensor_id, float sigma)
{
    constexpr size_t kChunk = size_t{1} << 20;
    const size_t chunks     = (v.size() + kChunk - 1) / kChunk;
    parallel_for(chunks, [&](size_t c) {
        std::mt19937_64 g(splitmix64(seed ^ splitmix64(tensor_id * 0x100000000ull + c)));
        std::uniform_int_distribution<int> di(-5, 5);
        std::normal_distribution<float> dn(0.f, sigma);
        const size_t end = std::min(v.size(), (c + 1) * kChunk);
        for(size_t i = c * kChunk; i < end; ++i)
            v[i] = encode_e4m3(data == Data::Int ? static_cast<float>(di(g)) : dn(g));
    });
}

struct Point
{
    ck_tile::index_t m, n;
};

std::vector<Point>
check_points(ck_tile::index_t M, ck_tile::index_t N, bool full, int per_tile, uint64_t seed)
{
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
    for(ck_tile::index_t tm = 0; tm < M; tm += kTileM)
        for(ck_tile::index_t tn = 0; tn < N; tn += kTileN)
        {
            const ck_tile::index_t m1 = std::min(M, tm + kTileM) - 1;
            const ck_tile::index_t n1 = std::min(N, tn + kTileN) - 1;
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

void launch(const auto& kargs, const Problem3& p, hipStream_t stream)
{
    const dim3 grids  = Kernel::GridSize(p.M, p.N, 1);
    const dim3 blocks = Kernel::BlockSize();
    ck_tile::launch_kernel(ck_tile::stream_config{stream, false},
                           ck_tile::make_kernel<1>(Kernel{}, grids, blocks, 0, kargs));
}

double tflops(const Problem3& p, double ms)
{
    return 2.0 * p.M * p.N * static_cast<double>(p.K) / (ms * 1e-3) / 1e12;
}

int run(const ck_tile::ArgParser& arg)
{
    const Problem3 p{arg.get_int("m"), arg.get_int("n"), arg.get_int("k")};
    const std::string mode = arg.get_str("mode");
    const uint64_t seed    = arg.get_uint64("seed");
    const float sigma      = arg.get_float("sigma");
    const bool is_verify   = mode == "verify-int" || mode == "verify-rand";
    const bool is_peak     = mode == "peak";
    if(!is_verify && !is_peak && mode != "sustained")
    {
        std::printf("unknown -mode=%s (verify-int, verify-rand, peak, sustained)\n", mode.c_str());
        return 2;
    }
    if(p.M <= 0 || p.N <= 0 || p.K <= 0 || p.M % kTileM != 0 || p.N % kTileN != 0 ||
       p.K % kTileK != 0)
    {
        std::printf("M, N must be multiples of 256 and K a multiple of 256 (no padding in this "
                    "instance); got %d %d %d\n",
                    p.M,
                    p.N,
                    p.K);
        return 2;
    }
    Data data = Data::Rand;
    if(mode == "verify-int" || (!is_verify && arg.get_str("data") == "int"))
        data = Data::Int;
    const char* data_name = data == Data::Int ? "int" : "rand";

    // Timing defaults per mode; -1 means "use the mode default".
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
                kBenchName,
                Kernel::GetName().c_str(),
                prop.name,
                prop.gcnArchName,
                prop.multiProcessorCount,
                HIP_VERSION,
                __clang_version__);

    if(const size_t bad = check_encoder(seed); bad != 0)
    {
        std::printf("host e4m3 encoder disagrees with ck_tile::type_convert on %zu values\n", bad);
        return 5;
    }

    const size_t a_elems = static_cast<size_t>(p.M) * p.K;
    const size_t b_elems = static_cast<size_t>(p.N) * p.K;
    const size_t c_elems = static_cast<size_t>(p.M) * p.N;
    std::vector<uint8_t> a(a_elems), b(b_elems);
    const auto t_fill = std::chrono::steady_clock::now();
    fill(a, data, seed, 0, sigma);
    fill(b, data, seed, 1, sigma);
    std::printf("host fill (%s, seed %llu, sigma %g): %.1f s\n",
                data_name,
                static_cast<unsigned long long>(seed),
                static_cast<double>(sigma),
                std::chrono::duration<double>(std::chrono::steady_clock::now() - t_fill).count());

    DeviceBuffers d;
    BENCH_HIP_CHECK(hipMalloc(&d.a, a_elems));
    BENCH_HIP_CHECK(hipMalloc(&d.b, b_elems));
    BENCH_HIP_CHECK(hipMalloc(&d.c, c_elems * sizeof(CDataType)));
    BENCH_HIP_CHECK(hipMemcpy(d.a, a.data(), a_elems, hipMemcpyHostToDevice));
    BENCH_HIP_CHECK(hipMemcpy(d.b, b.data(), b_elems, hipMemcpyHostToDevice));
    BENCH_HIP_CHECK(hipMemset(d.c, 0, c_elems * sizeof(CDataType)));

    const ck_tile::GemmHostArgs args(d.a, d.b, d.c, 1, p.M, p.N, p.K, p.K, p.K, p.N);
    const auto kargs = Kernel::MakeKernelArgs(args);
    if(!Kernel::IsSupportedArgument(kargs))
    {
        std::printf("Kernel::IsSupportedArgument rejected M=%d N=%d K=%d\n", p.M, p.N, p.K);
        return 2;
    }
    hipStream_t stream = nullptr;

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
            launch(kargs, p, stream);
            BENCH_HIP_CHECK(hipGetLastError());
            BENCH_HIP_CHECK(hipDeviceSynchronize());
            BENCH_HIP_CHECK(
                hipMemcpy(out.data(), d.c, c_elems * sizeof(CDataType), hipMemcpyDeviceToHost));
            if(l == 0)
                first = out;
            else if(std::memcmp(first.data(), out.data(), c_elems * sizeof(uint16_t)) != 0)
                ++nondeterministic;
        }

        const std::vector<float> dec = make_decode_table();
        const auto pts = check_points(p.M, p.N, full, arg.get_int("check_per_tile"), seed);
        std::vector<double> refs(pts.size()), sabs(pts.size());
        const auto t_ref = std::chrono::steady_clock::now();
        parallel_for(pts.size(), [&](size_t i) {
            const uint8_t* pa = &a[static_cast<size_t>(pts[i].m) * p.K];
            const uint8_t* pb = &b[static_cast<size_t>(pts[i].n) * p.K];
            double s = 0, sa = 0;
            for(ck_tile::index_t k = 0; k < p.K; ++k)
            {
                const double prod = static_cast<double>(dec[pa[k]]) * dec[pb[k]];
                s += prod;
                sa += std::fabs(prod);
            }
            refs[i] = s;
            sabs[i] = sa;
        });
        const double ref_s =
            std::chrono::duration<double>(std::chrono::steady_clock::now() - t_ref).count();

        double sum_sq = 0;
        for(double r : refs)
            sum_sq += r * r;
        const double rms = std::sqrt(sum_sq / static_cast<double>(refs.size()));
        size_t bad = 0, rel_counted = 0;
        double max_abs = 0, max_rel = 0;
        for(size_t i = 0; i < pts.size(); ++i)
        {
            const uint16_t got = first[static_cast<size_t>(pts[i].m) * p.N + pts[i].n];
            const double gotf  = bf16_to_float(got);
            const double err   = std::fabs(gotf - refs[i]);
            bool ok;
            if(data == Data::Int)
                ok = got == bf16_rne(static_cast<float>(refs[i]));
            else
                ok = err <= std::fabs(refs[i]) * 0x1p-7 + sabs[i] * 0x1p-16;
            max_abs = std::max(max_abs, err);
            if(std::fabs(refs[i]) >= 0.01 * rms && std::fabs(refs[i]) > 0)
            {
                max_rel = std::max(max_rel, err / std::fabs(refs[i]));
                ++rel_counted;
            }
            if(!ok && bad++ < 5)
                std::printf(
                    "mismatch m=%d n=%d ref=%.9g got=%.9g\n", pts[i].m, pts[i].n, refs[i], gotf);
        }
        const bool pass = bad == 0 && nondeterministic == 0;
        std::printf("{\"type\":\"verify\",\"bench\":\"%s\",\"mode\":\"%s\",\"M\":%d,\"N\":%d,"
                    "\"K\":%d,\"data\":\"%s\",\"seed\":%llu,\"sigma\":%g,\"check\":\"%s\","
                    "\"checked\":%zu,\"bad\":%zu,\"launches\":%d,\"nondeterministic_launches\":%zu,"
                    "\"max_abs_err\":%.6g,\"max_rel_err\":%.6g,\"rel_err_points\":%zu,"
                    "\"ref_seconds\":%.1f,\"result\":\"%s\"}\n",
                    kBenchName,
                    mode.c_str(),
                    p.M,
                    p.N,
                    p.K,
                    data_name,
                    static_cast<unsigned long long>(seed),
                    static_cast<double>(sigma),
                    full ? "full" : "sample",
                    pts.size(),
                    bad,
                    launches,
                    nondeterministic,
                    max_abs,
                    max_rel,
                    rel_counted,
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
            launch(kargs, p, stream);
        BENCH_HIP_CHECK(hipEventRecord(start, stream));
        for(int i = 0; i < repeat; ++i)
            launch(kargs, p, stream);
        BENCH_HIP_CHECK(hipEventRecord(stop, stream));
        BENCH_HIP_CHECK(hipEventSynchronize(stop));
        BENCH_HIP_CHECK(hipGetLastError());
        float total_ms = 0;
        BENCH_HIP_CHECK(hipEventElapsedTime(&total_ms, start, stop));
        const double ms = static_cast<double>(total_ms) / std::max(repeat, 1);
        samples.push_back(ms);
        std::printf("{\"type\":\"sample\",\"bench\":\"%s\",\"mode\":\"%s\",\"round\":%d,\"M\":%d,"
                    "\"N\":%d,\"K\":%d,\"data\":\"%s\",\"seed\":%llu,\"warmup\":%d,\"repeat\":%d,"
                    "\"gap_ms\":%d,\"ms\":%.6f,\"tflops\":%.2f}\n",
                    kBenchName,
                    mode.c_str(),
                    r,
                    p.M,
                    p.N,
                    p.K,
                    data_name,
                    static_cast<unsigned long long>(seed),
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
    double mean         = 0;
    for(double s : samples)
        mean += s / static_cast<double>(n);
    std::printf("{\"type\":\"summary\",\"bench\":\"%s\",\"mode\":\"%s\",\"M\":%d,\"N\":%d,"
                "\"K\":%d,\"data\":\"%s\",\"seed\":%llu,\"warmup\":%d,\"repeat\":%d,\"rounds\":%d,"
                "\"gap_ms\":%d,\"median_ms\":%.6f,\"best_ms\":%.6f,\"mean_ms\":%.6f,"
                "\"median_tflops\":%.2f,\"best_tflops\":%.2f,\"best_over_median\":%.4f}\n",
                kBenchName,
                mode.c_str(),
                p.M,
                p.N,
                p.K,
                data_name,
                static_cast<unsigned long long>(seed),
                warmup,
                repeat,
                rounds,
                gap_ms,
                median,
                best,
                mean,
                tflops(p, median),
                tflops(p, best),
                median / best);
    return 0;
}

} // namespace

int main(int argc, char* argv[])
{
    ck_tile::ArgParser arg;
    arg.insert("m", "4096", "M (multiple of 256)")
        .insert("n", "4096", "N (multiple of 256)")
        .insert("k", "65536", "K (multiple of 256)")
        .insert("mode", "peak", "verify-int | verify-rand | peak | sustained")
        .insert("data", "rand", "input data for peak/sustained: rand | int")
        .insert("seed", "12345", "RNG seed for the inputs")
        .insert("sigma", "0.1", "standard deviation of the rand inputs (before e4m3 rounding)")
        .insert("warmup", "-1", "untimed launches per sample (-1: peak 1, sustained 50)")
        .insert("repeat", "-1", "timed launches per sample (-1: peak 3, sustained 100)")
        .insert("rounds", "-1", "samples (-1: peak 15, sustained 1)")
        .insert("gap_ms", "-1", "idle before each sample in ms (-1: peak 25, sustained 0)")
        .insert("verify_launches", "5", "verify modes: launches, all outputs must be identical")
        .insert("check", "auto", "verify modes: full | sample | auto (full if M*N*K <= 2^33)")
        .insert("check_per_tile", "64", "verify modes, sampled check: points per 256x256 tile");
    if(!arg.parse(argc, argv))
        return 2;
    try
    {
        return run(arg);
    }
    catch(const std::exception& e)
    {
        std::printf("error: %s\n", e.what());
        return 3;
    }
}
