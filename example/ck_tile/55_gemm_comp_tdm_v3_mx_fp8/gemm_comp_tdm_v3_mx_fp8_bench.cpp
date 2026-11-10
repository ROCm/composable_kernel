// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// Correctness check and timed benchmark for MX FP8 GEMM on GemmPipelineAgBgCrCompTDMV3 (gfx1250),
// for any M x N x K with K a multiple of 128.
//
// fp8 (e4m3) A row-major and B column-major, one E8M0 scale per 32 K elements of every A row and
// every B column, fp32 accumulate, C row-major in fp16 (-c_dtype=fp16) or bf16 (-c_dtype=bf16).
// Warp tile 16x16x128, the scaled WMMA. Instances, chosen at run time:
//   -tile     32x256x256 | 32x128x256 | 32x64x256 | 64x256x256    4 waves (warps 2x2x1)
//             16x512x256 | 16x256x256 | 16x128x256 | 16x64x256    4 waves (warps 1x4x1)
//             256x256x256                             16 waves (warps 4x4x1)
//   -epilogue tdm | cshuffle   TdmEpilogue writes C with one TDM store clipped to M x N.
//                              CShuffle is padded (kPadM, kPadN) and also runs split-K.
//   -k_batch  N                split K into N ranges added into C atomically (CShuffle only; C is
//                              zeroed before every launch, inside the timed loop).
//
// Modes (-mode=):
//   verify-sparse  A and B elements nonzero with probability sqrt(16 / K), each -2, -1, 1 or 2;
//                  every scale 1/2, 1 or 2, chosen at random per (row or column, 32-element K
//                  block), so a scale applied to the wrong K block or row changes the result.
//                  Every product is a multiple of 1/4 of magnitude at most 16 and sum|terms| stays
//                  far below 2^24, so fp32 accumulation is exact in any order. Every partial sum
//                  of an output is a multiple of 1/4 no larger than its sum|terms|, which C holds
//                  exactly up to 2^(p-2) (p: C's significand bits, 11 fp16, 8 bf16); an output
//                  whose sum|terms| is at most half that, 256 for fp16 and 32 for bf16, is exact
//                  after any number of roundings of partial sums and must equal the reference bit
//                  for bit at any -k_batch. The rest use the bound below.
//   verify-rand    normal(0, sigma) rounded to e4m3, scales 2^e with e uniform in [-2, 2]. Checked
//                  against a double reference with |got - ref| <= |ref| * u + k_batch * S * u,
//                  where S = sum|terms| and u is C's unit roundoff (2^-11 fp16, 2^-8 bf16).
//   peak           -rounds samples; before each, -gap_ms of idle, -warmup untimed launches, then
//                  -repeat launches between two hipEvents. A sample is the mean time per launch.
//   sustained      the same loop with 50 warmup, 100 repeat, no idle gap and one round.
// The verify modes launch -verify_launches times; with k_batch 1 every output must be
// byte-identical to the first. The reference covers every output when M*N*K <= 2^33 (-check=auto)
// or -check=full, else -check_per_tile points in every 256x256 output tile plus its corners.
//
// Output: JSON lines ("type":"info", "verify", "sample", "summary"). Exit code: 0 pass, 1 wrong
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
#include "ck_tile/host/mx_processing.hpp"
#include "ck_tile/ops/epilogue.hpp"
#include "ck_tile/ops/gemm.hpp"
#include "ck_tile/ops/gemm/kernel/mx_gemm_kernel.hpp"

namespace {

using ADataType      = ck_tile::fp8_t;
using BDataType      = ck_tile::fp8_t;
using ScaleDataType  = ck_tile::e8m0_t;
using AccDataType    = float;
using RowMajor       = ck_tile::tensor_layout::gemm::RowMajor;
using ColumnMajor    = ck_tile::tensor_layout::gemm::ColumnMajor;
using ALayout        = RowMajor;
using BLayout        = ColumnMajor;
using CLayout        = RowMajor;
constexpr int kScale = 32; // K elements per scale

constexpr ck_tile::index_t kWarpM = 16;
constexpr ck_tile::index_t kWarpN = 16;
constexpr ck_tile::index_t kWarpK = 128;

// One compiled kernel per (tile, epilogue, C type). The configuration follows the gfx1250 MX
// gtests (test_mx_gemm_pipeline_util.hpp): transposed C for row-major C and a square warp tile,
// and an unpacked C tile for the CShuffle epilogue (comp_tdm_v3's block GEMM does not pack).
template <ck_tile::index_t TileM,
          ck_tile::index_t TileN,
          ck_tile::index_t TileK,
          ck_tile::index_t WarpsM,
          ck_tile::index_t WarpsN,
          bool TdmEpi,
          typename CDataType_>
struct Instance
{
    using CDataType                          = CDataType_;
    static constexpr bool kPad               = !TdmEpi; // CShuffle: partial C tiles; TDM clips
    static constexpr bool kTransC            = true;
    static constexpr ck_tile::index_t kTileM = TileM;
    static constexpr ck_tile::index_t kTileN = TileN;
    using TileShape       = ck_tile::TileGemmShape<ck_tile::sequence<TileM, TileN, TileK>,
                                                   ck_tile::sequence<WarpsM, WarpsN, 1>,
                                                   ck_tile::sequence<kWarpM, kWarpN, kWarpK>>;
    using Partitioner     = ck_tile::GemmSpatiallyLocalTilePartitioner<TileShape, 8, 4>;
    using Traits          = ck_tile::TileGemmUniversalTraits<kPad,
                                                             kPad,
                                                             false, // kPadK (MX GEMM forbids it)
                                                             true,  // DoubleSmemBuffer
                                                             ALayout,
                                                             BLayout,
                                                             CLayout,
                                                             kTransC,
                                                             false,  // StructuredSparsity
                                                             false,  // Persistent
                                                             1,      // NumWaveGroups
                                                             false>; // Preshuffle
    using Problem         = ck_tile::MxGemmPipelineProblem<ADataType,
                                                           BDataType,
                                                           AccDataType,
                                                           TileShape,
                                                           Traits,
                                                           ck_tile::GemmPipelineScheduler::Intrawave,
                                                           ck_tile::element_wise::PassThrough,
                                                           ck_tile::element_wise::PassThrough,
                                                           ADataType,
                                                           BDataType,
                                                           ScaleDataType,
                                                           ScaleDataType,
                                                           kScale>;
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
                                                             Problem::TransposeC,
                                                             1,     // kNumWaveGroups
                                                             false, // FixedVectorSize
                                                             1,     // VectorSizeC
                                                             1,     // BlockedXDLN_PerWarp
                                                             true,  // DoubleSmemBuffer
                                                             ADataType,
                                                             BDataType,
                                                             false>; // TilesPacked
    using Epilogue        = std::conditional_t<TdmEpi,
                                               ck_tile::TdmEpilogue<EpilogueProblem>,
                                               ck_tile::CShuffleEpilogue<EpilogueProblem>>;
    using Kernel          = ck_tile::MxGemmKernel<Partitioner, Pipeline, Epilogue>;
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

uint64_t splitmix64(uint64_t x)
{
    x += 0x9E3779B97F4A7C15ull;
    x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ull;
    x = (x ^ (x >> 27)) * 0x94D049BB133111EBull;
    return x ^ (x >> 31);
}

template <typename F>
void parallel_for(size_t n, F&& f)
{
    const unsigned hc = std::thread::hardware_concurrency();
    const size_t nt   = std::min<size_t>(hc == 0 ? 8u : std::min(hc, 64u), std::max<size_t>(n, 1));
    std::vector<std::thread> th;
    for(size_t t = 0; t < nt; ++t)
        th.emplace_back([&, t] {
            for(size_t i = n * t / nt; i < n * (t + 1) / nt; ++i)
                f(i);
        });
    for(auto& x : th)
        x.join();
}

// The reference rounded once to C's type, as raw bits, for the bit-for-bit checks.
template <typename CDataType>
uint16_t c_bits(double x)
{
    return ck_tile::bit_cast<uint16_t>(ck_tile::type_convert<CDataType>(static_cast<float>(x)));
}

template <typename CDataType>
constexpr double unit_roundoff()
{
    return std::is_same_v<CDataType, ck_tile::bf16_t> ? 0x1p-8 : 0x1p-11;
}

// verify-sparse: the largest sum|terms| whose output must be exact at any k_batch. Partial sums
// are multiples of 1/4, which C holds exactly up to 2^(p-2); half of that leaves a margin.
template <typename CDataType>
constexpr double sparse_exact_limit()
{
    return std::is_same_v<CDataType, ck_tile::bf16_t> ? 32.0 : 256.0;
}

struct Problem3
{
    ck_tile::index_t M, N, K;
};

double tflops(const Problem3& p, double ms)
{
    return 2.0 * p.M * p.N * static_cast<double>(p.K) / (ms * 1e-3) / 1e12;
}

struct DeviceBuffers
{
    void* a  = nullptr;
    void* b  = nullptr;
    void* sa = nullptr;
    void* sb = nullptr;
    void* c  = nullptr;
    ~DeviceBuffers()
    {
        for(void* p : {a, b, sa, sb, c})
            if(p != nullptr)
                (void)hipFree(p);
    }
};

// Host operands: element values and per-(row, K block) scale exponents, plus the encoded bytes.
struct HostOperand
{
    std::vector<ADataType> q;       // rows x K, fp8
    std::vector<float> v;           // rows x K, decoded fp8 values
    std::vector<int8_t> e;          // rows x (K / 32), scale exponents
    std::vector<ScaleDataType> raw; // padded_rows x (K / 32), E8M0 (natural layout)
};

HostOperand make_operand(ck_tile::index_t rows,
                         ck_tile::index_t padded_rows,
                         ck_tile::index_t K,
                         bool sparse,
                         float sigma,
                         uint64_t seed,
                         uint64_t id)
{
    const ck_tile::index_t ks = K / kScale;
    HostOperand o;
    o.q.resize(static_cast<size_t>(rows) * K);
    o.v.resize(o.q.size());
    o.e.resize(static_cast<size_t>(rows) * ks);
    o.raw.assign(static_cast<size_t>(padded_rows) * ks, ScaleDataType{static_cast<uint8_t>(127)});
    const double density = std::min(1.0, std::sqrt(16.0 / static_cast<double>(K)));
    parallel_for(static_cast<size_t>(rows), [&](size_t r) {
        std::mt19937_64 g(splitmix64(seed ^ splitmix64(id * 0x100000000ull + r)));
        std::uniform_real_distribution<double> du(0.0, 1.0);
        std::uniform_int_distribution<int> ds(0, 3);
        std::normal_distribution<float> dn(0.f, sigma);
        std::uniform_int_distribution<int> de(sparse ? -1 : -2, sparse ? 1 : 2);
        static constexpr float kSparse[4] = {-2.f, -1.f, 1.f, 2.f};
        for(ck_tile::index_t k = 0; k < K; ++k)
        {
            const float x  = sparse ? (du(g) < density ? kSparse[ds(g)] : 0.f) : dn(g);
            const size_t i = r * K + k;
            o.q[i]         = ck_tile::type_convert<ADataType>(x);
            o.v[i]         = ck_tile::type_convert<float>(o.q[i]);
        }
        for(ck_tile::index_t b = 0; b < ks; ++b)
        {
            const int ex      = de(g);
            o.e[r * ks + b]   = static_cast<int8_t>(ex);
            o.raw[r * ks + b] = ScaleDataType{static_cast<uint8_t>(127 + ex)};
        }
    });
    return o;
}

template <typename Inst>
int run(const ck_tile::ArgParser& arg, const std::string& bench_name)
{
    using Kernel    = typename Inst::Kernel;
    using CDataType = typename Inst::CDataType;
    const Problem3 p{arg.get_int("m"), arg.get_int("n"), arg.get_int("k")};
    const std::string mode = arg.get_str("mode");
    const uint64_t seed    = arg.get_uint64("seed");
    const float sigma      = arg.get_float("sigma");
    const int k_batch      = arg.get_int("k_batch");
    const bool is_verify   = mode == "verify-sparse" || mode == "verify-rand";
    const bool is_peak     = mode == "peak";
    if(!is_verify && !is_peak && mode != "sustained")
    {
        std::printf("unknown -mode=%s (verify-sparse, verify-rand, peak, sustained)\n",
                    mode.c_str());
        return 2;
    }
    if(p.M <= 0 || p.N <= 0 || p.K <= 0 || p.K % 128 != 0 || k_batch < 1)
    {
        std::printf(
            "need M, N > 0, K a positive multiple of 128 and k_batch >= 1; got %d %d %d %d\n",
            p.M,
            p.N,
            p.K,
            k_batch);
        return 2;
    }
    const bool sparse = mode == "verify-sparse" || (!is_verify && arg.get_str("data") == "sparse");
    int warmup = arg.get_int("warmup"), repeat = arg.get_int("repeat");
    int rounds = arg.get_int("rounds"), gap_ms = arg.get_int("gap_ms");
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
                "\"arch\":\"%s\",\"cus\":%d,\"hip_version\":%d}\n",
                bench_name.c_str(),
                Kernel::GetName().c_str(),
                prop.name,
                prop.gcnArchName,
                prop.multiProcessorCount,
                HIP_VERSION);

    // Scales: natural (rows, K / 32) E8M0, rows padded as the gtests do, then pre-shuffled into
    // the device layout.
    const ck_tile::index_t ks = p.K / kScale;
    const ck_tile::index_t pad_m =
        ck_tile::integer_least_multiple(p.M, std::max<ck_tile::index_t>(Inst::kTileM, 32));
    const ck_tile::index_t pad_n =
        ck_tile::integer_least_multiple(p.N, std::max<ck_tile::index_t>(Inst::kTileN, 32));
    const auto t_fill   = std::chrono::steady_clock::now();
    const HostOperand a = make_operand(p.M, pad_m, p.K, sparse, sigma, seed, 0);
    const HostOperand b = make_operand(p.N, pad_n, p.K, sparse, sigma, seed, 1);
    std::vector<ScaleDataType> sa_dev(a.raw.size()), sb_dev(b.raw.size());
    ck_tile::preShuffleScaleBuffer_gfx1250<ScaleDataType, kScale, true>(
        a.raw.data(), sa_dev.data(), pad_m, ks, kWarpM);
    ck_tile::preShuffleScaleBuffer_gfx1250<ScaleDataType, kScale, true>(
        b.raw.data(), sb_dev.data(), pad_n, ks, kWarpN);
    std::printf("host fill (%s, seed %llu): %.1f s\n",
                sparse ? "sparse" : "rand",
                static_cast<unsigned long long>(seed),
                std::chrono::duration<double>(std::chrono::steady_clock::now() - t_fill).count());

    const size_t c_elems = static_cast<size_t>(p.M) * p.N;
    DeviceBuffers d;
    BENCH_HIP_CHECK(hipMalloc(&d.a, a.q.size()));
    BENCH_HIP_CHECK(hipMalloc(&d.b, b.q.size()));
    BENCH_HIP_CHECK(hipMalloc(&d.sa, sa_dev.size()));
    BENCH_HIP_CHECK(hipMalloc(&d.sb, sb_dev.size()));
    BENCH_HIP_CHECK(hipMalloc(&d.c, c_elems * sizeof(CDataType)));
    BENCH_HIP_CHECK(hipMemcpy(d.a, a.q.data(), a.q.size(), hipMemcpyHostToDevice));
    BENCH_HIP_CHECK(hipMemcpy(d.b, b.q.data(), b.q.size(), hipMemcpyHostToDevice));
    BENCH_HIP_CHECK(hipMemcpy(d.sa, sa_dev.data(), sa_dev.size(), hipMemcpyHostToDevice));
    BENCH_HIP_CHECK(hipMemcpy(d.sb, sb_dev.data(), sb_dev.size(), hipMemcpyHostToDevice));
    BENCH_HIP_CHECK(hipMemset(d.c, 0, c_elems * sizeof(CDataType)));

    const ck_tile::MxGemmHostArgs<1, 1, 0> args({static_cast<const void*>(d.a)},
                                                {static_cast<const void*>(d.sa)},
                                                {static_cast<const void*>(d.b)},
                                                {static_cast<const void*>(d.sb)},
                                                {},
                                                d.c,
                                                k_batch,
                                                p.M,
                                                p.N,
                                                p.K,
                                                {p.K},
                                                {p.K},
                                                {},
                                                p.N);
    const auto kargs = Kernel::MakeKernelArgs(args);
    if(!Kernel::IsSupportedArgument(kargs))
    {
        std::printf("{\"type\":\"reject\",\"bench\":\"%s\",\"M\":%d,\"N\":%d,\"K\":%d,"
                    "\"k_batch\":%d,\"reason\":\"Kernel::IsSupportedArgument (set "
                    "CK_TILE_LOGGING=1 for details)\"}\n",
                    bench_name.c_str(),
                    p.M,
                    p.N,
                    p.K,
                    k_batch);
        return 2;
    }
    hipStream_t stream = nullptr;
    const dim3 grids   = Kernel::GridSize(p.M, p.N, k_batch);
    const dim3 blocks  = Kernel::BlockSize();
    auto launch        = [&] {
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
        std::vector<CDataType> out(c_elems), first(c_elems);
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
            else if(std::memcmp(first.data(), out.data(), c_elems * sizeof(CDataType)) != 0)
                ++nondeterministic;
        }

        // Points: every output, or a sample per 256x256 tile plus its corners.
        std::vector<std::pair<ck_tile::index_t, ck_tile::index_t>> pts;
        if(full)
        {
            pts.reserve(c_elems);
            for(ck_tile::index_t m = 0; m < p.M; ++m)
                for(ck_tile::index_t n = 0; n < p.N; ++n)
                    pts.emplace_back(m, n);
        }
        else
        {
            std::mt19937_64 g(splitmix64(seed ^ 0x5A5A5A5Aull));
            const int per_tile = arg.get_int("check_per_tile");
            for(ck_tile::index_t tm = 0; tm < p.M; tm += 256)
                for(ck_tile::index_t tn = 0; tn < p.N; tn += 256)
                {
                    const ck_tile::index_t m1 = std::min(p.M, tm + 256) - 1;
                    const ck_tile::index_t n1 = std::min(p.N, tn + 256) - 1;
                    pts.emplace_back(tm, tn);
                    pts.emplace_back(tm, n1);
                    pts.emplace_back(m1, tn);
                    pts.emplace_back(m1, n1);
                    std::uniform_int_distribution<ck_tile::index_t> dm(tm, m1), dn(tn, n1);
                    for(int i = 0; i < per_tile; ++i)
                        pts.emplace_back(dm(g), dn(g));
                }
        }
        // Reference in double: sum over K of a * 2^ea * b * 2^eb.
        std::vector<double> refs(pts.size()), sabs(pts.size());
        const auto t_ref = std::chrono::steady_clock::now();
        parallel_for(pts.size(), [&](size_t i) {
            const auto [m, n] = pts[i];
            const float* pa   = &a.v[static_cast<size_t>(m) * p.K];
            const float* pb   = &b.v[static_cast<size_t>(n) * p.K];
            const int8_t* ea  = &a.e[static_cast<size_t>(m) * ks];
            const int8_t* eb  = &b.e[static_cast<size_t>(n) * ks];
            double s = 0, sa = 0;
            for(ck_tile::index_t blk = 0; blk < ks; ++blk)
            {
                double part = 0, part_abs = 0;
                for(int j = 0; j < kScale; ++j)
                {
                    const double prod = static_cast<double>(pa[blk * kScale + j]) *
                                        static_cast<double>(pb[blk * kScale + j]);
                    part += prod;
                    part_abs += std::fabs(prod);
                }
                const double sc = std::ldexp(1.0, ea[blk] + eb[blk]);
                s += part * sc;
                sa += part_abs * sc;
            }
            refs[i] = s;
            sabs[i] = sa;
        });
        const double ref_s =
            std::chrono::duration<double>(std::chrono::steady_clock::now() - t_ref).count();

        constexpr double u  = unit_roundoff<CDataType>();
        size_t bad          = 0;
        size_t exact_points = 0;
        double max_abs      = 0;
        for(size_t i = 0; i < pts.size(); ++i)
        {
            const auto [m, n]       = pts[i];
            const CDataType got     = first[static_cast<size_t>(m) * p.N + n];
            const double gotf       = ck_tile::type_convert<float>(got);
            const double err        = std::fabs(gotf - refs[i]);
            const bool exact_sparse = sparse && sabs[i] <= sparse_exact_limit<CDataType>();
            exact_points += exact_sparse;
            const bool ok = exact_sparse
                                ? ck_tile::bit_cast<uint16_t>(got) == c_bits<CDataType>(refs[i])
                                : err <= std::fabs(refs[i]) * u + k_batch * sabs[i] * u;
            max_abs       = std::max(max_abs, err);
            if(!ok && bad++ < 5)
                std::printf("mismatch m=%d n=%d ref=%.9g got=%.9g sum|terms|=%.6g\n",
                            m,
                            n,
                            refs[i],
                            gotf,
                            sabs[i]);
        }
        const bool pass = bad == 0 && (nondeterministic == 0 || k_batch > 1);
        std::printf("{\"type\":\"verify\",\"bench\":\"%s\",\"mode\":\"%s\",\"M\":%d,\"N\":%d,"
                    "\"K\":%d,\"k_batch\":%d,\"seed\":%llu,\"check\":\"%s\",\"checked\":%zu,"
                    "\"exact_checked\":%zu,\"bad\":%zu,\"launches\":%d,"
                    "\"nondeterministic_launches\":%zu,\"max_abs_err\":%.6g,\"ref_seconds\":%.1f,"
                    "\"result\":\"%s\"}\n",
                    bench_name.c_str(),
                    mode.c_str(),
                    p.M,
                    p.N,
                    p.K,
                    k_batch,
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
                    "\"N\":%d,\"K\":%d,\"k_batch\":%d,\"warmup\":%d,\"repeat\":%d,\"gap_ms\":%d,"
                    "\"ms\":%.6f,\"tflops\":%.2f}\n",
                    bench_name.c_str(),
                    mode.c_str(),
                    r,
                    p.M,
                    p.N,
                    p.K,
                    k_batch,
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
                "\"K\":%d,\"k_batch\":%d,\"warmup\":%d,\"repeat\":%d,\"rounds\":%d,\"gap_ms\":%d,"
                "\"median_ms\":%.6f,\"best_ms\":%.6f,\"median_tflops\":%.2f,\"best_tflops\":%.2f,"
                "\"best_over_median\":%.4f}\n",
                bench_name.c_str(),
                mode.c_str(),
                p.M,
                p.N,
                p.K,
                k_batch,
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

template <bool TdmEpi, typename CDataType>
int dispatch_tile(const ck_tile::ArgParser& arg, const std::string& name)
{
    const std::string tile = arg.get_str("tile");
    if(tile == "32x256x256")
        return run<Instance<32, 256, 256, 2, 2, TdmEpi, CDataType>>(arg, name);
    if(tile == "32x128x256")
        return run<Instance<32, 128, 256, 2, 2, TdmEpi, CDataType>>(arg, name);
    if(tile == "64x256x256")
        return run<Instance<64, 256, 256, 2, 2, TdmEpi, CDataType>>(arg, name);
    if(tile == "256x256x256")
        return run<Instance<256, 256, 256, 4, 4, TdmEpi, CDataType>>(arg, name);
    // 16-row tiles: one warp tile in M, the four waves along N (warps 1x4x1)
    if(tile == "16x512x256")
        return run<Instance<16, 512, 256, 1, 4, TdmEpi, CDataType>>(arg, name);
    if(tile == "16x256x256")
        return run<Instance<16, 256, 256, 1, 4, TdmEpi, CDataType>>(arg, name);
    if(tile == "16x128x256")
        return run<Instance<16, 128, 256, 1, 4, TdmEpi, CDataType>>(arg, name);
    if(tile == "16x64x256")
        return run<Instance<16, 64, 256, 1, 4, TdmEpi, CDataType>>(arg, name);
    if(tile == "32x64x256")
        return run<Instance<32, 64, 256, 2, 2, TdmEpi, CDataType>>(arg, name);
    std::printf("unknown -tile=%s (32x256x256, 32x128x256, 32x64x256, 64x256x256, 256x256x256, "
                "16x512x256, 16x256x256, 16x128x256, 16x64x256)\n",
                tile.c_str());
    return 2;
}

template <typename CDataType>
int dispatch_epilogue(const ck_tile::ArgParser& arg, const std::string& name)
{
    const std::string epi = arg.get_str("epilogue");
    if(epi == "tdm")
        return dispatch_tile<true, CDataType>(arg, name);
    if(epi == "cshuffle")
        return dispatch_tile<false, CDataType>(arg, name);
    std::printf("unknown -epilogue=%s (tdm, cshuffle)\n", epi.c_str());
    return 2;
}

int dispatch(const ck_tile::ArgParser& arg)
{
    const std::string c    = arg.get_str("c_dtype");
    const std::string name = "ck_comp_tdm_v3_mxfp8_e8m0x32_rcr_c" + c + "_" + arg.get_str("tile") +
                             "_wt16x16x128_" + arg.get_str("epilogue") +
                             (arg.get_int("k_batch") > 1 ? "_kb" + arg.get_str("k_batch") : "");
    if(c == "fp16")
        return dispatch_epilogue<ck_tile::half_t>(arg, name);
    if(c == "bf16")
        return dispatch_epilogue<ck_tile::bf16_t>(arg, name);
    std::printf("unknown -c_dtype=%s (fp16, bf16)\n", c.c_str());
    return 2;
}

} // namespace

int main(int argc, char* argv[])
{
    ck_tile::ArgParser arg;
    arg.insert("m", "16", "M")
        .insert("n", "49152", "N")
        .insert("k", "32768", "K (a multiple of 128)")
        .insert("c_dtype", "fp16", "C data type: fp16 | bf16")
        .insert("epilogue", "tdm", "tdm | cshuffle (padded; split-K)")
        .insert("tile",
                "32x256x256",
                "32x256x256 | 32x128x256 | 32x64x256 | 64x256x256 | 256x256x256 | 16x512x256 | "
                "16x256x256 | 16x128x256 | 16x64x256")
        .insert("k_batch", "1", "split K into this many atomically added ranges (CShuffle only)")
        .insert("mode", "verify-sparse", "verify-sparse | verify-rand | peak | sustained")
        .insert("data", "rand", "input data for peak/sustained: rand | sparse")
        .insert("seed", "12345", "RNG seed for the inputs")
        .insert("sigma", "1.0", "standard deviation of the rand inputs (before e4m3 rounding)")
        .insert("warmup", "-1", "untimed launches per sample (-1: peak 1, sustained 50)")
        .insert("repeat", "-1", "timed launches per sample (-1: peak 3, sustained 100)")
        .insert("rounds", "-1", "samples (-1: peak 15, sustained 1)")
        .insert("gap_ms", "-1", "idle before each sample in ms (-1: peak 25, sustained 0)")
        .insert("verify_launches", "3", "verify modes: launches; with k_batch 1 all must match")
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
