// Stage-1 (2:4-sparse CK library-grade GEMM feasibility, 2026-07-20):
// warp-tile correctness proof for CK's ck_tile::SparseMmaPipeline
// (int8_t x int8_t -> int32_t, 16x16x32 WMMA SWMMAC, gfx1201), extending
// Stage-8's "raw atom executes" finding to "a validated PIPELINE produces
// correct 2:4-sparse GEMM results, checked against a CPU reference".
//
// This is a non-gtest standalone port of CK's OWN test methodology
// (test/ck_tile/core/arch/mma/pipeline/test_amdgcn_sparse_mma.cpp +
// pipeline_tests_helper.hpp), NOT independently re-derived -- reusing
// AMD's validated register-mapping logic (TileDistrEncRegMap) is much
// lower-risk than hand-deriving the per-lane sparse layout from scratch,
// and is exactly the "library value" this Stage-1 gate is asking about:
// does CK's own machinery work correctly for the SPARSE MmaOpFamily on
// gfx1201, not just compile.
//
// Data flow tested: dense A (int8, with 2:4 structured zeros already
// applied per-4-group -- CK's own convention: A is provided UNCOMPRESSED,
// SparseCompressTransform runs ON THE FLY inside Pipeline::exec()) x dense
// B -> int32 C, ONE warp, ONE 16x16x{32,64,128} tile.
#include "ck_tile/core/arch/arch.hpp"
#include "ck_tile/core/arch/mma/mma.hpp"
#include "ck_tile/core/arch/mma/sparse/sparse_mma_pipeline.hpp"
#include "ck_tile/core/arch/mma/utility/tile_distribution_encoding_calculator.hpp"
#include "ck_tile/core/arch/mma/utility/tile_distribution_encoding_register_mapper.hpp"
#include "ck_tile/core/numeric/integer.hpp"
#include "ck_tile/core/numeric/type_convert.hpp"
#include "ck_tile/core/numeric/vector_type.hpp"
#include "ck_tile/host/hip_check_error.hpp"
#include "ck_tile/host/kernel_launch.hpp"
#include "ck_tile/host/stream_config.hpp"

#include <hip/hip_runtime.h>

#include "real_a_tile.h"  // REAL_A_16x128: Quark int8 2:4 weight tile (arbitrary-position zeros)

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <random>
#include <algorithm>
#include <cmath>

using namespace ck_tile;
using namespace ck_tile::core::arch;
using namespace ck_tile::core::arch::mma;

// CRITICAL fix (found the hard way, matches this fork's own already-known
// __GFX12__/RDNA4 macro-visibility lesson): SparseMmaPipeline's DEFAULT
// CompilerTarget template param (getCMakeCompilerTarget()) resolves via
// the __gfx1201__ preprocessor macro, which clang's HIP frontend only
// predefines during the DEVICE compilation pass of a TU -- NEVER during
// the HOST pass. Any HOST-side code (like this file's run_test(), which
// computes sizeof(AWarpTensor) etc. to size/fill host buffers before
// upload) that instantiates SparseMmaPipeline<...> with the DEFAULTED
// target silently gets HOST-context resolution (__gfx1201__ undefined ->
// CK_TILE_ARCH_GFX1201=false -> falls through to an UNSUPPORTED dummy
// MmaOp with kM=kN=kK=1, kCompressionRatio=1) -- a COMPLETELY DIFFERENT,
// much smaller, wrong-shaped type than what the DEVICE kernel body (same
// source, device pass, __gfx1201__ defined) resolves. Host buffer sizing
// mismatched against device tensor layout -> heap corruption (confirmed:
// this exact bug crashed the first version of this probe with a glibc
// malloc assertion failure). FIX: specify CompilerTarget EXPLICITLY
// (a pure compile-time type, not macro-dependent) everywhere -- exactly
// what CK's OWN test file does (CompilerTargetGfx950 = decltype(...)),
// which I initially skipped by relying on the convenience default. Not a
// CK bug -- a usage requirement whenever a Pipeline type crosses the
// host/device boundary in one TU.
using Gfx1201Target = decltype(make_amdgcn_gfx12_target<amdgcn_target_id::GFX1201>());

// --- Reference (CPU, int64 accumulate -- exact for int8 x int8 up to K=~2^47, no overflow risk) ---
static void reference_matmul_i8(std::vector<int32_t> & C, const std::vector<int8_t> & A,
                                 const std::vector<int8_t> & B, uint32_t M, uint32_t N, uint32_t K) {
    for (uint32_t m = 0; m < M; ++m) {
        for (uint32_t n = 0; n < N; ++n) {
            int64_t acc = 0;
            for (uint32_t k = 0; k < K; ++k) {
                acc += (int64_t) A[m * K + k] * (int64_t) B[k * N + n];
            }
            C[m * N + n] = (int32_t) acc;
        }
    }
}

// Apply 2:4 sparsity: every group of 4 consecutive K elements keeps slots
// 0,2, zeros slots 1,3 (matches CK's own test convention exactly).
static void apply_sparse_pattern(std::vector<int8_t> & A, uint32_t M, uint32_t K) {
    for (uint32_t m = 0; m < M; ++m) {
        for (uint32_t k = 0; k < K; k += 4) {
            if (k + 1 < K) A[m * K + k + 1] = 0;
            if (k + 3 < K) A[m * K + k + 3] = 0;
        }
    }
}

template <typename Pipeline>
struct SparseGemmKernel {
    static constexpr int kBlockSize = 32; // wave32 on gfx1201

    __device__ void operator()(const void * a_per_lane, const void * b_per_lane, void * c_per_lane) const {
        using ATensor = typename Pipeline::AWarpTensor;
        using BTensor = typename Pipeline::BWarpTensor;
        using CTensor = typename Pipeline::CWarpTensor;

        const uint32_t lane = threadIdx.x;

        ATensor a;
        BTensor b;
        CTensor c;
        __builtin_memcpy(&a, static_cast<const uint8_t *>(a_per_lane) + lane * sizeof(ATensor), sizeof(ATensor));
        __builtin_memcpy(&b, static_cast<const uint8_t *>(b_per_lane) + lane * sizeof(BTensor), sizeof(BTensor));
        __builtin_memset(&c, 0, sizeof(CTensor));

        if constexpr (MmaOpTraits<typename Pipeline::MmaOp>::IsSupported) {
#ifdef DEBUG_DEVICE
            if (lane == 0) {
                ATensor a_dbg = a;
                constexpr index_t VecN = ATensor::get_thread_buffer_size();
                using RawVec = ext_vector_t<int8_t, VecN>;
                auto & raw = a_dbg.get_thread_buffer().template get_as<RawVec>().template at<0>();
                printf("  [DEV lane0] raw uncompressed a_vec (VecN=%d):", (int)VecN);
                for (index_t i = 0; i < VecN; ++i) printf(" %d", (int)raw[i]);
                printf("\n");
                auto ab_pair = Pipeline::ATransform::execExtVec(raw);
                auto & compressed = std::get<0>(ab_pair);
                auto & idxpk = std::get<1>(ab_pair);
                using CompVecTraits = vector_traits<std::remove_reference_t<decltype(compressed)>>;
                printf("  [DEV lane0] compressed a_vec (size=%d):", (int)CompVecTraits::vector_size);
                for (index_t i = 0; i < CompVecTraits::vector_size; ++i) printf(" %d", (int)compressed[i]);
                printf("\n  [DEV lane0] idx words:");
                for (auto w : idxpk.words) printf(" 0x%08x", (unsigned)w);
                printf("\n");
            }
#endif
            Pipeline::exec(a, b, c);
            __builtin_memcpy(static_cast<uint8_t *>(c_per_lane) + lane * sizeof(CTensor), &c, sizeof(CTensor));
        }
    }
};

// Per-lane layout construction, ported from pipeline_tests_helper.hpp's
// fill_a_fragments/fill_b_fragments/extract_c_matrix (AMD's own validated
// register-map logic -- TileDistrEncRegMap -- reused verbatim, not
// re-derived).
// FIX (Stage-17b): the old version rebuilt the register map from a *bare*
// TileDistrEncCalc<MmaOp> (all defaults: UncompressedA=false, kIter=1), which
// yields the COMPRESSED-A encoding (K dimension already halved by
// kCompressionRatio) -- the layout the *intrinsic* consumes post-compression,
// not the layout the *pipeline* expects the caller to supply pre-compression.
// It then hand-expanded that compressed coordinate back out via
// `k_local = k_compressed * kCompressionRatio + sub_pos`, which silently
// assumes the compressed-K stride advances in lockstep, whole-4-group chunks
// -- true only for the "zeros at slots 1,3" synthetic pattern, not for
// arbitrary-position 2:4 data (where compress_a_impl's own group-of-4 scan
// must see the TRUE contiguous [k*4 .. k*4+3] slice).
//
// SparseMmaPipeline already computes and PUBLICLY EXPOSES exactly the right
// encoding for this: `Pipeline::AWarpDstrEncoding` is built internally via
// `EncCalc = TileDistrEncCalc<MmaOp, CTranspose, SwizzleFactor, FragsK,
// AttrNumAccessAV, AttrNumAccessBV, /*UncompressedA=*/true>` (see
// sparse_mma_pipeline.hpp). With UncompressedA=true the K sub-dimension is
// NOT divided by kCompressionRatio, so `calc_matrix_indices_from_lane_vector`
// returns the GLOBAL, uncompressed (m, k) coordinate directly -- already in
// the exact per-lane flat vector order that SparseCompressTransform::exec()
// (and hence compress_a_impl's real contiguous-4 grouping) expects. Using
// Pipeline's own type instead of re-deriving it removes the bug at the root:
// no more manual compression-ratio math, no bm/bk frag loop needed (the K
// sub-dim already spans the WHOLE WaveTileK once kIter=FragsK is baked in).
template <typename Pipeline>
static void fill_a_fragments(typename Pipeline::AWarpTensor * a_per_lane,
                              const std::vector<int8_t> & A_matrix, uint32_t K, uint32_t waveSize) {
    using ARegMap = TileDistrEncRegMap<typename Pipeline::AWarpDstrEncoding>;
    using AFragScalar = typename Pipeline::ADataType;
    constexpr index_t a_vec_size = ARegMap::num_vector_items; // full uncompressed per-lane count

    for (uint32_t lane = 0; lane < waveSize; ++lane) {
        auto * lane_a = reinterpret_cast<AFragScalar *>(&a_per_lane[lane]);
        for (index_t v = 0; v < a_vec_size; ++v) {
            auto coords = ARegMap::calc_matrix_indices_from_lane_vector(lane, v);
            uint32_t m_global = coords[0];
            uint32_t k_global = coords[1];
            lane_a[v] = static_cast<AFragScalar>(A_matrix[m_global * K + k_global]);
#ifdef DEBUG_FILL_A
            if (m_global == 0 && K == 32) {
                printf("  [DBG m=0] lane=%2u v=%2d -> k=%2u val=%d\n", lane, (int)v, k_global, (int)lane_a[v]);
            }
#endif
        }
    }
}

template <typename Pipeline>
static void fill_b_fragments(typename Pipeline::BWarpTensor * b_per_lane,
                              const std::vector<int8_t> & B_matrix, uint32_t N, uint32_t waveSize) {
    // Same root-cause class as A (see fill_a_fragments comment): a bare
    // TileDistrEncCalc<MmaOp> defaults to kIter=1, describing ONE MmaOp::kK
    // fragment's worth of B. SparseMmaPipeline's real EncCalc uses kIter=FragsK
    // (baked into Pipeline::BWarpDstrEncoding), so its K sub-dim already spans
    // the WHOLE WaveTileK -- the old per-fragment bn/bk loop with frag_offset
    // happened to reproduce the right flat order only in combination with the
    // old (also-mismatched) A fill; once A is fixed to match Pipeline's true
    // encoding, B must match it too or the two mismatches stop canceling out.
    using BRegMap = TileDistrEncRegMap<typename Pipeline::BWarpDstrEncoding>;
    using BFragScalar = typename Pipeline::BDataType;
    constexpr index_t b_vec_size = BRegMap::num_vector_items; // spans the full WaveTileK already

    for (uint32_t lane = 0; lane < waveSize; ++lane) {
        auto * lane_b = reinterpret_cast<BFragScalar *>(&b_per_lane[lane]);
        for (index_t v = 0; v < b_vec_size; ++v) {
            auto coords = BRegMap::calc_matrix_indices_from_lane_vector(lane, v);
            uint32_t n_global = coords[0];
            uint32_t k_global = coords[1];
            lane_b[v] = static_cast<BFragScalar>(B_matrix[k_global * N + n_global]);
        }
    }
}

template <typename Pipeline>
static void extract_c_matrix(const typename Pipeline::CWarpTensor * c_per_lane,
                              std::vector<int32_t> & C_matrix, uint32_t N, uint32_t waveSize) {
    // C's encoding doesn't depend on kIter (see get_cwarp_dstr_encoding()), so this
    // was not actually part of the bug -- switched to Pipeline's own alias anyway
    // for consistency/robustness with A and B above.
    using CRegMap = TileDistrEncRegMap<typename Pipeline::CWarpDstrEncoding>;
    using CFragScalar = typename Pipeline::CDataType;

    constexpr uint32_t FragM = Pipeline::MmaOp::kM;
    constexpr uint32_t FragN = Pipeline::MmaOp::kN;
    constexpr uint32_t FragsM = Pipeline::FragsM;
    constexpr uint32_t FragsN = Pipeline::FragsN;
    constexpr index_t c_vec_size = CRegMap::num_vector_items;

    for (uint32_t lane = 0; lane < waveSize; ++lane) {
        auto * lane_c = reinterpret_cast<const CFragScalar *>(&c_per_lane[lane]);
        for (uint32_t bm = 0; bm < FragsM; ++bm) {
            for (uint32_t bn = 0; bn < FragsN; ++bn) {
                uint32_t frag_offset = (bm * FragsN + bn) * c_vec_size;
                for (index_t v = 0; v < c_vec_size; ++v) {
                    auto coords = CRegMap::calc_matrix_indices_from_lane_vector(lane, v);
                    uint32_t m_local = coords[0];
                    uint32_t n_local = coords[1];
                    uint32_t m_global = bm * FragM + m_local;
                    uint32_t n_global = bn * FragN + n_local;
                    C_matrix[m_global * N + n_global] = static_cast<int32_t>(lane_c[frag_offset + v]);
                }
            }
        }
    }
}

template <uint32_t WaveTileM, uint32_t WaveTileN, uint32_t WaveTileK>
static bool run_test(const char * label) {
    using Pipeline = SparseMmaPipeline<int8_t, int8_t, int32_t, WaveTileM, WaveTileN, WaveTileK,
                                        MmaAccumPolicy::ROW_MAJOR, false, 1, 1, 1, Gfx1201Target>;
    using AWarpTensor = typename Pipeline::AWarpTensor;
    using BWarpTensor = typename Pipeline::BWarpTensor;
    using CWarpTensor = typename Pipeline::CWarpTensor;

    const uint32_t M = WaveTileM, N = WaveTileN, K = WaveTileK, waveSize = 32;

    std::mt19937 rng(42);
    std::uniform_int_distribution<int> dist(-8, 8); // small range, avoids int8 overflow noise
    std::vector<int8_t> A(M * K), B(K * N);
#ifdef USE_REAL_TILE
    // A = REAL Quark int8 2:4 weight tile (q_proj L0, arbitrary-position zeros).
    // Tests whether CK's on-the-fly SparseCompressTransform handles genuine 2:4
    // (zeros anywhere in each 4-group). REAL_A_16x128 is row-major [16][128].
    for (uint32_t m = 0; m < M; ++m)
        for (uint32_t k = 0; k < K; ++k)
            A[m * K + k] = REAL_A_16x128[m * 128 + k];
    for (auto & v : B) v = (int8_t) dist(rng);
#else
    // Synthetic control: zeros in slots 1,3 (CK's assumed kept-pattern 0,2).
    for (auto & v : A) v = (int8_t) dist(rng);
    for (auto & v : B) v = (int8_t) dist(rng);
    apply_sparse_pattern(A, M, K);
#endif

    std::vector<int32_t> C_expected(M * N, 0), C_actual(M * N, 0);
    reference_matmul_i8(C_expected, A, B, M, N, K);
#ifdef DEBUG_DEVICE
    if (K == 32) {
        printf("  [HOST] B column 0, k=0..31:");
        for (uint32_t k = 0; k < K; ++k) printf(" %d", (int)B[k * N + 0]);
        printf("\n");
    }
#endif

    const size_t a_buf_size = waveSize * sizeof(AWarpTensor);
    const size_t b_buf_size = waveSize * sizeof(BWarpTensor);
    const size_t c_buf_size = waveSize * sizeof(CWarpTensor);
    std::vector<uint8_t> h_a(a_buf_size, 0), h_b(b_buf_size, 0), h_c(c_buf_size, 0);

    fill_a_fragments<Pipeline>(reinterpret_cast<AWarpTensor *>(h_a.data()), A, K, waveSize);
    fill_b_fragments<Pipeline>(reinterpret_cast<BWarpTensor *>(h_b.data()), B, N, waveSize);

    void *d_a, *d_b, *d_c;
    HIP_CHECK_ERROR(hipMalloc(&d_a, a_buf_size));
    HIP_CHECK_ERROR(hipMalloc(&d_b, b_buf_size));
    HIP_CHECK_ERROR(hipMalloc(&d_c, c_buf_size));
    HIP_CHECK_ERROR(hipMemcpy(d_a, h_a.data(), a_buf_size, hipMemcpyHostToDevice));
    HIP_CHECK_ERROR(hipMemcpy(d_b, h_b.data(), b_buf_size, hipMemcpyHostToDevice));
    HIP_CHECK_ERROR(hipMemset(d_c, 0, c_buf_size));

    ck_tile::launch_kernel(ck_tile::stream_config{},
        ck_tile::make_kernel(SparseGemmKernel<Pipeline>{}, dim3(1), dim3(waveSize), 0, d_a, d_b, d_c));
    HIP_CHECK_ERROR(hipDeviceSynchronize());

    HIP_CHECK_ERROR(hipMemcpy(h_c.data(), d_c, c_buf_size, hipMemcpyDeviceToHost));
    extract_c_matrix<Pipeline>(reinterpret_cast<const CWarpTensor *>(h_c.data()), C_actual, N, waveSize);

    HIP_CHECK_ERROR(hipFree(d_a));
    HIP_CHECK_ERROR(hipFree(d_b));
    HIP_CHECK_ERROR(hipFree(d_c));

    long max_abs_err = 0;
    for (size_t i = 0; i < C_actual.size(); ++i) {
        max_abs_err = std::max(max_abs_err, std::labs((long) C_actual[i] - (long) C_expected[i]));
    }
    const bool pass = (max_abs_err == 0);
    printf("[%s] M=%u N=%u K=%u -> max_abs_err=%ld %s\n", label, M, N, K, max_abs_err, pass ? "PASS" : "FAIL");
    if (!pass) {
        printf("  sample: C_expected[0]=%d C_actual[0]=%d\n", C_expected[0], C_actual[0]);
    }
    return pass;
}

int main() {
    printf("=== Stage-1: ck_tile::SparseMmaPipeline<int8_t,int8_t,int32_t> warp-tile correctness (gfx1201 WMMA SWMMAC) ===\n");
    bool all_pass = true;
    all_pass &= run_test<16, 16, 32>("K32_single_frag");
    all_pass &= run_test<16, 16, 64>("K64_2frag");
    all_pass &= run_test<16, 16, 128>("K128_4frag");
    printf("=== %s ===\n", all_pass ? "ALL PASS" : "SOME FAILED");
    return all_pass ? 0 : 1;
}
