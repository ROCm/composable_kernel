// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <cstdint>
#include <functional>
#include <vector>

#include <gtest/gtest.h>

#include "ck_tile/core/arch/arch.hpp"
#include "ck_tile/core/arch/mma/utility/tile_distribution_encoding_calculator.hpp"
#include "ck_tile/core/arch/mma/utility/tile_distribution_encoding_register_mapper.hpp"
#include "ck_tile/core/numeric/type_convert.hpp"
#include "ck_tile/core/numeric/vector_type.hpp"
#include "ck_tile/host/hip_check_error.hpp"
#include "ck_tile/host/kernel_launch.hpp"
#include <hip/hip_runtime.h>

#include "../get_cmake_targets_helper.hpp"

namespace mma_pipeline_test {

using namespace ck_tile;
using namespace ck_tile::core::arch;
using namespace ck_tile::core::arch::mma;
using namespace ck_tile::core::arch::testing;

inline bool hipTargetMatchesCmakeTargets(amdgcn_target_id arch)
{
    const auto cmake_targets = getCMakeGpuTargetIds();
    if(cmake_targets.count(arch) == 0)
    {
        // gfx12-generic and gfx11-generic make no difference with the specialized archs.
        // Some CI pipelines make use of that and configure the project with the generic
        // flags besides compiling for (f.e.) gfx1201.
        if(arch >= amdgcn_target_id::GFX1200 && arch <= amdgcn_target_id::GFX12_GENERIC)
        {
            return (cmake_targets.count(amdgcn_target_id::GFX12_GENERIC) > 0);
        }
        else if(arch >= amdgcn_target_id::GFX1100 && arch <= amdgcn_target_id::GFX11_GENERIC)
        {
            return (cmake_targets.count(amdgcn_target_id::GFX11_GENERIC) > 0);
        }
    }
    return true;
}
template <typename CType, typename AType, typename BType>
void reference_matmul(std::vector<CType>& C,
                      const std::vector<AType>& A,
                      const std::vector<BType>& B,
                      uint32_t M,
                      uint32_t N,
                      uint32_t K)
{
    for(uint32_t m = 0; m < M; ++m)
    {
        for(uint32_t n = 0; n < N; ++n)
        {
            float acc = 0.0f;
            for(uint32_t k = 0; k < K; ++k)
            {
                acc += type_convert<float>(A[m * K + k]) * type_convert<float>(B[k * N + n]);
            }
            C[m * N + n] = static_cast<CType>(acc);
        }
    }
}

template <typename T>
T deterministic_value(uint32_t row, uint32_t col, uint32_t minor_dim)
{
    float v = static_cast<float>((row * minor_dim + col) % 7 + 1) * 0.25f;
    return type_convert<T>(v);
}

// Apply 2:4 sparsity pattern to A matrix in-place (for sparse pipeline tests).
// Every group of 4 consecutive K elements keeps slots 0 and 2, zeros slots 1 and 3.
template <typename T>
void apply_sparse_pattern(std::vector<T>& A, uint32_t M, uint32_t K)
{
    for(uint32_t m = 0; m < M; ++m)
    {
        for(uint32_t k = 0; k < K; k += 4)
        {
            // Keep slots 0, 2. Zero out slots 1, 3.
            if(k + 1 < K)
                A[m * K + k + 1] = static_cast<T>(0);
            if(k + 3 < K)
                A[m * K + k + 3] = static_cast<T>(0);
        }
    }
}

// Fill per-lane A fragments from logical A[M][K] matrix.
// For dense pipelines: AVecType = InternalAVecT[FragsM][FragsK]
// For sparse pipelines: AVecType = ExternalAFragVecT[FragsM][FragsK] (uncompressed)
template <typename Pipeline, typename AScalar>
void fill_a_fragments(typename Pipeline::AVecType* a_per_lane,
                      const std::vector<AScalar>& A_matrix,
                      uint32_t K,
                      uint32_t waveSize)
{
    using MmaOp       = typename Pipeline::MmaOp;
    using ARegMap     = TileDistrEncRegMap<typename TileDistrEncCalc<MmaOp>::AWarpDstrEncoding>;
    using AFragScalar = typename vector_traits<typename MmaOp::AVecType>::scalar_type;

    constexpr uint32_t FragM  = Pipeline::FragM;
    constexpr uint32_t FragK  = Pipeline::FragK;
    constexpr uint32_t FragsM = Pipeline::FragsM;
    constexpr uint32_t FragsK = Pipeline::FragsK;

    constexpr uint32_t kCompressionRatio = MmaOp::kCompressionRatio;

    // The A register map maps (lane, vec_idx) -> (m_within_frag, k_within_frag)
    // For sparse: k_within_frag is in the compressed K domain (K / kCompressionRatio)
    constexpr index_t a_vec_size               = ARegMap::num_vector_items;
    constexpr index_t external_a_frag_vec_size = a_vec_size * kCompressionRatio;

    for(uint32_t lane = 0; lane < waveSize; ++lane)
    {
        auto* lane_a = reinterpret_cast<AFragScalar*>(&a_per_lane[lane]);

        for(uint32_t bm = 0; bm < FragsM; ++bm)
        {
            for(uint32_t bk = 0; bk < FragsK; ++bk)
            {
                uint32_t frag_offset = (bm * FragsK + bk) * external_a_frag_vec_size;

                if constexpr(kCompressionRatio > 1)
                {
                    // Sparse: fill external (uncompressed) vector
                    for(index_t ev = 0; ev < external_a_frag_vec_size; ++ev)
                    {
                        index_t compressed_v = ev / kCompressionRatio;
                        index_t sub_pos      = ev % kCompressionRatio;

                        auto coords =
                            ARegMap::calc_matrix_indices_from_lane_vector(lane, compressed_v);
                        uint32_t m_local      = coords[0];
                        uint32_t k_compressed = coords[1];
                        uint32_t k_local      = k_compressed * kCompressionRatio + sub_pos;

                        uint32_t m_global = bm * FragM + m_local;
                        uint32_t k_global = bk * FragK + k_local;

                        lane_a[frag_offset + ev] =
                            static_cast<AFragScalar>(A_matrix[m_global * K + k_global]);
                    }
                }
                else
                {
                    // Dense/Scale: direct mapping
                    for(index_t v = 0; v < a_vec_size; ++v)
                    {
                        auto coords      = ARegMap::calc_matrix_indices_from_lane_vector(lane, v);
                        uint32_t m_local = coords[0];
                        uint32_t k_local = coords[1];

                        uint32_t m_global = bm * FragM + m_local;
                        uint32_t k_global = bk * FragK + k_local;

                        lane_a[frag_offset + v] =
                            static_cast<AFragScalar>(A_matrix[m_global * K + k_global]);
                    }
                }
            }
        }
    }
}

// Fill per-lane B fragments from logical B[K][N] matrix.
// BVecType = InternalBVecT[FragsN][FragsK]
template <typename Pipeline, typename BScalar>
void fill_b_fragments(typename Pipeline::BVecType* b_per_lane,
                      const std::vector<BScalar>& B_matrix,
                      uint32_t N,
                      uint32_t waveSize)
{
    using MmaOp       = typename Pipeline::MmaOp;
    using BRegMap     = TileDistrEncRegMap<typename TileDistrEncCalc<MmaOp>::BWarpDstrEncoding>;
    using BFragScalar = typename vector_traits<typename MmaOp::BVecType>::scalar_type;

    constexpr uint32_t FragN  = Pipeline::FragN;
    constexpr uint32_t FragK  = Pipeline::FragK;
    constexpr uint32_t FragsN = Pipeline::FragsN;
    constexpr uint32_t FragsK = Pipeline::FragsK;

    constexpr index_t b_vec_size = BRegMap::num_vector_items;

    for(uint32_t lane = 0; lane < waveSize; ++lane)
    {
        auto* lane_b = reinterpret_cast<BFragScalar*>(&b_per_lane[lane]);

        for(uint32_t bn = 0; bn < FragsN; ++bn)
        {
            for(uint32_t bk = 0; bk < FragsK; ++bk)
            {
                uint32_t frag_offset = (bn * FragsK + bk) * b_vec_size;

                for(index_t v = 0; v < b_vec_size; ++v)
                {
                    auto coords      = BRegMap::calc_matrix_indices_from_lane_vector(lane, v);
                    uint32_t n_local = coords[0];
                    uint32_t k_local = coords[1];

                    uint32_t n_global = bn * FragN + n_local;
                    uint32_t k_global = bk * FragK + k_local;

                    // B matrix is stored as B[K][N]
                    lane_b[frag_offset + v] =
                        static_cast<BFragScalar>(B_matrix[k_global * N + n_global]);
                }
            }
        }
    }
}

// Extract C matrix from per-lane C fragments.
// CVecType = InternalCVecT[FragsM][FragsN]
template <typename Pipeline, typename CScalar>
void extract_c_matrix(const typename Pipeline::CVecType* c_per_lane,
                      std::vector<CScalar>& C_matrix,
                      uint32_t N,
                      uint32_t waveSize)
{
    using MmaOp       = typename Pipeline::MmaOp;
    using CRegMap     = TileDistrEncRegMap<typename TileDistrEncCalc<MmaOp>::CWarpDstrEncoding>;
    using CFragScalar = typename vector_traits<typename MmaOp::CVecType>::scalar_type;

    constexpr uint32_t FragM  = Pipeline::FragM;
    constexpr uint32_t FragN  = Pipeline::FragN;
    constexpr uint32_t FragsM = Pipeline::FragsM;
    constexpr uint32_t FragsN = Pipeline::FragsN;

    constexpr index_t c_vec_size = CRegMap::num_vector_items;

    for(uint32_t lane = 0; lane < waveSize; ++lane)
    {
        auto* lane_c = reinterpret_cast<const CFragScalar*>(&c_per_lane[lane]);

        for(uint32_t bm = 0; bm < FragsM; ++bm)
        {
            for(uint32_t bn = 0; bn < FragsN; ++bn)
            {
                uint32_t frag_offset = (bm * FragsN + bn) * c_vec_size;

                for(index_t v = 0; v < c_vec_size; ++v)
                {
                    auto coords      = CRegMap::calc_matrix_indices_from_lane_vector(lane, v);
                    uint32_t m_local = coords[0];
                    uint32_t n_local = coords[1];

                    uint32_t m_global = bm * FragM + m_local;
                    uint32_t n_global = bn * FragN + n_local;

                    C_matrix[m_global * N + n_global] =
                        static_cast<CScalar>(lane_c[frag_offset + v]);
                }
            }
        }
    }
}

/// Internal: runs the test with a fully resolved Pipeline type.
/// Called from run_pipeline_matrix_test after dispatching on compiler target.
template <typename Pipeline,
          typename KernelType,
          typename AScalar = fp16_t,
          typename BScalar = fp16_t,
          typename CScalar = fp32_t>
void run_pipeline_matrix_test_impl(uint32_t M,
                                   uint32_t N,
                                   uint32_t K,
                                   uint32_t waveSize,
                                   KernelType kernel,
                                   bool isSparse,
                                   bool transposeExpected = false,
                                   float referenceScale   = 1.0f)
{
    std::vector<AScalar> A_matrix(M * K);
    std::vector<BScalar> B_matrix(K * N);
    std::vector<CScalar> C_expected(M * N, static_cast<CScalar>(0));
    std::vector<CScalar> C_actual(M * N, static_cast<CScalar>(0));

    for(uint32_t m = 0; m < M; ++m)
        for(uint32_t k = 0; k < K; ++k)
            A_matrix[m * K + k] = deterministic_value<AScalar>(m, k, K);

    for(uint32_t k = 0; k < K; ++k)
        for(uint32_t n = 0; n < N; ++n)
            B_matrix[k * N + n] = deterministic_value<BScalar>(k, n, N);

    if(isSparse)
    {
        apply_sparse_pattern(A_matrix, M, K);
    }

    reference_matmul(C_expected, A_matrix, B_matrix, M, N, K);

    using AVecType = typename Pipeline::AVecType;
    using BVecType = typename Pipeline::BVecType;
    using CVecType = typename Pipeline::CVecType;

    const size_t a_buf_size = waveSize * sizeof(AVecType);
    const size_t b_buf_size = waveSize * sizeof(BVecType);
    const size_t c_buf_size = waveSize * sizeof(CVecType);

    std::vector<uint8_t> h_a(a_buf_size, 0);
    std::vector<uint8_t> h_b(b_buf_size, 0);
    std::vector<uint8_t> h_c(c_buf_size, 0);

    fill_a_fragments<Pipeline>(reinterpret_cast<AVecType*>(h_a.data()), A_matrix, K, waveSize);
    fill_b_fragments<Pipeline>(reinterpret_cast<BVecType*>(h_b.data()), B_matrix, N, waveSize);

    void *d_a, *d_b, *d_c;
    HIP_CHECK_ERROR(hipMalloc(&d_a, a_buf_size));
    HIP_CHECK_ERROR(hipMalloc(&d_b, b_buf_size));
    HIP_CHECK_ERROR(hipMalloc(&d_c, c_buf_size));

    HIP_CHECK_ERROR(hipMemcpy(d_a, h_a.data(), a_buf_size, hipMemcpyHostToDevice));
    HIP_CHECK_ERROR(hipMemcpy(d_b, h_b.data(), b_buf_size, hipMemcpyHostToDevice));
    HIP_CHECK_ERROR(hipMemset(d_c, 0, c_buf_size));

    ck_tile::launch_kernel(ck_tile::stream_config{},
                           ck_tile::make_kernel(kernel, dim3(1), dim3(waveSize), 0, d_a, d_b, d_c));
    HIP_CHECK_ERROR(hipDeviceSynchronize());

    HIP_CHECK_ERROR(hipMemcpy(h_c.data(), d_c, c_buf_size, hipMemcpyDeviceToHost));
    extract_c_matrix<Pipeline>(
        reinterpret_cast<const CVecType*>(h_c.data()), C_actual, N, waveSize);

    for(uint32_t m = 0; m < M; ++m)
    {
        for(uint32_t n = 0; n < N; ++n)
        {
            // When transposeExpected is true, the kernel computes C^T via SwapAB,
            // so compare actual C[m][n] against reference C[n][m].
            constexpr float relative_tolerance = 1e-2f;
            constexpr float absolute_tolerance = 1e-3f;

            float expected = transposeExpected ? static_cast<float>(C_expected[n * M + m])
                                               : static_cast<float>(C_expected[m * N + n]);
            expected *= referenceScale;
            float actual = static_cast<float>(C_actual[m * N + n]);
            EXPECT_NEAR(
                actual, expected, std::abs(expected) * relative_tolerance + absolute_tolerance)
                << "Mismatch at C[" << m << "][" << n << "]";
        }
    }

    HIP_CHECK_ERROR(hipFree(d_a));
    HIP_CHECK_ERROR(hipFree(d_b));
    HIP_CHECK_ERROR(hipFree(d_c));
}

/// @tparam PipelineFactory A template template that, given a CompilerTarget type, produces
///                         the Pipeline type:  PipelineFactory<Target>::type
/// @tparam KernelType      Kernel functor struct with kBlockSize and __device__ operator()
/// @tparam AScalar         Scalar type for A matrix (e.g., fp16_t)
/// @tparam BScalar         Scalar type for B matrix (e.g., fp16_t)
/// @tparam CScalar         Scalar type for C matrix (e.g., fp32_t)
/// @param  M               WaveTile M dimension
/// @param  N               WaveTile N dimension
/// @param  K               WaveTile K dimension
/// @param  shouldSkip      Predicate returning true if current device should skip
/// @param  kernel          Kernel functor instance to launch via make_kernel
/// @param  isSparse        Whether to apply 2:4 sparsity pattern to A
/// @param  transposeExpected When true, compare against transposed reference (for
/// SwapAB/TransposeC)
/// @param  referenceScale  Scalar multiplier applied to the reference matmul result before
///                         comparison (e.g., to account for scale-MMA scaling factors)
template <template <typename> class PipelineFactory,
          typename KernelType,
          typename AScalar = fp16_t,
          typename BScalar = fp16_t,
          typename CScalar = fp32_t>
void run_pipeline_matrix_test(uint32_t M,
                              uint32_t N,
                              uint32_t K,
                              std::function<bool(ck_tile::core::arch::amdgcn_target_id)> shouldSkip,
                              KernelType kernel,
                              bool isSparse          = false,
                              bool transposeExpected = false,
                              float referenceScale   = 1.0f)
{
    int devCount;
    hipDevice_t dev;
    HIP_CHECK_ERROR(hipGetDevice(&dev));
    HIP_CHECK_ERROR(hipGetDeviceCount(&devCount));

    hipDeviceProp_t devProp;
    HIP_CHECK_ERROR(hipGetDeviceProperties(&devProp, dev));

    auto currentArchId = hip_device_prop_gcn_arch_name_to_amdgcn_target_id(devProp.gcnArchName);
    if(devCount <= 0 || shouldSkip(currentArchId))
    {
        GTEST_SKIP() << "No HIP device found or arch (0x" << std::hex
                     << static_cast<int>(currentArchId) << ") not supported. Skipping test.";
    }

    if(!hipTargetMatchesCmakeTargets(currentArchId))
    {
        std::cout << "The GPU targets exposed by CMake are: ";
        for(const auto& target : getCMakeGpuTargetIds())
        {
            std::cout << "(0x" << std::hex << static_cast<int>(target) << ")\n";
        }
        FAIL() << "The HIP device (0x" << std::hex << static_cast<int>(currentArchId)
               << ") does not match the compiler target(s).";
    }

    const uint32_t waveSize = static_cast<uint32_t>(devProp.warpSize);

    bool dispatched = dispatchCompilerTarget(currentArchId, [&](auto target) {
        using CompilerTarget = decltype(target);
        using Pipeline       = typename PipelineFactory<CompilerTarget>::type;

        run_pipeline_matrix_test_impl<Pipeline, KernelType, AScalar, BScalar, CScalar>(
            M, N, K, waveSize, kernel, isSparse, transposeExpected, referenceScale);
    });

    if(!dispatched)
    {
        GTEST_SKIP() << "Cannot dispatch on HOST target.";
    }
}

} // namespace mma_pipeline_test
