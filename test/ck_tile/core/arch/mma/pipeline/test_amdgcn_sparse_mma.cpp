// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "pipeline_tests_helper.hpp"

#include "ck_tile/core/arch/arch.hpp"
#include "ck_tile/core/arch/mma/mma.hpp"
#include "ck_tile/core/arch/mma/mma_wavewise.hpp"
#include "ck_tile/core/arch/mma/sparse/sparse_mma_pipeline.hpp"
#include "ck_tile/core/numeric/ext_vector_base.hpp"
#include "ck_tile/core/numeric/half.hpp"
#include "ck_tile/core/numeric/integer.hpp"
#include "ck_tile/core/utility/type_traits.hpp"
#include "ck_tile/host/hip_check_error.hpp"
#include "ck_tile/host/kernel_launch.hpp"
#include "ck_tile/host/stream_config.hpp"

#include <gtest/gtest.h>
#include <hip/hip_runtime.h>

#include <tuple>
#include <type_traits>
#include <vector>
#include "ck_tile/core/arch/mma/utility/tile_distribution_encoding_calculator.hpp"
#include "ck_tile/core/arch/mma/utility/tile_distribution_encoding_register_mapper.hpp"
#include "ck_tile/core/numeric/pk_int4.hpp"
#include "ck_tile/core/numeric/type_convert.hpp"
#include "ck_tile/core/numeric/vector_type.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>

using namespace ck_tile;
using namespace ck_tile::core::arch;
using namespace ck_tile::core::arch::mma;

using CompilerTargetGfx950 = decltype(make_amdgcn_gfx9_target<amdgcn_target_id::GFX950>());

TEST(SparseMMATrait, SparseMfmaGfx950Specialization)
{
    // Test fp16 -> fp32 sparse MFMA for GFX950 (16x16x32)
    using TestSparseMfma16x16 = amdgcn_mma<fp16_t,
                                           fp16_t,
                                           fp32_t,
                                           16u,
                                           16u,
                                           32u,
                                           CompilerTargetGfx950,
                                           MmaOpFamily::SPARSE>;

    EXPECT_TRUE((std::is_same_v<typename TestSparseMfma16x16::OpType, MfmaOp> &&
                 TestSparseMfma16x16::OpFamily == MmaOpFamily::SPARSE))
        << "GFX950 sparse 16x16x32 should have SparseMFMAOp type";

    EXPECT_TRUE((is_mma_op_of_family_v<MmaOpFamily::SPARSE, TestSparseMfma16x16>))
        << "GFX950 sparse 16x16x32 should be detected as Sparse";
}

TEST(SparseMMATrait, MmaOpTraitsIntegration)
{
    // Create a sparse MMA op (16x16x32 fp16 specialization)
    using TestSparseMmma = amdgcn_mma<fp16_t,
                                      fp16_t,
                                      fp32_t,
                                      16u,
                                      16u,
                                      32u,
                                      CompilerTargetGfx950,
                                      MmaOpFamily::SPARSE>;

    // Get its traits
    using TestTraits = MmaOpTraits<TestSparseMmma>;

    // Verify trait detection
    EXPECT_TRUE(TestTraits::IsSparse) << "Sparse MMA should be detected as sparse";
    EXPECT_TRUE(TestTraits::IsSupported) << "Sparse MMA specialization should be supported";
    EXPECT_TRUE(TestTraits::IsMfma) << "Sparse MFMA should be detected as MFMA";
    EXPECT_FALSE(TestTraits::IsWmma) << "Sparse MFMA should not be detected as WMMA";
}

TEST(SparseMMATrait, TestConceptRequirements)
{
#if CK_TILE_CONCEPTS && CK_TILE_CONCEPTS_HEADER
    using TestSparseMmma = amdgcn_mma<fp16_t,
                                      fp16_t,
                                      fp32_t,
                                      16u,
                                      16u,
                                      32u,
                                      CompilerTargetGfx950,
                                      MmaOpFamily::SPARSE>;
    EXPECT_TRUE(MmaOpI<TestSparseMmma>);
#else
    GTEST_SKIP() << "Not compiled with concepts. Skipping test.";
#endif // CK_TILE_CONCEPTS && CK_TILE_CONCEPTS_HEADER
}

TEST(SparseMMATrait, DenseVsSparseDistinction)
{
    // Dense MFMA from mfma/mfma_gfx9.hpp
    using DenseMfma =
        amdgcn_mma<fp16_t, fp16_t, fp32_t, 16u, 16u, 32u, CompilerTargetGfx950, MmaOpFamily::DENSE>;

    // Sparse MFMA on GFX950
    using SparseMfma = amdgcn_mma<fp16_t,
                                  fp16_t,
                                  fp32_t,
                                  16u,
                                  16u,
                                  32u,
                                  CompilerTargetGfx950,
                                  MmaOpFamily::SPARSE>;

    // Verify they have different operation types
    EXPECT_TRUE((std::is_same_v<typename DenseMfma::OpType, typename SparseMfma::OpType> &&
                 DenseMfma::OpFamily != SparseMfma::OpFamily))
        << "Dense and Sparse MFMA should have the same OpType tags and different OpFamily";

    // Verify traits correctly identify them
    EXPECT_TRUE((MmaOpTraits<DenseMfma>::IsMfma && MmaOpTraits<DenseMfma>::IsDense &&
                 !MmaOpTraits<DenseMfma>::IsSparse && !MmaOpTraits<DenseMfma>::IsScale &&
                 MmaOpTraits<DenseMfma>::IsSupported))
        << "Dense MFMA should be identified correctly";

    EXPECT_TRUE((MmaOpTraits<SparseMfma>::IsSparse && MmaOpTraits<SparseMfma>::IsMfma &&
                 !MmaOpTraits<SparseMfma>::IsDense && !MmaOpTraits<SparseMfma>::IsScale &&
                 MmaOpTraits<SparseMfma>::IsSupported))
        << "Sparse MFMA should be identified correctly";
}

template <uint32_t CompressionRatio, typename Vec>
struct SparseTransformKernel
{
    static constexpr int kBlockSize = mma_pipeline_test::getCMakeWaveSize();

    __device__ void operator()(void* a, void* idx) const
    {
        using ResultT =
            decltype(SparseCompressTransform<CompressionRatio>::execExtVec(*static_cast<Vec*>(a)));
        using FirstT = std::tuple_element_t<0, ResultT>;
        using IdxT   = std::tuple_element_t<1, ResultT>;
        const auto& [vec, i] =
            SparseCompressTransform<CompressionRatio>::execExtVec(*static_cast<Vec*>(a));
        *reinterpret_cast<remove_cvref_t<FirstT>*>(a) = vec;
        __builtin_memcpy(idx, &i, sizeof(IdxT));
    }
};

// Generalized helper: runs the sparse transform kernel and verifies compressed output and index.
template <int NUM, int RATIO, typename Type>
void sparse_transform_verify(
    const std::vector<Type>& input,
    const std::vector<Type>& expected_output,
    const sparse::detail::SparseIdxPack<sparse::detail::idx_words_needed<NUM / RATIO>>&
        expected_idx)
{
    static_assert(RATIO == 2, "Extend functionality if other ratio is used.");
    ASSERT_EQ(static_cast<int>(input.size()), NUM);
    ASSERT_EQ(static_cast<int>(expected_output.size()), NUM / RATIO);

    constexpr int CompressedSize = NUM / RATIO;
    constexpr int IdxNumWords    = sparse::detail::idx_words_needed<CompressedSize>;
    using IdxType                = sparse::detail::SparseIdxPack<IdxNumWords>;

    int devCount;
    hipDevice_t dev;
    HIP_CHECK_ERROR(hipGetDevice(&dev));
    HIP_CHECK_ERROR(hipGetDeviceCount(&devCount));

    hipDeviceProp_t devProp;
    HIP_CHECK_ERROR(hipGetDeviceProperties(&devProp, dev));

    auto currentArchId = hip_device_prop_gcn_arch_name_to_amdgcn_target_id(devProp.gcnArchName);
    bool hasDevice     = static_cast<bool>(devCount > 0);

    // TODO: c++20 add check for arch id
    if(!hasDevice || (currentArchId == amdgcn_target_id::HOST))
    {
        GTEST_SKIP() << "No HIP device found. Skipping test.";
    }

    float* d_v;
    void* d_idx;

    static constexpr auto Size = sizeof(Type) * NUM;
    HIP_CHECK_ERROR(hipMalloc(&d_v, Size));
    HIP_CHECK_ERROR(hipMalloc(&d_idx, sizeof(IdxType)));

    // Copy inputs to device
    HIP_CHECK_ERROR(hipMemcpy(d_v, input.data(), Size, hipMemcpyHostToDevice));

    using Kernel = SparseTransformKernel<RATIO, ext_vector_t<Type, NUM>>;
    ck_tile::launch_kernel(ck_tile::stream_config{},
                           ck_tile::make_kernel(Kernel{}, dim3(1), dim3(32), 0, d_v, d_idx));
    HIP_CHECK_ERROR(hipDeviceSynchronize());

    std::vector<Type> h_out(NUM / RATIO, static_cast<Type>(0));
    HIP_CHECK_ERROR(hipMemcpy(h_out.data(), d_v, Size / RATIO, hipMemcpyDeviceToHost));
    IdxType h_idx{};
    HIP_CHECK_ERROR(hipMemcpy(&h_idx, d_idx, sizeof(IdxType), hipMemcpyDeviceToHost));

    EXPECT_EQ(h_idx.words[0], expected_idx.words[0]) << "Index mask mismatch (word 0)";
    for(int w = 1; w < IdxNumWords; ++w)
    {
        EXPECT_EQ(h_idx.words[w], expected_idx.words[w])
            << "Index mask mismatch (word " << w << ")";
    }
    for(int i = 0; i < NUM / RATIO; ++i)
    {
        EXPECT_EQ(h_out[i], expected_output[i]) << "Output mismatch at position " << i;
    }

    // Semantic index validation: each 2-bit field in h_idx encodes the original
    // slot (0-3) within the group of 4 that the corresponding compressed element
    // came from. Verify that the index is consistent with input and output.
    //
    // Note: when a group has fewer than 2 non-zeros, unused output slots contain
    // initialization values (from nonzero_elems init) that don't correspond to the
    // default index (slot 2). We only validate entries where the index was explicitly
    // set, i.e. where input[slot] is non-zero.
    for(int i = 0; i < CompressedSize; ++i)
    {
        const int word     = (2 * i) / 32;
        const int shift    = (2 * i) % 32;
        int slot           = (h_idx.words[word] >> shift) & 0b11;
        int group          = i / 2;
        Type input_at_slot = input[group * 4 + slot];
        // Only check when input at the indexed slot is non-zero (explicitly assigned)
        // or when both are zero (consistent default for all-zero groups).
        if(static_cast<float>(input_at_slot) != 0.0f || static_cast<float>(h_out[i]) == 0.0f)
        {
            EXPECT_EQ(h_out[i], input_at_slot)
                << "Index field " << i << " points to slot " << slot << " in group " << group
                << " but output[" << i << "] != input[" << (group * 4 + slot) << "]";
        }
    }

    HIP_CHECK_ERROR(hipFree(d_v));
    HIP_CHECK_ERROR(hipFree(d_idx));
}

// Helper: build expected index from a per-group 4-bit pattern, repeated for all groups.
// Each group of 4 input elements contributes 2 compressed elements -> 2 x 2-bit index fields = 4
// bits.
template <int NumGroups>
static auto build_repeated_group_idx(int32_t group_bits_4)
{
    constexpr int CompressedSize = NumGroups * 2;
    constexpr int NumWords       = sparse::detail::idx_words_needed<CompressedSize>;
    sparse::detail::SparseIdxPack<NumWords> idx{};
    for(int g = 0; g < NumGroups; ++g)
    {
        const int bit_pos = g * 4;
        const int word    = bit_pos / 32;
        const int shift   = bit_pos % 32;
        idx.words[word] |= (group_bits_4 << shift);
    }
    return idx;
}

// Helper: build expected index from alternating even/odd 4-bit group patterns.
template <int NumGroups>
static auto build_alternating_group_idx(int32_t even_bits_4, int32_t odd_bits_4)
{
    constexpr int CompressedSize = NumGroups * 2;
    constexpr int NumWords       = sparse::detail::idx_words_needed<CompressedSize>;
    sparse::detail::SparseIdxPack<NumWords> idx{};
    for(int g = 0; g < NumGroups; ++g)
    {
        const int bit_pos = g * 4;
        const int word    = bit_pos / 32;
        const int shift   = bit_pos % 32;
        idx.words[word] |= ((g % 2 == 0 ? even_bits_4 : odd_bits_4) << shift);
    }
    return idx;
}

// 1. Basic correctness: valid divisible sizes
// Input pattern: {1, 0, 3, 0, 5, 0, 7, 0, ...} -> non-zeros at slots 0,2
// Group idx pattern: field0=0b00 (slot 0), field1=0b10 (slot 2) -> 0b1000
template <int NUM, int RATIO, typename Type>
void sparse_transform_test_case()
{
    std::vector<Type> v(NUM);
    for(int i = 0; i < NUM; ++i)
    {
        v[i] = i % 2 == 0 ? i + 1 : 0;
    }

    std::vector<Type> expected_out(NUM / RATIO);
    for(int i = 0; i < NUM / RATIO; ++i)
    {
        expected_out[i] = v[i * 2];
    }

    auto expected_idx = build_repeated_group_idx<NUM / 4>(0b1000);
    sparse_transform_verify<NUM, RATIO, Type>(v, expected_out, expected_idx);
}

TEST(SparseTransformsTest, ValidCompressionRatio)
{
    // TODO: extend those when new sparse builtins are
    // introduced and use different type combinations
    sparse_transform_test_case<8, 2, fp16_t>();
    sparse_transform_test_case<16, 2, fp16_t>();
    sparse_transform_test_case<32, 2, fp16_t>();
    sparse_transform_test_case<64, 2, fp16_t>(); // multi-word SparseIdxPack
}

// All-zero input: no non-zeros in any group of 4.
// Each output pair defaults to {a_vec[slot2], a_vec[slot3]} = {0, 0},
// and the index uses default slot-2 encoding (0b10) for every 2-bit field.
// Group idx pattern: 0b1010
template <int NUM>
void sparse_transform_all_zero()
{
    using T = fp16_t;
    std::vector<T> input(NUM, static_cast<T>(0));
    std::vector<T> expected_output(NUM / 2, static_cast<T>(0));
    auto expected_idx = build_repeated_group_idx<NUM / 4>(0b1010);
    sparse_transform_verify<NUM, 2, T>(input, expected_output, expected_idx);
}

TEST(SparseTransformsTest, AllZeroInput)
{
    sparse_transform_all_zero<8>();
    sparse_transform_all_zero<16>();
    sparse_transform_all_zero<32>();
    sparse_transform_all_zero<64>(); // multi-word SparseIdxPack
}

// Single non-zero per group of 4 (at slot 3).
// Only j=3 triggers: nonzero_elems[0]=V, field0=0b11, pos becomes 1.
// The unused second compressed slot is a TRUE ZERO with the default idx
// (slot 2), so the group contributes V exactly once to any dot product.
//
// HISTORY: this test originally expected {V, V} -- the second slot leaking
// a_vec[slot3]=V through the pre-fix {a[2], a[3]} initialization of
// nonzero_elems. That expectation recorded the compress_a_impl default bug
// (bug 1 of this PR) as correct behavior: the leaked survivor, paired with
// the default idx slot, double-counts whenever a group has fewer than two
// non-zeros. The expectation below is the corrected semantics.
// Group idx pattern: field0=0b11, field1=0b10 (default) -> 0b1011
template <int NUM>
void sparse_transform_single_nonzero()
{
    using T = fp16_t;
    std::vector<T> input(NUM, static_cast<T>(0));
    std::vector<T> expected_output(NUM / 2);

    for(int g = 0; g < NUM / 4; ++g)
    {
        T val                      = static_cast<T>(g + 5);
        input[g * 4 + 3]           = val;
        expected_output[g * 2]     = val;
        expected_output[g * 2 + 1] = static_cast<T>(0); // true zero, not a leaked survivor
    }

    auto expected_idx = build_repeated_group_idx<NUM / 4>(0b1011);
    sparse_transform_verify<NUM, 2, T>(input, expected_output, expected_idx);
}

TEST(SparseTransformsTest, SingleNonZeroPerGroup)
{
    sparse_transform_single_nonzero<8>();
    sparse_transform_single_nonzero<16>();
    sparse_transform_single_nonzero<32>();
    sparse_transform_single_nonzero<64>(); // multi-word SparseIdxPack
}

// Non-zeros at slots 1 and 3 in each group.
// Input: {0, a, 0, b, ...}. Output: {a, b, ...}.
// Group idx pattern: field0=0b01 (slot 1), field1=0b11 (slot 3) -> 0b1101
template <int NUM>
void sparse_transform_slots_1_and_3()
{
    using T = fp16_t;
    std::vector<T> input(NUM, static_cast<T>(0));
    std::vector<T> expected_output(NUM / 2);

    for(int g = 0; g < NUM / 4; ++g)
    {
        T a                        = static_cast<T>(g * 2 + 3);
        T b                        = static_cast<T>(g * 2 + 4);
        input[g * 4 + 1]           = a;
        input[g * 4 + 3]           = b;
        expected_output[g * 2]     = a;
        expected_output[g * 2 + 1] = b;
    }

    auto expected_idx = build_repeated_group_idx<NUM / 4>(0b1101);
    sparse_transform_verify<NUM, 2, T>(input, expected_output, expected_idx);
}

TEST(SparseTransformsTest, NonZerosAtSlots1And3)
{
    sparse_transform_slots_1_and_3<8>();
    sparse_transform_slots_1_and_3<16>();
    sparse_transform_slots_1_and_3<32>();
    sparse_transform_slots_1_and_3<64>(); // multi-word SparseIdxPack
}

// Non-zeros at slots 0 and 3 in each group (non-adjacent).
// Input: {a, 0, 0, b, ...}. Output: {a, b, ...}.
// Group idx pattern: field0=0b00 (slot 0), field1=0b11 (slot 3) -> 0b1100
template <int NUM>
void sparse_transform_slots_0_and_3()
{
    using T = fp16_t;
    std::vector<T> input(NUM, static_cast<T>(0));
    std::vector<T> expected_output(NUM / 2);

    for(int g = 0; g < NUM / 4; ++g)
    {
        T a                        = static_cast<T>(g * 2 + 2);
        T b                        = static_cast<T>(g * 2 + 3);
        input[g * 4]               = a;
        input[g * 4 + 3]           = b;
        expected_output[g * 2]     = a;
        expected_output[g * 2 + 1] = b;
    }

    auto expected_idx = build_repeated_group_idx<NUM / 4>(0b1100);
    sparse_transform_verify<NUM, 2, T>(input, expected_output, expected_idx);
}

TEST(SparseTransformsTest, NonZerosAtSlots0And3)
{
    sparse_transform_slots_0_and_3<8>();
    sparse_transform_slots_0_and_3<16>();
    sparse_transform_slots_0_and_3<32>();
    sparse_transform_slots_0_and_3<64>(); // multi-word SparseIdxPack
}

// Mixed sparsity pattern: even groups have non-zeros at slots 0,2; odd groups at slots 1,3.
// Even group idx: field0=0b00, field1=0b10 -> 0b1000
// Odd  group idx: field0=0b01, field1=0b11 -> 0b1101
template <int NUM>
void sparse_transform_mixed()
{
    using T = fp16_t;
    std::vector<T> input(NUM, static_cast<T>(0));
    std::vector<T> expected_output(NUM / 2);

    for(int g = 0; g < NUM / 4; ++g)
    {
        T a = static_cast<T>(g * 2 + 1);
        T b = static_cast<T>(g * 2 + 2);
        if(g % 2 == 0)
        {
            // Slots 0, 2
            input[g * 4]     = a;
            input[g * 4 + 2] = b;
        }
        else
        {
            // Slots 1, 3
            input[g * 4 + 1] = a;
            input[g * 4 + 3] = b;
        }
        expected_output[g * 2]     = a;
        expected_output[g * 2 + 1] = b;
    }

    auto expected_idx = build_alternating_group_idx<NUM / 4>(0b1000, 0b1101);
    sparse_transform_verify<NUM, 2, T>(input, expected_output, expected_idx);
}

TEST(SparseTransformsTest, MixedSparsityPattern)
{
    sparse_transform_mixed<8>();
    sparse_transform_mixed<16>();
    sparse_transform_mixed<32>();
    sparse_transform_mixed<64>(); // multi-word SparseIdxPack
}

template <typename AType,
          typename BType,
          typename CType,
          uint32_t WaveTileM,
          uint32_t WaveTileN,
          uint32_t WaveTileK,
          MmaAccumPolicy AccumPolicy>
struct SparsePipelineKernel
{
    static constexpr int kBlockSize = mma_pipeline_test::getCMakeWaveSize();

    __device__ void
    operator()(const void* a_per_lane, const void* b_per_lane, void* c_per_lane) const
    {
        using CompilerTarget = decltype(get_compiler_target());
        using Pipeline       = SparseMmaPipeline<AType,
                                                 BType,
                                                 CType,
                                                 WaveTileM,
                                                 WaveTileN,
                                                 WaveTileK,
                                                 AccumPolicy,
                                                 false, // CTranspose
                                                 1,     // SwizzleFactor
                                                 1,     // AttrNumAccessAV
                                                 1,     // AttrNumAccessBV
                                                 false, // UsePackedNumAccess
                                                 CompilerTarget>;

        using ATensor = typename Pipeline::AWarpTensor;
        using BTensor = typename Pipeline::BWarpTensor;
        using CTensor = typename Pipeline::CWarpTensor;

        const uint32_t lane = threadIdx.x;

        ATensor a;
        BTensor b;
        CTensor c;
        __builtin_memcpy(
            &a, static_cast<const uint8_t*>(a_per_lane) + lane * sizeof(ATensor), sizeof(ATensor));
        __builtin_memcpy(
            &b, static_cast<const uint8_t*>(b_per_lane) + lane * sizeof(BTensor), sizeof(BTensor));
        __builtin_memset(&c, 0, sizeof(CTensor));

        if constexpr(MmaOpTraits<typename Pipeline::MmaOp>::IsSupported)
        {
            Pipeline::exec(a, b, c);
            __builtin_memcpy(
                static_cast<uint8_t*>(c_per_lane) + lane * sizeof(CTensor), &c, sizeof(CTensor));
        }
    }
};

namespace {
const auto should_skip = [](amdgcn_target_id currentArchId) {
    bool isSupportedWmma = (currentArchId >= amdgcn_target_id::GFX1200) &&
                           (currentArchId <= amdgcn_target_id::GFX12_GENERIC);
    bool isSupportedMfma =
        (currentArchId >= amdgcn_target_id::GFX942) && (currentArchId <= amdgcn_target_id::GFX950);
    return ((currentArchId == amdgcn_target_id::HOST) || !(isSupportedWmma || isSupportedMfma));
};
} // namespace

template <typename AType,
          typename BType,
          typename CType,
          uint32_t WaveTileM,
          uint32_t WaveTileN,
          uint32_t WaveTileK,
          MmaAccumPolicy AccumPolicy>
struct SparsePipelineFactory
{
    template <typename Target>
    struct Create
    {
        using type = SparseMmaPipeline<AType,
                                       BType,
                                       CType,
                                       WaveTileM,
                                       WaveTileN,
                                       WaveTileK,
                                       AccumPolicy,
                                       false, // CTranspose
                                       1,     // SwizzleFactor
                                       1,     // AttrNumAccessAV
                                       1,     // AttrNumAccessBV
                                       false, // UsePackedNumAccess
                                       Target>;
    };
};

template <typename AType,
          typename BType,
          typename CType,
          uint32_t WaveTileM,
          uint32_t WaveTileN,
          uint32_t WaveTileK,
          MmaAccumPolicy AccumPolicy = MmaAccumPolicy::ROW_MAJOR>
void SparsePipeline_Real_impl()
{
    using Factory =
        SparsePipelineFactory<AType, BType, CType, WaveTileM, WaveTileN, WaveTileK, AccumPolicy>;
    using Kernel =
        SparsePipelineKernel<AType, BType, CType, WaveTileM, WaveTileN, WaveTileK, AccumPolicy>;

    mma_pipeline_test::
        run_pipeline_matrix_test<Factory::template Create, Kernel, AType, BType, CType>(
            WaveTileM, WaveTileN, WaveTileK, should_skip, Kernel{}, /*isSparse=*/true);
}

// Full matrix verification: 16x16x32 single-fragment sparse pipeline (ROW_MAJOR)
TEST(SparseMmaPipeline, FullMatrixVerify_16x16x32)
{
    SparsePipeline_Real_impl<fp16_t, fp16_t, fp32_t, 16u, 16u, 32u>();
}

// Multi-fragment K: 16x16x64 -> 2 K fragments, tests internal K iteration (ROW_MAJOR)
TEST(SparseMmaPipeline, FullMatrixVerify_16x16x64)
{
    SparsePipeline_Real_impl<fp16_t, fp16_t, fp32_t, 16u, 16u, 64u>();
}

// Full matrix verification: 16x16x32 single-fragment sparse pipeline (COL_MAJOR)
TEST(SparseMmaPipeline, FullMatrixVerify_16x16x32_ColMajor)
{
    SparsePipeline_Real_impl<fp16_t, fp16_t, fp32_t, 16u, 16u, 32u, MmaAccumPolicy::COL_MAJOR>();
}

// Multi-fragment K: 16x16x64 -> 2 K fragments, tests internal K iteration (COL_MAJOR)
TEST(SparseMmaPipeline, FullMatrixVerify_16x16x64_ColMajor)
{
    SparsePipeline_Real_impl<fp16_t, fp16_t, fp32_t, 16u, 16u, 64u, MmaAccumPolicy::COL_MAJOR>();
}

// Multi-fragment K: 16x16x128 -> 4 K fragments, exercises multi-word SparseIdxPack (ROW_MAJOR)
TEST(SparseMmaPipeline, FullMatrixVerify_16x16x128)
{
    SparsePipeline_Real_impl<fp16_t, fp16_t, fp32_t, 16u, 16u, 128u>();
}

// Multi-fragment K: 16x16x256 -> 8 K fragments, exercises larger multi-word SparseIdxPack
// (ROW_MAJOR)
TEST(SparseMmaPipeline, FullMatrixVerify_16x16x256)
{
    SparsePipeline_Real_impl<fp16_t, fp16_t, fp32_t, 16u, 16u, 256u>();
}

// Multi-fragment K: 16x16x128 -> 4 K fragments (COL_MAJOR)
TEST(SparseMmaPipeline, FullMatrixVerify_16x16x128_ColMajor)
{
    SparsePipeline_Real_impl<fp16_t, fp16_t, fp32_t, 16u, 16u, 128u, MmaAccumPolicy::COL_MAJOR>();
}

// Multi-fragment K: 16x16x256 -> 8 K fragments (COL_MAJOR)
TEST(SparseMmaPipeline, FullMatrixVerify_16x16x256_ColMajor)
{
    SparsePipeline_Real_impl<fp16_t, fp16_t, fp32_t, 16u, 16u, 256u, MmaAccumPolicy::COL_MAJOR>();
}

// ===========================================================================
// End-to-end SparseMmaPipeline correctness on gfx12 (registered form of the
// PR's standalone repro; see the HISTORY note on SingleNonZeroPerGroup).
// Runtime-skipped on non-gfx12 devices.
// ===========================================================================

static bool device_is_gfx12()
{
    hipDeviceProp_t prop{};
    if(hipGetDeviceProperties(&prop, 0) != hipSuccess)
        return false;
    return std::strstr(prop.gcnArchName, "gfx12") != nullptr;
}

namespace sparse_swmmac_e2e {

// Warp-tile correctness repro for ck_tile::SparseMmaPipeline
// (int8_t x int8_t -> int32_t, 16x16x{32,64,128} WMMA SWMMAC, gfx1201):
// checks that a full pipeline (on-the-fly SparseCompressTransform inside
// Pipeline::exec()) produces exact 2:4-sparse GEMM results against a CPU
// reference.
//
// This is a non-gtest standalone port of CK's OWN test methodology
// (test/ck_tile/core/arch/mma/pipeline/test_amdgcn_sparse_mma.cpp +
// pipeline_tests_helper.hpp), NOT independently re-derived -- reusing
// AMD's validated register-mapping logic (TileDistrEncRegMap) is much
// lower-risk than hand-deriving the per-lane sparse layout from scratch.
//
// Data flow tested: dense A (int8, with 2:4 structured zeros already
// applied per-4-group -- CK's own convention: A is provided UNCOMPRESSED,
// SparseCompressTransform runs ON THE FLY inside Pipeline::exec()) x dense
// B -> int32 C, ONE warp, ONE 16x16x{32,64,128} tile.
//
// INPUT PATTERNS. The DEFAULT build uses an adversarial generated tile:
// deterministic 2:4 input covering every survivor-position case per
// 4-group -- all six two-survivor pairs {0,1},{0,2},{0,3},{1,2},{1,3},
// {2,3}, all four single-survivor positions, and the all-zero group. This
// is the configuration that FAILS before the compress_a_impl default fix
// and PASSES after it. Survivors at positions 1 and 3 are essential: the
// legacy canonical pattern (zeros at slots 1,3, survivors only at 0,2)
// cannot detect the bug, because the old {a[2], a[3]} default always read
// a guaranteed zero under it. That canonical pattern is retained as an
// opt-in control via -DUSE_CANONICAL_PATTERN (expected to pass on both
// fixed and unfixed trees).




// CRITICAL usage requirement (found the hard way): SparseMmaPipeline's
// DEFAULT CompilerTarget template param (getCMakeCompilerTarget()) resolves
// via the __gfx1201__ preprocessor macro, which clang's HIP frontend only
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
// what CK's OWN test file does (CompilerTargetGfx950 = decltype(...)).
// Not a CK bug -- a usage requirement whenever a Pipeline type crosses the
// host/device boundary in one TU.
using Gfx1201Target = decltype(make_amdgcn_gfx12_target<amdgcn_target_id::GFX1201>());

// --- Reference (CPU, int64 accumulate -- exact for int8 x int8 up to K=~2^47, no overflow risk) ---
static void reference_matmul_i8(std::vector<int32_t> & C, const std::vector<int8_t> & A,
                                 const std::vector<int8_t> & B, uint32_t M, uint32_t N, uint32_t K) {
    for (uint32_t m = 0; m < M; ++m) {
        for (uint32_t n = 0; n < N; ++n) {
            int64_t acc = 0;
            for (uint32_t k = 0; k < K; ++k) {
                acc += static_cast<int64_t>(A[m * K + k]) * static_cast<int64_t>(B[k * N + n]);
            }
            C[m * N + n] = static_cast<int32_t>(acc);
        }
    }
}

// Control pattern: every group of 4 consecutive K elements keeps slots
// 0,2, zeros slots 1,3 (matches CK's own historical test convention).
// This pattern CANNOT detect the compress_a_impl default bug (survivors
// never sit at positions 1/3), so it is expected to pass on both fixed
// and unfixed trees -- kept only as a control.
static void apply_sparse_pattern(std::vector<int8_t> & A, uint32_t M, uint32_t K) {
    for (uint32_t m = 0; m < M; ++m) {
        for (uint32_t k = 0; k < K; k += 4) {
            if (k + 1 < K) A[m * K + k + 1] = 0;
            if (k + 3 < K) A[m * K + k + 3] = 0;
        }
    }
}
// Adversarial 2:4 tile, generated deterministically: cycles every group
// through 11 cases -- the six two-survivor position pairs, four
// single-survivor positions, and one all-zero group -- so survivors land
// at EVERY position (including 3, which the pre-fix default silently
// duplicated). Values are derived from (row, group, position) so a
// misplaced or duplicated survivor shows up numerically instead of
// cancelling. This subsumes the failure mode originally found with a real
// Quark-quantized int8 2:4 weight tile (arbitrary-position zeros);
// generating the input here keeps the test self-contained and buildable
// from the repo alone.
static void generate_adversarial_a(std::vector<int8_t> & A, uint32_t M, uint32_t K) {
    static const int pair_cases[6][2] = {{0,1},{0,2},{0,3},{1,2},{1,3},{2,3}};
    std::fill(A.begin(), A.end(), static_cast<int8_t>(0));
    const uint32_t groups = K / 4;
    for (uint32_t m = 0; m < M; ++m) {
        for (uint32_t g = 0; g < groups; ++g) {
            const uint32_t c = (m * groups + g) % 11u;
            int positions[2] = {-1, -1};
            if (c < 6)       { positions[0] = pair_cases[c][0]; positions[1] = pair_cases[c][1]; }
            else if (c < 10) { positions[0] = static_cast<int>(c - 6); }
            // c == 10: all-zero group
            for (int s = 0; s < 2; ++s) {
                if (positions[s] < 0) continue;
                const uint32_t pos = static_cast<uint32_t>(positions[s]);
                int v = 1 + static_cast<int>((m * 13u + g * 5u + pos * 3u) % 14u); // 1..14, never 0
                if (((m + g + pos) & 1u) != 0) v = -v;
                A[m * K + g * 4 + pos] = static_cast<int8_t>(v);
            }
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
                printf("  [DEV lane0] raw uncompressed a_vec (VecN=%d):", static_cast<int>(VecN));
                for (index_t i = 0; i < VecN; ++i) printf(" %d", static_cast<int>(raw[i]));
                printf("\n");
                auto ab_pair = Pipeline::ATransform::execExtVec(raw);
                auto & compressed = std::get<0>(ab_pair);
                auto & idxpk = std::get<1>(ab_pair);
                using CompVecTraits = vector_traits<std::remove_reference_t<decltype(compressed)>>;
                printf("  [DEV lane0] compressed a_vec (size=%d):", static_cast<int>(CompVecTraits)::vector_size);
                for (index_t i = 0; i < CompVecTraits::vector_size; ++i) printf(" %d", static_cast<int>(compressed[i]));
                printf("\n  [DEV lane0] idx words:");
                for (auto w : idxpk.words) printf(" 0x%08x", static_cast<unsigned>(w));
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
// FIX: the old version rebuilt the register map from a *bare*
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
                printf("  [DBG m=0] lane=%2u v=%2d -> k=%2u val=%d\n", lane, static_cast<int>(v), k_global, static_cast<int>(lane_a[v]));
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
static bool run_test(const char * label, bool canonical) {
    using Pipeline = SparseMmaPipeline<int8_t, int8_t, int32_t, WaveTileM, WaveTileN, WaveTileK,
                                        MmaAccumPolicy::ROW_MAJOR, false, 1, 1, 1, Gfx1201Target>;
    using AWarpTensor = typename Pipeline::AWarpTensor;
    using BWarpTensor = typename Pipeline::BWarpTensor;
    using CWarpTensor = typename Pipeline::CWarpTensor;

    const uint32_t M = WaveTileM, N = WaveTileN, K = WaveTileK, waveSize = 32;

    std::mt19937 rng(42);
    std::uniform_int_distribution<int> dist(-8, 8); // small range, avoids int8 overflow noise
    std::vector<int8_t> A(M * K), B(K * N);
    if (canonical) {
        // Control: zeros in slots 1,3 (survivors only at 0,2). Cannot detect
        // the default bug -- expected to pass on both fixed and unfixed trees.
        for (auto & v : A) v = static_cast<int8_t>(dist(rng));
        for (auto & v : B) v = static_cast<int8_t>(dist(rng));
        apply_sparse_pattern(A, M, K);
    } else {
        // Default: adversarial generated tile, survivors at every position
        // (fails before the compress_a_impl default fix, passes after).
        generate_adversarial_a(A, M, K);
        for (auto & v : B) v = static_cast<int8_t>(dist(rng));
    }

    std::vector<int32_t> C_expected(M * N, 0), C_actual(M * N, 0);
    reference_matmul_i8(C_expected, A, B, M, N, K);
#ifdef DEBUG_DEVICE
    if (K == 32) {
        printf("  [HOST] B column 0, k=0..31:");
        for (uint32_t k = 0; k < K; ++k) printf(" %d", static_cast<int>(B[k * N + 0]));
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
        max_abs_err = std::max(max_abs_err, std::labs(static_cast<long>(C_actual[i]) - static_cast<long>(C_expected[i])));
    }
    const bool pass = (max_abs_err == 0);
    printf("[%s] M=%u N=%u K=%u -> max_abs_err=%ld %s\n", label, M, N, K, max_abs_err, pass ? "PASS" : "FAIL");
    if (!pass) {
        printf("  sample: C_expected[0]=%d C_actual[0]=%d\n", C_expected[0], C_actual[0]);
    }
    return pass;
}

} // namespace sparse_swmmac_e2e

TEST(SparseSwmmacE2E, AdversarialGeneratedTile)
{
    if(!device_is_gfx12())
        GTEST_SKIP() << "gfx12-only (SWMMAC end-to-end)";
    // Survivors at every position incl. 1 and 3 -- the configuration the
    // canonical pattern cannot detect (fails on the pre-fix tree).
    EXPECT_TRUE((sparse_swmmac_e2e::run_test<16, 16, 32>("K32_single_frag", false)));
    EXPECT_TRUE((sparse_swmmac_e2e::run_test<16, 16, 64>("K64_2frag", false)));
    EXPECT_TRUE((sparse_swmmac_e2e::run_test<16, 16, 128>("K128_4frag", false)));
}

TEST(SparseSwmmacE2E, CanonicalPatternControl)
{
    if(!device_is_gfx12())
        GTEST_SKIP() << "gfx12-only (SWMMAC end-to-end)";
    // Control: passes on both fixed and unfixed trees; preserved as evidence
    // that the legacy pattern cannot detect the compress default bug.
    EXPECT_TRUE((sparse_swmmac_e2e::run_test<16, 16, 32>("K32_control", true)));
    EXPECT_TRUE((sparse_swmmac_e2e::run_test<16, 16, 64>("K64_control", true)));
}

// ===========================================================================
// pk_int4_t (iu4) end-to-end -- the numerical coverage bugs 2 and 3 lacked.
// Ported, with permission implied by its MIT license and with attribution,
// from doplxyz's independent verification harness for this PR
// (github.com/doplxyz/ck3759-gfx1201-verification, pk4_e2e.cpp): dense
// int4 CPU oracle over LOGICAL values (never calls the transform under
// test), guard-banded host fill via CK's own register maps.
// K=128/256 additionally lock bug 2 and the checkATransformResult fix:
// idx word counts diverge only at FragsK > 1, and those shapes fail to
// COMPILE if either regresses.
// ===========================================================================

namespace sparse_pk4_e2e {

// End-to-end pk_int4_t (iu4) correctness test for ck_tile::SparseMmaPipeline on
// gfx1201 -- the numerical coverage that bugs 2 and 3 of PR #3759 currently lack.
//
// The PR's own test only ever instantiates <int8_t,int8_t,int32_t>, which takes
// the PackedSize==1 branch and emits v_swmmac_i32_16x16x32_iu8; no iu4
// instruction is generated, so neither the packed-nibble compression (bug 2) nor
// the idx mapping (bug 3) is exercised. The PR's ENABLE_PK4_CASE block is a
// printf and a TODO, not a test.
//
// ORACLE. A is supplied UNCOMPRESSED as logical 4-bit values; the CPU reference
// is a dense int4 x int4 -> int32 matmul over those same logical values. It
// never calls the compression transform under test, so it is not circular.
//
// HOST-SIDE FILL. The PR marks the packed register-map convention as unproven.
// It is resolved here by asking CK's own types rather than guessing (see
// pk4_layout_probe.cpp): for pk_int4_t the register map's vector index
// enumerates LOGICAL 4-bit elements, not physical bytes -- num_vector_items is
// 16 for a K=32 tile whose AWarpTensor is 8 bytes. Writing lane_a[v] as if it
// were a byte, the way the int8 fill does, would therefore overrun the tensor by
// 2x. Logical element v goes into byte v/2, high nibble for even v and low
// nibble for odd v, per CK_TILE_USE_PK4_LAYOUT_SHUFFLE.



// See the PR test's note: CompilerTarget must be explicit, because the defaulted
// one resolves differently in the host and device passes of the same TU.
using Gfx1201Target = decltype(make_amdgcn_gfx12_target<amdgcn_target_id::GFX1201>());

template <uint32_t K>
using Pk4Pipeline = SparseMmaPipeline<pk_int4_t, pk_int4_t, int32_t, 16, 16, K,
                                      MmaAccumPolicy::ROW_MAJOR, false, 1, 1, 1, Gfx1201Target>;

// --- CPU reference over LOGICAL int4 values, independent of the transform ----
static void reference_matmul_i4(std::vector<int32_t>& C, const std::vector<int8_t>& A,
                                const std::vector<int8_t>& B, uint32_t M, uint32_t N, uint32_t K)
{
    for(uint32_t m = 0; m < M; ++m)
        for(uint32_t n = 0; n < N; ++n)
        {
            int64_t acc = 0;
            for(uint32_t k = 0; k < K; ++k)
                acc += static_cast<int64_t>(A[m * K + k]) * static_cast<int64_t>(B[k * N + n]);
            C[m * N + n] = static_cast<int32_t>(acc);
        }
}

// Adversarial 2:4 tile in int4 range. Same construction as the PR's int8 tile --
// cycle every group through the six two-survivor position pairs, the four
// single-survivor positions and the all-zero group -- so survivors land at every
// position, including the ones the pre-fix default silently duplicated. Values
// depend on (row, group, position) so a misplaced survivor shows up numerically
// instead of cancelling.
static void generate_adversarial_a_i4(std::vector<int8_t>& A, uint32_t M, uint32_t K,
                                      int case_count[11], int value_seen[16])
{
    static const int pair_cases[6][2] = {{0, 1}, {0, 2}, {0, 3}, {1, 2}, {1, 3}, {2, 3}};
    std::fill(A.begin(), A.end(), static_cast<int8_t>(0));
    const uint32_t groups = K / 4;
    for(uint32_t m = 0; m < M; ++m)
        for(uint32_t g = 0; g < groups; ++g)
        {
            const uint32_t c   = (m * groups + g) % 11u;
            ++case_count[c];
            int positions[2]   = {-1, -1};
            if(c < 6)       { positions[0] = pair_cases[c][0]; positions[1] = pair_cases[c][1]; }
            else if(c < 10) { positions[0] = static_cast<int>(c - 6); }
            for(int s = 0; s < 2; ++s)
            {
                if(positions[s] < 0) continue;
                const uint32_t pos = static_cast<uint32_t>(positions[s]);
                // Full signed 4-bit magnitude range: +1..+7 and -1..-8, so the
                // end point -8 (the only value whose sign bit is set with a zero
                // magnitude field) is exercised too. Zero is never used for a
                // survivor, so pruning stays unambiguous.
                const uint32_t h = (m * 13u + g * 5u + pos * 3u);
                int v;
                if(((m + g + pos) & 1u) != 0) v = -static_cast<int>(1u + h % 8u);   // -1..-8
                else                          v =  static_cast<int>(1u + h % 7u);   // +1..+7
                A[m * K + g * 4 + pos] = static_cast<int8_t>(v);
                ++value_seen[v & 0xF];
            }
        }
}

static void generate_b_i4(std::vector<int8_t>& B, uint32_t K, uint32_t N, int value_seen[16])
{
    // B covers the whole signed 4-bit range including 0 and -8, and is
    // deliberately asymmetric in k and n so a transposed or swapped coordinate
    // interpretation cannot cancel out.
    for(uint32_t k = 0; k < K; ++k)
        for(uint32_t n = 0; n < N; ++n)
        {
            const int v = static_cast<int>((k * 5u + n * 3u) % 16u) - 8;            // -8..7, includes 0
            B[k * N + n] = static_cast<int8_t>(v);
            ++value_seen[v & 0xF];
        }
}

// Pack logical element index v into the lane's byte buffer, high nibble first.
static inline void put_logical_nibble(uint8_t* bytes, uint32_t v, int8_t val)
{
    const uint32_t byte = v / 2;
    const bool high     = (v % 2) == 0;   // CK_TILE_USE_PK4_LAYOUT_SHUFFLE
    const uint8_t nib   = static_cast<uint8_t>(val & 0x0F);
    if(high)
        bytes[byte] = static_cast<uint8_t>((bytes[byte] & 0x0F) | (nib << 4));
    else
        bytes[byte] = static_cast<uint8_t>((bytes[byte] & 0xF0) | nib);
}

template <typename Pipeline>
struct SparseGemmKernel
{
    static constexpr int kBlockSize = 32;
    __device__ void operator()(const void* a_per_lane, const void* b_per_lane,
                               void* c_per_lane) const
    {
        using ATensor = typename Pipeline::AWarpTensor;
        using BTensor = typename Pipeline::BWarpTensor;
        using CTensor = typename Pipeline::CWarpTensor;
        const uint32_t lane = threadIdx.x;

        ATensor a;
        BTensor b;
        CTensor c;
        __builtin_memcpy(&a, static_cast<const uint8_t*>(a_per_lane) + lane * sizeof(ATensor),
                         sizeof(ATensor));
        __builtin_memcpy(&b, static_cast<const uint8_t*>(b_per_lane) + lane * sizeof(BTensor),
                         sizeof(BTensor));
        __builtin_memset(&c, 0, sizeof(CTensor));

        if constexpr(MmaOpTraits<typename Pipeline::MmaOp>::IsSupported)
        {
            Pipeline::exec(a, b, c);
            __builtin_memcpy(static_cast<uint8_t*>(c_per_lane) + lane * sizeof(CTensor), &c,
                             sizeof(CTensor));
        }
    }
};

template <uint32_t M, uint32_t N, uint32_t K>
static bool run_pk4_test(const char* label)
{
    using Pipeline = Pk4Pipeline<K>;
    using ATensor  = typename Pipeline::AWarpTensor;
    using BTensor  = typename Pipeline::BWarpTensor;
    using CTensor  = typename Pipeline::CWarpTensor;
    using ARegMap  = TileDistrEncRegMap<typename Pipeline::AWarpDstrEncoding>;
    using BRegMap  = TileDistrEncRegMap<typename Pipeline::BWarpDstrEncoding>;
    constexpr index_t av = ARegMap::num_vector_items;
    constexpr index_t bv = BRegMap::num_vector_items;
    constexpr uint32_t waveSize = 32;

    // The fill below writes av logical nibbles into sizeof(ATensor) bytes; if
    // that accounting is ever wrong it is a buffer overrun, so assert it.
    static_assert(av == 2 * static_cast<index_t>(sizeof(ATensor)), "A: logical elements != 2 x bytes");
    static_assert(bv == 2 * static_cast<index_t>(sizeof(BTensor)), "B: logical elements != 2 x bytes");

    std::vector<int8_t> A(M * K), B(K * N);
    std::vector<int32_t> C_expected(M * N, 0), C_actual(M * N, 0);
    int a_case[11] = {0}, a_val[16] = {0}, b_val[16] = {0};
    generate_adversarial_a_i4(A, M, K, a_case, a_val);
    generate_b_i4(B, K, N, b_val);
    reference_matmul_i4(C_expected, A, B, M, N, K);

    // Coverage: assert the stimulus actually contains every 2:4 pattern case and
    // spans the signed 4-bit range, rather than assuming the cycle reached them.
    for(int i = 0; i < 11; ++i)
        if(a_case[i] == 0)
        {
            printf("[%s] STIMULUS GAP: 2:4 pattern case %d never generated\n", label, i);
            return false;
        }
    if(a_val[8] == 0 || b_val[8] == 0)   // 8 == -8, the signed end point
    {
        printf("[%s] STIMULUS GAP: -8 not present (A=%d B=%d)\n", label, a_val[8], b_val[8]);
        return false;
    }

    // Guard bands around every per-lane tensor. The host fill writes nibbles
    // through a byte pointer into an object representation, so an off-by-a-factor
    // in the logical/physical accounting would be a silent overrun; these make it
    // loud instead.
    constexpr size_t GUARD = 64;
    std::vector<uint8_t> abuf(waveSize * sizeof(ATensor) + 2 * GUARD, 0xCD);
    std::vector<uint8_t> bbuf(waveSize * sizeof(BTensor) + 2 * GUARD, 0xCD);
    auto* a_per_lane = reinterpret_cast<ATensor*>(abuf.data() + GUARD);
    auto* b_per_lane = reinterpret_cast<BTensor*>(bbuf.data() + GUARD);
    std::vector<CTensor> c_per_lane(waveSize);
    std::memset(a_per_lane, 0, waveSize * sizeof(ATensor));
    std::memset(b_per_lane, 0, waveSize * sizeof(BTensor));
    std::memset(c_per_lane.data(), 0, waveSize * sizeof(CTensor));

    for(uint32_t lane = 0; lane < waveSize; ++lane)
    {
        auto* ab = reinterpret_cast<uint8_t*>(&a_per_lane[lane]);
        for(index_t v = 0; v < av; ++v)
        {
            auto c = ARegMap::calc_matrix_indices_from_lane_vector(lane, v);
            put_logical_nibble(ab, static_cast<uint32_t>(v), A[c[0] * K + c[1]]);
        }
        auto* bb = reinterpret_cast<uint8_t*>(&b_per_lane[lane]);
        for(index_t v = 0; v < bv; ++v)
        {
            // B's register map returns (n, k), not (k, n) -- matching what the
            // PR's own fill_b_fragments does.
            auto c = BRegMap::calc_matrix_indices_from_lane_vector(lane, v);
            put_logical_nibble(bb, static_cast<uint32_t>(v), B[c[1] * N + c[0]]);
        }
    }

    auto guards_intact = [&](const std::vector<uint8_t>& buf, size_t payload) {
        for(size_t i = 0; i < GUARD; ++i)
            if(buf[i] != 0xCD || buf[GUARD + payload + i] != 0xCD) return false;
        return true;
    };
    if(!guards_intact(abuf, waveSize * sizeof(ATensor)) ||
       !guards_intact(bbuf, waveSize * sizeof(BTensor)))
    {
        printf("[%s] HOST FILL OVERRAN ITS BUFFER -- guard band corrupted\n", label);
        return false;
    }

    void *da, *db, *dc;
    HIP_CHECK_ERROR(hipMalloc(&da, waveSize * sizeof(ATensor)));
    HIP_CHECK_ERROR(hipMalloc(&db, waveSize * sizeof(BTensor)));
    HIP_CHECK_ERROR(hipMalloc(&dc, waveSize * sizeof(CTensor)));
    HIP_CHECK_ERROR(hipMemcpy(da, a_per_lane, waveSize * sizeof(ATensor),
                              hipMemcpyHostToDevice));
    HIP_CHECK_ERROR(hipMemcpy(db, b_per_lane, waveSize * sizeof(BTensor),
                              hipMemcpyHostToDevice));
    HIP_CHECK_ERROR(hipMemset(dc, 0, waveSize * sizeof(CTensor)));

    ck_tile::launch_kernel(
        ck_tile::stream_config{},
        ck_tile::make_kernel(SparseGemmKernel<Pipeline>{}, dim3(1), dim3(waveSize), 0, da, db, dc));
    HIP_CHECK_ERROR(hipDeviceSynchronize());
    HIP_CHECK_ERROR(hipMemcpy(c_per_lane.data(), dc, waveSize * sizeof(CTensor),
                              hipMemcpyDeviceToHost));
    HIP_CHECK_ERROR(hipFree(da));
    HIP_CHECK_ERROR(hipFree(db));
    HIP_CHECK_ERROR(hipFree(dc));

    using CRegMap = TileDistrEncRegMap<typename Pipeline::CWarpDstrEncoding>;
    for(uint32_t lane = 0; lane < waveSize; ++lane)
    {
        auto* lane_c = reinterpret_cast<int32_t*>(&c_per_lane[lane]);
        for(index_t v = 0; v < CRegMap::num_vector_items; ++v)
        {
            auto c = CRegMap::calc_matrix_indices_from_lane_vector(lane, v);
            C_actual[c[0] * N + c[1]] = lane_c[v];
        }
    }

    int64_t max_abs_err = 0;
    int first_bad = -1;
    for(uint32_t i = 0; i < M * N; ++i)
    {
        const int64_t e = std::llabs(static_cast<int64_t>(C_expected[i]) - static_cast<int64_t>(C_actual[i]));
        if(e > max_abs_err) max_abs_err = e;
        if(e != 0 && first_bad < 0) first_bad = static_cast<int>(i);
    }
    const bool pass = (max_abs_err == 0);
    printf("[%s] M=%u N=%u K=%u -> max_abs_err=%ld %s\n", label, M, N, K, static_cast<long>(max_abs_err),
           pass ? "PASS" : "FAIL");
    if(!pass)
        printf("  first mismatch at %d: expected=%d actual=%d\n", first_bad,
               C_expected[first_bad], C_actual[first_bad]);
    return pass;
}

} // namespace sparse_pk4_e2e

TEST(SparsePk4E2E, AdversarialInt4AllShapes)
{
    if(!device_is_gfx12())
        GTEST_SKIP() << "gfx12-only (iu4 SWMMAC end-to-end)";
    EXPECT_TRUE((sparse_pk4_e2e::run_pk4_test<16, 16, 32>("pk4_K32")));
    EXPECT_TRUE((sparse_pk4_e2e::run_pk4_test<16, 16, 64>("pk4_K64")));
    // Multi-fragment shapes: bug 2's idx-word accounting only diverges at
    // FragsK > 1; single-fragment shapes cannot distinguish it.
    EXPECT_TRUE((sparse_pk4_e2e::run_pk4_test<16, 16, 128>("pk4_K128")));
    EXPECT_TRUE((sparse_pk4_e2e::run_pk4_test<16, 16, 256>("pk4_K256")));
}
