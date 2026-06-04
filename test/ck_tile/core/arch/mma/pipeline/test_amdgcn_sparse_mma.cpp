// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <cstdint>
#include <gtest/gtest.h>
#include <iostream>
#include <numeric>

#include "ck_tile/core/arch/arch.hpp"
#include "ck_tile/core/arch/mma/amdgcn_mma.hpp"
#include "ck_tile/core/arch/mma/mma.hpp"
#include "ck_tile/core/arch/mma/mma_op_family.hpp"
#include "ck_tile/core/arch/mma/mma_selector.hpp"
#include "ck_tile/core/arch/mma/sparse/sparse_mma_pipeline.hpp"
#include <hip/hip_runtime.h>
#include "ck_tile/core/numeric/bfloat16.hpp"
#include "ck_tile/core/numeric/float8.hpp"
#include "ck_tile/core/numeric/half.hpp"
#include "ck_tile/core/numeric/integer.hpp"
#include "ck_tile/host/hip_check_error.hpp"
#include "ck_tile/core/arch/mma/mma_traits.hpp"
#include "ck_tile/core/utility/type_traits.hpp"

#include "pipeline_tests_helper.hpp"

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
                                           DefaultSparseMfmaCtrlFlags,
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
                                      DefaultSparseMfmaCtrlFlags,
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
                                      DefaultSparseMfmaCtrlFlags,
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
    using DenseMfma = amdgcn_mma<fp16_t,
                                 fp16_t,
                                 fp32_t,
                                 16u,
                                 16u,
                                 32u,
                                 DefaultMfmaCtrlFlags,
                                 CompilerTargetGfx950,
                                 MmaOpFamily::DENSE>;

    // Sparse MFMA on GFX950
    using SparseMfma = amdgcn_mma<fp16_t,
                                  fp16_t,
                                  fp32_t,
                                  16u,
                                  16u,
                                  32u,
                                  DefaultSparseMfmaCtrlFlags,
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

TEST(SparseMMATrait, SparseSelector)
{
    static_for<1, 33, 1>{}([](auto i) {
        using Selected = typename MmaDefaultSelector<fp16_t,
                                                     fp16_t,
                                                     fp32_t,
                                                     static_cast<uint32_t>(i),
                                                     static_cast<uint32_t>(i),
                                                     static_cast<uint32_t>(2 * i),
                                                     CompilerTargetGfx950,
                                                     MmaOpFamily::SPARSE>::SelectedOp;

        static constexpr bool isValid = (i == 16) || (i == 32);
        if constexpr(isValid)
        {
            // Selector should pick a sparse MFMA implementation
            EXPECT_TRUE(MmaOpTraits<Selected>::IsSparse);
            EXPECT_TRUE(MmaOpTraits<Selected>::IsMfma);
            EXPECT_TRUE(MmaOpTraits<Selected>::IsSupported);
            EXPECT_TRUE((std::is_same<typename Selected::OpType, MfmaOp>::value));
        }
        else
        {
            // Selector should pick the unsupported pass through
            EXPECT_FALSE(MmaOpTraits<Selected>::IsSupported);
        }
    });
}

template <uint32_t CompressionRatio, typename Vec>
struct SparseTransformKernel
{
    static constexpr int kBlockSize = mma_pipeline_test::getCMakeWaveSize();

    __device__ void operator()(void* a, void* idx) const
    {
        using ResultT =
            decltype(SparseCompressTransform<CompressionRatio>::exec(*static_cast<Vec*>(a)));
        using FirstT = std::tuple_element_t<0, ResultT>;
        using IdxT   = std::tuple_element_t<1, ResultT>;
        const auto& [vec, i] =
            SparseCompressTransform<CompressionRatio>::exec(*static_cast<Vec*>(a));
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
// nonzero_elems initializes to {a_vec[slot2]=0, a_vec[slot3]=V}.
// Only j=3 triggers: nonzero_elems[0]=V, field0=0b11, pos becomes 1.
// nonzero_elems[1] keeps its init V. Output: {V, V}.
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
        expected_output[g * 2 + 1] = val;
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
                                                 CompilerTarget>;

        using AVecType = typename Pipeline::AVecType;
        using BVecType = typename Pipeline::BVecType;
        using CVecType = typename Pipeline::CVecType;

        const uint32_t lane = threadIdx.x;

        AVecType a;
        BVecType b;
        CVecType c;
        __builtin_memcpy(&a,
                         static_cast<const uint8_t*>(a_per_lane) + lane * sizeof(AVecType),
                         sizeof(AVecType));
        __builtin_memcpy(&b,
                         static_cast<const uint8_t*>(b_per_lane) + lane * sizeof(BVecType),
                         sizeof(BVecType));
        __builtin_memset(&c, 0, sizeof(CVecType));

        if constexpr(MmaOpTraits<typename Pipeline::MmaOp>::IsSupported)
        {
            Pipeline::exec(a, b, c);
            __builtin_memcpy(
                static_cast<uint8_t*>(c_per_lane) + lane * sizeof(CVecType), &c, sizeof(CVecType));
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
