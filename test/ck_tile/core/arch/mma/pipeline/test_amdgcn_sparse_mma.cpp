// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <cstdint>
#include <gtest/gtest.h>
#include <iostream>
#include <numeric>

#include "ck_tile/core/arch/arch.hpp"
#include "ck_tile/core/arch/mma/amdgcn_mma.hpp"
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
    // Test fp16 → fp32 sparse MFMA for GFX950 (16x16x32)
    using TestSparseMfma16x16 = amdgcn_mma<fp16_t,
                                           fp16_t,
                                           fp32_t,
                                           16u,
                                           16u,
                                           32u,
                                           DefaultSparseMfmaCtrlFlags,
                                           CompilerTargetGfx950,
                                           MmaOpFamily::SPARSE>;

    static_assert(std::is_same_v<typename TestSparseMfma16x16::OpType, MfmaOp> &&
                      TestSparseMfma16x16::OpFamily == MmaOpFamily::SPARSE,
                  "GFX950 sparse 16x16x32 should have SparseMFMAOp type");

    static_assert(is_mma_op_of_family_v<MmaOpFamily::SPARSE, TestSparseMfma16x16>,
                  "GFX950 sparse 16x16x32 should be detected as Sparse");

    std::cout << "GFX950 sparse MFMA specialization is correct" << std::endl;
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
    static_assert(TestTraits::IsSparse, "Sparse MMA should be detected as sparse");
    static_assert(TestTraits::IsSupported, "Sparse MMA specialization should be supported");
    static_assert(TestTraits::IsMfma, "Sparse MFMA should be detected as MFMA");
    static_assert(!TestTraits::IsWmma, "Sparse MFMA should not be detected as WMMA");

    std::cout << "MmaOpTraits correctly integrates sparse operations" << std::endl;
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
    static_assert(MmaOpI<TestSparseMmma>);
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
    static_assert(std::is_same_v<typename DenseMfma::OpType, typename SparseMfma::OpType> &&
                      DenseMfma::OpFamily != SparseMfma::OpFamily,
                  "Dense and Sparse MFMA should have the same OpType tags and different OpFamily");

    // Verify traits correctly identify them
    static_assert(MmaOpTraits<DenseMfma>::IsMfma && MmaOpTraits<DenseMfma>::IsDense &&
                      !MmaOpTraits<DenseMfma>::IsSparse && !MmaOpTraits<DenseMfma>::IsScale &&
                      MmaOpTraits<DenseMfma>::IsSupported,
                  "Dense MFMA should be identified correctly");

    static_assert(MmaOpTraits<SparseMfma>::IsSparse && MmaOpTraits<SparseMfma>::IsMfma &&
                      !MmaOpTraits<SparseMfma>::IsDense && !MmaOpTraits<SparseMfma>::IsScale &&
                      MmaOpTraits<SparseMfma>::IsSupported,
                  "Sparse MFMA should be identified correctly");

    std::cout << "Dense and sparse MMA operations are correctly distinguished" << std::endl;
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
            static_assert(MmaOpTraits<Selected>::IsSparse);
            static_assert(MmaOpTraits<Selected>::IsMfma);
            static_assert(MmaOpTraits<Selected>::IsSupported);
            static_assert((std::is_same<typename Selected::OpType, MfmaOp>::value));
        }
        else
        {
            // Selector should pick the unsupported pass through
            static_assert(!MmaOpTraits<Selected>::IsSupported);
        }
    });
}

template <typename AType,
          typename BType,
          typename CType,
          uint32_t WaveTileM,
          uint32_t WaveTileN,
          uint32_t WaveTileK>
__global__ void test_sparse_accum_over_k(void* a, void* b, void* c, void* out)
{
    using Pipeline = SparseMmaPipeline<AType, BType, CType, WaveTileM, WaveTileN, WaveTileK>;

    using AVecType = typename Pipeline::AVecType;
    using BVecType = typename Pipeline::BVecType;
    using CVecType = typename Pipeline::CVecType;

    static constexpr uint32_t kIters = WaveTileK / Pipeline::MmaOp::kK;

    // Initialize the accumulator
    CVecType result = *reinterpret_cast<CVecType*>(c);

    // Accumulate input AxB over WaveTileK/FragK iterations
    for(uint32_t i = 0; i < kIters; ++i)
    {
        result = Pipeline::exec(
            *reinterpret_cast<AVecType*>(a), *reinterpret_cast<BVecType*>(b), result);
    }

    *reinterpret_cast<CVecType*>(out) = result;
}

// Live test on real hardware for sparse selection and execution.
TEST(SparseMMATrait, MmaSelector_Sparse_F16_F16_F32_16x16x32_Real)
{
    MmaPipelineTest<> test;
    const auto should_skip = [](amdgcn_target_id currentArchId) {
        bool isSupportedWmma = (currentArchId >= amdgcn_target_id::GFX1200) &&
                               (currentArchId <= amdgcn_target_id::GFX12_GENERIC);
        bool isSupportedMfma = (currentArchId >= amdgcn_target_id::GFX942) &&
                               (currentArchId <= amdgcn_target_id::GFX950);
        return ((currentArchId == amdgcn_target_id::HOST) || !(isSupportedWmma || isSupportedMfma));
    };
    const std::function<fp32_t(uint32_t)> validator = [](uint32_t waveTileK) {
        return static_cast<fp32_t>(waveTileK) / 2;
    };
    const auto kernel = [](uint32_t waveSize, void* a, void* b, void* c, void* out) {
        test_sparse_accum_over_k<MmaPipelineTest<>::AType,
                                 MmaPipelineTest<>::BType,
                                 MmaPipelineTest<>::CType,
                                 MmaPipelineTest<>::WaveTileM,
                                 MmaPipelineTest<>::WaveTileN,
                                 MmaPipelineTest<>::WaveTileK><<<1, waveSize>>>(a, b, c, out);
    };
    // Initialize A with 2:4 structured sparsity pattern: {1, 0, 1, 0, ...}
    // This ensures the sparse compression transform is actually exercised —
    // a no-op or broken compression would pass zeros through, causing incorrect results.
    const std::function<fp16_t(size_t)> sparseAInit = [](size_t i) -> fp16_t {
        return (i % 2 == 0) ? type_convert<fp16_t>(1) : type_convert<fp16_t>(0);
    };
    test.test_pipeline(should_skip, kernel, validator, sparseAInit);
}

template <uint32_t CompressionRatio, typename Vec>
__global__ void test_sparse_transform(void* a, void* idx)
{
    using ResultT =
        decltype(SparseCompressTransform<CompressionRatio>::exec(*static_cast<Vec*>(a)));
    using FirstT         = std::tuple_element_t<0, ResultT>;
    const auto& [vec, i] = SparseCompressTransform<CompressionRatio>::exec(*static_cast<Vec*>(a));
    *reinterpret_cast<remove_cvref_t<FirstT>*>(a) = vec;
    *reinterpret_cast<int32_t*>(idx)              = i;
}

// Generalized helper: runs the sparse transform kernel and verifies compressed output and index.
template <int NUM, int RATIO, typename Type>
void sparse_transform_verify(const std::vector<Type>& input,
                             const std::vector<Type>& expected_output,
                             int32_t expected_idx)
{
    static_assert(RATIO == 2, "Extend functionality if other ratio is used.");
    ASSERT_EQ(static_cast<int>(input.size()), NUM);
    ASSERT_EQ(static_cast<int>(expected_output.size()), NUM / RATIO);

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
    int32_t* d_idx;

    static constexpr auto Size = sizeof(Type) * NUM;
    HIP_CHECK_ERROR(hipMalloc(&d_v, Size));
    HIP_CHECK_ERROR(hipMalloc(&d_idx, sizeof(int32_t)));

    // Copy inputs to device
    HIP_CHECK_ERROR(hipMemcpy(d_v, input.data(), Size, hipMemcpyHostToDevice));

    test_sparse_transform<RATIO, ext_vector_t<Type, NUM>><<<1, 32>>>(d_v, d_idx);
    HIP_CHECK_ERROR(hipDeviceSynchronize());

    std::vector<Type> h_out(NUM / RATIO, static_cast<Type>(0));
    HIP_CHECK_ERROR(hipMemcpy(h_out.data(), d_v, Size / RATIO, hipMemcpyDeviceToHost));
    int32_t h_idx;
    HIP_CHECK_ERROR(hipMemcpy(&h_idx, d_idx, sizeof(int32_t), hipMemcpyDeviceToHost));

    EXPECT_EQ(h_idx, expected_idx) << "Index mask mismatch";
    for(int i = 0; i < NUM / RATIO; ++i)
    {
        EXPECT_EQ(h_out[i], expected_output[i]) << "Output mismatch at position " << i;
    }

    // Semantic index validation: each 2-bit field in h_idx encodes the original
    // slot (0–3) within the group of 4 that the corresponding compressed element
    // came from. Verify that the index is consistent with input and output.
    //
    // Note: when a group has fewer than 2 non-zeros, unused output slots contain
    // initialization values (from nonzero_elems init) that don't correspond to the
    // default index (slot 2). We only validate entries where the index was explicitly
    // set, i.e. where input[slot] is non-zero.
    constexpr int CompressedSize = NUM / RATIO;
    for(int i = 0; i < CompressedSize; ++i)
    {
        int slot           = (h_idx >> (2 * i)) & 0b11;
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
// Each group of 4 input elements contributes 2 compressed elements → 2 x 2-bit index fields = 4
// bits.
static int32_t build_repeated_group_idx(int num_groups, int32_t group_bits_4)
{
    int32_t idx = 0;
    for(int g = 0; g < num_groups; ++g)
        idx |= (group_bits_4 << (4 * g));
    return idx;
}

// Helper: build expected index from alternating even/odd 4-bit group patterns.
static int32_t build_alternating_group_idx(int num_groups, int32_t even_bits_4, int32_t odd_bits_4)
{
    int32_t idx = 0;
    for(int g = 0; g < num_groups; ++g)
        idx |= ((g % 2 == 0 ? even_bits_4 : odd_bits_4) << (4 * g));
    return idx;
}

// 1. Basic correctness: valid divisible sizes
// Input pattern: {1, 0, 3, 0, 5, 0, 7, 0, ...} → non-zeros at slots 0,2
// Group idx pattern: field0=0b00 (slot 0), field1=0b10 (slot 2) → 0b1000
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

    int32_t expected_idx = build_repeated_group_idx(NUM / 4, 0b1000);
    sparse_transform_verify<NUM, RATIO, Type>(v, expected_out, expected_idx);
}

TEST(SparseTransformsTest, ValidCompressionRatio)
{
    // TODO: extend those when new sparse builtins are
    // introduced and use different type combinations
    sparse_transform_test_case<8, 2, fp16_t>();
    sparse_transform_test_case<16, 2, fp16_t>();
    sparse_transform_test_case<32, 2, fp16_t>();
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
    int32_t expected_idx = build_repeated_group_idx(NUM / 4, 0b1010);
    sparse_transform_verify<NUM, 2, T>(input, expected_output, expected_idx);
}

TEST(SparseTransformsTest, AllZeroInput)
{
    sparse_transform_all_zero<8>();
    sparse_transform_all_zero<16>();
    sparse_transform_all_zero<32>();
}

// Single non-zero per group of 4 (at slot 3).
// nonzero_elems initializes to {a_vec[slot2]=0, a_vec[slot3]=V}.
// Only j=3 triggers: nonzero_elems[0]=V, field0=0b11, pos becomes 1.
// nonzero_elems[1] keeps its init V. Output: {V, V}.
// Group idx pattern: field0=0b11, field1=0b10 (default) → 0b1011
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

    int32_t expected_idx = build_repeated_group_idx(NUM / 4, 0b1011);
    sparse_transform_verify<NUM, 2, T>(input, expected_output, expected_idx);
}

TEST(SparseTransformsTest, SingleNonZeroPerGroup)
{
    sparse_transform_single_nonzero<8>();
    sparse_transform_single_nonzero<16>();
    sparse_transform_single_nonzero<32>();
}

// Non-zeros at slots 1 and 3 in each group.
// Input: {0, a, 0, b, ...}. Output: {a, b, ...}.
// Group idx pattern: field0=0b01 (slot 1), field1=0b11 (slot 3) → 0b1101
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

    int32_t expected_idx = build_repeated_group_idx(NUM / 4, 0b1101);
    sparse_transform_verify<NUM, 2, T>(input, expected_output, expected_idx);
}

TEST(SparseTransformsTest, NonZerosAtSlots1And3)
{
    sparse_transform_slots_1_and_3<8>();
    sparse_transform_slots_1_and_3<16>();
    sparse_transform_slots_1_and_3<32>();
}

// Non-zeros at slots 0 and 3 in each group (non-adjacent).
// Input: {a, 0, 0, b, ...}. Output: {a, b, ...}.
// Group idx pattern: field0=0b00 (slot 0), field1=0b11 (slot 3) → 0b1100
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

    int32_t expected_idx = build_repeated_group_idx(NUM / 4, 0b1100);
    sparse_transform_verify<NUM, 2, T>(input, expected_output, expected_idx);
}

TEST(SparseTransformsTest, NonZerosAtSlots0And3)
{
    sparse_transform_slots_0_and_3<8>();
    sparse_transform_slots_0_and_3<16>();
    sparse_transform_slots_0_and_3<32>();
}

// Mixed sparsity pattern: even groups have non-zeros at slots 0,2; odd groups at slots 1,3.
// Even group idx: field0=0b00, field1=0b10 → 0b1000
// Odd  group idx: field0=0b01, field1=0b11 → 0b1101
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

    int32_t expected_idx = build_alternating_group_idx(NUM / 4, 0b1000, 0b1101);
    sparse_transform_verify<NUM, 2, T>(input, expected_output, expected_idx);
}

TEST(SparseTransformsTest, MixedSparsityPattern)
{
    sparse_transform_mixed<8>();
    sparse_transform_mixed<16>();
    sparse_transform_mixed<32>();
}
