// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "ck_tile/core/arch/arch.hpp"
#include "ck_tile/core/arch/mma/mma_op_family.hpp"
#include "ck_tile/core/arch/mma/mma_wavewise.hpp"

#include "pipeline_tests_helper.hpp"
#include <memory>

using namespace ck_tile;
using namespace ck_tile::core::arch;
using namespace ck_tile::core::arch::mma;

template <typename AType,
          typename BType,
          typename CType,
          uint32_t WaveTileM,
          uint32_t WaveTileN,
          uint32_t WaveTileK,
          bool CTranspose>
__global__ void test_wavewise_pipeline(void* a, void* b, void* c, void* out)
{
    using CompilerTarget = decltype(get_compiler_target());

    using Pipeline = WaveWiseMmaPipeline<AType,
                                         BType,
                                         CType,
                                         WaveTileM,
                                         WaveTileN,
                                         WaveTileK,
                                         MmaOpFamily::DENSE,
                                         MmaAccumPolicy::ROW_MAJOR,
                                         CTranspose,
                                         CompilerTarget>;

    using AVecType = typename Pipeline::AVecType;
    using BVecType = typename Pipeline::BVecType;
    using CVecType = typename Pipeline::CVecType;

    auto result = Pipeline::exec(*reinterpret_cast<AVecType*>(a),
                                 *reinterpret_cast<BVecType*>(b),
                                 *reinterpret_cast<CVecType*>(c));

    if constexpr(MmaOpTraits<typename Pipeline::MmaOp>::IsSupported)
    {
        // When the MmaOp is Unsupported (default) it returns the CVecType by value
        // so this cast is impossible...
        __builtin_memcpy(out, static_cast<const void*>(result), sizeof(CVecType));
    }
}

namespace {
const auto should_skip = [](amdgcn_target_id currentArchId) {
    bool isSupportedWmma = false;
    bool isSupportedMfma =
        (currentArchId >= amdgcn_target_id::GFX942) && (currentArchId <= amdgcn_target_id::GFX950);
    return ((currentArchId == amdgcn_target_id::HOST) || !(isSupportedWmma || isSupportedMfma));
};
const std::function<fp32_t(uint32_t)> validator = [](uint32_t waveTileK) {
    return static_cast<fp32_t>(waveTileK);
};
} // namespace

TEST(WaveWiseMmaPipeline, testKIter)
{
    MmaPipelineTest<> test;
    const auto kernel = [](uint32_t waveSize, void* a, void* b, void* c, void* out) {
        test_wavewise_pipeline<MmaPipelineTest<>::AType,
                               MmaPipelineTest<>::BType,
                               MmaPipelineTest<>::CType,
                               MmaPipelineTest<>::WaveTileM,
                               MmaPipelineTest<>::WaveTileN,
                               MmaPipelineTest<>::WaveTileK,
                               false><<<1, waveSize>>>(a, b, c, out);
    };
    test.test_pipeline(should_skip, kernel, validator);
}

TEST(WaveWiseMmaPipeline, testKIterSwapAB)
{
    MmaPipelineTest<> test;
    const auto kernel = [](uint32_t waveSize, void* a, void* b, void* c, void* out) {
        test_wavewise_pipeline<MmaPipelineTest<>::AType,
                               MmaPipelineTest<>::BType,
                               MmaPipelineTest<>::CType,
                               MmaPipelineTest<>::WaveTileM,
                               MmaPipelineTest<>::WaveTileN,
                               MmaPipelineTest<>::WaveTileK,
                               true><<<1, waveSize>>>(a, b, c, out);
    };
    test.test_pipeline(should_skip, kernel, validator);
}
