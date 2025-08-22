#include "test_gemm_pipeline_kernel_types.hpp"
#include "test_gemm_pipeline_wmma_base.hpp"
#include "gtest/gtest.h"

template <typename T>
class TestCkTileGemmPipelineCompV3Wmma
    : public TestCkTileGemmPipelineWmmaBase<T, TestCkTileGemmPipelineCompV3Wmma<T>>
{
};

#define TEST_SUITE_NAME TestCkTileGemmPipelineCompV3Wmma

TYPED_TEST_SUITE(TestCkTileGemmPipelineCompV3Wmma, KernelTypesCompV3Wmma);

#include "test_gemm_pipeline_ut_cases.inc"

#undef TEST_SUITE_NAME
