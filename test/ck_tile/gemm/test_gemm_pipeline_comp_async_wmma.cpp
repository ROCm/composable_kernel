#include "test_gemm_pipeline_kernel_types.hpp"
#include "test_gemm_pipeline_wmma_base.hpp"
#include "gtest/gtest.h"

template <typename T>
class TestCkTileGemmPipelineCompAsyncWmma
    : public TestCkTileGemmPipelineWmmaBase<T, class TestCkTileGemmPipelineCompAsyncWmma<T>>
{
};

#define TEST_SUITE_NAME TestCkTileGemmPipelineCompAsyncWmma

TYPED_TEST_SUITE(TestCkTileGemmPipelineCompAsyncWmma, KernelTypesCompAsyncWmma);

#include "test_gemm_pipeline_ut_cases.inc"

#undef TEST_SUITE_NAME
