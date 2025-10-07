#include "test_gemm_pipeline_kernel_types.hpp"
#include "test_gemm_pipeline_util.hpp"
#include "gtest/gtest.h"

template <typename T>
class TestCkTileGemmPipelineCompAsync
    : public TestCkTileGemmPipeline<T, class TestCkTileGemmPipelineCompAsync<T>>
{
};

#define TEST_SUITE_NAME TestCkTileGemmPipelineCompAsync

TYPED_TEST_SUITE(TestCkTileGemmPipelineCompAsync, KernelTypesCompAsync);

#include "test_gemm_pipeline_ut_cases.inc"

#undef TEST_SUITE_NAME
