#include "test_gemm_pipeline_kernel_types.hpp"
#include "test_gemm_pipeline_util.hpp"
#include "gtest/gtest.h"

template <typename T>
class TestCkTileGemmPipelineFlatmm : public TestCkTileGemmPipeline<T>
{
};

#define TEST_SUITE_NAME TestCkTileGemmPipelineFlatmm

TYPED_TEST_SUITE(TestCkTileGemmPipelineFlatmm, KernelTypesFlatmm);

#include "test_gemm_pipeline_ut_cases.inc"

#undef TEST_SUITE_NAME
