#include "test_gemm_pipeline_kernel_types.hpp"
#include "test_gemm_pipeline_util.hpp"
#include "gtest/gtest.h"

template <typename T>
class TestCkTileGemmPipelinePersistent : public TestCkTileGemmPipeline<T>
{
};

#define TEST_SUITE_NAME TestCkTileGemmPipelinePersistent

TYPED_TEST_SUITE(TEST_SUITE_NAME, KernelTypesPersistent);

#include "test_gemm_pipeline_ut_cases.inc"

#undef TEST_SUITE_NAME
