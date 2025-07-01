#include "test_gemm_pipeline_kernel_types.hpp"
#include "test_gemm_pipeline_util.hpp"
#include "gtest/gtest.h"

template <typename T>
class TestCkTileGemmPipelineMemWmma : public TestCkTileGemmPipeline<T>
{
};

#define TEST_SUITE_NAME TestCkTileGemmPipelineMemWmma

TYPED_TEST_SUITE(TestCkTileGemmPipelineMemWmma, KernelTypesMemWmma);

#include "test_gemm_pipeline_ut_cases.inc"

#undef TEST_SUITE_NAME
