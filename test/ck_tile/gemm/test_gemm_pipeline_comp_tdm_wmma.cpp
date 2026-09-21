#include "test_gemm_pipeline_kernel_types.hpp"
#include "test_gemm_pipeline_wmma_base.hpp"
#include "gtest/gtest.h"

template <typename T>
class TestCkTileGemmPipelineCompTDMWmma
    : public TestCkTileGemmPipelineWmmaBase<T, TestCkTileGemmPipelineCompTDMWmma<T>>
{
};

#define TEST_SUITE_NAME TestCkTileGemmPipelineCompTDMWmma

TYPED_TEST_SUITE(TestCkTileGemmPipelineCompTDMWmma, KernelTypesCompTDMWmma);

#include "test_gemm_pipeline_ut_cases.inc"

#undef TEST_SUITE_NAME
