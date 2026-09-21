// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
#pragma once

// Large-tensor decomposition / INT_MAX element-count validation for fp4 (a4w4) grouped MX GEMM
// These cases exercise the all-GPU validation path (RunAllGpu): GPU data
// init, GPU reference, GPU compare. They apply only to the fp4 non-persistent CompAsync kernel
// row; for every other type in the suite they skip (RunAllGpu is fp4-only and the discarded
// if-constexpr branch is never instantiated for non-fp4 types).
//
// These tests are deliberately NOT registered with ctest (see the dedicated *_large_cases target
// in CMakeLists.txt) so they do not run in the default CI test pass; they are run explicitly
// (mirroring the CK *_large_cases / RUN_*_LARGE_CASES_TESTS convention).

// Stage A: cross-validate the new GPU reference against the trusted host reference on identical
// small shapes. Both Run() (host reference + check_err) and RunAllGpu() (GPU reference +
// gpu_verify) must pass before the GPU reference can be trusted at large scale.
TYPED_TEST(TEST_SUITE_NAME, GpuRefCrossValidate_Small)
{
    using ADataType = typename TestFixture::ADataType;
    if constexpr(std::is_same_v<ADataType, ck_tile::pk_fp4_t> &&
                 TestFixture::PipelineType == MxGemmPipelineType::CompAsync &&
                 !TestFixture::Persistent)
    {
        const int group_count = 2;
        const int kbatch      = 1;
        const std::vector<int> Ms{256, 512};
        const std::vector<int> Ns{512, 1024};
        const std::vector<int> Ks{512, 512};

        // Trusted host reference path first, then the new all-GPU path on the same shapes.
        this->Run(Ms, Ns, Ks, kbatch, group_count);
        this->RunAllGpu(Ms, Ns, Ks, kbatch, group_count);
    }
    else
    {
        GTEST_SKIP() << "GPU-reference cases apply only to fp4 non-persistent CompAsync.";
    }
}

// Stage B: the decomposition / int32-overflow case. Minimal shape that crosses the boundary:
// 2 groups, each per-group M*N = 100352*12288 = 1,233,125,376 < INT_MAX (and C byte span
// 2,466,250,752 > INT_MAX), aggregate total M*N = 2,466,250,752 > INT_MAX. This proves the
// host M-decomposition is what keeps every per-group buffer int32-safe (a single fused tensor
// would overflow the int32 element count), while the kernel still addresses C correctly even
// though the per-group C *byte* span exceeds INT_MAX. K is kept small (compute is irrelevant to
// the addressing property under test).
TYPED_TEST(TEST_SUITE_NAME, LargeM_decomposition_int32)
{
    using ADataType = typename TestFixture::ADataType;
    if constexpr(std::is_same_v<ADataType, ck_tile::pk_fp4_t> &&
                 TestFixture::PipelineType == MxGemmPipelineType::CompAsync &&
                 !TestFixture::Persistent)
    {
        const int group_count = 2;
        const int kbatch      = 1;
        const std::vector<int> Ms(group_count, 100352);
        const std::vector<int> Ns(group_count, 12288);
        const std::vector<int> Ks(group_count, 512);

        this->RunAllGpu(Ms, Ns, Ks, kbatch, group_count);
    }
    else
    {
        GTEST_SKIP() << "GPU-reference cases apply only to fp4 non-persistent CompAsync.";
    }
}
