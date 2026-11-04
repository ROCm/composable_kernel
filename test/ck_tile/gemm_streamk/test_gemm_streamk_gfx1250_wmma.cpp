// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "test_gemm_streamk_types.hpp"

template <typename Input, typename Pipeline, typename Reduction, typename Persistence>
using Gfx1250StreamKParams = std::tuple<Row,
                                        Col,
                                        Row,
                                        Input,
                                        Input,
                                        F32,
                                        std::conditional_t<std::is_same_v<Input, BF16>, BF16, F16>,
                                        I128,
                                        I128,
                                        ck_tile::number<64>,
                                        I16,
                                        I16,
                                        ck_tile::number<sizeof(Input) == 2 ? 32 : 64>,
                                        Persistence,
                                        Pipeline,
                                        Reduction>;

#if GFX1250_STREAMK_SMOKE
// The gfx1250 emulator runs one workgroup at a time, so a Linear or Tree tile owner would wait
// forever for later workgroups. Cover both pipelines, persistence modes and input widths with
// atomic reduction instead.
using Gfx1250StreamKTypes =
    ::testing::Types<Gfx1250StreamKParams<F16, CompV3, Atomic, NonPersistent>,
                     Gfx1250StreamKParams<F16, CompV4, Atomic, Persistent>,
                     Gfx1250StreamKParams<BF16, CompV4, Atomic, NonPersistent>,
                     Gfx1250StreamKParams<F8, CompV3, Atomic, NonPersistent>>;
#else
using Gfx1250StreamKTypes =
    ::testing::Types<Gfx1250StreamKParams<F16, CompV3, Atomic, NonPersistent>,
                     Gfx1250StreamKParams<F16, CompV3, Linear, NonPersistent>,
                     Gfx1250StreamKParams<F16, CompV3, Tree, NonPersistent>,
                     Gfx1250StreamKParams<F16, CompV4, Atomic, NonPersistent>,
                     Gfx1250StreamKParams<F16, CompV4, Linear, NonPersistent>,
                     Gfx1250StreamKParams<F16, CompV4, Tree, NonPersistent>,
                     Gfx1250StreamKParams<F16, CompV3, Atomic, Persistent>,
                     Gfx1250StreamKParams<F16, CompV3, Linear, Persistent>,
                     Gfx1250StreamKParams<F16, CompV3, Tree, Persistent>,
                     Gfx1250StreamKParams<F16, CompV4, Atomic, Persistent>,
                     Gfx1250StreamKParams<F16, CompV4, Linear, Persistent>,
                     Gfx1250StreamKParams<F16, CompV4, Tree, Persistent>,
                     Gfx1250StreamKParams<BF16, CompV3, Atomic, NonPersistent>,
                     Gfx1250StreamKParams<BF16, CompV3, Linear, NonPersistent>,
                     Gfx1250StreamKParams<BF16, CompV3, Tree, NonPersistent>,
                     Gfx1250StreamKParams<BF16, CompV4, Atomic, NonPersistent>,
                     Gfx1250StreamKParams<BF16, CompV4, Linear, NonPersistent>,
                     Gfx1250StreamKParams<BF16, CompV4, Tree, NonPersistent>,
                     Gfx1250StreamKParams<F8, CompV3, Atomic, NonPersistent>,
                     Gfx1250StreamKParams<F8, CompV3, Linear, NonPersistent>,
                     Gfx1250StreamKParams<F8, CompV3, Tree, NonPersistent>,
                     Gfx1250StreamKParams<F8, CompV4, Atomic, NonPersistent>,
                     Gfx1250StreamKParams<F8, CompV4, Linear, NonPersistent>,
                     Gfx1250StreamKParams<F8, CompV4, Tree, NonPersistent>,
                     Gfx1250StreamKParams<BF8, CompV3, Atomic, NonPersistent>,
                     Gfx1250StreamKParams<BF8, CompV3, Linear, NonPersistent>,
                     Gfx1250StreamKParams<BF8, CompV3, Tree, NonPersistent>,
                     Gfx1250StreamKParams<BF8, CompV4, Atomic, NonPersistent>,
                     Gfx1250StreamKParams<BF8, CompV4, Linear, NonPersistent>,
                     Gfx1250StreamKParams<BF8, CompV4, Tree, NonPersistent>>;
#endif

template <typename Params>
class TestGfx1250StreamK : public TestCkTileStreamK<Params>
{
};

TYPED_TEST_SUITE(TestGfx1250StreamK, Gfx1250StreamKTypes);

TYPED_TEST(TestGfx1250StreamK, SplitKTails)
{
    // Eight workgroups contribute to one tile. Exercise per-workgroup K-loop counts
    // 1, 2, 3, 4 and 5, including both tails of CompV4's double-buffered pipeline.
    for(int loops : {1, 2, 3, 4, 5})
        this->Run(128, 128, 8 * 64 * loops, 0, 0, 0, 8, true);
}

TYPED_TEST(TestGfx1250StreamK, UnevenSplitsAndPadding)
{
    this->Run(128, 384, 1088, 0, 0, 0, 8, true);
    this->Run(128, 384, 544, 0, 0, 0, 5, true);
    this->Run(136, 264, 1056, 0, 0, 0, 8, true);
}

TYPED_TEST(TestGfx1250StreamK, MixedDataParallelAndStreamK)
{
    // 33 output tiles with eight active workgroups forces DP and split-K sections;
    // persistent variants also reuse LDS across consecutive output tiles.
    this->Run(128, 33 * 128, 512, 0, 0, 0, 8, true);
}

template <typename T>
__global__ void AddAcrossWorkgroups(T* output)
{
    if(threadIdx.x == 0)
    {
        ck_tile::thread_buffer<T, 2> values;
        values[ck_tile::number<0>{}] = ck_tile::type_convert<T>(1);
        values[ck_tile::number<1>{}] = ck_tile::type_convert<T>(2);
        ck_tile::amd_buffer_atomic_add<T, 2>(values, output, 0, true, 2);
    }
}

template <typename T>
void CheckAtomicScope()
{
    constexpr int workgroups = 256;
    ck_tile::DeviceMem output(2 * sizeof(T));
    for(int repeat = 0; repeat < 8; ++repeat)
    {
        output.SetZero();
        hipLaunchKernelGGL(AddAcrossWorkgroups<T>,
                           dim3(workgroups),
                           dim3(32),
                           0,
                           nullptr,
                           static_cast<T*>(output.GetDeviceBuffer()));
        ASSERT_EQ(hipGetLastError(), hipSuccess);
        ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);
        T result[2];
        output.FromDevice(result);
        EXPECT_EQ(ck_tile::type_convert<float>(result[0]), workgroups);
        EXPECT_EQ(ck_tile::type_convert<float>(result[1]), 2 * workgroups);
    }
}

TEST(TestGfx1250BufferAtomic, DeviceScope)
{
    CheckAtomicScope<ck_tile::half_t>();
    CheckAtomicScope<float>();
    CheckAtomicScope<int32_t>();
}
