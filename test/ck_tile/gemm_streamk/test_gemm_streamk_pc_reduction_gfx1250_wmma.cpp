// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// GemmPipelineAgBgCrCompTDMProducerConsumer inside StreamKKernel with linear and tree reduction,
// over the CShuffle and TDM epilogues and several slot rings: only the math waves store and load
// partial tiles, and the results must equal the reference exactly.

#include <cstdint>
#include <string>

#include "test_gemm_streamk_pc_gfx1250_fixture.hpp"

using PCStreamKReductionTypes =
    ::testing::Types<PCStreamKParams<F16, Linear, CShuffleEpilogue, NonPersistent>,
                     PCStreamKParams<F16, Tree, CShuffleEpilogue, NonPersistent>,
                     PCStreamKParams<F16, Linear, TdmEpilogue, NonPersistent>,
                     PCStreamKParams<F16, Tree, TdmEpilogue, Persistent>,
                     PCStreamKParams<F16, Linear, CShuffleEpilogue, Persistent>,
                     PCStreamKParams<BF16, Tree, TdmEpilogue, NonPersistent>,
                     PCStreamKParams<BF16, Linear, CShuffleEpilogue, Persistent>,
                     PCStreamKParams<F16, Linear, CShuffleEpilogue, NonPersistent, Ring<2, 1>>,
                     PCStreamKParams<F16, Tree, TdmEpilogue, NonPersistent, Ring<3, 0>>,
                     PCStreamKParams<F16, Linear, TdmEpilogue, Persistent, Ring<5, 4>>,
                     // 128x64 tiles: AccumulatePartial adds the partial in two slices of two M
                     // iterations each; the 128x128 types above use four slices of one.
                     PCStreamKParams<F16, Linear, CShuffleEpilogue, NonPersistent, Ring<4, 0>, I64>,
                     PCStreamKParams<BF16, Tree, TdmEpilogue, Persistent, Ring<4, 0>, I64>>;

template <typename Params>
class TestGfx1250StreamKPCReduction : public TestGfx1250StreamKPC<Params>
{
};

TYPED_TEST_SUITE(TestGfx1250StreamKPCReduction, PCStreamKReductionTypes);

TYPED_TEST(TestGfx1250StreamKPCReduction, LoopCountsPerWorkgroup)
{
    this->LoopCountsPerWorkgroup();
}

TYPED_TEST(TestGfx1250StreamKPCReduction, SeveralSegmentsPerWorkgroup)
{
    this->SeveralSegmentsPerWorkgroup();
}

TYPED_TEST(TestGfx1250StreamKPCReduction, RaggedMAndK) { this->RaggedMAndK(); }

TYPED_TEST(TestGfx1250StreamKPCReduction, MixedDataParallelAndStreamK)
{
    this->MixedDataParallelAndStreamK();
}

TYPED_TEST(TestGfx1250StreamKPCReduction, DeepKExact) { this->DeepKExact(); }

TYPED_TEST(TestGfx1250StreamKPCReduction, WholeTileUnitsOverwriteC)
{
    this->WholeTileUnitsOverwriteC();
}

TYPED_TEST(TestGfx1250StreamKPCReduction, DataParallelWhenTilesMultipleOfNumCu)
{
    this->DataParallelWhenTilesMultipleOfNumCu();
}

TYPED_TEST(TestGfx1250StreamKPCReduction, RepeatRaceProneShapes)
{
    // A waiter that reads a partial before its producer's stores land fails only sometimes;
    // repeat the shapes with the most cross-workgroup waits.
    for(int repeat = 0; repeat < 5; ++repeat)
    {
        SCOPED_TRACE(::testing::Message() << "repeat=" << repeat);
        ASSERT_NO_FATAL_FAILURE(this->Run(TestFixture::M_Tile,
                                          TestFixture::N_Tile,
                                          TestFixture::kNumWorkgroups * TestFixture::K_Tile,
                                          TestFixture::kCheckedFill));
        if(this->IsSkipped())
        {
            return;
        }
        ASSERT_NO_FATAL_FAILURE(this->SeveralSegmentsPerWorkgroup());
    }
}

TYPED_TEST(TestGfx1250StreamKPCReduction, RaggedN)
{
    // The TDM epilogue clips a partial N tile itself; the CShuffle store would wrap it into the
    // next row of C, so that pairing rejects it.
    const auto M = TestFixture::M_Tile;
    const auto K = 8 * TestFixture::kNumWorkgroups * TestFixture::K_Tile;
    for(ck_tile::index_t N : {102, 121})
    {
        SCOPED_TRACE(::testing::Message() << "N=" << N);
        if constexpr(TestFixture::kTdmEpilogue)
        {
            ASSERT_GT(TestFixture::Partition(M, N, K).get_sk_tiles(), 0);
            ASSERT_NO_FATAL_FAILURE(this->Run(M, N, K, TestFixture::kCheckedFill));
            if(this->IsSkipped())
            {
                return;
            }
        }
        else
        {
            const auto* p = reinterpret_cast<void*>(std::uintptr_t{0x1000});
            auto* c       = reinterpret_cast<void*>(std::uintptr_t{0x1000});
            EXPECT_FALSE(TestFixture::IsArgSupported(TestFixture::MakeArgs(M, N, K, p, p, c),
                                                     TestFixture::kNumWorkgroups));
        }
    }
}

TYPED_TEST(TestGfx1250StreamKPCReduction, DeviceSizedPartition)
{
    // The default, device-sized budget (NumCU() * Occupancy()) splits one tile across every
    // workgroup: linear reduction chains wgs - 1 waits on workgroup 0, tree reduction log2(wgs)
    // levels. A flag that is never raised traps (CK_TILE_STREAMK_WAIT_SPIN_LIMIT) instead of
    // hanging. This does not prove co-residency: every wait targets a higher cta index, so with
    // in-order dispatch an occupancy overestimate only delays workgroups; the budget itself is
    // pinned by MoreWorkgroupsThanResidentRejected.
    const auto M   = TestFixture::M_Tile;
    const auto N   = TestFixture::N_Tile;
    const auto wgs = TestFixture::DeviceBudget();
    SCOPED_TRACE(::testing::Message() << "device budget=" << wgs);
    const auto K = 2 * wgs * TestFixture::K_Tile;
    const auto p = TestFixture::Partition(M, N, K, TestFixture::kDeviceWorkgroups);
    ASSERT_EQ(p.get_sk_ctas(), wgs);
    this->Run(M, N, K, TestFixture::Fill::Random, TestFixture::kDeviceWorkgroups);
}

TYPED_TEST(TestGfx1250StreamKPCReduction, MoreWorkgroupsThanResidentRejected)
{
    // The kernel's own device budget (NumCU() * Occupancy(), what MakeKernelArgs uses by default)
    // is the largest Stream-K workgroup count linear/tree reduction accepts; one more is rejected.
    const auto M  = TestFixture::M_Tile;
    const auto N  = TestFixture::N_Tile;
    const int cap = TestFixture::DeviceBudget();
    SCOPED_TRACE(::testing::Message() << "device budget=" << cap);
    const auto* p = reinterpret_cast<void*>(std::uintptr_t{0x1000});
    auto* c       = reinterpret_cast<void*>(std::uintptr_t{0x1000});
    for(const int wgs : {TestFixture::kNumWorkgroups, cap, cap + 1})
    {
        SCOPED_TRACE(::testing::Message() << "wgs=" << wgs);
        const auto K = 2 * wgs * TestFixture::K_Tile;
        ASSERT_EQ(TestFixture::Partition(M, N, K, wgs).get_sk_ctas(), wgs);
        EXPECT_EQ(TestFixture::IsArgSupported(TestFixture::MakeArgs(M, N, K, p, p, c), wgs),
                  wgs <= cap);
    }
}

TYPED_TEST(TestGfx1250StreamKPCReduction, HostMembersInstantiate)
{
    using Kernel           = typename TestFixture::Kernel;
    const std::string name = Kernel::GetName();
    EXPECT_NE(name.find(ck_tile::streamk_reduction_strategy_name(TestFixture::ReductionStrategy)),
              std::string::npos)
        << name;
    EXPECT_GT(Kernel::MaxOccupancyGridSize(ck_tile::stream_config{}).x, 0u);

    // The default budget is NumCU() * Occupancy(), and Occupancy() must respect the WGP's named
    // barrier pool, which hipOccupancy* does not model.
    int dev = 0;
    hipDeviceProp_t prop;
    ck_tile::hip_check_error(hipGetDevice(&dev));
    ck_tile::hip_check_error(hipGetDeviceProperties(&prop, dev));
    const auto budget = TestFixture::DeviceBudget();
    EXPECT_GE(budget, prop.multiProcessorCount);
    EXPECT_LE(budget, prop.multiProcessorCount * Kernel::kMaxResidentPerWgpByBarriers);
}
