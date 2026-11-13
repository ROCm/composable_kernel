// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// Shared fixture for GemmPipelineAgBgCrCompTDMProducerConsumer inside StreamKKernel. Each
// scenario pins on the host the partition it exists to exercise, then runs with guard bands
// around C and the workspace that must come back untouched. The atomic and linear/tree suites
// live in separate translation units so their kernels compile in parallel.

#pragma once

#include <cstddef>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <set>
#include <stdexcept>
#include <tuple>
#include <type_traits>
#include <vector>

#include "ck_tile/ops/epilogue/tdm_epilogue.hpp"
#include "test_gemm_streamk_types.hpp"

using CShuffleEpilogue =
    ck_tile::integral_constant<StreamKEpilogueType, StreamKEpilogueType::CShuffle>;
using TdmEpilogue = ck_tile::integral_constant<StreamKEpilogueType, StreamKEpilogueType::Tdm>;

template <ck_tile::index_t NumSlots, ck_tile::index_t PublishLag>
using Ring = ck_tile::GemmPipelineAgBgCrCompTDMProducerConsumerPolicy<NumSlots, PublishLag>;

template <typename Input,
          typename Reduction,
          typename Epilogue,
          typename Persistence,
          typename Policy = Ring<4, 0>,
          typename NTile  = I128>
using PCStreamKParams = std::tuple<Row,
                                   Col,
                                   Row,
                                   Input,
                                   Input,
                                   F32,
                                   Input,
                                   I128,
                                   NTile,
                                   I64,
                                   I16,
                                   I16,
                                   I32,
                                   Persistence,
                                   CompTDMProducerConsumer,
                                   Reduction,
                                   Epilogue,
                                   Policy>;

// With automatic NUMA balancing on, a workgroup waiting on a named barrier can be context-saved
// and restored by the driver without its barrier state, stalling the producer/consumer ring for
// good. Launches are skipped, visibly, rather than risk hanging the device.
inline bool numa_balancing_enabled()
{
    std::ifstream f("/proc/sys/kernel/numa_balancing");
    int value = 0;
    return (f >> value) && value != 0;
}

inline constexpr std::size_t kGuardBytes = 4096;
// -0.0 in both fp16 and bf16. A clipped out-of-range row contributes +0.0, and an atomic add of
// +0.0 turns -0.0 into +0.0, so the guard sees zero-valued atomic writes that any other pattern
// would absorb unchanged.
inline constexpr std::uint16_t kGuardHalf = 0x8000;

// A device allocation with kGuardBytes of a known pattern on each side of a zeroed payload.
class GuardedDeviceBuffer
{
    public:
    explicit GuardedDeviceBuffer(std::size_t payload_bytes)
        : payload_bytes_(payload_bytes), mem_(payload_bytes + 2 * kGuardBytes)
    {
        if(payload_bytes_ % sizeof(std::uint16_t) != 0)
        {
            throw std::invalid_argument("payload must be a whole number of 16-bit words");
        }
        const std::vector<std::uint16_t> pattern(mem_.GetBufferSize() / sizeof(std::uint16_t),
                                                 kGuardHalf);
        mem_.ToDevice(pattern.data());
        ck_tile::hip_check_error(hipMemset(payload(), 0, payload_bytes_));
    }

    // DeviceMem frees its pointer in its destructor but is implicitly copyable; a copy or move
    // of this wrapper would free the same allocation twice.
    GuardedDeviceBuffer(const GuardedDeviceBuffer&)            = delete;
    GuardedDeviceBuffer& operator=(const GuardedDeviceBuffer&) = delete;
    GuardedDeviceBuffer(GuardedDeviceBuffer&&)                 = delete;
    GuardedDeviceBuffer& operator=(GuardedDeviceBuffer&&)      = delete;
    ~GuardedDeviceBuffer()                                     = default;

    void* payload() const
    {
        return static_cast<unsigned char*>(mem_.GetDeviceBuffer()) + kGuardBytes;
    }

    void PayloadToHost(void* dst) const
    {
        ck_tile::hip_check_error(hipMemcpy(dst, payload(), payload_bytes_, hipMemcpyDeviceToHost));
    }

    // Number of 16-bit guard words that no longer hold the pattern.
    std::size_t CorruptedGuardWords() const
    {
        std::vector<std::uint16_t> words(mem_.GetBufferSize() / sizeof(std::uint16_t));
        mem_.FromDevice(words.data());
        const std::size_t guard = kGuardBytes / sizeof(std::uint16_t);
        const std::size_t tail  = (kGuardBytes + payload_bytes_) / sizeof(std::uint16_t);
        std::size_t corrupted   = 0;
        for(std::size_t i = 0; i < guard; ++i)
        {
            corrupted += words[i] != kGuardHalf;
            corrupted += words[tail + i] != kGuardHalf;
        }
        return corrupted;
    }

    private:
    std::size_t payload_bytes_;
    ck_tile::DeviceMem mem_;
};

template <typename Params>
class TestGfx1250StreamKPC : public TestCkTileStreamK<Params>
{
    protected:
    using Base      = TestCkTileStreamK<Params>;
    using ADataType = typename Base::ADataType;
    using BDataType = typename Base::BDataType;
    using CDataType = typename Base::CDataType;
    using ALayout   = typename Base::ALayout;
    using BLayout   = typename Base::BLayout;
    using CLayout   = typename Base::CLayout;
    using Kernel    = typename Base::template KernelType<>;

    static constexpr ck_tile::index_t M_Tile = Base::M_Tile;
    static constexpr ck_tile::index_t N_Tile = Base::N_Tile;
    static constexpr ck_tile::index_t K_Tile = Base::K_Tile;

    static constexpr bool kAtomic =
        Base::ReductionStrategy == ck_tile::StreamKReductionStrategy::Atomic;
    static constexpr bool kTdmEpilogue = Base::EpilogueType == StreamKEpilogueType::Tdm;

    // Passed as num_cu with occupancy one: a fixed workgroup budget, not the device's CU count,
    // so the partition does not depend on the device.
    static constexpr int kNumWorkgroups = 8;
    // Passed instead of kNumWorkgroups to size the partition from the device itself.
    static constexpr int kDeviceWorkgroups = 0;

    enum struct Fill
    {
        Random,
        // Integers in {-1, 0, 1}: linear and tree reduction sum fp32 partials of integers, so the
        // result must equal the reference exactly while it stays small enough for CDataType.
        SmallInt,
        // A and B all ones: every element of C is exactly K, so a dropped or repeated K
        // iteration shows up as an exact mismatch rather than inside a tolerance.
        Ones
    };

    // What C holds before the launch. NaN (0xFFFF in fp16 and bf16) is only valid where every
    // element of C is written with `set`: any atomic add into it would leave NaN behind.
    enum struct CInit
    {
        Zero,
        NaN
    };

    // The result fill for scenarios that compare against the reference: exact where possible.
    static constexpr Fill kCheckedFill = kAtomic ? Fill::Random : Fill::SmallInt;

    static_assert(std::is_same_v<ALayout, Row> && std::is_same_v<BLayout, Col> &&
                      std::is_same_v<CLayout, Row>,
                  "the default strides below assume row-major A, column-major B, row-major C");

    static ck_tile::StreamKHostArgs MakeArgs(ck_tile::index_t M,
                                             ck_tile::index_t N,
                                             ck_tile::index_t K,
                                             const void* a = nullptr,
                                             const void* b = nullptr,
                                             void* c       = nullptr)
    {
        return ck_tile::StreamKHostArgs{a, b, c, M, N, K, K, K, N};
    }

    // num_workgroups is passed as num_cu: with occupancy > 1 the partition has
    // num_workgroups * occupancy workgroups on num_workgroups CUs.
    static auto
    MakeKargs(const ck_tile::StreamKHostArgs& args, int num_workgroups, int occupancy = 1)
    {
        return num_workgroups == kDeviceWorkgroups
                   ? Kernel::MakeKernelArgs(args)
                   : Kernel::MakeKernelArgs(args, num_workgroups, occupancy);
    }

    static auto Partition(ck_tile::index_t M,
                          ck_tile::index_t N,
                          ck_tile::index_t K,
                          int num_workgroups = kNumWorkgroups,
                          int occupancy      = 1)
    {
        return MakeKargs(MakeArgs(M, N, K), num_workgroups, occupancy).tile_partitioner;
    }

    // The device-sized budget MakeKernelArgs uses by default: NumCU() * Occupancy().
    static int DeviceBudget()
    {
        return Partition(M_Tile, N_Tile, K_Tile, kDeviceWorkgroups).get_max_active_wgs();
    }

    void Run(ck_tile::index_t M,
             ck_tile::index_t N,
             ck_tile::index_t K,
             Fill fill          = Fill::Random,
             int num_workgroups = kNumWorkgroups,
             int occupancy      = 1,
             CInit c_init       = CInit::Zero)
    {
        // Only launches can stall; the host-side tests and partition checks still run.
        if(numa_balancing_enabled())
        {
            GTEST_SKIP() << "/proc/sys/kernel/numa_balancing is enabled; named-barrier pipelines "
                            "can stall under it. Set it to 0 to run this test.";
        }
        std::cout << "M: " << M << ", N: " << N << ", K: " << K << std::endl;

        ck_tile::HostTensor<ADataType> a_m_k({M, K}, {K, 1});
        ck_tile::HostTensor<BDataType> b_k_n({K, N}, {1, K});
        ck_tile::HostTensor<CDataType> c_m_n({M, N}, {N, 1});

        if(fill == Fill::Ones)
        {
            ck_tile::FillConstant<ADataType>{ck_tile::type_convert<ADataType>(1.f)}(a_m_k);
            ck_tile::FillConstant<BDataType>{ck_tile::type_convert<BDataType>(1.f)}(b_k_n);
        }
        else
        {
            const float bound = fill == Fill::SmallInt ? 1.f : 5.f;
            ck_tile::FillUniformDistributionIntegerValue<ADataType>{-bound, bound, 11939}(a_m_k);
            ck_tile::FillUniformDistributionIntegerValue<BDataType>{-bound, bound, 11940}(b_k_n);
        }

        ck_tile::DeviceMem a_buf(a_m_k.get_element_space_size_in_bytes());
        ck_tile::DeviceMem b_buf(b_k_n.get_element_space_size_in_bytes());
        GuardedDeviceBuffer c_buf(c_m_n.get_element_space_size_in_bytes());
        if(c_init == CInit::NaN)
        {
            ck_tile::hip_check_error(
                hipMemset(c_buf.payload(), 0xFF, c_m_n.get_element_space_size_in_bytes()));
        }
        a_buf.ToDevice(a_m_k.data());
        b_buf.ToDevice(b_k_n.data());

        auto kargs = MakeKargs(
            MakeArgs(M, N, K, a_buf.GetDeviceBuffer(), b_buf.GetDeviceBuffer(), c_buf.payload()),
            num_workgroups,
            occupancy);
        ASSERT_TRUE(Kernel::IsSupportedArgument(kargs));

        // Atomic reduction adds partial tiles straight into C, and a partition without Stream-K
        // workgroups has no partial tiles at all; neither may need a workspace, and a null
        // pointer makes any use fault instead of passing silently. Otherwise linear and tree
        // keep their flags and partial tiles there, guarded like C.
        const bool needs_workspace = !kAtomic && kargs.tile_partitioner.get_sk_ctas() > 0;
        const auto workspace_bytes = Kernel::GetWorkSpaceSize(kargs);
        if(needs_workspace)
        {
            ASSERT_GT(workspace_bytes, 0u);
        }
        else
        {
            ASSERT_EQ(workspace_bytes, 0u);
        }
        GuardedDeviceBuffer workspace(workspace_bytes);
        if(needs_workspace)
        {
            // The flags must start at zero; the partial tiles after them are poisoned with NaN
            // (all-ones fp32), so a waiter that reads a slot before its producer's stores land,
            // or a slot no wave wrote, turns C into NaN rather than a plausible value.
            const std::size_t flags_bytes = kargs.tile_partitioner.get_flags_buffer_size();
            ASSERT_LT(flags_bytes, static_cast<std::size_t>(workspace_bytes));
            ck_tile::hip_check_error(
                hipMemset(static_cast<unsigned char*>(workspace.payload()) + flags_bytes,
                          0xFF,
                          workspace_bytes - flags_bytes));
        }
        Kernel::SetWorkSpacePointer(kargs, needs_workspace ? workspace.payload() : nullptr);

        ck_tile::ignore = ck_tile::launch_kernel(
            ck_tile::stream_config{nullptr, false, 0, 0, 1},
            ck_tile::make_kernel<1>(
                Kernel{}, Kernel::GridSize(kargs.tile_partitioner), Kernel::BlockSize(), 0, kargs));
        ck_tile::hip_check_error(hipDeviceSynchronize());

        EXPECT_EQ(c_buf.CorruptedGuardWords(), 0u) << "C written outside its extent";
        if constexpr(!kAtomic)
        {
            EXPECT_EQ(workspace.CorruptedGuardWords(), 0u)
                << "workspace written outside its extent";
        }

        c_buf.PayloadToHost(c_m_n.data());

        if(fill == Fill::Ones)
        {
            std::size_t mismatches = 0;
            const auto expected    = ck_tile::bit_cast<uint32_t>(static_cast<float>(K));
            for(const auto& c : c_m_n.mData)
            {
                // Bitwise equality: a NaN never matches.
                mismatches +=
                    ck_tile::bit_cast<uint32_t>(ck_tile::type_convert<float>(c)) != expected;
            }
            EXPECT_EQ(mismatches, 0u) << "expected every element of C to be exactly K";
            return;
        }

        // Atomic reduction rounds each partial to CDataType before adding it, so only linear and
        // tree reduction can be held to equality.
        Base::ExpectMatchesReference(c_m_n,
                                     a_buf.GetDeviceBuffer(),
                                     b_buf.GetDeviceBuffer(),
                                     M,
                                     N,
                                     K,
                                     K,
                                     K,
                                     N,
                                     kargs.tile_partitioner.estimate_num_wgs_per_tile(),
                                     fill == Fill::SmallInt && !kAtomic);
    }

    void LoopCountsPerWorkgroup()
    {
        // One output tile shared by all workgroups, each running `loops` K iterations through the
        // slot ring: below, at, just past and twice around it; {1,2,3,4,5,6,9} for four slots.
        constexpr int S = Kernel::GemmPipeline::NumSlots;
        for(int loops : std::set<int>{1, 2, S - 1, S, S + 1, S + 2, 2 * S + 1})
        {
            SCOPED_TRACE(::testing::Message() << "loops=" << loops << " NumSlots=" << S);
            const auto K = kNumWorkgroups * K_Tile * loops;
            const auto p = Partition(M_Tile, N_Tile, K);
            ASSERT_EQ(p.get_sk_ctas(), kNumWorkgroups);
            ASSERT_EQ(p.get_iters_per_sk_cta(), loops);
            ASSERT_EQ(p.get_extra_iters(), 0);
            // Run's GTEST_SKIP/ASSERTs return from Run only; stop the loop with it.
            ASSERT_NO_FATAL_FAILURE(Run(M_Tile, N_Tile, K, kCheckedFill));
            if(::testing::Test::IsSkipped())
            {
                return;
            }
        }
    }

    void SeveralSegmentsPerWorkgroup()
    {
        // Each workgroup covers more K iterations than a tile holds, so it runs the pipeline on
        // two or three tiles in turn and the ring must come to rest between them.
        const auto M = M_Tile;
        const auto N = 11 * N_Tile;
        const auto K = 3 * K_Tile;
        const auto p = Partition(M, N, K);
        ASSERT_EQ(p.get_dp_tiles(), 0);
        ASSERT_GT(p.get_iters_per_sk_cta(), p.get_iters_per_tile());
        Run(M, N, K, kCheckedFill);
    }

    void RaggedMAndK()
    {
        // A partial last M tile and a partial last K tile, clipped by the TDM loads; N stays a
        // multiple of the tile (see RaggedN).
        const auto M = M_Tile + 8;
        const auto N = 2 * N_Tile;
        const auto K = 16 * K_Tile + K_Tile / 2;
        ASSERT_GT(Partition(M, N, K).get_sk_tiles(), 0);
        Run(M, N, K, kCheckedFill);
    }

    void MixedDataParallelAndStreamK()
    {
        // Data-parallel tiles run first, then the Stream-K tiles reuse the same ring.
        const auto M = M_Tile;
        const auto N = 33 * N_Tile;
        const auto K = 8 * K_Tile;
        const auto p = Partition(M, N, K);
        // More DP tiles than workgroups: each persistent workgroup reuses the ring across DP
        // tiles.
        ASSERT_GT(p.get_dp_tiles(), kNumWorkgroups);
        ASSERT_GT(p.get_sk_tiles(), 0);
        Run(M, N, K, kCheckedFill);
    }

    void WholeTileUnitsOverwriteC()
    {
        // One K iteration per tile, so every Stream-K unit covers whole tiles and no tile is
        // split: atomic reduction stores them with `set` (linear and tree always do). C starts
        // as NaN, so a unit that adds instead of storing, or a tile no unit wrote, fails.
        const auto M = M_Tile;
        const auto N = 12 * N_Tile;
        const auto K = K_Tile;
        const auto p = Partition(M, N, K);
        ASSERT_EQ(p.get_dp_tiles(), 0);
        ASSERT_EQ(p.get_sk_ctas(), kNumWorkgroups);
        ASSERT_EQ(p.get_iters_per_tile(), 1);
        ASSERT_GT(p.get_extra_iters(), 0) << "some units must cover more than one tile";
        Run(M, N, K, kCheckedFill, kNumWorkgroups, 1, CInit::NaN);
    }

    void DataParallelWhenTilesMultipleOfNumCu()
    {
        // kNumWorkgroups workgroups on half as many CUs (occupancy 2) and 12 tiles: 12 % 8 != 0
        // but 12 % 4 == 0, so the partition is all data-parallel. The persistent grid stays at
        // kNumWorkgroups and four workgroups take a second tile (the DP stride loop with
        // extra_dp_tiles != 0). Every tile is stored with `set`, so a NaN-filled C must come back
        // fully overwritten.
        constexpr int kNumCu     = kNumWorkgroups / 2;
        constexpr int kOccupancy = 2;
        const auto M             = M_Tile;
        const auto N             = 12 * N_Tile;
        const auto K             = 8 * K_Tile;
        const auto p             = Partition(M, N, K, kNumCu, kOccupancy);
        ASSERT_EQ(p.get_max_active_wgs(), kNumWorkgroups);
        ASSERT_EQ(p.get_sk_ctas(), 0);
        ASSERT_EQ(p.get_dp_tiles(), 12);
        if constexpr(std::decay_t<decltype(p)>::PERSISTENT)
        {
            ASSERT_EQ(p.grid_size().x, static_cast<unsigned>(kNumWorkgroups));
        }
        Run(M, N, K, kCheckedFill, kNumCu, kOccupancy, CInit::NaN);
    }

    void DeepKExact()
    {
        // 128 K iterations per tile, split across workgroups. With all-ones inputs every partial
        // and every intermediate sum is a multiple of K_Tile no larger than K, exact in fp16 and
        // bf16 while K <= 2^14 (bf16 spacing is 64 in [2^13, 2^14)), whatever the split and the
        // order of the atomic adds.
        const auto M = M_Tile;
        const auto N = 2 * N_Tile;
        const auto K = 128 * K_Tile;
        ASSERT_LE(K, 16384);
        const auto p = Partition(M, N, K);
        ASSERT_EQ(p.get_dp_tiles(), 0);
        ASSERT_LT(p.get_iters_per_sk_cta(), p.get_iters_per_tile());
        Run(M, N, K, Fill::Ones);
    }
};
