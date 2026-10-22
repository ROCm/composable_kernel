// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <cstdint>
#include <cstdlib>
#include <vector>

#include <gtest/gtest.h>

#include "ck_tile/core/arch/named_barrier.hpp"
#include "ck_tile/host/device_memory.hpp"
#include "ck_tile/host/device_prop.hpp"
#include "ck_tile/host/kernel_launch.hpp"

namespace {

// gfx1250 is RDNA, so a wave is 32 lanes. The kernel maps one slot element per lane.
constexpr int kWaveSize = 32;
constexpr int kNumSlots = 2;
constexpr int kNumSteps = 8;

using ring_pipe = ck_tile::named_barrier_pipeline<ck_tile::ring_spec<kNumSlots, 1, 1>>;

class Gfx125Device : public ::testing::Test
{
    protected:
    void SetUp() override
    {
        if(!ck_tile::is_gfx125_supported())
        {
            // CTest reports a skip as PASS, so on a runner that is meant to be gfx1250 this
            // would be indistinguishable from having run. Set CK_TILE_REQUIRE_GFX125=1 there
            // to make it a failure instead.
            const char* required = std::getenv("CK_TILE_REQUIRE_GFX125");
            if(required != nullptr && required[0] == '1')
            {
                FAIL() << "CK_TILE_REQUIRE_GFX125=1 but the device reports '"
                       << ck_tile::get_device_name() << "'";
            }
            GTEST_SKIP() << "hardware named barriers require gfx1250; device reports '"
                         << ck_tile::get_device_name() << "'";
        }
    }
};

// Suites run in registration order, so the assumption every other test rests on runs first.
using NamedBarrierAssumptionDevice = Gfx125Device;
using NamedBarrierSlotRingDevice   = Gfx125Device;
using NamedBarrierRingRunDevice    = Gfx125Device;

// Isolates the one semantic every handshake rests on and nothing has confirmed: a signal from a
// wave that never joined still counts toward the generation. Wave 0 only signals; wave 1 joins,
// signals and waits on a barrier whose member count needs both. If the assumption is false,
// wave 1 never wakes and the kernel hangs. The closing workgroup barrier keeps wave 0 from
// exiting straight after its signal, so that a hang here can only mean the assumption.
template <typename Pipe>
struct signal_without_join_kernel
{
    using barrier_pipeline = Pipe;
    using ring             = typename Pipe::template ring<0>;

    static constexpr ck_tile::index_t kBlockSize = 2 * kWaveSize;

    CK_TILE_DEVICE void operator()(int32_t* __restrict__ out) const
    {
        if constexpr(ring::kIsSupported)
        {
            const auto bar = Pipe::template init<signal_without_join_kernel>();
            if(ck_tile::get_warp_id() == 0)
            {
                ring::template producer<0>::template prime<0>(bar);
            }
            else
            {
                ring::consumer::template wait<0>(bar);
            }
            __syncthreads();
            out[threadIdx.x] = 1;
        }
        else
        {
            ck_tile::ignore = out;
        }
    }
};

// Never zero, and distinct per (step, producer, lane), so an unwritten output, a stale slot and
// the other producer's half are all told apart from a correctly delivered value.
constexpr int32_t fan_out_value(ck_tile::index_t step, int producer, int lane)
{
    return (step + 1) * 100000 + (producer + 1) * 1000 + lane + 1;
}

// Two producers feed two consumers through run(), the path the GEMM pipeline takes: a DATA
// generation needs both producers' signals, and every release fans out to each producer's own
// FREE barrier. Each producer owns one half of every slot, as the GEMM's A and B loaders do.
// Consumers come first so that, as in the GEMM, they occupy waves [0, kNumConsumers).
template <typename Pipe, ck_tile::index_t PublishLag>
struct fan_out_ring_kernel
{
    using barrier_pipeline = Pipe;
    using ring             = typename Pipe::template ring<0>;

    static constexpr int kSlots     = ring::kNumSlots;
    static constexpr int kProducers = ring::kNumProducerWaves;
    static constexpr int kConsumers = ring::kNumConsumerWaves;

    static constexpr ck_tile::index_t kBlockSize = (kProducers + kConsumers) * kWaveSize;

    CK_TILE_DEVICE void operator()(int32_t* __restrict__ out, ck_tile::index_t num_steps) const
    {
        if constexpr(ring::kIsSupported)
        {
            static_assert(!ring::kIsSupported || ck_tile::get_warp_size() == kWaveSize,
                          "one slot element per lane assumes wave32");

            __shared__ int32_t slots[kSlots][kProducers][kWaveSize];

            const ck_tile::index_t lane    = ck_tile::get_lane_id();
            const ck_tile::index_t wave_id = ck_tile::get_warp_id();
            const auto bar                 = Pipe::template init<fan_out_ring_kernel>();

            if(wave_id < kConsumers)
            {
                ring::consumer::run(bar, num_steps, [&](auto slot, ck_tile::index_t step) {
                    for(int p = 0; p < kProducers; ++p)
                    {
                        out[((wave_id * num_steps + step) * kProducers + p) * kWaveSize + lane] =
                            slots[slot][p][lane];
                    }
                    __threadfence_block();
                });
            }
            else
            {
                ck_tile::static_for<0, kProducers, 1>{}([&](auto p) {
                    constexpr int kProducer = decltype(p)::value;
                    if(wave_id == kConsumers + kProducer)
                    {
                        ring::template producer<kProducer>::template run<PublishLag>(
                            bar,
                            num_steps,
                            [&](auto slot, ck_tile::index_t step) {
                                slots[slot][kProducer][lane] = fan_out_value(step, kProducer, lane);
                            },
                            [](auto) { __threadfence_block(); });
                    }
                });
            }
        }
        else
        {
            ck_tile::ignore = out;
            ck_tile::ignore = num_steps;
        }
    }
};

// Launches one configuration and checks every consumer saw every step from both producers.
template <typename Pipe, ck_tile::index_t PublishLag>
void expect_fan_out_delivers(ck_tile::index_t num_steps)
{
    using kernel = fan_out_ring_kernel<Pipe, PublishLag>;
    const int out_elems =
        kernel::kConsumers * static_cast<int>(num_steps) * kernel::kProducers * kWaveSize;

    ck_tile::DeviceMem out_buf(out_elems * sizeof(int32_t));
    out_buf.SetBytePattern(0xFF);

    ck_tile::launch_and_check(ck_tile::stream_config{},
                              ck_tile::make_kernel(kernel{},
                                                   dim3(1),
                                                   dim3(kernel::kBlockSize),
                                                   0,
                                                   static_cast<int32_t*>(out_buf.GetDeviceBuffer()),
                                                   num_steps));
    ASSERT_EQ(hipDeviceSynchronize(), hipSuccess) << "kernel did not complete";

    std::vector<int32_t> out(out_elems);
    out_buf.FromDevice(out.data());

    for(int c = 0; c < kernel::kConsumers; ++c)
    {
        for(int step = 0; step < num_steps; ++step)
        {
            for(int p = 0; p < kernel::kProducers; ++p)
            {
                for(int lane = 0; lane < kWaveSize; ++lane)
                {
                    const int i =
                        ((c * num_steps + step) * kernel::kProducers + p) * kWaveSize + lane;
                    ASSERT_EQ(out[i], fan_out_value(step, p, lane))
                        << "slots " << kernel::kSlots << ", publish lag " << PublishLag
                        << ", steps " << num_steps << ": consumer " << c << " step " << step
                        << " producer " << p << " lane " << lane;
                }
            }
        }
    }
}

// Never zero, so an unwritten output element and an undrained slot both stay distinguishable
// from a legitimately delivered value.
constexpr int32_t slot_value(int step, int lane) { return (step + 1) * 1000 + lane + 1; }

// One producer wave feeds one consumer wave through the ring. A broken DATA handshake lets a
// consumer read a slot before it is filled; a broken FREE handshake lets the producer
// overwrite a slot the consumer has not drained. The consumer records what it saw per step
// rather than accumulating, because a sum cannot distinguish a reordered ring from an ordered
// one and duplicate/skip pairs cancel out of it exactly.
//
// Templated on the pipeline so `if constexpr` genuinely discards off gfx1250 -- in a
// non-template the discarded branch is still fully checked. This is how a real consumer
// (a pipeline) would be written.
template <typename Pipe>
struct slot_ring_kernel
{
    // Names this kernel's one barrier pipeline. init() rejects any other, which is what keeps a
    // kernel to a single barrier array.
    using barrier_pipeline = Pipe;

    using ring = typename Pipe::template ring<0>;

    static constexpr ck_tile::index_t kBlockSize = 2 * kWaveSize;

    // Used only by the supported branch below.
    [[maybe_unused]] static constexpr int kNumIters = kNumSteps / kNumSlots;
    static_assert(kNumSlots == 2 && kNumSteps % kNumSlots == 0,
                  "the loops below are unrolled for two slots and run whole trips round the ring, "
                  "dropping a ragged tail; run() handles any slot and step count");

    CK_TILE_DEVICE void operator()(int32_t* __restrict__ out) const
    {
        if constexpr(ring::kIsSupported)
        {
            // Dependent on Pipe, so it cannot fire from the discarded branch.
            static_assert(!ring::kIsSupported || ck_tile::get_warp_size() == kWaveSize,
                          "one wave per role and one slot element per lane assume wave32");

            // The ring's barriers are a separate hardware pool, so this is all the LDS needed.
            __shared__ int32_t p_slots[kNumSlots * kWaveSize];

            const ck_tile::index_t lane    = ck_tile::get_lane_id();
            const ck_tile::index_t wave_id = ck_tile::get_warp_id();

            // The token proves init() ran; every ring call requires one.
            const auto bar = Pipe::template init<slot_ring_kernel>();

            if(wave_id == 0)
            {
                using prod = typename ring::template producer<0>;

                // Every slot must be primed rather than waited on. On the first pass nothing
                // has been drained, so a wait would block on a FREE generation the consumer
                // cannot complete until this producer has filled the slot.
                p_slots[0 * kWaveSize + lane] = slot_value(0, lane);
                __threadfence_block();
                prod::template prime<0>(bar);

                p_slots[1 * kWaveSize + lane] = slot_value(1, lane);
                __threadfence_block();
                prod::template prime<1>(bar);

                for(int iter = 1; iter < kNumIters; ++iter)
                {
                    prod::template wait<0>(bar);
                    p_slots[0 * kWaveSize + lane] = slot_value(iter * kNumSlots + 0, lane);
                    __threadfence_block();
                    prod::template publish<0>(bar);

                    prod::template wait<1>(bar);
                    p_slots[1 * kWaveSize + lane] = slot_value(iter * kNumSlots + 1, lane);
                    __threadfence_block();
                    prod::template publish<1>(bar);
                }
            }
            else
            {
                using cons = typename ring::consumer;

                for(int iter = 0; iter < kNumIters; ++iter)
                {
                    cons::template wait<0>(bar);
                    const int32_t v0 = p_slots[0 * kWaveSize + lane];
                    __threadfence_block();
                    cons::template release<0>(bar);
                    out[(iter * kNumSlots + 0) * kWaveSize + lane] = v0;

                    cons::template wait<1>(bar);
                    const int32_t v1 = p_slots[1 * kWaveSize + lane];
                    __threadfence_block();
                    cons::template release<1>(bar);
                    out[(iter * kNumSlots + 1) * kWaveSize + lane] = v1;
                }
            }
        }
        else
        {
            ck_tile::ignore = out;
        }
    }
};

} // namespace

TEST_F(NamedBarrierAssumptionDevice, SignalWithoutJoinCountsTowardTheGeneration)
{
    using kernel = signal_without_join_kernel<ring_pipe>;

    ck_tile::DeviceMem out_buf(kernel::kBlockSize * sizeof(int32_t));
    out_buf.SetZero();

    ck_tile::launch_and_check(
        ck_tile::stream_config{},
        ck_tile::make_kernel(kernel{},
                             dim3(1),
                             dim3(kernel::kBlockSize),
                             0,
                             static_cast<int32_t*>(out_buf.GetDeviceBuffer())));
    ASSERT_EQ(hipDeviceSynchronize(), hipSuccess) << "kernel did not complete";

    std::vector<int32_t> out(kernel::kBlockSize);
    out_buf.FromDevice(out.data());
    for(int i = 0; i < kernel::kBlockSize; ++i)
    {
        ASSERT_EQ(out[i], 1) << "thread " << i << " never got past the barrier";
    }
}

TEST_F(NamedBarrierSlotRingDevice, ProducerConsumerHandshakeOrdersEveryStep)
{
    constexpr int kOutElems = kNumSteps * kWaveSize;

    ck_tile::DeviceMem out_buf(kOutElems * sizeof(int32_t));
    out_buf.SetBytePattern(0xFF);

    ck_tile::launch_and_check(
        ck_tile::stream_config{},
        ck_tile::make_kernel(slot_ring_kernel<ring_pipe>{},
                             dim3(1),
                             dim3(2 * kWaveSize),
                             0,
                             static_cast<int32_t*>(out_buf.GetDeviceBuffer())));
    ASSERT_EQ(hipDeviceSynchronize(), hipSuccess) << "kernel did not complete";

    std::vector<int32_t> out(kOutElems);
    out_buf.FromDevice(out.data());

    for(int step = 0; step < kNumSteps; ++step)
    {
        for(int lane = 0; lane < kWaveSize; ++lane)
        {
            ASSERT_EQ(out[step * kWaveSize + lane], slot_value(step, lane))
                << "step " << step << " lane " << lane;
        }
    }
}

TEST_F(NamedBarrierRingRunDevice, TwoProducersFeedEveryConsumerForAnyStepCount)
{
    using two_slots   = ck_tile::named_barrier_pipeline<ck_tile::ring_spec<2, 2, 2>>;
    using three_slots = ck_tile::named_barrier_pipeline<ck_tile::ring_spec<3, 2, 2>>;

    // Fewer steps than slots, exactly a slot's worth, and ragged tails past the first trip.
    for(ck_tile::index_t num_steps : {1, 2, 3, 7, 12})
    {
        expect_fan_out_delivers<two_slots, 0>(num_steps);
        expect_fan_out_delivers<two_slots, 1>(num_steps);
        expect_fan_out_delivers<three_slots, 0>(num_steps);
        expect_fan_out_delivers<three_slots, 2>(num_steps);
    }
}
