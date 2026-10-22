// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <algorithm>
#include <array>
#include <cstddef>
#include <deque>
#include <random>
#include <string>
#include <utility>
#include <vector>

#include <gtest/gtest.h>

#include "ck_tile/core/arch/named_barrier.hpp"

using ck_tile::index_t;
using ck_tile::kMaxNamedBarrierId;
using ck_tile::named_barrier_pipeline;
using ck_tile::ring_spec;

namespace {

// The device entry points cannot be instantiated off gfx1250, so what is testable here is
// the compile-time contract they rest on: the id allocation the backend must match.

// Flattens one ring's id space: DATA slots first, then FREE per producer.
template <typename Ring, index_t I>
constexpr index_t nth_id()
{
    constexpr index_t kSlots = Ring::kNumSlots;
    if constexpr(I < kSlots)
    {
        return Ring::template data_id<I>();
    }
    else
    {
        return Ring::template free_id<(I - kSlots) / kSlots, (I - kSlots) % kSlots>();
    }
}

template <typename Ring, std::size_t... Is>
constexpr auto collect_ids(std::index_sequence<Is...>)
{
    return std::array<index_t, sizeof...(Is)>{nth_id<Ring, static_cast<index_t>(Is)>()...};
}

template <typename Ring>
constexpr auto collect_ids()
{
    return collect_ids<Ring>(
        std::make_index_sequence<static_cast<std::size_t>(Ring::kNumBarriers)>{});
}

// A ring's ids must be a bijection onto the range its base claims, and every id must be one
// the hardware can address. An alias merges two handshakes into one, which surfaces as a
// hang rather than a wrong result.
template <typename Ring, index_t BaseId>
constexpr bool ids_are_bijective_and_addressable()
{
    const auto ids = collect_ids<Ring>();

    for(std::size_t i = 0; i < ids.size(); ++i)
    {
        // Addressable by S_BARRIER_SIGNAL/WAIT, and not the workgroup-wide id 0.
        if(ids[i] < 1 || ids[i] > kMaxNamedBarrierId)
        {
            return false;
        }
        // Inside this ring's own slice of the arena.
        if(ids[i] < BaseId || ids[i] >= BaseId + Ring::kNumBarriers)
        {
            return false;
        }
        for(std::size_t j = i + 1; j < ids.size(); ++j)
        {
            if(ids[i] == ids[j])
            {
                return false;
            }
        }
    }
    return true;
}

// Two rings of different depth sharing one arena: the case a single ring cannot express.
using asym_pipe = named_barrier_pipeline<ring_spec<3, 1, 2>, ring_spec<2, 1, 2>>;
using asym_a    = asym_pipe::ring<0>;
using asym_b    = asym_pipe::ring<1>;

// The largest pipeline the default pool holds.
using full_pipe = named_barrier_pipeline<ring_spec<3, 4, 1>>;
using full_ring = full_pipe::ring<0>;

// Two loaders, as in the gfx1250 GEMM pipeline, and two consumers.
using gemm_pipe = named_barrier_pipeline<ring_spec<2, 2, 2>>;
using gemm_ring = gemm_pipe::ring<0>;

} // namespace

TEST(NamedBarrierPipeline, RingsTileTheArenaWithoutOverlap)
{
    constexpr bool kAok    = ids_are_bijective_and_addressable<asym_a, 1>();
    constexpr bool kBok    = ids_are_bijective_and_addressable<asym_b, 7>();
    constexpr bool kFullOk = ids_are_bijective_and_addressable<full_ring, 1>();
    constexpr bool kGemmOk = ids_are_bijective_and_addressable<gemm_ring, 1>();

    EXPECT_TRUE(kAok) << "ring<0> ids alias or leave its slice";
    EXPECT_TRUE(kBok) << "ring<1> ids alias or leave its slice";
    EXPECT_TRUE(kFullOk) << "the saturating ring escapes [1, 15]";
    EXPECT_TRUE(kGemmOk) << "the 2P/2C ring aliases or escapes its slice";

    // The pipeline is exactly the sum of its rings, so the slices tile it with no gap.
    EXPECT_EQ(asym_a::kNumBarriers, 6);
    EXPECT_EQ(asym_b::kNumBarriers, 4);
    EXPECT_EQ(asym_pipe::kNumBarriers, 10);
    EXPECT_EQ(gemm_ring::kNumBarriers, 6);
    EXPECT_EQ(full_ring::kNumBarriers, 15);
    EXPECT_LE(full_pipe::kNumBarriers, kMaxNamedBarrierId);
}

TEST(NamedBarrierPipeline, BasesAreComputedNotWritten)
{
    // ring<1> starts where ring<0> ends. This is the invariant the backend must match, and
    // the extra parens keep the preprocessor from splitting on the template argument comma.
    EXPECT_EQ(asym_a::data_id<0>(), 1);
    EXPECT_EQ((asym_a::free_id<0, 2>()), 6);
    EXPECT_EQ(asym_b::data_id<0>(), 7);
    EXPECT_EQ((asym_b::free_id<0, 1>()), 10);
}

TEST(NamedBarrierSlotRing, MemberCountsIncludeTheWaiter)
{
    // Consumers wait on DATA, so every producer and every consumer must be a member. For the
    // 2P/2C pipeline that is what makes one wait mean "A and B are both ready".
    EXPECT_EQ(gemm_ring::kDataMemberCount, 2u + 2u);
    // One producer waits on its own FREE, so it and every consumer must be a member.
    EXPECT_EQ(gemm_ring::kFreeMemberCount, 1u + 2u);
}

TEST(NamedBarrierSlotRing, OccupiesNoLds)
{
    // Named barriers are a separate hardware pool; a pipeline aggregating LDS must see zero.
    EXPECT_EQ(asym_a::GetSmemSize(), 0);
    EXPECT_EQ(full_ring::GetSmemSize(), 0);
}

TEST(NamedBarrierRingSchedule, VisitsEveryStepOnceInOrderOnItsSlot)
{
    constexpr index_t kSlots = 3;
    for(index_t num_steps = 0; num_steps <= 4 * kSlots; ++num_steps)
    {
        std::vector<std::pair<index_t, index_t>> visits; // (slot, step)
        ck_tile::impl::ring_schedule<kSlots>::for_each_step(
            num_steps,
            [&](auto slot, index_t step) { visits.emplace_back(decltype(slot)::value, step); });

        ASSERT_EQ(static_cast<index_t>(visits.size()), num_steps) << "num_steps " << num_steps;
        for(index_t i = 0; i < num_steps; ++i)
        {
            EXPECT_EQ(visits[i].second, i) << "num_steps " << num_steps;
            EXPECT_EQ(visits[i].first, i % kSlots) << "num_steps " << num_steps;
        }
    }
}

namespace {

// Host model of the ring protocol.
//
// run() hands the protocol to drivers that are generic over their role operations, so the host
// can record exactly the handshakes a kernel issues and replay every wave's log against a model
// of the barrier semantics under many interleavings. The model encodes what named_barrier.hpp
// assumes, signal-without-join included: a generation completes on its member_count-th signal
// whether or not the signallers joined it. Passing here proves the protocol sound under those
// semantics, not that the hardware has them.

enum struct op_kind
{
    signal,
    arrive_and_wait,
    fill,
    retire,
    drain
};

struct ring_op
{
    op_kind kind;
    index_t a; // barrier id, slot, or the in-flight bound of a retire
    index_t b; // step, for fill and drain
};

using wave_log = std::vector<ring_op>;

// The drivers pass the token straight through to the role operations, so it carries the log.
struct recording_token
{
    wave_log* log;
};

template <typename Ring, index_t Producer>
struct recording_producer
{
    template <index_t Slot>
    static void wait(recording_token t)
    {
        t.log->push_back({op_kind::arrive_and_wait, Ring::template free_id<Producer, Slot>(), 0});
    }

    template <index_t Slot>
    static void publish(recording_token t)
    {
        t.log->push_back({op_kind::signal, Ring::template data_id<Slot>(), 0});
    }
};

template <typename Ring>
struct recording_consumer
{
    template <index_t Slot>
    static void wait(recording_token t)
    {
        t.log->push_back({op_kind::arrive_and_wait, Ring::template data_id<Slot>(), 0});
    }

    // The same fan-out as consumer::release: one signal on each producer's own FREE barrier.
    template <index_t Slot>
    static void release(recording_token t)
    {
        ck_tile::static_for<0, Ring::kNumProducerWaves, 1>{}([&](auto p) {
            t.log->push_back(
                {op_kind::signal, Ring::template free_id<decltype(p)::value, Slot>(), 0});
        });
    }
};

// One log per wave of a workgroup, producers first: what each issues for num_steps steps.
template <typename Ring, index_t PublishLag>
std::vector<wave_log> record_waves(index_t num_steps)
{
    std::vector<wave_log> waves(Ring::kNumProducerWaves + Ring::kNumConsumerWaves);

    ck_tile::static_for<0, Ring::kNumProducerWaves, 1>{}([&](auto p) {
        constexpr index_t kProducer = decltype(p)::value;
        wave_log& log               = waves[kProducer];
        using producer              = recording_producer<Ring, kProducer>;
        ck_tile::impl::drive_producer<producer, Ring::kNumSlots, PublishLag>(
            recording_token{&log},
            num_steps,
            [&](auto slot, index_t step) {
                log.push_back({op_kind::fill, decltype(slot)::value, step});
            },
            [&](auto in_flight) {
                log.push_back({op_kind::retire, decltype(in_flight)::value, 0});
            });
    });

    for(index_t c = 0; c < Ring::kNumConsumerWaves; ++c)
    {
        wave_log& log = waves[Ring::kNumProducerWaves + c];
        ck_tile::impl::drive_consumer<recording_consumer<Ring>, Ring::kNumSlots>(
            recording_token{&log}, num_steps, [&](auto slot, index_t step) {
                log.push_back({op_kind::drain, decltype(slot)::value, step});
            });
    }
    return waves;
}

// Replays one log per wave, producers first as record_waves() returns them, under the
// interleaving pick(runnable_waves) chooses. Returns the first promise the ring broke, or an
// empty string. Transfers land only when retired: the worst case for publishing too early.
template <typename Ring, typename Pick>
std::string replay(const std::vector<wave_log>& waves, index_t num_steps, Pick&& pick)
{
    constexpr index_t kSlots     = Ring::kNumSlots;
    constexpr index_t kProducers = Ring::kNumProducerWaves;
    static_assert(Ring::template data_id<0>() == 1, "the model indexes barriers from id 1");

    struct barrier_state
    {
        index_t member_count;
        index_t signals;
        index_t generation;
    };
    std::vector<barrier_state> barriers(Ring::kNumBarriers,
                                        barrier_state{Ring::kFreeMemberCount, 0, 0});
    ck_tile::static_for<0, kSlots, 1>{}([&](auto s) {
        barriers[Ring::template data_id<decltype(s)::value>() - 1].member_count =
            Ring::kDataMemberCount;
    });
    auto signal = [&](index_t id) {
        barrier_state& bar = barriers[id - 1];
        if(++bar.signals == bar.member_count)
        {
            bar.signals = 0;
            ++bar.generation;
        }
    };

    struct wave_state
    {
        std::size_t pc;
        index_t waiting_on; // barrier id, 0 when not waiting
        index_t joined_generation;
    };
    const index_t num_waves = static_cast<index_t>(waves.size());
    std::vector<wave_state> wave(num_waves, wave_state{0, 0, 0});
    auto blocked = [&](index_t w) {
        return wave[w].waiting_on != 0 &&
               barriers[wave[w].waiting_on - 1].generation <= wave[w].joined_generation;
    };

    // Each producer owns one part of every slot, as the GEMM's A and B loaders do.
    struct slot_part
    {
        index_t step;
        bool landed;
    };
    std::vector<std::vector<slot_part>> parts(kProducers,
                                              std::vector<slot_part>(kSlots, {-1, false}));
    std::vector<std::deque<std::pair<index_t, index_t>>> in_flight(kProducers); // (slot, step)
    std::vector<index_t> filled(kProducers, 0);
    std::vector<index_t> drained(num_waves - kProducers, 0);

    const auto step_msg = [](const char* what, index_t w, index_t step) {
        return std::string(what) + " (wave " + std::to_string(w) + ", step " +
               std::to_string(step) + ")";
    };

    for(;;)
    {
        std::vector<index_t> runnable;
        bool done = true;
        for(index_t w = 0; w < num_waves; ++w)
        {
            if(blocked(w))
            {
                done = false;
                continue;
            }
            wave[w].waiting_on = 0;
            if(wave[w].pc < waves[w].size())
            {
                done = false;
                runnable.push_back(w);
            }
        }
        if(done)
        {
            break;
        }
        if(runnable.empty())
        {
            std::string msg = "deadlock:";
            for(index_t w = 0; w < num_waves; ++w)
            {
                if(blocked(w))
                {
                    msg += " wave " + std::to_string(w) + " waits on barrier " +
                           std::to_string(wave[w].waiting_on) + ";";
                }
            }
            return msg;
        }

        const index_t w  = pick(runnable);
        const ring_op op = waves[w][wave[w].pc++];
        switch(op.kind)
        {
        case op_kind::signal: signal(op.a); break;
        case op_kind::arrive_and_wait:
            wave[w].waiting_on        = op.a;
            wave[w].joined_generation = barriers[op.a - 1].generation;
            signal(op.a);
            break;
        case op_kind::fill:
            if(op.a != op.b % kSlots || op.b != filled[w])
            {
                return step_msg("producer filled out of order or into the wrong slot", w, op.b);
            }
            for(index_t c = 0; c < num_waves - kProducers; ++c)
            {
                if(op.b >= kSlots && drained[c] <= op.b - kSlots)
                {
                    return step_msg(
                        "producer overwrote a slot a consumer had not drained", w, op.b);
                }
            }
            parts[w][op.a] = {op.b, false};
            in_flight[w].emplace_back(op.a, op.b);
            ++filled[w];
            break;
        case op_kind::retire:
            while(static_cast<index_t>(in_flight[w].size()) > op.a)
            {
                const auto oldest = in_flight[w].front();
                if(parts[w][oldest.first].step == oldest.second)
                {
                    parts[w][oldest.first].landed = true;
                }
                in_flight[w].pop_front();
            }
            break;
        case op_kind::drain:
            if(op.a != op.b % kSlots || op.b != drained[w - kProducers])
            {
                return step_msg("consumer drained out of order or the wrong slot", w, op.b);
            }
            for(index_t p = 0; p < kProducers; ++p)
            {
                if(parts[p][op.a].step != op.b || !parts[p][op.a].landed)
                {
                    return step_msg("consumer read a slot before every transfer landed", w, op.b);
                }
            }
            ++drained[w - kProducers];
            break;
        }
    }

    for(index_t p = 0; p < kProducers; ++p)
    {
        if(filled[p] != num_steps)
        {
            return step_msg("producer ran the wrong number of steps", p, filled[p]);
        }
    }
    for(index_t c = 0; c < num_waves - kProducers; ++c)
    {
        if(drained[c] != num_steps)
        {
            return step_msg("consumer ran the wrong number of steps", kProducers + c, drained[c]);
        }
    }
    for(index_t i = 0; i < Ring::kNumBarriers; ++i)
    {
        if(barriers[i].signals != 0)
        {
            return "barrier " + std::to_string(i + 1) + " ends holding " +
                   std::to_string(barriers[i].signals) + " signals: the ring is not at rest";
        }
    }
    return {};
}

index_t first_runnable(const std::vector<index_t>& runnable) { return runnable.front(); }
index_t last_runnable(const std::vector<index_t>& runnable) { return runnable.back(); }

struct round_robin
{
    index_t next = 0;
    index_t operator()(const std::vector<index_t>& runnable)
    {
        auto it =
            std::find_if(runnable.begin(), runnable.end(), [&](index_t w) { return w >= next; });
        const index_t w = (it == runnable.end()) ? runnable.front() : *it;
        next            = w + 1;
        return w;
    }
};

struct random_walk
{
    std::mt19937 rng;
    index_t operator()(const std::vector<index_t>& runnable)
    {
        return runnable[std::uniform_int_distribution<std::size_t>{0, runnable.size() - 1}(rng)];
    }
};

template <typename Spec, index_t PublishLag>
void expect_protocol_sound()
{
    using ring = typename named_barrier_pipeline<Spec>::template ring<0>;

    for(index_t num_steps = 0; num_steps <= 3 * ring::kNumSlots + 1; ++num_steps)
    {
        const auto waves   = record_waves<ring, PublishLag>(num_steps);
        const auto context = "slots " + std::to_string(ring::kNumSlots) + ", producers " +
                             std::to_string(ring::kNumProducerWaves) + ", consumers " +
                             std::to_string(ring::kNumConsumerWaves) + ", publish lag " +
                             std::to_string(PublishLag) + ", steps " + std::to_string(num_steps);

        EXPECT_EQ(replay<ring>(waves, num_steps, first_runnable), "") << context;
        EXPECT_EQ(replay<ring>(waves, num_steps, last_runnable), "") << context;
        EXPECT_EQ(replay<ring>(waves, num_steps, round_robin{}), "") << context;
        for(unsigned seed = 0; seed < 32; ++seed)
        {
            EXPECT_EQ(replay<ring>(waves, num_steps, random_walk{std::mt19937{seed}}), "")
                << context << ", seed " << seed;
        }
    }
}

} // namespace

TEST(NamedBarrierRingProtocol, SoundForEveryStepCountAndInterleaving)
{
    expect_protocol_sound<ring_spec<2, 1, 1>, 0>();
    expect_protocol_sound<ring_spec<2, 1, 1>, 1>();
    expect_protocol_sound<ring_spec<2, 2, 2>, 0>();
    expect_protocol_sound<ring_spec<2, 2, 2>, 1>();
    expect_protocol_sound<ring_spec<3, 2, 2>, 0>();
    expect_protocol_sound<ring_spec<3, 2, 2>, 1>();
    expect_protocol_sound<ring_spec<3, 2, 2>, 2>();
    // The producer/consumer GEMM test's rings: default policy <NumSlots=3, PublishLag=0>, two
    // A/B loader waves. [2, 2, 1] is what the fixture instantiates; [4, 4, 1] is forward
    // coverage for the 16-consumer-wave shape the tile engine can still generate. The consumer
    // count is the only input to the member counts the hardware barriers are armed with
    // (kDataMemberCount = producers + consumers, kFreeMemberCount = 1 + consumers), so both
    // must be replayed or one arming is unmodelled.
    expect_protocol_sound<ring_spec<3, 2, 4>, 0>(); // [2, 2, 1] warps
    expect_protocol_sound<ring_spec<3, 2, 4>, 1>();
    expect_protocol_sound<ring_spec<3, 2, 16>, 0>(); // [4, 4, 1] warps
    expect_protocol_sound<ring_spec<3, 2, 16>, 1>();
    expect_protocol_sound<ring_spec<4, 2, 1>, 3>();
    expect_protocol_sound<ring_spec<5, 2, 2>, 0>();
    expect_protocol_sound<ring_spec<5, 2, 2>, 4>();
}

TEST(NamedBarrierRingProtocol, ModelCatchesBrokenProtocols)
{
    // A model that passes everything proves nothing: break the protocol three ways.
    using ring               = named_barrier_pipeline<ring_spec<2, 1, 1>>::ring<0>;
    constexpr index_t kSteps = 5;
    const auto good          = record_waves<ring, 0>(kSteps);
    ASSERT_EQ(replay<ring>(good, kSteps, first_runnable), "");

    auto is_kind = [](op_kind kind) {
        return [kind](const ring_op& op) { return op.kind == kind; };
    };

    // Publishing without retiring the transfer: a consumer reads a slot still being filled.
    auto unretired = good;
    unretired[0].erase(
        std::remove_if(unretired[0].begin(), unretired[0].end(), is_kind(op_kind::retire)),
        unretired[0].end());
    EXPECT_NE(replay<ring>(unretired, kSteps, first_runnable), "");

    // Dropping the producer's last wait: the ring ends holding a release nobody answered.
    auto unanswered = good;
    ASSERT_EQ(unanswered[0].back().kind, op_kind::arrive_and_wait);
    unanswered[0].pop_back();
    EXPECT_NE(replay<ring>(unanswered, kSteps, first_runnable), "");

    // A consumer that never releases its first slot: the producer can never refill it.
    auto unreleased = good;
    auto& consumer  = unreleased[1];
    consumer.erase(std::find_if(consumer.begin(), consumer.end(), is_kind(op_kind::signal)));
    EXPECT_NE(replay<ring>(unreleased, kSteps, first_runnable).find("deadlock"), std::string::npos);
}
