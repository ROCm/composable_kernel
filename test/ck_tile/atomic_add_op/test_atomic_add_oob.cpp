// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <string>
#include <vector>

#include <gtest/gtest.h>
#include <hip/hip_runtime.h>

#include "ck_tile/core.hpp"

// On gfx942, bf16 atomic adds are issued as global atomics instead of buffer atomics.
// They must still drop accesses outside the buffer, as buffer atomics do in hardware,
// which check each packed pair on its own.

namespace {

using ck_tile::bf16_t;
using ck_tile::index_t;

constexpr index_t kGuardElements  = 16; // allocated before the buffer, to catch negative offsets
constexpr index_t kBufferElements = 32;
constexpr index_t kAllocElements  = kGuardElements + 64;
constexpr index_t kInRange        = 10;

template <bool Raw, index_t N>
__global__ void atomic_add_bf16([[maybe_unused]] bf16_t* p, [[maybe_unused]] index_t offset)
{
#if defined(__gfx942__)
    if(threadIdx.x != 0)
        return;
    ck_tile::thread_buffer<bf16_t, N> v;
    for(index_t i = 0; i < N; ++i)
        v[i] = ck_tile::type_convert<bf16_t>(1.0f);
    if constexpr(Raw)
        ck_tile::amd_buffer_atomic_add_raw<bf16_t, N>(v, p, offset, 0, true, kBufferElements);
    else
        ck_tile::amd_buffer_atomic_add<bf16_t, N>(v, p, offset, true, kBufferElements);
#endif
}

bool IsGfx942()
{
    hipDeviceProp_t props;
    if(hipGetDeviceProperties(&props, 0) != hipSuccess)
        return false;
    return std::string(props.gcnArchName).rfind("gfx942", 0) == 0;
}

// Returns the whole allocation after one N-wide atomic add at each offset into the buffer.
template <bool Raw, index_t N>
std::vector<float> RunAdds(const std::vector<index_t>& offsets)
{
    bf16_t* d = nullptr;
    EXPECT_EQ(hipMalloc(&d, kAllocElements * sizeof(bf16_t)), hipSuccess);
    EXPECT_EQ(hipMemset(d, 0, kAllocElements * sizeof(bf16_t)), hipSuccess);
    for(index_t offset : offsets)
        atomic_add_bf16<Raw, N><<<1, 64>>>(d + kGuardElements, offset);
    EXPECT_EQ(hipDeviceSynchronize(), hipSuccess);
    std::vector<bf16_t> h(kAllocElements);
    EXPECT_EQ(hipMemcpy(h.data(), d, kAllocElements * sizeof(bf16_t), hipMemcpyDeviceToHost),
              hipSuccess);
    EXPECT_EQ(hipFree(d), hipSuccess);
    std::vector<float> out(kAllocElements);
    for(index_t i = 0; i < kAllocElements; ++i)
        out[i] = ck_tile::type_convert<float>(h[i]);
    return out;
}

// Expects exactly the buffer elements in [first, last) to have been updated.
void ExpectUpdated(const std::vector<float>& out, index_t first, index_t last)
{
    for(index_t i = 0; i < kAllocElements; ++i)
    {
        const index_t offset = i - kGuardElements;
        const bool updated   = offset >= first && offset < last;
        EXPECT_EQ(out[i], updated ? 1.0f : 0.0f) << "buffer element " << offset;
    }
}

template <bool Raw>
void CheckPairAdds()
{
    if(!IsGfx942())
        GTEST_SKIP() << "bf16 global-atomic fallback is gfx942-only";
    // One pair inside the buffer, then pairs straddling its end, past its end, before its start.
    const auto out = RunAdds<Raw, 2>({kInRange, kBufferElements - 1, 40, -2});
    ExpectUpdated(out, kInRange, kInRange + 2);
}

// A vector straddling the end still updates its in-range pairs.
template <index_t N>
void CheckStraddlingVector()
{
    if(!IsGfx942())
        GTEST_SKIP() << "bf16 global-atomic fallback is gfx942-only";
    constexpr index_t offset = kBufferElements - N / 2;
    ExpectUpdated(RunAdds<false, N>({offset}), offset, kBufferElements);
}

} // namespace

TEST(AtomicAddOutOfBounds, Bf16BufferAtomicAdd) { CheckPairAdds<false>(); }

TEST(AtomicAddOutOfBounds, Bf16BufferAtomicAddRaw) { CheckPairAdds<true>(); }

TEST(AtomicAddOutOfBounds, Bf16x4StraddlingTheEnd) { CheckStraddlingVector<4>(); }

TEST(AtomicAddOutOfBounds, Bf16x8StraddlingTheEnd) { CheckStraddlingVector<8>(); }
