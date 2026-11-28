// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <gtest/gtest.h>
#include <random>
#include <vector>

#include "ck_tile/core.hpp"
#include "ck_tile/host.hpp"

namespace {

struct PhiloxInput
{
    unsigned long long seed;
    unsigned long long offset;
    unsigned long long subsequence;
};

__global__ void generate_philox(const PhiloxInput* inputs, uint4* outputs, size_t count)
{
    const size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i < count)
    {
        const auto input = inputs[i];
        outputs[i] = ck_tile::philox{input.seed, input.offset}.get_philox_4x32(input.subsequence);
    }
}

} // namespace

TEST(CkTilePhilox, DeviceMatchesHostBits)
{
    constexpr unsigned long long edges[] = {
        0, 1, 31, 32, 0xffffffffULL, 0x100000000ULL, 0x8000000000000000ULL, ~0ULL};
    std::vector<PhiloxInput> inputs;
    for(auto seed : edges)
        for(auto offset : edges)
            for(auto subsequence : edges)
                inputs.push_back({seed, offset, subsequence});

    std::mt19937_64 random(15342);
    for(int i = 0; i < 4096; ++i)
        inputs.push_back({random(), random(), random()});

    std::vector<uint4> outputs(inputs.size());
    ck_tile::DeviceMem device_inputs(inputs.size() * sizeof(PhiloxInput));
    ck_tile::DeviceMem device_outputs(outputs.size() * sizeof(uint4));
    device_inputs.ToDevice(inputs.data());
    generate_philox<<<(inputs.size() + 255) / 256, 256>>>(
        static_cast<const PhiloxInput*>(device_inputs.GetDeviceBuffer()),
        static_cast<uint4*>(device_outputs.GetDeviceBuffer()),
        inputs.size());
    ASSERT_EQ(hipGetLastError(), hipSuccess);
    ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);
    device_outputs.FromDevice(outputs.data());

    for(size_t i = 0; i < inputs.size(); ++i)
    {
        SCOPED_TRACE(i);
        const auto input = inputs[i];
        // Host Philox retains the widened multiply; compare all 128 output bits.
        const auto expected =
            ck_tile::philox{input.seed, input.offset}.get_philox_4x32(input.subsequence);
        EXPECT_EQ(outputs[i].x, expected.x);
        EXPECT_EQ(outputs[i].y, expected.y);
        EXPECT_EQ(outputs[i].z, expected.z);
        EXPECT_EQ(outputs[i].w, expected.w);
    }
}
