// SPDX-License-Identifier: MIT
// Copyright (c) Advanced Micro Devices, Inc. All rights reserved.

// Descriptor readback for the TDM dim-stride encoding.
//
// Packs strides through the real TDM_GROUP1 setters on device, copies the raw
// D# words back, and reconstructs each stride from the bitfield widths. A GEMM
// cannot stand in for this: a mis-encoded stride aims the DMA engine at an
// out-of-range address, so it surfaces as a runtime fault rather than a wrong
// result, and the TDM GEMM path has independent problems that would mask it.
//
// Layout under test (core/arch/amd_tdm_descriptor.hpp, TDM_GROUP1):
//
//   dim0_stride = sgpr5 (32 bits) ++ sgpr6[0:16]     -> splits at bit 32
//   dim1_stride = sgpr6[16:32]    ++ sgpr7 (32 bits) -> splits at bit 16
//
// Both fields are 48 bits wide. dim1 is the only field in the descriptor where
// `lo` is narrower than `hi`, which is what lets a symmetric `value >> 32` look
// correct: `lo` then survives by accidental truncation into its bitfield while
// bits [16:32] are dropped. That is invisible below 2^16 and wrong above it, so
// the cases below bracket that boundary.
//
// dim0 is exercised as a control. It splits at 32 genuinely and is encoded by
// separate code, so it must read back correctly regardless.

#include <cstdint>
#include <iterator>
#include <vector>

#include <gtest/gtest.h>

#include "ck_tile/core.hpp"
#include "ck_tile/host.hpp"
#include "ck_tile/host/kernel_launch.hpp"

namespace ck_tile {
namespace test {
namespace {

constexpr uint64_t k48Bit = (uint64_t{1} << 48) - 1;
constexpr int kNumWord    = 8; // D# group 1 is sgpr0..sgpr7

struct StrideCase
{
    uint64_t stride;
    const char* label;
};

const StrideCase kCases[] = {
    {1ull, "1"},
    {65535ull, "2^16 - 1"}, // last value a 16-bit lo half holds outright
    {65536ull, "2^16"},     // first value that needs the hi half
    {65537ull, "2^16 + 1"}, // both halves non-zero
    {131072ull, "2^17"},    //
    {1ull << 24, "2^24"},   //
    {1ull << 40, "2^40"},   // only reachable through a 32-bit hi half
    {k48Bit, "2^48 - 1"},   // every bit of both halves set
};

struct PackStridesKernel
{
    // One wave is ample -- there are only a handful of cases, and each lane
    // packs independently. Lanes past the case count exit immediately.
    static constexpr index_t kBlockSize = 64;

    // Only the device pass odr-uses this: kentry()'s body is `Kernel{}(args...)`
    // just under __HIP_DEVICE_COMPILE__, so the host pass sees a member that is
    // never called, and the enclosing anonymous namespace turns that into a
    // -Wunused-member-function error. Marked as DecodeLayoutProbe is in
    // test/ck_tile/fmha/test_fmha_bwd_tdm_layout.cpp.
    [[maybe_unused]] CK_TILE_DEVICE void
    operator()(const uint64_t* strides, int n, uint32_t* out) const
    {
        const int i = blockIdx.x * blockDim.x + threadIdx.x;
        if(i >= n)
            return;

        TDM_GROUP1 g{};
        g.tensorDimStride(0, strides[i]);
        g.tensorDimStride(1, strides[i]);

        uint32_t* w = out + i * kNumWord;
        w[0]        = g.sgpr0;
        w[1]        = g.sgpr1;
        w[2]        = g.sgpr2;
        w[3]        = g.sgpr3;
        w[4]        = g.sgpr4;
        w[5]        = g.sgpr5;
        w[6]        = g.sgpr6;
        w[7]        = g.sgpr7;
    }
};

// Reconstructed from the bitfield widths rather than by calling the setters'
// inverse -- a decoder derived from the encoder would cancel out the very bug
// this test exists to catch.
uint64_t read_dim0_stride(const uint32_t* w)
{
    return uint64_t{w[5]} | (uint64_t{w[6] & 0xFFFFu} << 32);
}

uint64_t read_dim1_stride(const uint32_t* w)
{
    return uint64_t{(w[6] >> 16) & 0xFFFFu} | (uint64_t{w[7]} << 16);
}

class TDMDescriptorStride : public ::testing::Test
{
    protected:
    std::vector<uint32_t> words;

    void SetUp() override
    {
        const int n = static_cast<int>(std::size(kCases));

        std::vector<uint64_t> strides(n);
        for(int i = 0; i < n; ++i)
            strides[i] = kCases[i].stride;

        DeviceMem stride_buf(sizeof(uint64_t) * n);
        DeviceMem word_buf(sizeof(uint32_t) * kNumWord * n);
        stride_buf.ToDevice(strides.data());

        PackStridesKernel pack_kernel;
        launch_kernel(stream_config{},
                      make_kernel(pack_kernel,
                                  dim3(1),
                                  dim3(PackStridesKernel::kBlockSize),
                                  0,
                                  static_cast<const uint64_t*>(stride_buf.GetDeviceBuffer()),
                                  n,
                                  static_cast<uint32_t*>(word_buf.GetDeviceBuffer())));

        words.resize(kNumWord * n);
        word_buf.FromDevice(words.data());
    }
};

TEST_F(TDMDescriptorStride, Dim1StrideRoundTrips)
{
    for(size_t i = 0; i < std::size(kCases); ++i)
    {
        const uint64_t expected = kCases[i].stride & k48Bit;
        EXPECT_EQ(read_dim1_stride(words.data() + i * kNumWord), expected)
            << "dim1 stride " << kCases[i].label << " did not survive the descriptor round trip";
    }
}

// Control: dim0 genuinely splits at 32 and is encoded separately, so it must
// round-trip whatever dim1 does. If this fails too, the defect is not the dim1
// split point and the diagnosis above is wrong.
TEST_F(TDMDescriptorStride, Dim0StrideRoundTripsAsControl)
{
    for(size_t i = 0; i < std::size(kCases); ++i)
    {
        const uint64_t expected = kCases[i].stride & k48Bit;
        EXPECT_EQ(read_dim0_stride(words.data() + i * kNumWord), expected)
            << "dim0 stride " << kCases[i].label << " did not survive the descriptor round trip";
    }
}

} // namespace
} // namespace test
} // namespace ck_tile
