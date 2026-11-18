// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// Descriptor readback for the TDM element size.
//
// The TDM counts dims, tile dims and strides in units of its data size, which is
// 1, 2, 4 or 8 bytes. A 12-byte pack of sixteen 6-bit values (pk_fp6x16_t) is
// none of those, so TDMDescriptor describes it in bytes: it uses the 1-byte data
// size and scales the contiguous dim, the contiguous tile dim and every stride by
// 12. A rank-2 pack descriptor must therefore equal, word for word, the descriptor
// of a 1-byte type whose contiguous extents and strides are 12 times larger.
//
// Builds both through the real createTDMDescriptor on device, copies every D#
// word back and compares them, then decodes pack descriptors to absolute byte
// values and, as controls, descriptors of 1-, 2-, 4- and 8-byte types, whose
// fields must equal their inputs. No TDM instruction is issued: the kernel only
// builds the descriptors and copies their words out.

#include <cstddef>
#include <cstdint>
#include <type_traits>
#include <vector>

#include <gtest/gtest.h>

#include "ck_tile/core.hpp"
#include "ck_tile/host.hpp"
#include "ck_tile/host/kernel_launch.hpp"

namespace ck_tile {
namespace test {
namespace {

constexpr int kNumWord     = 28; // D# groups 0..4: 4 + 8 + 4 + 4 + 8 words
constexpr int kGroup1Word  = 4;  // group 1 (sgpr0..sgpr7) starts after group 0
constexpr int kRank        = 2;  // the highest rank a 12-byte element supports
constexpr int kNumRowIndex = 16;
constexpr int kPackBytes   = static_cast<int>(sizeof(pk_fp6x16_t));

// Nothing is dereferenced and no TDM instruction is issued, so fixed addresses
// stand in for the global and LDS buffers. Group 0 carries them, which shows
// that a case's words were written at all.
constexpr uint64_t kGlobalAddr = 0x123456789000ull;
constexpr uint32_t kLdsAddr    = 0x400u;

template <typename T>
void expect_traits(const char* name,
                   index_t unit_bytes,
                   index_t units_per_element,
                   uint32_t data_size_code,
                   std::size_t alignment,
                   index_t max_box_dim0)
{
    using Traits = TDMElementTraits<T>;
    EXPECT_EQ(Traits::unit_bytes, unit_bytes) << name;
    EXPECT_EQ(Traits::units_per_element, units_per_element) << name;
    EXPECT_EQ(Traits::data_size_code, data_size_code) << name;
    EXPECT_EQ(Traits::alignment, alignment) << name;
    EXPECT_EQ(Traits::max_box_dim0, max_box_dim0) << name;
}

// Every type the TDM already moves is its own unit, so its descriptors are
// unchanged; double is the only 8-byte type checked here (data size code 3).
// The 12-byte pack is twelve 1-byte units; tile_dim0 holds at most 5461 packs.
TEST(TDMDescriptorElementSize, ElementTraits)
{
    expect_traits<fp8_t>("fp8_t", 1, 1, 0u, 1, 0xFFFF);
    expect_traits<bf8_t>("bf8_t", 1, 1, 0u, 1, 0xFFFF);
    expect_traits<int8_t>("int8_t", 1, 1, 0u, 1, 0xFFFF);
    expect_traits<pk_fp4_t>("pk_fp4_t", 1, 1, 0u, 1, 0xFFFF);
    expect_traits<half_t>("half_t", 2, 1, 1u, 2, 0xFFFF);
    expect_traits<bf16_t>("bf16_t", 2, 1, 1u, 2, 0xFFFF);
    expect_traits<float>("float", 4, 1, 2u, 4, 0xFFFF);
    expect_traits<double>("double", 8, 1, 3u, 8, 0xFFFF);
    expect_traits<pk_fp6x16_t>("pk_fp6x16_t", 1, 12, 0u, 4, 5461);
}

// Inputs as a caller passes them: counts of the element type.
struct DescCase
{
    uint32_t dims[kRank];
    uint64_t strides[kRank];
    uint16_t box[kRank];
    uint16_t row_index[kNumRowIndex];
    bool pad_enable;
    uint32_t pad_interval;
    uint32_t pad_amount;
};
static_assert(std::is_trivially_copyable_v<DescCase>,
              "DescCase is copied to the device byte for byte");

struct NamedCase
{
    const char* label;
    DescCase desc;
};

// Twin of a pack case for a 1-byte type: the contiguous dim, its tile dim and
// every stride in bytes; everything else as given. The scaled values must fit
// their fields: an overflowing case would wrap identically on device and here,
// and the comparison would then pass vacuously.
DescCase byte_twin(DescCase c)
{
    EXPECT_LE(int{c.box[0]}, TDMElementTraits<pk_fp6x16_t>::max_box_dim0);
    EXPECT_LE(c.dims[0], UINT32_MAX / static_cast<uint32_t>(kPackBytes));
    c.dims[0] *= kPackBytes;
    c.box[0] = static_cast<uint16_t>(c.box[0] * kPackBytes);
    for(uint64_t& s : c.strides)
    {
        EXPECT_LE(s, ((uint64_t{1} << 48) - 1) / static_cast<uint64_t>(kPackBytes));
        s *= kPackBytes;
    }
    return c;
}

template <typename T, bool Gather>
struct ReadBackKernel
{
    // One wave per case: getResourceDescriptorGroup reads every group through
    // the first active lane, so each case needs a wave of its own.
    static constexpr index_t kBlockSize = 32;

    // Only the device pass odr-uses this; see the note in tdm_descriptor_stride.cpp.
    [[maybe_unused]] CK_TILE_DEVICE void operator()(const DescCase* cases, uint32_t* out) const
    {
        const DescCase& c = cases[blockIdx.x];

        // The descriptor stores pointers to these and reads them only in
        // getResourceDescriptorGroup(), so they must outlive that call.
        uint32_t dims[kRank];
        uint64_t strides[kRank];
        uint16_t box[kRank];
        uint16_t row_index[kNumRowIndex];
        for(int i = 0; i < kRank; ++i)
        {
            dims[i]    = c.dims[i];
            strides[i] = c.strides[i];
            box[i]     = c.box[i];
        }
        for(int i = 0; i < kNumRowIndex; ++i)
            row_index[i] = c.row_index[i];

        TDMConfig config{};
        config.pad_enable              = c.pad_enable;
        config.pad_config.pad_interval = c.pad_interval;
        config.pad_config.pad_amount   = c.pad_amount;

        const auto desc = createTDMDescriptor<T, kRank, Gather>(
            reinterpret_cast<const void*>(kGlobalAddr),
            reinterpret_cast<void*>(static_cast<uintptr_t>(kLdsAddr)),
            dims,
            strides,
            box,
            config,
            Gather ? row_index : nullptr,
            TDMGatherIndexSize::Row16bit_Index);
        const auto groups = desc.getResourceDescriptorGroup();

        if(threadIdx.x != 0)
            return;
        uint32_t* w = out + blockIdx.x * kNumWord;
        int k       = 0;
        static_for<0, 5, 1>{}([&](auto g) {
            const auto v              = groups.at(g);
            constexpr int num_element = sizeof(v) / sizeof(int32_t);
            for(int e = 0; e < num_element; ++e)
                w[k++] = static_cast<uint32_t>(v[e]);
        });
    }
};

template <typename T, bool Gather>
std::vector<uint32_t> read_back(const std::vector<DescCase>& cases)
{
    using Kernel = ReadBackKernel<T, Gather>;

    DeviceMem case_buf(sizeof(DescCase) * cases.size());
    DeviceMem word_buf(sizeof(uint32_t) * kNumWord * cases.size());
    case_buf.ToDevice(cases.data());
    word_buf.SetZero();

    launch_kernel(stream_config{},
                  make_kernel(Kernel{},
                              dim3(static_cast<uint32_t>(cases.size())),
                              dim3(Kernel::kBlockSize),
                              0,
                              static_cast<const DescCase*>(case_buf.GetDeviceBuffer()),
                              static_cast<uint32_t*>(word_buf.GetDeviceBuffer())));

    std::vector<uint32_t> words(kNumWord * cases.size());
    word_buf.FromDevice(words.data());
    return words;
}

// Reconstructed from the bitfield widths rather than by calling the setters'
// inverse, so that a decoder cannot cancel out an encoder defect.
struct Group1Fields
{
    uint32_t data_size;
    uint32_t tensor_dim0;
    uint32_t tensor_dim1;
    uint32_t tile_dim0;
    uint32_t tile_dim1;
    uint64_t dim0_stride;
    uint64_t dim1_stride;
};

Group1Fields decode_group1(const uint32_t* words)
{
    const uint32_t* w = words + kGroup1Word;
    return Group1Fields{(w[0] >> 16) & 0x3u,
                        (w[1] >> 16) | ((w[2] & 0xFFFFu) << 16),
                        (w[2] >> 16) | ((w[3] & 0xFFFFu) << 16),
                        w[3] >> 16,
                        w[4] & 0xFFFFu,
                        uint64_t{w[5]} | (uint64_t{w[6] & 0xFFFFu} << 32),
                        uint64_t{(w[6] >> 16) & 0xFFFFu} | (uint64_t{w[7]} << 16)};
}

constexpr DescCase
rank2_case(uint32_t dim0, uint32_t dim1, uint64_t stride0, uint16_t box0, uint16_t box1)
{
    DescCase c{};
    c.dims[0]    = dim0;
    c.dims[1]    = dim1;
    c.strides[0] = stride0;
    c.strides[1] = 1;
    c.box[0]     = box0;
    c.box[1]     = box1;
    return c;
}

// Every field distinct, so that a field written from the wrong source cannot
// match by coincidence: 16 packs (256 values) remain in the row, the row pitch
// is 24 packs (384 values), the box is 8 packs (128 values) by 20 rows. In
// bytes: dim0 192, pitch 288, tile_dim0 96.
constexpr NamedCase kDistinctCase{"every field distinct", rank2_case(16, 64, 24, 8, 20)};

// In bytes: dim0 786444 (0xC000C) has both 16-bit halves non-zero, the dim0
// stride 0x100000008 needs the 16-bit high half, the dim1 stride 0x10008 needs
// the 32-bit high half, and tile_dim0 is 65532, the largest pack box.
constexpr NamedCase kSplitCase{"fields crossing their split points", [] {
                                   DescCase c   = rank2_case(65537, 70000, 0x15555556, 5461, 1);
                                   c.strides[1] = 0x1556;
                                   return c;
                               }()};

// A row of 128 6-bit values is 8 packs; strides and boxes in packs.
constexpr NamedCase kRank2Cases[] = {
    {"one wave's box of a 64x128 tile", rank2_case(8, 64, 8, 8, 16)},
    {"one box for the whole 64x128 tile", rank2_case(8, 64, 8, 8, 64)},
    {"row pitch of 6144 values", rank2_case(384, 512, 384, 8, 16)},
    kDistinctCase,
    kSplitCase,
    {"LDS padding fields pass through",
     [] {
         DescCase c     = kDistinctCase.desc;
         c.pad_enable   = true;
         c.pad_interval = 3;
         c.pad_amount   = 1;
         return c;
     }()},
};

// A case whose kernel wrote nothing reads back as zeros on both sides and would
// compare equal, so first check that group 0 holds the fixed addresses.
template <std::size_t N>
void expect_written(const std::vector<uint32_t>& words, const NamedCase (&cases)[N])
{
    ASSERT_EQ(words.size(), N * kNumWord);
    for(std::size_t i = 0; i < N; ++i)
    {
        const uint32_t* group0 = words.data() + i * kNumWord;
        EXPECT_EQ(group0[1], kLdsAddr) << cases[i].label << ": group 0 was not written";
        EXPECT_EQ(group0[2], static_cast<uint32_t>(kGlobalAddr))
            << cases[i].label << ": group 0 was not written";
    }
}

// Reads each case back as packs and its byte twin back as fp8, and expects
// identical words.
template <bool Gather, std::size_t N>
void expect_pack_matches_byte_twin(const NamedCase (&cases)[N])
{
    std::vector<DescCase> packs;
    std::vector<DescCase> twins;
    for(const NamedCase& c : cases)
    {
        packs.push_back(c.desc);
        twins.push_back(byte_twin(c.desc));
    }
    const std::vector<uint32_t> pack_words = read_back<pk_fp6x16_t, Gather>(packs);
    const std::vector<uint32_t> twin_words = read_back<fp8_t, Gather>(twins);
    expect_written(pack_words, cases);
    expect_written(twin_words, cases);
    ASSERT_EQ(pack_words.size(), twin_words.size());
    for(std::size_t i = 0; i < pack_words.size(); ++i)
    {
        EXPECT_EQ(pack_words[i], twin_words[i])
            << cases[i / kNumWord].label << ": word " << i % kNumWord
            << " differs between the pack and its 1-byte twin";
    }
}

TEST(TDMDescriptorElementSize, PackMatchesByteTwin)
{
    expect_pack_matches_byte_twin<false>(kRank2Cases);
}

TEST(TDMDescriptorElementSize, PackMatchesByteTwinGather)
{
    DescCase c = kDistinctCase.desc;
    for(int i = 0; i < kNumRowIndex; ++i)
        c.row_index[i] = static_cast<uint16_t>(3 * i + 1);
    const NamedCase cases[] = {{"gather: tile_dim0 scales, the row indices do not", c}};
    expect_pack_matches_byte_twin<true>(cases);
}

TEST(TDMDescriptorElementSize, PackDecodesToBytes)
{
    const NamedCase cases[] = {kDistinctCase, kSplitCase};
    const std::vector<uint32_t> words =
        read_back<pk_fp6x16_t, false>({cases[0].desc, cases[1].desc});
    expect_written(words, cases);

    const Group1Fields d = decode_group1(words.data());
    EXPECT_EQ(d.data_size, 0u);
    EXPECT_EQ(d.tensor_dim0, 192u);
    EXPECT_EQ(d.tensor_dim1, 64u);
    EXPECT_EQ(d.tile_dim0, 96u);
    EXPECT_EQ(d.tile_dim1, 20u);
    EXPECT_EQ(d.dim0_stride, uint64_t{288});
    EXPECT_EQ(d.dim1_stride, uint64_t{12});

    const Group1Fields s = decode_group1(words.data() + kNumWord);
    EXPECT_EQ(s.data_size, 0u);
    EXPECT_EQ(s.tensor_dim0, 786444u);
    EXPECT_EQ(s.tensor_dim1, 70000u);
    EXPECT_EQ(s.tile_dim0, 65532u);
    EXPECT_EQ(s.tile_dim1, 1u);
    EXPECT_EQ(s.dim0_stride, uint64_t{0x100000008});
    EXPECT_EQ(s.dim1_stride, uint64_t{0x10008});
}

// Control: types that are their own TDM unit keep every field as given.
template <typename T>
void expect_fields_unchanged(uint32_t data_size)
{
    const NamedCase cases[]           = {kDistinctCase, kSplitCase};
    const std::vector<uint32_t> words = read_back<T, false>({cases[0].desc, cases[1].desc});
    expect_written(words, cases);
    for(std::size_t i = 0; i < 2; ++i)
    {
        const Group1Fields f = decode_group1(words.data() + i * kNumWord);
        const DescCase& c    = cases[i].desc;
        EXPECT_EQ(f.data_size, data_size) << cases[i].label;
        EXPECT_EQ(f.tensor_dim0, c.dims[0]) << cases[i].label;
        EXPECT_EQ(f.tensor_dim1, c.dims[1]) << cases[i].label;
        EXPECT_EQ(f.tile_dim0, uint32_t{c.box[0]}) << cases[i].label;
        EXPECT_EQ(f.tile_dim1, uint32_t{c.box[1]}) << cases[i].label;
        EXPECT_EQ(f.dim0_stride, c.strides[0]) << cases[i].label;
        EXPECT_EQ(f.dim1_stride, c.strides[1]) << cases[i].label;
    }
}

TEST(TDMDescriptorElementSize, UnitSizedTypesKeepTheirFields)
{
    expect_fields_unchanged<fp8_t>(0u);
    expect_fields_unchanged<half_t>(1u);
    expect_fields_unchanged<float>(2u);
    expect_fields_unchanged<double>(3u);
}

} // namespace
} // namespace test
} // namespace ck_tile
