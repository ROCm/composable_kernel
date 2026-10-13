// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <optional>
#include <string>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

#include "ck_tile/host/device_memory.hpp"
#include "fmha_reference.hpp"
#include "fmha_test_common.hpp"

#ifndef DataTypeConfig
#define DataTypeConfig FmhaFwdFp16
#endif

namespace {

using Data = typename FmhaFwdTypeConfig<DataTypeConfig>::QDataType;
static_assert(std::is_same_v<DataTypeConfig, FmhaFwdFp16> ||
              std::is_same_v<DataTypeConfig, FmhaFwdBf16>);
constexpr const char* kDataType = std::is_same_v<DataTypeConfig, FmhaFwdFp16> ? "fp16" : "bf16";
namespace reference             = ck_tile::test::reference;

enum class SequencePointers
{
    Batch,
    Packed,
    PerSequence,
    Cumulative,
    AboveCuGrid,
};

struct ForwardCase
{
    const char* name;
    int dimension;
    int query_length;
    int key_length;
    mask_enum mask;
    bool bhsd;
    int query_heads;
    int kv_heads;
    SequencePointers pointers;
    int expected_n;
    bool scheduled          = true;
    int window_size_left    = -1;
    int window_size_right   = 0;
    int second_query_length = 63;
    int second_key_length   = 513;
    bool square_causal      = false;
    bool virtual_sink       = false;
};

// Sixteen bounded inputs, each checked with and without LSE in both dtype binaries.
// Q127 must select the M64 qr_tdm fallback; every other input must select sched.
constexpr std::array<ForwardCase, 16> kCases{{
    {"D64DenseK511", 64, 128, 511, mask_enum::no_mask, false, 2, 1, SequencePointers::Batch, 64},
    {"D64TopLeftK512",
     64,
     128,
     512,
     mask_enum::mask_top_left,
     true,
     2,
     1,
     SequencePointers::Batch,
     64},
    {"D64BottomRightK513",
     64,
     128,
     513,
     mask_enum::mask_bottom_right,
     false,
     2,
     1,
     SequencePointers::Batch,
     64},
    {"D64Q127Fallback",
     64,
     127,
     512,
     mask_enum::no_mask,
     true,
     1,
     1,
     SequencePointers::Batch,
     64,
     false},
    {"D128DenseK511", 128, 128, 511, mask_enum::no_mask, false, 2, 1, SequencePointers::Batch, 128},
    {"D128TopLeftK512",
     128,
     128,
     512,
     mask_enum::mask_top_left,
     true,
     2,
     1,
     SequencePointers::Batch,
     128},
    {"D128BottomRightK513",
     128,
     128,
     513,
     mask_enum::mask_bottom_right,
     false,
     2,
     1,
     SequencePointers::Batch,
     128},
    {"D128Q127Fallback",
     128,
     127,
     512,
     mask_enum::no_mask,
     true,
     1,
     1,
     SequencePointers::Batch,
     64,
     false},
    {"D192DenseK511N64",
     192,
     128,
     511,
     mask_enum::no_mask,
     false,
     2,
     1,
     SequencePointers::Batch,
     64},
    {"D192DenseK512N128",
     192,
     128,
     512,
     mask_enum::no_mask,
     true,
     2,
     1,
     SequencePointers::Batch,
     128},
    {"D192BottomRightK513N64",
     192,
     128,
     513,
     mask_enum::mask_bottom_right,
     false,
     2,
     1,
     SequencePointers::Batch,
     64},
    {"D192Q127Fallback",
     192,
     127,
     512,
     mask_enum::no_mask,
     true,
     1,
     1,
     SequencePointers::Batch,
     64,
     false},
    {"D192GroupGridAboveCuN64",
     192,
     128,
     512,
     mask_enum::no_mask,
     false,
     1,
     1,
     SequencePointers::AboveCuGrid,
     64},
    {"D192GroupPackedN128",
     192,
     128,
     512,
     mask_enum::no_mask,
     false,
     2,
     1,
     SequencePointers::Packed,
     128},
    {"D192GroupPerSequenceTopLeftN64",
     192,
     128,
     511,
     mask_enum::mask_top_left,
     false,
     2,
     1,
     SequencePointers::PerSequence,
     64},
    {"D192GroupCumulativeBottomRightN64",
     192,
     128,
     511,
     mask_enum::mask_bottom_right,
     true,
     2,
     1,
     SequencePointers::Cumulative,
     64},
}};

// These inputs extend coverage without changing the original sixteen inputs or their indices.
constexpr std::array<ForwardCase, 14> kAdditionalCases{{
    {"D64GroupDense", 64, 128, 512, mask_enum::no_mask, false, 2, 1, SequencePointers::Packed, 64},
    {"D128GroupDense",
     128,
     128,
     512,
     mask_enum::no_mask,
     true,
     2,
     1,
     SequencePointers::Packed,
     128},
    {"D64BatchLocalN64",
     64,
     256,
     513,
     mask_enum::mask_bottom_right,
     true,
     2,
     1,
     SequencePointers::Batch,
     64,
     true,
     31,
     7},
    {"D64GroupLocalN64",
     64,
     256,
     513,
     mask_enum::mask_bottom_right,
     false,
     2,
     1,
     SequencePointers::PerSequence,
     64,
     true,
     31,
     7},
    {"D128BatchLocalN128",
     128,
     256,
     513,
     mask_enum::mask_bottom_right,
     false,
     2,
     1,
     SequencePointers::Batch,
     128,
     true,
     31,
     7},
    {"D128GroupLocalN128",
     128,
     256,
     513,
     mask_enum::mask_bottom_right,
     true,
     2,
     1,
     SequencePointers::Cumulative,
     128,
     true,
     31,
     7},
    {"D192BatchLocalN64",
     192,
     256,
     513,
     mask_enum::mask_bottom_right,
     true,
     2,
     1,
     SequencePointers::Batch,
     64,
     true,
     31,
     7},
    {"D192GroupLocalN64",
     192,
     256,
     513,
     mask_enum::mask_bottom_right,
     false,
     2,
     1,
     SequencePointers::PerSequence,
     64,
     true,
     31,
     7},
    {"D64BatchMultiMSquareCausalBhsd",
     64,
     512,
     512,
     mask_enum::mask_top_left,
     true,
     2,
     1,
     SequencePointers::Batch,
     64,
     true,
     -1,
     0,
     512,
     512,
     true},
    {"D64GroupMultiMSquareCausalBhsd",
     64,
     512,
     512,
     mask_enum::mask_bottom_right,
     true,
     2,
     1,
     SequencePointers::Packed,
     64,
     true,
     -1,
     0,
     512,
     512,
     true},
    {"D128BatchMultiMSquareCausalBhsd",
     128,
     512,
     512,
     mask_enum::mask_top_left,
     true,
     2,
     1,
     SequencePointers::Batch,
     128,
     true,
     -1,
     0,
     512,
     512,
     true},
    {"D128GroupMultiMSquareCausalBhsd",
     128,
     512,
     512,
     mask_enum::mask_bottom_right,
     true,
     2,
     1,
     SequencePointers::Packed,
     128,
     true,
     -1,
     0,
     512,
     512,
     true},
    {"D192BatchMultiMSquareCausalN128Bhsd",
     192,
     512,
     512,
     mask_enum::mask_top_left,
     true,
     2,
     1,
     SequencePointers::Batch,
     128,
     true,
     -1,
     0,
     512,
     512,
     true},
    {"D192GroupMultiMSquareCausalN128Bhsd",
     192,
     512,
     512,
     mask_enum::mask_bottom_right,
     true,
     2,
     1,
     SequencePointers::Packed,
     128,
     true,
     -1,
     0,
     512,
     512,
     true},
}};

constexpr auto kVirtualSinkCases = [] {
    // Exercise masked and D192 group finite-sink fallback.
    std::array<ForwardCase, 8> cases{kCases[0],
                                     kCases[4],
                                     kCases[8],
                                     kCases[9],
                                     kCases[10],
                                     kCases[13],
                                     kAdditionalCases[0],
                                     kAdditionalCases[1]};
    constexpr std::array<const char*, 8> names{"D64VirtualSink",
                                               "D128VirtualSink",
                                               "D192N64VirtualSink",
                                               "D192N128VirtualSink",
                                               "D192BottomRightVirtualSink",
                                               "D192GroupVirtualSink",
                                               "D64GroupVirtualSink",
                                               "D128GroupVirtualSink"};
    for(std::size_t i = 0; i < cases.size(); ++i)
    {
        cases[i].name         = names[i];
        cases[i].virtual_sink = true;
    }
    return cases;
}();

// Keep heterogeneous average-K boundaries after all earlier inputs and indices.
constexpr std::array<ForwardCase, 2> kSelectorCases{{
    {"D192GroupPackedAverageBelowN64",
     192,
     128,
     511,
     mask_enum::no_mask,
     false,
     2,
     1,
     SequencePointers::Packed,
     64,
     true,
     -1,
     0,
     63,
     512},
    {"D192GroupPackedAverageExactN128",
     192,
     128,
     511,
     mask_enum::no_mask,
     false,
     2,
     1,
     SequencePointers::Packed,
     128,
     true,
     -1,
     0,
     63,
     513},
}};

constexpr int kCaseCount = static_cast<int>(kCases.size() + kAdditionalCases.size() +
                                            kVirtualSinkCases.size() + kSelectorCases.size());

const ForwardCase& GetCase(int index)
{
    if(index < static_cast<int>(kCases.size()))
        return kCases[index];
    index -= static_cast<int>(kCases.size());
    if(index < static_cast<int>(kAdditionalCases.size()))
        return kAdditionalCases[index];
    index -= static_cast<int>(kAdditionalCases.size());
    if(index < static_cast<int>(kVirtualSinkCases.size()))
        return kVirtualSinkCases[index];
    return kSelectorCases[index - kVirtualSinkCases.size()];
}

enum class ValuePattern
{
    Signed,
    Constant,
    Nonnegative,
};

std::vector<int32_t> Starts(const std::vector<int32_t>& lengths)
{
    std::vector<int32_t> starts(lengths.size() + 1, 0);
    std::partial_sum(lengths.begin(), lengths.end(), starts.begin() + 1);
    return starts;
}

struct Sequences
{
    std::vector<int32_t> q_lengths, k_lengths, q_starts, k_starts, q_cumulative, k_cumulative;
    bool group;
    int physical_q_rows, physical_k_rows;
};

Sequences MakeSequences(const ForwardCase& c, int num_cus)
{
    Sequences sequences;
    sequences.group = c.pointers != SequencePointers::Batch;
    const int batch = c.pointers == SequencePointers::AboveCuGrid ? num_cus + 1
                      : sequences.group || c.scheduled            ? 2
                                                                  : 1;
    sequences.q_lengths.assign(batch, c.query_length);
    sequences.k_lengths.assign(batch, c.key_length);
    if(c.pointers == SequencePointers::AboveCuGrid)
    {
        // max_seqlen_q is the actual maximum (128). All CU+1 groups are nonempty,
        // so get_num_blocks(128) = (CU+1) * 1 * ceil(128/128) exceeds CU.
        std::fill(sequences.q_lengths.begin() + 1, sequences.q_lengths.end(), 1);
    }
    else if(sequences.group)
    {
        sequences.q_lengths[1] = c.second_query_length;
        sequences.k_lengths[1] = c.second_key_length;
    }
    auto physical_q = sequences.q_lengths;
    auto physical_k = sequences.k_lengths;
    if(c.pointers == SequencePointers::PerSequence || c.pointers == SequencePointers::Cumulative)
    {
        // Real gaps separate the physical starts from either logical-length API.
        for(auto& length : physical_q)
            length += 9;
        for(auto& length : physical_k)
            length += 17;
    }
    sequences.q_starts        = Starts(physical_q);
    sequences.k_starts        = Starts(physical_k);
    sequences.q_cumulative    = Starts(sequences.q_lengths);
    sequences.k_cumulative    = Starts(sequences.k_lengths);
    sequences.physical_q_rows = sequences.group ? sequences.q_starts.back() : c.query_length;
    sequences.physical_k_rows = sequences.group ? sequences.k_starts.back() : c.key_length;
    return sequences;
}

struct Layout
{
    std::size_t head_stride, row_stride, batch_stride, elements;

    Layout(int batch, int heads, int physical_rows, int dimension, bool bhsd, bool group)
        : head_stride(bhsd ? static_cast<std::size_t>(physical_rows) * dimension : dimension),
          row_stride(bhsd ? dimension : static_cast<std::size_t>(heads) * dimension),
          batch_stride(static_cast<std::size_t>(heads) * physical_rows * dimension),
          elements(batch_stride * (group ? 1 : batch))
    {
    }

    std::size_t SequenceOffset(int batch, int32_t physical_start, bool group) const
    {
        return group ? static_cast<std::size_t>(physical_start) * row_stride
                     : static_cast<std::size_t>(batch) * batch_stride;
    }

    reference::FloatView View(const std::vector<float>& data, std::size_t offset) const
    {
        return {data.data(), data.size(), offset, head_stride, row_stride, 1};
    }
};

std::vector<Data> MakeInput(const Layout& layout,
                            const std::vector<int32_t>& lengths,
                            const std::vector<int32_t>& starts,
                            int heads,
                            int dimension,
                            bool group,
                            int phase,
                            ValuePattern value_pattern = ValuePattern::Signed)
{
    // Distinct padding exposes accidental inclusion of a physical gap in softmax.
    std::vector<Data> data(layout.elements, ck_tile::type_convert<Data>(2.0f));
    for(std::size_t batch = 0; batch < lengths.size(); ++batch)
        for(int head = 0; head < heads; ++head)
            for(int row = 0; row < lengths[batch]; ++row)
                for(int dim = 0; dim < dimension; ++dim)
                {
                    float value =
                        0.35f * std::sin((row + 1) * (0.061f + phase * 0.017f) +
                                         (dim + 1) * 0.137f + head * 0.53f + batch * 0.29f) +
                        0.11f * std::cos((row + 1) * (dim % 7 + 1) * 0.031f + phase * 0.4f);
                    if(value_pattern == ValuePattern::Constant)
                    {
                        // Fixed along K, distinct per head/dimension. Multiple
                        // mantissas expose a gain that BF16(1.01) can hide for V=1.
                        constexpr std::array<float, 4> values{0.75f, -1.25f, 1.5f, -1.75f};
                        value = values[(dim + head) % values.size()] * (head % 2 == 0 ? 1 : -1);
                    }
                    else if(value_pattern == ValuePattern::Nonnegative)
                        value = std::abs(value);
                    const auto index = layout.SequenceOffset(batch, starts[batch], group) +
                                       head * layout.head_stride + row * layout.row_stride + dim;
                    data[index] = ck_tile::type_convert<Data>(value);
                }
    return data;
}

std::vector<float> Decode(const std::vector<Data>& data)
{
    std::vector<float> decoded(data.size());
    std::transform(data.begin(), data.end(), decoded.begin(), [](Data value) {
        return ck_tile::type_convert<float>(value);
    });
    return decoded;
}

std::string ExpectedKernel(const ForwardCase& c, bool lse, bool group)
{
    const int value_dimension = c.dimension == 64 ? 64 : 128;
    const int m               = c.scheduled ? 128 : 64;
    const int n               = c.virtual_sink ? 64 : c.expected_n;
    // Finite virtual sinks use the qr_tdm fallback; sched is specialized for null sink.
    const bool use_scheduled = c.scheduled && !c.virtual_sink;
    std::string name         = "fmha_fwd_d" + std::to_string(c.dimension) + "_" + kDataType +
                       (group ? "_group_b" : "_batch_b") + std::to_string(m) + "x" +
                       std::to_string(n) + "x32x" + std::to_string(value_dimension) + "x32x" +
                       std::to_string(c.dimension) + "_r4x1x1_r4x1x1_w16x16x32_w16x16x32";
    if(use_scheduled)
    {
        const int occupancy = c.dimension == 64 ? 3 : c.expected_n == 64 ? 2 : 1;
        name += "_o" + std::to_string(occupancy) + "_qr_tdm_sched_vr_";
        if(c.dimension == 64)
            name += group || c.key_length % 64 != 0 ? "pssk" : "npad";
        else
            name += group || c.dimension == 128 ? "psskddv" : "pddv";
    }
    else if(c.virtual_sink)
    {
        const bool pad_sequence = group || c.key_length % 64 != 0;
        name += "_qr_tdm_vr_";
        if(c.dimension == 192)
            name += pad_sequence ? "psskddv" : "pddv";
        else
            name += pad_sequence ? "pssk" : "npad";
    }
    else
        name += c.dimension == 192 ? "_qr_tdm_vr_pddv" : "_qr_tdm_vr_npad";
    name += "_nlogits_nbias_";
    if(c.mask == mask_enum::no_mask)
        name += "nmask";
    else if(c.virtual_sink && !lse && c.window_size_left < 0 && c.window_size_right == 0 &&
            (c.mask == mask_enum::mask_top_left || c.mask == mask_enum::mask_bottom_right))
        // The simplified-mask codegen selects its dedicated causal fallback first.
        name += "mc";
    else
        name += "mask";
    name += lse ? "_lse" : "_nlse";
    name += "_ndropout_nskip_nqscale_ntrload";
    if(!use_scheduled)
        name += "_kvlp_plk";
    return name + "_nsink";
}

class QrTdmSchedForward : public ck_tile::test::Gfx125FmhaTest,
                          public ::testing::WithParamInterface<std::tuple<int, bool, ValuePattern>>
{
};

TEST_P(QrTdmSchedForward, GeneratedDispatcherAndIndependentFp32)
{
    const auto [case_index, lse, value_pattern] = GetParam();
    const auto& c                               = GetCase(case_index);
    SCOPED_TRACE(c.name);
    int device = 0;
    hipDeviceProp_t properties{};
    ASSERT_EQ(hipGetDevice(&device), hipSuccess);
    ASSERT_EQ(hipGetDeviceProperties(&properties, device), hipSuccess);
    ASSERT_GT(properties.multiProcessorCount, 0);
    const auto sequences      = MakeSequences(c, properties.multiProcessorCount);
    const int batch           = sequences.q_lengths.size();
    const int value_dimension = c.dimension == 64 ? 64 : 128;
    const Layout q_layout(
        batch, c.query_heads, sequences.physical_q_rows, c.dimension, c.bhsd, sequences.group);
    const Layout k_layout(
        batch, c.kv_heads, sequences.physical_k_rows, c.dimension, c.bhsd, sequences.group);
    const Layout v_layout(
        batch, c.kv_heads, sequences.physical_k_rows, value_dimension, c.bhsd, sequences.group);
    const Layout o_layout(
        batch, c.query_heads, sequences.physical_q_rows, value_dimension, c.bhsd, sequences.group);
    const Layout lse_layout(
        batch, c.query_heads, sequences.physical_q_rows, 1, true, sequences.group);
    const auto q       = MakeInput(q_layout,
                             sequences.q_lengths,
                             sequences.q_starts,
                             c.query_heads,
                             c.dimension,
                             sequences.group,
                             0);
    const auto k       = MakeInput(k_layout,
                             sequences.k_lengths,
                             sequences.k_starts,
                             c.kv_heads,
                             c.dimension,
                             sequences.group,
                             1);
    const auto v       = MakeInput(v_layout,
                             sequences.k_lengths,
                             sequences.k_starts,
                             c.kv_heads,
                             value_dimension,
                             sequences.group,
                             2,
                             value_pattern);
    const auto q_float = Decode(q);
    const auto k_float = Decode(k);
    const auto v_float = Decode(v);
    const float nan    = std::numeric_limits<float>::quiet_NaN();
    std::vector<Data> o(o_layout.elements, ck_tile::type_convert<Data>(nan));
    std::vector<float> lse_data(lse_layout.elements, nan);
    ck_tile::DeviceMem q_device(q.size() * sizeof(Data)), k_device(k.size() * sizeof(Data)),
        v_device(v.size() * sizeof(Data)), o_device(o.size() * sizeof(Data)),
        lse_device(lse_data.size() * sizeof(float));
    q_device.ToDevice(q.data());
    k_device.ToDevice(k.data());
    v_device.ToDevice(v.data());
    o_device.ToDevice(o.data());
    lse_device.ToDevice(lse_data.data());
    ck_tile::DeviceMem q_starts_device(sequences.q_starts.size() * sizeof(int32_t)),
        k_starts_device(sequences.k_starts.size() * sizeof(int32_t)),
        q_lengths_device(sequences.q_lengths.size() * sizeof(int32_t)),
        k_lengths_device(sequences.k_lengths.size() * sizeof(int32_t)),
        q_cumulative_device(sequences.q_cumulative.size() * sizeof(int32_t)),
        k_cumulative_device(sequences.k_cumulative.size() * sizeof(int32_t));
    q_starts_device.ToDevice(sequences.q_starts.data());
    k_starts_device.ToDevice(sequences.k_starts.data());
    q_lengths_device.ToDevice(sequences.q_lengths.data());
    k_lengths_device.ToDevice(sequences.k_lengths.data());
    q_cumulative_device.ToDevice(sequences.q_cumulative.data());
    k_cumulative_device.ToDevice(sequences.k_cumulative.data());

    std::vector<float> sink_logits(c.query_heads);
    for(int head = 0; head < c.query_heads; ++head)
        // Keep the V=0 term large enough that omitting it fails even without LSE.
        sink_logits[head] = head % 2 == 0 ? 8.0f : 10.0f;
    std::optional<ck_tile::DeviceMem> sink_device;
    if(c.virtual_sink)
    {
        sink_device.emplace(sink_logits.size() * sizeof(float));
        sink_device->ToDevice(sink_logits.data());
    }

    fmha_fwd_args args{};
    std::string selected_kernel;
    args.selected_kernel_name = &selected_kernel;
    args.q_ptr                = q_device.GetDeviceBuffer();
    args.k_ptr                = k_device.GetDeviceBuffer();
    args.v_ptr                = v_device.GetDeviceBuffer();
    args.o_ptr                = o_device.GetDeviceBuffer();
    args.lse_ptr              = lse ? lse_device.GetDeviceBuffer() : nullptr;
    args.sink_ptr             = sink_device ? sink_device->GetDeviceBuffer() : nullptr;
    args.batch                = batch;
    args.nhead_q              = c.query_heads;
    args.nhead_k              = c.kv_heads;
    args.hdim_q               = c.dimension;
    args.hdim_v               = value_dimension;
    args.seqlen_q             = sequences.group ? sequences.q_starts.back() : c.query_length;
    args.seqlen_k             = sequences.group ? sequences.k_starts.back() : c.key_length;
    args.max_seqlen_q   = *std::max_element(sequences.q_lengths.begin(), sequences.q_lengths.end());
    args.scale_s        = 0.19f;
    args.stride_q       = q_layout.row_stride;
    args.stride_k       = k_layout.row_stride;
    args.stride_v       = v_layout.row_stride;
    args.stride_o       = o_layout.row_stride;
    args.nhead_stride_q = q_layout.head_stride;
    args.nhead_stride_k = k_layout.head_stride;
    args.nhead_stride_v = v_layout.head_stride;
    args.nhead_stride_o = o_layout.head_stride;
    args.nhead_stride_lse  = lse_layout.head_stride;
    args.batch_stride_q    = q_layout.batch_stride;
    args.batch_stride_k    = k_layout.batch_stride;
    args.batch_stride_v    = v_layout.batch_stride;
    args.batch_stride_o    = o_layout.batch_stride;
    args.batch_stride_lse  = lse_layout.batch_stride;
    args.mask_type         = static_cast<int>(c.mask);
    args.window_size_left  = c.window_size_left;
    args.window_size_right = c.mask == mask_enum::no_mask ? -1 : c.window_size_right;
    if(sequences.group)
    {
        args.seqstart_q_ptr = q_starts_device.GetDeviceBuffer();
        args.seqstart_k_ptr = k_starts_device.GetDeviceBuffer();
    }
    if(c.pointers == SequencePointers::PerSequence)
    {
        args.seqlen_q_ptr = q_lengths_device.GetDeviceBuffer();
        args.seqlen_k_ptr = k_lengths_device.GetDeviceBuffer();
    }
    if(c.pointers == SequencePointers::Cumulative)
    {
        args.cu_seqlen_q_ptr = q_cumulative_device.GetDeviceBuffer();
        args.cu_seqlen_k_ptr = k_cumulative_device.GetDeviceBuffer();
    }
    const auto grid_blocks =
        static_cast<std::size_t>(batch) * c.query_heads * ((args.max_seqlen_q + 127) / 128);
    const int selector_begin =
        static_cast<int>(kCases.size() + kAdditionalCases.size() + kVirtualSinkCases.size());
    if(case_index >= selector_begin &&
       case_index < selector_begin + static_cast<int>(kSelectorCases.size()))
    {
        // All N128 predicates except average K are true; physical starts are packed.
        ASSERT_EQ(c.pointers, SequencePointers::Packed);
        ASSERT_TRUE(sequences.group);
        ASSERT_TRUE(c.scheduled);
        ASSERT_FALSE(c.virtual_sink);
        ASSERT_EQ(c.mask, mask_enum::no_mask);
        ASSERT_EQ(args.window_size_left, -1);
        ASSERT_EQ(args.window_size_right, -1);
        ASSERT_EQ(args.hdim_q, 192);
        ASSERT_EQ(args.hdim_v, 128);
        ASSERT_EQ(batch, 2);
        ASSERT_EQ(args.nhead_q, 2);
        ASSERT_EQ(args.nhead_k, 1);
        ASSERT_EQ(sequences.q_lengths, (std::vector<int32_t>{128, 63}));
        ASSERT_EQ(sequences.k_lengths, (std::vector<int32_t>{511, c.second_key_length}));
        ASSERT_EQ(sequences.q_starts, (std::vector<int32_t>{0, 128, 191}));
        ASSERT_EQ(sequences.k_starts, (std::vector<int32_t>{0, 511, args.seqlen_k}));
        ASSERT_EQ(sequences.q_starts, sequences.q_cumulative);
        ASSERT_EQ(sequences.k_starts, sequences.k_cumulative);
        ASSERT_EQ(args.seqlen_q, 191);
        ASSERT_EQ(args.max_seqlen_q, 128);
        ASSERT_EQ(args.seqlen_q_ptr, nullptr);
        ASSERT_EQ(args.seqlen_k_ptr, nullptr);
        ASSERT_EQ(args.cu_seqlen_q_ptr, nullptr);
        ASSERT_EQ(args.cu_seqlen_k_ptr, nullptr);
        ASSERT_NE(args.seqstart_q_ptr, nullptr);
        ASSERT_NE(args.seqstart_k_ptr, nullptr);
        ASSERT_EQ(grid_blocks, 4);
        ASSERT_LE(grid_blocks, static_cast<std::size_t>(properties.multiProcessorCount));
        ASSERT_EQ(args.seqlen_k, c.expected_n == 64 ? 512 * batch - 1 : 512 * batch);
        ASSERT_EQ(args.seqlen_k >= 512 * batch, c.expected_n == 128);
    }
    if(c.window_size_left >= 0)
    {
        ASSERT_EQ(c.mask, mask_enum::mask_bottom_right);
        ASSERT_GT(c.query_length, 128);
        for(int sequence = 0; sequence < batch; ++sequence)
        {
            // At Q origin zero, the first window tile already has a nonzero K origin.
            const int first_key = std::max(sequences.k_lengths[sequence] -
                                               sequences.q_lengths[sequence] - c.window_size_left,
                                           0);
            ASSERT_GT((first_key / c.expected_n) * c.expected_n, 0);
        }
        if(sequences.group)
        {
            ASSERT_GT(sequences.q_starts[1], sequences.q_lengths[0]);
            ASSERT_GT(sequences.k_starts[1], sequences.k_lengths[0]);
        }
    }
    if(c.square_causal)
    {
        ASSERT_TRUE(c.bhsd);
        ASSERT_GT(c.query_length, 128);
        ASSERT_LT(args.window_size_left, 0);
        ASSERT_EQ(args.window_size_right, 0);
        ASSERT_EQ(sequences.q_lengths, sequences.k_lengths);
        ASSERT_EQ(args.seqlen_q, args.seqlen_k);
    }
    if(c.scheduled && c.dimension == 192 && c.expected_n == 128)
    {
        ASSERT_GE(args.seqlen_k, 512 * (sequences.group ? batch : 1));
        ASSERT_EQ(args.seqlen_k_ptr, nullptr);
        ASSERT_EQ(args.cu_seqlen_k_ptr, nullptr);
        ASSERT_LE(grid_blocks, static_cast<std::size_t>(properties.multiProcessorCount));
        if(c.square_causal)
            ASSERT_TRUE(c.mask == mask_enum::mask_top_left ||
                        c.mask == mask_enum::mask_bottom_right);
        else
            ASSERT_EQ(c.mask, mask_enum::no_mask);
    }
    if(c.pointers == SequencePointers::AboveCuGrid)
    {
        // Every N128 predicate except grid<=CU is true. No fabricated maximum or
        // empty sequence is used to make this branch reachable on any CU count.
        ASSERT_EQ(args.max_seqlen_q, 128);
        ASSERT_EQ(args.seqlen_k, 512 * batch);
        ASSERT_EQ(args.seqlen_k_ptr, nullptr);
        ASSERT_EQ(args.cu_seqlen_k_ptr, nullptr);
        ASSERT_EQ(c.mask, mask_enum::no_mask);
        ASSERT_EQ(grid_blocks, static_cast<std::size_t>(properties.multiProcessorCount) + 1);
    }
    fmha_fwd_traits traits{};
    traits.hdim_q        = c.dimension;
    traits.hdim_v        = value_dimension;
    traits.data_type     = kDataType;
    traits.is_group_mode = sequences.group;
    traits.is_v_rowmajor = true;
    traits.mask_type     = c.mask;
    traits.bias_type     = bias_enum::no_bias;
    traits.has_lse       = lse;
    traits.qscale_type   = quant_scale_enum::no_scale;
    // A runtime V=0 sink does not request compile-time real-prefix sink tokens.
    ASSERT_FALSE(traits.has_sink);
    const ck_tile::stream_config stream{nullptr, false, 1, 0, 1};
    const float result = fmha_fwd(traits, args, stream);
    std::cout << '\n'
              << "generated_dispatch case=" << c.name << " dtype=" << kDataType << " lse=" << lse
              << " value_pattern=" << static_cast<int>(value_pattern)
              << " cus=" << properties.multiProcessorCount
              << " selector_m128_blocks=" << grid_blocks << " actual_kernel=" << selected_kernel
              << '\n';
    ASSERT_GE(result, 0) << "Generated dispatcher has no supported instance";
    ASSERT_EQ(hipGetLastError(), hipSuccess);
    ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);
    ASSERT_EQ(selected_kernel, ExpectedKernel(c, lse, sequences.group));
    o_device.FromDevice(o.data());
    if(lse)
        lse_device.FromDevice(lse_data.data());
    const auto o_float = Decode(o);

    std::vector<reference::Sequence> reference_sequences;
    std::vector<reference::Output> outputs;
    std::size_t logical_rows = 0;
    for(int sequence = 0; sequence < batch; ++sequence)
    {
        const auto query_offset =
            q_layout.SequenceOffset(sequence, sequences.q_starts[sequence], sequences.group);
        const auto key_offset =
            k_layout.SequenceOffset(sequence, sequences.k_starts[sequence], sequences.group);
        const auto value_offset =
            v_layout.SequenceOffset(sequence, sequences.k_starts[sequence], sequences.group);
        reference::Sequence input;
        input.q               = q_layout.View(q_float, query_offset);
        input.k               = k_layout.View(k_float, key_offset);
        input.v               = v_layout.View(v_float, value_offset);
        input.query_length    = sequences.q_lengths[sequence];
        input.key_length      = sequences.k_lengths[sequence];
        input.query_heads     = c.query_heads;
        input.kv_heads        = c.kv_heads;
        input.query_dimension = c.dimension;
        input.value_dimension = value_dimension;
        input.score_scale     = args.scale_s;
        if(c.virtual_sink)
            input.sink_logits = sink_logits;
        if(c.mask != mask_enum::no_mask)
        {
            // Signed causal/local intervals are independent of CK's mask implementation.
            const int alignment =
                c.mask == mask_enum::mask_bottom_right
                    ? sequences.k_lengths[sequence] - sequences.q_lengths[sequence]
                    : 0;
            input.is_masked = [alignment, left = c.window_size_left, right = c.window_size_right](
                                  std::size_t query, std::size_t key) {
                const auto center = static_cast<int64_t>(query) + alignment;
                const auto column = static_cast<int64_t>(key);
                return (left >= 0 && column < center - left) ||
                       (right >= 0 && column > center + right);
            };
        }
        logical_rows += input.query_length * input.query_heads;
        reference_sequences.push_back(std::move(input));
        const auto output_offset =
            o_layout.SequenceOffset(sequence, sequences.q_starts[sequence], sequences.group);
        reference::Output output{o_layout.View(o_float, output_offset), std::nullopt};
        if(lse)
        {
            const auto lse_offset =
                lse_layout.SequenceOffset(sequence, sequences.q_starts[sequence], sequences.group);
            output.lse = lse_layout.View(lse_data, lse_offset);
        }
        outputs.push_back(output);
    }
    // Empty sampled_query_rows means every logical row. The oracle retains FP32
    // softmax probabilities; it does not quantize P to match the device pipeline.
    const auto expected = reference::Compute(reference_sequences);
    const reference::AccuracyContract contract{
        std::is_same_v<DataTypeConfig, FmhaFwdFp16> ? 10 : 7,
        static_cast<std::size_t>(c.virtual_sink ? 64 : c.expected_n)};
    const auto metrics = reference::Compare(reference_sequences, expected, outputs, {}, contract);
    std::cout << std::setprecision(10) << "independent_fp32 checked_rows=" << metrics.checked_rows
              << " checked_o=" << metrics.checked_elements
              << " checked_lse=" << metrics.checked_lse_rows
              << " max_o_error=" << metrics.max_abs_error
              << " min_row_cosine=" << metrics.min_row_cosine
              << " mean_row_cosine=" << metrics.mean_row_cosine
              << " relative_l2=" << metrics.relative_rms_error
              << " max_row_relative_l2=" << metrics.max_row_relative_l2
              << " max_head_gain=" << metrics.max_head_gain
              << " max_component_ratio=" << metrics.max_component_ratio
              << " component_mismatches=" << metrics.component_mismatch_count
              << " strict_passed=" << metrics.strict_passed
              << " contract_passed=" << metrics.contract_passed
              << " max_lse_error=" << metrics.max_lse_abs_error
              << " nonfinite_o=" << metrics.nonfinite_output_count
              << " invalid_lse=" << metrics.invalid_lse_count << " passed=" << metrics.passed
              << '\n';
    EXPECT_EQ(metrics.checked_rows, logical_rows);
    EXPECT_EQ(metrics.checked_elements, logical_rows * value_dimension);
    EXPECT_EQ(metrics.scanned_output_elements, logical_rows * value_dimension);
    EXPECT_EQ(metrics.checked_lse_rows, lse ? logical_rows : 0);
    EXPECT_EQ(metrics.scanned_lse_rows, lse ? logical_rows : 0);
    EXPECT_TRUE(metrics.passed);
}

INSTANTIATE_TEST_SUITE_P(
    SmallGeneratedDispatch,
    QrTdmSchedForward,
    ::testing::Combine(
        ::testing::Range(0, kCaseCount),
        ::testing::Bool(),
        ::testing::Values(ValuePattern::Signed, ValuePattern::Constant, ValuePattern::Nonnegative)),
    [](const ::testing::TestParamInfo<QrTdmSchedForward::ParamType>& case_info) {
        return std::string{GetCase(std::get<0>(case_info.param)).name} +
               (std::get<1>(case_info.param) ? "Lse" : "NoLse") +
               (std::get<2>(case_info.param) == ValuePattern::Signed     ? "SignedV"
                : std::get<2>(case_info.param) == ValuePattern::Constant ? "ConstantV"
                                                                         : "NonnegativeV");
    });

} // namespace
