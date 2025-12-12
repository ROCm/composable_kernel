// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "ck_tile/builder/testing/error.hpp"
#include "ck_tile/builder/testing/tensor_buffer.hpp"
#include "ck_tile/builder/testing/validation.hpp"
#include "ck_tile/builder/factory/helpers/ck/conv_tensor_type.hpp"
#include "testing_utils.hpp"
#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include <span>
#include <vector>

namespace ckb = ck_tile::builder;
namespace ckt = ck_tile::builder::test;

using testing::ElementsAreArray;
using testing::Eq;
using testing::StrEq;

std::vector<size_t> make_packed_strides_row_major(std::span<const size_t> lengths)
{
    if(lengths.size() == 0)
        return {};

    std::vector<size_t> strides(lengths.size());
    strides[strides.size() - 1] = 1;
    size_t i                    = strides.size() - 1;
    while(i > 0)
    {
        --i;
        strides[i] = strides[i + 1] * lengths[i + 1];
    }
    return strides;
}

template <int BLOCK_SIZE, ckb::DataType DT, typename F>
__global__ __launch_bounds__(BLOCK_SIZE) //
    void fill_kernel(const uint64_t n, void* data, F f)
{
    using CKType = typename ckb::factory::internal::DataTypeToCK<DT>::type;
    auto* ptr    = static_cast<CKType*>(data);

    const auto gid = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    for(uint64_t i = gid; i < n; i += gridDim.x * BLOCK_SIZE)
    {
        f(ptr[i], i);
    }
}

template <ckb::DataType DT, typename F>
void fill_tensor_buffer(const ckt::TensorDescriptor<DT>& descriptor, void* buffer, F f)
{
    constexpr int block_size = 256;
    const auto kernel        = fill_kernel<block_size, DT, F>;
    int occupancy;
    ckt::check_hip(hipOccupancyMaxActiveBlocksPerMultiprocessor(&occupancy, kernel, block_size, 0));

    const auto num_elements = descriptor.get_element_space_size();

    kernel<<<occupancy, block_size>>>(num_elements, buffer, f);
    ckt::check_hip(hipGetLastError());
}

TEST(ValidationUtilities, MakePackedStrides)
{
    const std::vector<size_t> lengths = {5125, 623, 1177, 1534};
    const auto strides                = make_packed_strides_row_major(lengths);

    EXPECT_THAT(strides, ElementsAreArray({623 * 1177 * 1534, 1177 * 1534, 1534, 1}));
}

TEST(ValidationUtilities, FillTensorBuffer)
{
    const std::vector<size_t> lengths = {31, 54, 13};
    const auto strides                = make_packed_strides_row_major(lengths);
    ckt::TensorDescriptor<ckb::DataType::INT32> descriptor(lengths, strides);
    auto buffer = ckt::alloc_tensor_buffer(descriptor);

    fill_tensor_buffer(
        descriptor, buffer.get(), [](auto& value, size_t i) { value = static_cast<uint32_t>(i); });

    std::vector<uint32_t> h_buffer(descriptor.get_element_space_size());
    ckt::check_hip(hipMemcpy(
        h_buffer.data(), buffer.get(), h_buffer.size() * sizeof(uint32_t), hipMemcpyDeviceToHost));

    for(size_t i = 0; i < h_buffer.size(); ++i)
    {
        EXPECT_THAT(h_buffer[i], Eq(static_cast<uint32_t>(i)));
    }
}

TEST(ValidationReport, SingleCorrect)
{
    const std::vector<size_t> lengths = {52, 152, 224};
    const auto strides                = make_packed_strides_row_major(lengths);

    ckt::TensorDescriptor<ckb::DataType::FP32> descriptor(lengths, strides);

    auto a = ckt::alloc_tensor_buffer(descriptor);
    auto b = ckt::alloc_tensor_buffer(descriptor);

    // Generate a sort-of-random looking sequence
    auto generator = [](auto& value, size_t i) {
        value = i * static_cast<float>(10'000'019 % 768'351);
    };

    fill_tensor_buffer(descriptor, a.get(), generator);
    fill_tensor_buffer(descriptor, b.get(), generator);

    ckt::ValidationReport report;
    report.check("correct", descriptor, b.get(), a.get());

    EXPECT_THAT(report.get_errors().size(), Eq(0));
}

TEST(ValidationReport, SingleIncorrect)
{
    const std::vector<size_t> lengths = {100, 100, 100};
    const auto strides                = make_packed_strides_row_major(lengths);

    ckt::TensorDescriptor<ckb::DataType::FP16> descriptor(lengths, strides);

    auto a = ckt::alloc_tensor_buffer(descriptor);
    auto b = ckt::alloc_tensor_buffer(descriptor);

    fill_tensor_buffer(
        descriptor, a.get(), [](auto& value, [[maybe_unused]] size_t i) { value = 123; });
    fill_tensor_buffer(descriptor, b.get(), [](auto& value, [[maybe_unused]] size_t i) {
        value = i == 12345 ? 456 : i == 999999 ? 1 : 123;
    });

    ckt::ValidationReport report;
    report.check("incorrect", descriptor, b.get(), a.get());

    const auto errors = report.get_errors();

    EXPECT_THAT(errors.size(), Eq(1));
    EXPECT_THAT(errors[0].tensor_name, StrEq("incorrect"));
    EXPECT_THAT(errors[0].wrong_elements, Eq(2));
    EXPECT_THAT(errors[0].total_elements, Eq(descriptor.get_element_space_size()));
}

TEST(ValidationReport, MultipleSomeIncorrect)
{
    ckt::ValidationReport report;

    {
        const std::vector<size_t> lengths = {'R', 'O', 'C', 'm'};
        const auto strides                = make_packed_strides_row_major(lengths);
        ckt::TensorDescriptor<ckb::DataType::BF16> desc(lengths, strides);
        auto a = ckt::alloc_tensor_buffer(desc);
        auto b = ckt::alloc_tensor_buffer(desc);

        fill_tensor_buffer(desc, a.get(), [](auto& value, size_t i) {
            value = ck::type_convert<ck::bhalf_t>(i % 100);
        });
        fill_tensor_buffer(desc, a.get(), [](auto& value, size_t i) {
            value = ck::type_convert<ck::bhalf_t>(i % 101);
        });

        report.check("incorrect 1", desc, b.get(), a.get());
    }

    {
        const std::vector<size_t> lengths = {'H', 'I', 'P'};
        const auto strides                = make_packed_strides_row_major(lengths);
        ckt::TensorDescriptor<ckb::DataType::U8> desc(lengths, strides);
        auto a = ckt::alloc_tensor_buffer(desc);
        auto b = ckt::alloc_tensor_buffer(desc);

        fill_tensor_buffer(desc, a.get(), [](auto& value, size_t i) { value = "ROCm"[i % 4]; });
        fill_tensor_buffer(desc, b.get(), [](auto& value, size_t i) {
            switch(i % 4)
            {
            case 0: value = 'R'; break;
            case 1: value = 'O'; break;
            case 2: value = 'C'; break;
            case 3: value = 'm'; break;
            default: value = 'x'; break;
            }
        });

        report.check("correct", desc, b.get(), a.get());
    }

    {
        const std::vector<size_t> lengths = {'G', 'P', 'U'};
        const auto strides                = make_packed_strides_row_major(lengths);
        ckt::TensorDescriptor<ckb::DataType::INT32> desc(lengths, strides);
        auto a = ckt::alloc_tensor_buffer(desc);
        auto b = ckt::alloc_tensor_buffer(desc);

        fill_tensor_buffer(
            desc, a.get(), [](auto& value, [[maybe_unused]] size_t i) { value = 1; });
        fill_tensor_buffer(
            desc, a.get(), [](auto& value, [[maybe_unused]] size_t i) { value = 555; });

        report.check("incorrect 2", desc, b.get(), a.get());
    }

    const auto errors = report.get_errors();

    EXPECT_THAT(errors.size(), Eq(2));
    EXPECT_THAT(errors[0].tensor_name, StrEq("incorrect 1"));
    EXPECT_THAT(errors[0].wrong_elements, Eq(46840429));
    EXPECT_THAT(errors[1].tensor_name, StrEq("incorrect 2"));
    EXPECT_THAT(errors[1].wrong_elements, Eq(482800));
}
