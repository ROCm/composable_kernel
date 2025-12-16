// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "ck_tile/builder/testing/error.hpp"
#include "ck_tile/builder/testing/tensor_buffer.hpp"
#include "ck_tile/builder/testing/tensor_descriptor.hpp"
#include "ck_tile/builder/testing/validation.hpp"
#include "ck_tile/builder/testing/tensor_foreach.hpp"
#include "ck_tile/builder/factory/helpers/ck/conv_tensor_type.hpp"
#include "testing_utils.hpp"
#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include <span>
#include <array>

namespace ckb = ck_tile::builder;
namespace ckt = ck_tile::builder::test;

using testing::ElementsAreArray;
using testing::Eq;
using testing::StrEq;

template <ckb::DataType DT, size_t RANK, typename F>
void fill_tensor(const ckt::TensorDescriptor<DT, RANK>& desc, void* buffer, F f)
{
    const auto strides = desc.get_strides();
    ckt::tensor_foreach(desc.get_lengths(), [buffer, f, strides](auto index) {
        using CKType      = typename ckb::factory::internal::DataTypeToCK<DT>::type;
        auto* ptr         = static_cast<CKType*>(buffer);
        const auto offset = ckt::calculate_offset(index, strides);

        ptr[offset] = f(index);
    });
}

template <ckb::DataType DT, size_t RANK, typename F>
void fill_tensor_buffer(const ckt::TensorDescriptor<DT, RANK>& desc, void* buffer, F f)
{
    fill_tensor(desc.get_space_descriptor(), buffer, [f](auto index) { return f(index[0]); });
}

TEST(ValidationUtilities, MakePackedStrides)
{
    const ckt::Extent lengths = {5125, 623, 1177, 1534};
    const auto strides        = ckt::PackedRightLayout{}(lengths);

    EXPECT_THAT(strides, ElementsAreArray({623 * 1177 * 1534, 1177 * 1534, 1534, 1}));
}

TEST(ValidationUtilities, FillTensorBuffer)
{
    auto desc = ckt::make_descriptor<ckb::DataType::INT32>(ckt::Extent{31, 54, 13},
                                                           ckt::PackedRightLayout{});

    auto buffer = ckt::alloc_tensor_buffer(desc);

    fill_tensor_buffer(desc, buffer.get(), [](size_t i) { return static_cast<uint32_t>(i); });

    std::vector<uint32_t> h_buffer(desc.get_element_space_size());
    ckt::check_hip(hipMemcpy(
        h_buffer.data(), buffer.get(), h_buffer.size() * sizeof(uint32_t), hipMemcpyDeviceToHost));

    for(size_t i = 0; i < h_buffer.size(); ++i)
    {
        EXPECT_THAT(h_buffer[i], Eq(static_cast<uint32_t>(i)));
    }
}

TEST(ValidationReport, SingleCorrect)
{
    auto desc = ckt::make_descriptor<ckb::DataType::FP32>(ckt::Extent{52, 152, 224},
                                                          ckt::PackedRightLayout{});

    auto a = ckt::alloc_tensor_buffer(desc);
    auto b = ckt::alloc_tensor_buffer(desc);

    // Generate a sort-of-random looking sequence
    auto generator = [](size_t i) { return static_cast<float>(i * 10'000'019 % 768'351); };

    fill_tensor_buffer(desc, a.get(), generator);
    fill_tensor_buffer(desc, b.get(), generator);

    ckt::ValidationReport report;
    report.check("correct", desc, b.get(), a.get());

    EXPECT_THAT(report.get_errors().size(), Eq(0));
}

TEST(ValidationReport, SingleIncorrect)
{
    auto desc = ckt::make_descriptor<ckb::DataType::FP16>(ckt::Extent{100, 100, 100},
                                                          ckt::PackedRightLayout{});

    auto a = ckt::alloc_tensor_buffer(desc);
    auto b = ckt::alloc_tensor_buffer(desc);

    fill_tensor_buffer(desc, a.get(), []([[maybe_unused]] size_t i) { return 123; });
    fill_tensor_buffer(desc, b.get(), []([[maybe_unused]] size_t i) {
        return i == 12345 ? 456 : i == 999999 ? 1 : 123;
    });

    ckt::ValidationReport report;
    report.check("incorrect", desc, b.get(), a.get());

    const auto errors = report.get_errors();

    ASSERT_THAT(errors.size(), Eq(1));
    EXPECT_THAT(errors[0].tensor_name, StrEq("incorrect"));
    EXPECT_THAT(errors[0].wrong_elements, Eq(2));
    EXPECT_THAT(errors[0].total_elements, Eq(desc.get_element_space_size()));
}

TEST(ValidationReport, MultipleSomeIncorrect)
{
    ckt::ValidationReport report;

    {
        auto desc = ckt::make_descriptor<ckb::DataType::BF16, 4>({'R', 'O', 'C', 'm'},
                                                                 ckt::PackedRightLayout{});

        auto a = ckt::alloc_tensor_buffer(desc);
        auto b = ckt::alloc_tensor_buffer(desc);

        fill_tensor_buffer(
            desc, a.get(), [](size_t i) { return ck::type_convert<ck::bhalf_t>(i % 100); });
        fill_tensor_buffer(
            desc, b.get(), [](size_t i) { return ck::type_convert<ck::bhalf_t>(i % 101); });

        report.check("incorrect 1", desc, b.get(), a.get());
    }

    {
        auto desc =
            ckt::make_descriptor<ckb::DataType::U8, 3>({'H', 'I', 'P'}, ckt::PackedRightLayout{});

        auto a = ckt::alloc_tensor_buffer(desc);
        auto b = ckt::alloc_tensor_buffer(desc);

        fill_tensor_buffer(desc, a.get(), [](size_t i) { return "ROCm"[i % 4]; });
        fill_tensor_buffer(desc, b.get(), [](size_t i) {
            switch(i % 4)
            {
            case 0: return 'R';
            case 1: return 'O';
            case 2: return 'C';
            case 3: return 'm';
            default: return 'x';
            }
        });

        report.check("correct", desc, b.get(), a.get());
    }

    {
        auto desc = ckt::make_descriptor<ckb::DataType::INT32, 3>({'G', 'P', 'U'},
                                                                  ckt::PackedRightLayout{});

        auto a = ckt::alloc_tensor_buffer(desc);
        auto b = ckt::alloc_tensor_buffer(desc);

        fill_tensor_buffer(desc, a.get(), []([[maybe_unused]] size_t i) { return 1; });
        fill_tensor_buffer(desc, b.get(), []([[maybe_unused]] size_t i) { return 555; });

        report.check("incorrect 2", desc, b.get(), a.get());
    }

    const auto errors = report.get_errors();

    ASSERT_THAT(errors.size(), Eq(2));
    EXPECT_THAT(errors[0].tensor_name, StrEq("incorrect 1"));
    EXPECT_THAT(errors[0].wrong_elements, Eq(46840334));
    EXPECT_THAT(errors[1].tensor_name, StrEq("incorrect 2"));
    EXPECT_THAT(errors[1].wrong_elements, Eq(482800));
}
