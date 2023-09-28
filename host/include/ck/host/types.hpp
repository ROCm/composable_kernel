// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2023, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include <string>
#include <utility>
#include <unordered_map>
#include <vector>

namespace ck {
namespace host {

struct Solution
{

    Solution() = default;
    Solution(std::string str, std::unordered_map<std::string, std::string> values);
    std::string ToTemplateString() const;
    std::string ToTemplateParameter(const std::string& name) const;

    private:
    std::string template_str;
    std::unordered_map<std::string, std::string> template_values;
};

enum class DataType
{
    Half,
    Float,
    Int8,
    Int32
};

std::string ToString(DataType dt);

enum class Layout
{
    Row,
    Column
};

std::string ToString(Layout dl);

enum class GemmType
{
    Default
};

std::string ToString(GemmType gt);

struct TensorDesc
{
    DataType element;
    Layout layout;
};

std::string SequenceStr(const std::vector<int>& v);

std::string MakeTuple(const std::vector<std::string>& v);

template <int... xs>
const std::string S = SequenceStr({xs...});

constexpr const char* PassThrough = "ck::tensor_operation::element_wise::PassThrough";
constexpr const char* Bilinear    = "ck::tensor_operation::element_wise::Bilinear";

} // namespace host
} // namespace ck
