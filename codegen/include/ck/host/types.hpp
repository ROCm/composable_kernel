// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2024, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include <string>
#include <sstream>
#include <utility>
#include <unordered_map>
#include <vector>

namespace ck {
namespace host {

// holds the templated instance, substitues values into template from instancess
struct Solution
{

    Solution() = default;
    Solution(std::string str, std::unordered_map<std::string, std::string> values);
    std::string ToTemplateString() const;
    std::string GetTemplateParameter(const std::string& name) const;
    template <class T>
    T GetTemplateParameter(const std::string& name) const
    {
        T result;
        std::stringstream ss(GetTemplateParameter(name));
        ss >> result;
        return result;
    }

    private:
    std::string template_str;
    std::unordered_map<std::string, std::string> template_values;
};

// supported data types
enum class DataType
{
    Half,
    Float,
    Int8,
    Int32
};
std::string ToString(DataType dt);

// supported layouts: gemm and fwd conv
enum class Layout
{
    Row,
    Column,
    GKYXC,
    GKCYX,
    GNHWK,
    GNHWC,
    NHWGC,
    NHWGK
};
std::string ToString(Layout dl);
Layout ToLayout(bool Trans); // returns the layout for gemm

// supported GEMM types
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
