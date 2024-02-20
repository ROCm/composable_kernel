// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2023, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include <string>
#include <sstream>
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
Layout ToLayout(bool Trans);

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

enum struct GemmSpecialization
{
    Default,
    MPadding,
    NPadding,
    KPadding,
    MNPadding,
    MKPadding,
    NKPadding,
    MNKPadding,
    OPadding,
    MOPadding,
    NOPadding,
    KOPadding,
    MNOPadding,
    MKOPadding,
    NKOPadding,
    MNKOPadding,
};

inline std::string getGemmSpecializationString(const GemmSpecialization& s)
{
    switch(s)
    {
    case GemmSpecialization::Default: return "Default";
    case GemmSpecialization::MPadding: return "MPadding";
    case GemmSpecialization::NPadding: return "NPadding";
    case GemmSpecialization::KPadding: return "KPadding";
    case GemmSpecialization::MNPadding: return "MNPadding";
    case GemmSpecialization::MKPadding: return "MKPadding";
    case GemmSpecialization::NKPadding: return "NKPadding";
    case GemmSpecialization::MNKPadding: return "MNKPadding";
    case GemmSpecialization::OPadding: return "OPadding";
    case GemmSpecialization::MOPadding: return "MOPadding";
    case GemmSpecialization::NOPadding: return "NOPadding";
    case GemmSpecialization::KOPadding: return "KOPadding";
    case GemmSpecialization::MNOPadding: return "MNOPadding";
    case GemmSpecialization::MKOPadding: return "MKOPadding";
    case GemmSpecialization::NKOPadding: return "NKOPadding";
    case GemmSpecialization::MNKOPadding: return "MNKOPadding";
    default: return "Unrecognized specialization!";
    }
}

std::string SequenceStr(const std::vector<int>& v);

std::string MakeTuple(const std::vector<std::string>& v);

template <int... xs>
const std::string S = SequenceStr({xs...});

constexpr const char* PassThrough = "ck::tensor_operation::element_wise::PassThrough";
constexpr const char* Bilinear    = "ck::tensor_operation::element_wise::Bilinear";

} // namespace host
} // namespace ck
