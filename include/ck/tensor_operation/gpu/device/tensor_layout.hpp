// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2022, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

namespace ck {
namespace tensor_layout {

struct BaseTensorLayout
{
};

namespace gemm {

struct RowMajor : public BaseTensorLayout
{
    static constexpr const char* name = "RowMajor";
};

struct ColumnMajor : public BaseTensorLayout
{
    static constexpr const char* name = "ColumnMajor";
};
} // namespace gemm

namespace convolution {

// 1D Conv
struct NWC : public BaseTensorLayout
{
    static constexpr const char* name = "NWC";
};

struct KXC : public BaseTensorLayout
{
    static constexpr const char* name = "KXC";
};

struct NWK : public BaseTensorLayout
{
    static constexpr const char* name = "NWK";
};

struct NCW : public BaseTensorLayout
{
    static constexpr const char* name = "NCW";
};

struct KCX : public BaseTensorLayout
{
    static constexpr const char* name = "KCX";
};

struct NKW : public BaseTensorLayout
{
    static constexpr const char* name = "NKW";
};

// 2D Conv
struct NHWC : public BaseTensorLayout
{
    static constexpr const char* name = "NHWC";
};

struct KYXC : public BaseTensorLayout
{
    static constexpr const char* name = "KYXC";
};

struct NHWK : public BaseTensorLayout
{
    static constexpr const char* name = "NHWK";
};

struct NCHW : public BaseTensorLayout
{
    static constexpr const char* name = "NCHW";
};

struct KCYX : public BaseTensorLayout
{
    static constexpr const char* name = "KCYX";
};

struct NKHW : public BaseTensorLayout
{
    static constexpr const char* name = "NKHW";
};

// 3D Conv
struct NDHWC : public BaseTensorLayout
{
    static constexpr const char* name = "NDHWC";
};

struct KZYXC : public BaseTensorLayout
{
    static constexpr const char* name = "KZYXC";
};

struct NDHWK : public BaseTensorLayout
{
    static constexpr const char* name = "NDHWK";
};
struct NCDHW : public BaseTensorLayout
{
    static constexpr const char* name = "NCDHW";
};

struct KCZYX : public BaseTensorLayout
{
    static constexpr const char* name = "KCZYX";
};

struct NKDHW : public BaseTensorLayout
{
    static constexpr const char* name = "NKDHW";
};

} // namespace convolution

template <
    typename Layout,
    typename std::enable_if<std::is_base_of<BaseTensorLayout, Layout>::value, bool>::type = false>
std::ostream& operator<<(std::ostream& os, const Layout&)
{
    os << Layout::name;
    return os;
}

} // namespace tensor_layout
} // namespace ck
