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

// input tensor
// packed NWC/NHWC/NDHWC
struct NWC : public BaseTensorLayout
{
    static constexpr const char* name = "NWC";
};

struct NHWC : public BaseTensorLayout
{
    static constexpr const char* name = "NHWC";
};

struct NDHWC : public BaseTensorLayout
{
    static constexpr const char* name = "NDHWC";
};

// input tensor
// packed NCW/NCHW/NCDHW
struct NCW : public BaseTensorLayout
{
    static constexpr const char* name = "NCW";
};

struct NCHW : public BaseTensorLayout
{
    static constexpr const char* name = "NCHW";
};

struct NCDHW : public BaseTensorLayout
{
    static constexpr const char* name = "NCDHW";
};

// weight tensor
// packed KXC/KYXC/KZYXC
struct KXC : public BaseTensorLayout
{
    static constexpr const char* name = "KXC";
};

struct KYXC : public BaseTensorLayout
{
    static constexpr const char* name = "KYXC";
};

struct KZYXC : public BaseTensorLayout
{
    static constexpr const char* name = "KZYXC";
};

// weight tensor
// packed KCX/KCYX/KCZYX
struct KCX : public BaseTensorLayout
{
    static constexpr const char* name = "KCX";
};

struct KCYX : public BaseTensorLayout
{
    static constexpr const char* name = "KCYX";
};

struct KCZYX : public BaseTensorLayout
{
    static constexpr const char* name = "KCZYX";
};

// output tensor
// packed NWK/NHWK/NDHWK
struct NWK : public BaseTensorLayout
{
    static constexpr const char* name = "NWK";
};

struct NHWK : public BaseTensorLayout
{
    static constexpr const char* name = "NHWK";
};

struct NDHWK : public BaseTensorLayout
{
    static constexpr const char* name = "NDHWK";
};

// output tensor
// packed NKW/NKHW/NKDHW
struct NKW : public BaseTensorLayout
{
    static constexpr const char* name = "NKW";
};

struct NKHW : public BaseTensorLayout
{
    static constexpr const char* name = "NKHW";
};

struct NKDHW : public BaseTensorLayout
{
    static constexpr const char* name = "NKDHW";
};

// output tensor
// strided layout
struct NW_K : public BaseTensorLayout
{
    static constexpr const char* name = "NW_K";
};

struct NHW_K : public BaseTensorLayout
{
    static constexpr const char* name = "NHW_K";
};

struct NDHW_K : public BaseTensorLayout
{
    static constexpr const char* name = "NDHW_K";
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
