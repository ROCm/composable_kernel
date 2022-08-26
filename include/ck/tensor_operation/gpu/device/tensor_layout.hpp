// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2022, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#ifndef PP_DEFINE_LAYOUT_TYPE
#define PP_DEFINE_LAYOUT_TYPE(layout)                         \
    struct layout final : ck::tensor_layout::BaseTensorLayout \
    {                                                         \
        static constexpr const char* name = #layout;          \
    }
#endif

namespace ck {
namespace tensor_layout {

struct BaseTensorLayout
{
};

namespace gemm {

PP_DEFINE_LAYOUT_TYPE(RowMajor);
PP_DEFINE_LAYOUT_TYPE(ColumnMajor);

} // namespace gemm

namespace convolution {

// input tensor
// packed NCW/NCHW/NCDHW
PP_DEFINE_LAYOUT_TYPE(NCW);
PP_DEFINE_LAYOUT_TYPE(NCHW);
PP_DEFINE_LAYOUT_TYPE(NCDHW);

// packed GNCW/GNCHW/GNCDHW
PP_DEFINE_LAYOUT_TYPE(GNCW);
PP_DEFINE_LAYOUT_TYPE(GNCHW);
PP_DEFINE_LAYOUT_TYPE(GNCDHW);

// input tensor
// packed NWC/NHWC/NDHWC
PP_DEFINE_LAYOUT_TYPE(NWC);
PP_DEFINE_LAYOUT_TYPE(NHWC);
PP_DEFINE_LAYOUT_TYPE(NDHWC);

// input tensor
// packed GNWC/GNHWC/GNDHWC
PP_DEFINE_LAYOUT_TYPE(GNWC);
PP_DEFINE_LAYOUT_TYPE(GNHWC);
PP_DEFINE_LAYOUT_TYPE(GNDHWC);

// input tensor
// packed NWGC/NHWGC/NDHWGC
PP_DEFINE_LAYOUT_TYPE(NWGC);
PP_DEFINE_LAYOUT_TYPE(NHWGC);
PP_DEFINE_LAYOUT_TYPE(NDHWGC);

// input tensor
// strided layout
PP_DEFINE_LAYOUT_TYPE(G_NW_C);
PP_DEFINE_LAYOUT_TYPE(G_NHW_C);
PP_DEFINE_LAYOUT_TYPE(G_NDHW_C);

// weight tensor
// packed KCX/KCYX/KCZYX
PP_DEFINE_LAYOUT_TYPE(KCX);
PP_DEFINE_LAYOUT_TYPE(KCYX);
PP_DEFINE_LAYOUT_TYPE(KCZYX);

// weight tensor
// packed GKCX/GKCYX/GKCZYX
PP_DEFINE_LAYOUT_TYPE(GKCX);
PP_DEFINE_LAYOUT_TYPE(GKCYX);
PP_DEFINE_LAYOUT_TYPE(GKCZYX);

// weight tensor
// packed KXC/KYXC/KZYXC
PP_DEFINE_LAYOUT_TYPE(KXC);
PP_DEFINE_LAYOUT_TYPE(KYXC);
PP_DEFINE_LAYOUT_TYPE(KZYXC);

// weight tensor
// packed GKXC/GKYXC/GKZYXC
PP_DEFINE_LAYOUT_TYPE(GKXC);
PP_DEFINE_LAYOUT_TYPE(GKYXC);
PP_DEFINE_LAYOUT_TYPE(GKZYXC);

// weight tensor
// packed KXGC/KYXGC/KZYXGC
PP_DEFINE_LAYOUT_TYPE(KXGC);
PP_DEFINE_LAYOUT_TYPE(KYXGC);
PP_DEFINE_LAYOUT_TYPE(KZYXGC);

// weight tensor
// strided
PP_DEFINE_LAYOUT_TYPE(G_K_X_C);
PP_DEFINE_LAYOUT_TYPE(G_K_YX_C);
PP_DEFINE_LAYOUT_TYPE(G_K_ZYX_C);

// output tensor
// packed NKW/NKHW/NKDHW
PP_DEFINE_LAYOUT_TYPE(NKW);
PP_DEFINE_LAYOUT_TYPE(NKHW);
PP_DEFINE_LAYOUT_TYPE(NKDHW);

// output tensor
// packed GNKW/GNKHW/GNKDHW
PP_DEFINE_LAYOUT_TYPE(GNKW);
PP_DEFINE_LAYOUT_TYPE(GNKHW);
PP_DEFINE_LAYOUT_TYPE(GNKDHW);

// output tensor
// packed NWK/NHWK/NDHWK
PP_DEFINE_LAYOUT_TYPE(NWK);
PP_DEFINE_LAYOUT_TYPE(NHWK);
PP_DEFINE_LAYOUT_TYPE(NDHWK);

// output tensor
// packed GNWK/GNHWK/GNDHWK
PP_DEFINE_LAYOUT_TYPE(GNWK);
PP_DEFINE_LAYOUT_TYPE(GNHWK);
PP_DEFINE_LAYOUT_TYPE(GNDHWK);

// output tensor
// packed NWGK/NHWGK/NDHWGK
PP_DEFINE_LAYOUT_TYPE(NWGK);
PP_DEFINE_LAYOUT_TYPE(NHWGK);
PP_DEFINE_LAYOUT_TYPE(NDHWGK);

// output tensor
// strided layout
PP_DEFINE_LAYOUT_TYPE(G_NW_K);
PP_DEFINE_LAYOUT_TYPE(G_NHW_K);
PP_DEFINE_LAYOUT_TYPE(G_NDHW_K);

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
