// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <gtest/gtest.h>
#include <type_traits>

// Include the helper file we're testing
#include "ck_tile/builder/factory/helpers/conv_tensor_layout.hpp"

namespace {

namespace ckb = ::ck_tile::builder;
using ::ck_tile::builder::factory::internal::ConvTensorLayouts;
using ::ck_tile::builder::factory::internal::GetTensorLayout;
using ::ck_tile::builder::TensorLayout;
using enum ::ck_tile::builder::ConvDirection;

TEST(ConvTensorLayout, AssignsLayoutsFor1D_NWGC_GKXC_NWGK)
{
    using TensorLayouts =
        ConvTensorLayouts<TensorLayout::NWGC, TensorLayout::GKXC, TensorLayout::NWGK, 1, FORWARD>;

    EXPECT_TRUE((std::is_same_v<TensorLayouts::ALayout, ck::tensor_layout::convolution::NWGC>));
    EXPECT_TRUE((std::is_same_v<TensorLayouts::BLayout, ck::tensor_layout::convolution::GKXC>));
    EXPECT_TRUE((std::is_same_v<TensorLayouts::ELayout, ck::tensor_layout::convolution::NWGK>));
}

TEST(ConvTensorLayout, AssignsLayoutsFor1D_NGCW_GKXC_NGKW)
{
    using TensorLayouts =
        ConvTensorLayouts<TensorLayout::NGCW, TensorLayout::GKXC, TensorLayout::NGKW, 1, FORWARD>;

    EXPECT_TRUE((std::is_same_v<TensorLayouts::ALayout, ck::tensor_layout::convolution::NGCW>));
    EXPECT_TRUE((std::is_same_v<TensorLayouts::BLayout, ck::tensor_layout::convolution::GKXC>));
    EXPECT_TRUE((std::is_same_v<TensorLayouts::ELayout, ck::tensor_layout::convolution::NGKW>));
}

TEST(ConvTensorLayout, AssignsLayoutsFor1D_GNWC_GKXC_GNWK)
{
    using TensorLayouts =
        ConvTensorLayouts<TensorLayout::GNWC, TensorLayout::GKXC, TensorLayout::GNWK, 1, FORWARD>;

    EXPECT_TRUE((std::is_same_v<TensorLayouts::ALayout, ck::tensor_layout::convolution::GNWC>));
    EXPECT_TRUE((std::is_same_v<TensorLayouts::BLayout, ck::tensor_layout::convolution::GKXC>));
    EXPECT_TRUE((std::is_same_v<TensorLayouts::ELayout, ck::tensor_layout::convolution::GNWK>));
}

TEST(ConvTensorLayout, AssignsLayoutsFor1D_NGCW_GKCX_NGKW)
{
    using TensorLayouts =
        ConvTensorLayouts<TensorLayout::NGCW, TensorLayout::GKCX, TensorLayout::NGKW, 1, FORWARD>;

    EXPECT_TRUE((std::is_same_v<TensorLayouts::ALayout, ck::tensor_layout::convolution::NGCW>));
    EXPECT_TRUE((std::is_same_v<TensorLayouts::BLayout, ck::tensor_layout::convolution::GKCX>));
    EXPECT_TRUE((std::is_same_v<TensorLayouts::ELayout, ck::tensor_layout::convolution::NGKW>));
}

TEST(ConvTensorLayout, AssignsLayoutsFor2D_NGCHW_GKYXC_NGKHW)
{
    using TensorLayouts = ConvTensorLayouts<TensorLayout::NGCHW,
                                            TensorLayout::GKYXC,
                                            TensorLayout::NGKHW,
                                            2,
                                            FORWARD>;

    EXPECT_TRUE((std::is_same_v<TensorLayouts::ALayout, ck::tensor_layout::convolution::NGCHW>));
    EXPECT_TRUE((std::is_same_v<TensorLayouts::BLayout, ck::tensor_layout::convolution::GKYXC>));
    EXPECT_TRUE((std::is_same_v<TensorLayouts::ELayout, ck::tensor_layout::convolution::NGKHW>));
}

TEST(ConvTensorLayout, AssignsLayoutsFor2D_NHWGC_GKYXC_NHWGK)
{
    using TensorLayouts = ConvTensorLayouts<TensorLayout::NHWGC,
                                            TensorLayout::GKYXC,
                                            TensorLayout::NHWGK,
                                            2,
                                            FORWARD>;

    EXPECT_TRUE((std::is_same_v<TensorLayouts::ALayout, ck::tensor_layout::convolution::NHWGC>));
    EXPECT_TRUE((std::is_same_v<TensorLayouts::BLayout, ck::tensor_layout::convolution::GKYXC>));
    EXPECT_TRUE((std::is_same_v<TensorLayouts::ELayout, ck::tensor_layout::convolution::NHWGK>));
}

TEST(ConvTensorLayout, AssignsLayoutsFor2D_GNHWC_GKYXC_GNHWK)
{
    using TensorLayouts = ConvTensorLayouts<TensorLayout::GNHWC,
                                            TensorLayout::GKYXC,
                                            TensorLayout::GNHWK,
                                            2,
                                            FORWARD>;

    EXPECT_TRUE((std::is_same_v<TensorLayouts::ALayout, ck::tensor_layout::convolution::GNHWC>));
    EXPECT_TRUE((std::is_same_v<TensorLayouts::BLayout, ck::tensor_layout::convolution::GKYXC>));
    EXPECT_TRUE((std::is_same_v<TensorLayouts::ELayout, ck::tensor_layout::convolution::GNHWK>));
}

TEST(ConvTensorLayout, AssignsLayoutsFor2D_NGCHW_GKCYX_NGKHW)
{
    using TensorLayouts = ConvTensorLayouts<TensorLayout::NGCHW,
                                            TensorLayout::GKCYX,
                                            TensorLayout::NGKHW,
                                            2,
                                            FORWARD>;

    EXPECT_TRUE((std::is_same_v<TensorLayouts::ALayout, ck::tensor_layout::convolution::NGCHW>));
    EXPECT_TRUE((std::is_same_v<TensorLayouts::BLayout, ck::tensor_layout::convolution::GKCYX>));
    EXPECT_TRUE((std::is_same_v<TensorLayouts::ELayout, ck::tensor_layout::convolution::NGKHW>));
}

TEST(ConvTensorLayout, AssignsLayoutsFor3D_NGCDHW_GKCZYX_NGKDHW)
{
    using TensorLayouts = ConvTensorLayouts<TensorLayout::NGCDHW,
                                            TensorLayout::GKCZYX,
                                            TensorLayout::NGKDHW,
                                            3,
                                            FORWARD>;

    EXPECT_TRUE((std::is_same_v<TensorLayouts::ALayout, ck::tensor_layout::convolution::NGCDHW>));
    EXPECT_TRUE((std::is_same_v<TensorLayouts::BLayout, ck::tensor_layout::convolution::GKCZYX>));
    EXPECT_TRUE((std::is_same_v<TensorLayouts::ELayout, ck::tensor_layout::convolution::NGKDHW>));
}

TEST(ConvTensorLayout, AssignsLayoutsFor3D_NDHWGC_GKZYXC_NDHWGK)
{
    using TensorLayouts = ConvTensorLayouts<TensorLayout::NDHWGC,
                                            TensorLayout::GKZYXC,
                                            TensorLayout::NDHWGK,
                                            3,
                                            FORWARD>;

    EXPECT_TRUE((std::is_same_v<TensorLayouts::ALayout, ck::tensor_layout::convolution::NDHWGC>));
    EXPECT_TRUE((std::is_same_v<TensorLayouts::BLayout, ck::tensor_layout::convolution::GKZYXC>));
    EXPECT_TRUE((std::is_same_v<TensorLayouts::ELayout, ck::tensor_layout::convolution::NDHWGK>));
}

TEST(ConvTensorLayout, AssignsLayoutsFor3D_GNDHWC_GKZYXC_GNDHWK)
{
    using TensorLayouts = ConvTensorLayouts<TensorLayout::GNDHWC,
                                            TensorLayout::GKZYXC,
                                            TensorLayout::GNDHWK,
                                            3,
                                            FORWARD>;

    EXPECT_TRUE((std::is_same_v<TensorLayouts::ALayout, ck::tensor_layout::convolution::GNDHWC>));
    EXPECT_TRUE((std::is_same_v<TensorLayouts::BLayout, ck::tensor_layout::convolution::GKZYXC>));
    EXPECT_TRUE((std::is_same_v<TensorLayouts::ELayout, ck::tensor_layout::convolution::GNDHWK>));
}

} // namespace
