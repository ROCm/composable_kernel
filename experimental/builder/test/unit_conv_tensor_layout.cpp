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
using enum ::ck_tile::builder::ConvDirection;

TEST(ConvTensorLayout, AssignsLayoutsFor1D_NWGC_GKXC_NWGK)
{
    using TensorLayouts = ConvTensorLayouts<ckb::GroupConvLayout1D::NWGC_GKXC_NWGK, 1, FORWARD>;

    EXPECT_TRUE((std::is_same_v<TensorLayouts::ALayout, ck::tensor_layout::convolution::NWGC>));
    EXPECT_TRUE((std::is_same_v<TensorLayouts::BLayout, ck::tensor_layout::convolution::GKXC>));
    EXPECT_TRUE((std::is_same_v<TensorLayouts::ELayout, ck::tensor_layout::convolution::NWGK>));
}

TEST(ConvTensorLayout, AssignsLayoutsFor1D_NGCW_GKXC_NGKW)
{
    using TensorLayouts = ConvTensorLayouts<ckb::GroupConvLayout1D::NGCW_GKXC_NGKW, 1, FORWARD>;

    EXPECT_TRUE((std::is_same_v<TensorLayouts::ALayout, ck::tensor_layout::convolution::NGCW>));
    EXPECT_TRUE((std::is_same_v<TensorLayouts::BLayout, ck::tensor_layout::convolution::GKXC>));
    EXPECT_TRUE((std::is_same_v<TensorLayouts::ELayout, ck::tensor_layout::convolution::NGKW>));
}

TEST(ConvTensorLayout, AssignsLayoutsFor1D_GNWC_GKXC_GNWK)
{
    using TensorLayouts = ConvTensorLayouts<ckb::GroupConvLayout1D::GNWC_GKXC_GNWK, 1, FORWARD>;

    EXPECT_TRUE((std::is_same_v<TensorLayouts::ALayout, ck::tensor_layout::convolution::GNWC>));
    EXPECT_TRUE((std::is_same_v<TensorLayouts::BLayout, ck::tensor_layout::convolution::GKXC>));
    EXPECT_TRUE((std::is_same_v<TensorLayouts::ELayout, ck::tensor_layout::convolution::GNWK>));
}

TEST(ConvTensorLayout, AssignsLayoutsFor1D_NGCW_GKCX_NGKW)
{
    using TensorLayouts = ConvTensorLayouts<ckb::GroupConvLayout1D::NGCW_GKCX_NGKW, 1, FORWARD>;

    EXPECT_TRUE((std::is_same_v<TensorLayouts::ALayout, ck::tensor_layout::convolution::NGCW>));
    EXPECT_TRUE((std::is_same_v<TensorLayouts::BLayout, ck::tensor_layout::convolution::GKCX>));
    EXPECT_TRUE((std::is_same_v<TensorLayouts::ELayout, ck::tensor_layout::convolution::NGKW>));
}

TEST(ConvTensorLayout, AssignsLayoutsFor2D_NGCHW_GKYXC_NGKHW)
{
    using TensorLayouts = ConvTensorLayouts<ckb::GroupConvLayout2D::NGCHW_GKYXC_NGKHW, 2, FORWARD>;

    EXPECT_TRUE((std::is_same_v<TensorLayouts::ALayout, ck::tensor_layout::convolution::NGCHW>));
    EXPECT_TRUE((std::is_same_v<TensorLayouts::BLayout, ck::tensor_layout::convolution::GKYXC>));
    EXPECT_TRUE((std::is_same_v<TensorLayouts::ELayout, ck::tensor_layout::convolution::NGKHW>));
}

TEST(ConvTensorLayout, AssignsLayoutsFor2D_NHWGC_GKYXC_NHWGK)
{
    using TensorLayouts = ConvTensorLayouts<ckb::GroupConvLayout2D::NHWGC_GKYXC_NHWGK, 2, FORWARD>;

    EXPECT_TRUE((std::is_same_v<TensorLayouts::ALayout, ck::tensor_layout::convolution::NHWGC>));
    EXPECT_TRUE((std::is_same_v<TensorLayouts::BLayout, ck::tensor_layout::convolution::GKYXC>));
    EXPECT_TRUE((std::is_same_v<TensorLayouts::ELayout, ck::tensor_layout::convolution::NHWGK>));
}

TEST(ConvTensorLayout, AssignsLayoutsFor2D_GNHWC_GKYXC_GNHWK)
{
    using TensorLayouts = ConvTensorLayouts<ckb::GroupConvLayout2D::GNHWC_GKYXC_GNHWK, 2, FORWARD>;

    EXPECT_TRUE((std::is_same_v<TensorLayouts::ALayout, ck::tensor_layout::convolution::GNHWC>));
    EXPECT_TRUE((std::is_same_v<TensorLayouts::BLayout, ck::tensor_layout::convolution::GKYXC>));
    EXPECT_TRUE((std::is_same_v<TensorLayouts::ELayout, ck::tensor_layout::convolution::GNHWK>));
}

TEST(ConvTensorLayout, AssignsLayoutsFor2D_NGCHW_GKCYX_NGKHW)
{
    using TensorLayouts = ConvTensorLayouts<ckb::GroupConvLayout2D::NGCHW_GKCYX_NGKHW, 2, FORWARD>;

    EXPECT_TRUE((std::is_same_v<TensorLayouts::ALayout, ck::tensor_layout::convolution::NGCHW>));
    EXPECT_TRUE((std::is_same_v<TensorLayouts::BLayout, ck::tensor_layout::convolution::GKCYX>));
    EXPECT_TRUE((std::is_same_v<TensorLayouts::ELayout, ck::tensor_layout::convolution::NGKHW>));
}

TEST(ConvTensorLayout, AssignsLayoutsFor3D_NGCDHW_GKCZYX_NGKDHW)
{
    using TensorLayouts =
        ConvTensorLayouts<ckb::GroupConvLayout3D::NGCDHW_GKCZYX_NGKDHW, 3, FORWARD>;

    EXPECT_TRUE((std::is_same_v<TensorLayouts::ALayout, ck::tensor_layout::convolution::NGCDHW>));
    EXPECT_TRUE((std::is_same_v<TensorLayouts::BLayout, ck::tensor_layout::convolution::GKCZYX>));
    EXPECT_TRUE((std::is_same_v<TensorLayouts::ELayout, ck::tensor_layout::convolution::NGKDHW>));
}

TEST(ConvTensorLayout, AssignsLayoutsFor3D_NDHWGC_GKZYXC_NDHWGK)
{
    using TensorLayouts =
        ConvTensorLayouts<ckb::GroupConvLayout3D::NDHWGC_GKZYXC_NDHWGK, 3, FORWARD>;

    EXPECT_TRUE((std::is_same_v<TensorLayouts::ALayout, ck::tensor_layout::convolution::NDHWGC>));
    EXPECT_TRUE((std::is_same_v<TensorLayouts::BLayout, ck::tensor_layout::convolution::GKZYXC>));
    EXPECT_TRUE((std::is_same_v<TensorLayouts::ELayout, ck::tensor_layout::convolution::NDHWGK>));
}

TEST(ConvTensorLayout, AssignsLayoutsFor3D_GNDHWC_GKZYXC_GNDHWK)
{
    using TensorLayouts =
        ConvTensorLayouts<ckb::GroupConvLayout3D::GNDHWC_GKZYXC_GNDHWK, 3, FORWARD>;

    EXPECT_TRUE((std::is_same_v<TensorLayouts::ALayout, ck::tensor_layout::convolution::GNDHWC>));
    EXPECT_TRUE((std::is_same_v<TensorLayouts::BLayout, ck::tensor_layout::convolution::GKZYXC>));
    EXPECT_TRUE((std::is_same_v<TensorLayouts::ELayout, ck::tensor_layout::convolution::GNDHWK>));
}

} // namespace
