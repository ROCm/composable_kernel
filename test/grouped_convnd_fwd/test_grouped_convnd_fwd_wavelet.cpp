// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <cstdlib>
#include <iostream>
#include <initializer_list>
#include <vector>
#include <gtest/gtest.h>

#include "ck/utility/common_header.hpp"
#include "ck/host_utility/device_prop.hpp"
#include "profiler/profile_grouped_conv_fwd_impl.hpp"

static ck::index_t param_mask     = 0xffff;
static ck::index_t instance_index = -1;
using F16                         = ck::half_t;

template <typename Tuple>
class TestGroupedConvndFwdWavelet : public ::testing::Test
{
    protected:
    using InDataType   = std::tuple_element_t<0, Tuple>;
    using WeiDataType  = std::tuple_element_t<1, Tuple>;
    using OutDataType  = std::tuple_element_t<2, Tuple>;
    using AComputeType = std::tuple_element_t<3, Tuple>;
    using BComputeType = std::tuple_element_t<4, Tuple>;
    using InLayout     = std::tuple_element_t<5, Tuple>;
    using WeiLayout    = std::tuple_element_t<6, Tuple>;
    using OutLayout    = std::tuple_element_t<7, Tuple>;
    using IndexType    = ck::index_t;

    std::vector<ck::utils::conv::ConvParam> conv_params;
#if defined(CK_TEST_DISABLE_GPU_VALIDATION)
    static constexpr int verify_ = 1; // CPU reference
#else
    static constexpr int verify_ = 2; // GPU reference
#endif
    template <ck::index_t NDimSpatial>
    void Run()
    {
        EXPECT_FALSE(conv_params.empty());
        bool pass = true;
        for(size_t i = 0; i < conv_params.size(); i++)
        {
            if((param_mask & (1 << i)) == 0)
            {
                continue;
            }
            auto& param = conv_params[i];
            pass        = pass && ck::profiler::profile_grouped_conv_fwd_impl<NDimSpatial,
                                                                              InLayout,
                                                                              WeiLayout,
                                                                              OutLayout,
                                                                              InDataType,
                                                                              WeiDataType,
                                                                              OutDataType,
                                                                              AComputeType,
                                                                              BComputeType,
                                                                              IndexType>(
                               verify_, // do_verification
                               1,       // init_method: integer value
                               false,   // do_log
                               false,   // time_kernel
                               param,
                               ck::tensor_operation::element_wise::PassThrough{},
                               instance_index);
        }
        EXPECT_TRUE(pass);
    }
};

using namespace ck::tensor_layout::convolution;

// Wavelet XDL conv3d fwd instances use NDHWGC layout (NSpatialGC),
// which maps to Wave32Force16MNPerXDL=true on gfx11, so they are valid on gfx9 and gfx11.
using KernelTypes3d = ::testing::Types<std::tuple<F16, F16, F16, F16, F16, NDHWGC, GKZYXC, NDHWGK>>;

template <typename Tuple>
class TestGroupedConvndFwdWavelet3d : public TestGroupedConvndFwdWavelet<Tuple>
{
};

TYPED_TEST_SUITE(TestGroupedConvndFwdWavelet3d, KernelTypes3d);

TYPED_TEST(TestGroupedConvndFwdWavelet3d, Test3D)
{
    this->conv_params.clear();

    // Standard 3x3x3 conv
    this->conv_params.push_back(
        {3, 1, 4, 64, 64, {3, 3, 3}, {8, 8, 8}, {1, 1, 1}, {1, 1, 1}, {1, 1, 1}, {1, 1, 1}});

    // Strided conv (matching the example's stride=2 pattern in 3D)
    this->conv_params.push_back(
        {3, 1, 4, 64, 64, {3, 3, 3}, {16, 16, 16}, {2, 2, 2}, {1, 1, 1}, {1, 1, 1}, {1, 1, 1}});

    // Larger problem resembling the example (N=128, K=256, C=192 scaled down for test speed)
    this->conv_params.push_back(
        {3, 1, 8, 128, 64, {3, 3, 3}, {12, 12, 12}, {2, 2, 2}, {1, 1, 1}, {1, 1, 1}, {1, 1, 1}});

    // 1x1x1 filter, stride 1
    this->conv_params.push_back(
        {3, 1, 4, 64, 64, {1, 1, 1}, {8, 8, 8}, {1, 1, 1}, {1, 1, 1}, {0, 0, 0}, {0, 0, 0}});

    this->template Run<3>();
}
