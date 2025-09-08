// SPDX-License-Identifier: MIT
// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.

#include <gtest/gtest.h>
#include <vector>
#include <tuple>
#include <array>

#include "ck_tile/core.hpp"
#include "ck_tile/ops/grouped_convolution/utils/transform_conv_bwd_weight_to_gemm.hpp"
#include "ck_tile/ops/grouped_convolution/utils/convolution_specialization.hpp"


#ifdef __clang__
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wglobal-constructors"
#endif
using namespace ck_tile;

// Test configuration structure
template <index_t NDimSpatial, index_t NumGroupsToMerge = 1>
struct TestConfig
{
    static constexpr index_t NDim = NDimSpatial;
    static constexpr ConvolutionSpecialization ConvSpec = ConvolutionSpecialization::Default;
    static constexpr bool SplitN = false;
    static constexpr index_t NumberOfGroupsToMerge = NumGroupsToMerge;
    
    using ADataType = float;
    using CDataType = float;
    using IndexType = index_t;
    
    using TransformType = TransformConvBwdWeightToGemm<NDimSpatial,
                                                       ConvSpec,
                                                       NumGroupsToMerge,
                                                       SplitN,
                                                       ADataType,
                                                       CDataType,
                                                       IndexType>;
};

// Test configurations for different dimensions
using TestConfig1D_no_merge = TestConfig<1>;
using TestConfig2D_no_merge = TestConfig<2>;
using TestConfig3D_no_merge = TestConfig<3>;

constexpr index_t GroupsToMerge = 2;
using TestConfig1D_merge = TestConfig<1, GroupsToMerge>;
using TestConfig2D_merge = TestConfig<2, GroupsToMerge>;
using TestConfig3D_merge = TestConfig<3, GroupsToMerge>;

// Test class template
template <typename Config>
class TestTransformConvBwdWeightToGemm : public ::testing::Test
{
protected:
    static constexpr index_t NDim = Config::NDim;
    using TransformType = typename Config::TransformType;
    
    void SetUp() override
    {
        // Common test parameters
        G_ = 16;   // Groups
        N_ = 4;   // Batch size

        // Depthwise convolution
        K_ = 1; // Output channels per group
        C_ = 1;  // Input channels per group
        
        if constexpr (NDim == 1) {
            SetUp1D();
        } else if constexpr (NDim == 2) {
            SetUp2D();
        } else if constexpr (NDim == 3) {
            SetUp3D();
        }
    }
    
    void SetUp1D()
    {
        // 1D specific parameters
        Wi_ = 32;  // Input width
        Wo_ = 30;  // Output width
        X_ = 3;    // Filter width
        
        conv_strides_1d_ = {1};
        conv_dilations_1d_ = {1};
        input_left_pads_1d_ = {0};
        input_right_pads_1d_ = {0};
        
        // Set up dimension arrays
        a_g_n_c_wis_lengths_1d_ = {G_, N_, C_, Wi_};
        b_g_k_c_xs_lengths_1d_ = {G_, K_, C_, X_};
        c_g_n_k_wos_lengths_1d_ = {G_, N_, K_, Wo_};
    }
    
    void SetUp2D()
    {
        // 2D specific parameters
        Hi_ = 32;  // Input height
        Wi_ = 32;  // Input width
        Ho_ = 30;  // Output height
        Wo_ = 30;  // Output width
        Y_ = 3;    // Filter height
        X_ = 3;    // Filter width
        
        conv_strides_2d_ = {1, 1};
        conv_dilations_2d_ = {1, 1};
        input_left_pads_2d_ = {0, 0};
        input_right_pads_2d_ = {0, 0};
        
        // Set up dimension arrays
        a_g_n_c_wis_lengths_2d_ = {G_, N_, C_, Hi_, Wi_};
        b_g_k_c_xs_lengths_2d_ = {G_, K_, C_, Y_, X_};
        c_g_n_k_wos_lengths_2d_ = {G_, N_, K_, Ho_, Wo_};
    }
    
    void SetUp3D()
    {
        // 3D specific parameters
        Di_ = 16;  // Input depth
        Hi_ = 32;  // Input height
        Wi_ = 32;  // Input width
        Do_ = 14;  // Output depth
        Ho_ = 30;  // Output height
        Wo_ = 30;  // Output width
        Z_ = 3;    // Filter depth
        Y_ = 3;    // Filter height
        X_ = 3;    // Filter width
        
        conv_strides_3d_ = {1, 1, 1};
        conv_dilations_3d_ = {1, 1, 1};
        input_left_pads_3d_ = {0, 0, 0};
        input_right_pads_3d_ = {0, 0, 0};
        
        // Set up dimension arrays
        a_g_n_c_wis_lengths_3d_ = {G_, N_, C_, Di_, Hi_, Wi_};
        b_g_k_c_xs_lengths_3d_ = {G_, K_, C_, Z_, Y_, X_};
        c_g_n_k_wos_lengths_3d_ = {G_, N_, K_, Do_, Ho_, Wo_};
    }
    
    // Common parameters
    index_t G_, N_, K_, C_;
    index_t Wi_, Wo_, X_;
    index_t Hi_, Ho_, Y_;
    index_t Di_, Do_, Z_;
    
    // 1D arrays
    std::array<index_t, 1> conv_strides_1d_;
    std::array<index_t, 1> conv_dilations_1d_;
    std::array<index_t, 1> input_left_pads_1d_;
    std::array<index_t, 1> input_right_pads_1d_;
    std::array<index_t, 4> a_g_n_c_wis_lengths_1d_;
    std::array<index_t, 4> b_g_k_c_xs_lengths_1d_;
    std::array<index_t, 4> c_g_n_k_wos_lengths_1d_;
    
    // 2D arrays
    std::array<index_t, 2> conv_strides_2d_;
    std::array<index_t, 2> conv_dilations_2d_;
    std::array<index_t, 2> input_left_pads_2d_;
    std::array<index_t, 2> input_right_pads_2d_;
    std::array<index_t, 5> a_g_n_c_wis_lengths_2d_;
    std::array<index_t, 5> b_g_k_c_xs_lengths_2d_;
    std::array<index_t, 5> c_g_n_k_wos_lengths_2d_;
    
    // 3D arrays
    std::array<index_t, 3> conv_strides_3d_;
    std::array<index_t, 3> conv_dilations_3d_;
    std::array<index_t, 3> input_left_pads_3d_;
    std::array<index_t, 3> input_right_pads_3d_;
    std::array<index_t, 6> a_g_n_c_wis_lengths_3d_;
    std::array<index_t, 6> b_g_k_c_xs_lengths_3d_;
    std::array<index_t, 6> c_g_n_k_wos_lengths_3d_;
};

// Type lists for typed tests
using TestTypes = ::testing::Types<
  TestConfig1D_no_merge, 
  TestConfig2D_no_merge, 
  TestConfig3D_no_merge,
  TestConfig1D_merge, 
  TestConfig2D_merge, 
  TestConfig3D_merge>;

TYPED_TEST_SUITE(TestTransformConvBwdWeightToGemm, TestTypes);

// Test constructor
TYPED_TEST(TestTransformConvBwdWeightToGemm, Constructor)
{
    constexpr index_t NDim = TypeParam::NDim;
    
    if constexpr (NDim == 1) {
        typename TypeParam::TransformType transform(this->a_g_n_c_wis_lengths_1d_,
                                                   this->b_g_k_c_xs_lengths_1d_,
                                                   this->c_g_n_k_wos_lengths_1d_,
                                                   this->conv_strides_1d_,
                                                   this->conv_dilations_1d_,
                                                   this->input_left_pads_1d_,
                                                   this->input_right_pads_1d_);
        
        // Verify that the transformer was constructed successfully
        EXPECT_EQ(transform.G_, this->G_);
        EXPECT_EQ(transform.N_, this->N_);
        EXPECT_EQ(transform.K_, this->K_);
        EXPECT_EQ(transform.C_, this->C_);
        EXPECT_EQ(transform.Wi_, this->Wi_);
        EXPECT_EQ(transform.Wo_, this->Wo_);
        EXPECT_EQ(transform.X_, this->X_);
        EXPECT_EQ(transform.ZYX_, this->X_);
    } else if constexpr (NDim == 2) {
        typename TypeParam::TransformType transform(this->a_g_n_c_wis_lengths_2d_,
                                                   this->b_g_k_c_xs_lengths_2d_,
                                                   this->c_g_n_k_wos_lengths_2d_,
                                                   this->conv_strides_2d_,
                                                   this->conv_dilations_2d_,
                                                   this->input_left_pads_2d_,
                                                   this->input_right_pads_2d_);
        
        // Verify that the transformer was constructed successfully
        EXPECT_EQ(transform.G_, this->G_);
        EXPECT_EQ(transform.N_, this->N_);
        EXPECT_EQ(transform.K_, this->K_);
        EXPECT_EQ(transform.C_, this->C_);
        EXPECT_EQ(transform.Hi_, this->Hi_);
        EXPECT_EQ(transform.Wi_, this->Wi_);
        EXPECT_EQ(transform.Ho_, this->Ho_);
        EXPECT_EQ(transform.Wo_, this->Wo_);
        EXPECT_EQ(transform.Y_, this->Y_);
        EXPECT_EQ(transform.X_, this->X_);
        EXPECT_EQ(transform.ZYX_, this->Y_ * this->X_);
    } else if constexpr (NDim == 3) {
        typename TypeParam::TransformType transform(this->a_g_n_c_wis_lengths_3d_,
                                                   this->b_g_k_c_xs_lengths_3d_,
                                                   this->c_g_n_k_wos_lengths_3d_,
                                                   this->conv_strides_3d_,
                                                   this->conv_dilations_3d_,
                                                   this->input_left_pads_3d_,
                                                   this->input_right_pads_3d_);
        
        // Verify that the transformer was constructed successfully
        EXPECT_EQ(transform.G_, this->G_);
        EXPECT_EQ(transform.N_, this->N_);
        EXPECT_EQ(transform.K_, this->K_);
        EXPECT_EQ(transform.C_, this->C_);
        EXPECT_EQ(transform.Di_, this->Di_);
        EXPECT_EQ(transform.Hi_, this->Hi_);
        EXPECT_EQ(transform.Wi_, this->Wi_);
        EXPECT_EQ(transform.Do_, this->Do_);
        EXPECT_EQ(transform.Ho_, this->Ho_);
        EXPECT_EQ(transform.Wo_, this->Wo_);
        EXPECT_EQ(transform.Z_, this->Z_);
        EXPECT_EQ(transform.Y_, this->Y_);
        EXPECT_EQ(transform.X_, this->X_);
        EXPECT_EQ(transform.ZYX_, this->Z_ * this->Y_ * this->X_);
    }
}

// Test grid descriptors
TYPED_TEST(TestTransformConvBwdWeightToGemm, GridDescriptors)
{
    constexpr index_t NDim = TypeParam::NDim;
    constexpr auto I0 = number<0>{};
    constexpr auto I1 = number<1>{};
    constexpr auto I2 = number<2>{};
    constexpr auto I3 = number<3>{};
    constexpr auto I4 = number<4>{};
    constexpr auto I5 = number<5>{};
    
    constexpr index_t Gm = TypeParam::NumberOfGroupsToMerge;

    if constexpr (NDim == 1) 
    {
        typename TypeParam::TransformType transform(this->a_g_n_c_wis_lengths_1d_,
                                                   this->b_g_k_c_xs_lengths_1d_,
                                                   this->c_g_n_k_wos_lengths_1d_,
                                                   this->conv_strides_1d_,
                                                   this->conv_dilations_1d_,
                                                   this->input_left_pads_1d_,
                                                   this->input_right_pads_1d_);
        
        // Test individual grid descriptors
        auto out_grid_desc = transform.template make_out_grid_desc<1>();
        auto in_grid_desc = transform.template make_in_grid_desc<1>();
        auto wei_grid_desc = transform.template make_wei_grid_desc<1>();
        
        // Verify input grid descriptor dimensions
        EXPECT_EQ(in_grid_desc.get_length(I0), this->N_);
        EXPECT_EQ(in_grid_desc.get_length(I1), this->Wi_);
        
        // Verify output grid descriptor dimensions
        EXPECT_EQ(out_grid_desc.get_length(I0), this->K_);

        if constexpr (Gm > 1)
        {
            EXPECT_EQ(in_grid_desc.get_length(I2), Gm);
            EXPECT_EQ(in_grid_desc.get_length(I3), this->C_);

            EXPECT_EQ(wei_grid_desc.get_length(I0), Gm);
            EXPECT_EQ(wei_grid_desc.get_length(I1), this->K_);
            EXPECT_EQ(wei_grid_desc.get_length(I2), this->X_ * this->C_);

            EXPECT_EQ(out_grid_desc.get_length(I1), Gm);
            EXPECT_EQ(out_grid_desc.get_length(I2), this->N_ * this->Wo_);
        }
        else 
        {
            EXPECT_EQ(in_grid_desc.get_length(I2), this->C_);

            EXPECT_EQ(wei_grid_desc.get_length(I0), this->K_);
            EXPECT_EQ(wei_grid_desc.get_length(I1), this->X_ * this->C_);

            EXPECT_EQ(out_grid_desc.get_length(I1), this->N_ * this->Wo_);
        }
        
    } 
    else if constexpr (NDim == 2) 
    {
        typename TypeParam::TransformType transform(this->a_g_n_c_wis_lengths_2d_,
                                                   this->b_g_k_c_xs_lengths_2d_,
                                                   this->c_g_n_k_wos_lengths_2d_,
                                                   this->conv_strides_2d_,
                                                   this->conv_dilations_2d_,
                                                   this->input_left_pads_2d_,
                                                   this->input_right_pads_2d_);
        
        // Test individual grid descriptors
        auto out_grid_desc = transform.template make_out_grid_desc<2>();
        auto in_grid_desc = transform.template make_in_grid_desc<2>();
        auto wei_grid_desc = transform.template make_wei_grid_desc<2>();
        
        // Verify input grid descriptor dimensions
        EXPECT_EQ(in_grid_desc.get_length(I0), this->N_);
        EXPECT_EQ(in_grid_desc.get_length(I1), this->Hi_);
        EXPECT_EQ(in_grid_desc.get_length(I2), this->Wi_);

        // Verify output grid descriptor dimensions
        EXPECT_EQ(out_grid_desc.get_length(I0), this->K_);

        if constexpr (Gm > 1)
        {
            EXPECT_EQ(in_grid_desc.get_length(I3), Gm);
            EXPECT_EQ(in_grid_desc.get_length(I4), this->C_);

            EXPECT_EQ(wei_grid_desc.get_length(I0), Gm);
            EXPECT_EQ(wei_grid_desc.get_length(I1), this->K_);
            EXPECT_EQ(wei_grid_desc.get_length(I2), this->Y_ * this->X_ * this->C_);

            EXPECT_EQ(out_grid_desc.get_length(I1), Gm);
            EXPECT_EQ(out_grid_desc.get_length(I2), this->N_ * this->Ho_ * this->Wo_);
        }
        else 
        {
            EXPECT_EQ(in_grid_desc.get_length(I3), this->C_);

            EXPECT_EQ(wei_grid_desc.get_length(I0), this->K_);
            EXPECT_EQ(wei_grid_desc.get_length(I1), this->Y_ * this->X_ * this->C_);

            EXPECT_EQ(out_grid_desc.get_length(I1), this->N_ * this->Ho_ * this->Wo_);
        }
        
    } 
    else if constexpr (NDim == 3) 
    {
        typename TypeParam::TransformType transform(this->a_g_n_c_wis_lengths_3d_,
                                                   this->b_g_k_c_xs_lengths_3d_,
                                                   this->c_g_n_k_wos_lengths_3d_,
                                                   this->conv_strides_3d_,
                                                   this->conv_dilations_3d_,
                                                   this->input_left_pads_3d_,
                                                   this->input_right_pads_3d_);
        
        // Test individual grid descriptors
        auto out_grid_desc = transform.template make_out_grid_desc<3>();
        auto in_grid_desc = transform.template make_in_grid_desc<3>();
        auto wei_grid_desc = transform.template make_wei_grid_desc<3>();
        
        // Verify input grid descriptor dimensions
        EXPECT_EQ(in_grid_desc.get_length(I0), this->N_);
        EXPECT_EQ(in_grid_desc.get_length(I1), this->Di_);
        EXPECT_EQ(in_grid_desc.get_length(I2), this->Hi_);
        EXPECT_EQ(in_grid_desc.get_length(I3), this->Wi_);
        
        // Verify output grid descriptor dimensions
        EXPECT_EQ(out_grid_desc.get_length(I0), this->K_);

        if constexpr (Gm > 1)
        {
            EXPECT_EQ(in_grid_desc.get_length(I4), Gm);
            EXPECT_EQ(in_grid_desc.get_length(I5), this->C_);

            EXPECT_EQ(wei_grid_desc.get_length(I0), Gm);
            EXPECT_EQ(wei_grid_desc.get_length(I1), this->K_);
            EXPECT_EQ(wei_grid_desc.get_length(I2), this->Z_ * this->Y_ * this->X_ * this->C_);

            EXPECT_EQ(out_grid_desc.get_length(I1), Gm);
            EXPECT_EQ(out_grid_desc.get_length(I2), this->N_ * this->Do_ * this->Ho_ * this->Wo_);
        }
        else 
        {
            EXPECT_EQ(in_grid_desc.get_length(I4), this->C_);

            EXPECT_EQ(wei_grid_desc.get_length(I0), this->K_);
            EXPECT_EQ(wei_grid_desc.get_length(I1), this->Z_ * this->Y_ * this->X_ * this->C_);

            EXPECT_EQ(out_grid_desc.get_length(I1), this->N_ * this->Do_ * this->Ho_ * this->Wo_);
        }
    }
}

// Test ABC grid descriptors
TYPED_TEST(TestTransformConvBwdWeightToGemm, ABCGridDescriptors)
{
    constexpr index_t Gm = TypeParam::NumberOfGroupsToMerge;
    constexpr index_t NDim = TypeParam::NDim;
    constexpr auto I0 = number<0>{};
    constexpr auto I1 = number<1>{};
    constexpr auto I2 = number<2>{};
    
    if constexpr (NDim == 1) 
    {
        typename TypeParam::TransformType transform(this->a_g_n_c_wis_lengths_1d_,
                                                   this->b_g_k_c_xs_lengths_1d_,
                                                   this->c_g_n_k_wos_lengths_1d_,
                                                   this->conv_strides_1d_,
                                                   this->conv_dilations_1d_,
                                                   this->input_left_pads_1d_,
                                                   this->input_right_pads_1d_);
        
        // Test combined ABC grid descriptors
        const auto abc_descriptors = transform.template MakeABCGridDescriptor_A_K0_M_K1_B_K0_N_K1_C_M_N<1>();
        const auto& out_desc = abc_descriptors[I0]; // GEMM A ()
        const auto& in_desc = abc_descriptors[I1]; // GEMM B (N_gemm, K_gemm)
        const auto& wei_desc = abc_descriptors[I2]; // GEMM C

        EXPECT_EQ(out_desc.get_num_of_dimension(), 3);
        EXPECT_EQ(in_desc.get_num_of_dimension(), 2);
        EXPECT_EQ(wei_desc.get_num_of_dimension(), 3);
        
        EXPECT_EQ(in_desc.get_length(I0), this->X_ * this->C_);
        EXPECT_EQ(in_desc.get_length(I1), this->N_ * this->Wo_ * Gm);

        // // Verify GEMM M-dimension that should depend on Gm
        // EXPECT_EQ(out_desc.get_length(I1), this->K_ * Gm);
        // EXPECT_EQ(wei_desc.get_length(I0), this->K_ * Gm);
        
        // // Verify GEMM N-dimension that should depend on Gm.
        // EXPECT_EQ(in_desc.get_length(I1), this->X_ * this->C_ * Gm);
        // EXPECT_EQ(wei_desc.get_length(I1), this->X_ * this->C_ * Gm);
    } 
    else if constexpr (NDim == 2) 
    {
        typename TypeParam::TransformType transform(this->a_g_n_c_wis_lengths_2d_,
                                                   this->b_g_k_c_xs_lengths_2d_,
                                                   this->c_g_n_k_wos_lengths_2d_,
                                                   this->conv_strides_2d_,
                                                   this->conv_dilations_2d_,
                                                   this->input_left_pads_2d_,
                                                   this->input_right_pads_2d_);
        
        // Test combined ABC grid descriptors
        auto abc_descriptors = transform.template MakeABCGridDescriptor_A_K0_M_K1_B_K0_N_K1_C_M_N<2>();
        const auto& out_desc = abc_descriptors[I0];
        const auto& in_desc = abc_descriptors[I1];
        const auto& wei_desc = abc_descriptors[I2];
        
        EXPECT_EQ(out_desc.get_length(I1), this->K_ * Gm);
        EXPECT_EQ(wei_desc.get_length(I0), this->K_ * Gm);
        
        EXPECT_EQ(in_desc.get_length(I1), this->Y_ * this->X_ * this->C_ * Gm);
        EXPECT_EQ(wei_desc.get_length(I1),  this->Y_ *this->X_ * this->C_ * Gm);
        
    } 
    else if constexpr (NDim == 3) 
    {
        typename TypeParam::TransformType transform(this->a_g_n_c_wis_lengths_3d_,
                                                   this->b_g_k_c_xs_lengths_3d_,
                                                   this->c_g_n_k_wos_lengths_3d_,
                                                   this->conv_strides_3d_,
                                                   this->conv_dilations_3d_,
                                                   this->input_left_pads_3d_,
                                                   this->input_right_pads_3d_);
        
        // Test combined ABC grid descriptors
        auto abc_descriptors = transform.template MakeABCGridDescriptor_A_K0_M_K1_B_K0_N_K1_C_M_N<3>();
        const auto& out_desc = abc_descriptors[I0];
        const auto& in_desc = abc_descriptors[I1];
        const auto& wei_desc = abc_descriptors[I2];

        EXPECT_EQ(out_desc.get_length(I1), this->K_ * Gm);
        EXPECT_EQ(wei_desc.get_length(I0), this->K_ * Gm);
        
        EXPECT_EQ(in_desc.get_length(I1), this->Z_ * this->Y_ * this->X_ * this->C_ * Gm);
        EXPECT_EQ(wei_desc.get_length(I1), this->Z_ * this->Y_ *this->X_ * this->C_ * Gm);
    }
}

// Specialized test class for 2D convolution edge cases
template <typename Config>
class TestTransformConvBwdWeightToGemm2DEdgeCases : public ::testing::Test
{
protected:
    //using TransformType = TestConfig2D_no_merge::TransformType;
    using TransformType = typename Config::TransformType;
    
    void SetUp() override
    {
        // Common parameters
        G_ = 2; N_ = 4; K_ = 128; C_ = 64;
        Hi_ = Wi_ = 32; Ho_ = Wo_ = 30; Y_ = X_ = 3;
        
        conv_strides_ = {1, 1};
        conv_dilations_ = {1, 1};
        input_left_pads_ = {0, 0};
        input_right_pads_ = {0, 0};
        
        a_g_n_c_wis_lengths_ = {G_, N_, C_, Hi_, Wi_};
        b_g_k_c_xs_lengths_ = {G_, K_, C_, Y_, X_};
        c_g_n_k_wos_lengths_ = {G_, N_, K_, Ho_, Wo_};
    }
    
    index_t G_, N_, K_, C_, Hi_, Wi_, Ho_, Wo_, Y_, X_;
    std::array<index_t, 2> conv_strides_;
    std::array<index_t, 2> conv_dilations_;
    std::array<index_t, 2> input_left_pads_;
    std::array<index_t, 2> input_right_pads_;
    std::array<index_t, 5> a_g_n_c_wis_lengths_;
    std::array<index_t, 5> b_g_k_c_xs_lengths_;
    std::array<index_t, 5> c_g_n_k_wos_lengths_;
};

// Edge case typed tests
using EdgeCaseTypes = ::testing::Types<TestConfig2D_no_merge>;

TYPED_TEST_SUITE(TestTransformConvBwdWeightToGemm2DEdgeCases, EdgeCaseTypes);

TYPED_TEST(TestTransformConvBwdWeightToGemm2DEdgeCases, WithPadding)
{
    // Modify parameters to include padding
    this->input_left_pads_ = {1, 1};
    this->input_right_pads_ = {1, 1};
    
    // With padding, output size should be: (input + 2*pad - filter) / stride + 1
    // (32 + 2*1 - 3) / 1 + 1 = 32
    this->Ho_ = this->Wo_ = 32;
    this->c_g_n_k_wos_lengths_[3] = this->Ho_;
    this->c_g_n_k_wos_lengths_[4] = this->Wo_;
    
    typename TypeParam::TransformType transform(this->a_g_n_c_wis_lengths_,
                                             this->b_g_k_c_xs_lengths_,
                                             this->c_g_n_k_wos_lengths_,
                                             this->conv_strides_,
                                             this->conv_dilations_,
                                             this->input_left_pads_,
                                             this->input_right_pads_);
    
    // Verify padding was set correctly
    EXPECT_EQ(transform.InLeftPadH_, 1);
    EXPECT_EQ(transform.InLeftPadW_, 1);
    EXPECT_EQ(transform.InRightPadH_, 1);
    EXPECT_EQ(transform.InRightPadW_, 1);
    
    // Test that grid descriptors can still be created
    auto out_grid_desc = transform.template make_out_grid_desc<2>();
    
    EXPECT_EQ(out_grid_desc.get_length(number<1>{}), this->N_ * this->Ho_ * this->Wo_);
}

TYPED_TEST(TestTransformConvBwdWeightToGemm2DEdgeCases, WithStride)
{
    // Modify parameters to include stride
    this->conv_strides_ = {2, 2};
    
    // With stride 2, output size should be: (input - filter) / stride + 1
    // (32 - 3) / 2 + 1 = 15
    this->Ho_ = this->Wo_ = 15;
    this->c_g_n_k_wos_lengths_[3] = this->Ho_;
    this->c_g_n_k_wos_lengths_[4] = this->Wo_;
    
    typename TypeParam::TransformType transform(this->a_g_n_c_wis_lengths_,
                                             this->b_g_k_c_xs_lengths_,
                                             this->c_g_n_k_wos_lengths_,
                                             this->conv_strides_,
                                             this->conv_dilations_,
                                             this->input_left_pads_,
                                             this->input_right_pads_);
    
    // Verify stride was set correctly
    EXPECT_EQ(transform.ConvStrideH_, 2);
    EXPECT_EQ(transform.ConvStrideW_, 2);
    
    // Test that grid descriptors can still be created
    auto out_grid_desc = transform.template make_out_grid_desc<2>();
    EXPECT_EQ(out_grid_desc.get_length(number<1>{}), this->N_ * this->Ho_ * this->Wo_);
}

TYPED_TEST(TestTransformConvBwdWeightToGemm2DEdgeCases, WithDilation)
{
    // Modify parameters to include dilation
    this->conv_dilations_ = {2, 2};
    
    // With dilation 2, effective filter size is: (filter - 1) * dilation + 1 = (3 - 1) * 2 + 1 = 5
    // Output size: (32 - 5) / 1 + 1 = 28
    this->Ho_ = this->Wo_ = 28;
    this->c_g_n_k_wos_lengths_[3] = this->Ho_;
    this->c_g_n_k_wos_lengths_[4] = this->Wo_;
    
    typename TypeParam::TransformType transform(this->a_g_n_c_wis_lengths_,
                                             this->b_g_k_c_xs_lengths_,
                                             this->c_g_n_k_wos_lengths_,
                                             this->conv_strides_,
                                             this->conv_dilations_,
                                             this->input_left_pads_,
                                             this->input_right_pads_);
    
    // Verify dilation was set correctly
    EXPECT_EQ(transform.ConvDilationH_, 2);
    EXPECT_EQ(transform.ConvDilationW_, 2);
    
    // Test that grid descriptors can still be created
    auto out_grid_desc = transform.template make_out_grid_desc<2>();
    EXPECT_EQ(out_grid_desc.get_length(number<1>{}), this->N_ * this->Ho_ * this->Wo_);
}

#ifdef __clang__
#pragma clang diagnostic pop
#endif
