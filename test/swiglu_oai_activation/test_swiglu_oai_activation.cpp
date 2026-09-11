// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <algorithm>
#include <cmath>

#include "gtest/gtest.h"

// ck::swiglu_oai is the single source of truth shared by the XDL 2-stage MoE epilogue
// (Activation::swiglu_oai_and_mul) and this host unit test.
#include "ck/tensor_operation/gpu/device/tensor_layout.hpp"
#include "ck/tensor_operation/gpu/grid/gridwise_gemm_xdl_cshuffle_common.hpp"

namespace {

// Independent fp64 reference for the OAI / gpt-oss SwiGLU activation:
//   gate * sigmoid(alpha * gate) * (up + 1)
// with gate clamped to <= limit and up clamped to [-limit, limit]. Computed in double so
// we are not comparing the fp32 implementation under test against a copy of itself.
double ref_swiglu_oai(double gate, double up, double limit = 7.0, double alpha = 1.702)
{
    gate           = std::min(gate, limit);
    up             = std::min(std::max(up, -limit), limit);
    const double s = 1.0 / (1.0 + std::exp(alpha * -gate));
    return gate * s * (up + 1.0);
}

constexpr float kTol = 1e-3f;

} // namespace

// Values match the reference when no clamping is active.
TEST(SwigluOai, MatchesReferenceInRange)
{
    const float pts[][2] = {{0.f, 0.f}, {1.f, 2.f}, {-1.5f, 0.5f}, {3.f, -2.f}, {-4.f, -3.f}};
    for(const auto& p : pts)
    {
        const float got = ck::swiglu_oai(p[0], p[1]);
        const float ref = static_cast<float>(ref_swiglu_oai(p[0], p[1]));
        EXPECT_NEAR(got, ref, kTol) << "gate=" << p[0] << " up=" << p[1];
    }
}

// gate is upper-bounded to limit (7); it is NOT lower-clamped.
TEST(SwigluOai, GateUpperClamp)
{
    EXPECT_NEAR(ck::swiglu_oai(100.f, 1.f), static_cast<float>(ref_swiglu_oai(7.0, 1.0)), kTol);
    // gate >= limit all saturate to the same value.
    EXPECT_NEAR(ck::swiglu_oai(7.f, 1.f), ck::swiglu_oai(100.f, 1.f), kTol);
    // large negative gate passes through unclamped.
    EXPECT_NEAR(ck::swiglu_oai(-50.f, 1.f), static_cast<float>(ref_swiglu_oai(-50.0, 1.0)), kTol);
}

// up is symmetric-clamped to [-7, 7].
TEST(SwigluOai, UpSymmetricClamp)
{
    EXPECT_NEAR(ck::swiglu_oai(1.f, 100.f), static_cast<float>(ref_swiglu_oai(1.0, 7.0)), kTol);
    EXPECT_NEAR(ck::swiglu_oai(1.f, -100.f), static_cast<float>(ref_swiglu_oai(1.0, -7.0)), kTol);
}

// The "+1" shift on up is part of the OAI form: up == -1 zeroes the output exactly.
TEST(SwigluOai, UpPlusOneShiftZeroesOutput)
{
    EXPECT_FLOAT_EQ(ck::swiglu_oai(2.f, -1.f), 0.f);
    EXPECT_FLOAT_EQ(ck::swiglu_oai(-3.f, -1.f), 0.f);
}

// alpha defaults to 1.702 (gpt-oss); passing it explicitly must not change the result.
TEST(SwigluOai, DefaultAlphaMatchesExplicit)
{
    EXPECT_FLOAT_EQ(ck::swiglu_oai(2.f, 3.f), ck::swiglu_oai(2.f, 3.f, 7.0f, 1.702f));
}
