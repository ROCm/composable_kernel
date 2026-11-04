// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "test_amdgcn_mma_layout.inc"
TYPED_TEST_SUITE(TestMmaLayout, Gfx942Intrinsics);
TYPED_TEST(TestMmaLayout, Gfx942Intrinsics) { run_mma_layout_test<TypeParam>(); }

namespace {

// Signed, fractional TF32 values with exactly representable products and sums, so
// an exact comparison tests both input registers and the nonzero accumulator.
CK_TILE_HOST_DEVICE float input_a(index_t m, index_t k)
{
    return (m % 7 - 3) * 0.25f + (k + 1) * 0.03125f;
}

CK_TILE_HOST_DEVICE float input_b(index_t n, index_t k) { return (n % 5 - 2) * 0.5f - k * 0.0625f; }

CK_TILE_HOST_DEVICE float input_c(index_t m, index_t n) { return 0.5f + (m - n) * 0.125f; }

template <typename MmaOp>
struct Tf32AccumulationKernel
{
    static constexpr int kBlockSize = MmaOp::WaveSize;

    CK_TILE_DEVICE void operator()(float* output) const
    {
        using EncCalc = TileDistrEncCalc<MmaOp, false, 1>;
        using ARegMap = TileDistrEncRegMap<typename EncCalc::AWarpDstrEncoding>;
        using BRegMap = TileDistrEncRegMap<typename EncCalc::BWarpDstrEncoding>;
        using CRegMap = TileDistrEncRegMap<typename EncCalc::CWarpDstrEncoding>;

        typename MmaOp::AVecType a{};
        typename MmaOp::BVecType b{};
        typename MmaOp::CVecType c{};
        const index_t lane = threadIdx.x;

        for(index_t i = 0; i < ARegMap::num_vector_items; ++i)
        {
            const auto coords = ARegMap::calc_matrix_indices_from_lane_vector(lane, i);
            a[i]              = input_a(coords[0], coords[1]);
        }
        for(index_t i = 0; i < BRegMap::num_vector_items; ++i)
        {
            const auto coords = BRegMap::calc_matrix_indices_from_lane_vector(lane, i);
            b[i]              = input_b(coords[0], coords[1]);
        }
        for(index_t i = 0; i < CRegMap::num_vector_items; ++i)
        {
            const auto coords = CRegMap::calc_matrix_indices_from_lane_vector(lane, i);
            c[i]              = input_c(coords[0], coords[1]);
        }

        const auto result = MmaOp::exec(a, b, c);
        for(index_t i = 0; i < CRegMap::num_vector_items; ++i)
        {
            const auto coords = CRegMap::calc_matrix_indices_from_lane_vector(lane, i);
            output[coords[0] * MmaOp::kN + coords[1]] = result[i];
        }
    }
};

} // namespace

TYPED_TEST(TestMmaLayout, Gfx942Tf32Accumulation)
{
    using MmaOp = TypeParam;
    std::vector<float> output(MmaOp::kM * MmaOp::kN);
    ck_tile::DeviceMem device_output(output.size() * sizeof(float));

    ck_tile::launch_kernel(
        ck_tile::stream_config{nullptr, false, 0, 0, 1},
        ck_tile::make_kernel(Tf32AccumulationKernel<MmaOp>{},
                             dim3(1),
                             dim3(MmaOp::WaveSize),
                             0,
                             static_cast<float*>(device_output.GetDeviceBuffer())));
    device_output.FromDevice(output.data());

    for(index_t m = 0; m < MmaOp::kM; ++m)
    {
        for(index_t n = 0; n < MmaOp::kN; ++n)
        {
            float expected = input_c(m, n);
            for(index_t k = 0; k < MmaOp::kK; ++k)
                expected += input_a(m, k) * input_b(n, k);
            EXPECT_EQ(output[m * MmaOp::kN + n], expected) << "m=" << m << " n=" << n;
        }
    }
}
