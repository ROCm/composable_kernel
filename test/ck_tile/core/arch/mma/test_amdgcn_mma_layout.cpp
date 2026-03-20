// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <hip/hip_runtime.h>
#include <gtest/gtest.h>

#include "ck_tile/host/hip_check_error.hpp"
#include "ck_tile/host/stream_config.hpp"
#include "ck_tile/host/device_memory.hpp"
#include "ck_tile/host/kernel_launch.hpp"
#include "ck_tile/core/arch/arch.hpp"
#include "ck_tile/core/utility/env.hpp"

#include "test/ck_tile/core/arch/mma/test_amdgcn_mma_layout_util.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <vector>
#include <cstddef>
#include <string>
#include <tuple>

namespace ck  = ck_tile;
namespace mma = ck_tile::core::arch::mma;

// MMA register layout validation test for amdgcn_mma structs.
//
// Strategy: for every (m, k, n) triple in the tile, the test constructs a pair of input tensors
// A and B that contain exactly one non-zero element each, placed so that their product
// contributes to a single output element C(m, n):
//
//         A  (M x K)                B  (K x N)              C = A * B  (M x N)
//      . . . . . . . .           . . . . . . . .           . . . . . . . .
//      . . . . . . . .           . . . . . . . .           . . . . . . . .
//      . . . 1 . . . .           . . . . . . . .           . . . . . . . .
//      . . . . . . . .           . . . 1 . . . .           . . . . . 1 . .
//      . . . . . . . .           . . . . . . . .           . . . . . . . .
//         A(m,k) = 1                B(k,n) = 1                C(m,n) = 1
//
// The kernel uses RegisterMap to scatter A and B into the correct (lane, vecIdx) positions
// of the MMA fragment registers, executes the intrinsic, then uses RegisterMap again to
// gather back into C matrix. The position of "1" in C is checked against the expected (m, n)
// location.

namespace {

/**
 * @class MmaLayoutTestKernel
 * @brief Device kernel that performs C = AB using a given Mma op
 *
 * @tparam ADataType     Data type of tensor A elements
 * @tparam BDataType     Data type of tensor B elements
 * @tparam CDataType     Data type of tensor C elements
 * @tparam FragM         M-dimension of the MMA tile
 * @tparam FragN         N-dimension of the MMA tile
 * @tparam FragK         K-dimension of the MMA tile
 * @tparam BlockSize     HIP block size
 */
template <typename ADataType,
          typename BDataType,
          typename CDataType,
          uint32_t FragM,
          uint32_t FragN,
          uint32_t FragK,
          uint32_t BlockSize>
struct MmaLayoutTestKernel
{
    static constexpr int kBlockSize = BlockSize;

    __device__ void operator()(uint32_t* error_flags) const
    {
        using Selector =
            mma::MmaDefaultSelector<ADataType,
                                    BDataType,
                                    CDataType,
                                    FragM,
                                    FragN,
                                    FragK,
                                    decltype(ck_tile::core::arch::get_compiler_target()),
                                    mma::MmaOpFamily::DENSE>;
        using MmaOp = typename Selector::SelectedOp;

        if constexpr(mma::MmaOpTraits<MmaOp>::IsSupported)
        {
            using AVecType                = typename MmaOp::AVecType;
            using BVecType                = typename MmaOp::BVecType;
            using CVecType                = typename MmaOp::CVecType;
            constexpr uint32_t a_vec_size = vector_traits<AVecType>::vector_size;
            constexpr uint32_t b_vec_size = vector_traits<BVecType>::vector_size;
            constexpr uint32_t c_vec_size = vector_traits<CVecType>::vector_size;

            const uint32_t lane = threadIdx.x;

            AVecType a_frag{};
            BVecType b_frag{};
            CVecType c_frag{};

            // get (m, k, n), where "1" should be placed for this block
            const uint32_t case_idx = static_cast<uint32_t>(blockIdx.x);
            const uint32_t m        = case_idx / (MmaOp::kK * MmaOp::kN);
            const uint32_t k        = (case_idx / MmaOp::kN) % MmaOp::kK;
            const uint32_t n        = case_idx % MmaOp::kN;

            // place a single "1" in A/B fragments using (lane, vecIdx) -> (row, col) mapping
            for(uint32_t v = 0; v < a_vec_size; ++v)
            {
                auto a_coords = RegisterMap<MmaOp>::Register2AMap(lane, v);
                if(static_cast<uint32_t>(a_coords[0]) == m &&
                   static_cast<uint32_t>(a_coords[1]) == k)
                {
                    a_frag[v] = static_cast<ADataType>(1);
                }
            }

            for(uint32_t v = 0; v < b_vec_size; ++v)
            {
                auto b_coords = RegisterMap<MmaOp>::Register2BMap(lane, v);
                if(static_cast<uint32_t>(b_coords[0]) == n &&
                   static_cast<uint32_t>(b_coords[1]) == k)
                {
                    b_frag[v] = static_cast<BDataType>(1);
                }
            }

            c_frag = MmaOp::exec(a_frag, b_frag, c_frag);

            uint32_t err        = 0;
            const CDataType tol = static_cast<CDataType>(
                1.0e-1f); // TODO: this tolerance might not be suitable for all data types and
                          // should be revisited if we add more configurations
            for(uint32_t v = 0; v < c_vec_size; ++v)
            {
                auto c_coords    = RegisterMap<MmaOp>::Register2CMap(lane, v);
                const uint32_t i = static_cast<uint32_t>(c_coords[0]);
                const uint32_t j = static_cast<uint32_t>(c_coords[1]);

                const CDataType expected =
                    (i == m && j == n) ? static_cast<CDataType>(1) : static_cast<CDataType>(0);
                const CDataType value = static_cast<CDataType>(c_frag[v]);
                if(fabsf(static_cast<float>(value - expected)) > static_cast<float>(tol))
                {
                    err = 1;
                }
            }

            const uint32_t any_err = __any(err);
            if(threadIdx.x == 0)
            {
                error_flags[case_idx] = any_err;
            }
        }
    }
};

/**
 * @brief Test driver: runs the test for a given MMA configuration.
 *
 * The testlaunches (mkn) test cases (one per block) to check all possible positions of the "1" in
 * the A/B tensors.
 *   1. Constructs A and B tensors with a single 1 at A(m,k) and B(k,n).
 *   2. Executes MMA intrinsic to compute C tensor.
 *   3. Checks if C has the 1 in the expected position.
 *
 * @tparam Selector  Selector for the Mma operation
 * @return true if the test ran on hardware; false if skipped (no device or unsupported)
 */
template <typename Selector>
bool run_mma_layout_test()
{
    using MmaOp                       = typename Selector::SelectedOp;
    using MmaTraits                   = mma::MmaOpTraits<MmaOp>;
    using ADataType                   = typename MmaOp::ADataType;
    using BDataType                   = typename MmaOp::BDataType;
    using CDataType                   = typename MmaOp::CDataType;
    constexpr uint32_t FragM          = MmaOp::kM;
    constexpr uint32_t FragN          = MmaOp::kN;
    constexpr uint32_t FragK          = MmaOp::kK;
    constexpr auto selector_target_id = MmaTraits::CompilerTarget::TARGET_ID;
    constexpr auto selector_wave_size = MmaTraits::CompilerTarget::WAVE_SIZE_ID;

    int device_count = 0;
    hipDevice_t device{};
    HIP_CHECK_ERROR(hipGetDevice(&device));
    HIP_CHECK_ERROR(hipGetDeviceCount(&device_count));

    hipDeviceProp_t props{};
    HIP_CHECK_ERROR(hipGetDeviceProperties(&props, device));

    const auto runtime_target =
        ck_tile::core::arch::hip_device_prop_gcn_arch_name_to_amdgcn_target_id(props.gcnArchName);
    const bool has_device = device_count > 0;

    if(!has_device || runtime_target == ck_tile::core::arch::amdgcn_target_id::HOST ||
       runtime_target != selector_target_id ||
       props.warpSize != static_cast<int>(selector_wave_size))
    {
        return false;
    }

    constexpr uint32_t total_cases = FragM * FragK * FragN;
    ck_tile::DeviceMem d_errors(total_cases * sizeof(uint32_t));
    std::vector<uint32_t> h_errors(total_cases, 0u);

    auto* d_error_ptr = static_cast<uint32_t*>(d_errors.GetDeviceBuffer());

    std::ignore = hipGetLastError();

    using Kernel = MmaLayoutTestKernel<ADataType,
                                       BDataType,
                                       CDataType,
                                       FragM,
                                       FragN,
                                       FragK,
                                       static_cast<int>(selector_wave_size)>;

    std::ignore =
        ck_tile::launch_kernel(ck_tile::stream_config{nullptr, false, 0, 0, 1},
                               ck_tile::make_kernel(Kernel{},
                                                    dim3(total_cases),
                                                    dim3(static_cast<int>(selector_wave_size)),
                                                    0,
                                                    d_error_ptr));

    HIP_CHECK_ERROR(hipMemcpyAsync(
        h_errors.data(), d_error_ptr, d_errors.GetBufferSize(), hipMemcpyDeviceToHost));
    HIP_CHECK_ERROR(hipStreamSynchronize(nullptr));

    for(uint32_t case_idx = 0; case_idx < total_cases; ++case_idx)
    {
        const uint32_t m = case_idx / (FragK * FragN);
        const uint32_t k = (case_idx / FragN) % FragK;
        const uint32_t n = case_idx % FragN;

        EXPECT_EQ(h_errors[case_idx], 0u) << "Mismatch for m=" << m << " k=" << k << " n=" << n;
    }

    return true;
}

} // namespace

// ==================== Test configurations per target ====================
// TODO: currently we have only 1 specific target per test. This should be revisited to enable all
// the targets within the family (gfx12, gfx11, gfx9)
using MmaGfx1201CompilerTarget = decltype(ck_tile::core::arch::make_amdgcn_gfx12_target<
                                          ck_tile::core::arch::amdgcn_target_id::GFX1201>());
using MmaGfx90aCompilerTarget  = decltype(ck_tile::core::arch::make_amdgcn_gfx9_target<
                                          ck_tile::core::arch::amdgcn_target_id::GFX90A>());
using MmaGfx1100CompilerTarget = decltype(ck_tile::core::arch::make_amdgcn_gfx11_target<
                                          ck_tile::core::arch::amdgcn_target_id::GFX1100>());

using MmaGfx1201Selector = mma::MmaDefaultSelector<ck::fp16_t,
                                                   ck::fp16_t,
                                                   ck::fp32_t,
                                                   16u,
                                                   16u,
                                                   16u,
                                                   MmaGfx1201CompilerTarget,
                                                   mma::MmaOpFamily::DENSE>;
using MmaGfx90aSelector  = mma::MmaDefaultSelector<ck::fp16_t,
                                                   ck::fp16_t,
                                                   ck::fp32_t,
                                                   16u,
                                                   16u,
                                                   16u,
                                                   MmaGfx90aCompilerTarget,
                                                   mma::MmaOpFamily::DENSE>;
using MmaGfx1100Selector = mma::MmaDefaultSelector<ck::fp16_t,
                                                   ck::fp16_t,
                                                   ck::fp32_t,
                                                   16u,
                                                   16u,
                                                   16u,
                                                   MmaGfx1100CompilerTarget,
                                                   mma::MmaOpFamily::DENSE>;

// clang-format off
using KernelTypes = ::testing::Types<
    MmaGfx1201Selector,
    MmaGfx90aSelector,
    MmaGfx1100Selector
    >;
// clang-format on

template <typename Selector>
class TestMmaLayout : public ::testing::Test
{
};

TYPED_TEST_SUITE(TestMmaLayout, KernelTypes);

TYPED_TEST(TestMmaLayout, Mma_16x16x16_F16_F16_F32)
{
    bool executed = run_mma_layout_test<TypeParam>();

    if(!executed)
    {
        GTEST_SKIP() << "No supported HIP device found. Skipping test.";
    }
}
