// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "get_wave_size_helper.hpp"

#include "ck_tile/core/arch/arch.hpp"
#include "ck_tile/core/arch/mma/amdgcn_mma.hpp"
#include "ck_tile/core/arch/mma/mma_op_family.hpp"
#include "ck_tile/core/arch/mma/mma_selector.hpp"
#include "ck_tile/core/arch/mma/mma_traits.hpp"
#include "ck_tile/core/config.hpp"
#include "ck_tile/core/numeric/vector_type.hpp"
#include "ck_tile/host/hip_check_error.hpp"

#include <gtest/gtest.h>
#include <hip/hip_runtime.h>

#include <cstdint>
#include <type_traits>

using namespace ck_tile;
using namespace ck_tile::core::arch;
using namespace ck_tile::core::arch::mma;

// Dummy values for testing
constexpr uint32_t DummyTargetIdVal = 55555u;
using DummyCompilerTarget = amdgcn_target<static_cast<amdgcn_target_id>(DummyTargetIdVal)>;
struct DummyOpType;
struct DummyCtrlFlags
{
};

/** @brief Returns true if the given target id matches the dummy */
constexpr bool is_dummy_target(DummyCompilerTarget dummy)
{
    return static_cast<uint32_t>(dummy.TARGET_ID) == DummyTargetIdVal;
}

// Enable if for dummy architecture ID
// TODO: c++20 template <amdgcn_target_arch_id CompilerTarget>
template <typename CompilerTarget>
using enable_if_target_id_dummy_t = std::enable_if_t<is_dummy_target(CompilerTarget{})>;

// Specialization of amdgcn_mma for a supported dummy architecture.
// This way, we don't have to worry about underlying architectural details,
// and can focus on testing the mechanism of selecting supported vs unsupported architectures.
// TODO: c++20 template <amdgcn_target_arch_id CompilerTarget>
template <typename CompilerTarget>
// clang-format off
//               | A B C DataTypes      | MNK + WaveSize |AParams |BPar |CPar |
struct amdgcn_mma<fp32_t, fp32_t, fp32_t, 8u, 8u, 8u, DummyCtrlFlags, CompilerTarget, MmaOpFamily::DENSE, enable_if_target_id_dummy_t<CompilerTarget>>
: amdgcn_mma_base<fp32_t, fp32_t, fp32_t, 8u, 8u, 8u, 64u, 1, 1, 1, 1, 1, 1, 1, DummyOpType, MmaOpFamily::DENSE>
// clang-format on
{
    CK_TILE_DEVICE static CVecType
    exec(AVecType const& regsA, BVecType const& regsB, CVecType const& regsC)
    {
        return regsA + regsB + regsC; // Simple operation for testing
    }
};

// Have an alias so we can test supported arch vs unsupported arch
// TODO: c++20 template <amdgcn_target_arch_id CompilerTarget>
template <typename CompilerTarget>
using DummyAmdgcnMma = amdgcn_mma<fp32_t,
                                  fp32_t,
                                  fp32_t,
                                  8u,
                                  8u,
                                  8u,
                                  DummyCtrlFlags,
                                  CompilerTarget,
                                  MmaOpFamily::DENSE>;

/*! @struct MmaDefaultSelector
 * @brief For dummy Id only, instantiate tests for both MFMA and WMMA selectors so we can them both
 * @tparam ADataType      Data type of matrix A
 * @tparam BDataType      Data type of matrix B
 * @tparam CDataType      Data type of the accumulator
 * @tparam WaveTileM      Size of the M dimension of the WaveTile to decompose
 * @tparam WaveTileN      Size of the N dimension of the WaveTile to decompose
 * @tparam WaveTileK      Size of the K dimension of the WaveTile to decompose
 * @tparam CompilerTarget The compiler target
 * @tparam OpFamily       The MMA operation family
 */
template <typename ADataType,
          typename BDataType,
          typename CDataType,
          uint32_t WaveTileM,
          uint32_t WaveTileN,
          uint32_t WaveTileK,
          typename CompilerTarget,
          MmaOpFamily OpFamily>
// TODO: c++20 amdgcn_target_arch_id CompilerTarget>
// TODO: requires
struct MmaDefaultSelector<ADataType,
                          BDataType,
                          CDataType,
                          WaveTileM,
                          WaveTileN,
                          WaveTileK,
                          CompilerTarget,
                          OpFamily,
                          enable_if_all<enable_if_target_id_dummy_t<CompilerTarget>,
                                        std::enable_if_t<OpFamily == MmaOpFamily::DENSE>>>
{
    using SelectedOp = DummyAmdgcnMma<CompilerTarget>;
};

// Test case for supported architecture
TEST(TestAmdgcnMma, ArchSupported)
{
    // Instantiate MmaOp with the dummy supported CompilerTarget
    using MmaOp = DummyAmdgcnMma<DummyCompilerTarget>;

    EXPECT_TRUE((!std::is_same_v<typename MmaOp::OpType, Unsupported>));

    // Additional tests for DummyArchSupported: check all member variables and types

    // Check OpType
    EXPECT_TRUE(
        (std::is_same<typename MmaOp::OpType, DummyOpType>::value)); // OpType is DummyOpType
    // Check OpFamily
    EXPECT_TRUE((is_mma_op_of_family_v<MmaOpFamily::DENSE, MmaOp>));
}

// Test case for unsupported architecture
TEST(TestAmdgcnMma, ArchUnsupported)
{
    // Instantiate MmaOp with the dummy unsupported CompilerTarget (e.g., HOST)
    using MmaOp = DummyAmdgcnMma<amdgcn_target<>>;

    // OpType should be Unsupported
    EXPECT_TRUE((std::is_same<typename MmaOp::OpType, Unsupported>::value));
    // OpFamily should be Undefined
    EXPECT_TRUE((is_mma_op_of_family_v<MmaOpFamily::UNDEFINED, MmaOp>));
}

// Kernel to test amdgcn_mma::exec on device
template <typename MmaOp>
__global__ void test_amdgcn_mma_exec_kernel(typename MmaOp::AVecType* a,
                                            typename MmaOp::BVecType* b,
                                            typename MmaOp::CVecType* c,
                                            typename MmaOp::CVecType* out)
{
    // This is pseudo-mma behaviour to check the mechanics of mma.
    // All threads write to the same values, so ensure that
    // the inputs are uniform!
    *out = MmaOp::exec(*a, *b, *c);
}

TEST(TestAmdgcnMma, ArchSupportedExecDeviceOutput)
{
    using MmaOp    = DummyAmdgcnMma<DummyCompilerTarget>;
    using DataType = fp32_t;

    typename MmaOp::AVecType h_a;
    typename MmaOp::BVecType h_b;
    typename MmaOp::CVecType h_c;
    typename MmaOp::CVecType h_out;

    // Fill input vectors with known values
    for(size_t i = 0; i < sizeof(h_a) / sizeof(DataType); ++i)
    {
        reinterpret_cast<DataType*>(&h_a)[i] = static_cast<DataType>(i + 1);
    }
    for(size_t i = 0; i < sizeof(h_b) / sizeof(DataType); ++i)
    {
        reinterpret_cast<DataType*>(&h_b)[i] = static_cast<DataType>(i + 10);
    }
    for(size_t i = 0; i < sizeof(h_c) / sizeof(DataType); ++i)
    {
        reinterpret_cast<DataType*>(&h_c)[i] = static_cast<DataType>(i + 100);
    }

    typename MmaOp::AVecType* d_a;
    typename MmaOp::BVecType* d_b;
    typename MmaOp::CVecType* d_c;
    typename MmaOp::CVecType* d_out;

    HIP_CHECK_ERROR(hipMalloc(&d_a, sizeof(h_a)));
    HIP_CHECK_ERROR(hipMalloc(&d_b, sizeof(h_b)));
    HIP_CHECK_ERROR(hipMalloc(&d_c, sizeof(h_c)));
    HIP_CHECK_ERROR(hipMalloc(&d_out, sizeof(h_out)));

    HIP_CHECK_ERROR(hipMemcpy(d_a, &h_a, sizeof(h_a), hipMemcpyHostToDevice));
    HIP_CHECK_ERROR(hipMemcpy(d_b, &h_b, sizeof(h_b), hipMemcpyHostToDevice));
    HIP_CHECK_ERROR(hipMemcpy(d_c, &h_c, sizeof(h_c), hipMemcpyHostToDevice));

    test_amdgcn_mma_exec_kernel<MmaOp><<<1, 1>>>(d_a, d_b, d_c, d_out);
    HIP_CHECK_ERROR(hipDeviceSynchronize());

    HIP_CHECK_ERROR(hipMemcpy(&h_out, d_out, sizeof(h_out), hipMemcpyDeviceToHost));

    // Check that output matches expected: a + b + c
    for(size_t i = 0; i < sizeof(h_out) / sizeof(DataType); ++i)
    {
        DataType expected = reinterpret_cast<DataType*>(&h_a)[i] +
                            reinterpret_cast<DataType*>(&h_b)[i] +
                            reinterpret_cast<DataType*>(&h_c)[i];
        EXPECT_EQ(reinterpret_cast<DataType*>(&h_out)[i], expected);
    }

    HIP_CHECK_ERROR(hipFree(d_a));
    HIP_CHECK_ERROR(hipFree(d_b));
    HIP_CHECK_ERROR(hipFree(d_c));
    HIP_CHECK_ERROR(hipFree(d_out));
}

TEST(TestAmdgcnMma, ArchUnsupportedExecDeviceOutput)
{
    using MmaOp    = DummyAmdgcnMma<amdgcn_target<>>;
    using DataType = fp32_t;

    typename MmaOp::AVecType h_a{};
    typename MmaOp::BVecType h_b{};
    typename MmaOp::CVecType h_c{};
    typename MmaOp::CVecType h_out{};

    // Fill C with known values
    for(size_t i = 0; i < sizeof(h_c) / sizeof(DataType); ++i)
    {
        reinterpret_cast<DataType*>(&h_c)[i] = static_cast<DataType>(i + 1);
    }

    typename MmaOp::AVecType* d_a;
    typename MmaOp::BVecType* d_b;
    typename MmaOp::CVecType* d_c;
    typename MmaOp::CVecType* d_out;

    HIP_CHECK_ERROR(hipMalloc(&d_a, sizeof(h_a)));
    HIP_CHECK_ERROR(hipMalloc(&d_b, sizeof(h_b)));
    HIP_CHECK_ERROR(hipMalloc(&d_c, sizeof(h_c)));
    HIP_CHECK_ERROR(hipMalloc(&d_out, sizeof(h_out)));

    HIP_CHECK_ERROR(hipMemcpy(d_a, &h_a, sizeof(h_a), hipMemcpyHostToDevice));
    HIP_CHECK_ERROR(hipMemcpy(d_b, &h_b, sizeof(h_b), hipMemcpyHostToDevice));
    HIP_CHECK_ERROR(hipMemcpy(d_c, &h_c, sizeof(h_c), hipMemcpyHostToDevice));

    test_amdgcn_mma_exec_kernel<MmaOp><<<1, 1>>>(d_a, d_b, d_c, d_out);
    HIP_CHECK_ERROR(hipDeviceSynchronize());

    HIP_CHECK_ERROR(hipMemcpy(&h_out, d_out, sizeof(h_out), hipMemcpyDeviceToHost));

    // Check that output matches input C
    for(size_t i = 0; i < sizeof(h_c) / sizeof(DataType); ++i)
    {
        EXPECT_EQ(reinterpret_cast<DataType*>(&h_out)[i], reinterpret_cast<DataType*>(&h_c)[i]);
    }

    HIP_CHECK_ERROR(hipFree(d_a));
    HIP_CHECK_ERROR(hipFree(d_b));
    HIP_CHECK_ERROR(hipFree(d_c));
    HIP_CHECK_ERROR(hipFree(d_out));
}

// Test MmaOpTraits for supported DummyAmdgcnMma, including all member variables
TEST(TestAmdgcnMma, MmaOpTraitsSupportedMembers)
{
    using MmaOp = DummyAmdgcnMma<DummyCompilerTarget>;

    // Check MmaOpTraits member variables
    EXPECT_FALSE(MmaOpTraits<MmaOp>::IsMfma);
    EXPECT_FALSE(MmaOpTraits<MmaOp>::IsWmma);
    EXPECT_TRUE(MmaOpTraits<MmaOp>::IsSupported);
}

// Test MmaOpTraits for unsupported DummyAmdgcnMma, including all member variables
TEST(TestAmdgcnMma, MmaOpTraitsUnsupportedMembers)
{
    using MmaOp = DummyAmdgcnMma<amdgcn_target<>>;

    // Check MmaOpTraits member variables
    EXPECT_FALSE(MmaOpTraits<MmaOp>::IsMfma);
    EXPECT_FALSE(MmaOpTraits<MmaOp>::IsWmma);
    EXPECT_FALSE(MmaOpTraits<MmaOp>::IsSupported);
}

// Test MmaDefaultSelector for supported DummyAmdgcnMma
TEST(TestAmdgcnMma, MmaDefaultSelectorSupported)
{
    // Direct selection of the supported dummy instruction
    using SelectedMma = typename MmaDefaultSelector<fp32_t,
                                                    fp32_t,
                                                    fp32_t,
                                                    16u,
                                                    16u,
                                                    16u,
                                                    DummyCompilerTarget,
                                                    MmaOpFamily::DENSE>::SelectedOp;
    // Should select DummyAmdgcnMma specialization
    EXPECT_TRUE((std::is_same<SelectedMma, DummyAmdgcnMma<DummyCompilerTarget>>::value));
    // OpType should be DummyOpType
    EXPECT_TRUE((std::is_same<typename SelectedMma::OpType, DummyOpType>::value));
    // IsSupported should be true
    EXPECT_TRUE(MmaOpTraits<SelectedMma>::IsSupported);
}

// Test MmaDefaultSelector for unsupported DummyAmdgcnMma
TEST(TestAmdgcnMma, MmaDefaultSelectorUnsupported)
{
    // Direct selection of the unsupported dummy instruction
    using SelectedMma = MmaDefaultSelector<fp32_t,
                                           fp32_t,
                                           fp32_t,
                                           16u,
                                           16u,
                                           16u,
                                           amdgcn_target<>,
                                           MmaOpFamily::UNDEFINED>::SelectedOp;
    // OpType should be Unsupported
    EXPECT_TRUE((std::is_same<typename SelectedMma::OpType, Unsupported>::value));
    // IsSupported should be false
    EXPECT_FALSE(MmaOpTraits<SelectedMma>::IsSupported);
    // Compile-time check that print is instantiable for the default MmaOp
    (void)static_cast<void (*)(MmaOpTraits<SelectedMma> const&)>(print);
}

// Test MmaDefaultSelector for supported DummyAmdgcnMma on WaveTile sizes other than 16x16x16
// This tests that the selector can still pick the correct MMA op even if the WaveTile sizes differ
TEST(TestAmdgcnMma, MmaDefaultSelectorSupportedWaveTile)
{
    // Select indirectly with a WaveTile size of 256x128x64
    using SelectedMma = MmaDefaultSelector<fp32_t,
                                           fp32_t,
                                           fp32_t,
                                           256u,
                                           128u,
                                           64u,
                                           DummyCompilerTarget,
                                           MmaOpFamily::DENSE>::SelectedOp;
    // Should select DummyAmdgcnMma specialization
    EXPECT_TRUE((std::is_same<SelectedMma, DummyAmdgcnMma<DummyCompilerTarget>>::value));
    // OpType should be DummyOpType
    EXPECT_TRUE((std::is_same<typename SelectedMma::OpType, DummyOpType>::value));
    // IsSupported should be true
    EXPECT_TRUE(MmaOpTraits<SelectedMma>::IsSupported);
}

// Test MmaDefaultSelector for a different WaveTile size and supported arch
TEST(TestAmdgcnMma, MmaDefaultSelectorUnsupportedWaveTile)
{
    // This should fall back to unsupported since DummyAmdgcnMma only supports 16x16x16
    using SelectedMma = MmaDefaultSelector<fp32_t,
                                           fp32_t,
                                           fp32_t,
                                           8u,
                                           8u,
                                           8u,
                                           DummyCompilerTarget,
                                           MmaOpFamily::DENSE>::SelectedOp;
    EXPECT_FALSE((std::is_same<typename SelectedMma::OpType, Unsupported>::value));
    EXPECT_TRUE(MmaOpTraits<SelectedMma>::IsSupported);
}

// Test MmaDefaultSelector for a different data type (fp16_t) and unsupported arch
TEST(TestAmdgcnMma, MmaDefaultSelectorFp16Unsupported)
{
    using SelectedMma = MmaDefaultSelector<fp16_t,
                                           fp16_t,
                                           fp16_t,
                                           16u,
                                           16u,
                                           16u,
                                           amdgcn_target<>,
                                           MmaOpFamily::UNDEFINED>::SelectedOp;
    // Should select default amdgcn_mma (Unsupported)
    EXPECT_TRUE((std::is_same<typename SelectedMma::OpType, Unsupported>::value));
    EXPECT_FALSE(MmaOpTraits<SelectedMma>::IsSupported);
}

// Test on real hardware for MmaOp selection.
// This is not a GEMM kernel, but a simple test to ensure that the selected MmaOp works correctly on
// real hardware. Assumption: inputs are all 1's The multiply-accumulate functionality can be tested
// here by looping over the k dimension and accumulating the results. They should be equal to
// WaveTileK regardless of hardware.
template <typename ADataType,
          typename BDataType,
          typename CDataType,
          uint32_t WaveTileM,
          uint32_t WaveTileN,
          uint32_t WaveTileK>
__global__ void test_accum_over_k(void* a, void* b, void* c, void* out)
{
    using Selector = MmaDefaultSelector<ADataType,
                                        BDataType,
                                        CDataType,
                                        WaveTileM,
                                        WaveTileN,
                                        WaveTileK,
                                        decltype(get_compiler_target()),
                                        MmaOpFamily::DENSE>;

    using MmaOp    = typename Selector::SelectedOp;
    using CVecType = typename MmaOp::CVecType;

    static constexpr uint32_t kIters = WaveTileK / MmaOp::kK;

    // Initialize the accumulator
    CVecType result = *reinterpret_cast<typename MmaOp::CVecType*>(c);

    // Accumulate input AxB over WaveTileK/FragK iterations
    for(uint32_t i = 0; i < kIters; ++i)
    {
        result = MmaOp::exec(*reinterpret_cast<typename MmaOp::AVecType*>(a),
                             *reinterpret_cast<typename MmaOp::BVecType*>(b),
                             result);
    }

    *reinterpret_cast<typename MmaOp::CVecType*>(out) = result;
}

// Do a live test. At minimum, there should be a solution on real hardware for F16_F16_F32_16x16x32.
TEST(TestAmdgcnMma, MmaSelector_F16_F16_F32_16x16x32_Real)
{
    int devCount;
    hipDevice_t dev;
    HIP_CHECK_ERROR(hipGetDevice(&dev));
    HIP_CHECK_ERROR(hipGetDeviceCount(&devCount));

    hipDeviceProp_t devProp;
    HIP_CHECK_ERROR(hipGetDeviceProperties(&devProp, dev));

    auto currentArchId = hip_device_prop_gcn_arch_name_to_amdgcn_target_id(devProp.gcnArchName);
    bool hasDevice     = static_cast<bool>(devCount > 0);
    int deviceWarpSize = devProp.warpSize;

    // TODO: c++20 add check for arch id
    if(!hasDevice || (currentArchId == amdgcn_target_id::HOST))
    {
        GTEST_SKIP() << "No HIP device found. Skipping test.";
    }

    using AType = fp16_t;
    using BType = fp16_t;
    using CType = fp32_t;

    // WaveTile size, also the expected fragment size (MmaTile) from the selector.
    // Note: Actual FragK might be slightly different due to hardware implementation, but the
    // test_accum_over_k kernel will loop over the K dimension to ensure that the total K is
    // correct.
    static constexpr uint32_t WaveTileM = 16;
    static constexpr uint32_t WaveTileN = 16;
    static constexpr uint32_t WaveTileK = 32;
    static constexpr uint32_t FragM     = WaveTileM;
    static constexpr uint32_t FragN     = WaveTileN;
    static constexpr uint32_t FragK     = WaveTileK;

    // Gfx11 has input data duplication and no accumulator padding (MultiplierC = 1)
    // TODO: c++20 use is_target_family_gfx11(currentArchId)
    bool isGfx11 = (currentArchId >= amdgcn_target_id::GFX1100) &&
                   (currentArchId <= amdgcn_target_id::GFX11_GENERIC);
    uint32_t MultiplierA = isGfx11 ? 2 : 1;
    uint32_t MultiplierB = isGfx11 ? 2 : 1;
    uint32_t MultiplierC = 1;

    // The number of elements per thread
    uint32_t AElements = FragM * FragK / deviceWarpSize * MultiplierA;
    uint32_t BElements = FragN * FragK / deviceWarpSize * MultiplierB;
    uint32_t CElements = FragM * FragN / deviceWarpSize * MultiplierC;

    uint32_t ASize = AElements * sizeof(AType);
    uint32_t BSize = BElements * sizeof(BType);
    uint32_t CSize = CElements * sizeof(CType);

    // Initialize A and B to all 1's, C to all 0's
    std::vector<AType> h_a(AElements, static_cast<AType>(1));
    std::vector<BType> h_b(BElements, static_cast<BType>(1));
    std::vector<CType> h_c(CElements, static_cast<CType>(0));
    std::vector<CType> h_out(CElements, static_cast<CType>(0));

    AType* d_a;
    BType* d_b;
    CType* d_c;
    CType* d_out;

    HIP_CHECK_ERROR(hipMalloc(&d_a, ASize));
    HIP_CHECK_ERROR(hipMalloc(&d_b, BSize));
    HIP_CHECK_ERROR(hipMalloc(&d_c, CSize));
    HIP_CHECK_ERROR(hipMalloc(&d_out, CSize));

    // Copy inputs to device
    HIP_CHECK_ERROR(hipMemcpy(d_a, h_a.data(), ASize, hipMemcpyHostToDevice));
    HIP_CHECK_ERROR(hipMemcpy(d_b, h_b.data(), BSize, hipMemcpyHostToDevice));
    HIP_CHECK_ERROR(hipMemcpy(d_c, h_c.data(), CSize, hipMemcpyHostToDevice));

    const auto wave_size = getDeviceWaveSize();
    test_accum_over_k<AType, BType, CType, WaveTileM, WaveTileN, WaveTileK>
        <<<1, wave_size>>>(d_a, d_b, d_c, d_out);
    HIP_CHECK_ERROR(hipDeviceSynchronize());

    HIP_CHECK_ERROR(hipMemcpy(h_out.data(), d_out, CSize, hipMemcpyDeviceToHost));

    // Output should be WaveTileK for all elements, because the inputs are all 1's
    for(size_t i = 0; i < CElements; ++i)
    {
        CType expected = static_cast<CType>(WaveTileK);

        EXPECT_NEAR(h_out[i], expected, 1e-3);
    }

    HIP_CHECK_ERROR(hipFree(d_a));
    HIP_CHECK_ERROR(hipFree(d_b));
    HIP_CHECK_ERROR(hipFree(d_c));
    HIP_CHECK_ERROR(hipFree(d_out));
}

// Do a live test. At minimum, there should be a solution on real hardware for F16_F16_F32_16x16x32
// The selector should be able to pick the correct MmaOp as a multiple of 16x16x32, even if the
// WaveTile sizes are larger than 16x16x32. This tests that the selector can handle larger WaveTile
// sizes and still select the correct MmaOp.
TEST(TestAmdgcnMma, MmaSelector_F16_F16_F32_112x112x128_Real)
{
    int devCount;
    hipDevice_t dev;
    HIP_CHECK_ERROR(hipGetDevice(&dev));
    HIP_CHECK_ERROR(hipGetDeviceCount(&devCount));

    hipDeviceProp_t devProp;
    HIP_CHECK_ERROR(hipGetDeviceProperties(&devProp, dev));

    auto currentArchId = hip_device_prop_gcn_arch_name_to_amdgcn_target_id(devProp.gcnArchName);
    bool hasDevice     = static_cast<bool>(devCount > 0);
    int deviceWarpSize = devProp.warpSize;

    // TODO: c++20 add check for arch id
    if(!hasDevice || (currentArchId == amdgcn_target_id::HOST))
    {
        GTEST_SKIP() << "No HIP device found. Skipping test.";
    }

    using AType = fp16_t;
    using BType = fp16_t;
    using CType = fp32_t;

    // WaveTile size to test for decomposition.
    // We expect the selector to pick a 16x16 WaveTile
    static constexpr uint32_t WaveTileM = 112;
    static constexpr uint32_t WaveTileN = 112;
    static constexpr uint32_t WaveTileK = 128;

    // The expected fragment size from the selector (MmaTile, multiple of 16).
    // Note: Actual FragK might be slightly different due to hardware implementation, but the
    // test_accum_over_k kernel will loop over the K dimension to ensure that the total K is
    // correct.
    static constexpr uint32_t FragM = 16;
    static constexpr uint32_t FragN = 16;
    static constexpr uint32_t FragK = 32;

    // Gfx11 has input data duplication and no accumulator padding (MultiplierC = 1)
    // TODO: c++20 use is_target_family_gfx11(currentArchId)
    bool isGfx11 = (currentArchId >= amdgcn_target_id::GFX1100) &&
                   (currentArchId <= amdgcn_target_id::GFX11_GENERIC);
    uint32_t MultiplierA = isGfx11 ? 2 : 1;
    uint32_t MultiplierB = isGfx11 ? 2 : 1;
    uint32_t MultiplierC = 1;

    // The number of elements per thread
    uint32_t AElements = FragM * FragK / deviceWarpSize * MultiplierA;
    uint32_t BElements = FragN * FragK / deviceWarpSize * MultiplierB;
    uint32_t CElements = FragM * FragN / deviceWarpSize * MultiplierC;

    uint32_t ASize = AElements * sizeof(AType);
    uint32_t BSize = BElements * sizeof(BType);
    uint32_t CSize = CElements * sizeof(CType);

    // Initialize A and B to all 1's, C to all 0's
    std::vector<AType> h_a(AElements, static_cast<AType>(1));
    std::vector<BType> h_b(BElements, static_cast<BType>(1));
    std::vector<CType> h_c(CElements, static_cast<CType>(0));
    std::vector<CType> h_out(CElements, static_cast<CType>(0));

    AType* d_a;
    BType* d_b;
    CType* d_c;
    CType* d_out;

    HIP_CHECK_ERROR(hipMalloc(&d_a, ASize));
    HIP_CHECK_ERROR(hipMalloc(&d_b, BSize));
    HIP_CHECK_ERROR(hipMalloc(&d_c, CSize));
    HIP_CHECK_ERROR(hipMalloc(&d_out, CSize));

    // Copy inputs to device
    HIP_CHECK_ERROR(hipMemcpy(d_a, h_a.data(), ASize, hipMemcpyHostToDevice));
    HIP_CHECK_ERROR(hipMemcpy(d_b, h_b.data(), BSize, hipMemcpyHostToDevice));
    HIP_CHECK_ERROR(hipMemcpy(d_c, h_c.data(), CSize, hipMemcpyHostToDevice));

    const auto wave_size = getDeviceWaveSize();
    test_accum_over_k<AType, BType, CType, WaveTileM, WaveTileN, WaveTileK>
        <<<1, wave_size>>>(d_a, d_b, d_c, d_out);
    HIP_CHECK_ERROR(hipDeviceSynchronize());

    HIP_CHECK_ERROR(hipMemcpy(h_out.data(), d_out, CSize, hipMemcpyDeviceToHost));

    // Output should be WaveTileK for all elements, because the inputs are all 1's
    for(size_t i = 0; i < CElements; ++i)
    {
        CType expected = static_cast<CType>(WaveTileK);

        EXPECT_NEAR(h_out[i], expected, 1e-3);
    }

    HIP_CHECK_ERROR(hipFree(d_a));
    HIP_CHECK_ERROR(hipFree(d_b));
    HIP_CHECK_ERROR(hipFree(d_c));
    HIP_CHECK_ERROR(hipFree(d_out));
}
