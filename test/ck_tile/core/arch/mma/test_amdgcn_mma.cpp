// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <hip/hip_runtime.h>
#include <gtest/gtest.h>
#include "ck_tile/core/arch/arch.hpp"
#include "ck_tile/core/arch/mma/amdgcn_mma.hpp"
#include "ck_tile/core/arch/mma/mma_selector.hpp"
#include "ck_tile/core/utility/type_traits.hpp"
#include "ck_tile/host/hip_check_error.hpp"

using namespace ck_tile;
using namespace ck_tile::core::arch;
using namespace ck_tile::core::arch::mma;

// Dummy control flags for testing
struct DummyCtrlFlags
{
};

// Define a new dummy architecture ID for testing
constexpr uint32_t DummyGfxTargetId = 55555u;

// Define a new dummy op type
struct DummyOpType;

// Define an architecture test to include the dummy gfx target ID
template <uint32_t GfxTargetId>
struct is_dummy_arch_id : is_any_value_of<uint32_t, GfxTargetId, DummyGfxTargetId>
{
};

// Convenience evaluator
template <uint32_t GfxTargetId>
static constexpr bool is_dummy_arch_id_v = is_dummy_arch_id<GfxTargetId>::value;

// Enable if for dummy architecture ID
template <uint32_t GfxTargetId>
using enable_if_dummy_arch_id_t = std::enable_if_t<is_dummy_arch_id_v<GfxTargetId>>;

// Specialization of amdgcn_mma for a supported dummy architecture.
// This way, we don't have to worry about underlying architectural details,
// and can focus on testing the mechanism of selecting supported vs unsupported architectures.
template <uint32_t GfxTargetId>
struct amdgcn_mma<fp32_t,
                  fp32_t,
                  fp32_t,
                  16u,
                  16u,
                  16u,
                  DummyCtrlFlags,
                  GfxTargetId,
                  enable_if_dummy_arch_id_t<GfxTargetId>>
{
    // Mfma operation type
    using OpType = DummyOpType;

    // Register types
    using AVecType = ext_vector_t<fp32_t, 4>;
    using BVecType = ext_vector_t<fp32_t, 4>;
    using CVecType = ext_vector_t<fp32_t, 4>;

    // Layout constants
    static constexpr index_t kAMBlock = 1;
    static constexpr index_t kBNBlock = 2;

    static constexpr index_t kAMLane     = 3;
    static constexpr index_t kBNLane     = 4;
    static constexpr index_t kABKLane    = 5;
    static constexpr index_t kABKPerLane = 6;

    static constexpr index_t kCMLane     = 7;
    static constexpr index_t kCNLane     = 8;
    static constexpr index_t kCM0PerLane = 9;
    static constexpr index_t kCM1PerLane = 10;

    CK_TILE_DEVICE static CVecType
    exec(AVecType const& regsA, BVecType const& regsB, CVecType const& regsC)
    {
        return regsA + regsB + regsC; // Simple operation for testing
    }
};

// Have an alias so we can test supported arch vs unsupported arch
template <uint32_t GfxTargetId>
using DummyAmdgcnMma =
    amdgcn_mma<fp32_t, fp32_t, fp32_t, 16u, 16u, 16u, DummyCtrlFlags, GfxTargetId>;

/*! @struct MmaDefaultSelector
 * @brief For dummy Id only, instantiate tests for both MFMA and WMMA selectors so we can them both
 * @tparam ADataType Data type of matrix A
 * @tparam BDataType Data type of matrix B
 * @tparam CDataType Data type of the accumulator
 * @tparam FragM Size of the M dimension of the fragment to decompose
 * @tparam FragN Size of the N dimension of the fragment to decompose
 * @tparam FragK Size of the K dimension of the fragment to decompose
 * @tparam GfxTargetId Target architecture ID
 */
template <typename ADataType,
          typename BDataType,
          typename CDataType,
          uint32_t FragM,
          uint32_t FragN,
          uint32_t FragK,
          uint32_t GfxTargetId>
struct MmaDefaultSelector<ADataType,
                          BDataType,
                          CDataType,
                          FragM,
                          FragN,
                          FragK,
                          GfxTargetId,
                          enable_if_dummy_arch_id_t<GfxTargetId>>
{
    using SelectedOp = DummyAmdgcnMma<GfxTargetId>;
};

// Test case for supported architecture
TEST(TestAmdgcnMma, ArchSupported)
{
    // Instantiate MmaOp with the dummy supported GfxTargetId
    using MmaOp = DummyAmdgcnMma<DummyGfxTargetId>;

    EXPECT_TRUE((!std::is_same_v<typename MmaOp::OpType, Unsupported>));

    // Additional tests for DummyArchSupported: check all member variables and types

    // Check OpType
    EXPECT_TRUE(
        (std::is_same<typename MmaOp::OpType, DummyOpType>::value)); // OpType is DummyOpType

    // Check AVecType, BVecType, CVecType
    EXPECT_TRUE((std::is_same<typename MmaOp::AVecType, ext_vector_t<fp32_t, 4>>::value));
    EXPECT_TRUE((std::is_same<typename MmaOp::BVecType, ext_vector_t<fp32_t, 4>>::value));
    EXPECT_TRUE((std::is_same<typename MmaOp::CVecType, ext_vector_t<fp32_t, 4>>::value));

    // Check layout constants
    EXPECT_EQ(MmaOp::kAMBlock, 1);
    EXPECT_EQ(MmaOp::kBNBlock, 2);

    EXPECT_EQ(MmaOp::kAMLane, 3);
    EXPECT_EQ(MmaOp::kBNLane, 4);
    EXPECT_EQ(MmaOp::kABKLane, 5);
    EXPECT_EQ(MmaOp::kABKPerLane, 6);

    EXPECT_EQ(MmaOp::kCMLane, 7);
    EXPECT_EQ(MmaOp::kCNLane, 8);
    EXPECT_EQ(MmaOp::kCM0PerLane, 9);
    EXPECT_EQ(MmaOp::kCM1PerLane, 10);
}

// Test case for unsupported architecture
TEST(TestAmdgcnMma, ArchUnsupported)
{
    // Use an unsupported GfxTargetId (e.g., 1)
    constexpr uint32_t UnsupportedGfxTargetId = 1;

    // Instantiate MmaOp with the dummy unsupported GfxTargetId
    using MmaOp = DummyAmdgcnMma<UnsupportedGfxTargetId>;

    // OpType should be Unsupported
    EXPECT_TRUE((std::is_same<typename MmaOp::OpType, Unsupported>::value));

    // AVecType, BVecType, CVecType should match default
    EXPECT_TRUE((std::is_same<typename MmaOp::AVecType, ext_vector_t<fp32_t, 1>>::value));
    EXPECT_TRUE((std::is_same<typename MmaOp::BVecType, ext_vector_t<fp32_t, 1>>::value));
    EXPECT_TRUE((std::is_same<typename MmaOp::CVecType, ext_vector_t<fp32_t, 1>>::value));

    // Layout constants should match default values (typically 0)
    EXPECT_EQ(MmaOp::kAMBlock, 0);
    EXPECT_EQ(MmaOp::kBNBlock, 0);

    EXPECT_EQ(MmaOp::kAMLane, 0);
    EXPECT_EQ(MmaOp::kBNLane, 0);
    EXPECT_EQ(MmaOp::kABKLane, 0);
    EXPECT_EQ(MmaOp::kABKPerLane, 0);

    EXPECT_EQ(MmaOp::kCMLane, 0);
    EXPECT_EQ(MmaOp::kCNLane, 0);
    EXPECT_EQ(MmaOp::kCM0PerLane, 0);
    EXPECT_EQ(MmaOp::kCM1PerLane, 0);
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
    using MmaOp    = DummyAmdgcnMma<DummyGfxTargetId>;
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
    constexpr uint32_t UnsupportedGfxTargetId = 1;
    using MmaOp                               = DummyAmdgcnMma<UnsupportedGfxTargetId>;
    using DataType                            = fp32_t;

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

#include "ck_tile/core/arch/mma/mma_traits.hpp"

// Test MmaOpParams for supported DummyAmdgcnMma, including all member variables
TEST(TestAmdgcnMma, MmaOpParamsTraitsSupportedMembers)
{
    using MmaOp  = DummyAmdgcnMma<DummyGfxTargetId>;
    using Traits = MmaOpParams<MmaOp>;

    // Check MmaOpParams members
    EXPECT_TRUE((std::is_same<typename Traits::ADataType, fp32_t>::value));
    EXPECT_TRUE((std::is_same<typename Traits::BDataType, fp32_t>::value));
    EXPECT_TRUE((std::is_same<typename Traits::CDataType, fp32_t>::value));
    EXPECT_EQ(Traits::BlockM, 16u);
    EXPECT_EQ(Traits::BlockN, 16u);
    EXPECT_EQ(Traits::BlockK, 16u);
    EXPECT_TRUE((std::is_same<typename Traits::CtrlFlags, DummyCtrlFlags>::value));
}

// Test MmaOpParams for unsupported DummyAmdgcnMma, including all member variables
TEST(TestAmdgcnMma, MmaOpParamsUnsupportedMembers)
{
    constexpr uint32_t UnsupportedGfxTargetId = 1;
    using MmaOp                               = DummyAmdgcnMma<UnsupportedGfxTargetId>;
    using Traits                              = MmaOpParams<MmaOp>;

    // Check MmaOpParams members
    EXPECT_TRUE((std::is_same<typename Traits::ADataType, fp32_t>::value));
    EXPECT_TRUE((std::is_same<typename Traits::BDataType, fp32_t>::value));
    EXPECT_TRUE((std::is_same<typename Traits::CDataType, fp32_t>::value));
    EXPECT_EQ(Traits::BlockM, 16u);
    EXPECT_EQ(Traits::BlockN, 16u);
    EXPECT_EQ(Traits::BlockK, 16u);
    EXPECT_TRUE((std::is_same<typename Traits::CtrlFlags, DummyCtrlFlags>::value));
}

// Test MmaOpTraits for supported DummyAmdgcnMma, including all member variables
TEST(TestAmdgcnMma, MmaOpTraitsSupportedMembers)
{
    using MmaOp  = DummyAmdgcnMma<DummyGfxTargetId>;
    using Traits = MmaOpTraits<MmaOp>;

    // Check MmaOpTraits member variables
    EXPECT_TRUE((std::is_same<typename Traits::OpType, DummyOpType>::value));
    EXPECT_TRUE((std::is_same<typename Traits::AVecType, ext_vector_t<fp32_t, 4>>::value));
    EXPECT_TRUE((std::is_same<typename Traits::BVecType, ext_vector_t<fp32_t, 4>>::value));
    EXPECT_TRUE((std::is_same<typename Traits::CVecType, ext_vector_t<fp32_t, 4>>::value));
    EXPECT_EQ(Traits::kAMBlock, 1);
    EXPECT_EQ(Traits::kBNBlock, 2);
    EXPECT_EQ(Traits::kAMLane, 3);
    EXPECT_EQ(Traits::kBNLane, 4);
    EXPECT_EQ(Traits::kABKLane, 5);
    EXPECT_EQ(Traits::kABKPerLane, 6);
    EXPECT_EQ(Traits::kCMLane, 7);
    EXPECT_EQ(Traits::kCNLane, 8);
    EXPECT_EQ(Traits::kCM0PerLane, 9);
    EXPECT_EQ(Traits::kCM1PerLane, 10);
    EXPECT_FALSE(Traits::IsMfma);
    EXPECT_FALSE(Traits::IsWmma);
    EXPECT_TRUE(Traits::IsSupported);
}

// Test MmaOpTraits for unsupported DummyAmdgcnMma, including all member variables
TEST(TestAmdgcnMma, MmaOpTraitsUnsupportedMembers)
{
    constexpr uint32_t UnsupportedGfxTargetId = 1;
    using MmaOp                               = DummyAmdgcnMma<UnsupportedGfxTargetId>;
    using Traits                              = MmaOpTraits<MmaOp>;

    // Check MmaOpTraits member variables
    EXPECT_TRUE((std::is_same<typename Traits::OpType, Unsupported>::value));
    EXPECT_TRUE((std::is_same<typename Traits::AVecType, ext_vector_t<fp32_t, 1>>::value));
    EXPECT_TRUE((std::is_same<typename Traits::BVecType, ext_vector_t<fp32_t, 1>>::value));
    EXPECT_TRUE((std::is_same<typename Traits::CVecType, ext_vector_t<fp32_t, 1>>::value));
    EXPECT_EQ(Traits::kAMBlock, 0);
    EXPECT_EQ(Traits::kBNBlock, 0);
    EXPECT_EQ(Traits::kAMLane, 0);
    EXPECT_EQ(Traits::kBNLane, 0);
    EXPECT_EQ(Traits::kABKLane, 0);
    EXPECT_EQ(Traits::kABKPerLane, 0);
    EXPECT_EQ(Traits::kCMLane, 0);
    EXPECT_EQ(Traits::kCNLane, 0);
    EXPECT_EQ(Traits::kCM0PerLane, 0);
    EXPECT_EQ(Traits::kCM1PerLane, 0);
    EXPECT_FALSE(Traits::IsMfma);
    EXPECT_FALSE(Traits::IsWmma);
    EXPECT_FALSE(Traits::IsSupported);
}

// Test MmaDefaultSelector for supported DummyAmdgcnMma
TEST(TestAmdgcnMma, MmaDefaultSelectorSupported)
{
    // Direct selection of the supported dummy instruction
    using SelectedMma =
        typename MmaDefaultSelector<fp32_t, fp32_t, fp32_t, 16u, 16u, 16u, DummyGfxTargetId>::
            SelectedOp;
    // Should select DummyAmdgcnMma specialization
    EXPECT_TRUE((std::is_same<SelectedMma, DummyAmdgcnMma<DummyGfxTargetId>>::value));
    // OpType should be DummyOpType
    EXPECT_TRUE((std::is_same<typename SelectedMma::OpType, DummyOpType>::value));
    // IsSupported should be true
    EXPECT_TRUE(MmaOpTraits<SelectedMma>::IsSupported);
}

// Test MmaDefaultSelector for unsupported DummyAmdgcnMma
TEST(TestAmdgcnMma, MmaDefaultSelectorUnsupported)
{
    // Direct selection of the unsupported dummy instruction
    constexpr uint32_t UnsupportedGfxTargetId = 1;
    using SelectedMma =
        MmaDefaultSelector<fp32_t, fp32_t, fp32_t, 16u, 16u, 16u, UnsupportedGfxTargetId>::
            SelectedOp;
    // OpType should be Unsupported
    EXPECT_TRUE((std::is_same<typename SelectedMma::OpType, Unsupported>::value));
    // IsSupported should be false
    EXPECT_FALSE(MmaOpTraits<SelectedMma>::IsSupported);
}

// Test MmaDefaultSelector for supported DummyAmdgcnMma on fragment sizes other than 16x16x16
// This tests that the selector can still pick the correct MMA op even if the fragment sizes differ
TEST(TestAmdgcnMma, MmaDefaultSelectorSupportedFragment)
{
    // Select indirectly with a fragment size of 256x128x64
    using SelectedMma =
        MmaDefaultSelector<fp32_t, fp32_t, fp32_t, 256u, 128u, 64u, DummyGfxTargetId>::SelectedOp;
    // Should select DummyAmdgcnMma specialization
    EXPECT_TRUE((std::is_same<SelectedMma, DummyAmdgcnMma<DummyGfxTargetId>>::value));
    // OpType should be DummyOpType
    EXPECT_TRUE((std::is_same<typename SelectedMma::OpType, DummyOpType>::value));
    // IsSupported should be true
    EXPECT_TRUE(MmaOpTraits<SelectedMma>::IsSupported);
}

// Test MmaDefaultSelector for a different block size and supported arch
TEST(TestAmdgcnMma, MmaDefaultSelectorUnsupportedFragment)
{
    // This should fall back to unsupported since DummyAmdgcnMma only supports 16x16x16
    using SelectedMma =
        MmaDefaultSelector<fp32_t, fp32_t, fp32_t, 8u, 8u, 8u, DummyGfxTargetId>::SelectedOp;
    EXPECT_FALSE((std::is_same<typename SelectedMma::OpType, Unsupported>::value));
    EXPECT_TRUE(MmaOpTraits<SelectedMma>::IsSupported);
}

// Test MmaDefaultSelector for a different data type (fp16_t) and unsupported arch
TEST(TestAmdgcnMma, MmaDefaultSelectorFp16Unsupported)
{
    constexpr uint32_t UnsupportedGfxTargetId = 1;
    using SelectedMma =
        MmaDefaultSelector<fp16_t, fp16_t, fp16_t, 16u, 16u, 16u, UnsupportedGfxTargetId>::
            SelectedOp;
    // Should select default amdgcn_mma (Unsupported)
    EXPECT_TRUE((std::is_same<typename SelectedMma::OpType, Unsupported>::value));
    EXPECT_FALSE(MmaOpTraits<SelectedMma>::IsSupported);
}

// Kernel to test amdgcn_mma::exec on device
template <typename ADataType,
          typename BDataType,
          typename CDataType,
          uint32_t FragM,
          uint32_t FragN,
          uint32_t FragK>
__global__ void test_real_mma_selector_kernel(void* a, void* b, void* c, void* out)
{
    using Selector = MmaDefaultSelector<ADataType,
                                        BDataType,
                                        CDataType,
                                        FragM,
                                        FragN,
                                        FragK,
                                        get_target_arch_id()>;

    using MmaOp     = typename Selector::SelectedOp;
    using MmaTraits = MmaOpTraits<MmaOp>;

    using CVecType = typename MmaOp::CVecType;

    static constexpr uint32_t kIters = FragK / MmaTraits::BlockK;

    // Initialize the accumulator
    CVecType result = *reinterpret_cast<typename MmaOp::CVecType*>(c);

    // Accumulate input AxB over FragK/BlockK iterations
    for(uint32_t i = 0; i < kIters; ++i)
    {
        result = MmaOp::exec(*reinterpret_cast<typename MmaOp::AVecType*>(a),
                             *reinterpret_cast<typename MmaOp::BVecType*>(b),
                             result);
    }

    *reinterpret_cast<typename MmaOp::CVecType*>(out) = result;
}

// Do a live test. At minimum, there should be a solution on real hardware for F16_F16_F32_16x16x32
TEST(TestAmdgcnMma, F16_F16_F32_16x16x32_Real)
{
    int devCount;
    hipDevice_t dev;
    HIP_CHECK_ERROR(hipGetDevice(&dev));
    HIP_CHECK_ERROR(hipGetDeviceCount(&devCount));

    hipDeviceProp_t devProp;
    HIP_CHECK_ERROR(hipGetDeviceProperties(&devProp, dev));

    std::string deviceName(devProp.gcnArchName);

    bool isGfx11 = (deviceName.find("gfx1100") != std::string::npos) ||
                   (deviceName.find("gfx1101") != std::string::npos) ||
                   (deviceName.find("gfx1102") != std::string::npos) ||
                   (deviceName.find("gfx1153") != std::string::npos);

    bool hasDevice     = static_cast<bool>(devCount > 0);
    int deviceWarpSize = devProp.warpSize;

    if(!hasDevice)
    {
        GTEST_SKIP() << "No HIP device found. Skipping test.";
    }

    using AType = fp16_t;
    using BType = fp16_t;
    using CType = fp32_t;

    static constexpr uint32_t BlockM = 16;
    static constexpr uint32_t BlockN = 16;
    static constexpr uint32_t BlockK = 32;

    // Gfx11 has data duplication and accumulator padding
    uint32_t MultiplierA = isGfx11 ? 2 : 1;
    uint32_t MultiplierB = isGfx11 ? 2 : 1;
    uint32_t MultiplierC = ((isGfx11 && (sizeof(CType) < sizeof(float)))) ? 2 : 1;

    // The number of elements per thread
    uint32_t AElements = BlockM * BlockK / deviceWarpSize * MultiplierA;
    uint32_t BElements = BlockN * BlockK / deviceWarpSize * MultiplierB;
    uint32_t CElements = BlockM * BlockN / deviceWarpSize * MultiplierC;

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

    // Need at least 1 WG with 64 threads to get defined MFMA/WMMA behaviour
    test_real_mma_selector_kernel<AType, BType, CType, BlockM, BlockN, BlockK>
        <<<1, 64>>>(d_a, d_b, d_c, d_out);
    HIP_CHECK_ERROR(hipDeviceSynchronize());

    HIP_CHECK_ERROR(hipMemcpy(h_out.data(), d_out, CSize, hipMemcpyDeviceToHost));

    // Output should be BlockK for all elements, because the inputs are all 1's
    for(size_t i = 0; i < CElements; ++i)
    {
        // All threads write to the same values and they will be exactly the same as BlockK
        CType expected = static_cast<CType>(BlockK);

        EXPECT_NEAR(h_out[i], expected, 1e-3);
    }

    HIP_CHECK_ERROR(hipFree(d_a));
    HIP_CHECK_ERROR(hipFree(d_b));
    HIP_CHECK_ERROR(hipFree(d_c));
    HIP_CHECK_ERROR(hipFree(d_out));
}
