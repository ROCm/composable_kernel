// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "test_cshuffle_epilogue_util.hpp"

#include "ck_tile/host/device_prop.hpp"
#include "ck_tile/ops/epilogue/tdm_epilogue.hpp"
#include "ck_tile/ops/epilogue/tdm_multi_d_epilogue.hpp"
#include "ck_tile/ops/gemm.hpp"

#include <gtest/gtest.h>

#include <cstdlib>

namespace ck_tile {

// Overwrites smem right after Epilogue returns, as the next persistent tile's producers do.
// d_data backs the D tensors of a multi-D epilogue: M x N row-major, read once per D.
template <typename Epilogue, index_t M, index_t N>
__global__ void test_tdm_epilogue_lds_reuse_kernel(
    const typename Epilogue::Problem::AccDataType* __restrict__ input_data,
    typename Epilogue::Problem::ODataType* __restrict__ output_data,
    const typename Epilogue::Problem::ODataType* __restrict__ d_data,
    typename Epilogue::Problem::ODataType sentinel)
{
    // TDM exists only on gfx125; discard the body on the other targets of a mixed build.
    if constexpr(CK_TILE_ENABLE_TDM_FEATURE != 0)
    {
        using Problem     = typename Epilogue::Problem;
        using AccDataType = typename Problem::AccDataType;
        using ODataType   = typename Problem::ODataType;

        constexpr index_t kMPerBlock = Problem::kMPerBlock;
        constexpr index_t kNPerBlock = Problem::kNPerBlock;

        alignas(16) __shared__ char smem[Epilogue::GetSmemSize()];

        using WG = WarpGemmDispatcher<typename Epilogue::ATypeToUse,
                                      typename Epilogue::BTypeToUse,
                                      AccDataType,
                                      Problem::MPerXdl,
                                      Problem::NPerXdl,
                                      Problem::KPerXdl,
                                      Problem::isCTransposed>;

        constexpr index_t MIterPerWarp = kMPerBlock / (Problem::MWave * Problem::MPerXdl);
        constexpr index_t NIterPerWarp = kNPerBlock / (Problem::NWave * Problem::NPerXdl);

        constexpr auto c_block_outer_dstr_encoding = tile_distribution_encoding<
            sequence<>,
            tuple<sequence<MIterPerWarp, Problem::MWave>, sequence<NIterPerWarp, Problem::NWave>>,
            tuple<sequence<1, 2>>,
            tuple<sequence<1, 1>>,
            sequence<1, 2>,
            sequence<0, 0>>{};

        constexpr auto acc_distribution =
            make_static_tile_distribution(detail::make_embed_tile_distribution_encoding(
                c_block_outer_dstr_encoding, typename WG::CWarpDstrEncoding{}));
        auto acc_tile = make_static_distributed_tensor<AccDataType>(acc_distribution);

        auto input_tensor_view =
            make_naive_tensor_view<address_space_enum::global>(const_cast<AccDataType*>(input_data),
                                                               make_tuple(kMPerBlock, kNPerBlock),
                                                               make_tuple(kNPerBlock, 1),
                                                               number<1>{},
                                                               number<1>{});
        auto input_tile_window =
            make_tile_window(input_tensor_view,
                             make_tuple(number<kMPerBlock>{}, number<kNPerBlock>{}),
                             {0, 0},
                             acc_distribution);
        load_tile(acc_tile, input_tile_window);

        const index_t i_m = static_cast<index_t>(blockIdx.x) * kMPerBlock;

        auto output_tensor_view =
            make_naive_tensor_view<address_space_enum::global>(output_data,
                                                               make_tuple(M, N),
                                                               make_tuple(N, 1),
                                                               number<Epilogue::GetVectorSizeC()>{},
                                                               number<1>{});
        auto output_tile_window = make_tile_window(
            output_tensor_view, make_tuple(number<kMPerBlock>{}, number<kNPerBlock>{}), {i_m, 0});

        const auto ds_windows = generate_tuple(
            [&](auto) {
                auto d_tensor_view = make_naive_tensor_view<address_space_enum::global>(
                    const_cast<ODataType*>(d_data),
                    make_tuple(M, N),
                    make_tuple(N, 1),
                    number<1>{},
                    number<1>{});
                return make_tile_window(d_tensor_view,
                                        make_tuple(number<kMPerBlock>{}, number<kNPerBlock>{}),
                                        {i_m, 0});
            },
            number<Problem::NumDTensor>{});
        Epilogue{}(output_tile_window, acc_tile, ds_windows, smem);

        // Stands in for the persistent loop-top barrier.
        block_sync_lds();
        // 16-byte stores, last row first: what the TDM store reads last is overwritten first.
        constexpr index_t kFillElems = 16 / static_cast<index_t>(sizeof(ODataType));
        using fill_t                 = ext_vector_t<ODataType, kFillElems>;
        constexpr index_t kNumFills =
            Epilogue::GetSmemSize() / static_cast<index_t>(sizeof(fill_t));
        static_assert(kNumFills % Problem::kBlockSize == 0, "fill must split evenly");
        const fill_t fill = sentinel;
        // LDS cast by hand: CK_TILE_LDS_ADDR is empty; volatile blocks address-space inference.
        auto* lds = reinterpret_cast<volatile fill_t __attribute__((address_space(3)))*>(
            reinterpret_cast<uintptr_t>(smem));
        // volatile keeps the unread stores in order; on LDS it costs no s_wait.
        const index_t last_slice = kNumFills - 1 - static_cast<index_t>(threadIdx.x);
        static_for<0, kNumFills / Problem::kBlockSize, 1>{}(
            [&](auto k) { lds[last_slice - k.value * Problem::kBlockSize] = fill; });
    }
    else
    {
        ignore = input_data;
        ignore = output_data;
        ignore = d_data;
        ignore = sentinel;
    }
}

} // namespace ck_tile

using namespace ck_tile;

constexpr index_t kTileM = 64;
constexpr index_t kTileN = 64;
// Enough concurrent TDM stores that some are still in flight when the waves overwrite LDS.
constexpr index_t kNumBlocks = 256;
constexpr index_t kM         = kTileM * kNumBlocks;
constexpr index_t kN         = kTileN;

// Same epilogue problem as the persistent RCR F16 producer/consumer GEMM test config.
template <typename DsDataType, typename DsLayout, typename CDElementwise>
using TdmProblemWithDs = CShuffleEpilogueProblem<half_t,
                                                 half_t,
                                                 DsDataType,
                                                 float,
                                                 half_t,
                                                 DsLayout,
                                                 tensor_layout::gemm::RowMajor,
                                                 CDElementwise,
                                                 kTileM,
                                                 kTileN,
                                                 2,   // MWave
                                                 2,   // NWave
                                                 16,  // MPerXdl
                                                 16,  // NPerXdl
                                                 32,  // KPerXdl
                                                 true // isCTransposed
                                                 >;

using TdmProblem = TdmProblemWithDs<tuple<>, tuple<>, element_wise::PassThrough>;
// One zero D: E = C + 0, so the expected output is the same as the plain epilogue's.
using TdmMultiDProblem =
    TdmProblemWithDs<tuple<half_t>, tuple<tensor_layout::gemm::RowMajor>, element_wise::MultiDAdd>;

// True when the device lacks TDM. With CK_TILE_REQUIRE_GFX125=1 it has also recorded a
// failure, so the caller's GTEST_SKIP still reports FAILED (gtest: Skipped() == !Failed() && skip).
[[nodiscard]] bool skip_without_gfx125()
{
    if(is_gfx125_supported())
    {
        return false;
    }
    const char* required = std::getenv("CK_TILE_REQUIRE_GFX125");
    if(required != nullptr && required[0] == '1')
    {
        ADD_FAILURE() << "CK_TILE_REQUIRE_GFX125=1 but the device reports '" << get_device_name()
                      << "'";
    }
    return true;
}

// Runs Epilogue on kNumBlocks tiles, overwrites LDS right after it returns, and returns the
// number of output elements that do not match the input tile bit for bit.
template <typename Epilogue>
index_t count_mismatches_after_lds_reuse()
{
    using Problem = typename Epilogue::Problem;

    auto host_input = generate_unique_fp16_input<float, kTileM, kTileN>();
    DeviceMem input_buf(host_input.get_element_space_size_in_bytes());
    input_buf.ToDevice(host_input.data());

    HostTensor<half_t> host_output({kM, kN});
    host_output.SetZero();
    DeviceMem output_buf(host_output.get_element_space_size_in_bytes());
    output_buf.ToDevice(host_output.data());

    // Zero D: adding it is exact, so it cannot mask or create a mismatch.
    HostTensor<half_t> host_d({kM, kN});
    host_d.SetZero();
    DeviceMem d_buf(host_d.get_element_space_size_in_bytes());
    d_buf.ToDevice(host_d.data());

    int device    = 0;
    int warp_size = 0;
    HIP_CHECK_ERROR(hipGetDevice(&device));
    HIP_CHECK_ERROR(hipDeviceGetAttribute(&warp_size, hipDeviceAttributeWarpSize, device));
    const dim3 grid_size(kNumBlocks, 1, 1);
    const dim3 block_size(Problem::MWave * Problem::NWave * warp_size, 1, 1);

    // Inputs are positive fp16 normals, so -1 can only come from the overwritten LDS.
    const half_t sentinel = type_convert<half_t>(-1.0F);
    test_tdm_epilogue_lds_reuse_kernel<Epilogue, kM, kN>
        <<<grid_size, block_size>>>(static_cast<const float*>(input_buf.GetDeviceBuffer()),
                                    static_cast<half_t*>(output_buf.GetDeviceBuffer()),
                                    static_cast<const half_t*>(d_buf.GetDeviceBuffer()),
                                    sentinel);
    HIP_CHECK_ERROR(hipGetLastError());
    HIP_CHECK_ERROR(hipDeviceSynchronize());
    output_buf.FromDevice(host_output.data());

    // Bit-exact compare: inputs are fp16 values, so the epilogue's cast to half_t is lossless.
    index_t mismatches = 0;
    for(index_t m = 0; m < kM; ++m)
    {
        for(index_t n = 0; n < kN; ++n)
        {
            if(ck_tile::bit_cast<uint32_t>(type_convert<float>(host_output(m, n))) !=
               ck_tile::bit_cast<uint32_t>(host_input(m % kTileM, n)))
            {
                ++mismatches;
            }
        }
    }
    return mismatches;
}

TEST(TdmEpilogue, OverwritingLdsAfterReturnKeepsC)
{
    if(skip_without_gfx125())
    {
        GTEST_SKIP() << "TDM requires gfx1250; device reports '" << get_device_name() << "'";
    }
    EXPECT_EQ(count_mismatches_after_lds_reuse<TdmEpilogue<TdmProblem>>(), 0)
        << "TdmEpilogue returned while its TDM store still read LDS";
}

TEST(TdmMultiDEpilogue, OverwritingLdsAfterReturnKeepsC)
{
    if(skip_without_gfx125())
    {
        GTEST_SKIP() << "TDM requires gfx1250; device reports '" << get_device_name() << "'";
    }
    EXPECT_EQ(count_mismatches_after_lds_reuse<TdmMultiDEpilogue<TdmMultiDProblem>>(), 0)
        << "TdmMultiDEpilogue returned while its TDM store still read LDS";
}

// TdmEpilogue's split-K trap is deliberately not exercised: a device trap aborts the process
// through the HSA queue-error handler, a gtest death test would have to fork after HIP init,
// and on gfx1250 boxes without GPU recovery a trapped queue can wedge the device. The host
// rejection and the EpilogueSupportsSplitK elision below are what keep the trap unreachable.

// GemmKernel with a TDM pipeline and TdmEpilogue: the TDM store overwrites C, so a k_batch > 1
// launch would leave one split's partial sum. fp32 E is the case that used to slip through --
// fp16/bf16 E was already rejected by the odd C vector size. Host-only: nothing is launched and
// the null pointers are never dereferenced.
template <template <typename...> class Pipeline>
struct TdmGemmWithFp32E
{
    using Row = tensor_layout::gemm::RowMajor;
    using Col = tensor_layout::gemm::ColumnMajor;
    using Shape =
        TileGemmShape<sequence<kTileM, kTileN, 64>, sequence<2, 2, 1>, sequence<16, 16, 32>>;
    using Partitioner     = GemmSpatiallyLocalTilePartitioner<Shape, 8, 4>;
    using Traits          = TileGemmUniversalTraits<false, false, false, true, Row, Col, Row>;
    using Problem         = UniversalGemmPipelineProblem<half_t, half_t, float, Shape, Traits>;
    using EpilogueProblem = CShuffleEpilogueProblem<half_t,
                                                    half_t,
                                                    tuple<>,
                                                    float,
                                                    float,
                                                    tuple<>,
                                                    Row,
                                                    element_wise::PassThrough,
                                                    kTileM,
                                                    kTileN,
                                                    2,    // MWave
                                                    2,    // NWave
                                                    16,   // MPerXdl
                                                    16,   // NPerXdl
                                                    32,   // KPerXdl
                                                    false // isCTransposed
                                                    >;
    using Kernel = GemmKernel<Partitioner, Pipeline<Problem>, TdmEpilogue<EpilogueProblem>>;

    // K = 256 is 8 K-steps of 32, so k_batch = 2 clears the step-count guard: only the
    // epilogue's split-K support can reject it.
    static bool IsSupported(index_t k_batch)
    {
        constexpr index_t M = 4 * kTileM;
        constexpr index_t N = 4 * kTileN;
        constexpr index_t K = 256;
        const GemmHostArgs args(nullptr, nullptr, nullptr, k_batch, M, N, K, K, K, N);
        return Kernel::IsSupportedArgument(Kernel::MakeKernelArgs(args));
    }
};

// The constant also compiles out the device atomic_add arm, which would reach TdmEpilogue's trap.
TEST(TdmEpilogue, GemmKernelTdmV1RejectsSplitKWithFp32E)
{
    using Gemm = TdmGemmWithFp32E<GemmPipelineAgBgCrCompTDMV1>;
    EXPECT_FALSE(Gemm::Kernel::UniversalGemmKernel::EpilogueSupportsSplitK);
    EXPECT_TRUE(Gemm::IsSupported(1));
    EXPECT_FALSE(Gemm::IsSupported(2));
}

TEST(TdmEpilogue, GemmKernelTdmV2RejectsSplitKWithFp32E)
{
    using Gemm = TdmGemmWithFp32E<GemmPipelineAgBgCrCompTDMV2>;
    EXPECT_FALSE(Gemm::Kernel::UniversalGemmKernel::EpilogueSupportsSplitK);
    EXPECT_TRUE(Gemm::IsSupported(1));
    EXPECT_FALSE(Gemm::IsSupported(2));
}

int main(int argc, char** argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
