// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "test_cshuffle_epilogue_util.hpp"

#include "ck_tile/host/device_prop.hpp"
#include "ck_tile/ops/epilogue/tdm_epilogue.hpp"

#include <gtest/gtest.h>

#include <cstdlib>

namespace ck_tile {

// Overwrites smem right after TdmEpilogue returns, as the next persistent tile's producers do.
template <typename Problem, index_t M, index_t N>
__global__ void
test_tdm_epilogue_lds_reuse_kernel(const typename Problem::AccDataType* __restrict__ input_data,
                                   typename Problem::ODataType* __restrict__ output_data,
                                   typename Problem::ODataType sentinel)
{
    // TDM exists only on gfx125; discard the body on the other targets of a mixed build.
    if constexpr(CK_TILE_ENABLE_TDM_FEATURE != 0)
    {
        using Epilogue    = TdmEpilogue<Problem>;
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

        auto output_tensor_view =
            make_naive_tensor_view<address_space_enum::global>(output_data,
                                                               make_tuple(M, N),
                                                               make_tuple(N, 1),
                                                               number<Epilogue::GetVectorSizeC()>{},
                                                               number<1>{});
        auto output_tile_window =
            make_tile_window(output_tensor_view,
                             make_tuple(number<kMPerBlock>{}, number<kNPerBlock>{}),
                             {static_cast<index_t>(blockIdx.x) * kMPerBlock, 0});

        auto empty_ds = make_tuple();
        Epilogue{}(output_tile_window, acc_tile, empty_ds, smem);

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
using TdmProblem = CShuffleEpilogueProblem<half_t,
                                           half_t,
                                           tuple<>,
                                           float,
                                           half_t,
                                           tuple<>,
                                           tensor_layout::gemm::RowMajor,
                                           element_wise::PassThrough,
                                           kTileM,
                                           kTileN,
                                           2,   // MWave
                                           2,   // NWave
                                           16,  // MPerXdl
                                           16,  // NPerXdl
                                           32,  // KPerXdl
                                           true // isCTransposed
                                           >;

TEST(TdmEpilogue, OverwritingLdsAfterReturnKeepsC)
{
    if(!is_gfx125_supported())
    {
        const char* required = std::getenv("CK_TILE_REQUIRE_GFX125");
        if(required != nullptr && required[0] == '1')
        {
            FAIL() << "CK_TILE_REQUIRE_GFX125=1 but the device reports '" << get_device_name()
                   << "'";
        }
        GTEST_SKIP() << "TDM requires gfx1250; device reports '" << get_device_name() << "'";
    }

    auto host_input = generate_unique_fp16_input<float, kTileM, kTileN>();
    DeviceMem input_buf(host_input.get_element_space_size_in_bytes());
    input_buf.ToDevice(host_input.data());

    HostTensor<half_t> host_output({kM, kN});
    host_output.SetZero();
    DeviceMem output_buf(host_output.get_element_space_size_in_bytes());
    output_buf.ToDevice(host_output.data());

    int device    = 0;
    int warp_size = 0;
    HIP_CHECK_ERROR(hipGetDevice(&device));
    HIP_CHECK_ERROR(hipDeviceGetAttribute(&warp_size, hipDeviceAttributeWarpSize, device));
    const dim3 grid_size(kNumBlocks, 1, 1);
    const dim3 block_size(TdmProblem::MWave * TdmProblem::NWave * warp_size, 1, 1);

    // Inputs are positive fp16 normals, so -1 can only come from the overwritten LDS.
    const half_t sentinel = type_convert<half_t>(-1.0F);
    test_tdm_epilogue_lds_reuse_kernel<TdmProblem, kM, kN>
        <<<grid_size, block_size>>>(static_cast<const float*>(input_buf.GetDeviceBuffer()),
                                    static_cast<half_t*>(output_buf.GetDeviceBuffer()),
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
    EXPECT_EQ(mismatches, 0) << "TdmEpilogue returned while its TDM store still read LDS";
}

int main(int argc, char** argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
