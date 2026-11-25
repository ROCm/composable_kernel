// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// comp_tdm_scheduled with operands past 2^31 bytes (the wide A / E address paths), checked bit for
// bit on sampled rows: the first and last tile of rows and random ones. Needs a gfx1250 device with
// about 8 GB free.

#include "test_mx_gemm_tdm_scheduled_util.hpp"

namespace {

using namespace mx_tdm_scheduled_test;
using ck_tile::fp8_t;
using ck_tile::half_t;
using ck_tile::pk_fp4_t;

std::vector<index_t> sample_rows(index_t M, index_t tile_m)
{
    std::mt19937 rng(0xB10C);
    std::set<index_t> rows;
    for(index_t r = 0; r < std::min(M, tile_m); ++r)
    {
        rows.insert(r);
        rows.insert(M - 1 - r);
    }
    for(int i = 0; i < 64; ++i)
        rows.insert(static_cast<index_t>(rng() % M));
    return {rows.begin(), rows.end()};
}

template <typename Kernel>
void check_sampled_kernel(index_t M, index_t N, index_t K)
{
    using Pipeline = typename Kernel::GemmPipeline;
    using O        = typename Pipeline::ODataType;
    SCOPED_TRACE(Kernel::GetName() + " M" + std::to_string(M) + " N" + std::to_string(N) + " K" +
                 std::to_string(K));
    const Inputs in = make_kernel_inputs<Kernel>(M, N, K);
    const auto got  = run_device<Kernel>(in, M, N, K);
#ifdef MX_TDM_SCHEDULED_DRY_RUN
    return;
#endif
    const auto rows              = sample_rows(M, Pipeline::kTileM);
    const auto want              = run_reference<Kernel>(in, M, N, K, &rows);
    const long_index_t row_bytes = static_cast<long_index_t>(N) * sched::out_bytes<O>();
    std::vector<uint8_t> got_rows(want.size());
    for(std::size_t i = 0; i < rows.size(); ++i)
        std::memcpy(got_rows.data() + i * row_bytes, got.data() + rows[i] * row_bytes, row_bytes);
    EXPECT_EQ(first_mismatch<O>(got_rows, want, N), "");
}

template <typename Kernel>
void check_sampled(index_t M, index_t N, index_t K)
{
    for_tail_num<Kernel>(
        K, [&](auto variant) { check_sampled_kernel<typename decltype(variant)::type>(M, N, K); });
}

// A past 2^31 bytes: the 128x16x512 4x1-wave tile (tile_n 16) at M 65536, K 81920 (2.5 GiB of FP4
// A).
TEST(MxGemmTdmScheduledLargeCases, WideA)
{
    using K = Kernel<pk_fp4_t,
                     pk_fp4_t,
                     fp8_t,
                     128,
                     16,
                     512,
                     4,
                     1,
                     16,
                     Tu<1, 4, 0, false, false>,
                     ck_tile::MxTdmScheduledGemmTraits<false, false, true, false, false, 0>>;
    check_sampled<K>(65536, 16, 81920);
}

// E past 2^31 bytes: 33024 x 33024 FP16 (2.03 GiB).
TEST(MxGemmTdmScheduledLargeCases, WideD)
{
    using K = Kernel<pk_fp4_t,
                     pk_fp4_t,
                     half_t,
                     256,
                     256,
                     512,
                     2,
                     2,
                     32,
                     Tu<1, 1>,
                     ck_tile::MxTdmScheduledGemmTraits<false, false, false, false, true, 0>>;
    check_sampled<K>(33024, 33024, 1024);
}

} // namespace
