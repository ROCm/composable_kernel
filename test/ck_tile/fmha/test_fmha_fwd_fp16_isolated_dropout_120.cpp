// SPDX-License-Identifier: MIT
// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.

#include "example/ck_tile/01_fmha/fmha_fwd.hpp"
#include "example/ck_tile/01_fmha/fmha_fwd_runner.hpp"

#include "gtest/gtest.h"

#include <tuple>
#include <string>

using ::testing::Values;

using DataTypeConfig = FmhaFwdFp16;

// Random seed used for initializing input tensors
CK_TILE_DECLARE_ENV_VAR(CK_TILE_TEST_SEED, uint64_t, 123456)

#define CHECK_RESULT(result)                                      \
    do                                                            \
    {                                                             \
        if(result == fwd_result::no_instance)                     \
            GTEST_SKIP() << "No instance for current parameters"; \
        ASSERT_EQ(result, fwd_result::success);                   \
    } while(0)

const ck_tile::stream_config stream_config{
    nullptr, // stream_id_
    false,   // time_kernel_
    1,       // log_level_
    0,       // cold_niters_
    1,       // nrepeat_
    true,    // is_gpu_timer_
    false,   // flush_cache_
    1,       // rotating_count_
};

#define COMMON_ARGS \
    "uf", static_cast<uint32_t>(ck_tile::EnvValue(CK_TILE_ENV(CK_TILE_TEST_SEED))), 1, stream_config

constexpr auto qscale_str        = "n";
constexpr bool def_lse           = true;
constexpr bool def_is_v_rowmajor = true;

int adjust_seqlen(int seqlen) { return seqlen; }

// Isolated test for the failing configuration:
// TestCkTileFmhaFwd/Dropout.FmhaFwdFp16/126
// GetParam() = ((192, -1), batch, 0.5, (10, 123, false), (2, 4, 2, 280, 512, "0"))
TEST(IsolatedDropoutTest, FmhaFwdFp16_Config126_Dropout0_5)
{
    // Extracted parameters from the failing test
    auto hdim_q      = 192;
    auto hdim_v      = -1; // Will use hdim_q
    auto mode        = mode_enum::batch;
    auto p_drop      = 0.5f;
    auto drop_seed   = 10;
    auto drop_offset = 123;
    auto drop_prefs  = false;
    auto batch       = 2;
    auto nhead       = 4;
    auto nhead_k     = 2;
    auto seqlen_q    = 280;
    auto seqlen_k    = 512;
    auto mask_str    = "0";

    // Set hdim_v to hdim_q if it's -1
    if(hdim_v == -1)
    {
        hdim_v = hdim_q;
    }

    auto result = fmha_fwd_run<DataTypeConfig>(mode,
                                               batch,
                                               nhead,
                                               nhead_k,
                                               {adjust_seqlen(seqlen_q)},
                                               {adjust_seqlen(seqlen_k)},
                                               hdim_q,
                                               hdim_v,
                                               0,                 // seqlen_knew
                                               {-1},              // seqlen_qpads
                                               {-1},              // seqlen_kpads
                                               {},                // q_eff_lens_per_batch
                                               {},                // kv_eff_lens_per_batch
                                               0,                 // rotary_dim
                                               false,             // i_perm
                                               false,             // o_perm
                                               0,                 // scale_s
                                               0,                 // logits_soft_cap
                                               def_is_v_rowmajor, // is_v_rowmajor
                                               def_lse,           // lse
                                               0,                 // page_block_size
                                               false,             // use_cache_batch_idx
                                               "n",               // bias_str
                                               p_drop,            // p_drop
                                               drop_seed,         // drop_seed
                                               drop_offset,       // drop_offset
                                               drop_prefs,        // drop_prefs
                                               mask_str,          // mask_str
                                               qscale_str,
                                               true, // is_rotary_interleaved
                                               1,    // num_splits
                                               COMMON_ARGS);
    CHECK_RESULT(result);
}
