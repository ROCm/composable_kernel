// SPDX-License-Identifier: MIT
// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.

#define CHECK_RESULT(result)                                      \
    do                                                            \
    {                                                             \
        if(result == fwd_result::no_instance)                     \
            GTEST_SKIP() << "No instance for current parameters"; \
        ASSERT_EQ(result, fwd_result::success);                   \
    } while(0)

template <typename T>
class TestCkTileFmhaFwd : public testing::Test
{
};

const ck_tile::stream_config default_stream_config{
    nullptr, // stream_id_
    false,   // time_kernel_
    1,       // log_level_
    0,       // cold_niters_
    1,       // nrepeat_
    true,    // is_gpu_timer_
    false,   // flush_cache_
    1,       // rotating_count_
};

TYPED_TEST(TestCkTileFmhaFwd, Simple)
{
    using DataTypeConfig = TypeParam;

    auto bias = bias_info{bias_enum::no_bias, 0};

    constexpr bool squant = std::is_same_v<DataTypeConfig, FmhaFwdFp8>;

    auto result = fmha_fwd_run<DataTypeConfig>(mode_enum::batch,       // mode
                                               3,                      // batch
                                               4,                      // nhead
                                               2,                      // nhead_k
                                               "1024",                 // seqlen_q_str
                                               1024,                   // seqlen_q
                                               "512",                  // seqlen_k_str
                                               64,                     // hdim_q
                                               64,                     // hdim_v
                                               0,                      // seqlen_knew
                                               "0",                    // seqlen_kpad_str
                                               0,                      // rotary_dim
                                               false,                  // i_perm
                                               false,                  // o_perm
                                               0,                      // scale_s
                                               0,                      // logits_soft_cap
                                               false,                  // is_v_rowmajor
                                               false,                  // lse
                                               0,                      // page_block_size
                                               false,                  // use_cache_batch_idx
                                               bias,                   // bias
                                               0.0f,                   // p_drop
                                               123,                    // drop_seed
                                               1024,                   // drop_offset
                                               false,                  // drop_prefs
                                               "0",                    // mask_str
                                               16,                     // range_q
                                               16,                     // range_k
                                               16,                     // range_v
                                               1,                      // range_p
                                               16,                     // range_o
                                               squant,                 // squant
                                               false,                  // is_rotary_interleaved
                                               1,                      // num_splits
                                               squant ? "ufq" : "uf",  // init_method
                                               5675,                   // seed
                                               1,                      // do_validation
                                               default_stream_config); // stream_config)
    CHECK_RESULT(result);
}
