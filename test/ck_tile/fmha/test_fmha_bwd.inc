// SPDX-License-Identifier: MIT
// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.

template <typename T>
class TestCkTileFmhaBwd : public testing::Test
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

TYPED_TEST(TestCkTileFmhaBwd, Simple)
{
    using DataTypeConfig = TypeParam;

    auto bias = bias_info{bias_enum::no_bias, 0};
    auto mask = mask_info{mask_enum::no_mask, 0, 0, 0, 0, 0, 0};

    auto result = fmha_bwd_run<DataTypeConfig>(mode_enum::batch,       // mode
                                               3,                      // batch
                                               4,                      // nhead
                                               2,                      // nhead_k
                                               1024,                   // seqlen_q
                                               512,                    // seqlen_k
                                               64,                     // hdim_q
                                               64,                     // hdim_v
                                               false,                  // i_perm
                                               false,                  // o_perm
                                               0,                      // scale
                                               bias,                   // bias
                                               false,                  // use_dbias
                                               0.1f,                   // p_drop
                                               123,                    // drop_seed
                                               1024,                   // drop_offset
                                               true,                   // drop_prefs
                                               mask,                   // mask
                                               false,                  // deterministic
                                               "uf",                   // init_method
                                               67889,                  // seed
                                               1,                      // do_validation
                                               default_stream_config); // stream_config
    ASSERT_TRUE(result);
}
