// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT
#include "batched_transpose.hpp"

template <typename ts_type,
          ck_tile::index_t block_x,
          ck_tile::index_t block_y,
          ck_tile::index_t warp_x,
          ck_tile::index_t warp_y,
          bool kPadM,
          bool kPadN>
float batched_transpose_dispatch(batched_transpose_kargs& a, ck_tile::stream_config& s)
{
    uint32_t dim_stride = a.height * a.width;

    a.dim_stride  = dim_stride;
    a.dim_block_h = block_y;
    a.dim_block_w = block_x;

    using block_tile  = ck_tile::sequence<block_x, block_y>;
    using warp_layout = ck_tile::sequence<warp_x, warp_y>;

    using ts_problem =
        ck_tile::BatchedTransposeProblem<ts_type, block_tile, warp_layout, kPadM, kPadN>;
    using ts_pipeline = ck_tile::BatchedTransposePipeline<ts_problem>;

    using kernel = ck_tile::BatchedTransposeKernel<ts_pipeline>;

    auto kargs = kernel::MakeKargs(a);

    const dim3 grids      = kernel::GridSize(a);
    constexpr dim3 blocks = kernel::BlockSize();

    printf("Grid: %u %u %u\n", grids.x, grids.y, grids.z);
    printf("Block: %u %u %u\n", blocks.x, blocks.y, blocks.z);
    printf("kargs: kargs.batch %d kargs.height %d kargs.width %d kargs.dim_strid %d\n",
           kargs.batch,
           kargs.height,
           kargs.width,
           kargs.dim_stride);

    printf("Launching Kernel...\n");

    float ave_time = ck_tile::launch_kernel(
        s, ck_tile::make_kernel<blocks.x, 1>(kernel{}, grids, blocks, 0, kargs));

    printf("Kernel finished...\n");

    return ave_time;
}

// Param Comb: type_size, block_x & y, warp_x & y, thread_x & y
#define FOREACH_TRANSPOSE_PARAM(F)                       \
    F(fp8, ck_tile::fp8_t, 64, 64, 1, 1, true, true)     \
    F(fp8, ck_tile::fp8_t, 64, 64, 1, 1, false, false)   \
    F(fp16, ck_tile::fp16_t, 64, 64, 1, 1, true, true)   \
    F(fp16, ck_tile::fp16_t, 64, 64, 1, 1, false, false) \
    F(bf16, ck_tile::bf16_t, 64, 64, 1, 1, true, true)   \
    F(bf16, ck_tile::bf16_t, 64, 64, 1, 1, false, false)

// Macro that defines one static function per line
#define GEN_TRANSPOSE_FN(SHORT_NAME, REAL_TYPE, BX, BY, WX, WY, PADM, PADN)               \
    static float transpose_fn_##SHORT_NAME##_##BX##_##BY##_##WX##_##WY##_##PADM##_##PADN( \
        batched_transpose_kargs& a, ck_tile::stream_config& s)                            \
    {                                                                                     \
        return batched_transpose_dispatch<REAL_TYPE, BX, BY, WX, WY, PADM, PADN>(a, s);   \
    }

FOREACH_TRANSPOSE_PARAM(GEN_TRANSPOSE_FN)

float batched_transpose(batched_transpose_trait t,
                        batched_transpose_kargs a,
                        ck_tile::stream_config s)
{
    if(t.type == "fp8")
    {
        if(a.height % 64 == 0 && a.width % 64 == 0)
        {
            return transpose_fn_fp8_64_64_1_1_false_false(a, s);
        }
        else
        {
            return transpose_fn_fp8_64_64_1_1_true_true(a, s);
        }
    }
    else if(t.type == "fp16")
    {
        if(a.height % 64 == 0 && a.width % 64 == 0)
        {
            return transpose_fn_fp16_64_64_1_1_false_false(a, s);
        }
        else
        {
            return transpose_fn_fp16_64_64_1_1_true_true(a, s);
        }
    }
    else if(t.type == "bf16")
    {
        if(a.height % 64 == 0 && a.width % 64 == 0)
        {
            return transpose_fn_bf16_64_64_1_1_false_false(a, s);
        }
        else
        {
            return transpose_fn_bf16_64_64_1_1_true_true(a, s);
        }
    }
    return -1;
}
