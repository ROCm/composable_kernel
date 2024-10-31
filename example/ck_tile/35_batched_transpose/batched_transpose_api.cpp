#include "batched_transpose_api.hpp"

#define BATCHED_TRANSPOSE_DISPATCH()                                                   \
    using block_tile  = ck_tile::sequence<16, 16>;                                     \
    using warp_tile   = ck_tile::sequence<8, 8>;                                       \
    using thread_tile = ck_tile::sequence<1, 1>;                                       \
    using ts_problem =                                                                 \
        ck_tile::BatchedTransposeProblem<ts_type, block_tile, warp_tile, thread_tile>; \
    using ts_pipeline = ck_tile::BatchedTransposePipeline<ts_problem>;                 \
                                                                                       \
    using kernel = ck_tile::BatchedTransposeKernel<ts_pipeline>;                       \
                                                                                       \
    auto kargs = kernel::MakeKargs(a);                                                 \
                                                                                       \
    const dim3 grids      = kernel::GridSize(a);                                       \
    constexpr dim3 blocks = kernel::BlockSize();                                       \
                                                                                       \
    float ave_time = ck_tile::launch_kernel(                                           \
        s, ck_tile::make_kernel<blocks.x, 1>(kernel{}, grids, blocks, 0, kargs));      \
                                                                                       \
    return ave_time;

float batched_transpose(batched_transpose_trait t,
                        batched_transpose_kargs a,
                        ck_tile::stream_config s)
{
    if(t.type == "fp16")
    {
        using ts_type = ck_tile::fp16_t;
        if(t.layout == "NCHW")
        {
            BATCHED_TRANSPOSE_DISPATCH()
        }
        else if(t.layout == "NHWC")
        {
            BATCHED_TRANSPOSE_DISPATCH()
        }
    }
    else if(t.type == "bf16")
    {
        using ts_type = ck_tile::bf16_t;
        if(t.layout == "NCHW")
        {
            BATCHED_TRANSPOSE_DISPATCH()
        }
        else if(t.layout == "NHWC")
        {
            BATCHED_TRANSPOSE_DISPATCH()
        }
    }
    else if(t.type == "fp32")
    {
        using ts_type = ck_tile::fp32_t;
        if(t.layout == "NCHW")
        {
            BATCHED_TRANSPOSE_DISPATCH()
        }
        else if(t.layout == "NHWC")
        {
            BATCHED_TRANSPOSE_DISPATCH()
        }
    }
    else if(t.type == "int8")
    {
        using ts_type = ck_tile::int8_t;
        if(t.layout == "NCHW")
        {
            BATCHED_TRANSPOSE_DISPATCH()
        }
        else if(t.layout == "NHWC")
        {
            BATCHED_TRANSPOSE_DISPATCH()
        }
    }
    return -1;
}
