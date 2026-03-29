// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "topk_softmax_decode_api.hpp"

#define TOPK_SOFTMAX_DECODE_DISPATCH(experts_, use_softmax_)                                       \
    constexpr ck_tile::index_t ts_experts = experts_;                                              \
    constexpr bool ts_use_softmax         = use_softmax_;                                          \
    using ts_problem                      = ck_tile::TopkSoftmaxWarpPerRowProblem<ts_input_type,   \
                                                                                  ts_weight_type,  \
                                                                                  ts_index_type,   \
                                                                                  ts_experts,      \
                                                                                  ts_use_softmax>; \
    using ts_pipeline                     = ck_tile::TopkSoftmaxDecodePipeline<ts_problem>;        \
                                                                                                   \
    using kernel = ck_tile::TopkSoftmaxDecodeKernel<ts_pipeline>;                                  \
                                                                                                   \
    auto kargs = kernel::MakeKargs(a);                                                             \
                                                                                                   \
    const dim3 grids  = kernel::GridSize(a);                                                       \
    const dim3 blocks = kernel::BlockSize();                                                       \
                                                                                                   \
    float ave_time =                                                                               \
        ck_tile::launch_kernel(s, ck_tile::make_kernel<1>(kernel{}, grids, blocks, 0, kargs));     \
                                                                                                   \
    return ave_time;

#define TOPK_SOFTMAX_DECODE_EXPERT_LADDER(use_softmax_) \
    if(t.experts <= 8)                                  \
    {                                                   \
        TOPK_SOFTMAX_DECODE_DISPATCH(8, use_softmax_)   \
    }                                                   \
    else if(t.experts <= 16)                            \
    {                                                   \
        TOPK_SOFTMAX_DECODE_DISPATCH(16, use_softmax_)  \
    }                                                   \
    else if(t.experts <= 32)                            \
    {                                                   \
        TOPK_SOFTMAX_DECODE_DISPATCH(32, use_softmax_)  \
    }                                                   \
    else if(t.experts <= 64)                            \
    {                                                   \
        TOPK_SOFTMAX_DECODE_DISPATCH(64, use_softmax_)  \
    }                                                   \
    else if(t.experts <= 128)                           \
    {                                                   \
        TOPK_SOFTMAX_DECODE_DISPATCH(128, use_softmax_) \
    }                                                   \
    else if(t.experts <= 192)                           \
    {                                                   \
        TOPK_SOFTMAX_DECODE_DISPATCH(192, use_softmax_) \
    }                                                   \
    else if(t.experts <= 256)                           \
    {                                                   \
        TOPK_SOFTMAX_DECODE_DISPATCH(256, use_softmax_) \
    }                                                   \
    else if(t.experts <= 512)                           \
    {                                                   \
        TOPK_SOFTMAX_DECODE_DISPATCH(512, use_softmax_) \
    }                                                   \
    else if(t.experts <= 1024)                          \
    {                                                   \
        TOPK_SOFTMAX_DECODE_DISPATCH(1024, use_softmax_)\
    }

float topk_softmax_decode(topk_softmax_decode_trait t,
                          topk_softmax_decode_kargs a,
                          ck_tile::stream_config s)
{
    if(t.input_type == "fp16" && t.weight_type == "fp32" && t.activation == "softmax")
    {
        using ts_input_type  = ck_tile::fp16_t;
        using ts_weight_type = float;
        using ts_index_type  = ck_tile::index_t;
        TOPK_SOFTMAX_DECODE_EXPERT_LADDER(true)
    }
    else if(t.input_type == "bf16" && t.weight_type == "fp32" && t.activation == "softmax")
    {
        using ts_input_type  = ck_tile::bf16_t;
        using ts_weight_type = float;
        using ts_index_type  = ck_tile::index_t;
        TOPK_SOFTMAX_DECODE_EXPERT_LADDER(true)
    }
    else if(t.input_type == "fp16" && t.weight_type == "fp32" && t.activation == "sigmoid")
    {
        using ts_input_type  = ck_tile::fp16_t;
        using ts_weight_type = float;
        using ts_index_type  = ck_tile::index_t;
        TOPK_SOFTMAX_DECODE_EXPERT_LADDER(false)
    }
    else if(t.input_type == "bf16" && t.weight_type == "fp32" && t.activation == "sigmoid")
    {
        using ts_input_type  = ck_tile::bf16_t;
        using ts_weight_type = float;
        using ts_index_type  = ck_tile::index_t;
        TOPK_SOFTMAX_DECODE_EXPERT_LADDER(false)
    }
    return -1;
}
