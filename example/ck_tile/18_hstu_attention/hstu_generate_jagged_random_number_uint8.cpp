// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2025, Advanced Micro Devices, Inc. All rights reserved.

void hstu_generate_jagged_random_number_uint8(HstuGenerateRandUniformNumbersParams& param,
                                               hipStream_t stream)
{
    // only work for jagged mode
    using HstuRandUniformKernel_ = HstuRandUniformKernel<uint8_t, true>;

    const auto kargs = HstuRandUniformKernel_::MakeKargs(param.rand_val_ptr,
                                                         param.seqlen_q,
                                                         param.seqlen_k,
                                                         param.num_heads,
                                                         param.num_batches,
                                                         param.stride_seqlen_q,
                                                         param.stride_seqlen_k,
                                                         param.stride_nhead,
                                                         param.seqstart_q_ptr,
                                                         param.seqstart_k_ptr,
                                                         {param.philox_seed, param.philox_offset});

    dim3 kGridSize = HstuRandUniformKernel_::GridSize(
        param.num_batches, param.num_heads, param.seqlen_q, param.seqlen_k);
    dim3 kBlockSize                        = HstuRandUniformKernel_::BlockSize();
    constexpr ck_tile::index_t kBlockPerCu = HstuRandUniformKernel_::kBlockPerCu;

    (void)ck_tile::launch_kernel(ck_tile::stream_config{stream, false},
                                 ck_tile::make_kernel<kBlockPerCu>(
                                     HstuRandUniformKernel_{}, kGridSize, kBlockSize, 0, kargs));
};
