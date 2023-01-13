// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2022, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck/utility/data_type.hpp"
#include "ck/utility/philox_rand.hpp"

namespace ck {

template <typename ThreadSliceDesc_M_K>
struct BlockwiseDropout
{
    static constexpr auto I0         = Number<0>{};
    static constexpr auto I1         = Number<1>{};
    static constexpr index_t MRepeat = ThreadSliceDesc_M_K{}.GetLength(I0);
    static constexpr index_t KRepeat = ThreadSliceDesc_M_K{}.GetLength(I1);

    template <typename CThreadBuffer>
    __host__ __device__ void ApplyDropout(CThreadBuffer& in_thread_buf,
                                          ushort p_dropout_16bits,
                                          ck::philox ph,
                                          const int repeat_index,
                                          const int total_repeats)
    {

        auto if_dropout = [](bool keep, float val) { return keep ? val : float(0); };

        constexpr int tmp_size = MRepeat * KRepeat;
        int philox_calls       = tmp_size / 8;
        int tid                = get_thread_global_1d_id();
        unsigned long long uni_subsequence =
            tid * total_repeats * philox_calls + repeat_index * philox_calls;

        ushort tmp[tmp_size];
        for(int i = 0; i < philox_calls; i++)
        {
            ph.get_random_8x16((tmp + i * 8), (uni_subsequence + i));
        }

        block_sync_lds();

        for(int i = 0; i < tmp_size; i++)
        {
            printf("%u \n", tmp[i]);
        }

        int tmp_index = 0;
        static_for<0, MRepeat, 1>{}([&](auto iM) {
            static_for<0, KRepeat, 1>{}([&](auto iK) {
                auto offset = Number<ThreadSliceDesc_M_K{}.CalculateOffset(make_tuple(iM, iK))>{};
                in_thread_buf(offset) =
                    if_dropout(tmp[tmp_index] < p_dropout_16bits, in_thread_buf(offset));
                tmp_index = tmp_index + 1;
            });
        });
    }
};

} // namespace ck
