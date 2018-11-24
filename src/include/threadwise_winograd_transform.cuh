#pragma once
#include "constant_tensor_descriptor.cuh"

template <class TFloat,
          class InTransThreadDesc,  //{NPerThread, CPerThread, InTileSizeH, InTileSizeW}
          class WeiTransThreadDesc, //{KPerThread, CPerThread, InTileSizeH, InTileSizeW}
          class OutTransThreadDesc, //{NPerThread, KPerThread, InTileSizeH, InTileSizeW}
          unsigned InTileSizeH,
          unsigned InTileSizeW,
          unsigned S,
          unsigned R,
          unsigned OutTileSizeH,
          unsigned OutTileSizeW>
__device__ void
threadwise_winograd_calculate_transformed_output(InTransThreadDesc,
                                                 TFloat* const __restrict__ p_in_transform_thread,
                                                 WeiTransThreadDesc,
                                                 TFloat* const __restrict__ p_wei_transform_thread,
                                                 OutTransThreadDesc,
                                                 TFloat* __restrict__ p_out_transform_thread)
{
    constexpr auto I0 = Index<0>{};
    constexpr auto I1 = Index<1>{};
    constexpr auto I2 = Index<2>{};
    constexpr auto I3 = Index<3>{};

    constexpr auto in_transform_thread_desc  = InTransThreadDesc{};
    constexpr auto wei_transform_thread_desc = WeiTransThreadDesc{};
    constexpr auto out_transform_thread_desc = OutTransThreadDesc{};

    for(unsigned n = 0; n < out_transform_thread_desc.GetLength(I0); ++n)
    {
        for(unsigned k = 0; k < out_transform_thread_desc.GetLength(I1); ++k)
        {
            for(unsigned h = 0; h < out_transform_thread_desc.GetLength(I2); ++h)
            {
                for(unsigned w = 0; w < out_transform_thread_desc.GetLength(I3); ++w)
                {
                    for(unsigned c = 0; c < wei_transform_thread_desc.GetLength(I1); ++c)
                    {
                        const unsigned in_index  = in_transform_thread_desc.Get1dIndex(n, c, h, w);
                        const unsigned wei_index = wei_transform_thread_desc.Get1dIndex(k, c, h, w);
                        const unsigned out_index = out_transform_thread_desc.Get1dIndex(n, k, h, w);

                        p_out_transform_thread[out_index] +=
                            p_wei_transform_thread[wei_index] * p_in_transform_thread[in_index];
                    }
                }
            }
        }
    }
}

template <class TFloat,
          class OutTransThreadDesc, //{NPerThread, KPerThread,  InTileSizeH,  InTileSizeW}
          class OutThreadDesc,      //{NPerThread, CPerThread, OutTileSizeH, OutTileSizeW}
          unsigned InTileSizeH,
          unsigned InTileSizeW,
          unsigned S,
          unsigned R,
          unsigned OutTileSizeH,
          unsigned OutTileSizeW>
__device__ void
threadwise_winograd_reverse_transform_output(OutTransThreadDesc,
                                             TFloat* const __restrict__ p_out_transform_thread,
                                             OutThreadDesc,
                                             TFloat* __restrict__ p_out_thread)
{
    static_assert(InTileSizeH == 4, "wrong");
    static_assert(InTileSizeW == 4, "wrong");
    static_assert(S == 3, "wrong");
    static_assert(R == 3, "wrong");
    static_assert(OutTileSizeH == 2, "wrong");
    static_assert(OutTileSizeW == 2, "wrong");

    constexpr auto I0 = Index<0>{};
    constexpr auto I1 = Index<1>{};
    constexpr auto I2 = Index<2>{};
    constexpr auto I3 = Index<3>{};

    constexpr auto out_transform_thread_desc = OutTransThreadDesc{};
    constexpr auto out_thread_desc           = OutThreadDesc{};

    static_assert(InTileSizeH == out_transform_thread_desc.GetLength(I2), "wrong");
    static_assert(InTileSizeW == out_transform_thread_desc.GetLength(I3), "wrong");
    static_assert(OutTileSizeH == out_thread_desc.GetLength(I2), "wrong");
    static_assert(OutTileSizeW == out_thread_desc.GetLength(I3), "wrong");

    for(unsigned n = 0; n < out_thread_desc.GetLength(I0); ++n)
    {
        for(unsigned k = 0; k < out_thread_desc.GetLength(I1); ++k)
        {
            p_out_thread[out_thread_desc.Get1dIndex(n, k, 0, 0)] =
                p_out_transform_thread[out_transform_thread_desc.Get1dIndex(n, k, 0, 0)] +
                p_out_transform_thread[out_transform_thread_desc.Get1dIndex(n, k, 0, 1)] +
                p_out_transform_thread[out_transform_thread_desc.Get1dIndex(n, k, 0, 2)] +
                p_out_transform_thread[out_transform_thread_desc.Get1dIndex(n, k, 1, 0)] +
                p_out_transform_thread[out_transform_thread_desc.Get1dIndex(n, k, 1, 1)] +
                p_out_transform_thread[out_transform_thread_desc.Get1dIndex(n, k, 1, 2)] +
                p_out_transform_thread[out_transform_thread_desc.Get1dIndex(n, k, 2, 0)] +
                p_out_transform_thread[out_transform_thread_desc.Get1dIndex(n, k, 2, 1)] +
                p_out_transform_thread[out_transform_thread_desc.Get1dIndex(n, k, 2, 2)];

            p_out_thread[out_thread_desc.Get1dIndex(n, k, 0, 1)] =
                p_out_transform_thread[out_transform_thread_desc.Get1dIndex(n, k, 0, 1)] -
                p_out_transform_thread[out_transform_thread_desc.Get1dIndex(n, k, 0, 2)] -
                p_out_transform_thread[out_transform_thread_desc.Get1dIndex(n, k, 0, 3)] +
                p_out_transform_thread[out_transform_thread_desc.Get1dIndex(n, k, 1, 1)] -
                p_out_transform_thread[out_transform_thread_desc.Get1dIndex(n, k, 1, 2)] -
                p_out_transform_thread[out_transform_thread_desc.Get1dIndex(n, k, 1, 3)] +
                p_out_transform_thread[out_transform_thread_desc.Get1dIndex(n, k, 2, 1)] -
                p_out_transform_thread[out_transform_thread_desc.Get1dIndex(n, k, 2, 2)] -
                p_out_transform_thread[out_transform_thread_desc.Get1dIndex(n, k, 2, 3)];

            p_out_thread[out_thread_desc.Get1dIndex(n, k, 1, 0)] =
                p_out_transform_thread[out_transform_thread_desc.Get1dIndex(n, k, 1, 0)] +
                p_out_transform_thread[out_transform_thread_desc.Get1dIndex(n, k, 1, 1)] +
                p_out_transform_thread[out_transform_thread_desc.Get1dIndex(n, k, 1, 2)] -
                p_out_transform_thread[out_transform_thread_desc.Get1dIndex(n, k, 2, 0)] -
                p_out_transform_thread[out_transform_thread_desc.Get1dIndex(n, k, 2, 1)] -
                p_out_transform_thread[out_transform_thread_desc.Get1dIndex(n, k, 2, 2)] -
                p_out_transform_thread[out_transform_thread_desc.Get1dIndex(n, k, 3, 0)] -
                p_out_transform_thread[out_transform_thread_desc.Get1dIndex(n, k, 3, 1)] -
                p_out_transform_thread[out_transform_thread_desc.Get1dIndex(n, k, 3, 2)];

            p_out_thread[out_thread_desc.Get1dIndex(n, k, 1, 1)] =
                p_out_transform_thread[out_transform_thread_desc.Get1dIndex(n, k, 1, 1)] -
                p_out_transform_thread[out_transform_thread_desc.Get1dIndex(n, k, 1, 2)] -
                p_out_transform_thread[out_transform_thread_desc.Get1dIndex(n, k, 1, 3)] -
                p_out_transform_thread[out_transform_thread_desc.Get1dIndex(n, k, 2, 1)] +
                p_out_transform_thread[out_transform_thread_desc.Get1dIndex(n, k, 2, 2)] +
                p_out_transform_thread[out_transform_thread_desc.Get1dIndex(n, k, 2, 3)] -
                p_out_transform_thread[out_transform_thread_desc.Get1dIndex(n, k, 3, 1)] +
                p_out_transform_thread[out_transform_thread_desc.Get1dIndex(n, k, 3, 2)] +
                p_out_transform_thread[out_transform_thread_desc.Get1dIndex(n, k, 3, 3)];
        }
    }
}