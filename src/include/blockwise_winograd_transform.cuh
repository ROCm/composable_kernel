#pragma once
#include "constant_tensor_descriptor.cuh"

template <class TFloat,
          unsigned InTileSizeH,
          unsigned InTileSizeW,
          unsigned S,
          unsigned R,
          unsigned OutTileSizeH,
          unsigned OutTileSizeW,
          unsigned NPerBlock,
          unsigned CPerBlock,
          unsigned YPerBlock,
          unsigned XPerBlock,
          unsigned BlockSize>
__device__ void blockwise_winograd_transform_input(TFloat* const __restrict__ p_in,
                                                   TFloat* __restrict__ p_in_transform)
{
    p_in_transform[0] = 1;
}

template <class TFloat,
          unsigned InTileSizeH,
          unsigned InTileSizeW,
          unsigned S,
          unsigned R,
          unsigned OutTileSizeH,
          unsigned OutTileSizeW,
          unsigned KPerBlock,
          unsigned CPerBlock,
          unsigned BlockSize>
__device__ void blockwise_winograd_transform_weight(TFloat* const __restrict__ p_wei,
                                                    TFloat* __restrict__ p_wei_transform)
{
    p_wei_transform[0] = 1;
}