#pragma once
#include "constant_tensor_descriptor.cuh"

template <class TFloat, class InDesc, class WeiDesc, class OutDesc>
__device__ void threadwise_direct_convolution(InDesc,
                                              TFloat* const __restrict__ p_in,
                                              WeiDesc,
                                              TFloat* const __restrict__ p_wei,
                                              OutDesc,
                                              TFloat* __restrict__ p_out)
{
    constexpr auto I0 = Index<0>{};
    constexpr auto I1 = Index<1>{};
    constexpr auto I2 = Index<2>{};
    constexpr auto I3 = Index<3>{};

    constexpr auto in_desc  = InDesc{};
    constexpr auto wei_desc = WeiDesc{};
    constexpr auto out_desc = OutDesc{};

#if 0
    if(threadIdx.x == 0)
    {
        print_ConstantTensorDescriptor(in_desc);
        print_ConstantTensorDescriptor(wei_desc);
        print_ConstantTensorDescriptor(out_desc);
    }
#endif

    for(unsigned n = 0; n < out_desc.GetLength(I0); ++n)
    {
        for(unsigned k = 0; k < out_desc.GetLength(I1); ++k)
        {
            for(unsigned ho = 0; ho < out_desc.GetLength(I2); ++ho)
            {
                for(unsigned wo = 0; wo < out_desc.GetLength(I3); ++wo)
                {
                    for(unsigned c = 0; c < wei_desc.GetLength(I1); ++c)
                    {
                        for(unsigned s = 0; s < wei_desc.GetLength(I2); ++s)
                        {
                            for(unsigned r = 0; r < wei_desc.GetLength(I3); ++r)
                            {
                                const unsigned hi = ho + s;
                                const unsigned wi = wo + r;

                                const unsigned in_index =
                                    in_desc.GetStride(I0) * n + in_desc.GetStride(I1) * c +
                                    in_desc.GetStride(I2) * hi + in_desc.GetStride(I3) * wi;

                                const unsigned wei_index =
                                    wei_desc.GetStride(I0) * k + wei_desc.GetStride(I1) * c +
                                    wei_desc.GetStride(I2) * s + in_desc.GetStride(I3) * r;

                                const unsigned out_index =
                                    out_desc.GetStride(I0) * n + out_desc.GetStride(I1) * k +
                                    out_desc.GetStride(I2) * ho + out_desc.GetStride(I3) * wo;

                                p_out[out_index] += p_wei[wei_index] * p_in[in_index];

#if 0
                                if(threadIdx.x == 0)
                                {
                                    printf("threadwise_direct_convolution: 1: \t"
                                           "threadIdx.x %u\t"
                                           "out_index %u, p_out[out_index] %f, \t"
                                           "wei_index %u, p_wei[wei_index] %f, \t"
                                           "in_index %u, p_in[in_index] %f\n",
                                           threadIdx.x,
                                           out_index,
                                           p_out[out_index],
                                           wei_index,
                                           p_wei[wei_index],
                                           in_index,
                                           p_in[in_index]);
                                }
#endif
                            }
                        }
                    }
                }
            }
        }
    }
}
