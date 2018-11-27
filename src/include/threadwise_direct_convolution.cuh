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
    constexpr auto I0 = Number<0>{};
    constexpr auto I1 = Number<1>{};
    constexpr auto I2 = Number<2>{};
    constexpr auto I3 = Number<3>{};

    constexpr auto in_desc  = InDesc{};
    constexpr auto wei_desc = WeiDesc{};
    constexpr auto out_desc = OutDesc{};

#if 0
    if(blockIdx.x == 0 && threadIdx.x == 0)
    {
        print_ConstantTensorDescriptor(in_desc, "threadwise_direct_convolution: in_desc: ");
        print_ConstantTensorDescriptor(wei_desc, "threadwise_direct_convolution: wei_desc: ");
        print_ConstantTensorDescriptor(out_desc, "threadwise_direct_convolution: out_desc: ");
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

                                const unsigned in_index = in_desc.Get1dIndex(n, c, hi, wi);

                                const unsigned wei_index = wei_desc.Get1dIndex(k, c, s, r);

                                const unsigned out_index = out_desc.Get1dIndex(n, k, ho, wo);

                                p_out[out_index] += p_wei[wei_index] * p_in[in_index];

#if 0
                                //   if(threadIdx.x == 0)
                                {
                                    printf("threadwise_direct_convolution: \t"
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
