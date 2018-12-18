#pragma once
#include "constant_tensor_descriptor.cuh"

// optimized for scenario if p_in, p_wei, p_out are in register
template <class TFloat, class InDesc, class WeiDesc, class OutDesc>
__device__ void threadwise_direct_convolution_1(InDesc,
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

// optimized for scenario where p_in and p_wei are in LDS, p_out is in register
// break down a non-1x1 convolution into a sequence of 1x1 convolutions,
// load 1x1 weight into register, and do 1x1 convolution in register.
template <class TFloat, class InDesc, class WeiDesc, class OutDesc>
__device__ void threadwise_direct_convolution_2(InDesc,
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

    constexpr auto in_desc_lds  = InDesc{};
    constexpr auto wei_desc_lds = WeiDesc{};
    constexpr auto out_desc_reg = OutDesc{};

    constexpr auto in_desc_reg =
        make_ConstantTensorDescriptor(Sequence<in_desc_lds.GetLength(I0),
                                               in_desc_lds.GetLength(I1),
                                               out_desc_reg.GetLength(I2),
                                               out_desc_reg.GetLength(I3)>{});

    constexpr auto wei_desc_reg = make_ConstantTensorDescriptor(
        Sequence<wei_desc_lds.GetLength(I0), wei_desc_lds.GetLength(I1), 1, 1>{});

    TFloat p_in_reg[in_desc_reg.GetElementSpace()];
    TFloat p_wei_reg[wei_desc_reg.GetElementSpace()];

    constexpr unsigned in_w_new_read = 1;

    constexpr auto in_desc_reg_new_read =
        make_ConstantTensorDescriptor(Sequence<in_desc_reg.GetLength(I0),
                                               in_desc_reg.GetLength(I1),
                                               in_desc_reg.GetLength(I2),
                                               in_w_new_read>{});

    // loop over vertical direction
    for(unsigned s = 0; s < wei_desc_lds.GetLength(I2); ++s)
    {
#if 1
        // read first input
        threadwise_4d_tensor_copy(in_desc_lds,
                                  p_in + in_desc_lds.Get1dIndex(0, 0, s, 0),
                                  in_desc_reg,
                                  p_in_reg,
                                  in_desc_reg);

        // read first 1x1 weight
        threadwise_4d_tensor_copy(wei_desc_lds,
                                  p_wei + wei_desc_lds.Get1dIndex(0, 0, s, 0),
                                  wei_desc_reg,
                                  p_wei_reg,
                                  wei_desc_reg);

        // do first 1x1 conv
        threadwise_direct_convolution_1(
            in_desc_reg, p_in_reg, wei_desc_reg, p_wei_reg, out_desc_reg, p_out);

        // loop over horizontal direction
        for(unsigned r = 1; r < wei_desc_lds.GetLength(I3); ++r)
        {
            // read new weight
            threadwise_4d_tensor_copy(wei_desc_lds,
                                      p_wei + wei_desc_lds.Get1dIndex(0, 0, s, r),
                                      wei_desc_reg,
                                      p_wei_reg,
                                      wei_desc_reg);

            // shift old input to the left
            threadwise_4d_tensor_shift_down(in_desc_reg, p_in_reg, I3, Number<in_w_new_read>{});

            // read new input
            threadwise_4d_tensor_copy(
                in_desc_lds,
                p_in + in_desc_lds.Get1dIndex(0, 0, s, in_desc_reg.GetLength(I3) + r - 1),
                in_desc_reg,
                p_in_reg +
                    in_desc_reg.Get1dIndex(0, 0, 0, in_desc_reg.GetLength(I3) - in_w_new_read),
                in_desc_reg_new_read);

            // do 1x1 conv
            threadwise_direct_convolution_1(
                in_desc_reg, p_in_reg, wei_desc_reg, p_wei_reg, out_desc_reg, p_out);
        }
#elif 1
        // loop over horizontal direction
        for(unsigned r = 0; r < wei_desc_lds.GetLength(I3); ++r)
        {
            // read new weight
            threadwise_4d_tensor_copy(wei_desc_lds,
                                      p_wei + wei_desc_lds.Get1dIndex(0, 0, s, r),
                                      wei_desc_reg,
                                      p_wei_reg,
                                      wei_desc_reg);

            // read new input
            threadwise_4d_tensor_copy(in_desc_lds,
                                      p_in + in_desc_lds.Get1dIndex(0, 0, s, r),
                                      in_desc_reg,
                                      p_in_reg,
                                      in_desc_reg);

            // do 1x1 conv
            threadwise_direct_convolution_1(
                in_desc_reg, p_in_reg, wei_desc_reg, p_wei_reg, out_desc_reg, p_out);
        }
#endif
    }
}
