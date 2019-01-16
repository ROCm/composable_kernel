#pragma once
#include "ConstantTensorDescriptor.cuh"

// optimized for scenario if p_in, p_wei, p_out are in register
template <class Float, class InDesc, class WeiDesc, class OutDesc>
__device__ void threadwise_direct_convolution_1(InDesc,
                                                Float* const __restrict__ p_in,
                                                WeiDesc,
                                                Float* const __restrict__ p_wei,
                                                OutDesc,
                                                Float* __restrict__ p_out)
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

// Optimized for scenario if p_in and p_wei are in LDS, p_out are in register
// Copy in and wei into register before doing convolution
template <class Float, class InDesc, class WeiDesc, class OutDesc>
__device__ void threadwise_direct_convolution_2(InDesc,
                                                Float* const __restrict__ p_in,
                                                WeiDesc,
                                                Float* const __restrict__ p_wei,
                                                OutDesc,
                                                Float* __restrict__ p_out)
{
    constexpr auto in_desc  = InDesc{};
    constexpr auto wei_desc = WeiDesc{};
    constexpr auto out_desc = OutDesc{};

    constexpr auto in_reg_desc  = make_ConstantTensorDescriptor(in_desc.GetLengths());
    constexpr auto wei_reg_desc = make_ConstantTensorDescriptor(wei_desc.GetLengths());

    // register
    Float p_in_reg[in_reg_desc.GetElementSpace()];
    Float p_wei_reg[wei_reg_desc.GetElementSpace()];

    // copy input tensor into register
    threadwise_4d_tensor_copy(in_desc, p_in, in_reg_desc, p_in_reg, in_reg_desc.GetLengths());

    // copy input tensor into register
    threadwise_4d_tensor_copy(wei_desc, p_wei, wei_reg_desc, p_wei_reg, wei_reg_desc.GetLengths());

    // do convolution
    threadwise_direct_convolution_1(
        in_reg_desc, p_in_reg, wei_reg_desc, p_wei_reg, out_desc, p_out);
}

// optimized for scenario where p_in and p_wei are in LDS, p_out is in register
// break down a non-1x1 convolution into a sequence of 1x1 convolutions,
// load 1x1 weight into register, and do 1x1 convolution in register.
template <class Float, class InDesc, class WeiDesc, class OutDesc>
__device__ void threadwise_direct_convolution_3(InDesc,
                                                Float* const __restrict__ p_in,
                                                WeiDesc,
                                                Float* const __restrict__ p_wei,
                                                OutDesc,
                                                Float* __restrict__ p_out)
{
    constexpr auto I0 = Number<0>{};
    constexpr auto I1 = Number<1>{};
    constexpr auto I2 = Number<2>{};
    constexpr auto I3 = Number<3>{};

    constexpr auto in_desc  = InDesc{};
    constexpr auto wei_desc = WeiDesc{};
    constexpr auto out_desc = OutDesc{};

    constexpr auto in_reg_desc = make_ConstantTensorDescriptor(Sequence<in_desc.GetLength(I0),
                                                                        in_desc.GetLength(I1),
                                                                        out_desc.GetLength(I2),
                                                                        out_desc.GetLength(I3)>{});

    constexpr auto wei_reg_desc = make_ConstantTensorDescriptor(
        Sequence<wei_desc.GetLength(I0), wei_desc.GetLength(I1), 1, 1>{});

    Float p_in_reg[in_reg_desc.GetElementSpace()];
    Float p_wei_reg[wei_reg_desc.GetElementSpace()];

    constexpr unsigned in_w_new_read = 1;

    constexpr auto in_desc_reg_new_read =
        make_ConstantTensorDescriptor(Sequence<in_reg_desc.GetLength(I0),
                                               in_reg_desc.GetLength(I1),
                                               in_reg_desc.GetLength(I2),
                                               in_w_new_read>{});

#if 0
    // loop over vertical direction
    for(unsigned s = 0; s < wei_desc.GetLength(I2); ++s)
    {
        // read first input
        threadwise_4d_tensor_copy(in_desc,
                                  p_in + in_desc.Get1dIndex(0, 0, s, 0),
                                  in_reg_desc,
                                  p_in_reg,
                                  in_reg_desc.GetLengths());

        // read first 1x1 weight
        threadwise_4d_tensor_copy(wei_desc,
                                  p_wei + wei_desc.Get1dIndex(0, 0, s, 0),
                                  wei_reg_desc,
                                  p_wei_reg,
                                  wei_reg_desc.GetLengths());

        // do first 1x1 conv
        threadwise_direct_convolution_1(
            in_reg_desc, p_in_reg, wei_reg_desc, p_wei_reg, out_desc, p_out);

        // loop over horizontal direction
        for(unsigned r = 1; r < wei_desc.GetLength(I3); ++r)
        {
            // read new weight
            threadwise_4d_tensor_copy(wei_desc,
                                      p_wei + wei_desc.Get1dIndex(0, 0, s, r),
                                      wei_reg_desc,
                                      p_wei_reg,
                                      wei_reg_desc.GetLengths());

            // shift old input to the left
            threadwise_4d_tensor_shift_down(in_reg_desc, p_in_reg, I3, Number<in_w_new_read>{});

            // read new input
            threadwise_4d_tensor_copy(
                in_desc,
                p_in + in_desc.Get1dIndex(0, 0, s, r + in_reg_desc.GetLength(I3) - 1),
                in_reg_desc,
                p_in_reg +
                    in_reg_desc.Get1dIndex(0, 0, 0, in_reg_desc.GetLength(I3) - in_w_new_read),
                in_desc_reg_new_read.GetLengths());

            // do 1x1 conv
            threadwise_direct_convolution_1(
                in_reg_desc, p_in_reg, wei_reg_desc, p_wei_reg, out_desc, p_out);
        }
    }
#elif 1
    // loop over vertical direction
    for(unsigned s = 0; s < wei_desc.GetLength(I2); ++s)
    {
        // loop over horizontal direction
        for(unsigned r = 0; r < wei_desc.GetLength(I3); ++r)
        {
            // read new weight
            threadwise_4d_tensor_copy(wei_desc,
                                      p_wei + wei_desc.Get1dIndex(0, 0, s, r),
                                      wei_reg_desc,
                                      p_wei_reg,
                                      wei_reg_desc.GetLengths());

            // read new input
            threadwise_4d_tensor_copy(in_desc,
                                      p_in + in_desc.Get1dIndex(0, 0, s, r),
                                      in_reg_desc,
                                      p_in_reg,
                                      in_reg_desc.GetLengths());

            // do 1x1 conv
            threadwise_direct_convolution_1(
                in_reg_desc, p_in_reg, wei_reg_desc, p_wei_reg, out_desc, p_out);
        }
    }
#endif
}