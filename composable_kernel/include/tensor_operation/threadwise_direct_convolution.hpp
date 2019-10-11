#ifndef CK_THREADWISE_DIRECT_CONVOLUTION_HPP
#define CK_THREADWISE_DIRECT_CONVOLUTION_HPP

#include "common_header.hpp"
#include "ConstantTensorDescriptor_deprecated.hpp"
#include "threadwise_tensor_slice_copy.hpp"

namespace ck {

// optimized for scenario if p_in, p_wei, p_out are in register
template <class TInWei, class TOut, class InDesc, class WeiDesc, class OutDesc>
__device__ void threadwise_direct_convolution_1(InDesc,
                                                TInWei* const __restrict__ p_in,
                                                WeiDesc,
                                                TInWei* const __restrict__ p_wei,
                                                OutDesc,
                                                TOut* __restrict__ p_out)
{
    constexpr auto I0 = Number<0>{};
    constexpr auto I1 = Number<1>{};
    constexpr auto I2 = Number<2>{};
    constexpr auto I3 = Number<3>{};

    constexpr auto in_desc  = InDesc{};
    constexpr auto wei_desc = WeiDesc{};
    constexpr auto out_desc = OutDesc{};

#if 0
    if(blockIdx.x == 0 && get_thread_local_1d_id() == 0)
    {
        print_ConstantTensorDescriptor(in_desc, "threadwise_direct_convolution: in_desc: ");
        print_ConstantTensorDescriptor(wei_desc, "threadwise_direct_convolution: wei_desc: ");
        print_ConstantTensorDescriptor(out_desc, "threadwise_direct_convolution: out_desc: ");
    }
#endif

    for(index_t n = 0; n < out_desc.GetLength(I0); ++n)
    {
        for(index_t k = 0; k < out_desc.GetLength(I1); ++k)
        {
            for(index_t ho = 0; ho < out_desc.GetLength(I2); ++ho)
            {
                for(index_t wo = 0; wo < out_desc.GetLength(I3); ++wo)
                {
                    for(index_t c = 0; c < wei_desc.GetLength(I1); ++c)
                    {
                        for(index_t y = 0; y < wei_desc.GetLength(I2); ++y)
                        {
                            for(index_t x = 0; x < wei_desc.GetLength(I3); ++x)
                            {
                                const index_t hi = ho + y;
                                const index_t wi = wo + x;

                                const index_t in_index =
                                    in_desc.GetOffsetFromMultiIndex(n, c, hi, wi);

                                const index_t wei_index =
                                    wei_desc.GetOffsetFromMultiIndex(k, c, y, x);

                                const index_t out_index =
                                    out_desc.GetOffsetFromMultiIndex(n, k, ho, wo);

                                fused_multiply_accumulate(
                                    p_out[out_index], p_wei[wei_index], p_in[in_index]);
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
template <class TInWei, class TOut, class InDesc, class WeiDesc, class OutDesc>
__device__ void threadwise_direct_convolution_2(InDesc,
                                                TInWei* const __restrict__ p_in,
                                                WeiDesc,
                                                TInWei* const __restrict__ p_wei,
                                                OutDesc,
                                                TOut* __restrict__ p_out)
{
    constexpr auto in_desc  = InDesc{};
    constexpr auto wei_desc = WeiDesc{};
    constexpr auto out_desc = OutDesc{};

    constexpr auto in_reg_desc  = make_ConstantTensorDescriptor_packed(in_desc.GetLengths());
    constexpr auto wei_reg_desc = make_ConstantTensorDescriptor_packed(wei_desc.GetLengths());

    // register
    TInWei p_in_reg[in_reg_desc.GetElementSpace()];
    TInWei p_wei_reg[wei_reg_desc.GetElementSpace()];

    // copy input tensor into register
    threadwise_tensor_slice_copy(
        in_desc, p_in, in_reg_desc, p_in_reg, in_reg_desc.GetLengths(), Number<1>{});

    // copy input tensor into register
    threadwise_tensor_slice_copy(
        wei_desc, p_wei, wei_reg_desc, p_wei_reg, wei_reg_desc.GetLengths(), Number<1>{});

    // do convolution
    threadwise_direct_convolution_1(
        in_reg_desc, p_in_reg, wei_reg_desc, p_wei_reg, out_desc, p_out);
}

// optimized for scenario where p_in and p_wei are in LDS, p_out is in register
// break down a non-1x1 convolution into a sequence of 1x1 convolutions,
// load 1x1 weight into register, and do 1x1 convolution in register.
template <class Data, class InDesc, class WeiDesc, class OutDesc>
__device__ void threadwise_direct_convolution_3(InDesc,
                                                Data* const __restrict__ p_in,
                                                WeiDesc,
                                                Data* const __restrict__ p_wei,
                                                OutDesc,
                                                Data* __restrict__ p_out)
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

    Data p_in_reg[in_reg_desc.GetElementSpace()];
    Data p_wei_reg[wei_reg_desc.GetElementSpace()];

    constexpr index_t in_w_new_read = 1;

    constexpr auto in_desc_reg_new_read =
        make_ConstantTensorDescriptor(Sequence<in_reg_desc.GetLength(I0),
                                               in_reg_desc.GetLength(I1),
                                               in_reg_desc.GetLength(I2),
                                               in_w_new_read>{});

#if 0
    // this verison reused old input data in register, and read new data from LDS
    // loop over vertical direction
    for(index_t y = 0; y < wei_desc.GetLength(I2); ++y)
    {
        // read first input
        threadwise_4d_tensor_copy(in_desc,
                                  p_in + in_desc.GetOffsetFromMultiIndex(0, 0, y, 0),
                                  in_reg_desc,
                                  p_in_reg,
                                  in_reg_desc.GetLengths());

        // read first 1x1 weight
        threadwise_4d_tensor_copy(wei_desc,
                                  p_wei + wei_desc.GetOffsetFromMultiIndex(0, 0, y, 0),
                                  wei_reg_desc,
                                  p_wei_reg,
                                  wei_reg_desc.GetLengths());

        // do first 1x1 conv
        threadwise_direct_convolution_1(
            in_reg_desc, p_in_reg, wei_reg_desc, p_wei_reg, out_desc, p_out);

        // loop over horizontal direction
        for(index_t x = 1; x < wei_desc.GetLength(I3); ++x)
        {
            // read new weight
            threadwise_4d_tensor_copy(wei_desc,
                                      p_wei + wei_desc.GetOffsetFromMultiIndex(0, 0, y, x),
                                      wei_reg_desc,
                                      p_wei_reg,
                                      wei_reg_desc.GetLengths());

            // shift old input to the left
            threadwise_4d_tensor_shift_down(in_reg_desc, p_in_reg, I3, Number<in_w_new_read>{});

            // read new input
            threadwise_4d_tensor_copy(
                in_desc,
                p_in + in_desc.GetOffsetFromMultiIndex(0, 0, y, x + in_reg_desc.GetLength(I3) - 1),
                in_reg_desc,
                p_in_reg +
                    in_reg_desc.GetOffsetFromMultiIndex(0, 0, 0, in_reg_desc.GetLength(I3) - in_w_new_read),
                in_desc_reg_new_read.GetLengths());

            // do 1x1 conv
            threadwise_direct_convolution_1(
                in_reg_desc, p_in_reg, wei_reg_desc, p_wei_reg, out_desc, p_out);
        }
    }
#elif 1
    // this version read all input from LDS when filter moves
    // loop over vertical direction
    for(index_t y = 0; y < wei_desc.GetLength(I2); ++y)
    {
        // loop over horizontal direction
        for(index_t x = 0; x < wei_desc.GetLength(I3); ++x)
        {
            // read new weight
            threadwise_4d_tensor_copy(wei_desc,
                                      p_wei + wei_desc.GetOffsetFromMultiIndex(0, 0, y, x),
                                      wei_reg_desc,
                                      p_wei_reg,
                                      wei_reg_desc.GetLengths());

            // read new input
            threadwise_4d_tensor_copy(in_desc,
                                      p_in + in_desc.GetOffsetFromMultiIndex(0, 0, y, x),
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

} // namespace ck
#endif
