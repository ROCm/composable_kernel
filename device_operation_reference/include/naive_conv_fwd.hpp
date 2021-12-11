#ifndef NAIVE_CONV_FWD_HPP
#define NAIVE_CONV_FWD_HPP

namespace ck {
namespace ref {

/*
 * \brief naive implementation of 3D convolution. Layout is (NDHWC, KZYXC, NDHWK).
 *
 * \param N number of batches
 * \param K number of filters
 * \param C number of channels of weight
 * \param (Di, Hi, Wi) depth, height and width dimension of data
 * \param (Z, Y, X) depth, height and width dimensions of weights
 * \param (Do, Ho, Wo) depth, height and width dimension of output
 * \param (stride_z, stride_y, stride_x) strides
 * \param (dilation_z, dilation_y, dilation_x) dilations
 * \param (pad_z, pad_y, pad_x) pads
 */
template <typename TIn, typename TWei, typename TOut>
__global__ void naive_conv_fwd_ndhwc(const TIn* __restrict__ p_in,
                                     const TWei* __restrict__ p_wei,
                                     TOut* __restrict__ p_out,
                                     int N,
                                     int K,
                                     int C,
                                     int Di,
                                     int Hi,
                                     int Wi,
                                     int Z,
                                     int Y,
                                     int X,
                                     int Do,
                                     int Ho,
                                     int Wo,
                                     int stride_z,
                                     int stride_y,
                                     int stride_x,
                                     int dilation_z,
                                     int dilation_y,
                                     int dilation_x,
                                     int pad_z,
                                     int pad_y,
                                     int pad_x)
{
    const int tid            = blockIdx.x * blockDim.x + threadIdx.x;
    const int num_threads    = blockDim.x * gridDim.x;
    const long output_length = N * Do * Ho * Wo * K;

    const int out_strides[] = {Do * Ho * Wo * K, Ho * Wo * K, Wo * K, K};
    const int in_strides[]  = {Di * Hi * Wi * C, Hi * Wi * C, Wi * C, C};
    const int wei_strides[] = {Z * Y * X * C, Y * X * C, X * C, C};

    for(long ii = tid; ii < output_length; ii += num_threads)
    {
        const int n  = ii / out_strides[0];
        int k        = ii - n * out_strides[0];
        const int dO = k / out_strides[1];
        k -= dO * out_strides[1];
        const int ho = k / out_strides[2];
        k -= ho * out_strides[2];
        const int wo = k / out_strides[3];
        k -= wo * out_strides[3];

        double value = 0.0;

        const TIn* in_n   = p_in + static_cast<long>(n) * in_strides[0];
        const TWei* wei_k = p_wei + static_cast<long>(k) * wei_strides[0];

        for(int z = 0; z < Z; ++z)
        {
            int di              = stride_z * dO - pad_z + dilation_z * z;
            const TIn* in_n_di  = in_n + di * in_strides[1];
            const TWei* wei_k_z = wei_k + z * wei_strides[1];

            for(int y = 0; y < Y; ++y)
            {
                int hi                = stride_y * ho - pad_y + dilation_y * y;
                const TIn* in_n_di_hi = in_n_di + hi * in_strides[2];
                const TWei* wei_k_z_y = wei_k_z + y * wei_strides[2];

                for(int x = 0; x < X; ++x)
                {
                    int wi                   = stride_x * wo - pad_x + dilation_x * x;
                    const TIn* in_n_di_hi_wi = in_n_di_hi + wi * in_strides[3];
                    const TWei* wei_k_z_y_x  = wei_k_z_y + x * wei_strides[3];

                    if(di >= 0 && di < Di && hi >= 0 && hi < Hi && wi >= 0 && wi < Wi)
                    {
                        for(int c = 0; c < C; ++c)
                        {
                            value += static_cast<const double>(in_n_di_hi_wi[c]) *
                                     static_cast<const double>(wei_k_z_y_x[c]);
                        }
                    }
                }
            }
        }

        p_out[ii] = static_cast<TOut>(value);
    }
}
} // namespace ref
} // ck

#endif
