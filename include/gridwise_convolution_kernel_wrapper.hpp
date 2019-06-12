#ifndef CK_GRIDWISE_CONVOLUTION_KERNEL_WRAPPER
#define CK_GRIDWISE_CONVOLUTION_KERNEL_WRAPPER

template <class GridwiseConvolution, class T>
__global__ void run_gridwise_convolution_kernel(const T* const __restrict__ p_in_global,
                                                const T* const __restrict__ p_wei_global,
                                                T* const __restrict__ p_out_global)
{
    GridwiseConvolution{}.Run(p_in_global, p_wei_global, p_out_global);
}

#endif
