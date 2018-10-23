#pragma once
#include "device_tensor.cuh"

template <class TFloat, int NBlockDim>
__global__ void direct_convolution(DeviceTensorDescriptor in_desc,
                                   TFloat* const p_in,
                                   DeviceTensorDescriptor wei_desc,
                                   TFloat* const p_wei,
                                   DeviceTensorDescriptor out_desc,
                                   TFloat* p_out)
{
}
