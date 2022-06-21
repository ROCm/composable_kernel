#include <stdlib.h>
#include "config.hpp"
#include "device_gemm_reduce_xdl_cshuffle.hpp"
#include "element_wise_operation.hpp"
#include "device_5ary_elementwise.hpp"
#include "device_operation_instance.hpp"

namespace ck {
namespace tensor_operation {
namespace device {

using F16 = ck::half_t;
using F32 = float;

using inputType      = F16;
using MeanType       = F32;
using SquareMeanType = F32;
using GammaDataType  = F16;
using BetaDataType   = F16;
using outputType     = F16;

using Normalize = ck::tensor_operation::element_wise::Normalize;
using device_normalize_f16_f32_f32_f16_f16_instances = std::tuple<
    // clang-format off
    //###################|in | mean| square_mean| gamma| beta| out| ComputeDataType|  functor| NDim| MPerThread| in, mean, square_mean, gamma, beta, out ScalarPerVector|
    //###################|in | mean| square_mean| gamma| beta| out| ComputeDataType|  functor| NDim| MPerThread| in, mean, square_mean, gamma, beta, out ScalarPerVector|
    //###################|in | mean| square_mean| gamma| beta| out| ComputeDataType|  functor| NDim| MPerThread| in, mean, square_mean, gamma, beta, out ScalarPerVector|
    //###################|in | mean| square_mean| gamma| beta| out| ComputeDataType|  functor| NDim| MPerThread| in, mean, square_mean, gamma, beta, out ScalarPerVector|
    Device5AryElementwise<F16,  F32,         F32,   F16,  F16, F16,            F32, Normalize,    2,          8,  8,    1,           1,     8,    8,   8                >,
    Device5AryElementwise<F16,  F32,         F32,   F16,  F16, F16,            F32, Normalize,    2,          4,  4,    1,           1,     4,    4,   4                >,
    Device5AryElementwise<F16,  F32,         F32,   F16,  F16, F16,            F32, Normalize,    2,          2,  2,    1,           1,     2,    2,   2                >,
    Device5AryElementwise<F16,  F32,         F32,   F16,  F16, F16,            F32, Normalize,    2,          1,  1,    1,           1,     1,    1,   1                >
    // clang-format on
    >;

void add_device_normalize_f16_f32_f32_f16_f16_instances(
    std::vector<DeviceElementwisePtr<5, 1, 2, Normalize>>& instances)
{
    add_device_operation_instances(instances, device_normalize_f16_f32_f32_f16_f16_instances{});
}

} // namespace device
} // namespace tensor_operation
} // namespace ck
