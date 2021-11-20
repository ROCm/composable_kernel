#ifndef DEVICE_GEMM_HPP
#define DEVICE_GEMM_HPP

#include <iostream>
#include "device_base.hpp"

namespace ck {
namespace tensor_operation {
namespace device {

struct DeviceGemm : public BaseOperator
{
    virtual std::unique_ptr<BaseArgument>
    MakeArgumentPointer(const void* p_a,
                        const void* p_b,
                        void* p_c,
                        ck::index_t M,
                        ck::index_t N,
                        ck::index_t K,
                        ck::index_t StrideA,
                        ck::index_t StrideB,
                        ck::index_t StrideC,
                        std::unique_ptr<BaseGpuOperator> c_element_op_ptr) = 0;

    virtual std::unique_ptr<BaseInvoker> MakeInvokerPointer() = 0;
};

using DeviceGemmPtr = std::unique_ptr<DeviceGemm>;

} // namespace device
} // namespace tensor_operation
} // namespace ck
#endif
