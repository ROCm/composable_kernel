#ifndef CK_THREADWISE_GENERIC_TENSOR_OP_HPP
#define CK_THREADWISE_GENERIC_TENSOR_OP_HPP

#include "common_header.hpp"
#include "ConstantTensorDescriptor_deprecated.hpp"
#include "ConstantMergedTensorDescriptor_deprecated.hpp"

namespace ck {
template <class Float, class TDesc>
__device__ void threadwise_generic_tensor_set_zero(TDesc, Float* __restrict__ p)
{
    static_ford<decltype(TDesc::GetLengths())>{}([&](auto multi_id) {
        constexpr index_t offset = TDesc::GetOffsetFromMultiIndex(multi_id);

        p[offset] = static_cast<Float>(0);
    });
}

} // namespace ck
#endif
