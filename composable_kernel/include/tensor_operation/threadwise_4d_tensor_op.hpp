#ifndef CK_THREADWISE_4D_TENSOR_OP_HPP
#define CK_THREADWISE_4D_TENSOR_OP_HPP

#include "common_header.hpp"
#include "ConstantTensorDescriptor.hpp"

namespace ck {

template <class Float, class Desc, class IDim, class NShift>
__device__ void threadwise_4d_tensor_shift_down(Desc, Float* __restrict__ p, IDim, NShift)
{
    constexpr auto I0 = Number<0>{};
    constexpr auto I1 = Number<1>{};
    constexpr auto I2 = Number<2>{};
    constexpr auto I3 = Number<3>{};

    constexpr auto desc = Desc{};

#if 0
    if(get_thread_local_1d_id() == 0)
    {
        print_ConstantTensorDescriptor(desc, "threadwise_4d_tensor_shift_down: ");
    }
#endif

    constexpr index_t nshift = NShift::mValue;

    constexpr index_t did0_end =
        is_same<decltype(I0), IDim>::value ? desc.GetLength(I0) - nshift : desc.GetLength(I0);

    constexpr index_t did1_end =
        is_same<decltype(I1), IDim>::value ? desc.GetLength(I1) - nshift : desc.GetLength(I1);

    constexpr index_t did2_end =
        is_same<decltype(I2), IDim>::value ? desc.GetLength(I2) - nshift : desc.GetLength(I2);

    constexpr index_t did3_end =
        is_same<decltype(I3), IDim>::value ? desc.GetLength(I3) - nshift : desc.GetLength(I3);

    for(index_t did0 = 0; did0 < did0_end; ++did0)
    {
        for(index_t did1 = 0; did1 < did1_end; ++did1)
        {
            for(index_t did2 = 0; did2 < did2_end; ++did2)
            {
                for(index_t did3 = 0; did3 < did3_end; ++did3)
                {
                    const index_t dindex = desc.GetOffsetFromMultiIndex(did0, did1, did2, did3);

                    const index_t sindex = dindex + nshift * desc.GetStride(IDim{});

                    p[dindex] = p[sindex];
                }
            }
        }
    }
}

} // namespace ck
#endif
