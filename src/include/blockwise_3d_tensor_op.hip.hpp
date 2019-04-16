#pragma once
#include "common.hip.hpp"
#include "ConstantTensorDescriptor.hip.hpp"

template <index_t BlockSize,
          class Float,
          class SrcDesc,
          class DstDesc,
          class CopyLengths,
          index_t DataPerRead>
struct Blockwise3dTensorCopy1
{
    using vector_t = typename vector_type<Float, DataPerRead>::MemoryType;

    __device__ constexpr Blockwise3dTensorCopy1()
    {
        constexpr auto I0 = Number<0>{};
        constexpr auto I1 = Number<1>{};
        constexpr auto I2 = Number<2>{};

        static_assert(DataPerRead == 1 ||
                          (SrcDesc{}.GetStride(I2) == 1 && DstDesc{}.GetStride(I2) == 1),
                      "wrong! only support stride2 == 1 if DataPerRead > 1!\n");

        static_assert(DataPerRead == 1 || DataPerRead == 2 || DataPerRead == 4,
                      "wrong! only support DataPerRead == 1, 2 or 4!\n");

        static_assert(SrcDesc{}.GetStride(I1) % DataPerRead == 0 &&
                          DstDesc{}.GetStride(I1) % DataPerRead == 0,
                      "src and dst stride1 should be multiple of DataPerRead to keep alignment");

        // we allow out-of-bound read from src in D3 dimension,
        //   but we need to make sure dst stride2 is big enough,
        //   so that the out-of-bound write won't contaminate next line in dst
        constexpr index_t L2          = CopyLengths{}.Get(I2);
        constexpr index_t read_per_d2 = integer_divide_ceil(L2, DataPerRead);

        static_assert(read_per_d2 * DataPerRead <= DstDesc{}.GetStride(I1),
                      "wrong! out-of-bound write will contaminate next line!\n");
    }

    __device__ void Run(const Float* __restrict__ p_src, Float* __restrict__ p_dst) const
    {
        constexpr auto I0 = Number<0>{};
        constexpr auto I1 = Number<1>{};
        constexpr auto I2 = Number<2>{};

        constexpr auto src_desc = SrcDesc{};
        constexpr auto dst_desc = DstDesc{};

        constexpr index_t L0 = CopyLengths{}.Get(I0);
        constexpr index_t L1 = CopyLengths{}.Get(I1);
        constexpr index_t L2 = CopyLengths{}.Get(I2);

        constexpr index_t read_per_d2 = integer_divide_ceil(L2, DataPerRead);

        constexpr auto ref_desc = make_ConstantTensorDescriptor(Sequence<L0, L1, read_per_d2>{});

        constexpr index_t NLoop = ref_desc.GetElementSize() / BlockSize;

        auto f_copy = [&](index_t is) {
            index_t did[3];

            did[0] = is / ref_desc.GetStride(I0);

            is -= did[0] * ref_desc.GetStride(I0);

            did[1] = is / ref_desc.GetStride(I1);

            is -= did[1] * ref_desc.GetStride(I1);

            did[2] = is / ref_desc.GetStride(I2);

            const index_t src_index = src_desc.Get1dIndex(did[0], did[1], did[2] * DataPerRead);
            const index_t dst_index = dst_desc.Get1dIndex(did[0], did[1], did[2] * DataPerRead);

            *(reinterpret_cast<vector_t*>(p_dst + dst_index)) =
                *(reinterpret_cast<const vector_t*>(p_src + src_index));
        };

        for(index_t iloop = 0; iloop < NLoop; ++iloop)
        {
            index_t is = get_thread_local_1d_id() + iloop * BlockSize;

            f_copy(is);
        }

        constexpr bool has_tail = (ref_desc.GetElementSize() > NLoop * BlockSize);

        if(has_tail)
        {
            index_t is = get_thread_local_1d_id() + NLoop * BlockSize;

            if(is < ref_desc.GetElementSize())
            {
                f_copy(is);
            }
        }
    }
};
