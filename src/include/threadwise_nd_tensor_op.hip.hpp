#pragma once
#include "ConstantTensorDescriptor.hip.hpp"

// need to assume src and dst is aligned
template <class Float, class SrcDesc, class DstDesc, class SrcOpLengths, index_t DataPerRead>
__device__ void threadwise_nd_tensor_copy(SrcDesc,
                                          const Float* __restrict__ p_src,
                                          DstDesc,
                                          Float* __restrict__ p_dst,
                                          SrcOpLengths,
                                          Number<DataPerRead>)
{
    using vector_t = typename vector_type<Float, DataPerRead>::MemoryType;

    constexpr index_t nDim = SrcOpLengths::GetSize();

    static_assert(SrcDesc{}.GetDimension() == nDim && DstDesc{}.GetDimension() == nDim,
                  "wrong! dimension not consistent");

    constexpr auto src_desc = SrcDesc{};
    constexpr auto dst_desc = DstDesc{};
    constexpr auto ref_desc = make_ConstantTensorDescriptor(SrcOpLengths{});

#if 0
    if(get_thread_local_1d_id() == 0 && get_block_1d_id() == 0)
    {
        print_ConstantTensorDescriptor(src_desc, "src_desc");
        print_ConstantTensorDescriptor(dst_desc, "dst_desc");
        print_ConstantTensorDescriptor(ref_desc, "ref_desc");
    }
#endif

    static_assert(DataPerRead == 1 || (SrcDesc{}.GetStride(Number<nDim - 1>{}) == 1 &&
                                       DstDesc{}.GetStride(Number<nDim - 1>{}) == 1),
                  "wrong! only support stride[nDim-1] == 1!\n");

    static_assert(DataPerRead == 1 || DataPerRead == 2 || DataPerRead == 4,
                  "wrong! only support DataPerRead == 1, 2 or 4!\n");

    static_assert(
        SrcDesc{}.GetStride(Number<nDim - 2>{}) % DataPerRead == 0 &&
            DstDesc{}.GetStride(Number<nDim - 2>{}) % DataPerRead == 0,
        "wrong! src and dst stride[nDim-2] should be multiple of DataPerRead to keep alignment");

    constexpr index_t L_Back = SrcOpLengths{}.Back();

    static_assert(L_Back % DataPerRead == 0,
                  "wrong! lengths[nDim-1] should be evenly divided by DataPerRead");

    constexpr index_t nRead = L_Back / DataPerRead;

    static_ford<decltype(ref_desc.GetLengths().PopBack())>{}([=](auto Ids) {
        static_for<0, nRead, 1>{}([=](auto IRead) {
            constexpr auto multi_id = decltype(Ids){}.PushBack(Number<IRead.Get() * DataPerRead>{});

            const index_t src_index = src_desc.Get1dIndex(multi_id);

            const index_t dst_index = dst_desc.Get1dIndex(multi_id);

            *(reinterpret_cast<vector_t*>(&p_dst[dst_index])) =
                *(reinterpret_cast<const vector_t*>(&p_src[src_index]));
        });
    });
}
