#pragma once
#include "ConstantTensorDescriptor.hip.hpp"

// need to assume src and dst is aligned
template <class Float, class SrcDesc, class DstDesc, class SrcOpLengths, index_t DataPerRead>
__device__ void threadwise_tensor_slice_copy(SrcDesc,
                                             const Float* __restrict__ p_src,
                                             DstDesc,
                                             Float* __restrict__ p_dst,
                                             SrcOpLengths,
                                             Number<DataPerRead>)
{
    using vector_t = typename vector_type<Float, DataPerRead>::MemoryType;

    constexpr index_t nDim = SrcOpLengths::GetSize();

    static_assert(SrcDesc{}.GetNumOfDimension() == nDim && DstDesc{}.GetNumOfDimension() == nDim,
                  "wrong! dimension not consistent");

    constexpr auto src_desc = SrcDesc{};
    constexpr auto dst_desc = DstDesc{};
    constexpr auto ref_desc = make_ConstantTensorDescriptor_packed(SrcOpLengths{});

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
        static_for<0, nRead, 1>{}([&](auto IRead) {
            constexpr auto multi_id = decltype(Ids){}.PushBack(Number<IRead.Get() * DataPerRead>{});

            const index_t src_index = src_desc.GetOffsetFromMultiIndex(multi_id);

            const index_t dst_index = dst_desc.GetOffsetFromMultiIndex(multi_id);

            *(reinterpret_cast<vector_t*>(&p_dst[dst_index])) =
                *(reinterpret_cast<const vector_t*>(&p_src[src_index]));
        });
    });
}

// access in order of src
template <class SrcData,
          class DstData,
          class SrcDesc,
          class DstDesc,
          class SrcOpLengths,
          class MapDst2Src>
__device__ void
threadwise_tensor_slice_copy_reorder_given_dst2src_v1(SrcDesc,
                                                      const SrcData* __restrict__ p_src,
                                                      DstDesc,
                                                      DstData* __restrict__ p_dst,
                                                      SrcOpLengths,
                                                      MapDst2Src)
{
    constexpr auto src_desc = SrcDesc{};
    constexpr auto dst_desc = DstDesc{};

    ford<SrcOpLengths>{}([&](auto src_multi_id) {
        const auto dst_multi_id = reorder_array_given_new2old(src_multi_id, MapDst2Src{});

        const index_t dst_index = dst_desc.GetOffsetFromMultiIndex(dst_multi_id);

        const index_t src_index = src_desc.GetOffsetFromMultiIndex(src_multi_id);

        p_dst[dst_index] = p_src[src_index];
    });
}

// access in order of dst
template <class SrcData,
          class DstData,
          class SrcDesc,
          class DstDesc,
          class SrcOpLengths,
          class MapDst2Src>
__device__ void
threadwise_tensor_slice_copy_reorder_given_dst2src_v2(SrcDesc,
                                                      const SrcData* __restrict__ p_src,
                                                      DstDesc,
                                                      DstData* __restrict__ p_dst,
                                                      SrcOpLengths,
                                                      MapDst2Src)
{
    constexpr auto src_desc = SrcDesc{};
    constexpr auto dst_desc = DstDesc{};

    constexpr auto dst_op_lengths = SrcOpLengths{}.ReorderGivenNew2Old(MapDst2Src{});

    ford<decltype(dst_op_lengths)>{}([&](auto dst_multi_id) {
        const auto src_multi_id = reorder_array_given_old2new(dst_multi_id, MapDst2Src{});

        const index_t dst_index = dst_desc.GetOffsetFromMultiIndex(dst_multi_id);

        const index_t src_index = src_desc.GetOffsetFromMultiIndex(src_multi_id);

        p_dst[dst_index] = p_src[src_index];
    });
}

// access in order of dst
// manually pack data into vector before write
template <class Float,
          class SrcDesc,
          class DstDesc,
          class SrcOpLengths,
          class MapDst2Src,
          index_t DstDataPerWrite>
__device__ void
threadwise_tensor_slice_copy_reorder_given_dst2src_v3(SrcDesc,
                                                      const Float* __restrict__ p_src,
                                                      DstDesc,
                                                      Float* __restrict__ p_dst,
                                                      SrcOpLengths,
                                                      MapDst2Src,
                                                      Number<DstDataPerWrite>)
{
    using vector_t = typename vector_type<Float, DstDataPerWrite>::MemoryType;

    constexpr index_t nDim = SrcOpLengths::GetSize();

    static_assert(DstDataPerWrite == 1 || DstDesc{}.GetStride(Number<nDim - 1>{}) == 1,
                  "wrong! only support dst.stride[nDim-1] == 1, if DstDataPerWrite != 1");

    static_assert(DstDataPerWrite == 1 || DstDataPerWrite == 2 || DstDataPerWrite == 4,
                  "wrong! only support DstDataPerWrite == 1, 2 or 4");

    static_assert(
        DstDesc{}.GetStride(Number<nDim - 2>{}) % DstDataPerWrite == 0,
        "wrong! dst.stride[nDim-2] should be multiple of DstDataPerWrite to keep alignment");

    constexpr auto src_desc = SrcDesc{};
    constexpr auto dst_desc = DstDesc{};

    constexpr auto dst_op_lengths = SrcOpLengths{}.ReorderGivenNew2Old(MapDst2Src{});

    constexpr index_t L_Dst_Back = dst_op_lengths.Back();

    static_assert(L_Dst_Back % DstDataPerWrite == 0,
                  "wrong! dst.lengths[nDim-1] should be evenly divided by DstDataPerWrite");

    constexpr index_t nWrite = L_Dst_Back / DstDataPerWrite;

    ford<decltype(dst_op_lengths.PopBack())>{}([&](auto ids) {
        static_for<0, nWrite, 1>{}([&](auto IWrite) {
            vector_t dst_vec_data;

            // pack data
            static_for<0, DstDataPerWrite, 1>{}([&](auto IDstData) {
                const auto dst_multi_id =
                    ids.PushBack(IWrite.Get() * DstDataPerWrite + IDstData.Get());

                const auto src_multi_id = reorder_array_given_old2new(dst_multi_id, MapDst2Src{});

                const index_t src_index = src_desc.GetOffsetFromMultiIndex(src_multi_id);

                vector_type<Float, DstDataPerWrite>::SetScalar(
                    dst_vec_data, p_src[src_index], IDstData);
            });

            // write data
            const auto dst_multi_id = ids.PushBack(IWrite.Get() * DstDataPerWrite);

            const index_t dst_index = dst_desc.GetOffsetFromMultiIndex(dst_multi_id);

            *(reinterpret_cast<vector_t*>(&p_dst[dst_index])) = dst_vec_data;
        });
    });
}
