#ifndef CK_BLOCKWISE_3D_TENSOR_OP_HPP
#define CK_BLOCKWISE_3D_TENSOR_OP_HPP

#include "common_header.hpp"
#include "ConstantTensorDescriptor.hpp"

namespace ck {

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
        constexpr index_t read_per_d2 = math::integer_divide_ceil(L2, DataPerRead);

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

        constexpr index_t read_per_d2 = math::integer_divide_ceil(L2, DataPerRead);

        constexpr auto ref_desc = make_ConstantTensorDescriptor(Sequence<L0, L1, read_per_d2>{});

        constexpr index_t NLoop = ref_desc.GetElementSize() / BlockSize;

        auto f_copy = [&](index_t is) {
            index_t did[3];

            did[0] = is / ref_desc.GetStride(I0);

            is -= did[0] * ref_desc.GetStride(I0);

            did[1] = is / ref_desc.GetStride(I1);

            is -= did[1] * ref_desc.GetStride(I1);

            did[2] = is / ref_desc.GetStride(I2);

            const index_t src_index =
                src_desc.GetOffsetFromMultiIndex(did[0], did[1], did[2] * DataPerRead);
            const index_t dst_index =
                dst_desc.GetOffsetFromMultiIndex(did[0], did[1], did[2] * DataPerRead);

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

// starting point need to be aligned to float4 or float2 or float
// stride3 need to be 1 for both source and destination
template <index_t BlockSize,
          class Float,
          class SrcDesc,
          class DstDesc,
          class CopyLengths,
          class ThreadPerDims,
          index_t DataPerRead>
struct Blockwise3dTensorCopy3
{
    using vector_t = typename vector_type<Float, DataPerRead>::MemoryType;

    index_t mSrcMyThreadOffset;
    index_t mDstMyThreadOffset;

    __device__ Blockwise3dTensorCopy3()
    {
        constexpr auto I0 = Number<0>{};
        constexpr auto I1 = Number<1>{};
        constexpr auto I2 = Number<2>{};

        static_assert(DataPerRead == 1 ||
                          (SrcDesc{}.GetStride(I2) == 1 && DstDesc{}.GetStride(I2) == 1),
                      "wrong! only support stride3 == 1 if DataPerRead > 1!\n");

        static_assert(DataPerRead == 1 || DataPerRead == 2 || DataPerRead == 4,
                      "wrong! only support DataPerRead == 1, 2 or 4!\n");

        static_assert(
            SrcDesc{}.GetStride(I1) % DataPerRead == 0 &&
                DstDesc{}.GetStride(I1) % DataPerRead == 0,
            "wrong! src and dst stride1 should be multiple of DataPerRead to keep alignment");

        constexpr index_t L0 = CopyLengths{}.Get(I0);
        constexpr index_t L1 = CopyLengths{}.Get(I1);
        constexpr index_t L2 = CopyLengths{}.Get(I2);

        constexpr index_t thread_per_d0 = ThreadPerDims{}.Get(I0);
        constexpr index_t thread_per_d1 = ThreadPerDims{}.Get(I1);
        constexpr index_t thread_per_d2 = ThreadPerDims{}.Get(I2);

        // we allow out-of-bound read from src in D2 dimension,
        //   but we need to make sure dst stride is big enough,
        //   so that the out-of-bound write won't contaminate next line in dst
        constexpr index_t nloop_d2 = math::integer_divide_ceil(L2, thread_per_d2 * DataPerRead);

        static_assert(nloop_d2 * thread_per_d2 * DataPerRead <= DstDesc{}.GetStride(I1),
                      "wrong! out-of-bound write will contaminate next line!\n");

        static_assert(L0 % thread_per_d0 == 0 && L1 % thread_per_d1 == 0,
                      "wrong! L0, L1, L2 should be divided evenly!\n");

        static_assert(BlockSize >= thread_per_d0 * thread_per_d1 * thread_per_d2,
                      "wrrong! BlockSize is not big enough for ThreadPerDims!");

        constexpr index_t num_active_thread =
            reduce_on_sequence(ThreadPerDims{}, math::multiplies<index_t>{}, Number<1>{});

        if(BlockSize > num_active_thread)
        {
            if(get_thread_local_1d_id() >= num_active_thread)
            {
                return;
            }
        }

        constexpr auto thread_cluster_desc = make_ConstantTensorDescriptor(ThreadPerDims{});
        const auto thread_multi_id =
            thread_cluster_desc.GetMultiIndexFrom1dIndex(get_thread_local_1d_id());

        mSrcMyThreadOffset = SrcDesc{}.GetOffsetFromMultiIndex(
            thread_multi_id[0], thread_multi_id[1], thread_multi_id[2] * DataPerRead);

        mDstMyThreadOffset = DstDesc{}.GetOffsetFromMultiIndex(
            thread_multi_id[0], thread_multi_id[1], thread_multi_id[2] * DataPerRead);
    }

    __device__ void Run(const Float* __restrict__ p_src, Float* __restrict__ p_dst) const
    {
        constexpr auto I0 = Number<0>{};
        constexpr auto I1 = Number<1>{};
        constexpr auto I2 = Number<2>{};

        constexpr index_t L0 = CopyLengths{}.Get(I0);
        constexpr index_t L1 = CopyLengths{}.Get(I1);
        constexpr index_t L2 = CopyLengths{}.Get(I2);

        constexpr index_t thread_per_d0 = ThreadPerDims{}.Get(I0);
        constexpr index_t thread_per_d1 = ThreadPerDims{}.Get(I1);
        constexpr index_t thread_per_d2 = ThreadPerDims{}.Get(I2);

        constexpr index_t num_active_thread = thread_per_d0 * thread_per_d1 * thread_per_d2;

        if(BlockSize > num_active_thread)
        {
            if(get_thread_local_1d_id() >= num_active_thread)
            {
                return;
            }
        }

        constexpr index_t nloop_d0 = L0 / thread_per_d0;
        constexpr index_t nloop_d1 = L1 / thread_per_d1;
        constexpr index_t nloop_d2 = math::integer_divide_ceil(L2, thread_per_d2 * DataPerRead);

#pragma unroll
        for(index_t iloop_d0 = 0; iloop_d0 < nloop_d0; ++iloop_d0)
        {
#pragma unroll
            for(index_t iloop_d1 = 0; iloop_d1 < nloop_d1; ++iloop_d1)
            {
#pragma unroll
                for(index_t iloop_d2 = 0; iloop_d2 < nloop_d2; ++iloop_d2)
                {
                    const index_t src_offset =
                        SrcDesc{}.GetOffsetFromMultiIndex(iloop_d0 * thread_per_d0,
                                                          iloop_d1 * thread_per_d1,
                                                          iloop_d2 * thread_per_d2 * DataPerRead);

                    const index_t dst_offset =
                        DstDesc{}.GetOffsetFromMultiIndex(iloop_d0 * thread_per_d0,
                                                          iloop_d1 * thread_per_d1,
                                                          iloop_d2 * thread_per_d2 * DataPerRead);

                    *(reinterpret_cast<vector_t*>(&p_dst[dst_offset + mDstMyThreadOffset])) = *(
                        reinterpret_cast<const vector_t*>(&p_src[src_offset + mSrcMyThreadOffset]));
                }
            }
        }
    }

    __device__ static constexpr index_t GetRegisterBufferSize()
    {
        static_assert(is_same<Float, float>{}, "wrong! only support float!\n");

        constexpr auto I0 = Number<0>{};
        constexpr auto I1 = Number<1>{};
        constexpr auto I2 = Number<2>{};

        constexpr index_t L0 = CopyLengths{}.Get(I0);
        constexpr index_t L1 = CopyLengths{}.Get(I1);
        constexpr index_t L2 = CopyLengths{}.Get(I2);

        constexpr index_t thread_per_d0 = ThreadPerDims{}.Get(I0);
        constexpr index_t thread_per_d1 = ThreadPerDims{}.Get(I1);
        constexpr index_t thread_per_d2 = ThreadPerDims{}.Get(I2);

        constexpr index_t nloop_d0 = L0 / thread_per_d0;
        constexpr index_t nloop_d1 = L1 / thread_per_d1;
        constexpr index_t nloop_d2 = math::integer_divide_ceil(L2, thread_per_d2 * DataPerRead);

        return DataPerRead * nloop_d0 * nloop_d1 * nloop_d2;
    }

    __device__ void RunLoadRegisterBuffer(const Float* __restrict__ p_src,
                                          Float* __restrict__ p_clipboard) const
    {
        constexpr auto I0 = Number<0>{};
        constexpr auto I1 = Number<1>{};
        constexpr auto I2 = Number<2>{};

        constexpr index_t L0 = CopyLengths{}.Get(I0);
        constexpr index_t L1 = CopyLengths{}.Get(I1);
        constexpr index_t L2 = CopyLengths{}.Get(I2);

        constexpr index_t thread_per_d0 = ThreadPerDims{}.Get(I0);
        constexpr index_t thread_per_d1 = ThreadPerDims{}.Get(I1);
        constexpr index_t thread_per_d2 = ThreadPerDims{}.Get(I2);

        constexpr index_t num_active_thread = thread_per_d0 * thread_per_d1 * thread_per_d2;

        if(BlockSize > num_active_thread)
        {
            if(get_thread_local_1d_id() >= num_active_thread)
            {
                return;
            }
        }

        constexpr index_t nloop_d0 = L0 / thread_per_d0;
        constexpr index_t nloop_d1 = L1 / thread_per_d1;
        constexpr index_t nloop_d2 = math::integer_divide_ceil(L2, thread_per_d2 * DataPerRead);

        constexpr auto clipboard_desc =
            make_ConstantTensorDescriptor(Sequence<nloop_d0, nloop_d1, nloop_d2 * DataPerRead>{});

#pragma unroll
        for(index_t iloop_d0 = 0; iloop_d0 < nloop_d0; ++iloop_d0)
        {
#pragma unroll
            for(index_t iloop_d1 = 0; iloop_d1 < nloop_d1; ++iloop_d1)
            {
#pragma unroll
                for(index_t iloop_d2 = 0; iloop_d2 < nloop_d2; ++iloop_d2)
                {
                    const index_t src_offset =
                        SrcDesc{}.GetOffsetFromMultiIndex(iloop_d0 * thread_per_d0,
                                                          iloop_d1 * thread_per_d1,
                                                          iloop_d2 * thread_per_d2 * DataPerRead);

                    const index_t clipboard_offset = clipboard_desc.GetOffsetFromMultiIndex(
                        iloop_d0, iloop_d1, iloop_d2 * DataPerRead);

                    *(reinterpret_cast<vector_t*>(&p_clipboard[clipboard_offset])) = *(
                        reinterpret_cast<const vector_t*>(&p_src[src_offset + mSrcMyThreadOffset]));
                }
            }
        }
    }

    __device__ void RunStoreRegisterBuffer(const Float* __restrict__ p_clipboard,
                                           Float* __restrict__ p_dst) const
    {
        constexpr auto I0 = Number<0>{};
        constexpr auto I1 = Number<1>{};
        constexpr auto I2 = Number<2>{};

        constexpr index_t L0 = CopyLengths{}.Get(I0);
        constexpr index_t L1 = CopyLengths{}.Get(I1);
        constexpr index_t L2 = CopyLengths{}.Get(I2);

        constexpr index_t thread_per_d0 = ThreadPerDims{}.Get(I0);
        constexpr index_t thread_per_d1 = ThreadPerDims{}.Get(I1);
        constexpr index_t thread_per_d2 = ThreadPerDims{}.Get(I2);

        constexpr index_t num_active_thread = thread_per_d0 * thread_per_d1 * thread_per_d2;

        if(BlockSize > num_active_thread)
        {
            if(get_thread_local_1d_id() >= num_active_thread)
            {
                return;
            }
        }

        constexpr index_t nloop_d0 = L0 / thread_per_d0;
        constexpr index_t nloop_d1 = L1 / thread_per_d1;
        constexpr index_t nloop_d2 = math::integer_divide_ceil(L2, thread_per_d2 * DataPerRead);

        constexpr auto clipboard_desc =
            make_ConstantTensorDescriptor(Sequence<nloop_d0, nloop_d1, nloop_d2 * DataPerRead>{});

#pragma unroll
        for(index_t iloop_d0 = 0; iloop_d0 < nloop_d0; ++iloop_d0)
        {
#pragma unroll
            for(index_t iloop_d1 = 0; iloop_d1 < nloop_d1; ++iloop_d1)
            {
#pragma unroll
                for(index_t iloop_d2 = 0; iloop_d2 < nloop_d2; ++iloop_d2)
                {
                    const index_t clipboard_offset = clipboard_desc.GetOffsetFromMultiIndex(
                        iloop_d0, iloop_d1, iloop_d2 * DataPerRead);

                    const index_t dst_offset =
                        DstDesc{}.GetOffsetFromMultiIndex(iloop_d0 * thread_per_d0,
                                                          iloop_d1 * thread_per_d1,
                                                          iloop_d2 * thread_per_d2 * DataPerRead);

                    *(reinterpret_cast<vector_t*>(&p_dst[dst_offset + mDstMyThreadOffset])) =
                        *(reinterpret_cast<const vector_t*>(&p_clipboard[clipboard_offset]));
                }
            }
        }
    }
};

} // namespace ck

#endif
