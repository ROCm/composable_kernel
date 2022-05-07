#pragma once

#include "tuple.hpp"
#include "tensor_adaptor.hpp"
#include "multi_index_transform_helper.hpp"

namespace ck {
namespace tensor_operation {
namespace device {

struct BatchedGemmUtil
{
    template <index_t MPerBlock, index_t NPerBlock>
    static constexpr auto
    MakeBlock2CTileMap(index_t batch_count, index_t M, index_t N, index_t M01 = 1, index_t N01 = 1)
    {
        constexpr auto M1 = MPerBlock;
        constexpr auto N1 = NPerBlock;

        const auto M0 = M / M1;
        const auto N0 = N / N1;

        const auto M00 = M0 / M01;
        const auto N00 = N0 / N01;

        const auto g_m00_m01_n00_n01_to_m0_n0_block_cluster_adaptor =
            make_single_stage_tensor_adaptor(
                make_tuple(make_insert_transform(batch_count),
                           make_unmerge_transform(make_tuple(M00, M01)),
                           make_unmerge_transform(make_tuple(N00, N01))),
                make_tuple(Sequence<>{}, Sequence<0>{}, Sequence<1>{}),
                make_tuple(Sequence<0>{}, Sequence<1, 3>{}, Sequence<2, 4>{}));

        const auto globalblockid_to_m00_m01_n00_n01_block_cluster_adaptor =
            make_single_stage_tensor_adaptor(
                make_tuple(make_merge_transform(make_tuple(batch_count, M00, N00, M01, N01))),
                make_tuple(Sequence<0, 1, 2, 3, 4>{}),
                make_tuple(Sequence<0>{}));

        const auto globalblockid_to_m0_n0_block_cluster_adaptor =
            chain_tensor_adaptors(g_m00_m01_n00_n01_to_m0_n0_block_cluster_adaptor,
                                  globalblockid_to_m00_m01_n00_n01_block_cluster_adaptor);

        return globalblockid_to_m0_n0_block_cluster_adaptor;
    }

    struct ComputePtrOffsetOfStridedBatch
    {
        ComputePtrOffsetOfStridedBatch(index_t BatchStrideA,
                                       index_t BatchStrideB,
                                       index_t BatchStrideC)
            : BatchStrideA_(BatchStrideA), BatchStrideB_(BatchStrideB), BatchStrideC_(BatchStrideC)
        {
        }

        __host__ __device__ constexpr long_index_t GetAPtrOffset(index_t g_idx) const
        {
            return g_idx * static_cast<long_index_t>(BatchStrideA_);
        }

        __host__ __device__ constexpr long_index_t GetBPtrOffset(index_t g_idx) const
        {
            return g_idx * static_cast<long_index_t>(BatchStrideB_);
        }

        __host__ __device__ constexpr long_index_t GetCPtrOffset(index_t g_idx) const
        {
            return g_idx * static_cast<long_index_t>(BatchStrideC_);
        }

        private:
        index_t BatchStrideA_;
        index_t BatchStrideB_;
        index_t BatchStrideC_;
    };
};
} // namespace device
} // namespace tensor_operation
} // namespace ck
