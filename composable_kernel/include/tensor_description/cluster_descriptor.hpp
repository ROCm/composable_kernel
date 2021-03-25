#ifndef CK_CLUSTER_DESCRIPTOR_HPP
#define CK_CLUSTER_DESCRIPTOR_HPP

#include "common_header.hpp"

// TODO remove dependency on deprecated tensor descriptor
#include "tensor_descriptor.hpp"

namespace ck {

// a cluster map 1d index to N-d index
template <typename Lengths, typename ArrangeOrder>
struct ClusterDescriptor
{
    static constexpr index_t nDim = Lengths::Size();

    static constexpr auto mDesc = transform_tensor_descriptor(
        make_native_tensor_descriptor_packed(Lengths{}),
        make_tuple(Merge<decltype(Lengths::ReorderGivenNew2Old(ArrangeOrder{}))>{}),
        make_tuple(ArrangeOrder{}),
        make_tuple(Sequence<0>{}));

    __host__ __device__ constexpr ClusterDescriptor()
    {
        static_assert(Lengths::Size() == nDim && ArrangeOrder::Size() == nDim,
                      "wrong! size not the same");

        static_assert(is_valid_sequence_map<ArrangeOrder>{}, "wrong! ArrangeOrder is wrong");
    }

    __host__ __device__ static constexpr index_t GetElementSize() { return mDesc.GetElementSize(); }

    __host__ __device__ static constexpr auto CalculateClusterIndex(index_t idx_1d)
    {
        return mDesc.CalculateLowerIndex(MultiIndex<1>{idx_1d});
    }
};

template <typename Lengths,
          typename ArrangeOrder = typename arithmetic_sequence_gen<0, Lengths::Size(), 1>::type>
__host__ __device__ constexpr auto make_cluster_descriptor(
    Lengths, ArrangeOrder order = typename arithmetic_sequence_gen<0, Lengths::Size(), 1>::type{})
{
    return ClusterDescriptor<Lengths, decltype(order)>{};
}

} // namespace ck
#endif
