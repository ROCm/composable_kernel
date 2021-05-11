#ifndef CK_CLUSTER_DESCRIPTOR_HPP
#define CK_CLUSTER_DESCRIPTOR_HPP

#include "common_header.hpp"

// TODO remove dependency on deprecated tensor descriptor
#include "tensor_descriptor.hpp"
#include "tensor_adaptor.hpp"

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

#if 1
template <typename Lengths,
          typename ArrangeOrder = typename arithmetic_sequence_gen<0, Lengths::Size(), 1>::type>
__host__ __device__ constexpr auto make_cluster_descriptor_v2(
    const Lengths& lengths,
    ArrangeOrder order = typename arithmetic_sequence_gen<0, Lengths::Size(), 1>::type{})
{
    constexpr index_t ndim_low = Lengths::Size();

    const auto reordered_lengths = container_reorder_given_new2old(lengths, order);

    const auto low_lengths = generate_tuple(
        [&](auto idim_low) { return reordered_lengths[idim_low]; }, Number<ndim_low>{});

    const auto transform = make_merge_transform(low_lengths);

    constexpr auto low_dim_old_top_ids = ArrangeOrder{};

    constexpr auto up_dim_new_top_ids = Sequence<0>{};

    return make_single_stage_tensor_adaptor(
        make_tuple(transform), make_tuple(low_dim_old_top_ids), make_tuple(up_dim_new_top_ids));
}
#endif

} // namespace ck
#endif
