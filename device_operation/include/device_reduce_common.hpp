#ifndef DEVICE_REDUCE_COMMON_HPP
#define DEVICE_REDUCE_COMMON_HPP

#include <vector>

#include "common_header.hpp"
#include "reduction_enums.hpp"
#include "reduction_operator.hpp"

namespace ck {
namespace tensor_operation {
namespace device {

// template <typename preUnaryOpType, typename posUnaryOpType>
// using DeviceReducePtr = std::unique_ptr<DeviceReduce<preUnaryOpType, posUnaryOpType>>;

template <int rank, typename InnerDims>
std::pair<size_t, size_t> get_2d_lengths(const std::vector<int>& inLengths)
{
    static_assert(rank <= 6, "bigger rank size not supported!");

    size_t dim0_total_length = 1;
    size_t dim1_total_length = 1;

    static_for<0, InnerDims::Size(), 1>{}(
        [&](auto i) { dim1_total_length *= inLengths[InnerDims::At(i)]; });

    unsigned int flag = 0;

    static_for<0, InnerDims::Size(), 1>{}([&](auto i) { flag = flag | (0x1 << InnerDims::At(i)); });

    static_for<0, rank, 1>{}([&](auto i) {
        if(!(flag & (0x1 << i.value)))
            dim0_total_length *= inLengths[i.value];
    });

    return std::make_pair(dim0_total_length, dim1_total_length);
};

template <int x, typename Seq>
constexpr bool belong()
{
    bool inside = false;

    static_for<0, Seq::Size(), 1>{}([&](auto i) { inside = (inside || (x == Seq::At(i))); });

    return (inside);
};

template <int rank, typename InnerDims, int start = 0>
constexpr auto get_outer_dims()
{
    static_assert(rank <= 6, "bigger rank size not supported!");

    if constexpr(start >= rank)
        return Sequence<>{};
    else
    {
        if constexpr(!belong<start, InnerDims>())
            return merge_sequences(Sequence<start>{}, get_outer_dims<rank, InnerDims, start + 1>());
        else
            return get_outer_dims<rank, InnerDims, start + 1>();
    };
};

// helper functions using variadic template arguments
template <index_t... Ns>
static auto make_tuple_from_array_and_index_seq(const std::vector<int>& lengths, Sequence<Ns...>)
{
    return make_tuple(static_cast<index_t>(lengths[Ns])...);
};

template <index_t arraySize>
static auto make_tuple_from_array(const std::vector<int>& lengths, Number<arraySize>)
{
    static_assert(arraySize >= 1 && arraySize <= 6, "The tensor should have 1 to 6 dimensions");

    constexpr auto index_seq = typename arithmetic_sequence_gen<0, arraySize, 1>::type{};

    return make_tuple_from_array_and_index_seq(lengths, index_seq);
};

} // namespace device
} // namespace tensor_operation

} // namespace ck
#endif
