#pragma once
#include <ck/utility/sequence.hpp>

namespace ck_tile::builder {

// Helper struct to get the Sequence type from a constexpr ck::Array.
template <typename T, const T& Arr, typename>
struct ToSequenceHelper;

template <typename T, const T& Arr, std::size_t... Is>
struct ToSequenceHelper<T, Arr, std::index_sequence<Is...>>
{
    using type = ck::Sequence<Arr[Is]...>;
};

// The main interface to get the type
template <auto& Arr>
using ToSequence = typename ToSequenceHelper<
    std::remove_cvref_t<decltype(Arr)>,
    Arr,
    std::make_index_sequence<std::remove_reference_t<decltype(Arr)>::Size()>>::type;

} // namespace ck_tile::builder
