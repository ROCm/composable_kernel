#ifndef CK_PRINT_SEQUENCE_HPP
#define CK_PRINT_SEQUENCE_HPP

#include "sequence.hpp"

namespace ck {

template <index_t... Xs>
__host__ __device__ void print_sequence(const char* s, Sequence<Xs...>)
{
    constexpr index_t nsize = Sequence<Xs...>::Size();

    static_assert(nsize <= 10, "wrong!");

    static_if<nsize == 0>{}([&](auto) { printf("%s size %u, {}\n", s, nsize, Xs...); });

    static_if<nsize == 1>{}([&](auto) { printf("%s size %u, {%u}\n", s, nsize, Xs...); });

    static_if<nsize == 2>{}([&](auto) { printf("%s size %u, {%u %u}\n", s, nsize, Xs...); });

    static_if<nsize == 3>{}([&](auto) { printf("%s size %u, {%u %u %u}\n", s, nsize, Xs...); });

    static_if<nsize == 4>{}([&](auto) { printf("%s size %u, {%u %u %u %u}\n", s, nsize, Xs...); });

    static_if<nsize == 5>{}(
        [&](auto) { printf("%s size %u, {%u %u %u %u %u}\n", s, nsize, Xs...); });

    static_if<nsize == 6>{}(
        [&](auto) { printf("%s size %u, {%u %u %u %u %u %u}\n", s, nsize, Xs...); });

    static_if<nsize == 7>{}(
        [&](auto) { printf("%s size %u, {%u %u %u %u %u %u %u}\n", s, nsize, Xs...); });

    static_if<nsize == 8>{}(
        [&](auto) { printf("%s size %u, {%u %u %u %u %u %u %u %u}\n", s, nsize, Xs...); });

    static_if<nsize == 9>{}(
        [&](auto) { printf("%s size %u, {%u %u %u %u %u %u %u %u %u}\n", s, nsize, Xs...); });

    static_if<nsize == 10>{}(
        [&](auto) { printf("%s size %u, {%u %u %u %u %u %u %u %u %u %u}\n", s, nsize, Xs...); });
}

} // namespace ck

#endif
