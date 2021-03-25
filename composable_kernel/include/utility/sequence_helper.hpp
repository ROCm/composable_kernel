#ifndef CK_SEQUENCE_HELPER_HPP
#define CK_SEQUENCE_HELPER_HPP

#include "sequence_helper.hpp"

namespace ck {

template <typename F, index_t N>
__host__ __device__ constexpr auto generate_sequence(F, Number<N>)
{
    return typename sequence_gen<N, F>::type{};
}

} // namespace ck
#endif
