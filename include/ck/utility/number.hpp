#ifndef CK_NUMBER_HPP
#define CK_NUMBER_HPP

#include "integral_constant.hpp"

namespace ck {

template <index_t N>
using Number = integral_constant<index_t, N>;

} // namespace ck
#endif
