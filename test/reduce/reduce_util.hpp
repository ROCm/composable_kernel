#ifndef REDUCE_UTILS_HPP
#define REDUCE_UTILS_HPP

#include "data_type.hpp"

namespace ck {
namespace reduce_util {

template <typename T>
void to_f32_vector(const Tensor<T>& src, Tensor<float>& dst)
{
    for(int i = 0; i < src.mData.size(); ++i)
        dst.mData[i] = type_convert<float>(src.mData[i]);
}

} // namespace reduce_util

} // namespace ck
#endif
