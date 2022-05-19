#pragma once
#include <vector>

namespace ck {

template <typename Src, typename Dst>
inline std::vector<Dst> convert_vector_element_type(const std::vector<Src>& inData)
{
    std::vector<Dst> outData;

    for(auto elem : inData)
        outData.push_back(static_cast<Dst>(elem));

    return (outData);
};

}; // namespace ck
