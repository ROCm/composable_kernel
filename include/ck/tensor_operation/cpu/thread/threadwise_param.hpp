#ifndef CK_THREADWISE_PARAM_HPP
#define CK_THREADWISE_PARAM_HPP

#include "common_header.hpp"
#include "math.hpp"

namespace ck {
namespace cpu {

struct ThreadwiseGemmParam
{
    const void* p_a;
    const void* p_b;
    void* p_c;
    uint64_t Kr;
    uint64_t lda; // in unit of byte
    uint64_t ldb; // in unit of byte
    uint64_t ldc; // in unit of byte
    float alpha;
    uint32_t _pack0;
} __attribute__((packed));

} // namespace cpu
} // namespace ck

#endif
