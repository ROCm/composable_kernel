#ifndef CK_THREADWISE_GEMM_PARAM_HPP
#define CK_THREADWISE_GEMM_PARAM_HPP

#include "ck/utility/common_header.hpp"
#include "ck/utility/math.hpp"

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
    int accmulate_c; // if 1, need load C and add into current fma. if 0, direct store out c result
} __attribute__((packed));

} // namespace cpu
} // namespace ck

#endif
