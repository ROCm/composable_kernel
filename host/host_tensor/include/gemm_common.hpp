#ifndef GEMM_COMMON_HPP
#define GEMM_COMMON_HPP

enum GemmMatrixLayout
{
    MK_KN_MN, // 0
    KM_KN_MN, // 1
    KM_NK_MN, // 2
    MK_NK_MN, // 3
};

#endif
