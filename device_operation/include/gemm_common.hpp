#ifndef GEMM_COMMON_HPP
#define GEMM_COMMON_HPP

enum GemmMatrixLayout
{
    MK_KN_MN, // 0
    MK_NK_MN, // 1
    KM_KN_MN, // 2
    KM_NK_MN, // 3
    MK_KN_NM, // 4
    MK_NK_NM, // 5
    KM_KN_NM, // 6
    KM_NK_NM, // 7
};

enum GemmDataType
{
    F32_F32_F32, // 0
    F16_F16_F16, // 1
};

#endif
