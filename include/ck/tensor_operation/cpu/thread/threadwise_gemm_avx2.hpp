#ifndef CK_THREADWISE_GEMM_AVX2_HPP
#define CK_THREADWISE_GEMM_AVX2_HPP

#include "common_header.hpp"
#include "tensor_layout.hpp"
#include "math.hpp"
#include "threadwise_param.hpp"

namespace ck {
namespace cpu {

template <typename FloatA,
          typename FloatB,
          typename FloatC,
          index_t Mr,
          index_t Nr,
          typename ALayout, // default is k*m, trans->m*k
          typename BLayout, // default is n/8*k*n8, trans->k*n
          bool NonTemporalStore>
struct ThreadwiseGemmAvx2_MxN_6x16
{
    using ALayout_                          = ALayout;
    using BLayout_                          = BLayout;
    static constexpr auto Mr_               = Mr;
    static constexpr auto Nr_               = Nr;
    static constexpr auto NonTemporalStore_ = NonTemporalStore;

    __host__ constexpr ThreadwiseGemmAvx2_MxN_6x16()
    {
        static_assert(Mr <= 6 && Mr >= 1 && (Nr == 8 || Nr == 16), "wrong! Mr x Nr not valid");
    }
    __host__ static void Run(ThreadwiseGemmParam* param)
    {
        /*  6x16 ukernel
         *
         *         Mat_B
         *        |ymm12   |ymm13   |
         * Mat_A  +--------+--------+
         * ymm14  |ymm0    |ymm1    |  cycle 0
         * ymm15  |ymm2    |ymm3    |  cycle 1
         * ymm14  |ymm4    |ymm5    |  cycle 2
         * ymm15  |ymm6    |ymm7    |  cycle 3
         * ymm14  |ymm8    |ymm9    |  cycle 4
         * ymm15  |ymm10   |ymm11   | Mat_C   cycle 5
         *
         * ALayout:ColumnMajor (k*m), lda not needed
         * ALayout:RowMajor    (m*k), lda = k
         * BLayout:ColumnMajor (n/8*k*n8), ldb = k*n8. At least this should be 8 continuous n for a
         * ymm register BLayout:RowMajor    (k*n), ldb not needed
         *
         * lda/ldb/ldc all in unit of byte
         *
         */

        // clang-format off
        __asm__ __volatile__ (
        "L_GemmAvx2_MxN_6x16_Entry%=:\n"
            ".set   m_Mr,               %c[m_Mr]\n"
            ".set   m_Nr,               %c[m_Nr]\n"
            ".set   m_TransA,           %c[m_TransA]\n"
            ".set   m_TransB,           %c[m_TransB]\n"
            ".set   m_NTStore,          %c[m_NTStore]\n"
            ".set   m_ABytes,           %c[m_ABytes]\n"
            ".set   m_BBytes,           %c[m_BBytes]\n"
            ".set   m_CBytes,           %c[m_CBytes]\n"

            "movq     (%[m_param]),     %%rax\n"    // p_a
            "movq    8(%[m_param]),     %%rbx\n"    // p_b
            "movq   24(%[m_param]),     %%rsi\n"    // Kr
            ".if m_TransA != 0\n"
            "movq   32(%[m_param]),     %%rcx\n"    // lda
            ".endif\n"
            ".if m_TransB == 0\n"
            "movq   40(%[m_param]),     %%rdx\n"    // ldb
            ".endif\n"

            ".macro vbroadcastss_%= r_base, r_stride, i_scale, i_offset, ymm\n"
            ".if \\i_scale != 0\n"
            "vbroadcastss   \\i_offset(\\r_base, \\r_stride, \\i_scale), \\ymm\n"
            ".else\n"
            "vbroadcastss   \\i_offset(\\r_base), \\ymm\n"
            ".endif\n"
            ".endm\n"

            ".macro vmovups_%= r_base, r_stride, i_scale, i_offset, ymm\n"
            ".if \\i_scale != 0\n"
            "vmovups   \\i_offset(\\r_base, \\r_stride, \\i_scale), \\ymm\n"
            ".else\n"
            "vmovups   \\i_offset(\\r_base), \\ymm\n"
            ".endif\n"
            ".endm\n"

            ".macro vpbroadcastw_%= r_base, r_stride, i_scale, i_offset, xmm\n"
            ".if \\i_scale != 0\n"
            "vpbroadcastw   \\i_offset(\\r_base, \\r_stride, \\i_scale), \\xmm\n"
            ".else\n"
            "vpbroadcastw   \\i_offset(\\r_base), \\xmm\n"
            ".endif\n"
            ".endm\n"

            ".macro vcvtph2ps_%= r_base, r_stride, i_scale, i_offset, ymm\n"
            ".if \\i_scale != 0\n"
            "vcvtph2ps   \\i_offset(\\r_base, \\r_stride, \\i_scale), \\ymm\n"
            ".else\n"
            "vcvtph2ps   \\i_offset(\\r_base), \\ymm\n"
            ".endif\n"
            ".endm\n"

            ".macro vbroadcast_a%= i_k, i_m, ymm\n" // A in rax(r8, r9), lda in rcx
            ".if m_ABytes == 4\n"
                ".if m_TransA == 0\n"
                    "vbroadcastss_%= %%rax, 0, 0, (\\i_m + \\i_k * m_Mr) * m_ABytes, \\ymm\n"
                ".else\n"
                    ".if (\\i_m == 0) || (\\i_m == 1) || (\\i_m == 2)\n"
                        "vbroadcastss_%= %%rax, %%rcx, \\i_m, \\i_k * m_ABytes, \\ymm\n"
                    ".else\n"
                        "vbroadcastss_%= %%r8, %%rcx, \\i_m-3, \\i_k * m_ABytes, \\ymm\n"
                    ".endif\n"
                ".endif\n"
            ".else\n"
                ".if m_TransA == 0\n"
                    "vpbroadcastw_%= %%rax, 0, 0, (\\i_m + \\i_k * m_Mr) * m_ABytes, %%xmm15\n"
                ".else\n"
                    ".if (\\i_m == 0) || (\\i_m == 1) || (\\i_m == 2)\n"
                        "vpbroadcastw_%= %%rax, %%rcx, \\i_m, \\i_k * m_ABytes, %%xmm15\n"
                    ".else\n"
                        "vpbroadcastw_%= %%r8, %%rcx, \\i_m-3, \\i_k * m_ABytes, %%xmm15\n"
                    ".endif\n"
                ".endif\n"
                "vcvtph2ps  %%xmm15, \\ymm\n"
            ".endif\n"
            ".endm\n"

            ".macro vload_b%= i_k, i_n, ymm\n" // B in rbx, lda in rdx, i_n should be 0, 1
            ".if m_BBytes == 4\n"
                ".if m_TransB == 0\n"
                    "vmovups_%= %%rbx, %%rdx, \\i_n, \\i_k*m_BBytes*8, \\ymm\n"
                ".else\n"
                    "vmovups_%= %%rbx, 0, 0, (\\i_k*m_Nr + \\i_n*8)*m_BBytes, \\ymm\n"
                ".endif\n"
            ".else\n"
                ".if m_TransB == 0\n"
                    "vcvtph2ps_%= %%rbx, %%rdx, \\i_n, \\i_k*m_BBytes*8, \\ymm\n"
                ".else\n"
                    "vcvtph2ps_%= %%rbx, 0, 0, (\\i_k*m_Nr + \\i_n*8)*m_BBytes, \\ymm\n"
                ".endif\n"
            ".endif\n"
            ".endm\n"

            "                               vxorps %%ymm0,  %%ymm0,  %%ymm0 \n"
            ".if               (m_Nr > 8)\n vxorps %%ymm1,  %%ymm1,  %%ymm1 \n .endif\n"
            ".if (m_Mr > 1)              \n vxorps %%ymm2,  %%ymm2,  %%ymm2 \n .endif\n"
            ".if (m_Mr > 1) && (m_Nr > 8)\n vxorps %%ymm3,  %%ymm3,  %%ymm3 \n .endif\n"
            ".if (m_Mr > 2)              \n vxorps %%ymm4,  %%ymm4,  %%ymm4 \n .endif\n"
            ".if (m_Mr > 2) && (m_Nr > 8)\n vxorps %%ymm5,  %%ymm5,  %%ymm5 \n .endif\n"
            ".if (m_Mr > 3)              \n vxorps %%ymm6,  %%ymm6,  %%ymm6 \n .endif\n"
            ".if (m_Mr > 3) && (m_Nr > 8)\n vxorps %%ymm7,  %%ymm7,  %%ymm7 \n .endif\n"
            ".if (m_Mr > 4)              \n vxorps %%ymm8,  %%ymm8,  %%ymm8 \n .endif\n"
            ".if (m_Mr > 4) && (m_Nr > 8)\n vxorps %%ymm9,  %%ymm9,  %%ymm9 \n .endif\n"
            ".if (m_Mr > 5)              \n vxorps %%ymm10, %%ymm10, %%ymm10\n .endif\n"
            ".if (m_Mr > 5) && (m_Nr > 8)\n vxorps %%ymm11, %%ymm11, %%ymm11\n .endif\n"

            ".if m_TransA != 0\n"
            ".if m_Mr > 3\n"
            "lea    (%%rcx, %%rcx, 2),  %%r9\n"
            "lea    (%%rax, %%r9),  %%r8\n"
            ".endif\n"
            ".endif\n"

            "cmp $4, %%rsi\n"
            "jl L_GemmAvx2_MxN_6x16_K_Loop_Remain%=\n"
        "L_GemmAvx2_MxN_6x16_K_Loop_Start%=:\n"
            ".irp i_k, 0, 1, 2, 3\n"
            "                               vload_b%= \\i_k, 0,  %%ymm12\n"           // B
            ".if               (m_Nr > 8)\n vload_b%= \\i_k, 1,  %%ymm13\n .endif\n"  // B

            "                               vbroadcast_a%= \\i_k, 0, %%ymm14\n"                  // A broadcast 0
            ".if (m_Mr > 1)              \n vbroadcast_a%= \\i_k, 1, %%ymm15\n .endif\n"         // A broadcast 1
            "                               vfmadd231ps    %%ymm12, %%ymm14, %%ymm0\n"           // 0x0
            ".if               (m_Nr > 8)\n vfmadd231ps    %%ymm13, %%ymm14, %%ymm1\n .endif\n"  // 0x1
            ".if (m_Mr > 1)              \n vfmadd231ps    %%ymm12, %%ymm15, %%ymm2\n .endif\n"  // 1x0
            ".if (m_Mr > 1) && (m_Nr > 8)\n vfmadd231ps    %%ymm13, %%ymm15, %%ymm3\n .endif\n"  // 1x1

            ".if (m_Mr > 2)              \n vbroadcast_a%= \\i_k, 2, %%ymm14\n .endif\n"         // A broadcast 2
            ".if (m_Mr > 3)              \n vbroadcast_a%= \\i_k, 3, %%ymm15\n .endif\n"         // A broadcast 3
            ".if (m_Mr > 2)              \n vfmadd231ps    %%ymm12, %%ymm14, %%ymm4\n .endif\n"  // 2x0
            ".if (m_Mr > 2) && (m_Nr > 8)\n vfmadd231ps    %%ymm13, %%ymm14, %%ymm5\n .endif\n"  // 2x1
            ".if (m_Mr > 3)              \n vfmadd231ps    %%ymm12, %%ymm15, %%ymm6\n .endif\n"  // 3x0
            ".if (m_Mr > 3) && (m_Nr > 8)\n vfmadd231ps    %%ymm13, %%ymm15, %%ymm7\n .endif\n"  // 3x1

            ".if (m_Mr > 4)              \n vbroadcast_a%= \\i_k, 4, %%ymm14\n .endif\n"         // A broadcast 4
            ".if (m_Mr > 5)              \n vbroadcast_a%= \\i_k, 5, %%ymm15\n .endif\n"         // A broadcast 5
            ".if (m_Mr > 4)              \n vfmadd231ps    %%ymm12, %%ymm14, %%ymm8\n .endif\n"  // 4x0
            ".if (m_Mr > 4) && (m_Nr > 8)\n vfmadd231ps    %%ymm13, %%ymm14, %%ymm9\n .endif\n"  // 4x1
            ".if (m_Mr > 5)              \n vfmadd231ps    %%ymm12, %%ymm15, %%ymm10\n .endif\n" // 5x0
            ".if (m_Mr > 5) && (m_Nr > 8)\n vfmadd231ps    %%ymm13, %%ymm15, %%ymm11\n .endif\n" // 5x1
            ".endr\n"

            ".if m_TransA != 0\n"
            "               lea     4*m_ABytes(%%rax), %%rax\n"
            ".if m_Mr > 3\n lea     4*m_ABytes(%%r8),  %%r8\n  .endif\n"
            ".else\n"
            "               lea     m_Mr * 4 * m_ABytes(%%rax),  %%rax\n"
            ".endif\n"
            ".if m_TransB != 0\n"
            "               lea   m_Nr * 4 * m_BBytes(%%rbx), %%rbx\n"
            ".else\n"
            "               lea   8 * 4 * m_BBytes(%%rbx), %%rbx\n"
            ".endif\n"

            "sub        $4, %%rsi\n"
            "cmp        $4, %%rsi\n"
            "jge        L_GemmAvx2_MxN_6x16_K_Loop_Start%=\n"
            "testq      %%rsi, %%rsi\n"
            "je         L_GemmAvx2_MxN_6x16_K_Loop_End%=\n"
        "L_GemmAvx2_MxN_6x16_K_Loop_Remain%=:\n"
            "                               vload_b%=  0, 0,  %%ymm12\n"             // B
            ".if               (m_Nr > 8)\n vload_b%=  0, 1,  %%ymm13\n .endif\n"    // B

            "                               vbroadcast_a%= 0, 0, %%ymm14\n"                      // A broadcast 0
            ".if (m_Mr > 1)              \n vbroadcast_a%= 0, 1, %%ymm15\n .endif\n"             // A broadcast 1
            "                               vfmadd231ps    %%ymm12, %%ymm14, %%ymm0\n"           // 0x0
            ".if               (m_Nr > 8)\n vfmadd231ps    %%ymm13, %%ymm14, %%ymm1\n .endif\n"  // 0x1
            ".if (m_Mr > 1)              \n vfmadd231ps    %%ymm12, %%ymm15, %%ymm2\n .endif\n"  // 1x0
            ".if (m_Mr > 1) && (m_Nr > 8)\n vfmadd231ps    %%ymm13, %%ymm15, %%ymm3\n .endif\n"  // 1x1

            ".if (m_Mr > 2)              \n vbroadcast_a%= 0, 2, %%ymm14\n .endif\n"             // A broadcast 2
            ".if (m_Mr > 3)              \n vbroadcast_a%= 0, 3, %%ymm15\n .endif\n"             // A broadcast 3
            ".if (m_Mr > 2)              \n vfmadd231ps    %%ymm12, %%ymm14, %%ymm4\n .endif\n"  // 2x0
            ".if (m_Mr > 2) && (m_Nr > 8)\n vfmadd231ps    %%ymm13, %%ymm14, %%ymm5\n .endif\n"  // 2x1
            ".if (m_Mr > 3)              \n vfmadd231ps    %%ymm12, %%ymm15, %%ymm6\n .endif\n"  // 3x0
            ".if (m_Mr > 3) && (m_Nr > 8)\n vfmadd231ps    %%ymm13, %%ymm15, %%ymm7\n .endif\n"  // 3x1

            ".if (m_Mr > 4)              \n vbroadcast_a%= 0, 4, %%ymm14\n .endif\n"    // A broadcast 4
            ".if (m_Mr > 5)              \n vbroadcast_a%= 0, 5, %%ymm15\n .endif\n"    // A broadcast 5
            ".if (m_Mr > 4)              \n vfmadd231ps    %%ymm12, %%ymm14, %%ymm8\n .endif\n"  // 4x0
            ".if (m_Mr > 4) && (m_Nr > 8)\n vfmadd231ps    %%ymm13, %%ymm14, %%ymm9\n .endif\n"  // 4x1
            ".if (m_Mr > 5)              \n vfmadd231ps    %%ymm12, %%ymm15, %%ymm10\n .endif\n" // 5x0
            ".if (m_Mr > 5) && (m_Nr > 8)\n vfmadd231ps    %%ymm13, %%ymm15, %%ymm11\n .endif\n" // 5x1

            ".if m_TransA != 0\n"
            "               lea    m_ABytes(%%rax),    %%rax\n"
            ".if m_Mr > 3\n lea    m_ABytes(%%r8),     %%r8\n    .endif\n"
            ".else\n"
            "               lea    m_Mr * m_ABytes(%%rax), %%rax\n"
            ".endif\n"
            ".if m_TransB != 0\n"
            "               lea    m_Nr * m_BBytes(%%rbx), %%rbx\n"
            ".else\n"
            "               lea    8*m_BBytes(%%rbx), %%rbx\n"
            ".endif\n"

            "sub    $1, %%rsi\n"
            "jne    L_GemmAvx2_MxN_6x16_K_Loop_Remain%=\n"

        "L_GemmAvx2_MxN_6x16_K_Loop_End%=:\n"
            "mov   56(%[m_param]),  %%eax\n"    // alpha
            "cmp    $0x3f800000,    %%eax\n"
            "je     L_GemmAvx2_MxN_6x16_Update_C%=\n"
            "vbroadcastss   56(%[m_param]), %%ymm12\n"
            "                               vmulps    %%ymm12, %%ymm0,  %%ymm0 \n"              // 0x0
            ".if               (m_Nr > 8)\n vmulps    %%ymm12, %%ymm1,  %%ymm1 \n .endif\n"     // 0x1
            ".if (m_Mr > 1)              \n vmulps    %%ymm12, %%ymm2,  %%ymm2 \n .endif\n"     // 1x0
            ".if (m_Mr > 1) && (m_Nr > 8)\n vmulps    %%ymm12, %%ymm3,  %%ymm3 \n .endif\n"     // 1x1

            ".if (m_Mr > 2)              \n vmulps    %%ymm12, %%ymm4,  %%ymm4 \n .endif\n"     // 2x0
            ".if (m_Mr > 2) && (m_Nr > 8)\n vmulps    %%ymm12, %%ymm5,  %%ymm5 \n .endif\n"     // 2x1
            ".if (m_Mr > 3)              \n vmulps    %%ymm12, %%ymm6,  %%ymm6 \n .endif\n"     // 3x0
            ".if (m_Mr > 3) && (m_Nr > 8)\n vmulps    %%ymm12, %%ymm7,  %%ymm7 \n .endif\n"     // 3x1

            ".if (m_Mr > 4)              \n vmulps    %%ymm12, %%ymm8,  %%ymm8 \n .endif\n"     // 4x0
            ".if (m_Mr > 4) && (m_Nr > 8)\n vmulps    %%ymm12, %%ymm9,  %%ymm9 \n .endif\n"     // 4x1
            ".if (m_Mr > 5)              \n vmulps    %%ymm12, %%ymm10, %%ymm10\n .endif\n"     // 5x0
            ".if (m_Mr > 5) && (m_Nr > 8)\n vmulps    %%ymm12, %%ymm11, %%ymm11\n .endif\n"     // 5x1
        "L_GemmAvx2_MxN_6x16_Update_C%=:\n"
            "movq   16(%[m_param]),     %%rax\n"    // p_c
            "movq   48(%[m_param]),     %%rdi\n"    // ldc
            ".if (m_Mr > 1)\n lea  (%%rax, %%rdi, 1), %%rbx\n .endif\n"
            ".if (m_Mr > 2)\n lea  (%%rbx, %%rdi, 1), %%rcx\n .endif\n"
            ".if (m_Mr > 3)\n lea  (%%rcx, %%rdi, 1), %%rdx\n .endif\n"
            ".if (m_Mr > 4)\n lea  (%%rdx, %%rdi, 1), %%r8 \n .endif\n"
            ".if (m_Mr > 5)\n lea  (%%r8,  %%rdi, 1), %%r9 \n .endif\n"

            "                               vaddps  (%%rax),    %%ymm0,  %%ymm0 \n"
            ".if               (m_Nr > 8)\n vaddps  32(%%rax),  %%ymm1,  %%ymm1 \n .endif\n"
            ".if (m_Mr > 1)              \n vaddps  (%%rbx),    %%ymm2,  %%ymm2 \n .endif\n"
            ".if (m_Mr > 1) && (m_Nr > 8)\n vaddps  32(%%rbx),  %%ymm3,  %%ymm3 \n .endif\n"
            ".if (m_Mr > 2)              \n vaddps  (%%rcx),    %%ymm4,  %%ymm4 \n .endif\n"
            ".if (m_Mr > 2) && (m_Nr > 8)\n vaddps  32(%%rcx),  %%ymm5,  %%ymm5 \n .endif\n"
            ".if (m_Mr > 3)              \n vaddps  (%%rdx),    %%ymm6,  %%ymm6 \n .endif\n"
            ".if (m_Mr > 3) && (m_Nr > 8)\n vaddps  32(%%rdx),  %%ymm7,  %%ymm7 \n .endif\n"
            ".if (m_Mr > 4)              \n vaddps  (%%r8),     %%ymm8,  %%ymm8 \n .endif\n"
            ".if (m_Mr > 4) && (m_Nr > 8)\n vaddps  32(%%r8),   %%ymm9,  %%ymm9 \n .endif\n"
            ".if (m_Mr > 5)              \n vaddps  (%%r9),     %%ymm10, %%ymm10\n .endif\n"
            ".if (m_Mr > 5) && (m_Nr > 8)\n vaddps  32(%%r9),   %%ymm11, %%ymm11\n .endif\n"

            "                               vmovups %%ymm0,     (%%rax)  \n"
            ".if               (m_Nr > 8)\n vmovups %%ymm1,     32(%%rax)\n .endif\n"
            ".if (m_Mr > 1)              \n vmovups %%ymm2,     (%%rbx)  \n .endif\n"
            ".if (m_Mr > 1) && (m_Nr > 8)\n vmovups %%ymm3,     32(%%rbx)\n .endif\n"
            ".if (m_Mr > 2)              \n vmovups %%ymm4,     (%%rcx)  \n .endif\n"
            ".if (m_Mr > 2) && (m_Nr > 8)\n vmovups %%ymm5,     32(%%rcx)\n .endif\n"
            ".if (m_Mr > 3)              \n vmovups %%ymm6,     (%%rdx)  \n .endif\n"
            ".if (m_Mr > 3) && (m_Nr > 8)\n vmovups %%ymm7,     32(%%rdx)\n .endif\n"
            ".if (m_Mr > 4)              \n vmovups %%ymm8,     (%%r8)   \n .endif\n"
            ".if (m_Mr > 4) && (m_Nr > 8)\n vmovups %%ymm9,     32(%%r8) \n .endif\n"
            ".if (m_Mr > 5)              \n vmovups %%ymm10,    (%%r9)   \n .endif\n"
            ".if (m_Mr > 5) && (m_Nr > 8)\n vmovups %%ymm11,    32(%%r9) \n .endif\n"
        "L_GemmAvx2_MxN_6x16_Exit%=:\n"
            :
            :
            [m_Mr]          "i" (Mr),
            [m_Nr]          "i" (Nr),
            [m_TransA]      "i" (std::is_same<ck::tensor_layout::gemm::RowMajor, ALayout>::value ? 1 : 0),
            [m_TransB]      "i" (std::is_same<ck::tensor_layout::gemm::RowMajor, BLayout>::value ? 1 : 0),
            [m_NTStore]     "i" (NonTemporalStore),
            [m_ABytes]      "i" (sizeof(FloatA)),
            [m_BBytes]      "i" (sizeof(FloatB)),
            [m_CBytes]      "i" (sizeof(FloatC)),
            [m_param]       "r" (param)
            :
            "cc",
            "rax","rbx","rcx","rdx","rsi","rdi","r8","r9",
            "ymm0","ymm1","ymm2","ymm3","ymm4","ymm5","ymm6",
            "ymm7","ymm8","ymm9","ymm10","ymm11","ymm12","ymm13",
            "ymm14","ymm15"
        );
        // clang-format on
    }
};

template <typename FloatA,
          typename FloatB,
          typename FloatC,
          index_t Mr,
          index_t Nr,
          typename ALayout, // default is k*m, trans->m*k
          typename BLayout, // default is n/8*k*n8, trans->k*n
          bool NonTemporalStore>
struct ThreadwiseGemmAvx2_MxN_4x24
{
    using ALayout_                          = ALayout;
    using BLayout_                          = BLayout;
    static constexpr auto Mr_               = Mr;
    static constexpr auto Nr_               = Nr;
    static constexpr auto NonTemporalStore_ = NonTemporalStore;

    __host__ constexpr ThreadwiseGemmAvx2_MxN_4x24()
    {
        static_assert(Mr <= 4 && Mr >= 1 && (Nr == 8 || Nr == 16 || Nr == 24),
                      "wrong! Mr x Nr not valid");
    }
    __host__ static void Run(ThreadwiseGemmParam* param)
    {
        /*  4x24 ukernel
         *
         *         Mat_B
         *        |ymm12   |ymm13   |ymm14   |
         * Mat_A  +--------+--------+--------+
         * ymm15  |ymm0    |ymm1    |ymm2    |
         *        |ymm3    |ymm4    |ymm5    |
         *        |ymm6    |ymm7    |ymm8    |
         *        |ymm9    |ymm10   |ymm11   |
         *
         * ALayout:ColumnMajor (k*m), lda not needed
         * ALayout:RowMajor    (m*k), lda = k
         * BLayout:ColumnMajor (n/8*k*n8), ldb = k*n8. At least this should be 8 continuous n for a
         * ymm register BLayout:RowMajor    (k*n), ldb not needed
         *
         * lda/ldb/ldc all in unit of byte
         *
         */

        // clang-format off
        __asm__ __volatile__ (
        "L_GemmAvx2_MxN_4x24_Entry%=:\n"
            ".set   m_Mr,               %c[m_Mr]\n"
            ".set   m_Nr,               %c[m_Nr]\n"
            ".set   m_TransA,           %c[m_TransA]\n"
            ".set   m_TransB,           %c[m_TransB]\n"
            ".set   m_NTStore,          %c[m_NTStore]\n"
            ".set   m_ABytes,           %c[m_ABytes]\n"
            ".set   m_BBytes,           %c[m_BBytes]\n"
            ".set   m_CBytes,           %c[m_CBytes]\n"

            "movq     (%[m_param]),     %%rax\n"    // p_a
            "movq    8(%[m_param]),     %%rbx\n"    // p_b
            "movq   24(%[m_param]),     %%rsi\n"    // Kr
            ".if m_TransA != 0\n"
            "movq   32(%[m_param]),     %%rcx\n"    // lda
            ".endif\n"
            ".if m_TransB == 0\n"
            "movq   40(%[m_param]),     %%rdx\n"    // ldb
            ".endif\n"

            ".macro vbroadcastss_%= r_base, r_stride, i_scale, i_offset, ymm\n"
            ".if \\i_scale != 0\n"
            "vbroadcastss   \\i_offset(\\r_base, \\r_stride, \\i_scale), \\ymm\n"
            ".else\n"
            "vbroadcastss   \\i_offset(\\r_base), \\ymm\n"
            ".endif\n"
            ".endm\n"

            ".macro vmovups_%= r_base, r_stride, i_scale, i_offset, ymm\n"
            ".if \\i_scale != 0\n"
            "vmovups   \\i_offset(\\r_base, \\r_stride, \\i_scale), \\ymm\n"
            ".else\n"
            "vmovups   \\i_offset(\\r_base), \\ymm\n"
            ".endif\n"
            ".endm\n"

            ".macro vpbroadcastw_%= r_base, r_stride, i_scale, i_offset, xmm\n"
            ".if \\i_scale != 0\n"
            "vpbroadcastw   \\i_offset(\\r_base, \\r_stride, \\i_scale), \\xmm\n"
            ".else\n"
            "vpbroadcastw   \\i_offset(\\r_base), \\xmm\n"
            ".endif\n"
            ".endm\n"

            ".macro vcvtph2ps_%= r_base, r_stride, i_scale, i_offset, ymm\n"
            ".if \\i_scale != 0\n"
            "vcvtph2ps   \\i_offset(\\r_base, \\r_stride, \\i_scale), \\ymm\n"
            ".else\n"
            "vcvtph2ps   \\i_offset(\\r_base), \\ymm\n"
            ".endif\n"
            ".endm\n"

            ".macro vbroadcast_a%= i_k, i_m, ymm\n" // A in rax(r8), lda in rcx
            ".if m_ABytes == 4\n"
                ".if m_TransA == 0\n"
                    "vbroadcastss_%= %%rax, 0, 0, (\\i_m + \\i_k * m_Mr) * m_ABytes, \\ymm\n"
                ".else\n"
                    ".if (\\i_m == 0) || (\\i_m == 1)\n"
                        "vbroadcastss_%= %%rax, %%rcx, \\i_m, \\i_k * m_ABytes, \\ymm\n"
                    ".else\n"
                        "vbroadcastss_%= %%r8, %%rcx, \\i_m-2, \\i_k * m_ABytes, \\ymm\n"
                    ".endif\n"
                ".endif\n"
            ".else\n"
                ".if m_TransA == 0\n"
                    "vpbroadcastw_%= %%rax, 0, 0, (\\i_m + \\i_k * m_Mr) * m_ABytes, %%xmm15\n"
                ".else\n"
                    ".if (\\i_m == 0) || (\\i_m == 1)\n"
                        "vpbroadcastw_%= %%rax, %%rcx, \\i_m, \\i_k * m_ABytes, %%xmm15\n"
                    ".else\n"
                        "vpbroadcastw_%= %%r8, %%rcx, \\i_m-2, \\i_k * m_ABytes, %%xmm15\n"
                    ".endif\n"
                ".endif\n"
                "vcvtph2ps  %%xmm15, \\ymm\n"
            ".endif\n"
            ".endm\n"

            ".macro vload_b%= i_k, i_n, ymm\n" // B in rbx, lda in rdx, i_n should be 0, 1, 2
            ".if m_BBytes == 4\n"
                ".if m_TransB == 0\n"
                    "vmovups_%= %%rbx, %%rdx, \\i_n, \\i_k*m_BBytes*8, \\ymm\n"
                ".else\n"
                    "vmovups_%= %%rbx, 0, 0, (\\i_k*m_Nr + \\i_n*8)*m_BBytes, \\ymm\n"
                ".endif\n"
            ".else\n"
                ".if m_TransB == 0\n"
                    "vcvtph2ps_%= %%rbx, %%rdx, \\i_n, \\i_k*m_BBytes*8, \\ymm\n"
                ".else\n"
                    "vcvtph2ps_%= %%rbx, 0, 0, (\\i_k*m_Nr + \\i_n*8)*m_BBytes, \\ymm\n"
                ".endif\n"
            ".endif\n"
            ".endm\n"

            "                               vxorps %%ymm0,  %%ymm0,  %%ymm0 \n"
            ".if               (m_Nr > 8)\n vxorps %%ymm1,  %%ymm1,  %%ymm1 \n .endif\n"
            ".if               (m_Nr >16)\n vxorps %%ymm2,  %%ymm2,  %%ymm2 \n .endif\n"
            ".if (m_Mr > 1)              \n vxorps %%ymm3,  %%ymm3,  %%ymm3 \n .endif\n"
            ".if (m_Mr > 1) && (m_Nr > 8)\n vxorps %%ymm4,  %%ymm4,  %%ymm4 \n .endif\n"
            ".if (m_Mr > 1) && (m_Nr >16)\n vxorps %%ymm5,  %%ymm5,  %%ymm5 \n .endif\n"
            ".if (m_Mr > 2)              \n vxorps %%ymm6,  %%ymm6,  %%ymm6 \n .endif\n"
            ".if (m_Mr > 2) && (m_Nr > 8)\n vxorps %%ymm7,  %%ymm7,  %%ymm7 \n .endif\n"
            ".if (m_Mr > 2) && (m_Nr >16)\n vxorps %%ymm8,  %%ymm8,  %%ymm8 \n .endif\n"
            ".if (m_Mr > 3)              \n vxorps %%ymm9,  %%ymm9,  %%ymm9 \n .endif\n"
            ".if (m_Mr > 3) && (m_Nr > 8)\n vxorps %%ymm10, %%ymm10, %%ymm10\n .endif\n"
            ".if (m_Mr > 3) && (m_Nr > 8)\n vxorps %%ymm11, %%ymm11, %%ymm11\n .endif\n"

            ".if m_TransA != 0\n"
            ".if m_Mr > 2\n"
            "lea    (%%rax, %%rcx, 2),  %%r8\n"
            ".endif\n"
            ".endif\n"

            "cmp $4, %%rsi\n"
            "jl L_GemmAvx2_MxN_4x24_K_Loop_Remain%=\n"
        "L_GemmAvx2_MxN_4x24_K_Loop_Start%=:\n"
            ".irp i_k, 0, 1, 2, 3\n"
            "                               vload_b%= \\i_k, 0,  %%ymm12\n"           // B
            ".if               (m_Nr > 8)\n vload_b%= \\i_k, 1,  %%ymm13\n .endif\n"  // B
            ".if               (m_Nr >16)\n vload_b%= \\i_k, 2,  %%ymm14\n .endif\n"  // B

            "                               vbroadcast_a%= \\i_k, 0, %%ymm15\n"                  // A broadcast 0
            "                               vfmadd231ps    %%ymm12, %%ymm15, %%ymm0\n"           // 0x0
            ".if               (m_Nr > 8)\n vfmadd231ps    %%ymm13, %%ymm15, %%ymm1\n .endif\n"  // 0x1
            ".if               (m_Nr >16)\n vfmadd231ps    %%ymm14, %%ymm15, %%ymm2\n .endif\n"  // 0x2

            ".if (m_Mr > 1)              \n vbroadcast_a%= \\i_k, 1, %%ymm15\n .endif\n"         // A broadcast 1
            ".if (m_Mr > 1)              \n vfmadd231ps    %%ymm12, %%ymm15, %%ymm3\n .endif\n"  // 1x0
            ".if (m_Mr > 1) && (m_Nr > 8)\n vfmadd231ps    %%ymm13, %%ymm15, %%ymm4\n .endif\n"  // 1x1
            ".if (m_Mr > 1) && (m_Nr >16)\n vfmadd231ps    %%ymm14, %%ymm15, %%ymm5\n .endif\n"  // 1x2
            
            ".if (m_Mr > 2)              \n vbroadcast_a%= \\i_k, 2, %%ymm15\n .endif\n"         // A broadcast 2
            ".if (m_Mr > 2)              \n vfmadd231ps    %%ymm12, %%ymm15, %%ymm6\n .endif\n"  // 2x0
            ".if (m_Mr > 2) && (m_Nr > 8)\n vfmadd231ps    %%ymm13, %%ymm15, %%ymm7\n .endif\n"  // 2x1
            ".if (m_Mr > 2) && (m_Nr >16)\n vfmadd231ps    %%ymm14, %%ymm15, %%ymm8\n .endif\n"  // 2x2

            ".if (m_Mr > 3)              \n vbroadcast_a%= \\i_k, 3, %%ymm15\n .endif\n"         // A broadcast 3
            ".if (m_Mr > 3)              \n vfmadd231ps    %%ymm12, %%ymm15, %%ymm9\n  .endif\n" // 3x0
            ".if (m_Mr > 3) && (m_Nr > 8)\n vfmadd231ps    %%ymm13, %%ymm15, %%ymm10\n .endif\n" // 3x1
            ".if (m_Mr > 3) && (m_Nr >16)\n vfmadd231ps    %%ymm14, %%ymm15, %%ymm11\n .endif\n" // 3x2
            ".endr\n"

            ".if m_TransA != 0\n"
            "               lea     4*m_ABytes(%%rax), %%rax\n"
            ".if m_Mr > 2\n lea     4*m_ABytes(%%r8),  %%r8\n  .endif\n"
            ".else\n"
            "               lea     m_Mr * 4 * m_ABytes(%%rax),  %%rax\n"
            ".endif\n"
            ".if m_TransB != 0\n"
            "               lea   m_Nr * 4 * m_BBytes(%%rbx), %%rbx\n"
            ".else\n"
            "               lea   8 * 4 * m_BBytes(%%rbx), %%rbx\n"
            ".endif\n"

            "sub        $4, %%rsi\n"
            "cmp        $4, %%rsi\n"
            "jge        L_GemmAvx2_MxN_4x24_K_Loop_Start%=\n"
            "testq      %%rsi, %%rsi\n"
            "je         L_GemmAvx2_MxN_4x24_K_Loop_End%=\n"
        "L_GemmAvx2_MxN_4x24_K_Loop_Remain%=:\n"
            "                               vload_b%= 0, 0,  %%ymm12\n"           // B
            ".if               (m_Nr > 8)\n vload_b%= 0, 1,  %%ymm13\n .endif\n"  // B
            ".if               (m_Nr >16)\n vload_b%= 0, 2,  %%ymm14\n .endif\n"  // B

            "                               vbroadcast_a%= 0, 0, %%ymm15\n"                  // A broadcast 0
            "                               vfmadd231ps    %%ymm12, %%ymm15, %%ymm0\n"           // 0x0
            ".if               (m_Nr > 8)\n vfmadd231ps    %%ymm13, %%ymm15, %%ymm1\n .endif\n"  // 0x1
            ".if               (m_Nr >16)\n vfmadd231ps    %%ymm14, %%ymm15, %%ymm2\n .endif\n"  // 0x2

            ".if (m_Mr > 1)              \n vbroadcast_a%= 0, 1, %%ymm15\n .endif\n"         // A broadcast 1
            ".if (m_Mr > 1)              \n vfmadd231ps    %%ymm12, %%ymm15, %%ymm3\n .endif\n"  // 1x0
            ".if (m_Mr > 1) && (m_Nr > 8)\n vfmadd231ps    %%ymm13, %%ymm15, %%ymm4\n .endif\n"  // 1x1
            ".if (m_Mr > 1) && (m_Nr >16)\n vfmadd231ps    %%ymm14, %%ymm15, %%ymm5\n .endif\n"  // 1x2
            
            ".if (m_Mr > 2)              \n vbroadcast_a%= 0, 2, %%ymm15\n .endif\n"         // A broadcast 2
            ".if (m_Mr > 2)              \n vfmadd231ps    %%ymm12, %%ymm15, %%ymm6\n .endif\n"  // 2x0
            ".if (m_Mr > 2) && (m_Nr > 8)\n vfmadd231ps    %%ymm13, %%ymm15, %%ymm7\n .endif\n"  // 2x1
            ".if (m_Mr > 2) && (m_Nr >16)\n vfmadd231ps    %%ymm14, %%ymm15, %%ymm8\n .endif\n"  // 2x2

            ".if (m_Mr > 3)              \n vbroadcast_a%= 0, 3, %%ymm15\n .endif\n"         // A broadcast 3
            ".if (m_Mr > 3)              \n vfmadd231ps    %%ymm12, %%ymm15, %%ymm9\n  .endif\n" // 3x0
            ".if (m_Mr > 3) && (m_Nr > 8)\n vfmadd231ps    %%ymm13, %%ymm15, %%ymm10\n .endif\n" // 3x1
            ".if (m_Mr > 3) && (m_Nr >16)\n vfmadd231ps    %%ymm14, %%ymm15, %%ymm11\n .endif\n" // 3x2

            ".if m_TransA != 0\n"
            "               lea    m_ABytes(%%rax),    %%rax\n"
            ".if m_Mr > 3\n lea    m_ABytes(%%r8),     %%r8\n    .endif\n"
            ".else\n"
            "               lea    m_Mr * m_ABytes(%%rax), %%rax\n"
            ".endif\n"
            ".if m_TransB != 0\n"
            "               lea    m_Nr * m_BBytes(%%rbx), %%rbx\n"
            ".else\n"
            "               lea    8*m_BBytes(%%rbx), %%rbx\n"
            ".endif\n"

            "sub    $1, %%rsi\n"
            "jne    L_GemmAvx2_MxN_4x24_K_Loop_Remain%=\n"

        "L_GemmAvx2_MxN_4x24_K_Loop_End%=:\n"
            "mov   56(%[m_param]),  %%eax\n"    // alpha
            "cmp    $0x3f800000,    %%eax\n"
            "je     L_GemmAvx2_MxN_4x24_Update_C%=\n"
            "vbroadcastss   56(%[m_param]), %%ymm12\n"           
            "                               vmulps    %%ymm12, %%ymm0,  %%ymm0 \n"           // 0x0
            ".if               (m_Nr > 8)\n vmulps    %%ymm12, %%ymm1,  %%ymm1 \n .endif\n"  // 0x1
            ".if               (m_Nr >16)\n vmulps    %%ymm12, %%ymm2,  %%ymm2 \n .endif\n"  // 0x2
            ".if (m_Mr > 1)              \n vmulps    %%ymm12, %%ymm3,  %%ymm3 \n .endif\n"  // 1x0
            ".if (m_Mr > 1) && (m_Nr > 8)\n vmulps    %%ymm12, %%ymm4,  %%ymm4 \n .endif\n"  // 1x1
            ".if (m_Mr > 1) && (m_Nr >16)\n vmulps    %%ymm12, %%ymm5,  %%ymm5 \n .endif\n"  // 1x2
            ".if (m_Mr > 2)              \n vmulps    %%ymm12, %%ymm6,  %%ymm6 \n .endif\n"  // 2x0
            ".if (m_Mr > 2) && (m_Nr > 8)\n vmulps    %%ymm12, %%ymm7,  %%ymm7 \n .endif\n"  // 2x1
            ".if (m_Mr > 2) && (m_Nr >16)\n vmulps    %%ymm12, %%ymm8,  %%ymm8 \n .endif\n"  // 2x2
            ".if (m_Mr > 3)              \n vmulps    %%ymm12, %%ymm9,  %%ymm9 \n .endif\n"  // 3x0
            ".if (m_Mr > 3) && (m_Nr > 8)\n vmulps    %%ymm12, %%ymm10, %%ymm10\n .endif\n"  // 3x1
            ".if (m_Mr > 3) && (m_Nr >16)\n vmulps    %%ymm12, %%ymm11, %%ymm11\n .endif\n"  // 3x2

        "L_GemmAvx2_MxN_4x24_Update_C%=:\n"
            "movq   16(%[m_param]),     %%rax\n"    // p_c
            "movq   48(%[m_param]),     %%rdi\n"    // ldc
            ".if (m_Mr > 1)\n lea  (%%rax, %%rdi, 1), %%rbx\n .endif\n"
            ".if (m_Mr > 2)\n lea  (%%rbx, %%rdi, 1), %%rcx\n .endif\n"
            ".if (m_Mr > 3)\n lea  (%%rcx, %%rdi, 1), %%rdx\n .endif\n"

            "                               vaddps  (%%rax),    %%ymm0,  %%ymm0 \n"
            ".if               (m_Nr > 8)\n vaddps  32(%%rax),  %%ymm1,  %%ymm1 \n .endif\n"
            ".if               (m_Nr >16)\n vaddps  64(%%rax),  %%ymm2,  %%ymm2 \n .endif\n"
            ".if (m_Mr > 1)              \n vaddps  (%%rbx),    %%ymm3,  %%ymm3 \n .endif\n"
            ".if (m_Mr > 1) && (m_Nr > 8)\n vaddps  32(%%rbx),  %%ymm4,  %%ymm4 \n .endif\n"
            ".if (m_Mr > 1) && (m_Nr >16)\n vaddps  64(%%rbx),  %%ymm5,  %%ymm5 \n .endif\n"
            ".if (m_Mr > 2)              \n vaddps  (%%rcx),    %%ymm6,  %%ymm6 \n .endif\n"
            ".if (m_Mr > 2) && (m_Nr > 8)\n vaddps  32(%%rcx),  %%ymm7,  %%ymm7 \n .endif\n"
            ".if (m_Mr > 2) && (m_Nr >16)\n vaddps  64(%%rcx),  %%ymm8,  %%ymm8 \n .endif\n"
            ".if (m_Mr > 3)              \n vaddps  (%%rdx),    %%ymm9,  %%ymm9 \n .endif\n"
            ".if (m_Mr > 3) && (m_Nr > 8)\n vaddps  32(%%rdx),  %%ymm10, %%ymm10\n .endif\n"
            ".if (m_Mr > 3) && (m_Nr >16)\n vaddps  64(%%rdx),  %%ymm11, %%ymm11\n .endif\n"

            "                               vmovups %%ymm0,     (%%rax)  \n"
            ".if               (m_Nr > 8)\n vmovups %%ymm1,     32(%%rax)\n .endif\n"
            ".if               (m_Nr >16)\n vmovups %%ymm2,     64(%%rax)\n .endif\n"
            ".if (m_Mr > 1)              \n vmovups %%ymm3,     (%%rbx)  \n .endif\n"
            ".if (m_Mr > 1) && (m_Nr > 8)\n vmovups %%ymm4,     32(%%rbx)\n .endif\n"
            ".if (m_Mr > 1) && (m_Nr >16)\n vmovups %%ymm5,     64(%%rbx)\n .endif\n"
            ".if (m_Mr > 2)              \n vmovups %%ymm6,     (%%rcx)  \n .endif\n"
            ".if (m_Mr > 2) && (m_Nr > 8)\n vmovups %%ymm7,     32(%%rcx)\n .endif\n"
            ".if (m_Mr > 2) && (m_Nr >16)\n vmovups %%ymm8,     64(%%rcx)\n .endif\n"
            ".if (m_Mr > 3)              \n vmovups %%ymm9,     (%%rdx)  \n .endif\n"
            ".if (m_Mr > 3) && (m_Nr > 8)\n vmovups %%ymm10,    32(%%rdx)\n .endif\n"
            ".if (m_Mr > 3) && (m_Nr >16)\n vmovups %%ymm11,    64(%%rdx)\n .endif\n"
        "L_GemmAvx2_MxN_4x24_Exit%=:\n"
            :
            :
            [m_Mr]          "i" (Mr),
            [m_Nr]          "i" (Nr),
            [m_TransA]      "i" (std::is_same<ck::tensor_layout::gemm::RowMajor, ALayout>::value ? 1 : 0),
            [m_TransB]      "i" (std::is_same<ck::tensor_layout::gemm::RowMajor, BLayout>::value ? 1 : 0),
            [m_NTStore]     "i" (NonTemporalStore),
            [m_ABytes]      "i" (sizeof(FloatA)),
            [m_BBytes]      "i" (sizeof(FloatB)),
            [m_CBytes]      "i" (sizeof(FloatC)),
            [m_param]       "r" (param)
            :
            "cc",
            "rax","rbx","rcx","rdx","rsi","rdi","r8",
            "ymm0","ymm1","ymm2","ymm3","ymm4","ymm5","ymm6",
            "ymm7","ymm8","ymm9","ymm10","ymm11","ymm12","ymm13",
            "ymm14","ymm15"
        );
        // clang-format on
    }
};

} // namespace cpu
} // namespace ck
#endif
