#ifndef CK_THREADWISE_TENSOR_SLICE_TRANSFER_AVX2_SPECIALIZED_HPP
#define CK_THREADWISE_TENSOR_SLICE_TRANSFER_AVX2_SPECIALIZED_HPP

#include "common_header.hpp"
#include "data_type_cpu.hpp"
#include "../../gpu/thread/threadwise_tensor_slice_transfer.hpp"
#include "tensor_descriptor.hpp"
#include "tensor_descriptor_helper.hpp"
#include "tensor_space_filling_curve.hpp"
#include "dynamic_buffer_cpu.hpp"
#include "element_wise_operation_cpu.hpp"
#include "convolution_forward_specialization_cpu.hpp"
#include <immintrin.h>

namespace ck {
namespace cpu {

namespace avx2_util {

template <typename ElementwiseOp>
void memcpy32_avx2(void* dst, const void* src, const ck::index_t n, const ElementwiseOp& element_op)
{
    // 16-8-4-2-1 pattern
    ck::index_t i_n    = n;
    float* p_dst       = reinterpret_cast<float*>(dst);
    const float* p_src = reinterpret_cast<const float*>(src);
    while(i_n >= 16)
    {
        _mm256_storeu_ps(p_dst + 0, element_op.Apply(_mm256_loadu_ps(p_src + 0)));
        _mm256_storeu_ps(p_dst + 8, element_op.Apply(_mm256_loadu_ps(p_src + 8)));
        p_dst += 16;
        p_src += 16;
        i_n -= 16;
    }
    if(i_n & 8)
    {
        _mm256_storeu_ps(p_dst, element_op.Apply(_mm256_loadu_ps(p_src)));
        p_dst += 8;
        p_src += 8;
    }
    if(i_n & 4)
    {
        _mm_storeu_ps(p_dst, element_op.Apply(_mm_loadu_ps(p_src)));
        p_dst += 4;
        p_src += 4;
    }
    if(i_n & 2)
    {
#if defined(__GNUC__) && !defined(__clang__) && !defined(__llvm__)
        __m128i s = _mm_loadu_si64(p_src);
        __m128 v  = element_op.Apply(*reinterpret_cast<__m128*>(&s));
        _mm_storeu_si64(p_dst, *reinterpret_cast<__m128i*>(&v));
#else
        _mm_storeu_si64(p_dst, element_op.Apply(_mm_loadu_si64(p_src)));
#endif
        p_dst += 2;
        p_src += 2;
    }
    if(i_n & 1)
    {
        *p_dst = element_op.Apply(*p_src);
    }
}

template <typename ElementwiseOp>
void memcpy32_avx2_with_extra_2src(void* dst,
                                   const void* src,
                                   const void* src1,
                                   const void* src2,
                                   const ck::index_t n,
                                   const ElementwiseOp& element_op)
{
    // 16-8-4-2-1 pattern
    ck::index_t i_n     = n;
    float* p_dst        = reinterpret_cast<float*>(dst);
    const float* p_src  = reinterpret_cast<const float*>(src);
    const float* p_src1 = reinterpret_cast<const float*>(src1);
    const float* p_src2 = reinterpret_cast<const float*>(src2);
    while(i_n >= 16)
    {
        _mm256_storeu_ps(p_dst + 0,
                         element_op.Apply(_mm256_loadu_ps(p_src + 0),
                                          _mm256_loadu_ps(p_src1 + 0),
                                          _mm256_loadu_ps(p_src2 + 0)));
        _mm256_storeu_ps(p_dst + 8,
                         element_op.Apply(_mm256_loadu_ps(p_src + 8),
                                          _mm256_loadu_ps(p_src1 + 8),
                                          _mm256_loadu_ps(p_src2 + 8)));
        p_dst += 16;
        p_src += 16;
        p_src1 += 16;
        p_src2 += 16;
        i_n -= 16;
    }
    if(i_n & 8)
    {
        _mm256_storeu_ps(p_dst,
                         element_op.Apply(_mm256_loadu_ps(p_src),
                                          _mm256_loadu_ps(p_src1),
                                          _mm256_loadu_ps(p_src2)));
        p_dst += 8;
        p_src += 8;
        p_src1 += 8;
        p_src2 += 8;
    }
    if(i_n & 4)
    {
        _mm_storeu_ps(
            p_dst,
            element_op.Apply(_mm_loadu_ps(p_src), _mm_loadu_ps(p_src1), _mm_loadu_ps(p_src2)));
        p_dst += 4;
        p_src += 4;
        p_src1 += 4;
        p_src2 += 4;
    }
    if(i_n & 2)
    {
#if defined(__GNUC__) && !defined(__clang__) && !defined(__llvm__)
        __m128i s  = _mm_loadu_si64(p_src);
        __m128i s1 = _mm_loadu_si64(p_src1);
        __m128i s2 = _mm_loadu_si64(p_src2);

        __m128 v = element_op.Apply(*reinterpret_cast<__m128*>(&s),
                                    *reinterpret_cast<__m128*>(&s1),
                                    *reinterpret_cast<__m128*>(&s2));

        _mm_storeu_si64(p_dst, *reinterpret_cast<__m128i*>(&v));
#else
        _mm_storeu_si64(p_dst,
                        element_op.Apply(
                            _mm_loadu_si64(p_src), _mm_loadu_si64(p_src1), _mm_loadu_si64(p_src2)));
#endif
        p_dst += 2;
        p_src += 2;
        p_src1 += 2;
        p_src2 += 2;
    }
    if(i_n & 1)
    {
        *p_dst = element_op.Apply(*p_src, *p_src1, *p_src2);
    }
}

template <typename ElementwiseOp>
void memcpy32_avx2_with_extra_2src(void* dst,
                                   const void* src,
                                   float v_src1,
                                   const void* src2,
                                   const ck::index_t n,
                                   const ElementwiseOp& element_op)
{
    // 16-8-4-2-1 pattern
    ck::index_t i_n     = n;
    float* p_dst        = reinterpret_cast<float*>(dst);
    const float* p_src  = reinterpret_cast<const float*>(src);
    const float* p_src2 = reinterpret_cast<const float*>(src2);

    __m256 ymm_src1 = _mm256_set1_ps(*reinterpret_cast<const float*>(&v_src1));
    __m128 xmm_src1 = _mm_set1_ps(*reinterpret_cast<const float*>(&v_src1));

    while(i_n >= 16)
    {
        _mm256_storeu_ps(
            p_dst + 0,
            element_op.Apply(_mm256_loadu_ps(p_src + 0), ymm_src1, _mm256_loadu_ps(p_src2 + 0)));
        _mm256_storeu_ps(
            p_dst + 8,
            element_op.Apply(_mm256_loadu_ps(p_src + 8), ymm_src1, _mm256_loadu_ps(p_src2 + 8)));
        p_dst += 16;
        p_src += 16;
        p_src2 += 16;
        i_n -= 16;
    }
    if(i_n & 8)
    {
        _mm256_storeu_ps(
            p_dst, element_op.Apply(_mm256_loadu_ps(p_src), ymm_src1, _mm256_loadu_ps(p_src2)));
        p_dst += 8;
        p_src += 8;
        p_src2 += 8;
    }
    if(i_n & 4)
    {
        _mm_storeu_ps(p_dst, element_op.Apply(_mm_loadu_ps(p_src), xmm_src1, _mm_loadu_ps(p_src2)));
        p_dst += 4;
        p_src += 4;
        p_src2 += 4;
    }
    if(i_n & 2)
    {
#if defined(__GNUC__) && !defined(__clang__) && !defined(__llvm__)
        __m128i s  = _mm_loadu_si64(p_src);
        __m128i s2 = _mm_loadu_si64(p_src2);

        __m128 v = element_op.Apply(
            *reinterpret_cast<__m128*>(&s), xmm_src1, *reinterpret_cast<__m128*>(&s2));

        _mm_storeu_si64(p_dst, *reinterpret_cast<__m128i*>(&v));
#else
        _mm_storeu_si64(p_dst,
                        element_op.Apply(_mm_loadu_si64(p_src), xmm_src1, _mm_loadu_si64(p_src2)));
#endif
        p_dst += 2;
        p_src += 2;
        p_src2 += 2;
    }
    if(i_n & 1)
    {
        *p_dst = element_op.Apply(*p_src, v_src1, *p_src2);
    }
}

template <typename ElementwiseOp>
void memcpy32_avx2_with_extra_1src(void* dst,
                                   const void* src,
                                   const void* src_aux,
                                   const ck::index_t n,
                                   const ElementwiseOp& element_op)
{
    // 16-8-4-2-1 pattern
    ck::index_t i_n        = n;
    float* p_dst           = reinterpret_cast<float*>(dst);
    const float* p_src     = reinterpret_cast<const float*>(src);
    const float* p_src_aux = reinterpret_cast<const float*>(src_aux);
    while(i_n >= 16)
    {
        _mm256_storeu_ps(
            p_dst + 0,
            element_op.Apply(_mm256_loadu_ps(p_src + 0), _mm256_loadu_ps(p_src_aux + 0)));
        _mm256_storeu_ps(
            p_dst + 8,
            element_op.Apply(_mm256_loadu_ps(p_src + 8), _mm256_loadu_ps(p_src_aux + 8)));
        p_dst += 16;
        p_src += 16;
        p_src_aux += 16;
        i_n -= 16;
    }
    if(i_n & 8)
    {
        _mm256_storeu_ps(p_dst,
                         element_op.Apply(_mm256_loadu_ps(p_src), _mm256_loadu_ps(p_src_aux)));
        p_dst += 8;
        p_src += 8;
        p_src_aux += 8;
    }
    if(i_n & 4)
    {
        _mm_storeu_ps(p_dst, element_op.Apply(_mm_loadu_ps(p_src), _mm_loadu_ps(p_src_aux)));
        p_dst += 4;
        p_src += 4;
        p_src_aux += 4;
    }
    if(i_n & 2)
    {
#if defined(__GNUC__) && !defined(__clang__) && !defined(__llvm__)
        __m128i s  = _mm_loadu_si64(p_src);
        __m128i s1 = _mm_loadu_si64(p_src_aux);

        __m128 v =
            element_op.Apply(*reinterpret_cast<__m128*>(&s), *reinterpret_cast<__m128*>(&s1));

        _mm_storeu_si64(p_dst, *reinterpret_cast<__m128i*>(&v));
#else
        _mm_storeu_si64(p_dst, element_op.Apply(_mm_loadu_si64(p_src), _mm_loadu_si64(p_src_aux)));
#endif
        p_dst += 2;
        p_src += 2;
        p_src_aux += 2;
    }
    if(i_n & 1)
    {
        *p_dst = element_op.Apply(*p_src, *p_src_aux);
    }
}

inline void memset32_avx2(void* dst, const int32_t value, const ck::index_t n)
{
    // 16-8-4-2-1 pattern
    ck::index_t i_n = n;
    float* p_dst    = reinterpret_cast<float*>(dst);
    __m256 ymm      = _mm256_set1_ps(*reinterpret_cast<const float*>(&value));
    __m128 xmm      = _mm_set1_ps(*reinterpret_cast<const float*>(&value));
    while(i_n >= 16)
    {
        _mm256_storeu_ps(p_dst + 0, ymm);
        _mm256_storeu_ps(p_dst + 8, ymm);
        p_dst += 16;
        i_n -= 16;
    }
    if(i_n & 8)
    {
        _mm256_storeu_ps(p_dst, ymm);
        p_dst += 8;
    }
    if(i_n & 4)
    {
        _mm_storeu_ps(p_dst, xmm);
        p_dst += 4;
    }
    if(i_n & 2)
    {
#if defined(__GNUC__) && !defined(__clang__) && !defined(__llvm__)
        _mm_storeu_si64(p_dst, *reinterpret_cast<__m128i*>(&xmm));
#else
        _mm_storeu_si64(p_dst, xmm);
#endif
        p_dst += 2;
    }
    if(i_n & 1)
    {
        *p_dst = *reinterpret_cast<const float*>(&value);
    }
}

template <typename ElementwiseOp>
void transpose8x8_avx2(void* dst,
                       ck::index_t stride_dst,
                       const void* src,
                       ck::index_t stride_src,
                       const ElementwiseOp& element_op)
{
    // TODO: use vinsertf128 for better port usage. vpermf128 is slow
    __m256 r0, r1, r2, r3, r4, r5, r6, r7;
    __m256 t0, t1, t2, t3, t4, t5, t6, t7;

    float* p_dst       = reinterpret_cast<float*>(dst);
    const float* p_src = reinterpret_cast<const float*>(src);

    r0 = element_op.Apply(_mm256_loadu_ps(p_src + 0 * stride_src));
    r1 = element_op.Apply(_mm256_loadu_ps(p_src + 1 * stride_src));
    r2 = element_op.Apply(_mm256_loadu_ps(p_src + 2 * stride_src));
    r3 = element_op.Apply(_mm256_loadu_ps(p_src + 3 * stride_src));
    r4 = element_op.Apply(_mm256_loadu_ps(p_src + 4 * stride_src));
    r5 = element_op.Apply(_mm256_loadu_ps(p_src + 5 * stride_src));
    r6 = element_op.Apply(_mm256_loadu_ps(p_src + 6 * stride_src));
    r7 = element_op.Apply(_mm256_loadu_ps(p_src + 7 * stride_src));

    t0 = _mm256_unpacklo_ps(r0, r1);
    t1 = _mm256_unpackhi_ps(r0, r1);
    t2 = _mm256_unpacklo_ps(r2, r3);
    t3 = _mm256_unpackhi_ps(r2, r3);
    t4 = _mm256_unpacklo_ps(r4, r5);
    t5 = _mm256_unpackhi_ps(r4, r5);
    t6 = _mm256_unpacklo_ps(r6, r7);
    t7 = _mm256_unpackhi_ps(r6, r7);

    r0 = _mm256_shuffle_ps(t0, t2, _MM_SHUFFLE(1, 0, 1, 0));
    r1 = _mm256_shuffle_ps(t0, t2, _MM_SHUFFLE(3, 2, 3, 2));
    r2 = _mm256_shuffle_ps(t1, t3, _MM_SHUFFLE(1, 0, 1, 0));
    r3 = _mm256_shuffle_ps(t1, t3, _MM_SHUFFLE(3, 2, 3, 2));
    r4 = _mm256_shuffle_ps(t4, t6, _MM_SHUFFLE(1, 0, 1, 0));
    r5 = _mm256_shuffle_ps(t4, t6, _MM_SHUFFLE(3, 2, 3, 2));
    r6 = _mm256_shuffle_ps(t5, t7, _MM_SHUFFLE(1, 0, 1, 0));
    r7 = _mm256_shuffle_ps(t5, t7, _MM_SHUFFLE(3, 2, 3, 2));

    t0 = _mm256_permute2f128_ps(r0, r4, 0x20);
    t1 = _mm256_permute2f128_ps(r1, r5, 0x20);
    t2 = _mm256_permute2f128_ps(r2, r6, 0x20);
    t3 = _mm256_permute2f128_ps(r3, r7, 0x20);
    t4 = _mm256_permute2f128_ps(r0, r4, 0x31);
    t5 = _mm256_permute2f128_ps(r1, r5, 0x31);
    t6 = _mm256_permute2f128_ps(r2, r6, 0x31);
    t7 = _mm256_permute2f128_ps(r3, r7, 0x31);

    _mm256_storeu_ps(p_dst + 0 * stride_dst, t0);
    _mm256_storeu_ps(p_dst + 1 * stride_dst, t1);
    _mm256_storeu_ps(p_dst + 2 * stride_dst, t2);
    _mm256_storeu_ps(p_dst + 3 * stride_dst, t3);
    _mm256_storeu_ps(p_dst + 4 * stride_dst, t4);
    _mm256_storeu_ps(p_dst + 5 * stride_dst, t5);
    _mm256_storeu_ps(p_dst + 6 * stride_dst, t6);
    _mm256_storeu_ps(p_dst + 7 * stride_dst, t7);
}

} // namespace avx2_util

using ConvolutionForwardSpecialization_t =
    ck::tensor_operation::cpu::device::ConvolutionForwardSpecialization_t;
using ConvolutionForwardGemmKSpecialization_t =
    ck::tensor_operation::cpu::device::ConvolutionForwardGemmKSpecialization_t;

// assume input -> a matrix
// assume input -> MC * KC
template <typename SrcData,
          typename DstData,
          typename SrcDesc,
          typename DstDesc,
          typename ElementwiseOperation,
          bool BypassTransfer,
          ConvolutionForwardSpecialization_t ConvForwardSpecialization>
struct ThreadwiseTensorSliceTransferAvx2Specialization_ConvFwd_In_NHWC
{
    static constexpr ck::index_t nDim = SrcDesc::GetNumOfDimension();
    using Index                       = MultiIndex<nDim>;

    constexpr ThreadwiseTensorSliceTransferAvx2Specialization_ConvFwd_In_NHWC(
        const SrcDesc& src_desc,
        const Index&,
        const DstDesc&,
        const Index&,
        const ElementwiseOperation& element_op,
        const ConvolutionForwardGemmKSpecialization_t& gemm_k_spec)
        : element_op_(element_op), gemm_k_spec_(gemm_k_spec)
    {
        if constexpr(ConvForwardSpecialization ==
                     ConvolutionForwardSpecialization_t::Filter1x1Stride1Pad0)
        {
            N  = 1;
            Hi = 1;
            Wi = src_desc.GetTransforms()[Number<0>{}].GetUpperLengths()[Number<0>{}]; // gemm_m
            C  = src_desc.GetTransforms()[Number<0>{}].GetUpperLengths()[Number<1>{}]; // gemm_k

            Ho = 1;
            Wo = Wi;

            Fy = 1;
            Fx = 1;

            Dy = 1;
            Sy = 1;
            Dx = 1;
            Sx = 1;

            Py = 0;
            Px = 0;
        }
        else if constexpr(ConvForwardSpecialization ==
                          ConvolutionForwardSpecialization_t::Filter1x1Pad0)
        {
            N  = src_desc.GetTransforms()[Number<0>{}].GetUpperLengths()[Number<0>{}];
            Hi = src_desc.GetTransforms()[Number<0>{}].GetUpperLengths()[Number<1>{}];
            Wi = src_desc.GetTransforms()[Number<0>{}].GetUpperLengths()[Number<2>{}];
            C  = src_desc.GetTransforms()[Number<0>{}].GetUpperLengths()[Number<3>{}];

            Ho = src_desc.GetTransforms()[Number<2>{}].GetUpperLengths()[Number<0>{}];
            Wo = src_desc.GetTransforms()[Number<3>{}].GetUpperLengths()[Number<0>{}];

            Fy = 1;
            Fx = 1;

            Dy = 1;
            Sy = src_desc.GetTransforms()[Number<2>{}].coefficients_[Number<0>{}];
            Dx = 1;
            Sx = src_desc.GetTransforms()[Number<3>{}].coefficients_[Number<0>{}];

            Py = 0;
            Px = 0;
        }
        else
        {
            N  = src_desc.GetTransforms()[Number<0>{}].GetUpperLengths()[Number<0>{}];
            Hi = src_desc.GetTransforms()[Number<0>{}].GetUpperLengths()[Number<1>{}];
            Wi = src_desc.GetTransforms()[Number<0>{}].GetUpperLengths()[Number<2>{}];
            C  = src_desc.GetTransforms()[Number<0>{}].GetUpperLengths()[Number<3>{}];

            Ho = src_desc.GetTransforms()[Number<9>{}].low_lengths_[Number<1>{}];
            Wo = src_desc.GetTransforms()[Number<9>{}].low_lengths_[Number<2>{}];

            Fy = src_desc.GetTransforms()[Number<10>{}].low_lengths_[Number<0>{}];
            Fx = src_desc.GetTransforms()[Number<10>{}].low_lengths_[Number<1>{}];

            Dy = src_desc.GetTransforms()[Number<6>{}].coefficients_[Number<0>{}];
            Sy = src_desc.GetTransforms()[Number<6>{}].coefficients_[Number<1>{}];
            Dx = src_desc.GetTransforms()[Number<7>{}].coefficients_[Number<0>{}];
            Sx = src_desc.GetTransforms()[Number<7>{}].coefficients_[Number<1>{}];

            Py = src_desc.GetTransforms()[Number<2>{}].left_pad_length_;
            Px = src_desc.GetTransforms()[Number<3>{}].left_pad_length_;
        }

        // ihi = iho * s_stride_h + iy * s_dilation_h - s_pad_h
        // iwi = iwo * s_stride_w + ix * s_dilation_w - s_pad_w
        input_offset_acc_wi        = Sx * C;
        input_offset_ovf_wi_acc_hi = Sy * Wi * C - Wo * Sx * C;
        input_offset_ovf_hi_acc_n  = Hi * Wi * C - Ho * Sy * Wi * C;

        // input_offset_acc_c       = 1;
        input_offset_ovf_c_acc_x = Dx * C - C;
        input_offset_ovf_x_acc_y = Dy * Wi * C - Fx * Dx * C;

        src_offset = -Py * Wi * C - Px * C;

        i_n  = 0;
        i_c  = 0;
        i_hi = -Py;
        i_wi = -Px;
        i_ho = 0;
        i_wo = 0;
        i_y  = 0;
        i_x  = 0;

        i_gemm_k = 0;

#if 0
        printf("N:%d, Hi:%d, Wi:%d, C:%d, Ho:%d, Wo:%d, Fy:%d, Fx:%d, Dy:%d, Sy:%d, Dx:%d, Sx:%d, "
               "Py:%d, Px:%d\n",
               N,
               Hi,
               Wi,
               C,
               Ho,
               Wo,
               Fy,
               Fx,
               Dy,
               Sy,
               Dx,
               Sx,
               Py,
               Px);
#endif
    }

    void SetSrcSliceOrigin(const SrcDesc&, const Index& src_slice_origin_idx)
    {
        ck::index_t idx_m = src_slice_origin_idx[Number<0>{}];
        ck::index_t idx_k = src_slice_origin_idx[Number<1>{}];

        if constexpr(ConvForwardSpecialization ==
                     ConvolutionForwardSpecialization_t::Filter1x1Stride1Pad0)
        {
            i_wi       = idx_m;
            i_c        = idx_k;
            src_offset = i_wi * C + i_c;

            // printf("src_offset:%d, i_wi:%d, i_c:%d\n", src_offset, i_wi, i_c);
        }
        else if constexpr(ConvForwardSpecialization ==
                          ConvolutionForwardSpecialization_t::Filter1x1Pad0)
        {
            i_wo = idx_m % Wo;
            i_ho = (idx_m / Wo) % Ho;
            i_n  = (idx_m / Wo) / Ho;

            // ihi = iho * s_stride_h + iy * s_dilation_h - s_pad_h
            // iwi = iwo * s_stride_w + ix * s_dilation_w - s_pad_w
            i_c = idx_k;
            i_x = 0;
            i_y = 0;

            i_hi = i_ho * Sy;
            i_wi = i_wo * Sx;

            src_offset = i_n * Hi * Wi * C + i_hi * Wi * C + i_wi * C + i_c;

            i_gemm_k = idx_k;
        }
        else
        {
            i_wo = idx_m % Wo;
            i_ho = (idx_m / Wo) % Ho;
            i_n  = (idx_m / Wo) / Ho;

            // ihi = iho * s_stride_h + iy * s_dilation_h - s_pad_h
            // iwi = iwo * s_stride_w + ix * s_dilation_w - s_pad_w
            if(idx_k == 0)
            {
                i_c = 0;
                i_x = 0;
                i_y = 0;

                i_hi = i_ho * Sy - Py;
                i_wi = i_wo * Sx - Px;
            }
            else
            {
                i_c = idx_k % C;
                i_x = (idx_k / C) % Fx;
                i_y = (idx_k / C) / Fx;

                i_hi = i_ho * Sy + i_y * Dy - Py;
                i_wi = i_wo * Sx + i_x * Dx - Px;
            }

            src_offset = i_n * Hi * Wi * C + i_hi * Wi * C + i_wi * C + i_c;

            i_gemm_k = idx_k;
        }
    }

    void SetDstSliceOrigin(const DstDesc&, const Index&) {}

    template <typename SrcBuffer, typename DstBuffer, typename SliceLengths>
    void RunRead(const SrcDesc& src_desc,
                 SrcBuffer& src_buf,
                 const DstDesc& dst_desc,
                 DstBuffer& dst_buf,
                 const SliceLengths& slice_length)
    {
        if constexpr(BypassTransfer)
        {
            dst_buf.p_data_ = reinterpret_cast<float*>(src_buf.p_data_) + src_offset;
        }
        else
        {
            const ck::index_t m_per_block = slice_length[Number<0>{}];
            const ck::index_t k_per_block = slice_length[Number<1>{}];

            const float* p_src = reinterpret_cast<const float*>(src_buf.p_data_) + src_offset;
            float* p_dst       = reinterpret_cast<float*>(dst_buf.p_data_);

            // printf("src offset:%d, k_per_block:%d, m_per_block:%d\n", src_offset, k_per_block,
            // m_per_block);

            if constexpr(ConvForwardSpecialization ==
                         ConvolutionForwardSpecialization_t::Filter1x1Stride1Pad0)
            {
                ck::index_t i_m_itr = m_per_block;
                // standard 8-4-2-1 pattern
                while(i_m_itr >= 8)
                {
                    avx2_util::memcpy32_avx2(
                        p_dst + 0 * k_per_block, p_src + 0 * C, k_per_block, element_op_);
                    avx2_util::memcpy32_avx2(
                        p_dst + 1 * k_per_block, p_src + 1 * C, k_per_block, element_op_);
                    avx2_util::memcpy32_avx2(
                        p_dst + 2 * k_per_block, p_src + 2 * C, k_per_block, element_op_);
                    avx2_util::memcpy32_avx2(
                        p_dst + 3 * k_per_block, p_src + 3 * C, k_per_block, element_op_);
                    avx2_util::memcpy32_avx2(
                        p_dst + 4 * k_per_block, p_src + 4 * C, k_per_block, element_op_);
                    avx2_util::memcpy32_avx2(
                        p_dst + 5 * k_per_block, p_src + 5 * C, k_per_block, element_op_);
                    avx2_util::memcpy32_avx2(
                        p_dst + 6 * k_per_block, p_src + 6 * C, k_per_block, element_op_);
                    avx2_util::memcpy32_avx2(
                        p_dst + 7 * k_per_block, p_src + 7 * C, k_per_block, element_op_);

                    i_m_itr -= 8;
                    p_dst += 8 * k_per_block;
                    p_src += 8 * C;
                }
                if(i_m_itr & 4)
                {
                    avx2_util::memcpy32_avx2(
                        p_dst + 0 * k_per_block, p_src + 0 * C, k_per_block, element_op_);
                    avx2_util::memcpy32_avx2(
                        p_dst + 1 * k_per_block, p_src + 1 * C, k_per_block, element_op_);
                    avx2_util::memcpy32_avx2(
                        p_dst + 2 * k_per_block, p_src + 2 * C, k_per_block, element_op_);
                    avx2_util::memcpy32_avx2(
                        p_dst + 3 * k_per_block, p_src + 3 * C, k_per_block, element_op_);

                    p_dst += 4 * k_per_block;
                    p_src += 4 * C;
                }

                if(i_m_itr & 2)
                {
                    avx2_util::memcpy32_avx2(
                        p_dst + 0 * k_per_block, p_src + 0 * C, k_per_block, element_op_);
                    avx2_util::memcpy32_avx2(
                        p_dst + 1 * k_per_block, p_src + 1 * C, k_per_block, element_op_);

                    p_dst += 2 * k_per_block;
                    p_src += 2 * C;
                }

                if(i_m_itr & 1)
                {
                    avx2_util::memcpy32_avx2(
                        p_dst + 0 * k_per_block, p_src + 0 * C, k_per_block, element_op_);
                }
            }
            else if constexpr(ConvForwardSpecialization ==
                              ConvolutionForwardSpecialization_t::Filter1x1Pad0)
            {
                ck::index_t i_m_itr  = m_per_block;
                ck::index_t i_wo_itr = i_wo;
                ck::index_t i_ho_itr = i_ho;
                while(i_m_itr > 0)
                {
                    avx2_util::memcpy32_avx2(p_dst, p_src, k_per_block, element_op_);
                    p_dst += k_per_block;
                    i_wo_itr++;
                    p_src += input_offset_acc_wi;
                    if(i_wo_itr >= Wo)
                    {
                        i_wo_itr = 0;
                        i_ho_itr++;
                        p_src += input_offset_ovf_wi_acc_hi;
                    }
                    if(i_ho_itr >= Ho)
                    {
                        i_ho_itr = 0;
                        p_src += input_offset_ovf_hi_acc_n;
                    }

                    i_m_itr--;
                }
            }
            else
            {
                // ihi = iho * s_stride_h + iy * s_dilation_h - s_pad_h
                // iwi = iwo * s_stride_w + ix * s_dilation_w - s_pad_w
                if(gemm_k_spec_ == ConvolutionForwardGemmKSpecialization_t::NHWC_GemmKLoopOverC)
                {
                    // c % k_per_block == 0, so every time k_per_block here is the same
                    ck::index_t i_m_itr  = m_per_block;
                    ck::index_t i_wo_itr = i_wo;
                    ck::index_t i_ho_itr = i_ho;
                    ck::index_t i_wi_itr = i_wi;
                    ck::index_t i_hi_itr = i_hi;

                    while(i_m_itr > 0)
                    {
                        if((*reinterpret_cast<uint32_t*>(&i_hi_itr) <
                            *reinterpret_cast<uint32_t*>(&Hi)) &&
                           (*reinterpret_cast<uint32_t*>(&i_wi_itr) <
                            *reinterpret_cast<uint32_t*>(&Wi)))
                            avx2_util::memcpy32_avx2(p_dst, p_src, k_per_block, element_op_);
                        else
                            avx2_util::memset32_avx2(p_dst, 0, k_per_block);

                        p_dst += k_per_block;

                        i_wo_itr++;
                        i_wi_itr += Sx;
                        p_src += input_offset_acc_wi;
                        if(i_wo_itr >= Wo)
                        {
                            i_wo_itr = 0;
                            i_wi_itr -= Wo * Sx;
                            i_ho_itr++;
                            i_hi_itr += Sy;
                            p_src += input_offset_ovf_wi_acc_hi;
                        }

                        if(i_ho_itr >= Ho)
                        {
                            i_ho_itr = 0;
                            i_hi_itr -= Ho * Sy;
                            p_src += input_offset_ovf_hi_acc_n;
                        }

                        i_m_itr--;
                    }
                }
                else
                {
                    ck::index_t i_m_itr  = m_per_block;
                    ck::index_t i_wo_itr = i_wo;
                    ck::index_t i_ho_itr = i_ho;
                    ck::index_t i_wi_itr = i_wi;
                    ck::index_t i_hi_itr = i_hi;
                    // ihi = iho * s_stride_h + iy * s_dilation_h - s_pad_h
                    // iwi = iwo * s_stride_w + ix * s_dilation_w - s_pad_w
                    while(i_m_itr > 0)
                    {
                        /*** go along Gemm K ***/
                        const float* p_src_k   = p_src;
                        float* p_dst_k         = p_dst;
                        ck::index_t i_wi_itr_k = i_wi_itr;
                        ck::index_t i_hi_itr_k = i_hi_itr;
                        ck::index_t i_c_itr_k  = i_c;
                        // ck::index_t i_y_itr_k  = i_y;
                        ck::index_t i_x_itr_k = i_x;

                        ck::index_t i_k_itr = k_per_block;
                        while(i_k_itr > 0)
                        {
                            ck::index_t current_k_block_along_c =
                                ck::math::min(C - i_c_itr_k, i_k_itr);

                            // printf("current_k_block_along_c:%d, i_c_itr_k:%d, k_per_block:%d\n",
                            // current_k_block_along_c, i_c_itr_k,k_per_block); fflush(stdout);

                            if((*reinterpret_cast<uint32_t*>(&i_hi_itr_k) <
                                *reinterpret_cast<uint32_t*>(&Hi)) &&
                               (*reinterpret_cast<uint32_t*>(&i_wi_itr_k) <
                                *reinterpret_cast<uint32_t*>(&Wi)))
                                avx2_util::memcpy32_avx2(
                                    p_dst_k, p_src_k, current_k_block_along_c, element_op_);
                            else
                                avx2_util::memset32_avx2(p_dst_k, 0, current_k_block_along_c);

                            p_dst_k += current_k_block_along_c;
                            p_src_k += current_k_block_along_c;

                            i_c_itr_k += current_k_block_along_c;
                            if(i_c_itr_k >= C)
                            {
                                i_c_itr_k = 0;
                                i_x_itr_k++;
                                i_wi_itr_k += Dx;
                                p_src_k += input_offset_ovf_c_acc_x;
                            }
                            if(i_x_itr_k >= Fx)
                            {
                                i_x_itr_k = 0;
                                // i_y_itr_k++;
                                i_wi_itr_k -= Dx * Fx;
                                i_hi_itr_k += Dy;
                                p_src_k += input_offset_ovf_x_acc_y;
                            }

                            i_k_itr -= current_k_block_along_c;
                        }
                        /***  go along Gemm K ***/

                        p_dst += k_per_block;

                        i_wo_itr++;
                        i_wi_itr += Sx;
                        p_src += input_offset_acc_wi;
                        if(i_wo_itr >= Wo)
                        {
                            i_wo_itr = 0;
                            i_wi_itr -= Wo * Sx;
                            i_ho_itr++;
                            i_hi_itr += Sy;
                            p_src += input_offset_ovf_wi_acc_hi;
                        }

                        if(i_ho_itr >= Ho)
                        {
                            i_ho_itr = 0;
                            i_hi_itr -= Ho * Sy;
                            p_src += input_offset_ovf_hi_acc_n;
                        }

                        i_m_itr--;
                    }
                }
            }
        }
    }

    void MoveSrcSliceWindow(const SrcDesc& src_desc, const Index& src_slice_origin_step_idx)
    {
        ck::index_t move_k = src_slice_origin_step_idx[Number<1>{}];
        if constexpr(ConvForwardSpecialization ==
                     ConvolutionForwardSpecialization_t::Filter1x1Stride1Pad0)
        {
            // printf("  => move_k:%d, src offset:%d\n", move_k, src_offset);
            i_c += move_k;
            src_offset += move_k;
        }
        else if constexpr(ConvForwardSpecialization ==
                          ConvolutionForwardSpecialization_t::Filter1x1Pad0)
        {
            i_c += move_k;
            src_offset += move_k;
        }
        else
        {
            if(gemm_k_spec_ == ConvolutionForwardGemmKSpecialization_t::NHWC_GemmKLoopOverC)
            {
                // TODO: branch seems weird

                i_c += move_k;
                src_offset += move_k;

                if(i_c >= C)
                {
                    i_c = 0;
                    i_x++;
                    i_wi += Dx;
                    src_offset += Dx * C - C;
                }
                if(i_x >= Fx)
                {
                    i_x = 0;
                    // i_y++;
                    i_wi = i_wi - Fx * Dx;
                    i_hi += Dy;

                    src_offset += Dy * Wi * C - Fx * Dx * C;
                }
            }
            else
            {
                i_gemm_k += move_k;

                i_c = i_gemm_k % C;
                i_x = (i_gemm_k / C) % Fx;
                i_y = (i_gemm_k / C) / Fx;

                i_hi = i_ho * Sy + i_y * Dy - Py;
                i_wi = i_wo * Sx + i_x * Dx - Px;

                src_offset = i_n * Hi * Wi * C + i_hi * Wi * C + i_wi * C + i_c;
            }
        }
    }

    void MoveDstSliceWindow(const DstDesc&, const Index&) {}

    private:
    const ElementwiseOperation element_op_;
    const ConvolutionForwardGemmKSpecialization_t gemm_k_spec_;

    ck::index_t i_n;
    ck::index_t i_c;
    ck::index_t i_hi;
    ck::index_t i_wi;
    ck::index_t i_ho;
    ck::index_t i_wo;
    ck::index_t i_y;
    ck::index_t i_x;
    ck::index_t i_gemm_k;

    ck::index_t N;
    // ck::index_t K;
    ck::index_t C;
    ck::index_t Hi;
    ck::index_t Wi;
    ck::index_t Ho;
    ck::index_t Wo;

    ck::index_t Sy;
    ck::index_t Sx;

    ck::index_t Dy;
    ck::index_t Dx;

    ck::index_t Py;
    ck::index_t Px;

    ck::index_t Fy;
    ck::index_t Fx;

    intptr_t input_offset_acc_wi;
    intptr_t input_offset_ovf_wi_acc_hi;
    intptr_t input_offset_ovf_hi_acc_n;

    // intptr_t input_offset_acc_c;
    intptr_t input_offset_ovf_c_acc_x;
    intptr_t input_offset_ovf_x_acc_y;

    intptr_t src_offset; // keep this as pointer type in case we have negative offset
};

template <typename SrcData,
          typename DstData,
          typename SrcDesc,
          typename DstDesc,
          typename ElementwiseOperation,
          bool BypassTransfer,
          ConvolutionForwardSpecialization_t ConvForwardSpecialization>
struct ThreadwiseTensorSliceTransferAvx2Specialization_ConvFwd_Wei_KYXC
{
    static constexpr ck::index_t nDim = SrcDesc::GetNumOfDimension();
    using Index                       = MultiIndex<nDim>;

    // using SrcCoord = decltype(make_tensor_coordinate(SrcDesc{}, Index{}));
    // using DstCoord = decltype(make_tensor_coordinate(DstDesc{}, Index{}));

    constexpr ThreadwiseTensorSliceTransferAvx2Specialization_ConvFwd_Wei_KYXC(
        const SrcDesc& src_desc,
        const Index& src_slice_origin,
        const DstDesc& dst_desc,
        const Index& dst_slice_origin,
        const ElementwiseOperation& element_op)
        : element_op_(element_op)
    {
        GemmN1 = src_desc.GetTransforms()[Number<3>{}].GetUpperLengths()[Number<1>{}];
        GemmN  = src_desc.GetTransforms()[Number<0>{}].GetUpperLengths()[Number<0>{}];
        GemmK  = src_desc.GetTransforms()[Number<0>{}].GetUpperLengths()[Number<1>{}];
    }

    void SetSrcSliceOrigin(const SrcDesc&, const Index& src_slice_origin_idx)
    {
        ck::index_t idx_n0 = src_slice_origin_idx[Number<0>{}];
        ck::index_t idx_k  = src_slice_origin_idx[Number<1>{}];
        ck::index_t idx_n1 = src_slice_origin_idx[Number<2>{}];

        i_gemm_n = idx_n0 * GemmN1 + idx_n1;
        // i_gemm_k = idx_k;

        src_offset = idx_n0 * GemmK * GemmN1 + idx_k + idx_n1 * GemmN1; // Note we transpose here

        // printf("xxxx  i_gemm_n:%d, i_gemm_k:%d, src_offset:%d\n", i_gemm_n, i_gemm_k,
        // src_offset);
    }

    void SetDstSliceOrigin(const DstDesc&, const Index&) {}

    template <typename SrcBuffer, typename DstBuffer, typename SliceLengths>
    void RunRead(const SrcDesc&,
                 SrcBuffer& src_buf,
                 const DstDesc& dst_desc,
                 DstBuffer& dst_buf,
                 const SliceLengths& slice_length)
    {
        if constexpr(BypassTransfer)
        {
            // KYXC weigh should not support this
            dst_buf.p_data_ = reinterpret_cast<float*>(src_buf.p_data_) + src_offset;
        }
        else
        {
            const ck::index_t n_per_block = slice_length[Number<0>{}] * slice_length[Number<2>{}];
            const ck::index_t k_per_block = slice_length[Number<1>{}];

            // printf(" >>>> %d, %d, %d -> %d(%dx%d), %d\n", GemmN, GemmK, GemmN1, n_per_block,
            //     dst_desc.GetTransforms()[Number<0>{}]
            //         .GetUpperLengths()[Number<0>{}],
            //         dst_desc.GetTransforms()[Number<0>{}]
            //         .GetUpperLengths()[Number<2>{}],
            // k_per_block);

            const float* p_src = reinterpret_cast<const float*>(src_buf.p_data_) + src_offset;
            float* p_dst       = reinterpret_cast<float*>(dst_buf.p_data_);

            // n * k -> n0 * k * n1, n1 = 8, n0 = n/8
            for(index_t i_n_itr = 0; i_n_itr < n_per_block; i_n_itr += 8)
            {
                ck::index_t current_n_8 = ck::math::min(GemmN - (i_n_itr + i_gemm_n), 8);
                ck::index_t i_k_itr     = k_per_block;
                if(current_n_8 == 8)
                {
                    const float* p_src_k = p_src;
                    float* p_dst_k       = p_dst;
                    while(i_k_itr >= 8)
                    {
                        avx2_util::transpose8x8_avx2(p_dst_k, 8, p_src_k, GemmK, element_op_);
                        p_dst_k += 8 * 8;
                        p_src_k += 8;
                        i_k_itr -= 8;
                    }
                    if(i_k_itr & 4)
                    {
                        p_dst_k[0 * 8 + 0] = element_op_.Apply(p_src_k[0 * GemmK + 0]);
                        p_dst_k[0 * 8 + 1] = element_op_.Apply(p_src_k[1 * GemmK + 0]);
                        p_dst_k[0 * 8 + 2] = element_op_.Apply(p_src_k[2 * GemmK + 0]);
                        p_dst_k[0 * 8 + 3] = element_op_.Apply(p_src_k[3 * GemmK + 0]);
                        p_dst_k[0 * 8 + 4] = element_op_.Apply(p_src_k[4 * GemmK + 0]);
                        p_dst_k[0 * 8 + 5] = element_op_.Apply(p_src_k[5 * GemmK + 0]);
                        p_dst_k[0 * 8 + 6] = element_op_.Apply(p_src_k[6 * GemmK + 0]);
                        p_dst_k[0 * 8 + 7] = element_op_.Apply(p_src_k[7 * GemmK + 0]);

                        p_dst_k[1 * 8 + 0] = element_op_.Apply(p_src_k[0 * GemmK + 1]);
                        p_dst_k[1 * 8 + 1] = element_op_.Apply(p_src_k[1 * GemmK + 1]);
                        p_dst_k[1 * 8 + 2] = element_op_.Apply(p_src_k[2 * GemmK + 1]);
                        p_dst_k[1 * 8 + 3] = element_op_.Apply(p_src_k[3 * GemmK + 1]);
                        p_dst_k[1 * 8 + 4] = element_op_.Apply(p_src_k[4 * GemmK + 1]);
                        p_dst_k[1 * 8 + 5] = element_op_.Apply(p_src_k[5 * GemmK + 1]);
                        p_dst_k[1 * 8 + 6] = element_op_.Apply(p_src_k[6 * GemmK + 1]);
                        p_dst_k[1 * 8 + 7] = element_op_.Apply(p_src_k[7 * GemmK + 1]);

                        p_dst_k[2 * 8 + 0] = element_op_.Apply(p_src_k[0 * GemmK + 2]);
                        p_dst_k[2 * 8 + 1] = element_op_.Apply(p_src_k[1 * GemmK + 2]);
                        p_dst_k[2 * 8 + 2] = element_op_.Apply(p_src_k[2 * GemmK + 2]);
                        p_dst_k[2 * 8 + 3] = element_op_.Apply(p_src_k[3 * GemmK + 2]);
                        p_dst_k[2 * 8 + 4] = element_op_.Apply(p_src_k[4 * GemmK + 2]);
                        p_dst_k[2 * 8 + 5] = element_op_.Apply(p_src_k[5 * GemmK + 2]);
                        p_dst_k[2 * 8 + 6] = element_op_.Apply(p_src_k[6 * GemmK + 2]);
                        p_dst_k[2 * 8 + 7] = element_op_.Apply(p_src_k[7 * GemmK + 2]);

                        p_dst_k[3 * 8 + 0] = element_op_.Apply(p_src_k[0 * GemmK + 3]);
                        p_dst_k[3 * 8 + 1] = element_op_.Apply(p_src_k[1 * GemmK + 3]);
                        p_dst_k[3 * 8 + 2] = element_op_.Apply(p_src_k[2 * GemmK + 3]);
                        p_dst_k[3 * 8 + 3] = element_op_.Apply(p_src_k[3 * GemmK + 3]);
                        p_dst_k[3 * 8 + 4] = element_op_.Apply(p_src_k[4 * GemmK + 3]);
                        p_dst_k[3 * 8 + 5] = element_op_.Apply(p_src_k[5 * GemmK + 3]);
                        p_dst_k[3 * 8 + 6] = element_op_.Apply(p_src_k[6 * GemmK + 3]);
                        p_dst_k[3 * 8 + 7] = element_op_.Apply(p_src_k[7 * GemmK + 3]);

                        p_dst_k += 4 * 8;
                        p_src_k += 4;
                    }
                    if(i_k_itr & 2)
                    {
                        p_dst_k[0 * 8 + 0] = element_op_.Apply(p_src_k[0 * GemmK + 0]);
                        p_dst_k[0 * 8 + 1] = element_op_.Apply(p_src_k[1 * GemmK + 0]);
                        p_dst_k[0 * 8 + 2] = element_op_.Apply(p_src_k[2 * GemmK + 0]);
                        p_dst_k[0 * 8 + 3] = element_op_.Apply(p_src_k[3 * GemmK + 0]);
                        p_dst_k[0 * 8 + 4] = element_op_.Apply(p_src_k[4 * GemmK + 0]);
                        p_dst_k[0 * 8 + 5] = element_op_.Apply(p_src_k[5 * GemmK + 0]);
                        p_dst_k[0 * 8 + 6] = element_op_.Apply(p_src_k[6 * GemmK + 0]);
                        p_dst_k[0 * 8 + 7] = element_op_.Apply(p_src_k[7 * GemmK + 0]);

                        p_dst_k[1 * 8 + 0] = element_op_.Apply(p_src_k[0 * GemmK + 1]);
                        p_dst_k[1 * 8 + 1] = element_op_.Apply(p_src_k[1 * GemmK + 1]);
                        p_dst_k[1 * 8 + 2] = element_op_.Apply(p_src_k[2 * GemmK + 1]);
                        p_dst_k[1 * 8 + 3] = element_op_.Apply(p_src_k[3 * GemmK + 1]);
                        p_dst_k[1 * 8 + 4] = element_op_.Apply(p_src_k[4 * GemmK + 1]);
                        p_dst_k[1 * 8 + 5] = element_op_.Apply(p_src_k[5 * GemmK + 1]);
                        p_dst_k[1 * 8 + 6] = element_op_.Apply(p_src_k[6 * GemmK + 1]);
                        p_dst_k[1 * 8 + 7] = element_op_.Apply(p_src_k[7 * GemmK + 1]);

                        p_dst_k += 2 * 8;
                        p_src_k += 2;
                    }
                    if(i_k_itr & 1)
                    {
                        p_dst_k[0 * 8 + 0] = element_op_.Apply(p_src_k[0 * GemmK + 0]);
                        p_dst_k[0 * 8 + 1] = element_op_.Apply(p_src_k[1 * GemmK + 0]);
                        p_dst_k[0 * 8 + 2] = element_op_.Apply(p_src_k[2 * GemmK + 0]);
                        p_dst_k[0 * 8 + 3] = element_op_.Apply(p_src_k[3 * GemmK + 0]);
                        p_dst_k[0 * 8 + 4] = element_op_.Apply(p_src_k[4 * GemmK + 0]);
                        p_dst_k[0 * 8 + 5] = element_op_.Apply(p_src_k[5 * GemmK + 0]);
                        p_dst_k[0 * 8 + 6] = element_op_.Apply(p_src_k[6 * GemmK + 0]);
                        p_dst_k[0 * 8 + 7] = element_op_.Apply(p_src_k[7 * GemmK + 0]);
                    }
                }
                else
                {
                    const float* p_src_k = p_src;
                    float* p_dst_k       = p_dst;

                    for(index_t i_sub_n = 0; i_sub_n < 8; i_sub_n++)
                    {
                        for(index_t i_sub_k = 0; i_sub_k < k_per_block; i_sub_k++)
                        {
                            ck::index_t i_current_n_itr = i_n_itr + i_sub_n + i_gemm_n;

                            float v = i_current_n_itr < GemmN
                                          ? element_op_.Apply(p_src_k[i_sub_n * GemmK + i_sub_k])
                                          : .0f;

                            p_dst_k[i_sub_k * 8 + i_sub_n] = v;
                        }
                    }
                }

                p_dst += 8 * k_per_block;
                p_src += 8 * GemmK;
            }
        }
    }

    // src_slice_origin_step_idx need to be known at compile-time, for performance reason
    void MoveSrcSliceWindow(const SrcDesc& src_desc, const Index& src_slice_origin_step_idx)
    {
        ck::index_t move_k  = src_slice_origin_step_idx[Number<1>{}];
        ck::index_t move_n0 = src_slice_origin_step_idx[Number<0>{}];

        // i_gemm_k += move_k;

        // printf("wei move:%d\n", move_k); fflush(stdout);

        src_offset += move_k + move_n0 * GemmK * GemmN1;
    }

    // dst_slice_origin_step_idx need to be known at compile-time, for performance reason
    void MoveDstSliceWindow(const DstDesc&, const Index&) {}

    private:
    const ElementwiseOperation element_op_;

    ck::index_t i_gemm_n;
    // ck::index_t i_gemm_k;

    // ck::index_t GemmN0;
    ck::index_t GemmN1;
    ck::index_t GemmN;
    ck::index_t GemmK;

    intptr_t src_offset;
};

template <typename SrcData,
          typename DstData,
          typename SrcDesc,
          typename DstDesc,
          typename ElementwiseOperation,
          bool BypassTransfer,
          ConvolutionForwardSpecialization_t ConvForwardSpecialization>
struct ThreadwiseTensorSliceTransferAvx2Specialization_ConvFwd_Wei_KYXCK8
{
    static constexpr ck::index_t nDim = SrcDesc::GetNumOfDimension();
    using Index                       = MultiIndex<nDim>;

    constexpr ThreadwiseTensorSliceTransferAvx2Specialization_ConvFwd_Wei_KYXCK8(
        const SrcDesc& src_desc,
        const Index& src_slice_origin,
        const DstDesc& dst_desc,
        const Index& dst_slice_origin,
        const ElementwiseOperation& element_op)
        : element_op_(element_op)
    {
        GemmN1 =
            src_desc.GetTransforms()[Number<0>{}].GetUpperLengths()[Number<2>{}]; // Need to be 8
        GemmN = src_desc.GetTransforms()[Number<0>{}].GetUpperLengths()[Number<0>{}];
        GemmK = src_desc.GetTransforms()[Number<0>{}].GetUpperLengths()[Number<1>{}];
    }

    void SetSrcSliceOrigin(const SrcDesc&, const Index& src_slice_origin_idx)
    {
        ck::index_t idx_n0 = src_slice_origin_idx[Number<0>{}];
        ck::index_t idx_k  = src_slice_origin_idx[Number<1>{}];
        ck::index_t idx_n1 = src_slice_origin_idx[Number<2>{}];

        src_offset = idx_n0 * GemmK * GemmN1 + idx_k * GemmN1 + idx_n1;

        // printf("xxxx  i_gemm_n:%d, i_gemm_k:%d, src_offset:%d\n", i_gemm_n, i_gemm_k,
        // src_offset);
    }

    void SetDstSliceOrigin(const DstDesc&, const Index&) {}

    template <typename SrcBuffer, typename DstBuffer, typename SliceLengths>
    void RunRead(const SrcDesc&,
                 SrcBuffer& src_buf,
                 const DstDesc& dst_desc,
                 DstBuffer& dst_buf,
                 const SliceLengths& slice_length)
    {
        if constexpr(BypassTransfer)
        {
            dst_buf.p_data_ = reinterpret_cast<float*>(src_buf.p_data_) + src_offset;
        }
        else
        {
            const ck::index_t n0_per_block = slice_length[Number<0>{}];
            const ck::index_t k_n1_per_block =
                slice_length[Number<1>{}] * slice_length[Number<2>{}];
            const ck::index_t SrcStride_K_N1 = GemmK * slice_length[Number<2>{}];

            // printf(" >>>> %d, %d, %d -> %d(%dx%d), %d\n", GemmN, GemmK, GemmN1, n_per_block,
            //     dst_desc.GetTransforms()[Number<0>{}]
            //         .GetUpperLengths()[Number<0>{}],
            //         dst_desc.GetTransforms()[Number<0>{}]
            //         .GetUpperLengths()[Number<2>{}],
            // k_per_block);

            const float* p_src = reinterpret_cast<const float*>(src_buf.p_data_) + src_offset;
            float* p_dst       = reinterpret_cast<float*>(dst_buf.p_data_);

            // n0 * k * n1
            index_t i_n0_itr = n0_per_block;
            while(i_n0_itr >= 8)
            {
                avx2_util::memcpy32_avx2(p_dst + 0 * k_n1_per_block,
                                         p_src + 0 * SrcStride_K_N1,
                                         k_n1_per_block,
                                         element_op_);
                avx2_util::memcpy32_avx2(p_dst + 1 * k_n1_per_block,
                                         p_src + 1 * SrcStride_K_N1,
                                         k_n1_per_block,
                                         element_op_);
                avx2_util::memcpy32_avx2(p_dst + 2 * k_n1_per_block,
                                         p_src + 2 * SrcStride_K_N1,
                                         k_n1_per_block,
                                         element_op_);
                avx2_util::memcpy32_avx2(p_dst + 3 * k_n1_per_block,
                                         p_src + 3 * SrcStride_K_N1,
                                         k_n1_per_block,
                                         element_op_);
                avx2_util::memcpy32_avx2(p_dst + 4 * k_n1_per_block,
                                         p_src + 4 * SrcStride_K_N1,
                                         k_n1_per_block,
                                         element_op_);
                avx2_util::memcpy32_avx2(p_dst + 5 * k_n1_per_block,
                                         p_src + 5 * SrcStride_K_N1,
                                         k_n1_per_block,
                                         element_op_);
                avx2_util::memcpy32_avx2(p_dst + 6 * k_n1_per_block,
                                         p_src + 6 * SrcStride_K_N1,
                                         k_n1_per_block,
                                         element_op_);
                avx2_util::memcpy32_avx2(p_dst + 7 * k_n1_per_block,
                                         p_src + 7 * SrcStride_K_N1,
                                         k_n1_per_block,
                                         element_op_);

                i_n0_itr -= 8;
                p_dst += 8 * k_n1_per_block;
                p_src += 8 * SrcStride_K_N1;
            }
            if(i_n0_itr & 4)
            {
                avx2_util::memcpy32_avx2(p_dst + 0 * k_n1_per_block,
                                         p_src + 0 * SrcStride_K_N1,
                                         k_n1_per_block,
                                         element_op_);
                avx2_util::memcpy32_avx2(p_dst + 1 * k_n1_per_block,
                                         p_src + 1 * SrcStride_K_N1,
                                         k_n1_per_block,
                                         element_op_);
                avx2_util::memcpy32_avx2(p_dst + 2 * k_n1_per_block,
                                         p_src + 2 * SrcStride_K_N1,
                                         k_n1_per_block,
                                         element_op_);
                avx2_util::memcpy32_avx2(p_dst + 3 * k_n1_per_block,
                                         p_src + 3 * SrcStride_K_N1,
                                         k_n1_per_block,
                                         element_op_);

                p_dst += 4 * k_n1_per_block;
                p_src += 4 * SrcStride_K_N1;
            }
            if(i_n0_itr & 2)
            {
                avx2_util::memcpy32_avx2(p_dst + 0 * k_n1_per_block,
                                         p_src + 0 * SrcStride_K_N1,
                                         k_n1_per_block,
                                         element_op_);
                avx2_util::memcpy32_avx2(p_dst + 1 * k_n1_per_block,
                                         p_src + 1 * SrcStride_K_N1,
                                         k_n1_per_block,
                                         element_op_);

                p_dst += 2 * k_n1_per_block;
                p_src += 2 * SrcStride_K_N1;
            }
            if(i_n0_itr & 1)
            {
                avx2_util::memcpy32_avx2(p_dst + 0 * k_n1_per_block,
                                         p_src + 0 * SrcStride_K_N1,
                                         k_n1_per_block,
                                         element_op_);
            }
        }
    }

    // src_slice_origin_step_idx need to be known at compile-time, for performance reason
    void MoveSrcSliceWindow(const SrcDesc& src_desc, const Index& src_slice_origin_step_idx)
    {
        ck::index_t move_n0 = src_slice_origin_step_idx[Number<0>{}];
        ck::index_t move_k  = src_slice_origin_step_idx[Number<1>{}];
        ck::index_t move_n1 = src_slice_origin_step_idx[Number<2>{}];

        // i_gemm_k += move_k;

        // printf("wei move:%d\n", move_k); fflush(stdout);

        src_offset += move_n0 * GemmK * GemmN1 + move_k * GemmN1 + move_n1;
    }

    // dst_slice_origin_step_idx need to be known at compile-time, for performance reason
    void MoveDstSliceWindow(const DstDesc&, const Index&) {}

    private:
    const ElementwiseOperation element_op_;

    ck::index_t i_gemm_n;
    // ck::index_t i_gemm_k;

    // ck::index_t GemmN0;
    ck::index_t GemmN1;
    ck::index_t GemmN;
    ck::index_t GemmK;

    intptr_t src_offset;
};

template <typename SrcData,
          typename DstData,
          typename SrcDesc,
          typename DstDesc,
          typename ElementwiseOperation,
          bool BypassTransfer,
          ConvolutionForwardSpecialization_t ConvForwardSpecialization>
struct ThreadwiseTensorSliceTransferAvx2Specialization_ConvFwd_Wei_YXCK
{
    static constexpr ck::index_t nDim = SrcDesc::GetNumOfDimension();
    using Index                       = MultiIndex<nDim>;

    constexpr ThreadwiseTensorSliceTransferAvx2Specialization_ConvFwd_Wei_YXCK(
        const SrcDesc& src_desc,
        const Index&,
        const DstDesc&,
        const Index&,
        const ElementwiseOperation& element_op)
        : element_op_(element_op)
    {
        GemmK = src_desc.GetTransforms()[Number<0>{}].GetUpperLengths()[Number<0>{}];
        GemmN = src_desc.GetTransforms()[Number<0>{}].GetUpperLengths()[Number<1>{}];
    }

    void SetSrcSliceOrigin(const SrcDesc&, const Index& src_slice_origin_idx)
    {
        ck::index_t idx_k = src_slice_origin_idx[Number<0>{}];
        ck::index_t idx_n = src_slice_origin_idx[Number<1>{}];

        src_offset = idx_k * GemmN + idx_n;
    }

    void SetDstSliceOrigin(const DstDesc&, const Index&) {}

    template <typename SrcBuffer, typename DstBuffer, typename SliceLengths>
    void RunRead(const SrcDesc&,
                 SrcBuffer& src_buf,
                 const DstDesc& dst_desc,
                 DstBuffer& dst_buf,
                 const SliceLengths& slice_length)
    {
        if constexpr(BypassTransfer)
        {
            dst_buf.p_data_ = reinterpret_cast<float*>(src_buf.p_data_) + src_offset;
        }
        else
        {
            const ck::index_t k_per_block = slice_length[Number<0>{}];
            const ck::index_t n_per_block = slice_length[Number<1>{}];

            const float* p_src = reinterpret_cast<const float*>(src_buf.p_data_) + src_offset;
            float* p_dst       = reinterpret_cast<float*>(dst_buf.p_data_);

            // k * n
            index_t i_k_itr = k_per_block;
            while(i_k_itr >= 8)
            {
                avx2_util::memcpy32_avx2(
                    p_dst + 0 * n_per_block, p_src + 0 * GemmN, n_per_block, element_op_);
                avx2_util::memcpy32_avx2(
                    p_dst + 1 * n_per_block, p_src + 1 * GemmN, n_per_block, element_op_);
                avx2_util::memcpy32_avx2(
                    p_dst + 2 * n_per_block, p_src + 2 * GemmN, n_per_block, element_op_);
                avx2_util::memcpy32_avx2(
                    p_dst + 3 * n_per_block, p_src + 3 * GemmN, n_per_block, element_op_);
                avx2_util::memcpy32_avx2(
                    p_dst + 4 * n_per_block, p_src + 4 * GemmN, n_per_block, element_op_);
                avx2_util::memcpy32_avx2(
                    p_dst + 5 * n_per_block, p_src + 5 * GemmN, n_per_block, element_op_);
                avx2_util::memcpy32_avx2(
                    p_dst + 6 * n_per_block, p_src + 6 * GemmN, n_per_block, element_op_);
                avx2_util::memcpy32_avx2(
                    p_dst + 7 * n_per_block, p_src + 7 * GemmN, n_per_block, element_op_);

                i_k_itr -= 8;
                p_dst += 8 * n_per_block;
                p_src += 8 * GemmN;
            }
            if(i_k_itr & 4)
            {
                avx2_util::memcpy32_avx2(
                    p_dst + 0 * n_per_block, p_src + 0 * GemmN, n_per_block, element_op_);
                avx2_util::memcpy32_avx2(
                    p_dst + 1 * n_per_block, p_src + 1 * GemmN, n_per_block, element_op_);
                avx2_util::memcpy32_avx2(
                    p_dst + 2 * n_per_block, p_src + 2 * GemmN, n_per_block, element_op_);
                avx2_util::memcpy32_avx2(
                    p_dst + 3 * n_per_block, p_src + 3 * GemmN, n_per_block, element_op_);

                p_dst += 4 * n_per_block;
                p_src += 4 * GemmN;
            }
            if(i_k_itr & 2)
            {
                avx2_util::memcpy32_avx2(
                    p_dst + 0 * n_per_block, p_src + 0 * GemmN, n_per_block, element_op_);
                avx2_util::memcpy32_avx2(
                    p_dst + 1 * n_per_block, p_src + 1 * GemmN, n_per_block, element_op_);

                p_dst += 2 * n_per_block;
                p_src += 2 * GemmN;
            }
            if(i_k_itr & 1)
            {
                avx2_util::memcpy32_avx2(
                    p_dst + 0 * n_per_block, p_src + 0 * GemmN, n_per_block, element_op_);
            }
        }
    }

    // src_slice_origin_step_idx need to be known at compile-time, for performance reason
    void MoveSrcSliceWindow(const SrcDesc& src_desc, const Index& src_slice_origin_step_idx)
    {
        ck::index_t move_k = src_slice_origin_step_idx[Number<0>{}];
        ck::index_t move_n = src_slice_origin_step_idx[Number<1>{}];

        src_offset += move_k * GemmN + move_n;
    }

    // dst_slice_origin_step_idx need to be known at compile-time, for performance reason
    void MoveDstSliceWindow(const DstDesc&, const Index&) {}

    private:
    const ElementwiseOperation element_op_;

    ck::index_t GemmN;
    ck::index_t GemmK;

    intptr_t src_offset;
};

template <typename SrcData,
          typename DstData,
          typename SrcDesc,
          typename DstDesc,
          typename ElementwiseOperation,
          bool BypassTransfer,
          ConvolutionForwardSpecialization_t ConvForwardSpecialization>
struct ThreadwiseTensorSliceTransferAvx2Specialization_MatC_Store_MxN
{
    static constexpr ck::index_t nDim = SrcDesc::GetNumOfDimension();
    using Index                       = MultiIndex<nDim>;

    constexpr ThreadwiseTensorSliceTransferAvx2Specialization_MatC_Store_MxN(
        const SrcDesc& src_desc,
        const Index&,
        const DstDesc& dst_desc,
        const Index&,
        const ElementwiseOperation& element_op)
        : element_op_(element_op)
    {
        DstGemmM = dst_desc.GetTransforms()[Number<0>{}].GetUpperLengths()[Number<0>{}];
        DstGemmN = dst_desc.GetTransforms()[Number<0>{}].GetUpperLengths()[Number<1>{}];

        src_offset = 0;
        dst_offset = 0;
    }

    void SetSrcSliceOrigin(const SrcDesc&, const Index& src_slice_origin_idx)
    {
        if constexpr(BypassTransfer)
        {
            auto i_src_gemm_m = src_slice_origin_idx[Number<0>{}];
            auto i_src_gemm_n = src_slice_origin_idx[Number<1>{}];

            src_offset = i_src_gemm_m * DstGemmN + i_src_gemm_n;
        }
    }

    void SetDstSliceOrigin(const DstDesc&, const Index& dst_slice_origin_idx)
    {
        i_dst_gemm_m = dst_slice_origin_idx[Number<0>{}];
        i_dst_gemm_n = dst_slice_origin_idx[Number<1>{}];

        dst_offset = i_dst_gemm_m * DstGemmN + i_dst_gemm_n;
    }

    template <typename SrcBuffer, typename DstBuffer, typename SliceLengths>
    void RunRead(const SrcDesc& src_desc,
                 SrcBuffer& src_buf,
                 const DstDesc& dst_desc,
                 DstBuffer& dst_buf,
                 const SliceLengths& slice_length)
    {
        if constexpr(BypassTransfer)
        {
            dst_buf.p_data_ = reinterpret_cast<float*>(src_buf.p_data_) + src_offset;
        }
    }

    template <typename SrcBuffer, typename DstBuffer, typename SliceLengths>
    void RunWrite(const SrcDesc& src_desc,
                  SrcBuffer& src_buf,
                  const DstDesc& dst_desc,
                  DstBuffer& dst_buf,
                  const SliceLengths& slice_length)
    {
        if constexpr(BypassTransfer)
        {
            // src_buf.p_data_ = reinterpret_cast<float*>(dst_buf.p_data_) + src_offset;
            if constexpr(!std::is_same<ElementwiseOperation,
                                       ck::tensor_operation::cpu::element_wise::PassThrough>::value)
            {
                const ck::index_t m_per_block = slice_length[Number<0>{}];
                const ck::index_t n_per_block = slice_length[Number<1>{}];

                const ck::index_t current_n = ck::math::min(DstGemmN - i_dst_gemm_n, n_per_block);

                float* p_dst = reinterpret_cast<float*>(dst_buf.p_data_) + dst_offset;

                ck::index_t i_m_itr = m_per_block;

                // printf("xxxx %d, current_n:%d, DstGemmN:%d, n_per_block:%d,
                // dst_offset:%d\n",__LINE__, current_n,
                //  DstGemmN, n_per_block, dst_offset);fflush(stdout);

                // standard 8-4-2-1 pattern
                while(i_m_itr >= 8)
                {
                    avx2_util::memcpy32_avx2(
                        p_dst + 0 * DstGemmN, p_dst + 0 * DstGemmN, current_n, element_op_);
                    avx2_util::memcpy32_avx2(
                        p_dst + 1 * DstGemmN, p_dst + 1 * DstGemmN, current_n, element_op_);
                    avx2_util::memcpy32_avx2(
                        p_dst + 2 * DstGemmN, p_dst + 2 * DstGemmN, current_n, element_op_);
                    avx2_util::memcpy32_avx2(
                        p_dst + 3 * DstGemmN, p_dst + 3 * DstGemmN, current_n, element_op_);
                    avx2_util::memcpy32_avx2(
                        p_dst + 4 * DstGemmN, p_dst + 4 * DstGemmN, current_n, element_op_);
                    avx2_util::memcpy32_avx2(
                        p_dst + 5 * DstGemmN, p_dst + 5 * DstGemmN, current_n, element_op_);
                    avx2_util::memcpy32_avx2(
                        p_dst + 6 * DstGemmN, p_dst + 6 * DstGemmN, current_n, element_op_);
                    avx2_util::memcpy32_avx2(
                        p_dst + 7 * DstGemmN, p_dst + 7 * DstGemmN, current_n, element_op_);

                    i_m_itr -= 8;
                    p_dst += 8 * DstGemmN;
                }

                if(i_m_itr & 4)
                {
                    avx2_util::memcpy32_avx2(
                        p_dst + 0 * DstGemmN, p_dst + 0 * DstGemmN, current_n, element_op_);
                    avx2_util::memcpy32_avx2(
                        p_dst + 1 * DstGemmN, p_dst + 1 * DstGemmN, current_n, element_op_);
                    avx2_util::memcpy32_avx2(
                        p_dst + 2 * DstGemmN, p_dst + 2 * DstGemmN, current_n, element_op_);
                    avx2_util::memcpy32_avx2(
                        p_dst + 3 * DstGemmN, p_dst + 3 * DstGemmN, current_n, element_op_);

                    p_dst += 4 * DstGemmN;
                }

                if(i_m_itr & 2)
                {
                    avx2_util::memcpy32_avx2(
                        p_dst + 0 * DstGemmN, p_dst + 0 * DstGemmN, current_n, element_op_);
                    avx2_util::memcpy32_avx2(
                        p_dst + 1 * DstGemmN, p_dst + 1 * DstGemmN, current_n, element_op_);

                    p_dst += 2 * DstGemmN;
                }

                if(i_m_itr & 1)
                {
                    avx2_util::memcpy32_avx2(
                        p_dst + 0 * DstGemmN, p_dst + 0 * DstGemmN, current_n, element_op_);
                }
            }
        }
        else
        {
            const ck::index_t m_per_block = slice_length[Number<0>{}];
            const ck::index_t n_per_block = slice_length[Number<1>{}];

            const ck::index_t current_n = ck::math::min(DstGemmN - i_dst_gemm_n, n_per_block);

            const float* p_src = reinterpret_cast<float*>(src_buf.p_data_) + src_offset;
            float* p_dst       = reinterpret_cast<float*>(dst_buf.p_data_) + dst_offset;

            ck::index_t i_m_itr = m_per_block;

            // printf("xxxx %d, current_n:%d, DstGemmN:%d, n_per_block:%d\n",__LINE__, current_n,
            // DstGemmN, n_per_block);fflush(stdout);

            // standard 8-4-2-1 pattern
            while(i_m_itr >= 8)
            {
                avx2_util::memcpy32_avx2(
                    p_dst + 0 * DstGemmN, p_src + 0 * n_per_block, current_n, element_op_);
                avx2_util::memcpy32_avx2(
                    p_dst + 1 * DstGemmN, p_src + 1 * n_per_block, current_n, element_op_);
                avx2_util::memcpy32_avx2(
                    p_dst + 2 * DstGemmN, p_src + 2 * n_per_block, current_n, element_op_);
                avx2_util::memcpy32_avx2(
                    p_dst + 3 * DstGemmN, p_src + 3 * n_per_block, current_n, element_op_);
                avx2_util::memcpy32_avx2(
                    p_dst + 4 * DstGemmN, p_src + 4 * n_per_block, current_n, element_op_);
                avx2_util::memcpy32_avx2(
                    p_dst + 5 * DstGemmN, p_src + 5 * n_per_block, current_n, element_op_);
                avx2_util::memcpy32_avx2(
                    p_dst + 6 * DstGemmN, p_src + 6 * n_per_block, current_n, element_op_);
                avx2_util::memcpy32_avx2(
                    p_dst + 7 * DstGemmN, p_src + 7 * n_per_block, current_n, element_op_);

                i_m_itr -= 8;
                p_dst += 8 * DstGemmN;
                p_src += 8 * n_per_block;
            }

            if(i_m_itr & 4)
            {
                avx2_util::memcpy32_avx2(
                    p_dst + 0 * DstGemmN, p_src + 0 * n_per_block, current_n, element_op_);
                avx2_util::memcpy32_avx2(
                    p_dst + 1 * DstGemmN, p_src + 1 * n_per_block, current_n, element_op_);
                avx2_util::memcpy32_avx2(
                    p_dst + 2 * DstGemmN, p_src + 2 * n_per_block, current_n, element_op_);
                avx2_util::memcpy32_avx2(
                    p_dst + 3 * DstGemmN, p_src + 3 * n_per_block, current_n, element_op_);

                p_dst += 4 * DstGemmN;
                p_src += 4 * n_per_block;
            }

            if(i_m_itr & 2)
            {
                avx2_util::memcpy32_avx2(
                    p_dst + 0 * DstGemmN, p_src + 0 * n_per_block, current_n, element_op_);
                avx2_util::memcpy32_avx2(
                    p_dst + 1 * DstGemmN, p_src + 1 * n_per_block, current_n, element_op_);

                p_dst += 2 * DstGemmN;
                p_src += 2 * n_per_block;
            }

            if(i_m_itr & 1)
            {
                avx2_util::memcpy32_avx2(
                    p_dst + 0 * DstGemmN, p_src + 0 * n_per_block, current_n, element_op_);
            }

            // printf("xxxx %d\n",__LINE__);fflush(stdout);
        }
    }

    // src_slice_origin_step_idx need to be known at compile-time, for performance reason
    void MoveSrcSliceWindow(const SrcDesc&, const Index&) {}

    // dst_slice_origin_step_idx need to be known at compile-time, for performance reason
    void MoveDstSliceWindow(const DstDesc&, const Index&) {}

    private:
    const ElementwiseOperation element_op_;

    ck::index_t i_dst_gemm_m;
    ck::index_t i_dst_gemm_n;

    ck::index_t DstGemmM;
    ck::index_t DstGemmN;

    intptr_t src_offset;
    intptr_t dst_offset;
};

template <typename SrcData,
          typename Src1Data, // for Bias, per dimension
          typename Src2Data, // for Residual, per pixel
          typename DstData,
          typename SrcDesc,
          typename Src1Desc,
          typename Src2Desc,
          typename DstDesc,
          typename ElementwiseOperation,
          bool BypassTransfer,
          bool Src1AlongDim0> // if true, src1 has dim along M, false, src1 has dim along N
struct ThreadwiseTensorSliceTransferAvx2Specialization_MatC_Store_Bias_Residual_MxN
{
    static constexpr ck::index_t nDim = SrcDesc::GetNumOfDimension();
    using Index                       = MultiIndex<nDim>;

    constexpr ThreadwiseTensorSliceTransferAvx2Specialization_MatC_Store_Bias_Residual_MxN(
        const SrcDesc& src_desc,
        const Index&,
        const DstDesc& dst_desc,
        const Index&,
        const ElementwiseOperation& element_op)
        : element_op_(element_op)
    {
        DstGemmM = dst_desc.GetTransforms()[Number<0>{}].GetUpperLengths()[Number<0>{}];
        DstGemmN = dst_desc.GetTransforms()[Number<0>{}].GetUpperLengths()[Number<1>{}];

        src_offset  = 0;
        src1_offset = 1;
        src2_offset = 2;
        dst_offset  = 0;
    }

    void SetSrcSliceOrigin(const SrcDesc&, const Index& src_slice_origin_idx)
    {
        if constexpr(BypassTransfer)
        {
            auto i_src_gemm_m = src_slice_origin_idx[Number<0>{}];
            auto i_src_gemm_n = src_slice_origin_idx[Number<1>{}];

            src_offset = i_src_gemm_m * DstGemmN + i_src_gemm_n;
        }
    }

    void SetSrc1SliceOrigin(const SrcDesc&, const Index& src_slice_origin_idx)
    {
        if constexpr(Src1AlongDim0)
        {
            auto i_src_gemm_m = src_slice_origin_idx[Number<0>{}];
            // auto i_src_gemm_n = src_slice_origin_idx[Number<1>{}];

            src1_offset = i_src_gemm_m;
        }
        else
        {
            auto i_src_gemm_n = src_slice_origin_idx[Number<1>{}];
            src1_offset       = i_src_gemm_n;
        }
    }

    void SetSrc2SliceOrigin(const SrcDesc&, const Index& src_slice_origin_idx)
    {
        auto i_src_gemm_m = src_slice_origin_idx[Number<0>{}];
        auto i_src_gemm_n = src_slice_origin_idx[Number<1>{}];

        src2_offset = i_src_gemm_m * DstGemmN + i_src_gemm_n;
    }

    void SetDstSliceOrigin(const DstDesc&, const Index& dst_slice_origin_idx)
    {
        i_dst_gemm_m = dst_slice_origin_idx[Number<0>{}];
        i_dst_gemm_n = dst_slice_origin_idx[Number<1>{}];

        dst_offset = i_dst_gemm_m * DstGemmN + i_dst_gemm_n;
    }

    template <typename SrcBuffer,
              typename Src1Buffer,
              typename Src2Buffer,
              typename DstBuffer,
              typename SliceLengths>
    void RunRead(const SrcDesc&,
                 SrcBuffer& src_buf,
                 const Src1Desc&,
                 Src1Buffer&,
                 const Src2Desc&,
                 Src2Buffer&,
                 const DstDesc&,
                 DstBuffer& dst_buf,
                 const SliceLengths&)
    {
        if constexpr(BypassTransfer)
        {
            dst_buf.p_data_ = reinterpret_cast<float*>(src_buf.p_data_) + src_offset;
        }
    }

    template <typename SrcBuffer,
              typename Src1Buffer,
              typename Src2Buffer,
              typename DstBuffer,
              typename SliceLengths>
    void RunWrite(const SrcDesc& src_desc,
                  SrcBuffer& src_buf,
                  const Src1Desc& src1_desc,
                  Src1Buffer& src1_buf,
                  const Src2Desc& src2_desc,
                  Src2Buffer& src2_buf,
                  const DstDesc& dst_desc,
                  DstBuffer& dst_buf,
                  const SliceLengths& slice_length)
    {
        if constexpr(BypassTransfer)
        {
            // src_buf.p_data_ = reinterpret_cast<float*>(dst_buf.p_data_) + src_offset;
            if constexpr(!std::is_same<ElementwiseOperation,
                                       ck::tensor_operation::cpu::element_wise::PassThrough>::value)
            {
                const ck::index_t m_per_block = slice_length[Number<0>{}];
                const ck::index_t n_per_block = slice_length[Number<1>{}];

                const ck::index_t current_n = ck::math::min(DstGemmN - i_dst_gemm_n, n_per_block);

                float* p_dst = reinterpret_cast<float*>(dst_buf.p_data_) + dst_offset;
                const float* p_src1 =
                    reinterpret_cast<const float*>(src1_buf.p_data_) + src1_offset;
                const float* p_src2 =
                    reinterpret_cast<const float*>(src2_buf.p_data_) + src2_offset;

                ck::index_t i_m_itr = m_per_block;

                // printf("xxxx %d, current_n:%d, DstGemmN:%d, n_per_block:%d,
                // dst_offset:%d\n",__LINE__, current_n,
                //  DstGemmN, n_per_block, dst_offset);fflush(stdout);

                // standard 8-4-2-1 pattern
                if constexpr(Src1AlongDim0)
                {
                    while(i_m_itr >= 8)
                    {
                        avx2_util::memcpy32_avx2_with_extra_2src(p_dst + 0 * DstGemmN,
                                                                 p_dst + 0 * DstGemmN,
                                                                 *(p_src1 + 0),
                                                                 p_src2 + 0 * DstGemmN,
                                                                 current_n,
                                                                 element_op_);
                        avx2_util::memcpy32_avx2_with_extra_2src(p_dst + 1 * DstGemmN,
                                                                 p_dst + 1 * DstGemmN,
                                                                 *(p_src1 + 1),
                                                                 p_src2 + 1 * DstGemmN,
                                                                 current_n,
                                                                 element_op_);
                        avx2_util::memcpy32_avx2_with_extra_2src(p_dst + 2 * DstGemmN,
                                                                 p_dst + 2 * DstGemmN,
                                                                 *(p_src1 + 2),
                                                                 p_src2 + 2 * DstGemmN,
                                                                 current_n,
                                                                 element_op_);
                        avx2_util::memcpy32_avx2_with_extra_2src(p_dst + 3 * DstGemmN,
                                                                 p_dst + 3 * DstGemmN,
                                                                 *(p_src1 + 3),
                                                                 p_src2 + 3 * DstGemmN,
                                                                 current_n,
                                                                 element_op_);
                        avx2_util::memcpy32_avx2_with_extra_2src(p_dst + 4 * DstGemmN,
                                                                 p_dst + 4 * DstGemmN,
                                                                 *(p_src1 + 4),
                                                                 p_src2 + 4 * DstGemmN,
                                                                 current_n,
                                                                 element_op_);
                        avx2_util::memcpy32_avx2_with_extra_2src(p_dst + 5 * DstGemmN,
                                                                 p_dst + 5 * DstGemmN,
                                                                 *(p_src1 + 5),
                                                                 p_src2 + 5 * DstGemmN,
                                                                 current_n,
                                                                 element_op_);
                        avx2_util::memcpy32_avx2_with_extra_2src(p_dst + 6 * DstGemmN,
                                                                 p_dst + 6 * DstGemmN,
                                                                 *(p_src1 + 6),
                                                                 p_src2 + 6 * DstGemmN,
                                                                 current_n,
                                                                 element_op_);
                        avx2_util::memcpy32_avx2_with_extra_2src(p_dst + 7 * DstGemmN,
                                                                 p_dst + 7 * DstGemmN,
                                                                 *(p_src1 + 7),
                                                                 p_src2 + 7 * DstGemmN,
                                                                 current_n,
                                                                 element_op_);

                        i_m_itr -= 8;
                        p_dst += 8 * DstGemmN;
                        p_src1 += 8;
                        p_src2 += 8 * DstGemmN;
                    }

                    if(i_m_itr & 4)
                    {
                        avx2_util::memcpy32_avx2_with_extra_2src(p_dst + 0 * DstGemmN,
                                                                 p_dst + 0 * DstGemmN,
                                                                 *(p_src1 + 0),
                                                                 p_src2 + 0 * DstGemmN,
                                                                 current_n,
                                                                 element_op_);
                        avx2_util::memcpy32_avx2_with_extra_2src(p_dst + 1 * DstGemmN,
                                                                 p_dst + 1 * DstGemmN,
                                                                 *(p_src1 + 1),
                                                                 p_src2 + 1 * DstGemmN,
                                                                 current_n,
                                                                 element_op_);
                        avx2_util::memcpy32_avx2_with_extra_2src(p_dst + 2 * DstGemmN,
                                                                 p_dst + 2 * DstGemmN,
                                                                 *(p_src1 + 2),
                                                                 p_src2 + 2 * DstGemmN,
                                                                 current_n,
                                                                 element_op_);
                        avx2_util::memcpy32_avx2_with_extra_2src(p_dst + 3 * DstGemmN,
                                                                 p_dst + 3 * DstGemmN,
                                                                 *(p_src1 + 3),
                                                                 p_src2 + 3 * DstGemmN,
                                                                 current_n,
                                                                 element_op_);

                        p_dst += 4 * DstGemmN;
                        p_src1 += 4;
                        p_src2 += 4 * DstGemmN;
                    }

                    if(i_m_itr & 2)
                    {
                        avx2_util::memcpy32_avx2_with_extra_2src(p_dst + 0 * DstGemmN,
                                                                 p_dst + 0 * DstGemmN,
                                                                 *(p_src1 + 0),
                                                                 p_src2 + 0 * DstGemmN,
                                                                 current_n,
                                                                 element_op_);
                        avx2_util::memcpy32_avx2_with_extra_2src(p_dst + 1 * DstGemmN,
                                                                 p_dst + 1 * DstGemmN,
                                                                 *(p_src1 + 1),
                                                                 p_src2 + 1 * DstGemmN,
                                                                 current_n,
                                                                 element_op_);

                        p_dst += 2 * DstGemmN;
                        p_src1 += 2;
                        p_src2 += 2 * DstGemmN;
                    }

                    if(i_m_itr & 1)
                    {
                        avx2_util::memcpy32_avx2_with_extra_2src(p_dst + 0 * DstGemmN,
                                                                 p_dst + 0 * DstGemmN,
                                                                 *(p_src1 + 0),
                                                                 p_src2 + 0 * DstGemmN,
                                                                 current_n,
                                                                 element_op_);
                    }
                }
                else
                {
                    while(i_m_itr >= 8)
                    {
                        avx2_util::memcpy32_avx2_with_extra_2src(p_dst + 0 * DstGemmN,
                                                                 p_dst + 0 * DstGemmN,
                                                                 p_src1,
                                                                 p_src2 + 0 * DstGemmN,
                                                                 current_n,
                                                                 element_op_);
                        avx2_util::memcpy32_avx2_with_extra_2src(p_dst + 1 * DstGemmN,
                                                                 p_dst + 1 * DstGemmN,
                                                                 p_src1,
                                                                 p_src2 + 1 * DstGemmN,
                                                                 current_n,
                                                                 element_op_);
                        avx2_util::memcpy32_avx2_with_extra_2src(p_dst + 2 * DstGemmN,
                                                                 p_dst + 2 * DstGemmN,
                                                                 p_src1,
                                                                 p_src2 + 2 * DstGemmN,
                                                                 current_n,
                                                                 element_op_);
                        avx2_util::memcpy32_avx2_with_extra_2src(p_dst + 3 * DstGemmN,
                                                                 p_dst + 3 * DstGemmN,
                                                                 p_src1,
                                                                 p_src2 + 3 * DstGemmN,
                                                                 current_n,
                                                                 element_op_);
                        avx2_util::memcpy32_avx2_with_extra_2src(p_dst + 4 * DstGemmN,
                                                                 p_dst + 4 * DstGemmN,
                                                                 p_src1,
                                                                 p_src2 + 4 * DstGemmN,
                                                                 current_n,
                                                                 element_op_);
                        avx2_util::memcpy32_avx2_with_extra_2src(p_dst + 5 * DstGemmN,
                                                                 p_dst + 5 * DstGemmN,
                                                                 p_src1,
                                                                 p_src2 + 5 * DstGemmN,
                                                                 current_n,
                                                                 element_op_);
                        avx2_util::memcpy32_avx2_with_extra_2src(p_dst + 6 * DstGemmN,
                                                                 p_dst + 6 * DstGemmN,
                                                                 p_src1,
                                                                 p_src2 + 6 * DstGemmN,
                                                                 current_n,
                                                                 element_op_);
                        avx2_util::memcpy32_avx2_with_extra_2src(p_dst + 7 * DstGemmN,
                                                                 p_dst + 7 * DstGemmN,
                                                                 p_src1,
                                                                 p_src2 + 7 * DstGemmN,
                                                                 current_n,
                                                                 element_op_);

                        i_m_itr -= 8;
                        p_dst += 8 * DstGemmN;
                        p_src2 += 8 * DstGemmN;
                    }

                    if(i_m_itr & 4)
                    {
                        avx2_util::memcpy32_avx2_with_extra_2src(p_dst + 0 * DstGemmN,
                                                                 p_dst + 0 * DstGemmN,
                                                                 p_src1,
                                                                 p_src2 + 0 * DstGemmN,
                                                                 current_n,
                                                                 element_op_);
                        avx2_util::memcpy32_avx2_with_extra_2src(p_dst + 1 * DstGemmN,
                                                                 p_dst + 1 * DstGemmN,
                                                                 p_src1,
                                                                 p_src2 + 1 * DstGemmN,
                                                                 current_n,
                                                                 element_op_);
                        avx2_util::memcpy32_avx2_with_extra_2src(p_dst + 2 * DstGemmN,
                                                                 p_dst + 2 * DstGemmN,
                                                                 p_src1,
                                                                 p_src2 + 2 * DstGemmN,
                                                                 current_n,
                                                                 element_op_);
                        avx2_util::memcpy32_avx2_with_extra_2src(p_dst + 3 * DstGemmN,
                                                                 p_dst + 3 * DstGemmN,
                                                                 p_src1,
                                                                 p_src2 + 3 * DstGemmN,
                                                                 current_n,
                                                                 element_op_);

                        p_dst += 4 * DstGemmN;
                        p_src2 += 4 * DstGemmN;
                    }

                    if(i_m_itr & 2)
                    {
                        avx2_util::memcpy32_avx2_with_extra_2src(p_dst + 0 * DstGemmN,
                                                                 p_dst + 0 * DstGemmN,
                                                                 p_src1,
                                                                 p_src2 + 0 * DstGemmN,
                                                                 current_n,
                                                                 element_op_);
                        avx2_util::memcpy32_avx2_with_extra_2src(p_dst + 1 * DstGemmN,
                                                                 p_dst + 1 * DstGemmN,
                                                                 p_src1,
                                                                 p_src2 + 1 * DstGemmN,
                                                                 current_n,
                                                                 element_op_);

                        p_dst += 2 * DstGemmN;
                        p_src2 += 2 * DstGemmN;
                    }

                    if(i_m_itr & 1)
                    {
                        avx2_util::memcpy32_avx2_with_extra_2src(p_dst + 0 * DstGemmN,
                                                                 p_dst + 0 * DstGemmN,
                                                                 p_src1,
                                                                 p_src2 + 0 * DstGemmN,
                                                                 current_n,
                                                                 element_op_);
                    }
                }
            }
        }
        else
        {
            const ck::index_t m_per_block = slice_length[Number<0>{}];
            const ck::index_t n_per_block = slice_length[Number<1>{}];

            const ck::index_t current_n = ck::math::min(DstGemmN - i_dst_gemm_n, n_per_block);

            const float* p_src  = reinterpret_cast<const float*>(src_buf.p_data_) + src_offset;
            float* p_dst        = reinterpret_cast<float*>(dst_buf.p_data_) + dst_offset;
            const float* p_src1 = reinterpret_cast<const float*>(src1_buf.p_data_) + src1_offset;
            const float* p_src2 = reinterpret_cast<const float*>(src2_buf.p_data_) + src2_offset;

            ck::index_t i_m_itr = m_per_block;

            // printf("xxxx %d, current_n:%d, DstGemmN:%d, n_per_block:%d\n",__LINE__, current_n,
            // DstGemmN, n_per_block);fflush(stdout);

            // standard 8-4-2-1 pattern
            if constexpr(Src1AlongDim0)
            {
                while(i_m_itr >= 8)
                {
                    avx2_util::memcpy32_avx2_with_extra_2src(p_dst + 0 * DstGemmN,
                                                             p_src + 0 * n_per_block,
                                                             *(p_src1 + 0),
                                                             p_src2 + 0 * DstGemmN,
                                                             current_n,
                                                             element_op_);
                    avx2_util::memcpy32_avx2_with_extra_2src(p_dst + 1 * DstGemmN,
                                                             p_src + 1 * n_per_block,
                                                             *(p_src1 + 1),
                                                             p_src2 + 1 * DstGemmN,
                                                             current_n,
                                                             element_op_);
                    avx2_util::memcpy32_avx2_with_extra_2src(p_dst + 2 * DstGemmN,
                                                             p_src + 2 * n_per_block,
                                                             *(p_src1 + 2),
                                                             p_src2 + 2 * DstGemmN,
                                                             current_n,
                                                             element_op_);
                    avx2_util::memcpy32_avx2_with_extra_2src(p_dst + 3 * DstGemmN,
                                                             p_src + 3 * n_per_block,
                                                             *(p_src1 + 3),
                                                             p_src2 + 3 * DstGemmN,
                                                             current_n,
                                                             element_op_);
                    avx2_util::memcpy32_avx2_with_extra_2src(p_dst + 4 * DstGemmN,
                                                             p_src + 4 * n_per_block,
                                                             *(p_src1 + 4),
                                                             p_src2 + 4 * DstGemmN,
                                                             current_n,
                                                             element_op_);
                    avx2_util::memcpy32_avx2_with_extra_2src(p_dst + 5 * DstGemmN,
                                                             p_src + 5 * n_per_block,
                                                             *(p_src1 + 5),
                                                             p_src2 + 5 * DstGemmN,
                                                             current_n,
                                                             element_op_);
                    avx2_util::memcpy32_avx2_with_extra_2src(p_dst + 6 * DstGemmN,
                                                             p_src + 6 * n_per_block,
                                                             *(p_src1 + 6),
                                                             p_src2 + 6 * DstGemmN,
                                                             current_n,
                                                             element_op_);
                    avx2_util::memcpy32_avx2_with_extra_2src(p_dst + 7 * DstGemmN,
                                                             p_src + 7 * n_per_block,
                                                             *(p_src1 + 7),
                                                             p_src2 + 7 * DstGemmN,
                                                             current_n,
                                                             element_op_);

                    i_m_itr -= 8;
                    p_dst += 8 * DstGemmN;
                    p_src += 8 * n_per_block;
                    p_src1 += 8;
                    p_src2 += 8 * DstGemmN;
                }

                if(i_m_itr & 4)
                {
                    avx2_util::memcpy32_avx2_with_extra_2src(p_dst + 0 * DstGemmN,
                                                             p_src + 0 * n_per_block,
                                                             *(p_src1 + 0),
                                                             p_src2 + 0 * DstGemmN,
                                                             current_n,
                                                             element_op_);
                    avx2_util::memcpy32_avx2_with_extra_2src(p_dst + 1 * DstGemmN,
                                                             p_src + 1 * n_per_block,
                                                             *(p_src1 + 1),
                                                             p_src2 + 1 * DstGemmN,
                                                             current_n,
                                                             element_op_);
                    avx2_util::memcpy32_avx2_with_extra_2src(p_dst + 2 * DstGemmN,
                                                             p_src + 2 * n_per_block,
                                                             *(p_src1 + 2),
                                                             p_src2 + 2 * DstGemmN,
                                                             current_n,
                                                             element_op_);
                    avx2_util::memcpy32_avx2_with_extra_2src(p_dst + 3 * DstGemmN,
                                                             p_src + 3 * n_per_block,
                                                             *(p_src1 + 3),
                                                             p_src2 + 3 * DstGemmN,
                                                             current_n,
                                                             element_op_);

                    p_dst += 4 * DstGemmN;
                    p_src += 4 * n_per_block;
                    p_src1 += 4;
                    p_src2 += 4 * DstGemmN;
                }

                if(i_m_itr & 2)
                {
                    avx2_util::memcpy32_avx2_with_extra_2src(p_dst + 0 * DstGemmN,
                                                             p_src + 0 * n_per_block,
                                                             *(p_src1 + 0),
                                                             p_src2 + 0 * DstGemmN,
                                                             current_n,
                                                             element_op_);
                    avx2_util::memcpy32_avx2_with_extra_2src(p_dst + 1 * DstGemmN,
                                                             p_src + 1 * n_per_block,
                                                             *(p_src1 + 1),
                                                             p_src2 + 1 * DstGemmN,
                                                             current_n,
                                                             element_op_);

                    p_dst += 2 * DstGemmN;
                    p_src += 2 * n_per_block;
                    p_src1 += 2;
                    p_src2 += 2 * DstGemmN;
                }

                if(i_m_itr & 1)
                {
                    avx2_util::memcpy32_avx2_with_extra_2src(p_dst + 0 * DstGemmN,
                                                             p_src + 0 * n_per_block,
                                                             *(p_src1 + 0),
                                                             p_src2 + 0 * DstGemmN,
                                                             current_n,
                                                             element_op_);
                }
            }
            else
            {
                while(i_m_itr >= 8)
                {
                    avx2_util::memcpy32_avx2_with_extra_2src(p_dst + 0 * DstGemmN,
                                                             p_src + 0 * n_per_block,
                                                             p_src1,
                                                             p_src2 + 0 * DstGemmN,
                                                             current_n,
                                                             element_op_);
                    avx2_util::memcpy32_avx2_with_extra_2src(p_dst + 1 * DstGemmN,
                                                             p_src + 1 * n_per_block,
                                                             p_src1,
                                                             p_src2 + 1 * DstGemmN,
                                                             current_n,
                                                             element_op_);
                    avx2_util::memcpy32_avx2_with_extra_2src(p_dst + 2 * DstGemmN,
                                                             p_src + 2 * n_per_block,
                                                             p_src1,
                                                             p_src2 + 2 * DstGemmN,
                                                             current_n,
                                                             element_op_);
                    avx2_util::memcpy32_avx2_with_extra_2src(p_dst + 3 * DstGemmN,
                                                             p_src + 3 * n_per_block,
                                                             p_src1,
                                                             p_src2 + 3 * DstGemmN,
                                                             current_n,
                                                             element_op_);
                    avx2_util::memcpy32_avx2_with_extra_2src(p_dst + 4 * DstGemmN,
                                                             p_src + 4 * n_per_block,
                                                             p_src1,
                                                             p_src2 + 4 * DstGemmN,
                                                             current_n,
                                                             element_op_);
                    avx2_util::memcpy32_avx2_with_extra_2src(p_dst + 5 * DstGemmN,
                                                             p_src + 5 * n_per_block,
                                                             p_src1,
                                                             p_src2 + 5 * DstGemmN,
                                                             current_n,
                                                             element_op_);
                    avx2_util::memcpy32_avx2_with_extra_2src(p_dst + 6 * DstGemmN,
                                                             p_src + 6 * n_per_block,
                                                             p_src1,
                                                             p_src2 + 6 * DstGemmN,
                                                             current_n,
                                                             element_op_);
                    avx2_util::memcpy32_avx2_with_extra_2src(p_dst + 7 * DstGemmN,
                                                             p_src + 7 * n_per_block,
                                                             p_src1,
                                                             p_src2 + 7 * DstGemmN,
                                                             current_n,
                                                             element_op_);

                    i_m_itr -= 8;
                    p_dst += 8 * DstGemmN;
                    p_src += 8 * n_per_block;
                    p_src2 += 8 * DstGemmN;
                }

                if(i_m_itr & 4)
                {
                    avx2_util::memcpy32_avx2_with_extra_2src(p_dst + 0 * DstGemmN,
                                                             p_src + 0 * n_per_block,
                                                             p_src1,
                                                             p_src2 + 0 * DstGemmN,
                                                             current_n,
                                                             element_op_);
                    avx2_util::memcpy32_avx2_with_extra_2src(p_dst + 1 * DstGemmN,
                                                             p_src + 1 * n_per_block,
                                                             p_src1,
                                                             p_src2 + 1 * DstGemmN,
                                                             current_n,
                                                             element_op_);
                    avx2_util::memcpy32_avx2_with_extra_2src(p_dst + 2 * DstGemmN,
                                                             p_src + 2 * n_per_block,
                                                             p_src1,
                                                             p_src2 + 2 * DstGemmN,
                                                             current_n,
                                                             element_op_);
                    avx2_util::memcpy32_avx2_with_extra_2src(p_dst + 3 * DstGemmN,
                                                             p_src + 3 * n_per_block,
                                                             p_src1,
                                                             p_src2 + 3 * DstGemmN,
                                                             current_n,
                                                             element_op_);

                    p_dst += 4 * DstGemmN;
                    p_src += 4 * n_per_block;
                    p_src2 += 4 * DstGemmN;
                }

                if(i_m_itr & 2)
                {
                    avx2_util::memcpy32_avx2_with_extra_2src(p_dst + 0 * DstGemmN,
                                                             p_src + 0 * n_per_block,
                                                             p_src1,
                                                             p_src2 + 0 * DstGemmN,
                                                             current_n,
                                                             element_op_);
                    avx2_util::memcpy32_avx2_with_extra_2src(p_dst + 1 * DstGemmN,
                                                             p_src + 1 * n_per_block,
                                                             p_src1,
                                                             p_src2 + 1 * DstGemmN,
                                                             current_n,
                                                             element_op_);

                    p_dst += 2 * DstGemmN;
                    p_src += 2 * n_per_block;
                    p_src2 += 2 * DstGemmN;
                }

                if(i_m_itr & 1)
                {
                    avx2_util::memcpy32_avx2_with_extra_2src(p_dst + 0 * DstGemmN,
                                                             p_src + 0 * n_per_block,
                                                             p_src1,
                                                             p_src2 + 0 * DstGemmN,
                                                             current_n,
                                                             element_op_);
                }
            }

            // printf("xxxx %d\n",__LINE__);fflush(stdout);
        }
    }

    // src_slice_origin_step_idx need to be known at compile-time, for performance reason
    void MoveSrcSliceWindow(const SrcDesc&, const Index&) {}

    // dst_slice_origin_step_idx need to be known at compile-time, for performance reason
    void MoveDstSliceWindow(const DstDesc&, const Index&) {}

    private:
    const ElementwiseOperation element_op_;

    ck::index_t i_dst_gemm_m;
    ck::index_t i_dst_gemm_n;

    ck::index_t DstGemmM;
    ck::index_t DstGemmN;

    intptr_t src_offset;
    intptr_t src1_offset;
    intptr_t src2_offset;
    intptr_t dst_offset;
};

} // namespace cpu
} // namespace ck

#endif
