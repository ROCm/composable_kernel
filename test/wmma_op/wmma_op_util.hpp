// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck/ck.hpp"
#include "ck/tensor_operation/gpu/device/tensor_layout.hpp"
#include "ck/library/utility/check_err.hpp"
#include "ck/library/utility/device_memory.hpp"
#include "ck/library/utility/host_tensor.hpp"
#include "ck/library/utility/host_tensor_generator.hpp"
#include "ck/library/reference_tensor_operation/cpu/reference_gemm.hpp"
#include "ck/utility/amd_wmma.hpp"
#include "ck/host_utility/device_prop.hpp"

namespace ck {
namespace wmma_op_util {

#if defined(__gfx125__)

#define CK_WMMA_CALL_INTRIN_1(dst_fmt, src0_fmt, size)              \
    intrin_wmma_##dst_fmt##_16x16x##size##_##src0_fmt<16, 16>::Run( \
        reg_a, reg_b, reg_d.GetVectorTypeReference(Number<0>{}))

#define CK_WMMA_CALL_INTRIN_2(dst_fmt, src0_fmt, src1_fmt, size)              \
    intrin_wmma_##dst_fmt##_16x16x##size##_##src0_fmt##src1_fmt<16, 16>::Run( \
        reg_a, reg_b, reg_d.GetVectorTypeReference(Number<0>{}))

#define CK_WMMA_CALL_INTRIN_3(dst_fmt, acc_fmt, src0_fmt, size)              \
    intrin_wmma_##dst_fmt##acc_fmt##_16x16x##size##_##src0_fmt<16, 16>::Run( \
        reg_a,                                                               \
        reg_b,                                                               \
        reg_c.GetVectorTypeReference(Number<0>{}),                           \
        reg_d.GetVectorTypeReference(Number<0>{}))

#define CK_WMMA_CALL_INTRIN_4(dst_fmt, src0_fmt, neg_a, neg_b, size)              \
    intrin_wmma_##dst_fmt##_16x16x##size##_##src0_fmt<16, 16, neg_a, neg_b>::Run( \
        reg_a, reg_b, reg_d.GetVectorTypeReference(Number<0>{}))

template <typename T, index_t kValue, typename = void>
struct WMMAVecType
{
    static_assert(sizeof(T) == 0, "VecType is not specialized for this type");
};

// fp64 specialization
template <typename T, index_t kValue>
struct WMMAVecType<T, kValue, ck::enable_if_t<ck::is_same_v<T, double>>>
{
    static constexpr bool layoutTransform = false;
    static constexpr int ToIntDim         = 1;

    template <typename D>
    constexpr static bool is_compatible()
    {
        return ck::is_same_v<T, D>;
    }

    using VecT                = vector_type<T, kValue / 2>;
    using ViewT               = vector_type<T, 1>;
    static constexpr int size = kValue / 2;
};

// fp32 specialization
template <typename T, index_t kValue>
struct WMMAVecType<T, kValue, ck::enable_if_t<ck::is_same_v<T, float>>>
{
    static constexpr bool layoutTransform = false;
    static constexpr int ToIntDim         = 1;

    template <typename D>
    constexpr static bool is_compatible()
    {
        return ck::is_same_v<T, D>;
    }

    using VecT                = vector_type<T, kValue / 2>;
    using ViewT               = vector_type<T, 1>;
    static constexpr int size = kValue / 2;
};

// fp16 specialization
template <typename T, index_t kValue>
struct WMMAVecType<T,
                   kValue,
                   ck::enable_if_t<ck::is_same_v<T, ck::half_t> || ck::is_same_v<T, ck::bhalf_t>>>
{
    static constexpr bool layoutTransform = false;
    static constexpr int ToIntDim         = 2;

    template <typename D>
    constexpr static bool is_compatible()
    {
        return ck::is_same_v<T, D>;
    }

    using VecT                = vector_type<T, kValue / 2>;
    using ViewT               = vector_type<T, 2>;
    static constexpr int size = kValue / 2;
};

// f8 ocp or bf8 ocp specialization
template <typename T, index_t kValue>
struct WMMAVecType<
    T,
    kValue,
    ck::enable_if_t<ck::is_same_v<T, ck::f8_ocp_t> || ck::is_same_v<T, ck::bf8_ocp_t>>>
{
    static constexpr bool layoutTransform = true;
    static constexpr int ToIntDim         = 4;

    template <typename D>
    constexpr static bool is_compatible()
    {
        return ck::is_same_v<D, ck::f8_t> || ck::is_same_v<D, ck::bf8_t>;
    }

    using VecT                = vector_type<typename T::data_type, kValue / 2>;
    using ViewT               = vector_type<typename T::data_type, 4>;
    static constexpr int size = kValue / 2;
};

// f8 fnuz or bf8 fnuz specialization
template <typename T, index_t kValue>
struct WMMAVecType<
    T,
    kValue,
    ck::enable_if_t<ck::is_same_v<T, ck::f8_fnuz_t> || ck::is_same_v<T, ck::bf8_fnuz_t>>>
{
    static constexpr bool layoutTransform = true;
    static constexpr int ToIntDim         = 4;

    template <typename D>
    constexpr static bool is_compatible()
    {
        return ck::is_same_v<D, ck::f8_t> || ck::is_same_v<D, ck::bf8_t>;
    }

    using VecT                = vector_type<T, kValue / 2>;
    using ViewT               = vector_type<T, 4>;
    static constexpr int size = kValue / 2;
};

// int8 specialization
template <typename T, index_t kValue>
struct WMMAVecType<T, kValue, ck::enable_if_t<ck::is_same_v<T, int8_t>>>
{
    static constexpr bool layoutTransform = true;
    static constexpr int ToIntDim         = 4;
    template <typename D>
    constexpr static bool is_compatible()
    {
        return ck::is_same_v<T, D>;
    }

    using VecT                = vector_type<T, kValue / 2>;
    using ViewT               = vector_type<T, 4>;
    static constexpr int size = kValue / 2;
};

// gfx1250 builtin_wmma_naive_selector
template <typename srcAType, typename srcBType, typename dstType, typename accType, index_t kValue>
__device__ void builtin_wmma_naive_selector(
    const typename WMMAVecType<srcAType, kValue>::VecT::type& reg_a,
    const typename WMMAVecType<srcBType, kValue>::VecT::type& reg_b,
    StaticBufferTupleOfVector<AddressSpaceEnum::Vgpr, accType, 1, 8, true>& reg_c,
    StaticBufferTupleOfVector<AddressSpaceEnum::Vgpr, dstType, 1, 8, true>& reg_d)
{
    constexpr int size = kValue;

    // if accType and dstType the same
    if constexpr(std::is_same_v<accType, dstType>)
    {
        if constexpr((ck::is_same_v<srcAType, ck::bf8_t> || ck::is_same_v<srcAType, ck::f8_t>) &&
                     (ck::is_same_v<srcBType, ck::bf8_t> || ck::is_same_v<srcBType, ck::f8_t>))
        {
            if constexpr(ck::is_same_v<dstType, ck::half_t>)
            {
                if constexpr(ck::is_same_v<srcAType, ck::bf8_t> &&
                             ck::is_same_v<srcBType, ck::bf8_t>)
                {
                    if constexpr(size == 64)
                        CK_WMMA_CALL_INTRIN_2(f16, bf8, bf8, 64);
                    else if constexpr(size == 128)
                        CK_WMMA_CALL_INTRIN_2(f16, bf8, bf8, 128);
                    else
                        static_assert(false);
                }
                else if constexpr(ck::is_same_v<srcAType, ck::bf8_t> &&
                                  ck::is_same_v<srcBType, ck::f8_t>)
                {
                    if constexpr(size == 64)
                        CK_WMMA_CALL_INTRIN_2(f16, bf8, f8, 64);
                    else if constexpr(size == 128)
                        CK_WMMA_CALL_INTRIN_2(f16, bf8, f8, 128);
                    else
                        static_assert(false);
                }
                else if constexpr(ck::is_same_v<srcAType, ck::f8_t> &&
                                  ck::is_same_v<srcBType, ck::bf8_t>)
                {
                    if constexpr(size == 64)
                        CK_WMMA_CALL_INTRIN_2(f16, f8, bf8, 64);
                    else if constexpr(size == 128)
                        CK_WMMA_CALL_INTRIN_2(f16, f8, bf8, 128);
                    else
                        static_assert(false);
                }
                else if constexpr(ck::is_same_v<srcAType, ck::f8_t> &&
                                  ck::is_same_v<srcBType, ck::f8_t>)
                {
                    if constexpr(size == 64)
                        CK_WMMA_CALL_INTRIN_2(f16, f8, f8, 64);
                    else if constexpr(size == 128)
                        CK_WMMA_CALL_INTRIN_2(f16, f8, f8, 128);
                    else
                        static_assert(false);
                }
                else
                {
                    static_assert(false);
                }
            }
            else if constexpr(ck::is_same_v<dstType, float>)
            {
                if constexpr(ck::is_same_v<srcAType, ck::bf8_t> &&
                             ck::is_same_v<srcBType, ck::bf8_t>)
                {
                    if constexpr(size == 64)
                        CK_WMMA_CALL_INTRIN_2(f32, bf8, bf8, 64);
                    else if constexpr(size == 128)
                        CK_WMMA_CALL_INTRIN_2(f32, bf8, bf8, 128);
                    else
                        static_assert(false);
                }
                else if constexpr(ck::is_same_v<srcAType, ck::bf8_t> &&
                                  ck::is_same_v<srcBType, ck::f8_t>)
                {
                    if constexpr(size == 64)
                        CK_WMMA_CALL_INTRIN_2(f32, bf8, f8, 64);
                    else if constexpr(size == 128)
                        CK_WMMA_CALL_INTRIN_2(f32, bf8, f8, 128);
                    else
                        static_assert(false);
                }
                else if constexpr(ck::is_same_v<srcAType, ck::f8_t> &&
                                  ck::is_same_v<srcBType, ck::bf8_t>)
                {
                    if constexpr(size == 64)
                        CK_WMMA_CALL_INTRIN_2(f32, f8, bf8, 64);
                    else if constexpr(size == 128)
                        CK_WMMA_CALL_INTRIN_2(f32, f8, bf8, 128);
                    else
                        static_assert(false);
                }
                else if constexpr(ck::is_same_v<srcAType, ck::f8_t> &&
                                  ck::is_same_v<srcBType, ck::f8_t>)
                {
                    if constexpr(size == 64)
                        CK_WMMA_CALL_INTRIN_2(f32, f8, f8, 64);
                    else if constexpr(size == 128)
                        CK_WMMA_CALL_INTRIN_2(f32, f8, f8, 128);
                    else
                        static_assert(false);
                }
                else
                {
                    static_assert(false);
                }
            }
        }
        else
        { // not fp8 or bf8
            if constexpr(std::is_same_v<srcAType, ck::half_t> &&
                         std::is_same_v<srcBType, ck::half_t> &&
                         std::is_same_v<dstType, ck::half_t>)
            {
                CK_WMMA_CALL_INTRIN_1(f16, f16, 32);
            }
            else if constexpr(std::is_same_v<srcAType, ck::half_t> &&
                              std::is_same_v<srcBType, ck::half_t> &&
                              std::is_same_v<dstType, float>)
            {
                CK_WMMA_CALL_INTRIN_1(f32, f16, 32);
            }
            else if constexpr(std::is_same_v<srcAType, ck::bhalf_t> &&
                              std::is_same_v<srcBType, ck::bhalf_t> &&
                              std::is_same_v<dstType, float>)
            {
                CK_WMMA_CALL_INTRIN_1(f32, bf16, 32);
            }
            else if constexpr(std::is_same_v<srcAType, ck::bhalf_t> &&
                              std::is_same_v<srcBType, ck::bhalf_t> &&
                              std::is_same_v<dstType, ck::bhalf_t>)
            {
                CK_WMMA_CALL_INTRIN_1(bf16, bf16, 32);
            }
            else if constexpr(std::is_same_v<srcAType, int8_t> &&
                              std::is_same_v<srcBType, int8_t> && std::is_same_v<dstType, int32_t>)
            {
                CK_WMMA_CALL_INTRIN_4(i32, iu8, true, true, 64);
            }
            else if constexpr(std::is_same_v<srcAType, float> && std::is_same_v<srcBType, float> &&
                              std::is_same_v<dstType, float>)
            {
                CK_WMMA_CALL_INTRIN_1(f32, f32, 4);
            }
            else
            {
                static_assert(false);
            }
        }
    }
    else if constexpr(!std::is_same_v<accType, dstType>)
    {
        if constexpr(std::is_same_v<accType, float> && std::is_same_v<dstType, ck::bhalf_t>)
        {
            CK_WMMA_CALL_INTRIN_3(bf16, f32, bf16, 32);
        }
        else
        {
            static_assert(false);
        }
    }
    else
    {
        static_assert(false);
    }
}

template <typename srcA_t, typename srcB_t, typename dst_t, typename acc_t, ck::index_t kValue>
__global__ void matmul(const srcA_t* a, const srcB_t* b, dst_t* c)
{
    static_assert(WMMAVecType<srcA_t, kValue>::template is_compatible<srcB_t>(),
                  "the data format for srcA and srcB is unsupported in gfx1250");
    using srcA_cast_T    = WMMAVecType<srcA_t, kValue>::ViewT;
    using srcB_cast_T    = WMMAVecType<srcB_t, kValue>::ViewT;
    using srcA_cast_type = typename srcA_cast_T::type;
    using srcB_cast_type = typename srcB_cast_T::type;

    using srcA_vec      = typename WMMAVecType<srcA_t, kValue>::VecT;
    using srcB_vec      = typename WMMAVecType<srcB_t, kValue>::VecT;
    using srcA_vec_type = srcA_vec::type;
    using srcB_vec_type = srcB_vec::type;

    const srcA_cast_type* a_ptr = reinterpret_cast<const srcA_cast_type*>(a);
    const srcB_cast_type* b_ptr = reinterpret_cast<const srcB_cast_type*>(b);

    srcA_vec a_frag = {};
    srcB_vec b_frag = {};

    srcA_vec a_temp = {};
    srcB_vec b_temp = {};

    using acc_vec = StaticBufferTupleOfVector<AddressSpaceEnum::Vgpr, acc_t, 1, 8, true>;
    acc_vec acc_thread_buf_;

    using dst_vec = StaticBufferTupleOfVector<AddressSpaceEnum::Vgpr, dst_t, 1, 8, true>;
    dst_vec dst_thread_buf_;

    // Num elements per 32B packed chunk
    constexpr int ToIntDim = WMMAVecType<srcA_t, kValue>::ToIntDim;

    // to int dim is 1 for float, 2 for half; base dim assumption is 16
    constexpr int SRC_DIM = WMMAVecType<srcA_t, kValue>::size / ToIntDim;

    // 2 threads per a row
    constexpr int ROW_SIZE = 2 * SRC_DIM;

    // 16 is base dim assumption, 2 is for a input and b input both
    constexpr int LDS_DIM = 2 * 16 * ROW_SIZE;

    constexpr int LDS_B_START = LDS_DIM / 2;

    constexpr int QUADRANT_SIZE = ROW_SIZE / 4;

    constexpr int BLOCK_SIZE = 4 * QUADRANT_SIZE;
    __shared__ srcA_cast_type p_shared[LDS_DIM];

    // strongly-type compile time index value of 0 for template containers
    static constexpr auto I0 = Number<0>{};

    // use a directly, as lds_shared allocated with A; A better be bigger than B if mixed
    const srcA_cast_type* local_a_ptr = p_shared;

    // get pointer as B type given LDS allocated with A
    const srcB_cast_type* local_b_ptr = reinterpret_cast<const srcB_cast_type*>(p_shared);

    const int lIdx    = threadIdx.x;
    const int lane    = lIdx % 32; // wave size
    const int lowHigh = lane / 16;

    bool use_QUADS = true;

    if(use_QUADS == true)
    {

        // load A to registers using QUADRANTS
        static_for<0, QUADRANT_SIZE, 1>{}([&](auto ele) {
            int i      = ele;
            int j      = ele + QUADRANT_SIZE * 2;
            int rowIdx = lane % 16;

            int offset1 = (rowIdx * ROW_SIZE) + (i + (lowHigh * QUADRANT_SIZE));
            int offset2 = (rowIdx * ROW_SIZE) + (j + (lowHigh * QUADRANT_SIZE));

            a_temp.template AsType<srcA_cast_type>()(ele) = a_ptr[offset1];
            a_temp.template AsType<srcA_cast_type>()(Number<ele + QUADRANT_SIZE>{}) =
                a_ptr[offset2];
        });

        // load B to registers using QUADRANTS
        static_for<0, QUADRANT_SIZE, 1>{}([&](auto ele) {
            int i      = ele;
            int j      = ele + QUADRANT_SIZE * 2;
            int rowIdx = lane % 16;

            int offset1 = (rowIdx * ROW_SIZE) + (i + (lowHigh * QUADRANT_SIZE));
            int offset2 = (rowIdx * ROW_SIZE) + (j + (lowHigh * QUADRANT_SIZE));

            b_temp.template AsType<srcB_cast_type>()(ele) = b_ptr[offset1];
            b_temp.template AsType<srcB_cast_type>()(Number<ele + QUADRANT_SIZE>{}) =
                b_ptr[offset2];
        });

        // Load A into LDS with quadrants
        static_for<0, QUADRANT_SIZE, 1>{}([&](auto ele) {
            int rowIdx = lIdx % 16;
            int hi     = lIdx / 16;

            // Each thread gets a block based on rowIdx and hi
            int base = rowIdx * BLOCK_SIZE + hi * QUADRANT_SIZE;

            // Write first quadrant
            p_shared[base + ele] = a_temp.template AsType<srcA_cast_type>()(ele);

            // Write second quadrant (offset by 2 quad sizes)
            p_shared[base + ele + QUADRANT_SIZE * 2] =
                a_temp.template AsType<srcA_cast_type>()(Number<ele + QUADRANT_SIZE>{});
        });

        // Load B into LDS with quadrants -- OK
        static_for<0, QUADRANT_SIZE, 1>{}([&](auto ele) {
            int rowIdx = lIdx % 16;
            int hi     = lIdx / 16;

            // Each thread gets a block based on rowIdx and hi
            int base = rowIdx * BLOCK_SIZE + hi * QUADRANT_SIZE + LDS_B_START;

            // Write first quadrant
            int idx1       = base + ele;
            p_shared[idx1] = b_temp.template AsType<srcB_cast_type>()(ele);

            // Write second quadrant (offset by 2 quad sizes)
            int idx2 = base + ele + QUADRANT_SIZE * 2;
            p_shared[idx2] =
                b_temp.template AsType<srcB_cast_type>()(Number<ele + QUADRANT_SIZE>{});
        });

        __syncthreads();

        // Construct a_frag and b_frag for WMMA call -- OK
        static_for<0, QUADRANT_SIZE, 1>{}([&](auto ele) {
            int rowIdx = lIdx % 16;
            int hi     = lIdx / 16;

            // base for a
            int base_a = rowIdx * BLOCK_SIZE + hi * QUADRANT_SIZE;
            int base_b = rowIdx * BLOCK_SIZE + hi * QUADRANT_SIZE + LDS_B_START;

            int idx1_a = base_a + ele;
            int idx2_a = base_a + ele + QUADRANT_SIZE * 2;
            int idx1_b = base_b + ele;
            int idx2_b = base_b + ele + QUADRANT_SIZE * 2;

            // index for first quadrant access
            a_frag.template AsType<srcA_cast_type>()(ele) = local_a_ptr[idx1_a];
            b_frag.template AsType<srcB_cast_type>()(ele) = local_b_ptr[idx1_b];

            // index for second quadrant access
            a_frag.template AsType<srcA_cast_type>()(Number<ele + QUADRANT_SIZE>{}) =
                local_a_ptr[idx2_a];
            b_frag.template AsType<srcB_cast_type>()(Number<ele + QUADRANT_SIZE>{}) =
                local_b_ptr[idx2_b];
        });
    }
    else // Don't use quads
    {
        // Load A to registers without quadrants
        static_for<0, SRC_DIM, 1>{}([&](auto ele) {
            int i      = ele;
            int rowIdx = lane % 16;

            int offset1 = (rowIdx * ROW_SIZE) + (i + (lowHigh * SRC_DIM));

            a_temp.template AsType<srcA_cast_type>()(ele) = a_ptr[offset1];
        });

        // Load B to registers without quadrants
        static_for<0, SRC_DIM, 1>{}([&](auto ele) {
            int i      = ele;
            int rowIdx = lane % 16;

            int offset1 = (rowIdx * ROW_SIZE) + (i + (lowHigh * SRC_DIM));

            b_temp.template AsType<srcB_cast_type>()(ele) = b_ptr[offset1];
        });

        // Load A into LDS without quadrants
        static_for<0, SRC_DIM, 1>{}([&](auto ele) {
            int rowIdx = lIdx % 16;
            int hi     = lIdx / 16;

            // Each thread gets a block based on rowIdx and hi
            int base = rowIdx * ROW_SIZE + hi * SRC_DIM;

            // Write first quadrant
            p_shared[base + ele] = a_temp.template AsType<srcA_cast_type>()(ele);
        });

        // Load B into LDS without quadrants
        static_for<0, SRC_DIM, 1>{}([&](auto ele) {
            int rowIdx = lIdx % 16;
            int hi     = lIdx / 16;

            // Each thread gets a block based on rowIdx and hi
            int base = rowIdx * ROW_SIZE + hi * SRC_DIM + LDS_B_START;

            // Write first quadrant
            int idx1       = base + ele;
            p_shared[idx1] = b_temp.template AsType<srcB_cast_type>()(ele);
        });

        static_for<0, SRC_DIM, 1>{}([&](auto ele) {
            int rowIdx = lIdx % 16;
            int hi     = lIdx / 16;

            // base for a
            int base_a = rowIdx * ROW_SIZE + hi * SRC_DIM;
            int base_b = rowIdx * ROW_SIZE + hi * SRC_DIM + LDS_B_START;

            int idx1_a = base_a + ele;
            int idx1_b = base_b + ele;

            // index for first quadrant access
            a_frag.template AsType<srcA_cast_type>()(ele) = local_a_ptr[idx1_a];
            b_frag.template AsType<srcB_cast_type>()(ele) = local_b_ptr[idx1_b];
        });
    }

    __syncthreads();

    // Call the WMMA intrinsic selector
    builtin_wmma_naive_selector<srcA_t, srcB_t, dst_t, acc_t, kValue>(
        a_frag.template AsType<srcA_vec_type>()(I0),
        b_frag.template AsType<srcB_vec_type>()(I0),
        acc_thread_buf_,
        dst_thread_buf_);

    static_for<0, 8, 1>{}([&](auto ele) {
        int lowHi         = lIdx / 16;
        int col           = lIdx % 16;
        int row           = (lowHi) * 8 + static_cast<int>(ele);
        c[col + 16 * row] = dst_thread_buf_[Number<ele>{}];
    });
}

template <typename srcA_t, typename srcB_t, typename dst_t, typename acc_t, ck::index_t kValue>
__global__ void matmul_swizzle_a(const srcA_t* a, const srcB_t* b, dst_t* c)
{
    static_assert(WMMAVecType<srcA_t, kValue>::template is_compatible<srcB_t>(),
                  "the data format for srcA and srcB is unsupported in gfx1250");
    using srcA_cast_T    = WMMAVecType<srcA_t, kValue>::ViewT;
    using srcB_cast_T    = WMMAVecType<srcB_t, kValue>::ViewT;
    using srcA_cast_type = typename srcA_cast_T::type;
    using srcB_cast_type = typename srcB_cast_T::type;

    using srcA_vec      = typename WMMAVecType<srcA_t, kValue>::VecT;
    using srcB_vec      = typename WMMAVecType<srcB_t, kValue>::VecT;
    using srcA_vec_type = srcA_vec::type;
    using srcB_vec_type = srcB_vec::type;

    const srcA_cast_type* a_ptr = reinterpret_cast<const srcA_cast_type*>(a);
    const srcB_cast_type* b_ptr = reinterpret_cast<const srcB_cast_type*>(b);

    srcA_vec a_frag = {};
    srcB_vec b_frag = {};

    using acc_vec = StaticBufferTupleOfVector<AddressSpaceEnum::Vgpr, acc_t, 1, 8, true>;
    acc_vec acc_thread_buf_;

    using dst_vec = StaticBufferTupleOfVector<AddressSpaceEnum::Vgpr, dst_t, 1, 8, true>;
    dst_vec dst_thread_buf_;

    // Num elements per 32B packed chunk
    constexpr int ToIntDim = WMMAVecType<srcA_t, kValue>::ToIntDim;

    constexpr int SRC_DIM = WMMAVecType<srcA_t, kValue>::size / ToIntDim;

    // 2 threads per a row
    constexpr int ROW_SIZE = 2 * SRC_DIM;

    constexpr int QUADRANT_SIZE = ROW_SIZE / 4;

    // strongly-type compile time index value of 0 for template containers
    static constexpr auto I0 = Number<0>{};

    const int lIdx = threadIdx.x;

    bool use_QUADS = true;

    if(use_QUADS == true)
    {
        constexpr int BLOCK_SIZE = 4 * QUADRANT_SIZE;
        static_for<0, QUADRANT_SIZE, 1>{}([&](auto ele) {
            int rowIdx = lIdx % 16;
            int hi     = lIdx / 16;

            // base for a
            int base_a = rowIdx * BLOCK_SIZE + hi * QUADRANT_SIZE;
            int base_b = rowIdx * BLOCK_SIZE + hi * QUADRANT_SIZE;

            int idx1_a = base_a + ele;
            int idx2_a = base_a + ele + QUADRANT_SIZE * 2;
            int idx1_b = base_b + ele;
            int idx2_b = base_b + ele + QUADRANT_SIZE * 2;

            // index for first quadrant access
            a_frag.template AsType<srcA_cast_type>()(ele) = a_ptr[idx1_a];
            b_frag.template AsType<srcB_cast_type>()(ele) = b_ptr[idx1_b];

            // index for second quadrant access
            a_frag.template AsType<srcA_cast_type>()(Number<ele + QUADRANT_SIZE>{}) = a_ptr[idx2_a];
            b_frag.template AsType<srcB_cast_type>()(Number<ele + QUADRANT_SIZE>{}) = b_ptr[idx2_b];
        });
    }
    else
    {
        static_for<0, SRC_DIM, 1>{}([&](auto ele) {
            int rowIdx = lIdx % 16;
            int hi     = lIdx / 16;

            // base for a
            int base = rowIdx * ROW_SIZE + hi * SRC_DIM;
            int idx1 = base + ele;

            // index for first quadrant access
            a_frag.template AsType<srcA_cast_type>()(ele) = a_ptr[idx1];
            b_frag.template AsType<srcB_cast_type>()(ele) = b_ptr[idx1];
        });
    }

    __syncthreads();

    builtin_wmma_naive_selector<srcA_t, srcB_t, dst_t, acc_t, kValue>(
        a_frag.template AsType<srcA_vec_type>()(I0),
        b_frag.template AsType<srcB_vec_type>()(I0),
        acc_thread_buf_,
        dst_thread_buf_);

    static_for<0, 8, 1>{}([&](auto ele) {
        int lowHi = lIdx / 16;
        int col   = lIdx % 16;
        int row   = (lowHi) * 8 + static_cast<int>(ele);
        c[col + 16 * row] =
            dst_thread_buf_[Number<ele>{}]; // idea each thread contiguous along column
    });
}

// template <typename src_t, typename dst_t, typename acc_t, index_t acc_num>
template <typename src_t, typename dst_t, typename acc_t>
__global__ void matmul(const src_t* a, const src_t* b, dst_t* c)
{
    ignore = a;
    ignore = b;
    ignore = c;
    printf("---------- Running gfx1250 matmul - DISABLED original matmul ----------\n");
}

template <typename src_t, typename dst_t, typename acc_t>
__global__ void matmul_swizzle_a(const src_t* a, const src_t* b, dst_t* c)
{
    ignore = a;
    ignore = b;
    ignore = c;
    printf("---------- Running gfx1250 matmul - DISABLED original matmul_swizzle_a ----------\n");
}

// #endif
#else

template <typename AccType>
struct WMMA_ACCNumber_traits
{
    static constexpr index_t ACC_NUMBER = 8;
};

template <>
struct WMMA_ACCNumber_traits<ck::half_t>
{
    static constexpr index_t ACC_NUMBER = 16;
};

template <>
struct WMMA_ACCNumber_traits<ck::bhalf_t>
{
    static constexpr index_t ACC_NUMBER = 16;
};

template <typename src_vec, typename acc_vec>
__device__ void builtin_wmma_naive_selector(const src_vec&, const src_vec&, acc_vec&)
{
}

template <>
__device__ void
builtin_wmma_naive_selector<half16_t,
                            StaticBufferTupleOfVector<AddressSpaceEnum::Vgpr, float, 1, 8, true>>(
    const half16_t& reg_a,
    const half16_t& reg_b,
    StaticBufferTupleOfVector<AddressSpaceEnum::Vgpr, float, 1, 8, true>& reg_c)
{
    intrin_wmma_f32_16x16x16_f16_w32<16, 16>::Run(
        reg_a, reg_b, reg_c.GetVectorTypeReference(Number<0>{}));
}

template <>
__device__ void
builtin_wmma_naive_selector<bhalf16_t,
                            StaticBufferTupleOfVector<AddressSpaceEnum::Vgpr, float, 1, 8, true>>(
    const bhalf16_t& reg_a,
    const bhalf16_t& reg_b,
    StaticBufferTupleOfVector<AddressSpaceEnum::Vgpr, float, 1, 8, true>& reg_c)
{
    intrin_wmma_f32_16x16x16_bf16_w32<16, 16>::Run(
        reg_a, reg_b, reg_c.GetVectorTypeReference(Number<0>{}));
}

template <>
__device__ void
builtin_wmma_naive_selector<half16_t,
                            StaticBufferTupleOfVector<AddressSpaceEnum::Vgpr, half_t, 1, 16, true>>(
    const half16_t& reg_a,
    const half16_t& reg_b,
    StaticBufferTupleOfVector<AddressSpaceEnum::Vgpr, half_t, 1, 16, true>& reg_c)
{
    intrin_wmma_f16_16x16x16_f16_w32<16, 16, 0>::Run(
        reg_a, reg_b, reg_c.GetVectorTypeReference(Number<0>{}));
}

template <>
__device__ void builtin_wmma_naive_selector<
    bhalf16_t,
    StaticBufferTupleOfVector<AddressSpaceEnum::Vgpr, bhalf_t, 1, 16, true>>(
    const bhalf16_t& reg_a,
    const bhalf16_t& reg_b,
    StaticBufferTupleOfVector<AddressSpaceEnum::Vgpr, bhalf_t, 1, 16, true>& reg_c)
{
    intrin_wmma_bf16_16x16x16_bf16_w32<16, 16, 0>::Run(
        reg_a, reg_b, reg_c.GetVectorTypeReference(Number<0>{}));
}

template <>
__device__ void
builtin_wmma_naive_selector<int8x16_t,
                            StaticBufferTupleOfVector<AddressSpaceEnum::Vgpr, int32_t, 1, 8, true>>(
    const int8x16_t& reg_a,
    const int8x16_t& reg_b,
    StaticBufferTupleOfVector<AddressSpaceEnum::Vgpr, int32_t, 1, 8, true>& reg_c)
{
    intrin_wmma_i32_16x16x16_iu8_w32<16, 16, true, true, false>::Run(
        reg_a, reg_b, reg_c.GetVectorTypeReference(Number<0>{}));
}

#ifdef CK_EXPERIMENTAL_BIT_INT_EXTENSION_INT4
template <>
__device__ void
builtin_wmma_naive_selector<int4x16_t,
                            StaticBufferTupleOfVector<AddressSpaceEnum::Vgpr, int32_t, 1, 8, true>>(
    const int4x16_t& reg_a,
    const int4x16_t& reg_b,
    StaticBufferTupleOfVector<AddressSpaceEnum::Vgpr, int32_t, 1, 8, true>& reg_c)
{
    intrin_wmma_i32_16x16x16_iu4_w32<16, 16, true, true, false>::Run(
        reg_a, reg_b, reg_c.GetVectorTypeReference(Number<0>{}));
}
#endif

template <typename src_t, typename dst_t, typename acc_t>
__global__ void matmul(const src_t* a, const src_t* b, dst_t* c)
{
    __shared__ src_t p_shared[16 * 16 * 2];
    const int lIdx = threadIdx.x;
    // a and b fragments are stored in 8 VGPRs each, in packed format, so 16 elements each for a and
    // b a_frag will store one column of the 16x16 matrix tile b_frag will store one row of the
    // 16x16 matrix tile
    using src_vec             = typename vector_type<src_t, 16>::type;
    constexpr index_t acc_num = WMMA_ACCNumber_traits<acc_t>::ACC_NUMBER;
    src_vec a_frag            = {};
    src_vec b_frag            = {};

    src_vec a_temp = {};
    src_vec b_temp = {};
    // initialize c fragment to 0
    using acc_vec = StaticBufferTupleOfVector<AddressSpaceEnum::Vgpr, acc_t, 1, acc_num, true>;
    acc_vec c_thread_buf_;

    // lane is (0-31) mod 16 instead of 0-31 due to matrix replication
    const int lane    = lIdx % 16;
    const int lane_lo = lIdx / 2;
    const int lane_hi = lIdx % 2;
    for(int ele = 0; ele < 8; ++ele)
    {
        a_temp[ele] = a[8 * lane_hi + 16 * lane_lo + ele];
    }

    for(int ele = 0; ele < 8; ++ele)
    {
        b_temp[ele] = b[8 * lane_hi + 16 * lane_lo + ele];
    }

    __syncthreads();

    for(int ele = 0; ele < 8; ++ele)
    {
        p_shared[8 * 16 * lane_hi + 8 * lane_lo + ele] = a_temp[ele];
    }

    for(int ele = 0; ele < 8; ++ele)
    {
        p_shared[8 * 16 * lane_hi + 8 * lane_lo + ele + 16 * 16] = b_temp[ele];
    }

    __syncthreads();

    for(int ele = 0; ele < 16; ++ele)
    {
        b_frag[ele] = p_shared[(ele / 8) * 16 * 8 + 8 * lane + ele % 8 + 16 * 16];
    }
    // follow origin design
    for(int ele = 0; ele < 16; ++ele)
    {
        a_frag[ele] = p_shared[(ele / 8) * 16 * 8 + 8 * lane + ele % 8];
    }

    __syncthreads();

    // sync threads, similar to mma_sync
    // __syncthreads();
    builtin_wmma_naive_selector<src_vec, acc_vec>(a_frag, b_frag, c_thread_buf_);
    // since only fp16_fp32 asm wmma implemented for experiment purpose, restrict test case to fp16
    // when enable this ck::amd_assembly_wmma_f32_16x16x16_f16_w32(a_frag, b_frag,
    // c_thread_buf_.GetVectorTypeReference(Number<0>{}).template AsType<float8_t>()(Number<0>{}));
    __syncthreads();
    // wait for results, similar to mma_sync
    static_for<0, 8, 1>{}([&](auto ele) {
        const int r = ele * 2 + (lIdx / 16);
        // store results from unpacked c_thread_buf_ output
        c[16 * r + lane] = ck::type_convert<dst_t>(c_thread_buf_[Number<ele * acc_num / 8>{}]);
    });
}

template <typename src_t, typename dst_t, typename acc_t>
__global__ void matmul_swizzle_a(const src_t* a, const src_t* b, dst_t* c)
{
    const int lIdx = threadIdx.x;

    using src_vec             = typename vector_type<src_t, 16>::type;
    constexpr index_t acc_num = WMMA_ACCNumber_traits<acc_t>::ACC_NUMBER;
    src_vec a_frag            = {};
    src_vec b_frag            = {};
    using acc_vec = StaticBufferTupleOfVector<AddressSpaceEnum::Vgpr, acc_t, 1, acc_num, true>;
    acc_vec c_thread_buf_;

    const int lane = lIdx % 16;

    for(int ele = 0; ele < 16; ++ele)
    {
        b_frag[ele] = b[16 * lane + ele];
    }

    const int offset_m = (((lane & 1) << 3) | (lane >> 1));
    for(int ele = 0; ele < 16; ++ele)
    {
        a_frag[ele] = a[16 * offset_m + ele];
    }

    __syncthreads();
    builtin_wmma_naive_selector<src_vec, acc_vec>(a_frag, b_frag, c_thread_buf_);
    __syncthreads();

    static_for<0, 8, 1>{}([&](auto ele) {
        const int blk = lIdx / 16;
        const int r   = ele;
        c[16 * 8 * blk + 16 * r + lane] =
            ck::type_convert<dst_t>(c_thread_buf_[Number<ele * acc_num / 8>{}]);
    });
}

template <typename srcA_t, typename srcB_t, typename dst_t, typename acc_t, ck::index_t kValue>
__global__ void matmul(const srcA_t* a, const srcB_t* b, dst_t* c)
{
    ignore = a;
    ignore = b;
    ignore = c;
}

template <typename srcA_t, typename srcB_t, typename dst_t, typename acc_t, ck::index_t kValue>
__global__ void matmul_swizzle_a(const srcA_t* a, const srcB_t* b, dst_t* c)
{
    ignore = a;
    ignore = b;
    ignore = c;
}
#endif

struct GemmParams
{
    GemmParams() : M(16), N(16), K(16), StrideA(16), StrideB(16), StrideC(16), alpha(1), beta(0) {}

    ck::index_t M;
    ck::index_t N;
    ck::index_t K;

    ck::index_t StrideA;
    ck::index_t StrideB;
    ck::index_t StrideC;

    float alpha;
    float beta;
};

template <typename GemmInstance,
          typename ADataType,
          typename BDataType,
          typename CDataType,
          typename AElementwiseOperation,
          typename BElementwiseOperation,
          typename CElementwiseOperation>
void RunHostGEMM(const Tensor<ADataType>& A,
                 const Tensor<BDataType>& B,
                 Tensor<CDataType>& C,
                 AElementwiseOperation a_element_op,
                 BElementwiseOperation b_element_op,
                 CElementwiseOperation c_element_op)
{
    auto ref_gemm     = GemmInstance{};
    auto ref_invoker  = ref_gemm.MakeInvoker();
    auto ref_argument = ref_gemm.MakeArgument(A, B, C, a_element_op, b_element_op, c_element_op);

    ref_invoker.Run(ref_argument);
}

template <typename KernelType, typename ADataType, typename BDataType, typename CDataType>
bool RunDeviceGEMM(KernelType kernel,
                   const Tensor<ADataType>& A,
                   const Tensor<BDataType>& B,
                   Tensor<CDataType>& C)
{
    DeviceMem a_m_k_device_buf(sizeof(ADataType) * A.mDesc.GetElementSpaceSize());
    DeviceMem b_n_k_device_buf(sizeof(BDataType) * B.mDesc.GetElementSpaceSize());
    DeviceMem c_m_n_device_buf(sizeof(CDataType) * C.mDesc.GetElementSpaceSize());

    a_m_k_device_buf.ToDevice(A.mData.data());
    b_n_k_device_buf.ToDevice(B.mData.data());
    kernel<<<1, 32>>>(static_cast<const ADataType*>(a_m_k_device_buf.GetDeviceBuffer()),
                      static_cast<const BDataType*>(b_n_k_device_buf.GetDeviceBuffer()),
                      static_cast<CDataType*>(c_m_n_device_buf.GetDeviceBuffer()));
    hipError_t err = hipGetLastError();
    if(err != hipSuccess)
    {
        std::cerr << "HIP kernel launch error: " << hipGetErrorString(err) << std::endl;
    }
    c_m_n_device_buf.FromDevice(C.mData.data());

    return true;
}

template <typename DeviceWmma,
          typename ADataType,
          typename BDataType,
          typename CDataType,
          typename GPUAccDataType,
          typename CPUAccDataType,
          typename ALayout,
          typename BLayout,
          typename CLayout,
          typename AElementwiseOperation,
          typename BElementwiseOperation,
          typename CElementwiseOperation,
          index_t kValue>
struct TestWmma
{
    auto PrepareGemmTensor(const ck::wmma_op_util::GemmParams& params)
    {
        auto f_host_tensor_descriptor =
            [](std::size_t row, std::size_t col, std::size_t stride, auto layout) {
                if(std::is_same<decltype(layout), ck::tensor_layout::gemm::RowMajor>::value)
                {
                    return HostTensorDescriptor(std::vector<std::size_t>({row, col}),
                                                std::vector<std::size_t>({stride, 1}));
                }
                else
                {
                    return HostTensorDescriptor(std::vector<std::size_t>({row, col}),
                                                std::vector<std::size_t>({1, stride}));
                }
            };

        Tensor<ADataType> a_m_k(
            f_host_tensor_descriptor(params.M, params.K, params.StrideA, ALayout{}));
        Tensor<BDataType> b_n_k(
            f_host_tensor_descriptor(params.K, params.N, params.StrideB, BLayout{}));
        Tensor<CDataType> c_m_n_host_result(
            f_host_tensor_descriptor(params.M, params.N, params.StrideC, CLayout{}));
        Tensor<CDataType> c_m_n_device_result(
            f_host_tensor_descriptor(params.M, params.N, params.StrideC, CLayout{}));

        auto f_generate_tensor_value = [](auto& tensor, auto type) {
            using dataType = decltype(type);

            tensor.GenerateTensorValue(GeneratorTensor_2<dataType>{-5, 5});
        };

        f_generate_tensor_value(a_m_k, ADataType{});
        f_generate_tensor_value(b_n_k, BDataType{});

        return std::make_tuple(a_m_k, b_n_k, c_m_n_host_result, c_m_n_device_result);
    }

    template <typename DataType>
    void dump_tensor(Tensor<DataType> mat)
    {
        std::cout << "mat [ " << std::endl;

        auto len = mat.GetLengths();
        for(uint32_t i = 0; i < len[0]; i++)
        {
            std::cout << "    [";
            for(uint32_t j = 0; j < len[1]; j++)
            {
                std::vector<std::size_t> idx({i, j});
                std::cout << ck::type_convert<float>(mat(idx)) << ", ";
            }
            std::cout << "]" << std::endl;
        }
        std::cout << "]" << std::endl;
    }

    auto operator()(const DeviceWmma& wmma_kernel)
    {
        // Arrange
        ck::wmma_op_util::GemmParams params;
        params.M       = 16;
        params.N       = 16;
        params.K       = kValue;
        params.StrideA = kValue;
        params.StrideB = kValue;
        params.StrideC = 16;

        auto host_tensors = PrepareGemmTensor(params);

        const Tensor<ADataType>& a  = std::get<0>(host_tensors);
        const Tensor<BDataType>& b  = std::get<1>(host_tensors);
        Tensor<CDataType>& c_host   = std::get<2>(host_tensors);
        Tensor<CDataType>& c_device = std::get<3>(host_tensors);

        auto a_element_op = AElementwiseOperation{};
        auto b_element_op = BElementwiseOperation{};
        auto c_element_op = CElementwiseOperation{};

        using ReferenceGemmInstance =
            ck::tensor_operation::host::ReferenceGemm<ADataType,
                                                      BDataType,
                                                      CDataType,
                                                      CPUAccDataType,
                                                      AElementwiseOperation,
                                                      BElementwiseOperation,
                                                      CElementwiseOperation>;
        ck::wmma_op_util::RunHostGEMM<ReferenceGemmInstance>(
            a, b, c_host, a_element_op, b_element_op, c_element_op);

        // Act
        bool is_supported = (ck::is_gfx11_supported() || ck::is_gfx12_supported()) &&
                            ck::wmma_op_util::RunDeviceGEMM(wmma_kernel, a, b, c_device);

        if(is_supported)
        {
            // Assert
            bool res = false;
            if(std::is_same<CDataType, float>::value)
            {
                res = ck::utils::check_err(c_device.mData, c_host.mData);
                std::cout << (res ? "SUCCESS" : "FAILURE") << std::endl;
            }
            else if(std::is_same<CDataType, ck::half_t>::value)
            {
                res = ck::utils::check_err(c_device.mData, c_host.mData);
                std::cout << (res ? "SUCCESS" : "FAILURE") << std::endl;
            }
            else if(std::is_same<CDataType, ck::bhalf_t>::value)
            {
                // 0.5 Pixel Error Tolerance is introduced by Accumulator difference.
                // BF16 WMMA Accumulator is in BF16 Type while On Host-side Accumulator is Float.
                res = ck::utils::check_err(
                    c_device.mData, c_host.mData, "Error: Incorrect results!", 0, 1.0);
                std::cout << (res ? "SUCCESS" : "FAILURE") << std::endl;
            }
            else if(std::is_same<CDataType, int8_t>::value)
            {
                res = ck::utils::check_err(c_device.mData, c_host.mData);
                std::cout << (res ? "SUCCESS" : "FAILURE") << std::endl;
            }
            else if(std::is_same<CDataType, double>::value)
            {
                res = ck::utils::check_err(c_device.mData, c_host.mData);
                std::cout << (res ? "SUCCESS" : "FAILURE") << std::endl;
            }
            else if(std::is_same<CDataType, int32_t>::value)
            {
                res = ck::utils::check_err(c_device.mData, c_host.mData);
                std::cout << (res ? "SUCCESS" : "FAILURE") << std::endl;
            }
            else
            {
                std::cout << "UNSUPPORTED CDataType" << std::endl;
            }

            return res;
        }
        else
        {
            return true;
        }
    }
};

} // namespace wmma_op_util
} // namespace ck
