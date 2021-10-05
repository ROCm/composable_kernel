#ifndef CK_AMD_XDLOPS_HPP
#define CK_AMD_XDLOPS_HPP

#include "data_type.hpp"

namespace ck {

// A, B, C, cbsz, abid, blgp
// fp32
extern "C" __device__ float32_t llvm_intrin_amdgcn_mfma_f32_32x32x1f32(
    float, float, float32_t, int, int, int) __asm("llvm.amdgcn.mfma.f32.32x32x1f32");

extern "C" __device__ float16_t llvm_intrin_amdgcn_mfma_f32_32x32x2f32(
    float, float, float16_t, int, int, int) __asm("llvm.amdgcn.mfma.f32.32x32x2f32");

extern "C" __device__ float4_t llvm_intrin_amdgcn_mfma_f32_16x16x4f32(
    float, float, float4_t, int, int, int) __asm("llvm.amdgcn.mfma.f32.16x16x4f32");

extern "C" __device__ float16_t llvm_intrin_amdgcn_mfma_f32_16x16x1f32(
    float, float, float16_t, int, int, int) __asm("llvm.amdgcn.mfma.f32.16x16x1f32");

extern "C" __device__ float4_t llvm_intrin_amdgcn_mfma_f32_4x4x1f32(
    float, float, float4_t, int, int, int) __asm("llvm.amdgcn.mfma.f32.4x4x1f32");

// fp16
extern "C" __device__ float32_t llvm_intrin_amdgcn_mfma_f32_32x32x4f16(
    half4_t, half4_t, float32_t, int, int, int) __asm("llvm.amdgcn.mfma.f32.32x32x4f16");

extern "C" __device__ float16_t llvm_intrin_amdgcn_mfma_f32_32x32x8f16(
    half4_t, half4_t, float16_t, int, int, int) __asm("llvm.amdgcn.mfma.f32.32x32x8f16");

extern "C" __device__ float4_t llvm_intrin_amdgcn_mfma_f32_16x16x16f16(
    half4_t, half4_t, float4_t, int, int, int) __asm("llvm.amdgcn.mfma.f32.16x16x16f16");

extern "C" __device__ float16_t llvm_intrin_amdgcn_mfma_f32_16x16x4f16(
    half4_t, half4_t, float16_t, int, int, int) __asm("llvm.amdgcn.mfma.f32.16x16x4f16");

extern "C" __device__ float4_t llvm_intrin_amdgcn_mfma_f32_4x4x4f16(
    half4_t, half4_t, float4_t, int, int, int) __asm("llvm.amdgcn.mfma.f32.4x4x4f16");

// bfp16
extern "C" __device__ float16_t llvm_intrin_amdgcn_mfma_f32_32x32x8bf16_1k(
    ushort4_t, ushort4_t, float16_t, int, int, int) __asm("llvm.amdgcn.mfma.f32.32x32x8bf16.1k");

extern "C" __device__ float4_t llvm_intrin_amdgcn_mfma_f32_16x16x16bf16_1k(
    ushort4_t, ushort4_t, float4_t, int, int, int) __asm("llvm.amdgcn.mfma.f32.16x16x16bf16.1k");

extern "C" __device__ float32_t llvm_intrin_amdgcn_mfma_f32_32x32x2bf16(
    ushort2_t, ushort2_t, float32_t, int, int, int) __asm("llvm.amdgcn.mfma.f32.32x32x2bf16");

extern "C" __device__ float16_t llvm_intrin_amdgcn_mfma_f32_32x32x4bf16(
    ushort2_t, ushort2_t, float16_t, int, int, int) __asm("llvm.amdgcn.mfma.f32.32x32x4bf16");

extern "C" __device__ float4_t llvm_intrin_amdgcn_mfma_f32_16x16x8bf16(
    ushort2_t, ushort2_t, float4_t, int, int, int) __asm("llvm.amdgcn.mfma.f32.16x16x8bf16");

extern "C" __device__ float16_t llvm_intrin_amdgcn_mfma_f32_16x16x2bf16(
    ushort2_t, ushort2_t, float16_t, int, int, int) __asm("llvm.amdgcn.mfma.f32.16x16x2bf16");

extern "C" __device__ float4_t llvm_intrin_amdgcn_mfma_f32_4x4x2bf16(
    ushort2_t, ushort2_t, float4_t, int, int, int) __asm("llvm.amdgcn.mfma.f32.4x4x2bf16");

// fp32
template <index_t MPerWave, index_t NPerWave>
struct intrin_mfma_f32_32x32x1f32;

template <>
struct intrin_mfma_f32_32x32x1f32<64, 64>
{
    template <class FloatC>
    __device__ static void Run(const float& reg_a, const float& reg_b, FloatC& reg_c)
    {
        reg_c.template AsType<float32_t>()(Number<0>{}) = llvm_intrin_amdgcn_mfma_f32_32x32x1f32(
            reg_a, reg_b, reg_c.template AsType<float32_t>()[Number<0>{}], 1, 0, 0);
        reg_c.template AsType<float32_t>()(Number<1>{}) = llvm_intrin_amdgcn_mfma_f32_32x32x1f32(
            reg_a, reg_b, reg_c.template AsType<float32_t>()[Number<1>{}], 1, 1, 0);
    }
};

template <>
struct intrin_mfma_f32_32x32x1f32<32, 64>
{
    template <class FloatC>
    __device__ static void Run(const float& reg_a, const float& reg_b, FloatC& reg_c)
    {
        reg_c.template AsType<float32_t>()(Number<0>{}) = llvm_intrin_amdgcn_mfma_f32_32x32x1f32(
            reg_a, reg_b, reg_c.template AsType<float32_t>()[Number<0>{}], 1, 0, 0);
    }
};

template <index_t MPerWave, index_t NPerWave>
struct intrin_mfma_f32_32x32x2f32;

template <>
struct intrin_mfma_f32_32x32x2f32<32, 32>
{
    template <class FloatC>
    __device__ static void Run(const float& reg_a, const float& reg_b, FloatC& reg_c)
    {
        reg_c.template AsType<float16_t>()(Number<0>{}) = llvm_intrin_amdgcn_mfma_f32_32x32x2f32(
            reg_a, reg_b, reg_c.template AsType<float16_t>()[Number<0>{}], 0, 0, 0);
    }
};

template <index_t MPerWave, index_t NPerWave>
struct intrin_mfma_f32_16x16x4f32;

template <>
struct intrin_mfma_f32_16x16x4f32<16, 16>
{
    template <class FloatC>
    __device__ static void Run(const float& reg_a, const float& reg_b, FloatC& reg_c)
    {
        reg_c.template AsType<float4_t>()(Number<0>{}) = llvm_intrin_amdgcn_mfma_f32_16x16x4f32(
            reg_a, reg_b, reg_c.template AsType<float4_t>()[Number<0>{}], 0, 0, 0);
    }
};

template <index_t MPerWave, index_t NPerWave>
struct intrin_mfma_f32_16x16x1f32;

template <>
struct intrin_mfma_f32_16x16x1f32<16, 64>
{
    template <class FloatC>
    __device__ static void Run(const float& reg_a, const float& reg_b, FloatC& reg_c)
    {

        reg_c.template AsType<float16_t>()(Number<0>{}) = llvm_intrin_amdgcn_mfma_f32_16x16x1f32(
            reg_a, reg_b, reg_c.template AsType<float16_t>()[Number<0>{}], 2, 0, 0);
    }
};

template <index_t MPerWave, index_t NPerWave>
struct intrin_mfma_f32_4x4x1f32;

template <>
struct intrin_mfma_f32_4x4x1f32<4, 64>
{
    template <class FloatC>
    __device__ static void Run(const float& reg_a, const float& reg_b, FloatC& reg_c)
    {
        reg_c.template AsType<float4_t>()(Number<0>{}) = llvm_intrin_amdgcn_mfma_f32_4x4x1f32(
            reg_a, reg_b, reg_c.template AsType<float4_t>()[Number<0>{}], 4, 0, 0);
    }
};

template <>
struct intrin_mfma_f32_4x4x1f32<8, 64>
{
    template <class FloatC>
    __device__ static void Run(const float& reg_a, const float& reg_b, FloatC& reg_c)
    {
        reg_c.template AsType<float4_t>()(Number<0>{}) = llvm_intrin_amdgcn_mfma_f32_4x4x1f32(
            reg_a, reg_b, reg_c.template AsType<float4_t>()[Number<0>{}], 4, 0, 0);
        reg_c.template AsType<float4_t>()(Number<1>{}) = llvm_intrin_amdgcn_mfma_f32_4x4x1f32(
            reg_a, reg_b, reg_c.template AsType<float4_t>()[Number<1>{}], 4, 1, 0);
    }
};

// fp16
template <index_t MPerWave, index_t NPerWave>
struct intrin_mfma_f32_32x32x4f16;

template <>
struct intrin_mfma_f32_32x32x4f16<64, 64>
{
    template <class FloatC>
    __device__ static void Run(const half4_t& reg_a, const half4_t& reg_b, FloatC& reg_c)
    {
        reg_c.template AsType<float32_t>()(Number<0>{}) = llvm_intrin_amdgcn_mfma_f32_32x32x4f16(
            reg_a, reg_b, reg_c.template AsType<float32_t>()[Number<0>{}], 1, 0, 0);
        reg_c.template AsType<float32_t>()(Number<1>{}) = llvm_intrin_amdgcn_mfma_f32_32x32x4f16(
            reg_a, reg_b, reg_c.template AsType<float32_t>()[Number<1>{}], 1, 1, 0);
    }
};

template <>
struct intrin_mfma_f32_32x32x4f16<32, 64>
{
    template <class FloatC>
    __device__ static void Run(const half4_t& reg_a, const half4_t& reg_b, FloatC& reg_c)
    {
        reg_c.template AsType<float32_t>()(Number<0>{}) = llvm_intrin_amdgcn_mfma_f32_32x32x4f16(
            reg_a, reg_b, reg_c.template AsType<float32_t>()[Number<0>{}], 1, 0, 0);
    }
};

template <index_t MPerWave, index_t NPerWave>
struct intrin_mfma_f32_32x32x8f16;

template <>
struct intrin_mfma_f32_32x32x8f16<32, 32>
{
    template <class FloatC>
    __device__ static void Run(const half4_t& reg_a, const half4_t& reg_b, FloatC& reg_c)
    {
        reg_c.template AsType<float16_t>()(Number<0>{}) = llvm_intrin_amdgcn_mfma_f32_32x32x8f16(
            reg_a, reg_b, reg_c.template AsType<float16_t>()[Number<0>{}], 0, 0, 0);
    }
};

template <index_t MPerWave, index_t NPerWave>
struct intrin_mfma_f32_16x16x16f16;

template <>
struct intrin_mfma_f32_16x16x16f16<16, 16>
{
    template <class FloatC>
    __device__ static void Run(const half4_t& reg_a, const half4_t& reg_b, FloatC& reg_c)
    {
        reg_c.template AsType<float4_t>()(Number<0>{}) = llvm_intrin_amdgcn_mfma_f32_16x16x16f16(
            reg_a, reg_b, reg_c.template AsType<float4_t>()[Number<0>{}], 0, 0, 0);
    }
};

template <index_t MPerWave, index_t NPerWave>
struct intrin_mfma_f32_16x16x4f16;

template <>
struct intrin_mfma_f32_16x16x4f16<16, 64>
{
    template <class FloatC>
    __device__ static void Run(const half4_t& reg_a, const half4_t& reg_b, FloatC& reg_c)
    {
        reg_c.template AsType<float16_t>()(Number<0>{}) = llvm_intrin_amdgcn_mfma_f32_16x16x4f16(
            reg_a, reg_b, reg_c.template AsType<float16_t>()[Number<0>{}], 2, 0, 0);
    }
};

template <index_t MPerWave, index_t NPerWave>
struct intrin_mfma_f32_4x4x4f16;

template <>
struct intrin_mfma_f32_4x4x4f16<4, 64>
{
    template <class FloatC>
    __device__ static void Run(const half4_t& reg_a, const half4_t& reg_b, FloatC& reg_c)
    {
        reg_c.template AsType<float4_t>()(Number<0>{}) = llvm_intrin_amdgcn_mfma_f32_4x4x4f16(
            reg_a, reg_b, reg_c.template AsType<float4_t>()[Number<0>{}], 4, 0, 0);
    }
};

template <>
struct intrin_mfma_f32_4x4x4f16<8, 64>
{
    template <class FloatC>
    __device__ static void Run(const half4_t& reg_a, const half4_t& reg_b, FloatC& reg_c)
    {
        reg_c.template AsType<float4_t>()(Number<0>{}) = llvm_intrin_amdgcn_mfma_f32_4x4x4f16(
            reg_a, reg_b, reg_c.template AsType<float4_t>()[Number<0>{}], 4, 0, 0);
        reg_c.template AsType<float4_t>()(Number<1>{}) = llvm_intrin_amdgcn_mfma_f32_4x4x4f16(
            reg_a, reg_b, reg_c.template AsType<float4_t>()[Number<1>{}], 4, 1, 0);
    }
};

// bfp16
template <index_t MPerWave, index_t NPerWave>
struct intrin_mfma_f32_32x32x8bf16_1k;

template <>
struct intrin_mfma_f32_32x32x8bf16_1k<32, 32>
{
    template <class FloatC>
    __device__ static void Run(const ushort4_t& reg_a, const ushort4_t& reg_b, FloatC& reg_c)
    {
        reg_c.template AsType<float16_t>()(Number<0>{}) =
            llvm_intrin_amdgcn_mfma_f32_32x32x8bf16_1k(
                reg_a, reg_b, reg_c.template AsType<float16_t>()[Number<0>{}], 0, 0, 0);
    }
};

template <index_t MPerWave, index_t NPerWave>
struct intrin_mfma_f32_16x16x16bf16_1k;

template <>
struct intrin_mfma_f32_16x16x16bf16_1k<16, 16>
{
    template <class FloatC>
    __device__ static void Run(const ushort4_t& reg_a, const ushort4_t& reg_b, FloatC& reg_c)
    {
        reg_c.template AsType<float4_t>()(Number<0>{}) =
            llvm_intrin_amdgcn_mfma_f32_16x16x16bf16_1k(
                reg_a, reg_b, reg_c.template AsType<float4_t>()[Number<0>{}], 0, 0, 0);
    }
};

template <index_t MPerWave, index_t NPerWave>
struct intrin_mfma_f32_32x32x4bf16;

template <>
struct intrin_mfma_f32_32x32x4bf16<32, 32>
{
    template <class FloatC>
    __device__ static void Run(const ushort2_t& reg_a, const ushort2_t& reg_b, FloatC& reg_c)
    {
        reg_c.template AsType<float16_t>()(Number<0>{}) = llvm_intrin_amdgcn_mfma_f32_32x32x4bf16(
            reg_a, reg_b, reg_c.template AsType<float16_t>()[Number<0>{}], 0, 0, 0);
    }
};

template <index_t MPerWave, index_t NPerWave>
struct intrin_mfma_f32_16x16x8bf16;

template <>
struct intrin_mfma_f32_16x16x8bf16<16, 16>
{
    template <class FloatC>
    __device__ static void Run(const ushort2_t& reg_a, const ushort2_t& reg_b, FloatC& reg_c)
    {
        reg_c.template AsType<float4_t>()(Number<0>{}) = llvm_intrin_amdgcn_mfma_f32_16x16x8bf16(
            reg_a, reg_b, reg_c.template AsType<float4_t>()[Number<0>{}], 0, 0, 0);
    }
};

} // namespace ck
#endif
