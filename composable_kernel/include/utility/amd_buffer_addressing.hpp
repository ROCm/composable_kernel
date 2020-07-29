#ifndef CK_AMD_BUFFER_ADDRESSING_HPP
#define CK_AMD_BUFFER_ADDRESSING_HPP

#include "float_type.hpp"

namespace ck {

// For 128 bit SGPRs to supply resource constant in buffer instructions
// https://rocm-documentation.readthedocs.io/en/latest/GCN_ISA_Manuals/testdocbook.html#vector-memory-buffer-instructions
template <typename T>
union BufferResourceConstant
{
    int32x4_t data;
    T* address[2];
    int32_t range[4];
    int32_t config[4];
};

__device__ float __llvm_amdgcn_buffer_load_f32(int32x4_t srsrc,
                                               index_t vindex,
                                               index_t offset,
                                               bool glc,
                                               bool slc) __asm("llvm.amdgcn.buffer.load.f32");

__device__ float2_t
__llvm_amdgcn_buffer_load_f32x2(int32x4_t srsrc,
                                index_t vindex,
                                index_t offset,
                                bool glc,
                                bool slc) __asm("llvm.amdgcn.buffer.load.v2f32");

__device__ float4_t
__llvm_amdgcn_buffer_load_f32x4(int32x4_t srsrc,
                                index_t vindex,
                                index_t offset,
                                bool glc,
                                bool slc) __asm("llvm.amdgcn.buffer.load.v4f32");

__device__ half_t __llvm_amdgcn_buffer_load_f16(int32x4_t srsrc,
                                                index_t vindex,
                                                index_t offset,
                                                bool glc,
                                                bool slc) __asm("llvm.amdgcn.buffer.load.f16");

__device__ half2_t __llvm_amdgcn_buffer_load_f16x2(int32x4_t srsrc,
                                                   index_t vindex,
                                                   index_t offset,
                                                   bool glc,
                                                   bool slc) __asm("llvm.amdgcn.buffer.load.v2f16");

__device__ half4_t __llvm_amdgcn_buffer_load_f16x4(int32x4_t srsrc,
                                                   index_t vindex,
                                                   index_t offset,
                                                   bool glc,
                                                   bool slc) __asm("llvm.amdgcn.buffer.load.v4f16");

__device__ ushort __llvm_amdgcn_buffer_load_bf16(int32x4_t srsrc,
                                                 index_t vindex,
                                                 index_t offset,
                                                 bool glc,
                                                 bool slc) __asm("llvm.amdgcn.buffer.load.bf16");

__device__ ushort2_t
__llvm_amdgcn_buffer_load_bf16x2(int32x4_t srsrc,
                                 index_t vindex,
                                 index_t offset,
                                 bool glc,
                                 bool slc) __asm("llvm.amdgcn.buffer.load.v2bf16");

__device__ ushort4_t
__llvm_amdgcn_buffer_load_bf16x4(int32x4_t srsrc,
                                 index_t vindex,
                                 index_t offset,
                                 bool glc,
                                 bool slc) __asm("llvm.amdgcn.buffer.load.v4bf16");

__device__ void __llvm_amdgcn_buffer_store_f32(float vdata,
                                               int32x4_t srsrc,
                                               index_t vindex,
                                               index_t offset,
                                               bool glc,
                                               bool slc) __asm("llvm.amdgcn.buffer.store.f32");

__device__ void __llvm_amdgcn_buffer_store_f32x2(float2_t vdata,
                                                 int32x4_t srsrc,
                                                 index_t vindex,
                                                 index_t offset,
                                                 bool glc,
                                                 bool slc) __asm("llvm.amdgcn.buffer.store.v2f32");

__device__ void __llvm_amdgcn_buffer_store_f32x4(float4_t vdata,
                                                 int32x4_t srsrc,
                                                 index_t vindex,
                                                 index_t offset,
                                                 bool glc,
                                                 bool slc) __asm("llvm.amdgcn.buffer.store.v4f32");

__device__ void __llvm_amdgcn_buffer_store_f16(half_t vdata,
                                               int32x4_t srsrc,
                                               index_t vindex,
                                               index_t offset,
                                               bool glc,
                                               bool slc) __asm("llvm.amdgcn.buffer.store.f16");

__device__ void __llvm_amdgcn_buffer_store_f16x2(half2_t vdata,
                                                 int32x4_t srsrc,
                                                 index_t vindex,
                                                 index_t offset,
                                                 bool glc,
                                                 bool slc) __asm("llvm.amdgcn.buffer.store.v2f16");

__device__ void __llvm_amdgcn_buffer_store_f16x4(half4_t vdata,
                                                 int32x4_t srsrc,
                                                 index_t vindex,
                                                 index_t offset,
                                                 bool glc,
                                                 bool slc) __asm("llvm.amdgcn.buffer.store.v4f16");

__device__ void __llvm_amdgcn_buffer_store_bf16(ushort vdata,
                                                int32x4_t srsrc,
                                                index_t vindex,
                                                index_t offset,
                                                bool glc,
                                                bool slc) __asm("llvm.amdgcn.buffer.store.bf16");

__device__ void
__llvm_amdgcn_buffer_store_bf16x2(ushort2_t vdata,
                                  int32x4_t srsrc,
                                  index_t vindex,
                                  index_t offset,
                                  bool glc,
                                  bool slc) __asm("llvm.amdgcn.buffer.store.v2bf16");

__device__ void
__llvm_amdgcn_buffer_store_bf16x4(ushort4_t vdata,
                                  int32x4_t srsrc,
                                  index_t vindex,
                                  index_t offset,
                                  bool glc,
                                  bool slc) __asm("llvm.amdgcn.buffer.store.v4bf16");

__device__ void
__llvm_amdgcn_buffer_atomic_add_f32(float vdata,
                                    int32x4_t srsrc,
                                    index_t vindex,
                                    index_t offset,
                                    bool slc) __asm("llvm.amdgcn.buffer.atomic.fadd.f32");

// buffer_load requires:
//   1) p_src_thread must be in global memory space, p_dst_thread must be vgpr
//   2) p_src_thread to be a wavewise pointer.
// It is user's responsibility to make sure that is true.
template <typename T, index_t VectorSize>
__device__ typename vector_type<T, VectorSize>::MemoryType
amd_buffer_load(const T* p_src_wave,
                index_t src_thread_data_offset,
                bool src_thread_data_valid,
                index_t src_elemenst_space);

// buffer_store requires:
//   1) p_src_thread must be in vgpr space, p_dst_thread must be global memory
//   2) p_dst_thread to be a wavewise pointer.
// It is user's responsibility to make sure that is true.
template <typename T, index_t VectorSize>
__device__ void amd_buffer_store(const T* p_src_thread,
                                 T* p_dst_wave,
                                 index_t dst_thread_data_offset,
                                 bool dst_thread_data_valid,
                                 index_t dst_data_range);

// buffer_atomic requires:
//   1) p_src_thread must be in vgpr space, p_dst_thread must be global memory
//   2) p_dst_thread to be a wavewise pointer.
// It is user's responsibility to make sure that is true.
template <typename T, index_t VectorSize>
__device__ void amd_buffer_atomic_add(const T* p_src_thread,
                                      T* p_dst_wave,
                                      index_t dst_thread_data_offset,
                                      bool dst_thread_data_valid,
                                      index_t dst_data_range);

template <>
__device__ float amd_buffer_load<float, 1>(const float* p_src_wave,
                                           index_t src_thread_data_offset,
                                           bool src_thread_data_valid,
                                           index_t src_data_range)
{
    BufferResourceConstant<float> src_wave_buffer_resource;

    // wavewise base address (64 bit)
    src_wave_buffer_resource.address[0] = const_cast<float*>(p_src_wave);
    // wavewise range (32 bit)
    src_wave_buffer_resource.range[2] = src_data_range * sizeof(float);
    // wavewise setting (32 bit)
    src_wave_buffer_resource.config[3] = 0x00027000;

    index_t src_thread_addr_offset = src_thread_data_offset * sizeof(float);

#if 1 // debug
#if !CK_EXPERIMENTAL_AMD_BUFFER_ADDRESSING_USE_OFFSET_TRICK
    return __llvm_amdgcn_buffer_load_f32(src_wave_buffer_resource.data,
                                         0,
                                         src_thread_data_valid ? src_thread_addr_offset
                                                               : 0xffffffff,
                                         false,
                                         false);
#else
    uint32_t src_addr_shift = src_thread_data_valid ? 0 : 0x7fffffff;

    return __llvm_amdgcn_buffer_load_f32(
        src_wave_buffer_resource.data, 0, src_addr_shift + src_thread_addr_offset, false, false);
#endif
#else
    return src_thread_data_valid
               ? __llvm_amdgcn_buffer_load_f32(
                     src_wave_buffer_resource.data, 0, src_thread_addr_offset, false, false)
               : 0;
#endif
}

template <>
__device__ float2_t amd_buffer_load<float, 2>(const float* p_src_wave,
                                              index_t src_thread_data_offset,
                                              bool src_thread_data_valid,
                                              index_t src_data_range)
{
    BufferResourceConstant<float> src_wave_buffer_resource;

    // wavewise base address (64 bit)
    src_wave_buffer_resource.address[0] = const_cast<float*>(p_src_wave);
    // wavewise range (32 bit)
    src_wave_buffer_resource.range[2] = src_data_range * sizeof(float);
    // wavewise setting (32 bit)
    src_wave_buffer_resource.config[3] = 0x00027000;

    index_t src_thread_addr_offset = src_thread_data_offset * sizeof(float);

#if !CK_EXPERIMENTAL_AMD_BUFFER_ADDRESSING_USE_OFFSET_TRICK
    return __llvm_amdgcn_buffer_load_f32x2(src_wave_buffer_resource.data,
                                           0,
                                           src_thread_data_valid ? src_thread_addr_offset
                                                                 : 0xffffffff,
                                           false,
                                           false);
#else
    uint32_t src_addr_shift = src_thread_data_valid ? 0 : 0x7fffffff;

    return __llvm_amdgcn_buffer_load_f32x2(
        src_wave_buffer_resource.data, 0, src_addr_shift + src_thread_addr_offset, false, false);
#endif
}

template <>
__device__ float4_t amd_buffer_load<float, 4>(const float* p_src_wave,
                                              index_t src_thread_data_offset,
                                              bool src_thread_data_valid,
                                              index_t src_data_range)
{
    BufferResourceConstant<float> src_wave_buffer_resource;

    // wavewise base address (64 bit)
    src_wave_buffer_resource.address[0] = const_cast<float*>(p_src_wave);
    // wavewise range (32 bit)
    src_wave_buffer_resource.range[2] = src_data_range * sizeof(float);
    // wavewise setting (32 bit)
    src_wave_buffer_resource.config[3] = 0x00027000;

    index_t src_thread_addr_offset = src_thread_data_offset * sizeof(float);

#if !CK_EXPERIMENTAL_AMD_BUFFER_ADDRESSING_USE_OFFSET_TRICK
    return __llvm_amdgcn_buffer_load_f32x4(src_wave_buffer_resource.data,
                                           0,
                                           src_thread_data_valid ? src_thread_addr_offset
                                                                 : 0xffffffff,
                                           false,
                                           false);
#else
    uint32_t src_addr_shift = src_thread_data_valid ? 0 : 0x7fffffff;

    return __llvm_amdgcn_buffer_load_f32x4(
        src_wave_buffer_resource.data, 0, src_addr_shift + src_thread_addr_offset, false, false);
#endif
}

template <>
__device__ half_t amd_buffer_load<half_t, 1>(const half_t* p_src_wave,
                                             index_t src_thread_data_offset,
                                             bool src_thread_data_valid,
                                             index_t src_data_range)
{
    BufferResourceConstant<half_t> src_wave_buffer_resource;

    // wavewise base address (64 bit)
    src_wave_buffer_resource.address[0] = const_cast<half_t*>(p_src_wave);
    // wavewise range (32 bit)
    src_wave_buffer_resource.range[2] = src_data_range * sizeof(half_t);
    // wavewise setting (32 bit)
    src_wave_buffer_resource.config[3] = 0x00027000;

#if !CK_WORKAROUND_SWDEV_231101
    index_t src_thread_addr_offset = src_thread_data_offset * sizeof(half_t);

#if !CK_EXPERIMENTAL_AMD_BUFFER_ADDRESSING_USE_OFFSET_TRICK
    return __llvm_amdgcn_buffer_load_f16(src_wave_buffer_resource.data,
                                         0,
                                         src_thread_data_valid ? src_thread_addr_offset
                                                               : 0xffffffff,
                                         false,
                                         false);
#else
    uint32_t src_addr_shift = src_thread_data_valid ? 0 : 0x7fffffff;

    return __llvm_amdgcn_buffer_load_f16(
        src_wave_buffer_resource.data, 0, src_addr_shift + src_thread_addr_offset, false, false);
#endif
#else
    return src_thread_data_valid ? p_src_wave[src_thread_data_offset] : 0;
#endif
}

template <>
__device__ half2_t amd_buffer_load<half_t, 2>(const half_t* p_src_wave,
                                              index_t src_thread_data_offset,
                                              bool src_thread_data_valid,
                                              index_t src_data_range)
{
    BufferResourceConstant<half_t> src_wave_buffer_resource;

    // wavewise base address (64 bit)
    src_wave_buffer_resource.address[0] = const_cast<half_t*>(p_src_wave);
    // wavewise range (32 bit)
    src_wave_buffer_resource.range[2] = src_data_range * sizeof(half_t);
    // wavewise setting (32 bit)
    src_wave_buffer_resource.config[3] = 0x00027000;

    index_t src_thread_addr_offset = src_thread_data_offset * sizeof(half_t);

#if !CK_EXPERIMENTAL_AMD_BUFFER_ADDRESSING_USE_OFFSET_TRICK
    float dst_out_tmp =
        __llvm_amdgcn_buffer_load_f32(src_wave_buffer_resource.data,
                                      0,
                                      src_thread_data_valid ? src_thread_addr_offset : 0xffffffff,
                                      false,
                                      false);
#else
    uint32_t src_addr_shift = src_thread_data_valid ? 0 : 0x7fffffff;

    float dst_out_tmp = __llvm_amdgcn_buffer_load_f32(
        src_wave_buffer_resource.data, 0, src_addr_shift + src_thread_addr_offset, false, false);
#endif

    return *reinterpret_cast<half2_t*>(&dst_out_tmp);
}

template <>
__device__ half4_t amd_buffer_load<half_t, 4>(const half_t* p_src_wave,
                                              index_t src_thread_data_offset,
                                              bool src_thread_data_valid,
                                              index_t src_data_range)
{
    BufferResourceConstant<half_t> src_wave_buffer_resource;

    // wavewise base address (64 bit)
    src_wave_buffer_resource.address[0] = const_cast<half_t*>(p_src_wave);
    // wavewise range (32 bit)
    src_wave_buffer_resource.range[2] = src_data_range * sizeof(half_t);
    // wavewise setting (32 bit)
    src_wave_buffer_resource.config[3] = 0x00027000;

    index_t src_thread_addr_offset = src_thread_data_offset * sizeof(half_t);

#if !CK_EXPERIMENTAL_AMD_BUFFER_ADDRESSING_USE_OFFSET_TRICK
    float2_t dst_out_tmp =
        __llvm_amdgcn_buffer_load_f32x2(src_wave_buffer_resource.data,
                                        0,
                                        src_thread_data_valid ? src_thread_addr_offset : 0xffffffff,
                                        false,
                                        false);
#else
    uint32_t src_addr_shift = src_thread_data_valid ? 0 : 0x7fffffff;

    float2_t dst_out_tmp = __llvm_amdgcn_buffer_load_f32x2(
        src_wave_buffer_resource.data, 0, src_addr_shift + src_thread_addr_offset, false, false);
#endif

    return *reinterpret_cast<half4_t*>(&dst_out_tmp);
}

template <>
__device__ half8_t amd_buffer_load<half_t, 8>(const half_t* p_src_wave,
                                              index_t src_thread_data_offset,
                                              bool src_thread_data_valid,
                                              index_t src_data_range)
{
    BufferResourceConstant<half_t> src_wave_buffer_resource;

    // wavewise base address (64 bit)
    src_wave_buffer_resource.address[0] = const_cast<half_t*>(p_src_wave);
    // wavewise range (32 bit)
    src_wave_buffer_resource.range[2] = src_data_range * sizeof(half_t);
    // wavewise setting (32 bit)
    src_wave_buffer_resource.config[3] = 0x00027000;

    index_t src_thread_addr_offset = src_thread_data_offset * sizeof(half_t);

#if !CK_EXPERIMENTAL_AMD_BUFFER_ADDRESSING_USE_OFFSET_TRICK
    float4_t dst_out_tmp =
        __llvm_amdgcn_buffer_load_f32x4(src_wave_buffer_resource.data,
                                        0,
                                        src_thread_data_valid ? src_thread_addr_offset : 0xffffffff,
                                        false,
                                        false);
#else
    uint32_t src_addr_shift = src_thread_data_valid ? 0 : 0x7fffffff;

    float4_t dst_out_tmp = __llvm_amdgcn_buffer_load_f32x4(
        src_wave_buffer_resource.data, 0, src_addr_shift + src_thread_addr_offset, false, false);
#endif

    return *reinterpret_cast<half8_t*>(&dst_out_tmp);
}

template <>
__device__ ushort amd_buffer_load<ushort, 1>(const ushort* p_src_wave,
                                             index_t src_thread_data_offset,
                                             bool src_thread_data_valid,
                                             index_t src_data_range)
{
    BufferResourceConstant<ushort> src_wave_buffer_resource;

    // wavewise base address (64 bit)
    src_wave_buffer_resource.address[0] = const_cast<ushort*>(p_src_wave);
    // wavewise range (32 bit)
    src_wave_buffer_resource.range[2] = src_data_range * sizeof(ushort);
    // wavewise setting (32 bit)
    src_wave_buffer_resource.config[3] = 0x00027000;

#if !CK_WORKAROUND_SWDEV_231101
    index_t src_thread_addr_offset = src_thread_data_offset * sizeof(ushort);

#if !CK_EXPERIMENTAL_AMD_BUFFER_ADDRESSING_USE_OFFSET_TRICK
    return __llvm_amdgcn_buffer_load_bf16(src_wave_buffer_resource.data,
                                          0,
                                          src_thread_data_valid ? src_thread_addr_offset
                                                                : 0xffffffff,
                                          false,
                                          false);
#else
    uint32_t src_addr_shift = src_thread_data_valid ? 0 : 0x7fffffff;

    return __llvm_amdgcn_buffer_load_bf16(
        src_wave_buffer_resource.data, 0, src_addr_shift + src_thread_addr_offset, false, false);
#endif

#else
    return src_thread_data_valid ? p_src_wave[src_thread_data_offset] : 0;
#endif
}

template <>
__device__ ushort2_t amd_buffer_load<ushort, 2>(const ushort* p_src_wave,
                                                index_t src_thread_data_offset,
                                                bool src_thread_data_valid,
                                                index_t src_data_range)
{
    BufferResourceConstant<ushort> src_wave_buffer_resource;

    // wavewise base address (64 bit)
    src_wave_buffer_resource.address[0] = const_cast<ushort*>(p_src_wave);
    // wavewise range (32 bit)
    src_wave_buffer_resource.range[2] = src_data_range * sizeof(ushort);
    // wavewise setting (32 bit)
    src_wave_buffer_resource.config[3] = 0x00027000;

    index_t src_thread_addr_offset = src_thread_data_offset * sizeof(ushort);

#if !CK_EXPERIMENTAL_AMD_BUFFER_ADDRESSING_USE_OFFSET_TRICK
    float dst_out_tmp =
        __llvm_amdgcn_buffer_load_f32(src_wave_buffer_resource.data,
                                      0,
                                      src_thread_data_valid ? src_thread_addr_offset : 0xffffffff,
                                      false,
                                      false);
#else
    uint32_t src_addr_shift = src_thread_data_valid ? 0 : 0x7fffffff;

    float dst_out_tmp = __llvm_amdgcn_buffer_load_f32(
        src_wave_buffer_resource.data, 0, src_addr_shift + src_thread_addr_offset, false, false);
#endif

    return *reinterpret_cast<ushort2_t*>(&dst_out_tmp);
}

template <>
__device__ ushort4_t amd_buffer_load<ushort, 4>(const ushort* p_src_wave,
                                                index_t src_thread_data_offset,
                                                bool src_thread_data_valid,
                                                index_t src_data_range)
{
    BufferResourceConstant<ushort> src_wave_buffer_resource;

    // wavewise base address (64 bit)
    src_wave_buffer_resource.address[0] = const_cast<ushort*>(p_src_wave);
    // wavewise range (32 bit)
    src_wave_buffer_resource.range[2] = src_data_range * sizeof(ushort);
    // wavewise setting (32 bit)
    src_wave_buffer_resource.config[3] = 0x00027000;

    index_t src_thread_addr_offset = src_thread_data_offset * sizeof(ushort);

#if !CK_EXPERIMENTAL_AMD_BUFFER_ADDRESSING_USE_OFFSET_TRICK
    float2_t dst_out_tmp =
        __llvm_amdgcn_buffer_load_f32x2(src_wave_buffer_resource.data,
                                        0,
                                        src_thread_data_valid ? src_thread_addr_offset : 0xffffffff,
                                        false,
                                        false);
#else
    uint32_t src_addr_shift = src_thread_data_valid ? 0 : 0x7fffffff;

    float2_t dst_out_tmp = __llvm_amdgcn_buffer_load_f32x2(
        src_wave_buffer_resource.data, 0, src_addr_shift + src_thread_addr_offset, false, false);
#endif

    return *reinterpret_cast<ushort4_t*>(&dst_out_tmp);
}

template <>
__device__ ushort8_t amd_buffer_load<ushort, 8>(const ushort* p_src_wave,
                                                index_t src_thread_data_offset,
                                                bool src_thread_data_valid,
                                                index_t src_data_range)
{
    BufferResourceConstant<ushort> src_wave_buffer_resource;

    // wavewise base address (64 bit)
    src_wave_buffer_resource.address[0] = const_cast<ushort*>(p_src_wave);
    // wavewise range (32 bit)
    src_wave_buffer_resource.range[2] = src_data_range * sizeof(ushort);
    // wavewise setting (32 bit)
    src_wave_buffer_resource.config[3] = 0x00027000;

    index_t src_thread_addr_offset = src_thread_data_offset * sizeof(ushort);

#if !CK_EXPERIMENTAL_AMD_BUFFER_ADDRESSING_USE_OFFSET_TRICK
    float4_t dst_out_tmp =
        __llvm_amdgcn_buffer_load_f32x4(src_wave_buffer_resource.data,
                                        0,
                                        src_thread_data_valid ? src_thread_addr_offset : 0xffffffff,
                                        false,
                                        false);
#else
    uint32_t src_addr_shift = src_thread_data_valid ? 0 : 0x7fffffff;

    float4_t dst_out_tmp = __llvm_amdgcn_buffer_load_f32x4(
        src_wave_buffer_resource.data, 0, src_addr_shift + src_thread_addr_offset, false, false);
#endif

    return *reinterpret_cast<ushort8_t*>(&dst_out_tmp);
}

template <>
__device__ void amd_buffer_store<float, 1>(const float* p_src_thread,
                                           float* p_dst_wave,
                                           index_t dst_thread_data_offset,
                                           bool dst_thread_data_valid,
                                           index_t dst_data_range)
{
    BufferResourceConstant<float> dst_wave_buffer_resource;

    // wavewise base address (64 bit)
    dst_wave_buffer_resource.address[0] = p_dst_wave;
    // wavewise range (32 bit)
    dst_wave_buffer_resource.range[2] = dst_data_range * sizeof(float);
    // wavewise setting (32 bit)
    dst_wave_buffer_resource.config[3] = 0x00027000;

    index_t dst_thread_addr_offset = dst_thread_data_offset * sizeof(float);

#if 1 // debug
#if !CK_EXPERIMENTAL_AMD_BUFFER_ADDRESSING_USE_OFFSET_TRICK
    __llvm_amdgcn_buffer_store_f32(*p_src_thread,
                                   dst_wave_buffer_resource.data,
                                   0,
                                   dst_thread_data_valid ? dst_thread_addr_offset : 0xffffffff,
                                   false,
                                   false);
#else
    uint32_t dst_addr_shift = dst_thread_data_valid ? 0 : 0x7fffffff;

    __llvm_amdgcn_buffer_store_f32(*p_src_thread,
                                   dst_wave_buffer_resource.data,
                                   0,
                                   dst_addr_shift + dst_thread_addr_offset,
                                   false,
                                   false);
#endif
#else
    if(dst_thread_data_valid)
    {
        __llvm_amdgcn_buffer_store_f32(
            *p_src_thread, dst_wave_buffer_resource.data, 0, dst_thread_addr_offset, false, false);
    }
#endif
}

template <>
__device__ void amd_buffer_store<float, 2>(const float* p_src_thread,
                                           float* p_dst_wave,
                                           index_t dst_thread_data_offset,
                                           bool dst_thread_data_valid,
                                           index_t dst_data_range)
{
    BufferResourceConstant<float> dst_wave_buffer_resource;

    // wavewise base address (64 bit)
    dst_wave_buffer_resource.address[0] = p_dst_wave;
    // wavewise range (32 bit)
    dst_wave_buffer_resource.range[2] = dst_data_range * sizeof(float);
    // wavewise setting (32 bit)
    dst_wave_buffer_resource.config[3] = 0x00027000;

    index_t dst_thread_addr_offset = dst_thread_data_offset * sizeof(float);

#if !CK_EXPERIMENTAL_AMD_BUFFER_ADDRESSING_USE_OFFSET_TRICK
    __llvm_amdgcn_buffer_store_f32x2(*reinterpret_cast<const float2_t*>(p_src_thread),
                                     dst_wave_buffer_resource.data,
                                     0,
                                     dst_thread_data_valid ? dst_thread_addr_offset : 0xffffffff,
                                     false,
                                     false);
#else
    uint32_t dst_addr_shift = dst_thread_data_valid ? 0 : 0x7fffffff;

    __llvm_amdgcn_buffer_store_f32x2(*reinterpret_cast<const float2_t*>(p_src_thread),
                                     dst_wave_buffer_resource.data,
                                     0,
                                     dst_addr_shift + dst_thread_addr_offset,
                                     false,
                                     false);
#endif
}

template <>
__device__ void amd_buffer_store<float, 4>(const float* p_src_thread,
                                           float* p_dst_wave,
                                           index_t dst_thread_data_offset,
                                           bool dst_thread_data_valid,
                                           index_t dst_data_range)
{
    BufferResourceConstant<float> dst_wave_buffer_resource;

    // wavewise base address (64 bit)
    dst_wave_buffer_resource.address[0] = p_dst_wave;
    // wavewise range (32 bit)
    dst_wave_buffer_resource.range[2] = dst_data_range * sizeof(float);
    // wavewise setting (32 bit)
    dst_wave_buffer_resource.config[3] = 0x00027000;

    index_t dst_thread_addr_offset = dst_thread_data_offset * sizeof(float);

#if !CK_EXPERIMENTAL_AMD_BUFFER_ADDRESSING_USE_OFFSET_TRICK
    __llvm_amdgcn_buffer_store_f32x4(*reinterpret_cast<const float4_t*>(p_src_thread),
                                     dst_wave_buffer_resource.data,
                                     0,
                                     dst_thread_data_valid ? dst_thread_addr_offset : 0xffffffff,
                                     false,
                                     false);
#else
    uint32_t dst_addr_shift = dst_thread_data_valid ? 0 : 0x7fffffff;

    __llvm_amdgcn_buffer_store_f32x4(*reinterpret_cast<const float4_t*>(p_src_thread),
                                     dst_wave_buffer_resource.data,
                                     0,
                                     dst_addr_shift + dst_thread_addr_offset,
                                     false,
                                     false);
#endif
}

template <>
__device__ void amd_buffer_store<half_t, 1>(const half_t* p_src_thread,
                                            half_t* p_dst_wave,
                                            index_t dst_thread_data_offset,
                                            bool dst_thread_data_valid,
                                            index_t dst_data_range)
{
    BufferResourceConstant<half_t> dst_wave_buffer_resource;

    // wavewise base address (64 bit)
    dst_wave_buffer_resource.address[0] = p_dst_wave;
    // wavewise range (32 bit)
    dst_wave_buffer_resource.range[2] = dst_data_range * sizeof(half_t);
    // wavewise setting (32 bit)
    dst_wave_buffer_resource.config[3] = 0x00027000;

#if !CK_WORKAROUND_SWDEV_231101
    index_t dst_thread_addr_offset = dst_thread_data_offset * sizeof(half_t);

#if !CK_EXPERIMENTAL_AMD_BUFFER_ADDRESSING_USE_OFFSET_TRICK
    __llvm_amdgcn_buffer_store_f16(*p_src_thread,
                                   dst_wave_buffer_resource.data,
                                   0,
                                   dst_thread_data_valid ? dst_thread_addr_offset : 0xffffffff,
                                   false,
                                   false);
#else
    uint32_t dst_addr_shift = dst_thread_data_valid ? 0 : 0x7fffffff;

    __llvm_amdgcn_buffer_store_f16(*p_src_thread,
                                   dst_wave_buffer_resource.data,
                                   0,
                                   dst_addr_shift + dst_thread_addr_offset,
                                   false,
                                   false);
#endif

#else
    if(dst_thread_data_valid)
    {
        p_dst_wave[dst_thread_data_offset] = *p_src_thread;
    }
#endif
}

template <>
__device__ void amd_buffer_store<half_t, 2>(const half_t* p_src_thread,
                                            half_t* p_dst_wave,
                                            index_t dst_thread_data_offset,
                                            bool dst_thread_data_valid,
                                            index_t dst_data_range)
{
    BufferResourceConstant<half_t> dst_wave_buffer_resource;

    // wavewise base address (64 bit)
    dst_wave_buffer_resource.address[0] = p_dst_wave;
    // wavewise range (32 bit)
    dst_wave_buffer_resource.range[2] = dst_data_range * sizeof(half_t);
    // wavewise setting (32 bit)
    dst_wave_buffer_resource.config[3] = 0x00027000;

    index_t dst_thread_addr_offset = dst_thread_data_offset * sizeof(half_t);

    const float* p_src_tmp = reinterpret_cast<const float*>(p_src_thread);

#if !CK_EXPERIMENTAL_AMD_BUFFER_ADDRESSING_USE_OFFSET_TRICK
    __llvm_amdgcn_buffer_store_f32(*p_src_tmp,
                                   dst_wave_buffer_resource.data,
                                   0,
                                   dst_thread_data_valid ? dst_thread_addr_offset : 0xffffffff,
                                   false,
                                   false);
#else
    uint32_t dst_addr_shift = dst_thread_data_valid ? 0 : 0x7fffffff;

    __llvm_amdgcn_buffer_store_f32(*p_src_tmp,
                                   dst_wave_buffer_resource.data,
                                   0,
                                   dst_addr_shift + dst_thread_addr_offset,
                                   false,
                                   false);
#endif
}

template <>
__device__ void amd_buffer_store<half_t, 4>(const half_t* p_src_thread,
                                            half_t* p_dst_wave,
                                            index_t dst_thread_data_offset,
                                            bool dst_thread_data_valid,
                                            index_t dst_data_range)
{
    BufferResourceConstant<half_t> dst_wave_buffer_resource;

    // wavewise base address (64 bit)
    dst_wave_buffer_resource.address[0] = p_dst_wave;
    // wavewise range (32 bit)
    dst_wave_buffer_resource.range[2] = dst_data_range * sizeof(half_t);
    // wavewise setting (32 bit)
    dst_wave_buffer_resource.config[3] = 0x00027000;

    index_t dst_thread_addr_offset = dst_thread_data_offset * sizeof(half_t);

    const float2_t* p_src_tmp = reinterpret_cast<const float2_t*>(p_src_thread);

#if !CK_EXPERIMENTAL_AMD_BUFFER_ADDRESSING_USE_OFFSET_TRICK
    __llvm_amdgcn_buffer_store_f32x2(*p_src_tmp,
                                     dst_wave_buffer_resource.data,
                                     0,
                                     dst_thread_data_valid ? dst_thread_addr_offset : 0xffffffff,
                                     false,
                                     false);
#else
    uint32_t dst_addr_shift = dst_thread_data_valid ? 0 : 0x7fffffff;

    __llvm_amdgcn_buffer_store_f32x2(*p_src_tmp,
                                     dst_wave_buffer_resource.data,
                                     0,
                                     dst_addr_shift + dst_thread_addr_offset,
                                     false,
                                     false);
#endif
}

template <>
__device__ void amd_buffer_store<half_t, 8>(const half_t* p_src_thread,
                                            half_t* p_dst_wave,
                                            index_t dst_thread_data_offset,
                                            bool dst_thread_data_valid,
                                            index_t dst_data_range)
{
    BufferResourceConstant<half_t> dst_wave_buffer_resource;

    // wavewise base address (64 bit)
    dst_wave_buffer_resource.address[0] = p_dst_wave;
    // wavewise range (32 bit)
    dst_wave_buffer_resource.range[2] = dst_data_range * sizeof(half_t);
    // wavewise setting (32 bit)
    dst_wave_buffer_resource.config[3] = 0x00027000;

    index_t dst_thread_addr_offset = dst_thread_data_offset * sizeof(half_t);

    const float4_t* p_src_tmp = reinterpret_cast<const float4_t*>(p_src_thread);

#if !CK_EXPERIMENTAL_AMD_BUFFER_ADDRESSING_USE_OFFSET_TRICK
    __llvm_amdgcn_buffer_store_f32x4(*p_src_tmp,
                                     dst_wave_buffer_resource.data,
                                     0,
                                     dst_thread_data_valid ? dst_thread_addr_offset : 0xffffffff,
                                     false,
                                     false);
#else
    uint32_t dst_addr_shift = dst_thread_data_valid ? 0 : 0x7fffffff;

    __llvm_amdgcn_buffer_store_f32x4(*p_src_tmp,
                                     dst_wave_buffer_resource.data,
                                     0,
                                     dst_addr_shift + dst_thread_addr_offset,
                                     false,
                                     false);
#endif
}

template <>
__device__ void amd_buffer_store<ushort, 1>(const ushort* p_src_thread,
                                            ushort* p_dst_wave,
                                            index_t dst_thread_data_offset,
                                            bool dst_thread_data_valid,
                                            index_t dst_data_range)
{
    BufferResourceConstant<ushort> dst_wave_buffer_resource;

    // wavewise base address (64 bit)
    dst_wave_buffer_resource.address[0] = p_dst_wave;
    // wavewise range (32 bit)
    dst_wave_buffer_resource.range[2] = dst_data_range * sizeof(ushort);
    // wavewise setting (32 bit)
    dst_wave_buffer_resource.config[3] = 0x00027000;

#if !CK_WORKAROUND_SWDEV_231101
    index_t dst_thread_addr_offset = dst_thread_data_offset * sizeof(ushort);

#if !CK_EXPERIMENTAL_AMD_BUFFER_ADDRESSING_USE_OFFSET_TRICK
    __llvm_amdgcn_buffer_store_bf16(*p_src_thread,
                                    dst_wave_buffer_resource.data,
                                    0,
                                    dst_thread_data_valid ? dst_thread_addr_offset : 0xffffffff,
                                    false,
                                    false);
#else
    uint32_t dst_addr_shift = dst_thread_data_valid ? 0 : 0x7fffffff;

    __llvm_amdgcn_buffer_store_bf16(*p_src_thread,
                                    dst_wave_buffer_resource.data,
                                    0,
                                    dst_addr_shift + dst_thread_addr_offset,
                                    false,
                                    false);
#endif

#else
    if(dst_thread_data_valid)
    {
        p_dst_wave[dst_thread_data_offset] = *p_src_thread;
    }
#endif
}

template <>
__device__ void amd_buffer_store<ushort, 2>(const ushort* p_src_thread,
                                            ushort* p_dst_wave,
                                            index_t dst_thread_data_offset,
                                            bool dst_thread_data_valid,
                                            index_t dst_data_range)
{
    BufferResourceConstant<ushort> dst_wave_buffer_resource;

    // wavewise base address (64 bit)
    dst_wave_buffer_resource.address[0] = p_dst_wave;
    // wavewise range (32 bit)
    dst_wave_buffer_resource.range[2] = dst_data_range * sizeof(ushort);
    // wavewise setting (32 bit)
    dst_wave_buffer_resource.config[3] = 0x00027000;

    index_t dst_thread_addr_offset = dst_thread_data_offset * sizeof(ushort);

    const float* p_src_tmp = reinterpret_cast<const float*>(p_src_thread);

#if !CK_EXPERIMENTAL_AMD_BUFFER_ADDRESSING_USE_OFFSET_TRICK
    __llvm_amdgcn_buffer_store_f32(*p_src_tmp,
                                   dst_wave_buffer_resource.data,
                                   0,
                                   dst_thread_data_valid ? dst_thread_addr_offset : 0xffffffff,
                                   false,
                                   false);
#else
    uint32_t dst_addr_shift = dst_thread_data_valid ? 0 : 0x7fffffff;

    __llvm_amdgcn_buffer_store_f32(*p_src_tmp,
                                   dst_wave_buffer_resource.data,
                                   0,
                                   dst_addr_shift + dst_thread_addr_offset,
                                   false,
                                   false);
#endif
}

template <>
__device__ void amd_buffer_store<ushort, 4>(const ushort* p_src_thread,
                                            ushort* p_dst_wave,
                                            index_t dst_thread_data_offset,
                                            bool dst_thread_data_valid,
                                            index_t dst_data_range)
{
    BufferResourceConstant<ushort> dst_wave_buffer_resource;

    // wavewise base address (64 bit)
    dst_wave_buffer_resource.address[0] = p_dst_wave;
    // wavewise range (32 bit)
    dst_wave_buffer_resource.range[2] = dst_data_range * sizeof(ushort);
    // wavewise setting (32 bit)
    dst_wave_buffer_resource.config[3] = 0x00027000;

    index_t dst_thread_addr_offset = dst_thread_data_offset * sizeof(ushort);

    const float2_t* p_src_tmp = reinterpret_cast<const float2_t*>(p_src_thread);

#if !CK_EXPERIMENTAL_AMD_BUFFER_ADDRESSING_USE_OFFSET_TRICK
    __llvm_amdgcn_buffer_store_f32x2(*p_src_tmp,
                                     dst_wave_buffer_resource.data,
                                     0,
                                     dst_thread_data_valid ? dst_thread_addr_offset : 0xffffffff,
                                     false,
                                     false);
#else
    uint32_t dst_addr_shift = dst_thread_data_valid ? 0 : 0x7fffffff;

    __llvm_amdgcn_buffer_store_f32x2(*p_src_tmp,
                                     dst_wave_buffer_resource.data,
                                     0,
                                     dst_addr_shift + dst_thread_addr_offset,
                                     false,
                                     false);
#endif
}

template <>
__device__ void amd_buffer_store<ushort, 8>(const ushort* p_src_thread,
                                            ushort* p_dst_wave,
                                            index_t dst_thread_data_offset,
                                            bool dst_thread_data_valid,
                                            index_t dst_data_range)
{
    BufferResourceConstant<ushort> dst_wave_buffer_resource;

    // wavewise base address (64 bit)
    dst_wave_buffer_resource.address[0] = p_dst_wave;
    // wavewise range (32 bit)
    dst_wave_buffer_resource.range[2] = dst_data_range * sizeof(ushort);
    // wavewise setting (32 bit)
    dst_wave_buffer_resource.config[3] = 0x00027000;

    index_t dst_thread_addr_offset = dst_thread_data_offset * sizeof(ushort);

    const float4_t* p_src_tmp = reinterpret_cast<const float4_t*>(p_src_thread);

#if !CK_EXPERIMENTAL_AMD_BUFFER_ADDRESSING_USE_OFFSET_TRICK
    __llvm_amdgcn_buffer_store_f32x4(*p_src_tmp,
                                     dst_wave_buffer_resource.data,
                                     0,
                                     dst_thread_data_valid ? dst_thread_addr_offset : 0xffffffff,
                                     false,
                                     false);
#else
    uint32_t dst_addr_shift = dst_thread_data_valid ? 0 : 0x7fffffff;

    __llvm_amdgcn_buffer_store_f32x4(*p_src_tmp,
                                     dst_wave_buffer_resource.data,
                                     0,
                                     dst_addr_shift + dst_thread_addr_offset,
                                     false,
                                     false);
#endif
}

template <>
__device__ void amd_buffer_atomic_add<float, 1>(const float* p_src_thread,
                                                float* p_dst_wave,
                                                index_t dst_thread_data_offset,
                                                bool dst_thread_data_valid,
                                                index_t dst_data_range)
{
    BufferResourceConstant<float> dst_wave_buffer_resource;

    // wavewise base address (64 bit)
    dst_wave_buffer_resource.address[0] = p_dst_wave;
    // wavewise range (32 bit)
    dst_wave_buffer_resource.range[2] = dst_data_range * sizeof(float);
    // wavewise setting (32 bit)
    dst_wave_buffer_resource.config[3] = 0x00027000;

    index_t dst_thread_addr_offset = dst_thread_data_offset * sizeof(float);

#if !CK_EXPERIMENTAL_AMD_BUFFER_ADDRESSING_USE_OFFSET_TRICK
    __llvm_amdgcn_buffer_atomic_add_f32(*p_src_thread,
                                        dst_wave_buffer_resource.data,
                                        0,
                                        dst_thread_data_valid ? dst_thread_addr_offset : 0xffffffff,
                                        false);
#else
    uint32_t dst_addr_shift = dst_thread_data_valid ? 0 : 0x7fffffff;

    __llvm_amdgcn_buffer_atomic_add_f32(*p_src_thread,
                                        dst_wave_buffer_resource.data,
                                        0,
                                        dst_addr_shift + dst_thread_addr_offset,
                                        false);
#endif
}

template <>
__device__ void amd_buffer_atomic_add<float, 2>(const float* p_src_thread,
                                                float* p_dst_wave,
                                                index_t dst_thread_data_offset,
                                                bool dst_thread_data_valid,
                                                index_t dst_data_range)
{
    BufferResourceConstant<float> dst_wave_buffer_resource;

    // wavewise base address (64 bit)
    dst_wave_buffer_resource.address[0] = p_dst_wave;
    // wavewise range (32 bit)
    dst_wave_buffer_resource.range[2] = dst_data_range;
    // wavewise setting (32 bit)
    dst_wave_buffer_resource.config[3] = 0x00027000;

    index_t dst_thread_addr_offset = dst_thread_data_offset * sizeof(float);

#if !CK_EXPERIMENTAL_AMD_BUFFER_ADDRESSING_USE_OFFSET_TRICK
    for(index_t i = 0; i < 2; ++i)
    {
        __llvm_amdgcn_buffer_atomic_add_f32(
            p_src_thread[i],
            dst_wave_buffer_resource.data,
            0,
            dst_thread_data_valid ? (dst_thread_addr_offset + i * sizeof(float)) : 0xffffffff,
            false);
    }
#else
    uint32_t dst_addr_shift = dst_thread_data_valid ? 0 : 0x7fffffff;

    for(index_t i = 0; i < 2; ++i)
    {
        __llvm_amdgcn_buffer_atomic_add_f32(p_src_thread[i],
                                            dst_wave_buffer_resource.data,
                                            0,
                                            dst_addr_shift + dst_thread_addr_offset +
                                                i * sizeof(float),
                                            false);
    }
#endif
}

template <>
__device__ void amd_buffer_atomic_add<float, 4>(const float* p_src_thread,
                                                float* p_dst_wave,
                                                index_t dst_thread_data_offset,
                                                bool dst_thread_data_valid,
                                                index_t dst_data_range)
{
    BufferResourceConstant<float> dst_wave_buffer_resource;

    // wavewise base address (64 bit)
    dst_wave_buffer_resource.address[0] = p_dst_wave;
    // wavewise range (32 bit)
    dst_wave_buffer_resource.range[2] = dst_data_range * sizeof(float);
    // wavewise setting (32 bit)
    dst_wave_buffer_resource.config[3] = 0x00027000;

    index_t dst_thread_addr_offset = dst_thread_data_offset * sizeof(float);

#if !CK_EXPERIMENTAL_AMD_BUFFER_ADDRESSING_USE_OFFSET_TRICK
    for(index_t i = 0; i < 4; ++i)
    {
        __llvm_amdgcn_buffer_atomic_add_f32(
            p_src_thread[i],
            dst_wave_buffer_resource.data,
            0,
            dst_thread_data_valid ? (dst_thread_addr_offset + i * sizeof(float)) : 0xffffffff,
            false);
    }
#else
    uint32_t dst_addr_shift = dst_thread_data_valid ? 0 : 0x7fffffff;

    for(index_t i = 0; i < 4; ++i)
    {
        __llvm_amdgcn_buffer_atomic_add_f32(p_src_thread[i],
                                            dst_wave_buffer_resource.data,
                                            0,
                                            dst_addr_shift + dst_thread_addr_offset +
                                                i * sizeof(float),
                                            false);
    }
#endif
}

} // namespace ck
#endif
