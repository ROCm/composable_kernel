#ifndef CK_AMD_BUFFER_ADDRESSING_HPP
#define CK_AMD_BUFFER_ADDRESSING_HPP

#include "float_type.hpp"
#include "amd_buffer_addressing_v2.hpp"

namespace ck {

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
__device__ half_t
__llvm_amdgcn_raw_buffer_load_f16(int32x4_t rsrc,
                                  index_t voffset,
                                  index_t soffset,
                                  index_t glc_slc) __asm("llvm.amdgcn.raw.buffer.load.f16");

__device__ ushort
__llvm_amdgcn_raw_buffer_load_bf16(int32x4_t rsrc,
                                   index_t voffset,
                                   index_t soffset,
                                   index_t glc_slc) __asm("llvm.amdgcn.raw.buffer.load.bf16");

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

__device__ void
__llvm_amdgcn_raw_buffer_store_f16(half_t vdata,
                                   int32x4_t rsrc,
                                   index_t voffset,
                                   index_t soffset,
                                   index_t glc_slc) __asm("llvm.amdgcn.raw.buffer.store.f16");

__device__ void
__llvm_amdgcn_raw_buffer_store_bf16(ushort vdata,
                                    int32x4_t rsrc,
                                    index_t voffset,
                                    index_t soffset,
                                    index_t glc_slc) __asm("llvm.amdgcn.raw.buffer.store.bf16");

#if CK_USE_AMD_BUFFER_ATOMIC_FADD
#if CK_HIP_VERSION_FLAT >= 3010020405
// starting ROCm-3.10, the return type becomes float
__device__ float
#else
__device__ void
#endif
__llvm_amdgcn_buffer_atomic_add_f32(float vdata,
                                    int32x4_t rsrc,
                                    index_t vindex,
                                    index_t offset,
                                    bool slc) __asm("llvm.amdgcn.buffer.atomic.fadd.f32");
#endif

// buffer_load requires:
//   1) p_src_wave must be in global memory space
//   2) p_src_wave to be a wavewise pointer.
// It is user's responsibility to make sure that is true.
template <typename T, index_t VectorSize>
__device__ typename vector_type<T, VectorSize>::type amd_buffer_load(const T* p_src_wave,
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
    BufferResource<float> src_wave_buffer_resource;

    // wavewise base address (64 bit)
    src_wave_buffer_resource.address[0] = const_cast<float*>(p_src_wave);
    // wavewise range (32 bit)
    src_wave_buffer_resource.range[2] = src_data_range * sizeof(float);
    // wavewise setting (32 bit)
    src_wave_buffer_resource.config[3] = CK_BUFFER_RESOURCE_3RD_DWORD;

    index_t src_thread_addr_offset = src_thread_data_offset * sizeof(float);

#if CK_EXPERIMENTAL_USE_BUFFER_LOAD_OOB_CHECK_OFFSET_TRICK
    uint32_t src_addr_shift = src_thread_data_valid ? 0 : 0x7fffffff;

    return __llvm_amdgcn_buffer_load_f32(
        src_wave_buffer_resource.data, 0, src_addr_shift + src_thread_addr_offset, false, false);
#else
    float tmp = __llvm_amdgcn_buffer_load_f32(
        src_wave_buffer_resource.data, 0, src_thread_addr_offset, false, false);

    return src_thread_data_valid ? tmp : float(0);
#endif
}

template <>
__device__ float2_t amd_buffer_load<float, 2>(const float* p_src_wave,
                                              index_t src_thread_data_offset,
                                              bool src_thread_data_valid,
                                              index_t src_data_range)
{
    BufferResource<float> src_wave_buffer_resource;

    // wavewise base address (64 bit)
    src_wave_buffer_resource.address[0] = const_cast<float*>(p_src_wave);
    // wavewise range (32 bit)
    src_wave_buffer_resource.range[2] = src_data_range * sizeof(float);
    // wavewise setting (32 bit)
    src_wave_buffer_resource.config[3] = CK_BUFFER_RESOURCE_3RD_DWORD;

    index_t src_thread_addr_offset = src_thread_data_offset * sizeof(float);

#if CK_EXPERIMENTAL_USE_BUFFER_LOAD_OOB_CHECK_OFFSET_TRICK
    uint32_t src_addr_shift = src_thread_data_valid ? 0 : 0x7fffffff;

    return __llvm_amdgcn_buffer_load_f32x2(
        src_wave_buffer_resource.data, 0, src_addr_shift + src_thread_addr_offset, false, false);
#else
    float2_t tmp = __llvm_amdgcn_buffer_load_f32x2(
        src_wave_buffer_resource.data, 0, src_thread_addr_offset, false, false);

    return src_thread_data_valid ? tmp : float2_t(0);
#endif
}

template <>
__device__ float4_t amd_buffer_load<float, 4>(const float* p_src_wave,
                                              index_t src_thread_data_offset,
                                              bool src_thread_data_valid,
                                              index_t src_data_range)
{
    BufferResource<float> src_wave_buffer_resource;

    // wavewise base address (64 bit)
    src_wave_buffer_resource.address[0] = const_cast<float*>(p_src_wave);
    // wavewise range (32 bit)
    src_wave_buffer_resource.range[2] = src_data_range * sizeof(float);
    // wavewise setting (32 bit)
    src_wave_buffer_resource.config[3] = CK_BUFFER_RESOURCE_3RD_DWORD;

    index_t src_thread_addr_offset = src_thread_data_offset * sizeof(float);

#if CK_EXPERIMENTAL_USE_BUFFER_LOAD_OOB_CHECK_OFFSET_TRICK
    uint32_t src_addr_shift = src_thread_data_valid ? 0 : 0x7fffffff;

    return __llvm_amdgcn_buffer_load_f32x4(
        src_wave_buffer_resource.data, 0, src_addr_shift + src_thread_addr_offset, false, false);
#else
    float4_t tmp = __llvm_amdgcn_buffer_load_f32x4(
        src_wave_buffer_resource.data, 0, src_thread_addr_offset, false, false);

    return src_thread_data_valid ? tmp : float4_t(0);
#endif
}

template <>
__device__ half_t amd_buffer_load<half_t, 1>(const half_t* p_src_wave,
                                             index_t src_thread_data_offset,
                                             bool src_thread_data_valid,
                                             index_t src_data_range)
{
    BufferResource<half_t> src_wave_buffer_resource;

    // wavewise base address (64 bit)
    src_wave_buffer_resource.address[0] = const_cast<half_t*>(p_src_wave);
    // wavewise range (32 bit)
    src_wave_buffer_resource.range[2] = src_data_range * sizeof(half_t);
    // wavewise setting (32 bit)
    src_wave_buffer_resource.config[3] = CK_BUFFER_RESOURCE_3RD_DWORD;

    index_t src_thread_addr_offset = src_thread_data_offset * sizeof(half_t);

#if CK_EXPERIMENTAL_USE_BUFFER_LOAD_OOB_CHECK_OFFSET_TRICK
    uint32_t src_addr_shift = src_thread_data_valid ? 0 : 0x7fffffff;

    // current code cannot isolate Soffset and Voffset, so Soffset is hard-coded to 0, and
    // everything is passed to Voffset
    return __llvm_amdgcn_raw_buffer_load_f16(
        src_wave_buffer_resource.data, src_addr_shift + src_thread_addr_offset, 0, 0);
#else
    half_t zero(0);

    // current code cannot isolate Soffset and Voffset, so Soffset is hard-coded to 0, and
    // everything is passed to Voffset
    return src_thread_data_valid ? __llvm_amdgcn_raw_buffer_load_f16(
                                       src_wave_buffer_resource.data, src_thread_addr_offset, 0, 0)
                                 : zero;
#endif
}

template <>
__device__ half2_t amd_buffer_load<half_t, 2>(const half_t* p_src_wave,
                                              index_t src_thread_data_offset,
                                              bool src_thread_data_valid,
                                              index_t src_data_range)
{
    BufferResource<half_t> src_wave_buffer_resource;

    // wavewise base address (64 bit)
    src_wave_buffer_resource.address[0] = const_cast<half_t*>(p_src_wave);
    // wavewise range (32 bit)
    src_wave_buffer_resource.range[2] = src_data_range * sizeof(half_t);
    // wavewise setting (32 bit)
    src_wave_buffer_resource.config[3] = CK_BUFFER_RESOURCE_3RD_DWORD;

    index_t src_thread_addr_offset = src_thread_data_offset * sizeof(half_t);

#if CK_EXPERIMENTAL_USE_BUFFER_LOAD_OOB_CHECK_OFFSET_TRICK
    uint32_t src_addr_shift = src_thread_data_valid ? 0 : 0x7fffffff;

    float dst_out_tmp = __llvm_amdgcn_buffer_load_f32(
        src_wave_buffer_resource.data, 0, src_addr_shift + src_thread_addr_offset, false, false);

    return *reinterpret_cast<half2_t*>(&dst_out_tmp);
#else
    half2_t zeros(0);

    float dst_out_tmp = __llvm_amdgcn_buffer_load_f32(
        src_wave_buffer_resource.data, 0, src_thread_addr_offset, false, false);

    return src_thread_data_valid ? *reinterpret_cast<half2_t*>(&dst_out_tmp) : zeros;
#endif
}

template <>
__device__ half4_t amd_buffer_load<half_t, 4>(const half_t* p_src_wave,
                                              index_t src_thread_data_offset,
                                              bool src_thread_data_valid,
                                              index_t src_data_range)
{
    BufferResource<half_t> src_wave_buffer_resource;

    // wavewise base address (64 bit)
    src_wave_buffer_resource.address[0] = const_cast<half_t*>(p_src_wave);
    // wavewise range (32 bit)
    src_wave_buffer_resource.range[2] = src_data_range * sizeof(half_t);
    // wavewise setting (32 bit)
    src_wave_buffer_resource.config[3] = CK_BUFFER_RESOURCE_3RD_DWORD;

    index_t src_thread_addr_offset = src_thread_data_offset * sizeof(half_t);

#if CK_EXPERIMENTAL_USE_BUFFER_LOAD_OOB_CHECK_OFFSET_TRICK
    uint32_t src_addr_shift = src_thread_data_valid ? 0 : 0x7fffffff;

    float2_t dst_out_tmp = __llvm_amdgcn_buffer_load_f32x2(
        src_wave_buffer_resource.data, 0, src_addr_shift + src_thread_addr_offset, false, false);

    return *reinterpret_cast<half4_t*>(&dst_out_tmp);
#else
    half4_t zeros(0);

    float2_t dst_out_tmp = __llvm_amdgcn_buffer_load_f32x2(
        src_wave_buffer_resource.data, 0, src_thread_addr_offset, false, false);

    return src_thread_data_valid ? *reinterpret_cast<half4_t*>(&dst_out_tmp) : zeros;
#endif
}

template <>
__device__ half8_t amd_buffer_load<half_t, 8>(const half_t* p_src_wave,
                                              index_t src_thread_data_offset,
                                              bool src_thread_data_valid,
                                              index_t src_data_range)
{
    BufferResource<half_t> src_wave_buffer_resource;

    // wavewise base address (64 bit)
    src_wave_buffer_resource.address[0] = const_cast<half_t*>(p_src_wave);
    // wavewise range (32 bit)
    src_wave_buffer_resource.range[2] = src_data_range * sizeof(half_t);
    // wavewise setting (32 bit)
    src_wave_buffer_resource.config[3] = CK_BUFFER_RESOURCE_3RD_DWORD;

    index_t src_thread_addr_offset = src_thread_data_offset * sizeof(half_t);

#if CK_EXPERIMENTAL_USE_BUFFER_LOAD_OOB_CHECK_OFFSET_TRICK
    uint32_t src_addr_shift = src_thread_data_valid ? 0 : 0x7fffffff;

    float4_t dst_out_tmp = __llvm_amdgcn_buffer_load_f32x4(
        src_wave_buffer_resource.data, 0, src_addr_shift + src_thread_addr_offset, false, false);

    return *reinterpret_cast<half8_t*>(&dst_out_tmp);
#else
    half8_t zeros(0);

    float4_t dst_out_tmp = __llvm_amdgcn_buffer_load_f32x4(
        src_wave_buffer_resource.data, 0, src_thread_addr_offset, false, false);

    return src_thread_data_valid ? *reinterpret_cast<half8_t*>(&dst_out_tmp) : zeros;
#endif
}

template <>
__device__ ushort amd_buffer_load<ushort, 1>(const ushort* p_src_wave,
                                             index_t src_thread_data_offset,
                                             bool src_thread_data_valid,
                                             index_t src_data_range)
{
    BufferResource<ushort> src_wave_buffer_resource;

    // wavewise base address (64 bit)
    src_wave_buffer_resource.address[0] = const_cast<ushort*>(p_src_wave);
    // wavewise range (32 bit)
    src_wave_buffer_resource.range[2] = src_data_range * sizeof(ushort);
    // wavewise setting (32 bit)
    src_wave_buffer_resource.config[3] = CK_BUFFER_RESOURCE_3RD_DWORD;

    index_t src_thread_addr_offset = src_thread_data_offset * sizeof(ushort);

#if CK_EXPERIMENTAL_USE_BUFFER_LOAD_OOB_CHECK_OFFSET_TRICK
    uint32_t src_addr_shift = src_thread_data_valid ? 0 : 0x7fffffff;

    // current code cannot isolate Soffset and Voffset, so Soffset is hard-coded to 0, and
    // everything is passed to Voffset
    return __llvm_amdgcn_raw_buffer_load_bf16(
        src_wave_buffer_resource.data, src_addr_shift + src_thread_addr_offset, 0, 0);
#else
    ushort zero(0);

    // current code cannot isolate Soffset and Voffset, so Soffset is hard-coded to 0, and
    // everything is passed to Voffset
    return src_thread_data_valid ? __llvm_amdgcn_raw_buffer_load_bf16(
                                       src_wave_buffer_resource.data, src_thread_addr_offset, 0, 0)
                                 : zero;
#endif
}

template <>
__device__ ushort2_t amd_buffer_load<ushort, 2>(const ushort* p_src_wave,
                                                index_t src_thread_data_offset,
                                                bool src_thread_data_valid,
                                                index_t src_data_range)
{
    BufferResource<ushort> src_wave_buffer_resource;

    // wavewise base address (64 bit)
    src_wave_buffer_resource.address[0] = const_cast<ushort*>(p_src_wave);
    // wavewise range (32 bit)
    src_wave_buffer_resource.range[2] = src_data_range * sizeof(ushort);
    // wavewise setting (32 bit)
    src_wave_buffer_resource.config[3] = CK_BUFFER_RESOURCE_3RD_DWORD;

    index_t src_thread_addr_offset = src_thread_data_offset * sizeof(ushort);

#if CK_EXPERIMENTAL_USE_BUFFER_LOAD_OOB_CHECK_OFFSET_TRICK
    uint32_t src_addr_shift = src_thread_data_valid ? 0 : 0x7fffffff;

    float dst_out_tmp = __llvm_amdgcn_buffer_load_f32(
        src_wave_buffer_resource.data, 0, src_addr_shift + src_thread_addr_offset, false, false);

    return *reinterpret_cast<ushort2_t*>(&dst_out_tmp);
#else
    ushort2_t zeros(0);

    float dst_out_tmp = __llvm_amdgcn_buffer_load_f32(
        src_wave_buffer_resource.data, 0, src_thread_addr_offset, false, false);

    return src_thread_data_valid ? *reinterpret_cast<ushort2_t*>(&dst_out_tmp) : zeros;
#endif
}

template <>
__device__ ushort4_t amd_buffer_load<ushort, 4>(const ushort* p_src_wave,
                                                index_t src_thread_data_offset,
                                                bool src_thread_data_valid,
                                                index_t src_data_range)
{
    BufferResource<ushort> src_wave_buffer_resource;

    // wavewise base address (64 bit)
    src_wave_buffer_resource.address[0] = const_cast<ushort*>(p_src_wave);
    // wavewise range (32 bit)
    src_wave_buffer_resource.range[2] = src_data_range * sizeof(ushort);
    // wavewise setting (32 bit)
    src_wave_buffer_resource.config[3] = CK_BUFFER_RESOURCE_3RD_DWORD;

    index_t src_thread_addr_offset = src_thread_data_offset * sizeof(ushort);

#if CK_EXPERIMENTAL_USE_BUFFER_LOAD_OOB_CHECK_OFFSET_TRICK
    uint32_t src_addr_shift = src_thread_data_valid ? 0 : 0x7fffffff;

    float2_t dst_out_tmp = __llvm_amdgcn_buffer_load_f32x2(
        src_wave_buffer_resource.data, 0, src_addr_shift + src_thread_addr_offset, false, false);

    return *reinterpret_cast<ushort4_t*>(&dst_out_tmp);
#else
    ushort4_t zeros(0);

    float2_t dst_out_tmp = __llvm_amdgcn_buffer_load_f32x2(
        src_wave_buffer_resource.data, 0, src_thread_addr_offset, false, false);

    return src_thread_data_valid ? *reinterpret_cast<ushort4_t*>(&dst_out_tmp) : zeros;
#endif
}

template <>
__device__ ushort8_t amd_buffer_load<ushort, 8>(const ushort* p_src_wave,
                                                index_t src_thread_data_offset,
                                                bool src_thread_data_valid,
                                                index_t src_data_range)
{
    BufferResource<ushort> src_wave_buffer_resource;

    // wavewise base address (64 bit)
    src_wave_buffer_resource.address[0] = const_cast<ushort*>(p_src_wave);
    // wavewise range (32 bit)
    src_wave_buffer_resource.range[2] = src_data_range * sizeof(ushort);
    // wavewise setting (32 bit)
    src_wave_buffer_resource.config[3] = CK_BUFFER_RESOURCE_3RD_DWORD;

    index_t src_thread_addr_offset = src_thread_data_offset * sizeof(ushort);

#if CK_EXPERIMENTAL_USE_BUFFER_LOAD_OOB_CHECK_OFFSET_TRICK
    uint32_t src_addr_shift = src_thread_data_valid ? 0 : 0x7fffffff;

    float4_t dst_out_tmp = __llvm_amdgcn_buffer_load_f32x4(
        src_wave_buffer_resource.data, 0, src_addr_shift + src_thread_addr_offset, false, false);

    return *reinterpret_cast<ushort8_t*>(&dst_out_tmp);
#else
    ushort8_t zeros(0);

    float4_t dst_out_tmp = __llvm_amdgcn_buffer_load_f32x4(
        src_wave_buffer_resource.data, 0, src_thread_addr_offset, false, false);

    return src_thread_data_valid ? *reinterpret_cast<ushort8_t*>(&dst_out_tmp) : zeros;
#endif
}

template <>
__device__ void amd_buffer_store<float, 1>(const float* p_src_thread,
                                           float* p_dst_wave,
                                           index_t dst_thread_data_offset,
                                           bool dst_thread_data_valid,
                                           index_t dst_data_range)
{
    BufferResource<float> dst_wave_buffer_resource;

    // wavewise base address (64 bit)
    dst_wave_buffer_resource.address[0] = p_dst_wave;
    // wavewise range (32 bit)
    dst_wave_buffer_resource.range[2] = dst_data_range * sizeof(float);
    // wavewise setting (32 bit)
    dst_wave_buffer_resource.config[3] = CK_BUFFER_RESOURCE_3RD_DWORD;

    index_t dst_thread_addr_offset = dst_thread_data_offset * sizeof(float);

#if CK_EXPERIMENTAL_USE_BUFFER_STORE_OOB_CHECK_OFFSET_TRICK
    uint32_t dst_addr_shift = dst_thread_data_valid ? 0 : 0x7fffffff;

    __llvm_amdgcn_buffer_store_f32(*p_src_thread,
                                   dst_wave_buffer_resource.data,
                                   0,
                                   dst_addr_shift + dst_thread_addr_offset,
                                   false,
                                   false);
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
    BufferResource<float> dst_wave_buffer_resource;

    // wavewise base address (64 bit)
    dst_wave_buffer_resource.address[0] = p_dst_wave;
    // wavewise range (32 bit)
    dst_wave_buffer_resource.range[2] = dst_data_range * sizeof(float);
    // wavewise setting (32 bit)
    dst_wave_buffer_resource.config[3] = CK_BUFFER_RESOURCE_3RD_DWORD;

    index_t dst_thread_addr_offset = dst_thread_data_offset * sizeof(float);

#if CK_EXPERIMENTAL_USE_BUFFER_STORE_OOB_CHECK_OFFSET_TRICK
    uint32_t dst_addr_shift = dst_thread_data_valid ? 0 : 0x7fffffff;

    __llvm_amdgcn_buffer_store_f32x2(*reinterpret_cast<const float2_t*>(p_src_thread),
                                     dst_wave_buffer_resource.data,
                                     0,
                                     dst_addr_shift + dst_thread_addr_offset,
                                     false,
                                     false);
#else
    if(dst_thread_data_valid)
    {
        __llvm_amdgcn_buffer_store_f32x2(*reinterpret_cast<const float2_t*>(p_src_thread),
                                         dst_wave_buffer_resource.data,
                                         0,
                                         dst_thread_addr_offset,
                                         false,
                                         false);
    }
#endif
}

template <>
__device__ void amd_buffer_store<float, 4>(const float* p_src_thread,
                                           float* p_dst_wave,
                                           index_t dst_thread_data_offset,
                                           bool dst_thread_data_valid,
                                           index_t dst_data_range)
{
    BufferResource<float> dst_wave_buffer_resource;

    // wavewise base address (64 bit)
    dst_wave_buffer_resource.address[0] = p_dst_wave;
    // wavewise range (32 bit)
    dst_wave_buffer_resource.range[2] = dst_data_range * sizeof(float);
    // wavewise setting (32 bit)
    dst_wave_buffer_resource.config[3] = CK_BUFFER_RESOURCE_3RD_DWORD;

    index_t dst_thread_addr_offset = dst_thread_data_offset * sizeof(float);

#if CK_EXPERIMENTAL_USE_BUFFER_STORE_OOB_CHECK_OFFSET_TRICK
    uint32_t dst_addr_shift = dst_thread_data_valid ? 0 : 0x7fffffff;

    __llvm_amdgcn_buffer_store_f32x4(*reinterpret_cast<const float4_t*>(p_src_thread),
                                     dst_wave_buffer_resource.data,
                                     0,
                                     dst_addr_shift + dst_thread_addr_offset,
                                     false,
                                     false);
#else
    if(dst_thread_data_valid)
    {
        __llvm_amdgcn_buffer_store_f32x4(*reinterpret_cast<const float4_t*>(p_src_thread),
                                         dst_wave_buffer_resource.data,
                                         0,
                                         dst_thread_addr_offset,
                                         false,
                                         false);
    }
#endif
}

template <>
__device__ void amd_buffer_store<half_t, 1>(const half_t* p_src_thread,
                                            half_t* p_dst_wave,
                                            index_t dst_thread_data_offset,
                                            bool dst_thread_data_valid,
                                            index_t dst_data_range)
{
    BufferResource<half_t> dst_wave_buffer_resource;

    // wavewise base address (64 bit)
    dst_wave_buffer_resource.address[0] = p_dst_wave;
    // wavewise range (32 bit)
    dst_wave_buffer_resource.range[2] = dst_data_range * sizeof(half_t);
    // wavewise setting (32 bit)
    dst_wave_buffer_resource.config[3] = CK_BUFFER_RESOURCE_3RD_DWORD;

    index_t dst_thread_addr_offset = dst_thread_data_offset * sizeof(half_t);

#if CK_EXPERIMENTAL_USE_BUFFER_STORE_OOB_CHECK_OFFSET_TRICK
    uint32_t dst_addr_shift = dst_thread_data_valid ? 0 : 0x7fffffff;

    // current code cannot isolate Soffset and Voffset, so Soffset is hard-coded to 0, and
    // everything is passed to Voffset
    __llvm_amdgcn_raw_buffer_store_f16(*p_src_thread,
                                       dst_wave_buffer_resource.data,
                                       dst_addr_shift + dst_thread_addr_offset,
                                       0,
                                       0);
#else
    if(dst_thread_data_valid)
    {
        // current code cannot isolate Soffset and Voffset, so Soffset is hard-coded to 0, and
        // everything is passed to Voffset
        __llvm_amdgcn_raw_buffer_store_f16(
            *p_src_thread, dst_wave_buffer_resource.data, dst_thread_addr_offset, 0, 0);
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
    BufferResource<half_t> dst_wave_buffer_resource;

    // wavewise base address (64 bit)
    dst_wave_buffer_resource.address[0] = p_dst_wave;
    // wavewise range (32 bit)
    dst_wave_buffer_resource.range[2] = dst_data_range * sizeof(half_t);
    // wavewise setting (32 bit)
    dst_wave_buffer_resource.config[3] = CK_BUFFER_RESOURCE_3RD_DWORD;

    index_t dst_thread_addr_offset = dst_thread_data_offset * sizeof(half_t);

    const float* p_src_tmp = reinterpret_cast<const float*>(p_src_thread);

#if CK_EXPERIMENTAL_USE_BUFFER_STORE_OOB_CHECK_OFFSET_TRICK
    uint32_t dst_addr_shift = dst_thread_data_valid ? 0 : 0x7fffffff;

    __llvm_amdgcn_buffer_store_f32(*p_src_tmp,
                                   dst_wave_buffer_resource.data,
                                   0,
                                   dst_addr_shift + dst_thread_addr_offset,
                                   false,
                                   false);
#else
    if(dst_thread_data_valid)
    {
        __llvm_amdgcn_buffer_store_f32(
            *p_src_tmp, dst_wave_buffer_resource.data, 0, dst_thread_addr_offset, false, false);
    }
#endif
}

template <>
__device__ void amd_buffer_store<half_t, 4>(const half_t* p_src_thread,
                                            half_t* p_dst_wave,
                                            index_t dst_thread_data_offset,
                                            bool dst_thread_data_valid,
                                            index_t dst_data_range)
{
    BufferResource<half_t> dst_wave_buffer_resource;

    // wavewise base address (64 bit)
    dst_wave_buffer_resource.address[0] = p_dst_wave;
    // wavewise range (32 bit)
    dst_wave_buffer_resource.range[2] = dst_data_range * sizeof(half_t);
    // wavewise setting (32 bit)
    dst_wave_buffer_resource.config[3] = CK_BUFFER_RESOURCE_3RD_DWORD;

    index_t dst_thread_addr_offset = dst_thread_data_offset * sizeof(half_t);

    const float2_t* p_src_tmp = reinterpret_cast<const float2_t*>(p_src_thread);

#if CK_EXPERIMENTAL_USE_BUFFER_STORE_OOB_CHECK_OFFSET_TRICK
    uint32_t dst_addr_shift = dst_thread_data_valid ? 0 : 0x7fffffff;

    __llvm_amdgcn_buffer_store_f32x2(*p_src_tmp,
                                     dst_wave_buffer_resource.data,
                                     0,
                                     dst_addr_shift + dst_thread_addr_offset,
                                     false,
                                     false);
#else
    if(dst_thread_data_valid)
    {
        __llvm_amdgcn_buffer_store_f32x2(
            *p_src_tmp, dst_wave_buffer_resource.data, 0, dst_thread_addr_offset, false, false);
    }
#endif
}

template <>
__device__ void amd_buffer_store<half_t, 8>(const half_t* p_src_thread,
                                            half_t* p_dst_wave,
                                            index_t dst_thread_data_offset,
                                            bool dst_thread_data_valid,
                                            index_t dst_data_range)
{
    BufferResource<half_t> dst_wave_buffer_resource;

    // wavewise base address (64 bit)
    dst_wave_buffer_resource.address[0] = p_dst_wave;
    // wavewise range (32 bit)
    dst_wave_buffer_resource.range[2] = dst_data_range * sizeof(half_t);
    // wavewise setting (32 bit)
    dst_wave_buffer_resource.config[3] = CK_BUFFER_RESOURCE_3RD_DWORD;

    index_t dst_thread_addr_offset = dst_thread_data_offset * sizeof(half_t);

    const float4_t* p_src_tmp = reinterpret_cast<const float4_t*>(p_src_thread);

#if CK_EXPERIMENTAL_USE_BUFFER_STORE_OOB_CHECK_OFFSET_TRICK
    uint32_t dst_addr_shift = dst_thread_data_valid ? 0 : 0x7fffffff;

    __llvm_amdgcn_buffer_store_f32x4(*p_src_tmp,
                                     dst_wave_buffer_resource.data,
                                     0,
                                     dst_addr_shift + dst_thread_addr_offset,
                                     false,
                                     false);
#else
    if(dst_thread_data_valid)
    {
        __llvm_amdgcn_buffer_store_f32x4(
            *p_src_tmp, dst_wave_buffer_resource.data, 0, dst_thread_addr_offset, false, false);
    }
#endif
}

template <>
__device__ void amd_buffer_store<ushort, 1>(const ushort* p_src_thread,
                                            ushort* p_dst_wave,
                                            index_t dst_thread_data_offset,
                                            bool dst_thread_data_valid,
                                            index_t dst_data_range)
{
    BufferResource<ushort> dst_wave_buffer_resource;

    // wavewise base address (64 bit)
    dst_wave_buffer_resource.address[0] = p_dst_wave;
    // wavewise range (32 bit)
    dst_wave_buffer_resource.range[2] = dst_data_range * sizeof(ushort);
    // wavewise setting (32 bit)
    dst_wave_buffer_resource.config[3] = CK_BUFFER_RESOURCE_3RD_DWORD;

    index_t dst_thread_addr_offset = dst_thread_data_offset * sizeof(ushort);

#if CK_EXPERIMENTAL_USE_BUFFER_STORE_OOB_CHECK_OFFSET_TRICK
    uint32_t dst_addr_shift = dst_thread_data_valid ? 0 : 0x7fffffff;

    __llvm_amdgcn_raw_buffer_store_bf16(*p_src_thread,
                                        dst_wave_buffer_resource.data,
                                        dst_addr_shift + dst_thread_addr_offset,
                                        0,
                                        0);
#else
    if(dst_thread_data_valid)
    {
        __llvm_amdgcn_raw_buffer_store_bf16(
            *p_src_thread, dst_wave_buffer_resource.data, dst_thread_addr_offset, 0, 0);
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
    BufferResource<ushort> dst_wave_buffer_resource;

    // wavewise base address (64 bit)
    dst_wave_buffer_resource.address[0] = p_dst_wave;
    // wavewise range (32 bit)
    dst_wave_buffer_resource.range[2] = dst_data_range * sizeof(ushort);
    // wavewise setting (32 bit)
    dst_wave_buffer_resource.config[3] = CK_BUFFER_RESOURCE_3RD_DWORD;

    index_t dst_thread_addr_offset = dst_thread_data_offset * sizeof(ushort);

    const float* p_src_tmp = reinterpret_cast<const float*>(p_src_thread);

#if CK_EXPERIMENTAL_USE_BUFFER_STORE_OOB_CHECK_OFFSET_TRICK
    uint32_t dst_addr_shift = dst_thread_data_valid ? 0 : 0x7fffffff;

    __llvm_amdgcn_buffer_store_f32(*p_src_tmp,
                                   dst_wave_buffer_resource.data,
                                   0,
                                   dst_addr_shift + dst_thread_addr_offset,
                                   false,
                                   false);
#else
    if(dst_thread_data_valid)
    {
        __llvm_amdgcn_buffer_store_f32(
            *p_src_tmp, dst_wave_buffer_resource.data, 0, dst_thread_addr_offset, false, false);
    }
#endif
}

template <>
__device__ void amd_buffer_store<ushort, 4>(const ushort* p_src_thread,
                                            ushort* p_dst_wave,
                                            index_t dst_thread_data_offset,
                                            bool dst_thread_data_valid,
                                            index_t dst_data_range)
{
    BufferResource<ushort> dst_wave_buffer_resource;

    // wavewise base address (64 bit)
    dst_wave_buffer_resource.address[0] = p_dst_wave;
    // wavewise range (32 bit)
    dst_wave_buffer_resource.range[2] = dst_data_range * sizeof(ushort);
    // wavewise setting (32 bit)
    dst_wave_buffer_resource.config[3] = CK_BUFFER_RESOURCE_3RD_DWORD;

    index_t dst_thread_addr_offset = dst_thread_data_offset * sizeof(ushort);

    const float2_t* p_src_tmp = reinterpret_cast<const float2_t*>(p_src_thread);

#if CK_EXPERIMENTAL_USE_BUFFER_STORE_OOB_CHECK_OFFSET_TRICK
    uint32_t dst_addr_shift = dst_thread_data_valid ? 0 : 0x7fffffff;

    __llvm_amdgcn_buffer_store_f32x2(*p_src_tmp,
                                     dst_wave_buffer_resource.data,
                                     0,
                                     dst_addr_shift + dst_thread_addr_offset,
                                     false,
                                     false);
#else
    if(dst_thread_data_valid)
    {
        __llvm_amdgcn_buffer_store_f32x2(
            *p_src_tmp, dst_wave_buffer_resource.data, 0, dst_thread_addr_offset, false, false);
    }
#endif
}

template <>
__device__ void amd_buffer_store<ushort, 8>(const ushort* p_src_thread,
                                            ushort* p_dst_wave,
                                            index_t dst_thread_data_offset,
                                            bool dst_thread_data_valid,
                                            index_t dst_data_range)
{
    BufferResource<ushort> dst_wave_buffer_resource;

    // wavewise base address (64 bit)
    dst_wave_buffer_resource.address[0] = p_dst_wave;
    // wavewise range (32 bit)
    dst_wave_buffer_resource.range[2] = dst_data_range * sizeof(ushort);
    // wavewise setting (32 bit)
    dst_wave_buffer_resource.config[3] = CK_BUFFER_RESOURCE_3RD_DWORD;

    index_t dst_thread_addr_offset = dst_thread_data_offset * sizeof(ushort);

    const float4_t* p_src_tmp = reinterpret_cast<const float4_t*>(p_src_thread);

#if CK_EXPERIMENTAL_USE_BUFFER_STORE_OOB_CHECK_OFFSET_TRICK
    uint32_t dst_addr_shift = dst_thread_data_valid ? 0 : 0x7fffffff;

    __llvm_amdgcn_buffer_store_f32x4(*p_src_tmp,
                                     dst_wave_buffer_resource.data,
                                     0,
                                     dst_addr_shift + dst_thread_addr_offset,
                                     false,
                                     false);
#else
    if(dst_thread_data_valid)
    {
        __llvm_amdgcn_buffer_store_f32x4(
            *p_src_tmp, dst_wave_buffer_resource.data, 0, dst_thread_addr_offset, false, false);
    }
#endif
}

#if CK_USE_AMD_BUFFER_ATOMIC_FADD
template <>
__device__ void amd_buffer_atomic_add<float, 1>(const float* p_src_thread,
                                                float* p_dst_wave,
                                                index_t dst_thread_data_offset,
                                                bool dst_thread_data_valid,
                                                index_t dst_data_range)
{
    BufferResource<float> dst_wave_buffer_resource;

    // wavewise base address (64 bit)
    dst_wave_buffer_resource.address[0] = p_dst_wave;
    // wavewise range (32 bit)
    dst_wave_buffer_resource.range[2] = dst_data_range * sizeof(float);
    // wavewise setting (32 bit)
    dst_wave_buffer_resource.config[3] = CK_BUFFER_RESOURCE_3RD_DWORD;

    index_t dst_thread_addr_offset = dst_thread_data_offset * sizeof(float);

#if CK_EXPERIMENTAL_USE_BUFFER_ATOMIC_OOB_CHECK_OFFSET_TRICK
    uint32_t dst_addr_shift = dst_thread_data_valid ? 0 : 0x7fffffff;

    __llvm_amdgcn_buffer_atomic_add_f32(*p_src_thread,
                                        dst_wave_buffer_resource.data,
                                        0,
                                        dst_addr_shift + dst_thread_addr_offset,
                                        false);
#else
    if(dst_thread_data_valid)
    {
        __llvm_amdgcn_buffer_atomic_add_f32(
            *p_src_thread, dst_wave_buffer_resource.data, 0, dst_thread_addr_offset, false);
    }
#endif
}

template <>
__device__ void amd_buffer_atomic_add<float, 2>(const float* p_src_thread,
                                                float* p_dst_wave,
                                                index_t dst_thread_data_offset,
                                                bool dst_thread_data_valid,
                                                index_t dst_data_range)
{
    BufferResource<float> dst_wave_buffer_resource;

    // wavewise base address (64 bit)
    dst_wave_buffer_resource.address[0] = p_dst_wave;
    // wavewise range (32 bit)
    dst_wave_buffer_resource.range[2] = dst_data_range;
    // wavewise setting (32 bit)
    dst_wave_buffer_resource.config[3] = CK_BUFFER_RESOURCE_3RD_DWORD;

    index_t dst_thread_addr_offset = dst_thread_data_offset * sizeof(float);

#if CK_EXPERIMENTAL_USE_BUFFER_ATOMIC_OOB_CHECK_OFFSET_TRICK
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
#else
    if(dst_thread_data_valid)
    {
        for(index_t i = 0; i < 2; ++i)
        {
            __llvm_amdgcn_buffer_atomic_add_f32(p_src_thread[i],
                                                dst_wave_buffer_resource.data,
                                                0,
                                                dst_thread_addr_offset + i * sizeof(float),
                                                false);
        }
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
    BufferResource<float> dst_wave_buffer_resource;

    // wavewise base address (64 bit)
    dst_wave_buffer_resource.address[0] = p_dst_wave;
    // wavewise range (32 bit)
    dst_wave_buffer_resource.range[2] = dst_data_range * sizeof(float);
    // wavewise setting (32 bit)
    dst_wave_buffer_resource.config[3] = CK_BUFFER_RESOURCE_3RD_DWORD;

    index_t dst_thread_addr_offset = dst_thread_data_offset * sizeof(float);

#if CK_EXPERIMENTAL_USE_BUFFER_ATOMIC_OOB_CHECK_OFFSET_TRICK
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
#else
    if(dst_thread_data_valid)
    {
        for(index_t i = 0; i < 4; ++i)
        {
            __llvm_amdgcn_buffer_atomic_add_f32(p_src_thread[i],
                                                dst_wave_buffer_resource.data,
                                                0,
                                                dst_thread_addr_offset + i * sizeof(float),
                                                false);
        }
    }
#endif
}
#endif // CK_USE_AMD_BUFFER_ATOMIC_FADD

} // namespace ck
#endif
