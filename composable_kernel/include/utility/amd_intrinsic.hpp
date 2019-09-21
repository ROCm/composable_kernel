#ifndef CK_AMD_INTRINSIC_HPP
#define CK_AMD_INTRINSIC_HPP

#include "vector_type.hpp"

namespace ck {

__device__ float __llvm_amdgcn_buffer_load(int32x4_t rsrc,
                                           uint32_t vindex,
                                           uint32_t offset,
                                           bool glc,
                                           bool slc) __asm("llvm.amdgcn.buffer.load");

__device__ vector_type<float, 2>::MemoryType
__llvm_amdgcn_buffer_loadx2(int32x4_t rsrc,
                            uint32_t vindex,
                            uint32_t offset,
                            bool glc,
                            bool slc) __asm("llvm.amdgcn.buffer.load.dwordx2");

__device__ vector_type<float, 4>::MemoryType
__llvm_amdgcn_buffer_loadx4(int32x4_t rsrc,
                            uint32_t vindex,
                            uint32_t offset,
                            bool glc,
                            bool slc) __asm("llvm.amdgcn.buffer.load.dwordx4");

__device__ void __llvm_amdgcn_buffer_store(float vdata,
                                           int32x4_t rsrc,
                                           uint32_t vindex,
                                           uint32_t offset,
                                           bool glc,
                                           bool slc) __asm("llvm.amdgcn.buffer.store");

__device__ void __llvm_amdgcn_buffer_storex2(vector_type<float, 2>::MemoryType vdata,
                                             int32x4_t rsrc,
                                             uint32_t vindex,
                                             uint32_t offset,
                                             bool glc,
                                             bool slc) __asm("llvm.amdgcn.buffer.store.dwordx2");

__device__ void __llvm_amdgcn_buffer_storex4(vector_type<float, 4>::MemoryType vdata,
                                             int32x4_t rsrc,
                                             uint32_t vindex,
                                             uint32_t offset,
                                             bool glc,
                                             bool slc) __asm("llvm.amdgcn.buffer.store.dwordx4");

// buffer_load and buffer_store
template <typename T, index_t VectorSize>
__device__ typename vector_type<T, VectorSize>::MemoryType __buffer_load(
    const T* p_src_block, uint32_t src_thread_data_offset, uint32_t src_const_data_offset);

template <typename T, index_t VectorSize>
__device__ void __buffer_store(const typename vector_type<T, VectorSize>::MemoryType& src,
                               T* p_dst_block,
                               uint32_t dst_thread_data_offset,
                               uint32_t dst_const_data_offset);

template <>
__device__ float __buffer_load<float, 1>(const float* p_src_block,
                                         uint32_t src_thread_data_offset,
                                         uint32_t src_const_data_offset)
{
#if 0
    float dst;

    uint32_t src_thread_addr_offset = src_thread_data_offset * sizeof(float);
    uint32_t src_const_addr_offset  = src_const_data_offset * sizeof(float);

    int32x4_t src_block_setting{0};
    // fill in byte 0 - 1
    *reinterpret_cast<float**>(&src_block_setting) = const_cast<float*>(p_src_block);
    // fill in byte 2
    reinterpret_cast<int*>(&src_block_setting)[2] = -1;
    // fill in byte 3
    reinterpret_cast<int*>(&src_block_setting)[3] = 0x00027000;

    asm volatile("\n \
    buffer_load_dword %0, %1, %2, %3 offen offset:0 \n \
    s_waitcnt 0 \n \
    "
                 : "=v"(dst)
                 : "v"(src_thread_addr_offset), "s"(src_block_setting), "s"(src_const_addr_offset));

    return dst;
#else
    float dst;

    uint32_t src_thread_addr_offset = src_thread_data_offset * sizeof(float);
    uint32_t src_const_addr_offset  = src_const_data_offset * sizeof(float);

    int32x4_t src_block_setting{0};
    // fill in byte 0 - 1
    *reinterpret_cast<float**>(&src_block_setting) = const_cast<float*>(p_src_block);
    // fill in byte 2
    reinterpret_cast<int*>(&src_block_setting)[2] = -1;
    // fill in byte 3
    reinterpret_cast<int*>(&src_block_setting)[3] = 0x00027000;

    dst = __llvm_amdgcn_buffer_load(
        src_block_setting, 0, src_thread_addr_offset + src_const_addr_offset, false, false);

    return dst;
#endif
}

template <>
__device__ vector_type<float, 2>::MemoryType __buffer_load<float, 2>(
    const float* p_src_block, uint32_t src_thread_data_offset, uint32_t src_const_data_offset)
{
#if 0
    vector_type<float, 2>::MemoryType dst;

    uint32_t src_thread_addr_offset = src_thread_data_offset * sizeof(float);
    uint32_t src_const_addr_offset  = src_const_data_offset * sizeof(float);

    int32x4_t src_block_setting{0};
    // fill in byte 0 - 1
    *reinterpret_cast<float**>(&src_block_setting) = const_cast<float*>(p_src_block);
    // fill in byte 2
    reinterpret_cast<int*>(&src_block_setting)[2] = -1;
    // fill in byte 3
    reinterpret_cast<int*>(&src_block_setting)[3] = 0x00027000;

    asm volatile("\n \
    buffer_load_dwordx2 %0, %1, %2, %3 offen offset:0 \n \
    s_waitcnt 0 \n \
    "
                 : "=v"(dst)
                 : "v"(src_thread_addr_offset), "s"(src_block_setting), "s"(src_const_addr_offset));

    return dst;
#else
    vector_type<float, 2>::MemoryType dst;

    uint32_t src_thread_addr_offset = src_thread_data_offset * sizeof(float);
    uint32_t src_const_addr_offset  = src_const_data_offset * sizeof(float);

    int32x4_t src_block_setting{0};
    // fill in byte 0 - 1
    *reinterpret_cast<float**>(&src_block_setting) = const_cast<float*>(p_src_block);
    // fill in byte 2
    reinterpret_cast<int*>(&src_block_setting)[2] = -1;
    // fill in byte 3
    reinterpret_cast<int*>(&src_block_setting)[3] = 0x00027000;

    dst = __llvm_amdgcn_buffer_loadx2(
        src_block_setting, 0, src_thread_addr_offset + src_const_addr_offset, false, false);

    return dst;
#endif
}

template <>
__device__ vector_type<float, 4>::MemoryType __buffer_load<float, 4>(
    const float* p_src_block, uint32_t src_thread_data_offset, uint32_t src_const_data_offset)
{
#if 0
    vector_type<float, 4>::MemoryType dst;

    uint32_t src_thread_addr_offset = src_thread_data_offset * sizeof(float);
    uint32_t src_const_addr_offset  = src_const_data_offset * sizeof(float);

    int32x4_t src_block_setting{0};
    // fill in byte 0 - 1
    *reinterpret_cast<float**>(&src_block_setting) = const_cast<float*>(p_src_block);
    // fill in byte 2
    reinterpret_cast<int*>(&src_block_setting)[2] = -1;
    // fill in byte 3
    reinterpret_cast<int*>(&src_block_setting)[3] = 0x00027000;

    asm volatile("\n \
    buffer_load_dwordx4 %0, %1, %2, %3 offen offset:0 \n \
    s_waitcnt 0 \n \
    "
                 : "=v"(dst)
                 : "v"(src_thread_addr_offset), "s"(src_block_setting), "s"(src_const_addr_offset));

    return dst;
#elif 1
    vector_type<float, 4>::MemoryType dst;

    uint32_t src_thread_addr_offset = src_thread_data_offset * sizeof(float);
    uint32_t src_const_addr_offset  = src_const_data_offset * sizeof(float);

    int32x4_t src_block_setting{0};
    // fill in byte 0 - 1
    *reinterpret_cast<float**>(&src_block_setting) = const_cast<float*>(p_src_block);
    // fill in byte 2
    reinterpret_cast<int*>(&src_block_setting)[2] = -1;
    // fill in byte 3
    reinterpret_cast<int*>(&src_block_setting)[3] = 0x00027000;

    dst = __llvm_amdgcn_buffer_loadx4(
        src_block_setting, 0, src_thread_addr_offset + src_const_addr_offset, false, false);

    return dst;
#endif
}

template <>
__device__ void __buffer_store<float, 1>(const float& src,
                                         float* p_dst_block,
                                         uint32_t dst_thread_data_offset,
                                         uint32_t dst_const_data_offset)
{
#if 0
    uint32_t dst_thread_addr_offset = dst_thread_data_offset * sizeof(float);
    uint32_t dst_const_addr_offset  = dst_const_data_offset * sizeof(float);

    int32x4_t dst_block_setting{0};
    // fill in byte 0 - 1
    *reinterpret_cast<float**>(&dst_block_setting) = p_dst_block;
    // fill in byte 2
    reinterpret_cast<int*>(&dst_block_setting)[2] = -1;
    // fill in byte 3
    reinterpret_cast<int*>(&dst_block_setting)[3] = 0x00027000;

    asm volatile("\n \
    buffer_store_dword %1, %2, %0, %3 offen offset:0 \n \
    "
                 :
                 : "s"(dst_block_setting),
                   "v"(src),
                   "v"(dst_thread_addr_offset),
                   "s"(dst_const_addr_offset));
#else
    uint32_t dst_thread_addr_offset = dst_thread_data_offset * sizeof(float);
    uint32_t dst_const_addr_offset  = dst_const_data_offset * sizeof(float);

    int32x4_t dst_block_setting{0};
    // fill in byte 0 - 1
    *reinterpret_cast<float**>(&dst_block_setting) = p_dst_block;
    // fill in byte 2
    reinterpret_cast<int*>(&dst_block_setting)[2] = -1;
    // fill in byte 3
    reinterpret_cast<int*>(&dst_block_setting)[3] = 0x00027000;

    __llvm_amdgcn_buffer_store(
        src, dst_block_setting, 0, dst_thread_addr_offset + dst_const_addr_offset, false, false);
#endif
}

template <>
__device__ void __buffer_store<float, 2>(const vector_type<float, 2>::MemoryType& src,
                                         float* p_dst_block,
                                         uint32_t dst_thread_data_offset,
                                         uint32_t dst_const_data_offset)
{
#if 0
    uint32_t dst_thread_addr_offset = dst_thread_data_offset * sizeof(float);
    uint32_t dst_const_addr_offset  = dst_const_data_offset * sizeof(float);

    int32x4_t dst_block_setting{0};
    // fill in byte 0 - 1
    *reinterpret_cast<float**>(&dst_block_setting) = p_dst_block;
    // fill in byte 2
    reinterpret_cast<int*>(&dst_block_setting)[2] = -1;
    // fill in byte 3
    reinterpret_cast<int*>(&dst_block_setting)[3] = 0x00027000;

    asm volatile("\n \
    buffer_store_dwordx2 %1, %2, %0, %3 offen offset:0 \n \
    "
                 :
                 : "s"(dst_block_setting),
                   "v"(src),
                   "v"(dst_thread_addr_offset),
                   "s"(dst_const_addr_offset));
#else
    uint32_t dst_thread_addr_offset = dst_thread_data_offset * sizeof(float);
    uint32_t dst_const_addr_offset  = dst_const_data_offset * sizeof(float);

    int32x4_t dst_block_setting{0};
    // fill in byte 0 - 1
    *reinterpret_cast<float**>(&dst_block_setting) = p_dst_block;
    // fill in byte 2
    reinterpret_cast<int*>(&dst_block_setting)[2] = -1;
    // fill in byte 3
    reinterpret_cast<int*>(&dst_block_setting)[3] = 0x00027000;

    __llvm_amdgcn_buffer_storex2(
        src, dst_block_setting, 0, dst_thread_addr_offset + dst_const_addr_offset, false, false);
#endif
}

template <>
__device__ void __buffer_store<float, 4>(const vector_type<float, 4>::MemoryType& src,
                                         float* p_dst_block,
                                         uint32_t dst_thread_data_offset,
                                         uint32_t dst_const_data_offset)
{
#if 0
    uint32_t dst_thread_addr_offset = dst_thread_data_offset * sizeof(float);
    uint32_t dst_const_addr_offset  = dst_const_data_offset * sizeof(float);

    int32x4_t dst_block_setting{0};
    // fill in byte 0 - 1
    *reinterpret_cast<float**>(&dst_block_setting) = p_dst_block;
    // fill in byte 2
    reinterpret_cast<int*>(&dst_block_setting)[2] = -1;
    // fill in byte 3
    reinterpret_cast<int*>(&dst_block_setting)[3] = 0x00027000;

    asm volatile("\n \
    buffer_store_dwordx4 %1, %2, %0, %3 offen offset:0 \n \
    "
                 :
                 : "s"(dst_block_setting),
                   "v"(src),
                   "v"(dst_thread_addr_offset),
                   "s"(dst_const_addr_offset));
#else
    uint32_t dst_thread_addr_offset = dst_thread_data_offset * sizeof(float);
    uint32_t dst_const_addr_offset  = dst_const_data_offset * sizeof(float);

    int32x4_t dst_block_setting{0};
    // fill in byte 0 - 1
    *reinterpret_cast<float**>(&dst_block_setting) = p_dst_block;
    // fill in byte 2
    reinterpret_cast<int*>(&dst_block_setting)[2] = -1;
    // fill in byte 3
    reinterpret_cast<int*>(&dst_block_setting)[3] = 0x00027000;

    __llvm_amdgcn_buffer_storex4(
        src, dst_block_setting, 0, dst_thread_addr_offset + dst_const_addr_offset, false, false);
#endif
}

} // namespace ck
#endif
