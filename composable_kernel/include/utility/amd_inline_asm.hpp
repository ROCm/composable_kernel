#ifndef CK_AMD_INLINE_ASM_HPP
#define CK_AMD_INLINE_ASM_HPP

#include "vector_type.hpp"

namespace ck {

// cast a pointer of LDS to its address
extern "C" __attribute__((address_space(3))) __device__ void* __to_local(void* p);

// global_load and global_store
template <typename T, index_t VectorSize>
__device__ typename vector_type<T, VectorSize>::MemoryType __global_load(
    const T* p_src_block, uint32_t src_thread_data_offset, uint32_t src_const_data_offset);

template <typename T, index_t VectorSize>
__device__ void __global_store(const typename vector_type<T, VectorSize>::MemoryType& src,
                               T* p_dst_block,
                               uint32_t dst_thread_data_offset,
                               uint32_t dst_const_data_offset);

template <>
__device__ float __global_load<float, 1>(const float* p_src_block,
                                         uint32_t src_thread_data_offset,
                                         uint32_t src_const_data_offset)
{
    float dst;

#if 0   // source code
    dst = p_src_block[src_const_data_offset + src_thread_data_offset];
#elif 0 // use VGPR only
    const float* src_thread_addr_offset_u64 =
        p_src_block + src_const_data_offset + src_thread_data_offset;

    asm volatile("\n \
     global_load_dword %0, %1 off offset:0 \n \
     s_waitcnt 0 \n \
     "
                 : "=v"(dst)
                 : "v"(src_thread_addr_offset_u64));
#elif 0 // use VGPR and SGPR, do compute on VALU
    uint64_t src_thread_addr_offset_u64 =
        (src_thread_data_offset + src_const_data_offset) * sizeof(float);

    asm volatile("\n \
     global_load_dword %0, %1, %2, offset:0 \n \
     s_waitcnt 0 \n \
     "
                 : "=v"(dst)
                 : "v"(src_thread_addr_offset_u64), "s"(p_src_block));
#elif 1 // use VGPR and SGPR, do compute on SALU
    uint64_t src_thread_addr_offset_u64 =
        static_cast<uint64_t>(src_thread_data_offset * sizeof(float));

    const float* p_src_block_with_offset = p_src_block + src_const_data_offset;

    asm volatile("\n \
     global_load_dword %0, %1, %2, offset:0 \n \
     s_waitcnt 0 \n \
     "
                 : "=v"(dst)
                 : "v"(src_thread_addr_offset_u64), "s"(p_src_block_with_offset));
#endif

    return dst;
}

template <>
__device__ vector_type<float, 2>::MemoryType __global_load<float, 2>(
    const float* p_src_block, uint32_t src_thread_data_offset, uint32_t src_const_data_offset)
{
    using vector_t = vector_type<float, 2>::MemoryType;

    vector_t dst;

#if 0   // source code
    dst = *reinterpret_cast<const vector_t*>(&p_src_block[src_const_data_offset + src_thread_data_offset]);
#elif 0 // use VGPR only
    const float* src_thread_addr_offset_u64 =
        p_src_block + src_const_data_offset + src_thread_data_offset;

    asm volatile("\n \
     global_load_dwordx2 %0, %1 off offset:0 \n \
     s_waitcnt 0 \n \
     "
                 : "=v"(dst)
                 : "v"(src_thread_addr_offset_u64));
#elif 0 // use VGPR and SGPR, do compute on VALU
    uint64_t src_thread_addr_offset_u64 =
        (src_thread_data_offset + src_const_data_offset) * sizeof(float);

    asm volatile("\n \
     global_load_dwordx2 %0, %1, %2, offset:0 \n \
     s_waitcnt 0 \n \
     "
                 : "=v"(dst)
                 : "v"(src_thread_addr_offset_u64), "s"(p_src_block));
#elif 1 // use VGPR and SGPR, do compute on SALU
    uint64_t src_thread_addr_offset_u64 =
        static_cast<uint64_t>(src_thread_data_offset * sizeof(float));

    const float* p_src_block_with_offset = p_src_block + src_const_data_offset;

    asm volatile("\n \
     global_load_dwordx2 %0, %1, %2, offset:0 \n \
     s_waitcnt 0 \n \
     "
                 : "=v"(dst)
                 : "v"(src_thread_addr_offset_u64), "s"(p_src_block_with_offset));
#endif

    return dst;
}

template <>
__device__ vector_type<float, 4>::MemoryType __global_load<float, 4>(
    const float* p_src_block, uint32_t src_thread_data_offset, uint32_t src_const_data_offset)
{
    using vector_t = vector_type<float, 4>::MemoryType;

    vector_t dst;

#if 0   // source code
    dst = *reinterpret_cast<const vector_t*>(&p_src_block[src_const_data_offset + src_thread_data_offset]);
#elif 0 // use VGPR only
    const float* src_thread_addr_offset_u64 =
        p_src_block + src_const_data_offset + src_thread_data_offset;

    asm volatile("\n \
     global_load_dwordx4 %0, %1 off offset:0 \n \
     s_waitcnt 0 \n \
     "
                 : "=v"(dst)
                 : "v"(src_thread_addr_offset_u64));
#elif 0 // use VGPR and SGPR, do compute on VALU
    uint64_t src_thread_addr_offset_u64 =
        (src_thread_data_offset + src_const_data_offset) * sizeof(float);

    asm volatile("\n \
     global_load_dwordx4 %0, %1, %2, offset:0 \n \
     s_waitcnt 0 \n \
     "
                 : "=v"(dst)
                 : "v"(src_thread_addr_offset_u64), "s"(p_src_block));
#elif 1 // use VGPR and SGPR, do compute on SALU
    uint64_t src_thread_addr_offset_u64 =
        static_cast<uint64_t>(src_thread_data_offset * sizeof(float));

    const float* p_src_block_with_offset = p_src_block + src_const_data_offset;

    asm volatile("\n \
     global_load_dwordx4 %0, %1, %2, offset:0 \n \
     s_waitcnt 0 \n \
     "
                 : "=v"(dst)
                 : "v"(src_thread_addr_offset_u64), "s"(p_src_block_with_offset));
#endif

    return dst;
}

template <>
__device__ void __global_store<float, 1>(const float& src,
                                         float* p_dst_block,
                                         uint32_t dst_thread_data_offset,
                                         uint32_t dst_const_data_offset)
{
#if 0 // compute on VALU
    uint64_t dst_thread_data_offset_u64 = (dst_thread_data_offset + dst_const_data_offset) * sizeof(float);

    asm volatile("\n \
     global_store_dword %0, %1, %2, offset:0 \n \
     "
                 :
                 : "v"(dst_thread_data_offset_u64), "v"(src), "s"(p_dst_block));
#else // compute on SALU
    uint64_t dst_thread_data_offset_u64 = dst_thread_data_offset * sizeof(float);

    float* p_dst_block_with_offset = p_dst_block + dst_const_data_offset;

    asm volatile("\n \
     global_store_dword %0, %1, %2, offset:0 \n \
     "
                 :
                 : "v"(dst_thread_data_offset_u64), "v"(src), "s"(p_dst_block_with_offset));
#endif
}

// __buffer_load and __buffer_store
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
}

template <>
__device__ vector_type<float, 2>::MemoryType __buffer_load<float, 2>(
    const float* p_src_block, uint32_t src_thread_data_offset, uint32_t src_const_data_offset)
{
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
}

template <>
__device__ vector_type<float, 4>::MemoryType __buffer_load<float, 4>(
    const float* p_src_block, uint32_t src_thread_data_offset, uint32_t src_const_data_offset)
{
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
}

template <>
__device__ void __buffer_store<float, 1>(const float& src,
                                         float* p_dst_block,
                                         uint32_t dst_thread_data_offset,
                                         uint32_t dst_const_data_offset)
{
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
}

__device__ void vmcnt(index_t cnt)
{
    if(cnt == 0)
    {
        asm volatile("\n \
                s_waitcnt vmcnt(0) \n \
                " ::);
    }
    else if(cnt == 1)
    {
        asm volatile("\n \
                s_waitcnt vmcnt(1) \n \
                " ::);
    }
    else if(cnt == 2)
    {
        asm volatile("\n \
                s_waitcnt vmcnt(2) \n \
                " ::);
    }
    else if(cnt == 4)
    {
        asm volatile("\n \
                s_waitcnt vmcnt(2) \n \
                " ::);
    }
    else
    {
        assert(false);
    }
}

__device__ void lgkmcnt(index_t cnt)
{
    if(cnt == 0)
    {
        asm volatile("\n \
                s_waitcnt lgkmcnt(0) \n \
                " ::);
    }
    else if(cnt == 1)
    {
        asm volatile("\n \
                s_waitcnt lgkmcnt(1) \n \
                " ::);
    }
    else if(cnt == 2)
    {
        asm volatile("\n \
                s_waitcnt lgkmcnt(2) \n \
                " ::);
    }
    else if(cnt == 3)
    {
        asm volatile("\n \
                s_waitcnt lgkmcnt(3) \n \
                " ::);
    }
    else if(cnt == 4)
    {
        asm volatile("\n \
                s_waitcnt lgkmcnt(4) \n \
                " ::);
    }
    else
    {
        assert(false);
    }
}

__device__ void outerProduct1x4(const float* a, const float* b, float* c)
{
    asm volatile("\n \
            v_mac_f32 %0, %4, %5 \n \
            v_mac_f32 %1, %4, %6 \n \
            v_mac_f32 %2, %4, %7 \n \
            v_mac_f32 %3, %4, %8 \n \
            "
                 : "=v"(c[0]), "=v"(c[1]), "=v"(c[2]), "=v"(c[3])
                 : "v"(a[0]),
                   "v"(b[0]),
                   "v"(b[1]),
                   "v"(b[2]),
                   "v"(b[3]),
                   "0"(c[0]),
                   "1"(c[1]),
                   "2"(c[2]),
                   "3"(c[3]));
}

__device__ void outerProduct1x4(const float& a,
                                const vector_type<float, 4>::MemoryType& b,
                                vector_type<float, 4>::MemoryType& c)
{
    outerProduct1x4(&a, reinterpret_cast<const float*>(&b), reinterpret_cast<float*>(&c));
}

__device__ void outerProduct2x4(const vector_type<float, 2>::MemoryType& a,
                                const vector_type<float, 4>::MemoryType& b,
                                vector_type<float, 4>::MemoryType& c0,
                                vector_type<float, 4>::MemoryType& c1)
{
    outerProduct1x4(a.x, b, c0);
    outerProduct1x4(a.y, b, c1);
}

__device__ void outerProduct4x4(const vector_type<float, 4>::MemoryType& a,
                                const vector_type<float, 4>::MemoryType& b,
                                vector_type<float, 4>::MemoryType& c0,
                                vector_type<float, 4>::MemoryType& c1,
                                vector_type<float, 4>::MemoryType& c2,
                                vector_type<float, 4>::MemoryType& c3)
{
    outerProduct1x4(a.x, b, c0);
    outerProduct1x4(a.y, b, c1);
    outerProduct1x4(a.z, b, c2);
    outerProduct1x4(a.w, b, c3);
}

__device__ void outerProduct8x8(const vector_type<float, 4>::MemoryType* a,
                                const vector_type<float, 4>::MemoryType* b,
                                vector_type<float, 4>::MemoryType* c)
{
    outerProduct4x4(a[0], b[0], c[0], c[2], c[4], c[6]);
    outerProduct4x4(a[0], b[1], c[1], c[3], c[5], c[7]);
    outerProduct4x4(a[1], b[0], c[8], c[10], c[12], c[14]);
    outerProduct4x4(a[1], b[1], c[9], c[11], c[13], c[15]);
}

__device__ void ds_read_b128(vector_type<float, 4>::MemoryType& r, void* lds, index_t offset = 0)
{
    if(offset == 0)
    {
        asm volatile("\n \
                ds_read_b128 %0, %1 offset:0\n \
                "
                     : "=v"(r)
                     : "v"(__to_local(lds)));
    }
    if(offset == 64)
    {
        asm volatile("\n \
                ds_read_b128 %0, %1 offset:64\n \
                "
                     : "=v"(r)
                     : "v"(__to_local(lds)));
    }
    if(offset == 128)
    {
        asm volatile("\n \
                ds_read_b128 %0, %1 offset:128\n \
                "
                     : "=v"(r)
                     : "v"(__to_local(lds)));
    }
    if(offset == 192)
    {
        asm volatile("\n \
                ds_read_b128 %0, %1 offset:192\n \
                "
                     : "=v"(r)
                     : "v"(__to_local(lds)));
    }
    if(offset == 256)
    {
        asm volatile("\n \
                ds_read_b128 %0, %1 offset:256\n \
                "
                     : "=v"(r)
                     : "v"(__to_local(lds)));
    }
    if(offset == 320)
    {
        asm volatile("\n \
                ds_read_b128 %0, %1 offset:320\n \
                "
                     : "=v"(r)
                     : "v"(__to_local(lds)));
    }
    if(offset == 384)
    {
        asm volatile("\n \
                ds_read_b128 %0, %1 offset:384\n \
                "
                     : "=v"(r)
                     : "v"(__to_local(lds)));
    }
    if(offset == 448)
    {
        asm volatile("\n \
                ds_read_b128 %0, %1 offset:448\n \
                "
                     : "=v"(r)
                     : "v"(__to_local(lds)));
    }
    if(offset == 512)
    {
        asm volatile("\n \
                ds_read_b128 %0, %1 offset:512\n \
                "
                     : "=v"(r)
                     : "v"(__to_local(lds)));
    }
    if(offset == 576)
    {
        asm volatile("\n \
                ds_read_b128 %0, %1 offset:576\n \
                "
                     : "=v"(r)
                     : "v"(__to_local(lds)));
    }
    if(offset == 640)
    {
        asm volatile("\n \
                ds_read_b128 %0, %1 offset:640\n \
                "
                     : "=v"(r)
                     : "v"(__to_local(lds)));
    }
    if(offset == 704)
    {
        asm volatile("\n \
                ds_read_b128 %0, %1 offset:704\n \
                "
                     : "=v"(r)
                     : "v"(__to_local(lds)));
    }
    if(offset == 768)
    {
        asm volatile("\n \
                ds_read_b128 %0, %1 offset:768\n \
                "
                     : "=v"(r)
                     : "v"(__to_local(lds)));
    }
    if(offset == 832)
    {
        asm volatile("\n \
                ds_read_b128 %0, %1 offset:832\n \
                "
                     : "=v"(r)
                     : "v"(__to_local(lds)));
    }
    if(offset == 896)
    {
        asm volatile("\n \
                ds_read_b128 %0, %1 offset:896\n \
                "
                     : "=v"(r)
                     : "v"(__to_local(lds)));
    }
    if(offset == 960)
    {
        asm volatile("\n \
                ds_read_b128 %0, %1 offset:960\n \
                "
                     : "=v"(r)
                     : "v"(__to_local(lds)));
    }
    if(offset == 1024)
    {
        asm volatile("\n \
                ds_read_b128 %0, %1 offset:1024\n \
                "
                     : "=v"(r)
                     : "v"(__to_local(lds)));
    }
    if(offset == 1088)
    {
        asm volatile("\n \
                ds_read_b128 %0, %1 offset:1088\n \
                "
                     : "=v"(r)
                     : "v"(__to_local(lds)));
    }
    if(offset == 1152)
    {
        asm volatile("\n \
                ds_read_b128 %0, %1 offset:1152\n \
                "
                     : "=v"(r)
                     : "v"(__to_local(lds)));
    }
    if(offset == 1216)
    {
        asm volatile("\n \
                ds_read_b128 %0, %1 offset:1216\n \
                "
                     : "=v"(r)
                     : "v"(__to_local(lds)));
    }
    if(offset == 1280)
    {
        asm volatile("\n \
                ds_read_b128 %0, %1 offset:1280\n \
                "
                     : "=v"(r)
                     : "v"(__to_local(lds)));
    }
    if(offset == 1344)
    {
        asm volatile("\n \
                ds_read_b128 %0, %1 offset:1344\n \
                "
                     : "=v"(r)
                     : "v"(__to_local(lds)));
    }
    if(offset == 1408)
    {
        asm volatile("\n \
                ds_read_b128 %0, %1 offset:1408\n \
                "
                     : "=v"(r)
                     : "v"(__to_local(lds)));
    }
    if(offset == 1472)
    {
        asm volatile("\n \
                ds_read_b128 %0, %1 offset:1472\n \
                "
                     : "=v"(r)
                     : "v"(__to_local(lds)));
    }
    if(offset == 1536)
    {
        asm volatile("\n \
                ds_read_b128 %0, %1 offset:1536\n \
                "
                     : "=v"(r)
                     : "v"(__to_local(lds)));
    }
    if(offset == 1600)
    {
        asm volatile("\n \
                ds_read_b128 %0, %1 offset:1600\n \
                "
                     : "=v"(r)
                     : "v"(__to_local(lds)));
    }
    if(offset == 1664)
    {
        asm volatile("\n \
                ds_read_b128 %0, %1 offset:1664\n \
                "
                     : "=v"(r)
                     : "v"(__to_local(lds)));
    }
    if(offset == 1728)
    {
        asm volatile("\n \
                ds_read_b128 %0, %1 offset:1728\n \
                "
                     : "=v"(r)
                     : "v"(__to_local(lds)));
    }
    if(offset == 1792)
    {
        asm volatile("\n \
                ds_read_b128 %0, %1 offset:1792\n \
                "
                     : "=v"(r)
                     : "v"(__to_local(lds)));
    }
    if(offset == 1856)
    {
        asm volatile("\n \
                ds_read_b128 %0, %1 offset:1856\n \
                "
                     : "=v"(r)
                     : "v"(__to_local(lds)));
    }
    if(offset == 1920)
    {
        asm volatile("\n \
                ds_read_b128 %0, %1 offset:1920\n \
                "
                     : "=v"(r)
                     : "v"(__to_local(lds)));
    }
    if(offset == 1984)
    {
        asm volatile("\n \
                ds_read_b128 %0, %1 offset:1984\n \
                "
                     : "=v"(r)
                     : "v"(__to_local(lds)));
    }
    if(offset == 2048)
    {
        asm volatile("\n \
                ds_read_b128 %0, %1 offset:2048\n \
                "
                     : "=v"(r)
                     : "v"(__to_local(lds)));
    }
    if(offset == 2112)
    {
        asm volatile("\n \
                ds_read_b128 %0, %1 offset:2112\n \
                "
                     : "=v"(r)
                     : "v"(__to_local(lds)));
    }
    if(offset == 2176)
    {
        asm volatile("\n \
                ds_read_b128 %0, %1 offset:2176\n \
                "
                     : "=v"(r)
                     : "v"(__to_local(lds)));
    }
    if(offset == 2240)
    {
        asm volatile("\n \
                ds_read_b128 %0, %1 offset:2240\n \
                "
                     : "=v"(r)
                     : "v"(__to_local(lds)));
    }
    if(offset == 2304)
    {
        asm volatile("\n \
                ds_read_b128 %0, %1 offset:2304\n \
                "
                     : "=v"(r)
                     : "v"(__to_local(lds)));
    }
    if(offset == 2368)
    {
        asm volatile("\n \
                ds_read_b128 %0, %1 offset:2368\n \
                "
                     : "=v"(r)
                     : "v"(__to_local(lds)));
    }
    if(offset == 2432)
    {
        asm volatile("\n \
                ds_read_b128 %0, %1 offset:2432\n \
                "
                     : "=v"(r)
                     : "v"(__to_local(lds)));
    }
    if(offset == 2496)
    {
        asm volatile("\n \
                ds_read_b128 %0, %1 offset:2496\n \
                "
                     : "=v"(r)
                     : "v"(__to_local(lds)));
    }
    if(offset == 2560)
    {
        asm volatile("\n \
                ds_read_b128 %0, %1 offset:2560\n \
                "
                     : "=v"(r)
                     : "v"(__to_local(lds)));
    }
    if(offset == 2624)
    {
        asm volatile("\n \
                ds_read_b128 %0, %1 offset:2624\n \
                "
                     : "=v"(r)
                     : "v"(__to_local(lds)));
    }
    if(offset == 2688)
    {
        asm volatile("\n \
                ds_read_b128 %0, %1 offset:2688\n \
                "
                     : "=v"(r)
                     : "v"(__to_local(lds)));
    }
    if(offset == 2752)
    {
        asm volatile("\n \
                ds_read_b128 %0, %1 offset:2752\n \
                "
                     : "=v"(r)
                     : "v"(__to_local(lds)));
    }
    if(offset == 2816)
    {
        asm volatile("\n \
                ds_read_b128 %0, %1 offset:2816\n \
                "
                     : "=v"(r)
                     : "v"(__to_local(lds)));
    }
    if(offset == 2880)
    {
        asm volatile("\n \
                ds_read_b128 %0, %1 offset:2880\n \
                "
                     : "=v"(r)
                     : "v"(__to_local(lds)));
    }
    if(offset == 2944)
    {
        asm volatile("\n \
                ds_read_b128 %0, %1 offset:2944\n \
                "
                     : "=v"(r)
                     : "v"(__to_local(lds)));
    }
    if(offset == 3008)
    {
        asm volatile("\n \
                ds_read_b128 %0, %1 offset:3008\n \
                "
                     : "=v"(r)
                     : "v"(__to_local(lds)));
    }
    if(offset == 3072)
    {
        asm volatile("\n \
                ds_read_b128 %0, %1 offset:3072\n \
                "
                     : "=v"(r)
                     : "v"(__to_local(lds)));
    }
    if(offset == 3136)
    {
        asm volatile("\n \
                ds_read_b128 %0, %1 offset:3136\n \
                "
                     : "=v"(r)
                     : "v"(__to_local(lds)));
    }
    if(offset == 3200)
    {
        asm volatile("\n \
                ds_read_b128 %0, %1 offset:3200\n \
                "
                     : "=v"(r)
                     : "v"(__to_local(lds)));
    }
    if(offset == 3264)
    {
        asm volatile("\n \
                ds_read_b128 %0, %1 offset:3264\n \
                "
                     : "=v"(r)
                     : "v"(__to_local(lds)));
    }
    if(offset == 3328)
    {
        asm volatile("\n \
                ds_read_b128 %0, %1 offset:3328\n \
                "
                     : "=v"(r)
                     : "v"(__to_local(lds)));
    }
    if(offset == 3392)
    {
        asm volatile("\n \
                ds_read_b128 %0, %1 offset:3392\n \
                "
                     : "=v"(r)
                     : "v"(__to_local(lds)));
    }
    if(offset == 3456)
    {
        asm volatile("\n \
                ds_read_b128 %0, %1 offset:3456\n \
                "
                     : "=v"(r)
                     : "v"(__to_local(lds)));
    }
    if(offset == 3520)
    {
        asm volatile("\n \
                ds_read_b128 %0, %1 offset:3520\n \
                "
                     : "=v"(r)
                     : "v"(__to_local(lds)));
    }
    if(offset == 3584)
    {
        asm volatile("\n \
                ds_read_b128 %0, %1 offset:3584\n \
                "
                     : "=v"(r)
                     : "v"(__to_local(lds)));
    }
    if(offset == 3648)
    {
        asm volatile("\n \
                ds_read_b128 %0, %1 offset:3648\n \
                "
                     : "=v"(r)
                     : "v"(__to_local(lds)));
    }
    if(offset == 3712)
    {
        asm volatile("\n \
                ds_read_b128 %0, %1 offset:3712\n \
                "
                     : "=v"(r)
                     : "v"(__to_local(lds)));
    }
    if(offset == 3776)
    {
        asm volatile("\n \
                ds_read_b128 %0, %1 offset:3776\n \
                "
                     : "=v"(r)
                     : "v"(__to_local(lds)));
    }
    if(offset == 3840)
    {
        asm volatile("\n \
                ds_read_b128 %0, %1 offset:3840\n \
                "
                     : "=v"(r)
                     : "v"(__to_local(lds)));
    }
    if(offset == 3904)
    {
        asm volatile("\n \
                ds_read_b128 %0, %1 offset:3904\n \
                "
                     : "=v"(r)
                     : "v"(__to_local(lds)));
    }
    if(offset == 3968)
    {
        asm volatile("\n \
                ds_read_b128 %0, %1 offset:3968\n \
                "
                     : "=v"(r)
                     : "v"(__to_local(lds)));
    }
    if(offset == 4032)
    {
        asm volatile("\n \
                ds_read_b128 %0, %1 offset:4032\n \
                "
                     : "=v"(r)
                     : "v"(__to_local(lds)));
    }
    if(offset == 4096)
    {
        asm volatile("\n \
                ds_read_b128 %0, %1 offset:4096\n \
                "
                     : "=v"(r)
                     : "v"(__to_local(lds)));
    }
}

__device__ void
ds_write_b128(const vector_type<float, 4>::MemoryType& r, void* lds, index_t offset = 0)
{
    if(offset == 0)
    {
        asm volatile("\n \
            ds_write_b128 %0, %1 \n \
            "
                     :
                     : "v"(__to_local(lds)), "v"(r));
    }
    else
    {
        assert(false);
    }
}

} // namespace ck
#endif
