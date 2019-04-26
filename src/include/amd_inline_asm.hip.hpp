#pragma once
#include "common.hip.hpp"

#define NO_VM_WAIT 0
#define NO_LGKM_WAIT 0
#define NO_DS_READ 0
#define NO_DS_WRITE 0
#define NO_GLB_READ 0

// cast a pointer of LDS to its address
extern "C" __attribute__((address_space(3))) void* __to_local(void* p)[[hc]];

__device__ void vmcnt(index_t cnt)
{
#if !NO_VM_WAIT
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
#endif
}

__device__ void lgkmcnt(index_t cnt)
{
#if !NO_LGKM_WAIT
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
#endif
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
#if 0
    asm volatile(
            "\n \
            v_mac_f32 %0, %4, %5 \n \
            v_mac_f32 %1, %4, %6 \n \
            v_mac_f32 %2, %4, %7 \n \
            v_mac_f32 %3, %4, %8 \n \
            "
            :
            :"v"(c.x),"v"(c.y),"v"(c.z),"v"(c.w), \
            "v"(a.x),"v"(b.x),"v"(b.y),"v"(b.z),"v"(b.w)
            );
#else
    outerProduct1x4(&a, (float*)&b, (float*)&c);
#endif
}

__device__ void outerProduct4x4(const vector_type<float, 4>::MemoryType& a,
                                const vector_type<float, 4>::MemoryType& b,
                                vector_type<float, 4>::MemoryType& c0,
                                vector_type<float, 4>::MemoryType& c1,
                                vector_type<float, 4>::MemoryType& c2,
                                vector_type<float, 4>::MemoryType& c3)
{
#if 0
    asm volatile(
            "\n \
            v_mac_f32 %0, %4, %5 \n \
            v_mac_f32 %1, %4, %6 \n \
            v_mac_f32 %2, %4, %7 \n \
            v_mac_f32 %3, %4, %8 \n \
            "
            :
            :"v"(c0.x),"v"(c0.y),"v"(c0.z),"v"(c0.w), \
            "v"(a.x),"v"(b.x),"v"(b.y),"v"(b.z),"v"(b.w)
            );
    asm volatile(
            "\n \
            v_mac_f32 %0, %4, %5 \n \
            v_mac_f32 %1, %4, %6 \n \
            v_mac_f32 %2, %4, %7 \n \
            v_mac_f32 %3, %4, %8 \n \
            "
            :
            :"v"(c1.x),"v"(c1.y),"v"(c1.z),"v"(c1.w), \
            "v"(a.y),"v"(b.x),"v"(b.y),"v"(b.z),"v"(b.w)
            );
    asm volatile(
            "\n \
            v_mac_f32 %0, %4, %5 \n \
            v_mac_f32 %1, %4, %6 \n \
            v_mac_f32 %2, %4, %7 \n \
            v_mac_f32 %3, %4, %8 \n \
            "
            :
            :"v"(c2.x),"v"(c2.y),"v"(c2.z),"v"(c2.w), \
            "v"(a.z),"v"(b.x),"v"(b.y),"v"(b.z),"v"(b.w)
            );
    asm volatile(
            "\n \
            v_mac_f32 %0, %4, %5 \n \
            v_mac_f32 %1, %4, %6 \n \
            v_mac_f32 %2, %4, %7 \n \
            v_mac_f32 %3, %4, %8 \n \
            "
            :
            :"v"(c3.x),"v"(c3.y),"v"(c3.z),"v"(c3.w), \
            "v"(a.w),"v"(b.x),"v"(b.y),"v"(b.z),"v"(b.w)
            );
#else
    outerProduct1x4(a.x, b, c0);
    outerProduct1x4(a.y, b, c1);
    outerProduct1x4(a.z, b, c2);
    outerProduct1x4(a.w, b, c3);
#endif
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
#if !NO_DS_READ
    if(offset == 0)
    {
        asm volatile("\n \
                ds_read_b128 %0, %1 \n \
                "
                     : "=v"(r)
                     : "v"(__to_local(lds)));
    }
    else if(offset == 128)
    {
        asm volatile("\n \
                ds_read_b128 %0, %1 offset:128 \n \
                "
                     : "=v"(r)
                     : "v"(__to_local(lds)));
    }
    else if(offset == 256)
    {
        asm volatile("\n \
                ds_read_b128 %0, %1 offset:256 \n \
                "
                     : "=v"(r)
                     : "v"(__to_local(lds)));
    }
    else if(offset == 384)
    {
        asm volatile("\n \
                ds_read_b128 %0, %1 offset:384 \n \
                "
                     : "=v"(r)
                     : "v"(__to_local(lds)));
    }
    else if(offset == 512)
    {
        asm volatile("\n \
                ds_read_b128 %0, %1 offset:512 \n \
                "
                     : "=v"(r)
                     : "v"(__to_local(lds)));
    }
    else if(offset == 640)
    {
        asm volatile("\n \
                ds_read_b128 %0, %1 offset:640 \n \
                "
                     : "=v"(r)
                     : "v"(__to_local(lds)));
    }
    else if(offset == 768)
    {
        asm volatile("\n \
                ds_read_b128 %0, %1 offset:768 \n \
                "
                     : "=v"(r)
                     : "v"(__to_local(lds)));
    }
    else if(offset == 896)
    {
        asm volatile("\n \
                ds_read_b128 %0, %1 offset:896 \n \
                "
                     : "=v"(r)
                     : "v"(__to_local(lds)));
    }
    else if(offset == 1024)
    {
        asm volatile("\n \
                ds_read_b128 %0, %1 offset:1024 \n \
                "
                     : "=v"(r)
                     : "v"(__to_local(lds)));
    }
    else if(offset == 1152)
    {
        asm volatile("\n \
                ds_read_b128 %0, %1 offset:1152 \n \
                "
                     : "=v"(r)
                     : "v"(__to_local(lds)));
    }
    else if(offset == 1280)
    {
        asm volatile("\n \
                ds_read_b128 %0, %1 offset:1280 \n \
                "
                     : "=v"(r)
                     : "v"(__to_local(lds)));
    }
    else if(offset == 1408)
    {
        asm volatile("\n \
                ds_read_b128 %0, %1 offset:1408 \n \
                "
                     : "=v"(r)
                     : "v"(__to_local(lds)));
    }
    else if(offset == 1536)
    {
        asm volatile("\n \
                ds_read_b128 %0, %1 offset:1536 \n \
                "
                     : "=v"(r)
                     : "v"(__to_local(lds)));
    }
    else if(offset == 1664)
    {
        asm volatile("\n \
                ds_read_b128 %0, %1 offset:1664 \n \
                "
                     : "=v"(r)
                     : "v"(__to_local(lds)));
    }
    else if(offset == 1792)
    {
        asm volatile("\n \
                ds_read_b128 %0, %1 offset:1792 \n \
                "
                     : "=v"(r)
                     : "v"(__to_local(lds)));
    }
    else if(offset == 1920)
    {
        asm volatile("\n \
                ds_read_b128 %0, %1 offset:1920 \n \
                "
                     : "=v"(r)
                     : "v"(__to_local(lds)));
    }
    else if(offset == 2048)
    {
        asm volatile("\n \
                ds_read_b128 %0, %1 offset:2048 \n \
                "
                     : "=v"(r)
                     : "v"(__to_local(lds)));
    }
    else if(offset == 2176)
    {
        asm volatile("\n \
                ds_read_b128 %0, %1 offset:2176 \n \
                "
                     : "=v"(r)
                     : "v"(__to_local(lds)));
    }
    else if(offset == 2304)
    {
        asm volatile("\n \
                ds_read_b128 %0, %1 offset:2304 \n \
                "
                     : "=v"(r)
                     : "v"(__to_local(lds)));
    }
    else if(offset == 2432)
    {
        asm volatile("\n \
                ds_read_b128 %0, %1 offset:2432 \n \
                "
                     : "=v"(r)
                     : "v"(__to_local(lds)));
    }
    else if(offset == 2560)
    {
        asm volatile("\n \
                ds_read_b128 %0, %1 offset:2560 \n \
                "
                     : "=v"(r)
                     : "v"(__to_local(lds)));
    }
    else if(offset == 2688)
    {
        asm volatile("\n \
                ds_read_b128 %0, %1 offset:2688 \n \
                "
                     : "=v"(r)
                     : "v"(__to_local(lds)));
    }
    else if(offset == 2816)
    {
        asm volatile("\n \
                ds_read_b128 %0, %1 offset:2816 \n \
                "
                     : "=v"(r)
                     : "v"(__to_local(lds)));
    }
    else if(offset == 2944)
    {
        asm volatile("\n \
                ds_read_b128 %0, %1 offset:2944 \n \
                "
                     : "=v"(r)
                     : "v"(__to_local(lds)));
    }
    else if(offset == 3072)
    {
        asm volatile("\n \
                ds_read_b128 %0, %1 offset:3072 \n \
                "
                     : "=v"(r)
                     : "v"(__to_local(lds)));
    }
    else if(offset == 3200)
    {
        asm volatile("\n \
                ds_read_b128 %0, %1 offset:3200 \n \
                "
                     : "=v"(r)
                     : "v"(__to_local(lds)));
    }
    else if(offset == 3328)
    {
        asm volatile("\n \
                ds_read_b128 %0, %1 offset:3328 \n \
                "
                     : "=v"(r)
                     : "v"(__to_local(lds)));
    }
    else if(offset == 3456)
    {
        asm volatile("\n \
                ds_read_b128 %0, %1 offset:3456 \n \
                "
                     : "=v"(r)
                     : "v"(__to_local(lds)));
    }
    else if(offset == 3584)
    {
        asm volatile("\n \
                ds_read_b128 %0, %1 offset:3584 \n \
                "
                     : "=v"(r)
                     : "v"(__to_local(lds)));
    }
    else if(offset == 3712)
    {
        asm volatile("\n \
                ds_read_b128 %0, %1 offset:3712 \n \
                "
                     : "=v"(r)
                     : "v"(__to_local(lds)));
    }
    else if(offset == 3840)
    {
        asm volatile("\n \
                ds_read_b128 %0, %1 offset:3840 \n \
                "
                     : "=v"(r)
                     : "v"(__to_local(lds)));
    }
    else if(offset == 3968)
    {
        asm volatile("\n \
                ds_read_b128 %0, %1 offset:3968 \n \
                "
                     : "=v"(r)
                     : "v"(__to_local(lds)));
    }
    else if(offset == 4096)
    {
        asm volatile("\n \
                ds_read_b128 %0, %1 offset:4096 \n \
                "
                     : "=v"(r)
                     : "v"(__to_local(lds)));
    }
    else if(offset == 4352)
    {
        asm volatile("\n \
                ds_read_b128 %0, %1 offset:4352 \n \
                "
                     : "=v"(r)
                     : "v"(__to_local(lds)));
    }
    else
    {
        assert(false);
    }
#endif
}

__device__ void global_load(vector_type<float, 4>::MemoryType& r,
                            const vector_type<float, 4>::MemoryType* ptr,
                            index_t offset = 0)
{
#if !NO_GLB_READ
    if(offset == 0)
    {
        asm volatile("\n \
                global_load_dwordx4 %0, %1, off \n \
                "
                     : "=v"(r)
                     : "v"(ptr));
    }
    else
    {
        assert(false);
    }
#endif
}

__device__ void
ds_write_b128(const vector_type<float, 4>::MemoryType& r, void* lds, index_t offset = 0)
{
#if !NO_DS_WRITE
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
#endif
}
