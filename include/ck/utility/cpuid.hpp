#ifndef CK_CPUID_HPP
#define CK_CPUID_HPP

namespace ck {
namespace cpu {

enum cpuid_vendor
{
    cpuid_vendor_intel = 0,
    cpuid_vendor_amd   = 1,
    cpuid_vendor_other = 2,
};

enum cpuid_cache_type
{
    cpuid_cache_type_null    = 0,
    cpuid_cache_type_dcache  = 1,
    cpuid_cache_type_icache  = 2,
    cpuid_cache_type_unified = 3,
};

struct cpuid_raw
{
    uint32_t eax{0};
    uint32_t ebx{0};
    uint32_t ecx{0};
    uint32_t edx{0};
};

struct cpuid_cache_detail
{
    uint32_t size{0};
    uint32_t type{0};
    uint32_t cache_line_size{0};
    uint32_t associativity{0};
    uint32_t sets{0};
    uint32_t partitions{0};
    uint32_t shared_by_procs{0};  // in HT, usually maybe 2 threads per core, hence for L1/L2,
                                  // usually this maybe 2, unless turn of HT
    uint32_t cores_per_socket{0}; // hardware cores in a physical socket. there maybe multiple
                                  // sockets on the chip. TODO: may not needed?
    uint32_t flags{0};
};

struct cpuid_cache_hierarchy
{
    cpuid_cache_detail l1i;
    cpuid_cache_detail l1d;
    cpuid_cache_detail l2;
    cpuid_cache_detail l3;
    cpuid_cache_detail l4;
};

static inline cpuid_raw cpuid(uint32_t eax, uint32_t ecx)
{
    // some leaf feature require ecx value.
    // for others, ecx actually not used.
    uint32_t ebx, edx;
    asm __volatile__("mov    %0,  %%eax\n"
                     "mov    %2,  %%ecx\n"
                     "cpuid\n"
                     "mov    %%eax, %0\n"
                     "mov    %%ebx, %1\n"
                     "mov    %%ecx, %2\n"
                     "mov    %%edx, %3\n"
                     : "=r"(eax), "=r"(ebx), "=r"(ecx), "=r"(edx)
                     : "0"(eax), "2"(ecx));
    return {eax, ebx, ecx, edx};
}

static inline cpuid_vendor cpuid_query_vendor()
{
    cpuid_raw r = cpuid(0, 0);
    if(r.ebx == 0x756E6547U /*Genu*/ && r.edx == 0x49656E69U /*ineI*/ &&
       r.ecx == 0x6C65746EU /*ntel*/)
    {
        return cpuid_vendor_intel;
    }
    if(r.ebx == 0x68747541U /*Auth*/ && r.edx == 0x69746E65U /*enti*/ &&
       r.ecx == 0x444D4163U /*cAMD*/)
    {
        return cpuid_vendor_amd;
    }
    if(r.ebx == 0x69444D41U /*AMDi*/ && r.edx == 0x74656273U /*sbet*/ &&
       r.ecx == 0x21726574U /*ter */)
    {
        return cpuid_vendor_amd;
    }
    if(r.ebx == 0x20444D41U /*AMD */ && r.edx == 0x45425349U /*ISBE*/ &&
       r.ecx == 0x52455454U /*TTER*/)
    {
        return cpuid_vendor_amd;
    }
    return cpuid_vendor_other;
}

static inline cpuid_cache_hierarchy cpuid_query_cache()
{
    cpuid_cache_hierarchy cache_hierarchy;
    cpuid_vendor vendor    = cpuid_query_vendor();
    uint32_t leaf_cache_id = vendor == cpuid_vendor_amd ? 0x8000001d : 0x4;
    printf("leaf_cache_id:%u, vendor:%d\n", leaf_cache_id, vendor);

    for(uint32_t ecx_idx = 0;; ecx_idx++)
    {
        cpuid_raw r         = cpuid(leaf_cache_id, ecx_idx);
        uint32_t cache_type = r.eax & 0x1f;
        if(cache_type == cpuid_cache_type_null)
            break; // Null, no more cache

        uint32_t cache_level           = (r.eax >> 5) & 0x7;
        uint32_t cache_shared_by_cores = 1 + ((r.eax >> 14) & 0xfff);
        uint32_t cache_lpp_cores       = 1 + ((r.eax >> 26) & 0x3f);

        uint32_t cache_line_size     = 1 + (r.ebx & 0xfff);
        uint32_t cache_partitions    = 1 + ((r.ebx >> 12) & 0x3ff);
        uint32_t cache_associativity = 1 + (r.ebx >> 22);

        uint32_t cache_sets = 1 + r.ecx;

        switch(cache_level)
        {
        case 1:
            if(cache_type == cpuid_cache_type_dcache || cache_type == cpuid_cache_type_unified)
            {
                cache_hierarchy.l1d.size =
                    cache_partitions * cache_sets * cache_associativity * cache_line_size;
                cache_hierarchy.l1d.type             = cache_type;
                cache_hierarchy.l1d.cache_line_size  = cache_line_size;
                cache_hierarchy.l1d.associativity    = cache_associativity;
                cache_hierarchy.l1d.sets             = cache_sets;
                cache_hierarchy.l1d.partitions       = cache_partitions;
                cache_hierarchy.l1d.shared_by_procs  = cache_shared_by_cores;
                cache_hierarchy.l1d.cores_per_socket = cache_lpp_cores;
            }
            else if(cache_type == cpuid_cache_type_icache)
            {
                cache_hierarchy.l1i.size =
                    cache_partitions * cache_sets * cache_associativity * cache_line_size;
                cache_hierarchy.l1i.type             = cache_type;
                cache_hierarchy.l1i.cache_line_size  = cache_line_size;
                cache_hierarchy.l1i.associativity    = cache_associativity;
                cache_hierarchy.l1i.sets             = cache_sets;
                cache_hierarchy.l1i.partitions       = cache_partitions;
                cache_hierarchy.l1i.shared_by_procs  = cache_shared_by_cores;
                cache_hierarchy.l1i.cores_per_socket = cache_lpp_cores;
            }
            break;
        case 2:
            if(cache_type == cpuid_cache_type_dcache || cache_type == cpuid_cache_type_unified)
            {
                cache_hierarchy.l2.size =
                    cache_partitions * cache_sets * cache_associativity * cache_line_size;
                cache_hierarchy.l2.type             = cache_type;
                cache_hierarchy.l2.cache_line_size  = cache_line_size;
                cache_hierarchy.l2.associativity    = cache_associativity;
                cache_hierarchy.l2.sets             = cache_sets;
                cache_hierarchy.l2.partitions       = cache_partitions;
                cache_hierarchy.l2.shared_by_procs  = cache_shared_by_cores;
                cache_hierarchy.l2.cores_per_socket = cache_lpp_cores;
            }
            break;
        case 3:
            if(cache_type == cpuid_cache_type_dcache || cache_type == cpuid_cache_type_unified)
            {
                cache_hierarchy.l3.size =
                    cache_partitions * cache_sets * cache_associativity * cache_line_size;
                cache_hierarchy.l3.type             = cache_type;
                cache_hierarchy.l3.cache_line_size  = cache_line_size;
                cache_hierarchy.l3.associativity    = cache_associativity;
                cache_hierarchy.l3.sets             = cache_sets;
                cache_hierarchy.l3.partitions       = cache_partitions;
                cache_hierarchy.l3.shared_by_procs  = cache_shared_by_cores;
                cache_hierarchy.l3.cores_per_socket = cache_lpp_cores;
            }
            break;
        case 4:
            if(cache_type == cpuid_cache_type_dcache || cache_type == cpuid_cache_type_unified)
            {
                cache_hierarchy.l4.size =
                    cache_partitions * cache_sets * cache_associativity * cache_line_size;
                cache_hierarchy.l4.type             = cache_type;
                cache_hierarchy.l4.cache_line_size  = cache_line_size;
                cache_hierarchy.l4.associativity    = cache_associativity;
                cache_hierarchy.l4.sets             = cache_sets;
                cache_hierarchy.l4.partitions       = cache_partitions;
                cache_hierarchy.l4.shared_by_procs  = cache_shared_by_cores;
                cache_hierarchy.l4.cores_per_socket = cache_lpp_cores;
            }
            break;
        }
    }

    return cache_hierarchy;
}

} // namespace cpu
} // namespace ck
#endif
