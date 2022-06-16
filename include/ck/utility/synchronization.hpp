#ifndef CK_SYNCHRONIZATION_AMD_HPP
#define CK_SYNCHRONIZATION_AMD_HPP

#include "config.hpp"

namespace ck {

__device__ void block_sync_lds()
{
#if CK_EXPERIMENTAL_BLOCK_SYNC_LDS_WITHOUT_SYNC_VMEM
    asm volatile("\
    s_waitcnt lgkmcnt(0) \n \
    s_barrier \
    " ::);
#else
    __syncthreads();
#endif
}
__device__ void s_nop()
{
    asm volatile("\
    s_nop 0 \n \
    " ::);
}

__device__ void s_barrier()
{
    asm volatile("\
    s_barrier \
    " ::);
}

} // namespace ck
#endif
