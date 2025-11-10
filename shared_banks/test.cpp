#include <hip/hip_runtime.h>

__global__ void conflict_simulator()
{
  // __shared__ int buf[64];
  // buf[threadIdx.x] = threadIdx.x; // -> 0 conflicts	
  //
  // __shared__ int buf[2 * 64];
  // buf[2 * threadIdx.x] = threadIdx.x; // -> 1 conflict per access (2-way)
  //
  __shared__ int buf[4 * 64];
  buf[4 * threadIdx.x] = threadIdx.x; // -> 3 conflicts per access (4-way)
}

int main()
{
  conflict_simulator<<<1, 64, 0, 0>>>();
  return 0;
}
