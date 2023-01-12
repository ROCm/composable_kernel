#pragma once

namespace ck {

class philox
{
    public:
    __device__ inline philox(unsigned long long seed,
                             unsigned long long subsequence,
                             unsigned long long offset)
        : key(reinterpret_cast<const uint2&>(seed))
    {

        ull2* tmp = reinterpret_cast<ull2*>(&counter);
        tmp->x    = offset / 4;
        tmp->y    = subsequence;
    }

    __device__ inline uint4 operator()()
    {

        uint4 counter_ = counter;
        uint2 key_     = key;
// 7-round philox
#pragma unroll
        for(int i = 0; i < 6; i++)
        {
            counter_ = single_round(counter_, key_);
            key_.x += kPhilox10A;
            key_.y += kPhilox10B;
        }
        uint4 output = single_round(counter_, key_);
        incr();
        /*
        if ((threadIdx.x == 0) && (blockIdx.x == 0) && (blockIdx.y == 0)) {
            printf("Philox counter: %u, %u, %u, %u\n", counter.x, counter.y, counter.z, counter.w);
            printf("Philox output: %u, %u, %u, %u\n", output.x, output.y, output.z, output.w);
        }
        if ((threadIdx.x == 1) && (blockIdx.x == 0) && (blockIdx.y == 0)) {
            printf("Philox counter: %u, %u, %u, %u\n", counter.x, counter.y, counter.z, counter.w);
            printf("Philox output: %u, %u, %u, %u\n", output.x, output.y, output.z, output.w);
        }
        */
        return output;
    }

    __device__ inline uint4 operator()(const unsigned long long subsequence)
    {

        uint4 counter_ = counter;
        ull2* tmp      = reinterpret_cast<ull2*>(&counter_);
        tmp->y         = subsequence;

        uint2 key_ = key;
// 7-round philox
#pragma unroll
        for(int i = 0; i < 6; i++)
        {
            counter_ = single_round(counter_, key_);
            key_.x += kPhilox10A;
            key_.y += kPhilox10B;
        }
        uint4 output = single_round(counter_, key_);
        return output;
    }

    private:
    struct ull2
    {
        uint64_t x;
        uint64_t y;
    };
    uint4 counter;
    const uint2 key;

    __device__ uint4 incr(uint4 ctr)
    {

        uint4 res;
        uint4 tmp;
        // res.x = ctr.x + 1;
        asm volatile("v_mov_b32 %8  1; \n"
                     "v_mov_b32 %9  0; \n"
                     "v_mov_b32 %10 0; \n"
                     "v_mov_b32 %11 0; \n"
                     "v_add_co_u32     %0, %4, %8;  "
                     "v_addc_co_u32    %1, %5, %9;  "
                     "v_addc_co_u32    %2, %6, %10; "
                     "v_addc_u32       %3, %7, %11; "
                     : "=v"(res.x), "=v"(res.y), "=v"(res.z), "=v"(res.w)
                     : "v"(ctr.x),
                       "v"(ctr.y),
                       "v"(ctr.z),
                       "v"(ctr.w),
                       "v"(tmp.x),
                       "v"(tmp.y),
                       "v"(tmp.z),
                       "v"(tmp.w));
        return res;
    }

    __device__ inline void incr() { counter = incr(counter); }

    __device__ uint2 mulhilo32(const unsigned int a, const unsigned int b)
    {
        uint2* res;
        uint2 tmp_res;
        asm("v_mul_hi_u32    %0, %2, %3\n\t"
            "v_mul_lo_u32    %1, %2, %3\n\t"
            : "=v"(tmp_res.x), "=v"(tmp_res.y)
            : "v"(a), "v"(b));
        res = &tmp_res;
        return *res;
    }

    __device__ inline uint4 single_round(const uint4 ctr, const uint2 i_key)
    {

        uint2 res0 = mulhilo32(kPhiloxSA, ctr.x);
        uint2 res1 = mulhilo32(kPhiloxSB, ctr.z);
        uint4 ret  = {res1.y ^ ctr.y ^ i_key.x, res1.x, res0.y ^ ctr.w ^ i_key.y, res0.x};
        return ret;
    }

    static const unsigned long kPhilox10A = 0x9E3779B9;
    static const unsigned long kPhilox10B = 0xBB67AE85;
    static const unsigned long kPhiloxSA  = 0xD2511F53;
    static const unsigned long kPhiloxSB  = 0xCD9E8D57;
};

__device__ void uint4_to_ushort8(const uint4 a, ushort* b)
{
    uint* b_tmp = reinterpret_cast<uint*>(b);
    b_tmp[0]    = a.x;
    b_tmp[1]    = a.y;
    b_tmp[2]    = a.z;
    b_tmp[3]    = a.w;
}

} // namespace ck

/*
__global__ void rand_use(){

    const int tidx_global = blockIdx.x * blockDim.x + threadIdx.x;
    ushort tmp[8];
    ck::philox ph0(0, tidx_global, 0);
    ck::uint4_to_ushort8(ph0(), tmp);
    __syncthreads();
    for(int i = 0; i < 8; i++){
      printf("rand num at %d is %u \n", i, tmp[i]);
    }
}

int main(){

    int blockSize, gridSize;

    blockSize = 4;
    gridSize = 4;

    hipLaunchKernelGGL(rand_use, dim3(gridSize), dim3(blockSize), 0, 0);

    hipDeviceSynchronize();

    return 0;

}
*/
