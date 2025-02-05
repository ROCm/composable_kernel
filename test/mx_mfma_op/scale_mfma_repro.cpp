#include <hip/hip_ext.h>
#include <hip/hip_runtime.h>

__global__ void kernel()
{
    using dataAB = uint8_t __attribute__((ext_vector_type(32)));
    using dataC  = float __attribute__((ext_vector_type(16)));
    using dataX  = int32_t __attribute__((ext_vector_type(2)));

    dataAB regA(0x38);
    dataAB regB(0x38);
    dataC regC(1.0f);
    // dataC regCin(1.0f);
#if 1
    // dataX xa{127, 127};   // 1.0
    dataX xa(127 & 0xFF); // 1.0
    dataX xb(127 & 0xFF); // 1.0
#else
    dataX xa(0);
    dataX xb(0);
#endif
#if 0
    if(threadIdx.x == 0)
    {
        // xa = 127; // 1.0
        for(size_t i = 0; i < 32; i++)
        {
            regA[i] = 0x38; // 1.0
        }
        for(size_t i = 0; i < 32; i++)
        {
            regB[i] = 0x38; // 1.0
        }
        printf("thread: %u -- xA: %x\n", threadIdx.x, xa[threadIdx.x / 32]);
        printf("thread: %u -- xB: %x\n", threadIdx.x, xb[threadIdx.x / 32]);
    }

    if(threadIdx.x == 32)
    {
        // xa = 126; // 0.5
        for(size_t i = 0; i < 32; i++)
        {
            regA[i] = 0xC0; // -2.0
        }
        for(size_t i = 0; i < 32; i++)
        {
            regB[i] = 0x38; // 1.0
        }
        printf("thread: %u -- xA: %x\n", threadIdx.x, xa[threadIdx.x / 32]);
        printf("thread: %u -- xB: %x\n", threadIdx.x, xb[threadIdx.x / 32]);
    }
#endif

    __syncthreads();
    printf("thread: %u -- regA: %x %x %x %x %x %x %x %x %x %x %x %x %x %x %x %x %x %x %x %x %x %x "
           "%x %x %x %x %x %x %x %x %x %x\n",
           threadIdx.x,
           regA[0],
           regA[1],
           regA[2],
           regA[3],
           regA[4],
           regA[5],
           regA[6],
           regA[7],
           regA[8],
           regA[9],
           regA[10],
           regA[11],
           regA[12],
           regA[13],
           regA[14],
           regA[15],
           regA[16],
           regA[17],
           regA[18],
           regA[19],
           regA[20],
           regA[21],
           regA[22],
           regA[23],
           regA[24],
           regA[25],
           regA[26],
           regA[27],
           regA[28],
           regA[29],
           regA[30],
           regA[31]);

    printf("thread: %u -- regB: %x %x %x %x %x %x %x %x %x %x %x %x %x %x %x %x %x %x %x %x %x %x "
           "%x %x %x %x %x %x %x %x %x %x\n",
           threadIdx.x,
           regB[0],
           regB[1],
           regB[2],
           regB[3],
           regB[4],
           regB[5],
           regB[6],
           regB[7],
           regB[8],
           regB[9],
           regB[10],
           regB[11],
           regB[12],
           regB[13],
           regB[14],
           regB[15],
           regB[16],
           regB[17],
           regB[18],
           regB[19],
           regB[20],
           regB[21],
           regB[22],
           regB[23],
           regB[24],
           regB[25],
           regB[26],
           regB[27],
           regB[28],
           regB[29],
           regB[30],
           regB[31]);

    //__builtin_amdgcn_mfma_ld_scale_b32(xb[threadIdx.x / 32], 0, 0);
    regC = __builtin_amdgcn_mfma_scale_f32_32x32x64_f8f6f4(regA,
                                                           regB,
                                                           regC,
                                                           0, // cbsz
                                                           0, // blgp
                                                           0,
                                                           xa[threadIdx.x / 32],
                                                           0,
                                                           xb[threadIdx.x / 32]);

    __syncthreads();

    printf("thread: %u -- regC: %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f\n",
           threadIdx.x,
           regC[0],
           regC[1],
           regC[2],
           regC[3],
           regC[4],
           regC[5],
           regC[6],
           regC[7],
           regC[8],
           regC[9],
           regC[10],
           regC[11],
           regC[12],
           regC[13],
           regC[14],
           regC[15]);

    // printf("thread: %u -- regCin: %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f\n",
    //        threadIdx.x,
    //        regCin[0],
    //        regCin[1],
    //        regCin[2],
    //        regCin[3],
    //        regCin[4],
    //        regCin[5],
    //        regCin[6],
    //        regCin[7],
    //        regCin[8],
    //        regCin[9],
    //        regCin[10],
    //        regCin[11],
    //        regCin[12],
    //        regCin[13],
    //        regCin[14],
    //        regCin[15]);
}

int main()
{
    kernel<<<1, 64>>>();
    return 0;
}