#include <hip/hip_ext.h>
#include <hip/hip_runtime.h>

template <typename Y,
          typename X,
          typename std::enable_if<sizeof(X) == sizeof(Y), bool>::type = false>
__host__ __device__ constexpr Y bit_cast(const X& x)
{
    static_assert(__has_builtin(__builtin_bit_cast), "");
    static_assert(sizeof(X) == sizeof(Y), "Do not support cast between different size of type");

    return __builtin_bit_cast(Y, x);
}

__global__ void kernel_a_scale_mapping()
{
    using dataAB = uint8_t __attribute__((ext_vector_type(32)));
    using dataC  = float __attribute__((ext_vector_type(16)));
    using dataX  = int32_t __attribute__((ext_vector_type(2)));

    dataAB regA(0);
    dataAB regB(0);
    dataC regC(0.0f);
    // dataC regCin(1.0f);
#if 0
    dataX xa{0x3F800000, 0x3F800000}; 
    dataX xb(0x3F800000);
#elif 0
    dataX xa{0x3F000000, 0x3F000000};
    dataX xb(0x3F800000);
#elif 0
    dataX xa{0x3F800000, 0x3F000000};
    dataX xb(0x3F800000);
#elif 0
    dataX xa{0x7F, 0x7E}; // expect 64 at c(0,0)
    dataX xb(0x3F800000);
// dataX xb(0x7F);
#elif 1
    dataX xa{bit_cast<int32_t>(threadIdx.x) + 0x7F,
             bit_cast<int32_t>(threadIdx.x) + 0x7F}; // 127{2^0}, 127+1{2^1},...,127+63{2^63}
    dataX xb(0x3F800000);
#else
    dataX xa(0);
    dataX xb(0);
#endif

    // fill first column of B with 1.0
    if(threadIdx.x == 0 || threadIdx.x == 32)
    {
        for(size_t i = 0; i < 32; i++)
        {
            regB[i] = 0x38; // 1.0
        }
    }

    // verify scale mapping for each row
    for(int rowId = 0; rowId < 32; rowId++)
    {
        for(int testId = 0; testId < 64; testId++)
        {
            if(threadIdx.x == 0 && false)
            {
                printf("testId: %d\n", testId);
            }
            regA = dataAB(0);
            regC = dataC(0.0f);

            if(threadIdx.x == 0 + rowId && testId < 32) // row 0
            {
                // set a(0,testId) = 1.0
                regA[testId] = 0x38; // 1.0
            }
            else if(threadIdx.x == 32 + rowId && 32 <= testId) // row 0
            {
                // set a(0,testId) = 1.0
                regA[testId % 32] = 0x38; // 1.0
            }

            __syncthreads();
#if 0
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
#endif
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
            if(threadIdx.x % 32 == 0 && false) // row 0
            {
                printf(
                    "thread: %u -- xA: %x\n", threadIdx.x, bit_cast<int32_t>(xa[threadIdx.x / 32]));
                printf(
                    "thread: %u -- xB: %x\n", threadIdx.x, bit_cast<int32_t>(xb[threadIdx.x / 32]));
            }
            // Size              |   BLOCK_N  |   BLOCK_N   |
            // N                 | 0  ...  31 |  0  ...  31 |
            // Thread Id         | 0  ...  31 | 32  ...  63 | Vector
            // Register Element   ------------ -------------  Element
            // Reg0              |     M0     |     M4      | v[0]
            // Reg1              |     M1     |     M5      | v[1]
            // Reg2              |     M2     |     M6      | v[2]
            // Reg3              |     M3     |     M7      | v[3]
            //                    ____________ _____________
            // Reg4              |     M8     |     M12     | v[4]
            // Reg5              |     M9     |     M13     | v[5]
            // Reg6              |     M10    |     M14     | v[6]
            // Reg7              |     M11    |     M15     | v[7]
            //                    ____________ _____________
            // Reg8              |     M16    |     M20     | v[8]
            // Reg9              |     M17    |     M21     | v[9]
            // Reg10             |     M18    |     M22     | v[10]
            // Reg11             |     M19    |     M23     | v[11]
            //                    ____________ _____________
            // Reg12             |     M24    |     M28     | v[12]
            // Reg13             |     M25    |     M29     | v[13]
            // Reg14             |     M26    |     M30     | v[14]
            // Reg15             |     M27    |     M31     | v[15]
            if(threadIdx.x == 0 || threadIdx.x == 32)
            {
                auto majChunkId = rowId / 8; //{0,1,2,3}
                auto minChunkId = rowId % 8; //{0,1,2,3,4,5,6,7}

                if(minChunkId < 4 && threadIdx.x == 0)
                {
                    printf("a(%d,%d) is scaled from thread %f\n",
                           rowId,
                           testId,
                           log2f(regC[4 * majChunkId + minChunkId]));

                    // printf("ax(%.0f)*a(%d,%d) ",
                    //        log2f(regC[4 * majChunkId + minChunkId]),
                    //        rowId,
                    //        testId);
                }
                else if(minChunkId >= 4 && threadIdx.x == 32)
                {
                    // printf("ax(%.0f)*a(%d,%d) ",
                    //        log2f(regC[4 * majChunkId + minChunkId - 4]),
                    //        rowId,
                    //        testId);

                    printf("a(%d,%d) is scaled from thread %f\n",
                           rowId,
                           testId,
                           log2f(regC[4 * majChunkId + minChunkId - 4]));
                }
            }
#if 0
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
#endif
        }
        if(threadIdx.x == 32)
        {
            printf("\n");
        }
    }
}

__global__ void kernel_b_scale_mapping()
{
    using dataAB = uint8_t __attribute__((ext_vector_type(32)));
    using dataC  = float __attribute__((ext_vector_type(16)));
    using dataX  = int32_t __attribute__((ext_vector_type(2)));

    dataAB regA(0);
    dataAB regB(0);
    dataC regC(0.0f);

    dataX xa(0x3F800000);
    dataX xb{bit_cast<int32_t>(threadIdx.x) + 0x7F,
             bit_cast<int32_t>(threadIdx.x) + 0x7F}; // 127{2^0}, 127+1{2^1},...,127+63{2^63}

    // fill first row of A with 1.0
    if(threadIdx.x == 0 || threadIdx.x == 32)
    {
        for(size_t i = 0; i < 32; i++)
        {
            regA[i] = 0x38; // 1.0
        }
    }

    // verify scale mapping for each row
    for(int colId = 0; colId < 32; colId++)
    {
        for(int testId = 0; testId < 64; testId++)
        {
            if(threadIdx.x == 0 && false)
            {
                printf("testId: %d\n", testId);
            }
            regB = dataAB(0);
            regC = dataC(0.0f);

            if(threadIdx.x == 0 + colId && testId < 32) // first half
            {
                // set b(testId, colId) = 1.0
                regB[testId] = 0x38; // 1.0
            }
            else if(threadIdx.x == 32 + colId && 32 <= testId) // upper 32 entries
            {
                // set b(testId, colId) = 1.0
                regB[testId % 32] = 0x38; // 1.0
            }

            __syncthreads();

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

            if(threadIdx.x % 32 == 0 && false) // row 0
            {
                printf(
                    "thread: %u -- xA: %x\n", threadIdx.x, bit_cast<int32_t>(xa[threadIdx.x / 32]));
                printf(
                    "thread: %u -- xB: %x\n", threadIdx.x, bit_cast<int32_t>(xb[threadIdx.x / 32]));
            }
            // Size              |   BLOCK_N  |   BLOCK_N   |
            // N                 | 0  ...  31 |  0  ...  31 |
            // Thread Id         | 0  ...  31 | 32  ...  63 | Vector
            // Register Element   ------------ -------------  Element
            // Reg0              |     M0     |     M4      | v[0]
            // Reg1              |     M1     |     M5      | v[1]
            // Reg2              |     M2     |     M6      | v[2]
            // Reg3              |     M3     |     M7      | v[3]
            //                    ____________ _____________
            // Reg4              |     M8     |     M12     | v[4]
            // Reg5              |     M9     |     M13     | v[5]
            // Reg6              |     M10    |     M14     | v[6]
            // Reg7              |     M11    |     M15     | v[7]
            //                    ____________ _____________
            // Reg8              |     M16    |     M20     | v[8]
            // Reg9              |     M17    |     M21     | v[9]
            // Reg10             |     M18    |     M22     | v[10]
            // Reg11             |     M19    |     M23     | v[11]
            //                    ____________ _____________
            // Reg12             |     M24    |     M28     | v[12]
            // Reg13             |     M25    |     M29     | v[13]
            // Reg14             |     M26    |     M30     | v[14]
            // Reg15             |     M27    |     M31     | v[15]
            if(threadIdx.x == colId)
            {
                printf("b(%d,%d) is scaled from thread %f\n", testId, colId, log2f(regC[0]));
            }
        }
        if(threadIdx.x == 32)
        {
            printf("\n");
        }
    }
}

int main()
{
    kernel_a_scale_mapping<<<1, 64>>>();
    kernel_b_scale_mapping<<<1, 64>>>();
    return 0;
}