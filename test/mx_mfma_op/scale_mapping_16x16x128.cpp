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
    using dataC  = float __attribute__((ext_vector_type(4)));
    using dataX  = int32_t __attribute__((ext_vector_type(4)));

    // A[ 16, 128]
    // B[128,  16]
    // C[ 16,  16]

    dataAB regA(0);
    dataAB regB(0);
    dataC regC(0.0f);

    dataX xa(bit_cast<int32_t>(threadIdx.x) + 0x7F); // 127{2^0}, 127+1{2^1},...,127+63{2^63}
    dataX xb(0x3F800000);

    // fill first column of B with 1.0
    if(threadIdx.x % 16 == 0)
    {
        for(size_t i = 0; i < 32; i++)
        {
            regB[i] = 0x38; // 1.0
        }
    }

    // verify scale mapping for each row
    for(int rowId = 0; rowId < 16; rowId++)
    {
        for(int testId = 0; testId < 128; testId++)
        {
            if(threadIdx.x == 0 && false)
            {
                printf("testId: %d\n", testId);
            }
            regA = dataAB(0);
            regC = dataC(0.0f);

            if(threadIdx.x % 16 == rowId)
            {
                if(threadIdx.x / 16 == testId / 32)
                {
                    // set a(rowId,testId) = 1.0
                    regA[testId % 32] = 0x38; // 1.0
                }
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
            regC = __builtin_amdgcn_mfma_scale_f32_16x16x128_f8f6f4(regA,
                                                                    regB,
                                                                    regC,
                                                                    0, // cbsz
                                                                    0, // blgp
                                                                    0,
                                                                    xa[threadIdx.x / 16],
                                                                    0,
                                                                    xb[threadIdx.x / 16]);

            __syncthreads();

            if(threadIdx.x % 32 == 0 && false) // row 0
            {
                printf(
                    "thread: %u -- xA: %x\n", threadIdx.x, bit_cast<int32_t>(xa[threadIdx.x / 32]));
                printf(
                    "thread: %u -- xB: %x\n", threadIdx.x, bit_cast<int32_t>(xb[threadIdx.x / 32]));
            }

            // Here we want to store a 16x16 block of data.
            //
            // Size              |   BLOCK_N  |   BLOCK_N   |   BLOCK_N   |   BLOCK_N   |
            // N                 | 0  ...  15 |  0  ...  15 | 0  ...  15  |  0  ...  15 |
            // Thread Id         | 0  ...  15 | 16  ...  31 | 32  ... 47  | 48  ...  63 | Vector
            // Register Element   ------------ ------------- ------------ -------------- Element
            // Reg0              |     M0     |     M4      |     M8      |     M12     | v[0]
            // Reg1              |     M1     |     M5      |     M9      |     M13     | v[1]
            // Reg2              |     M2     |     M6      |     M10     |     M14     | v[2]
            // Reg3              |     M3     |     M7      |     M11     |     M15     | v[3]
            if(threadIdx.x % 16 == 0) // {0, 16, 32, 48} -- C colId == 0
            {
                if(threadIdx.x / 16 == rowId / 4)
                {
                    printf("a(%d,%d) is scaled from thread %f\n",
                           rowId,
                           testId,
                           log2f(regC[rowId % 4]));
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
    using dataC  = float __attribute__((ext_vector_type(4)));
    using dataX  = int32_t __attribute__((ext_vector_type(4)));

    // A[ 16, 128]
    // B[128,  16]
    // C[ 16,  16]

    dataAB regA(0);
    dataAB regB(0);
    dataC regC(0.0f);

    dataX xa(0x3F800000);
    dataX xb(bit_cast<int32_t>(threadIdx.x) + 0x7F); // 127{2^0}, 127+1{2^1},...,127+63{2^63}

    // fill first row of A with 1.0
    if(threadIdx.x % 16 == 0)
    {
        for(size_t i = 0; i < 32; i++)
        {
            regA[i] = 0x38; // 1.0
        }
    }

    // verify scale mapping for each row
    for(int colId = 0; colId < 16; colId++)
    {
        for(int testId = 0; testId < 128; testId++)
        {
            regB = dataAB(0);
            regC = dataC(0.0f);

            if(threadIdx.x % 16 == colId)
            {
                if(threadIdx.x / 16 == testId / 32)
                {
                    // set a(rowId,testId) = 1.0
                    regB[testId % 32] = 0x38; // 1.0
                }
            }
            __syncthreads();

            regC = __builtin_amdgcn_mfma_scale_f32_16x16x128_f8f6f4(regA,
                                                                    regB,
                                                                    regC,
                                                                    0, // cbsz
                                                                    0, // blgp
                                                                    0,
                                                                    xa[threadIdx.x / 16],
                                                                    0,
                                                                    xb[threadIdx.x / 16]);

            __syncthreads();

            if(threadIdx.x % 32 == 0 && false) // row 0
            {
                printf(
                    "thread: %u -- xA: %x\n", threadIdx.x, bit_cast<int32_t>(xa[threadIdx.x / 32]));
                printf(
                    "thread: %u -- xB: %x\n", threadIdx.x, bit_cast<int32_t>(xb[threadIdx.x / 32]));
            }

            // Here we want to store a 16x16 block of data.
            //
            // Size              |   BLOCK_N  |   BLOCK_N   |   BLOCK_N   |   BLOCK_N   |
            // N                 | 0  ...  15 |  0  ...  15 | 0  ...  15  |  0  ...  15 |
            // Thread Id         | 0  ...  15 | 16  ...  31 | 32  ... 47  | 48  ...  63 | Vector
            // Register Element   ------------ ------------- ------------ -------------- Element
            // Reg0              |     M0     |     M4      |     M8      |     M12     | v[0]
            // Reg1              |     M1     |     M5      |     M9      |     M13     | v[1]
            // Reg2              |     M2     |     M6      |     M10     |     M14     | v[2]
            // Reg3              |     M3     |     M7      |     M11     |     M15     | v[3]
            if(threadIdx.x == colId)
            {
                printf("b(%d,%d) is scaled from thread %f\n", testId, colId, log2f(regC[0]));
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

int main()
{
    kernel_a_scale_mapping<<<1, 64>>>();
    kernel_b_scale_mapping<<<1, 64>>>();
    return 0;
}