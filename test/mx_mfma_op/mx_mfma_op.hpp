#pragma once

#include "ck/ck.hpp"

#include "ck/library/utility/host_tensor.hpp"
#include "ck/library/utility/device_memory.hpp"
#include "ck/tensor_operation/gpu/device/tensor_layout.hpp"
#include "ck/tensor_operation/gpu/warp/xdlops_gemm.hpp"
#include "ck/library/utility/host_tensor_generator.hpp"
#include "ck/library/reference_tensor_operation/cpu/reference_gemm.hpp"
#include "ck/library/utility/check_err.hpp"

namespace ck {
namespace mx_mfma_test {

// MFMA instructions supported in this test
enum class MFMA_F8F6F4
{
    F32_16x16x128 =
        static_cast<int>(MfmaInstr::mfma_f32_16x16x128f8f6f4), // V_MFMA_F32_16X16X128_F8F6F4
    F32_32x32x64 =
        static_cast<int>(MfmaInstr::mfma_f32_32x32x64f8f6f4) // V_MFMA_F32_32X32X64_F8F6F4
};

template <typename AFragT, typename BFragT, typename AccumFragT, int32_t BLOCK_M, int32_t BLOCK_N>
struct mfma_type_selector;

template <typename AFragT, typename BFragT, typename AccumFragT>
struct mfma_type_selector<AFragT, BFragT, AccumFragT, 16, 16>
{
    __device__ void operator()(AFragT const& fragA, BFragT const& fragB, AccumFragT& fragAcc)
    {
#if 1
        auto op = mfma_type<MfmaInstr::mfma_f32_16x16x128f8f6f4>{};
        op.template run<16, 16, AFragT, BFragT, AccumFragT>(fragA, fragB, fragAcc);
#else
        ignore = fragA;
        ignore = fragB;
        ignore = fragAcc;
#endif
    }
};

template <typename AFragT, typename BFragT, typename AccumFragT>
struct mfma_type_selector<AFragT, BFragT, AccumFragT, 32, 32>
{
    __device__ void operator()(AFragT const& fragA, BFragT const& fragB, AccumFragT& fragAcc)
    {
#if 1
        auto op = mfma_type<MfmaInstr::mfma_f32_32x32x64f8f6f4>{};
        op.template run<32, 32, AFragT, BFragT, AccumFragT>(fragA, fragB, fragAcc);
#else
        ignore = fragA;
        ignore = fragB;
        ignore = fragAcc;
#endif
    }
};

template <typename VecT>
static constexpr int32_t vectorSize(const VecT&)
{
    return scalar_type<VecT>::vector_size;
}

// Define a load function for input A blocks:
// Size: (BLOCK_M x BLOCK_K)
// ASSUMPTION:
// - We want contiguous BLOCK_M sized column neighbors in register.
// - Data is in col_major format
// This means:
// - From A we will load K columns of size BLOCK_M to satisfy our input data
template <typename AType, typename AFragT, int32_t BLOCK_M, int32_t BLOCK_K>
__device__ AFragT load_A_col_major(AType const* input_ptr)
{
    // Here we want to load a BLOCK_M x BLOCK_K block of data.
    static constexpr uint32_t VW = vectorSize(AFragT{});
    using ARawT                  = typename scalar_type<AFragT>::type;
    using AScalarFragT           = vector_type<ARawT, VW>::type;

    // To start the loading process, let's visualize in 2D coords.
    // Each thread will load 32 elements.
    // We need to know where they start, and where the next elements are.
    auto startCoord2D = std::make_pair(threadIdx.x % BLOCK_M,         // Row
                                       (threadIdx.x / BLOCK_M) * VW); // Col
    auto stepCoord2D  = std::make_pair(0u, 1u);

    // Flatten to 1D col_major offsets.
    auto col_major = [](auto const& coord, auto ld) { return coord.first + coord.second * ld; };

    // BLOCK_M is a stride in A matrix
    auto startOffset = col_major(startCoord2D, BLOCK_M);
    auto kOffset     = col_major(stepCoord2D, BLOCK_M);
// kOffset == BLOCK_M
// This means every BLOCK_M element is loaded into output vector
#if 0
    auto fragA = AScalarFragT{
        bit_cast<ARawT>(input_ptr[startOffset]),                 // XXX v[0] = Reg 0 [0:7]
        bit_cast<ARawT>(input_ptr[startOffset + 1 * kOffset]),   // XXX v[1] = Reg 0 [8:15]
        bit_cast<ARawT>(input_ptr[startOffset + 2 * kOffset]),   // XXX v[2] = Reg 0 [16:23]
        bit_cast<ARawT>(input_ptr[startOffset + 3 * kOffset]),   // XXX v[3] = Reg 0 [24:31]
        bit_cast<ARawT>(input_ptr[startOffset + 4 * kOffset]),   // XXX v[4] = Reg 1 [0:7]
        bit_cast<ARawT>(input_ptr[startOffset + 5 * kOffset]),   // XXX v[5] = Reg 1 [8:15]
        bit_cast<ARawT>(input_ptr[startOffset + 6 * kOffset]),   // XXX v[6] = Reg 1 [16:23]
        bit_cast<ARawT>(input_ptr[startOffset + 7 * kOffset]),   // XXX v[7] = Reg 1 [24:31]
        bit_cast<ARawT>(input_ptr[startOffset + 8 * kOffset]),   // XXX v[8] = Reg 2 [0:7]
        bit_cast<ARawT>(input_ptr[startOffset + 9 * kOffset]),   // XXX v[9] = Reg 2 [8:15]
        bit_cast<ARawT>(input_ptr[startOffset + 10 * kOffset]),  // XXX v[10] = Reg 2 [16:23]
        bit_cast<ARawT>(input_ptr[startOffset + 11 * kOffset]),  // XXX v[11] = Reg 2 [24:31]
        bit_cast<ARawT>(input_ptr[startOffset + 12 * kOffset]),  // XXX v[12] = Reg 3 [0:7]
        bit_cast<ARawT>(input_ptr[startOffset + 13 * kOffset]),  // XXX v[13] = Reg 3 [8:15]
        bit_cast<ARawT>(input_ptr[startOffset + 14 * kOffset]),  // XXX v[14] = Reg 3 [16:23]
        bit_cast<ARawT>(input_ptr[startOffset + 15 * kOffset]),  // XXX v[15] = Reg 3 [24:31]
        bit_cast<ARawT>(input_ptr[startOffset + 16 * kOffset]),  // XXX v[16] = Reg 4 [0:7]
        bit_cast<ARawT>(input_ptr[startOffset + 17 * kOffset]),  // XXX v[17] = Reg 4 [8:15]
        bit_cast<ARawT>(input_ptr[startOffset + 18 * kOffset]),  // XXX v[18] = Reg 4 [16:23]
        bit_cast<ARawT>(input_ptr[startOffset + 19 * kOffset]),  // XXX v[19] = Reg 4 [24:31]
        bit_cast<ARawT>(input_ptr[startOffset + 20 * kOffset]),  // XXX v[20] = Reg 5 [0:7]
        bit_cast<ARawT>(input_ptr[startOffset + 21 * kOffset]),  // XXX v[21] = Reg 5 [8:15]
        bit_cast<ARawT>(input_ptr[startOffset + 22 * kOffset]),  // XXX v[22] = Reg 5 [16:23]
        bit_cast<ARawT>(input_ptr[startOffset + 23 * kOffset]),  // XXX v[23] = Reg 5 [24:31]
        bit_cast<ARawT>(input_ptr[startOffset + 24 * kOffset]),  // XXX v[24] = Reg 6 [0:7]
        bit_cast<ARawT>(input_ptr[startOffset + 25 * kOffset]),  // XXX v[25] = Reg 6 [8:15]
        bit_cast<ARawT>(input_ptr[startOffset + 26 * kOffset]),  // XXX v[26] = Reg 6 [16:23]
        bit_cast<ARawT>(input_ptr[startOffset + 27 * kOffset]),  // XXX v[27] = Reg 6 [24:31]
        bit_cast<ARawT>(input_ptr[startOffset + 28 * kOffset]),  // XXX v[28] = Reg 7 [0:7]
        bit_cast<ARawT>(input_ptr[startOffset + 29 * kOffset]),  // XXX v[29] = Reg 7 [8:15]
        bit_cast<ARawT>(input_ptr[startOffset + 30 * kOffset]),  // XXX v[30] = Reg 7 [16:23]
        bit_cast<ARawT>(input_ptr[startOffset + 31 * kOffset])}; // XXX v[31] = Reg 7 [24:31]
#else
    auto fragA = AScalarFragT{};
    static_for<0, VW, 1>{}([&](auto i) {
        fragA[static_cast<int>(i)] =
            bit_cast<ARawT>(input_ptr[startOffset + static_cast<int>(i) * kOffset]);
    });
#endif
    return fragA;
}

// Define a load function for input B blocks:
// Size: (BLOCK_K x BLOCK_N)
// ASSUMPTION:
// - We want contiguous BLOCK_N sized row neighbors in register.
// - Data is in row_major format
// This means:
// - From B we will load K rows of size BLOCK_N to satisfy our input data
template <typename BType, typename BFragT, int32_t BLOCK_K, int32_t BLOCK_N>
__device__ BFragT load_B_col_major(BType const* input_ptr)
{
    // Here we want to load a BLOCK_K x BLOCK_N block of data.
    static constexpr uint32_t VW = vectorSize(BFragT{});

    // To start the loading process, let's visualize in 2D coords.
    // Each thread will load 32 elements.
    // We need to know where they start, and where the next elements are.
    auto startCoord2D = std::make_pair((threadIdx.x / BLOCK_N) * VW, // Row
                                       threadIdx.x % BLOCK_N);       // Col
    // auto stepCoord2D  = std::make_pair(1u, 0u);

    // Flatten to 1D col_major offsets.
    auto col_major = [](auto const& coord, auto ld) { return coord.first + coord.second * ld; };

    auto startOffset = col_major(startCoord2D, BLOCK_K);
    // auto kOffset     = col_major(stepCoord2D, BLOCK_K);

    // kOffset == 1
    auto const* fragPtr = reinterpret_cast<BFragT const*>(input_ptr + startOffset);
    return *fragPtr;
}

// Define a store function for C
// Size: (BLOCK_M x BLOCK_N)
// ASSUMPTION:
// - We want contiguous BLOCK_N sized row neighbors in register.
// - Data is in col_major format
// This means:
// - From C we will load BLOCK_M rows of size BLOCK_N to satisfy our input data
template <typename CType, typename CFragT, int32_t BLOCK_M, int32_t BLOCK_N>
struct store_C_col_major;

// Here we want to store a 16x16 block of data.
//
// Size              |   BLOCK_N  |   BLOCK_N   |   BLOCK_N   |   BLOCK_N    | Vector
// Register Element  | 0  ...  15 | 16  ...  31 | 32  ...  47 | 48  ...   63 | Element
//                    ____________ _____________ _____________ ______________
// Reg0              |     M0     |     M4      |     M8      |     M12      | v[0]
// Reg1              |     M1     |     M5      |     M9      |     M13      | v[1]
// Reg2              |     M2     |     M6      |     M10     |     M14      | v[2]
// Reg3              |     M3     |     M7      |     M11     |     M15      | v[3]
template <typename CType, typename CFragT>
struct store_C_col_major<CType, CFragT, 16, 16>
{
    __device__ void operator()(CType* output, CFragT cFrag)
    {
        static constexpr uint32_t VW  = vectorSize(cFrag); // 4
        static constexpr uint32_t Dim = 16;

#if 1
        for(int i = 0; i < vectorSize(cFrag); ++i)
        {
            printf("threadIdx.x = %d; cFrag[%d] = %f\n",
                   static_cast<int>(threadIdx.x),
                   i,
                   static_cast<float>(cFrag[i]));
        }
#endif

        // Each thread will load 4 elements.
        // We need to know where they start, and where the next elements are.
        auto startCoord2D = std::make_pair((threadIdx.x / Dim) * VW, // Row
                                           threadIdx.x % Dim);       // Col
        // auto stepCoord2D  = std::make_pair(1u, 0u);

        // Flatten to 1D col_major offsets.
        auto col_major = [](auto const& coord, auto ld) { return coord.first + coord.second * ld; };

        auto startOffset = col_major(startCoord2D, 16);
        // auto kOffset     = col_major(stepCoord2D, 16); // 1

        // kOffset == 1
        auto* fragPtr = reinterpret_cast<CFragT*>(output + startOffset);
        *fragPtr      = cFrag;
    }
};

// Here we want to store a 32x32 block of data.
// Register Mapping:

// Size              |   BLOCK_N  |   BLOCK_N   | Vector
// Register Element  | 0  ...  31 | 32  ...  63 | Element
//                    ____________ _____________
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

template <typename CType, typename CFragT>
struct store_C_col_major<CType, CFragT, 32, 32>
{
    __device__ void operator()(CType* output, CFragT cFrag)
    {
        static constexpr uint32_t WAVE_SIZE      = 64;
        static constexpr uint32_t VW             = 4;
        static constexpr uint32_t Dim            = 32;
        static constexpr uint32_t M_PER_VW_CHUNK = VW * WAVE_SIZE / 32; // 8

#if 1
        for(int i = 0; i < vectorSize(cFrag); ++i)
        {
            printf("threadIdx.x = %d; cFrag[%d] = %f\n",
                   static_cast<int>(threadIdx.x),
                   i,
                   static_cast<float>(cFrag[i]));
        }
#endif

        auto startCoord2D = std::make_pair((threadIdx.x / Dim) * VW, // Row
                                           threadIdx.x % Dim);       // Col

        // Minor step for each 'chunk'
        // auto minorStepCoord2D = std::make_pair(1u, 0u);

        // Major step between 'chunks'
        auto majorStepCoord2D = std::make_pair(M_PER_VW_CHUNK, 0);

        // Flatten to 1D col_major offsets.
        auto col_major = [](auto const& coord, auto ld) { return coord.first + coord.second * ld; };

        auto startOffset = col_major(startCoord2D, 32);
        // auto kMinorOffset = col_major(minorStepCoord2D, 32); // 1
        auto kMajorOffset = col_major(majorStepCoord2D, 32); // 8

        // kMinorOffset == 1.
        // This means we can vector store 4 contiguous elements at a time.
        using CRawT        = typename scalar_type<CFragT>::type;
        using CScalarFragT = vector_type<CRawT, VW>::type;
        union
        {
            CFragT frag;
            CScalarFragT chunks[vectorSize(CFragT{}) / VW];
        } fragC{cFrag}; // Initialize with input fragment

        *(reinterpret_cast<CScalarFragT*>(output + startOffset))                = fragC.chunks[0];
        *(reinterpret_cast<CScalarFragT*>(output + startOffset + kMajorOffset)) = fragC.chunks[1];
        *(reinterpret_cast<CScalarFragT*>(output + startOffset + 2 * kMajorOffset)) =
            fragC.chunks[2];
        *(reinterpret_cast<CScalarFragT*>(output + startOffset + 3 * kMajorOffset)) =
            fragC.chunks[3];
    }
};

template <typename AType,
          typename BType,
          typename CType,
          typename AccType,
          int32_t BLOCK_M,
          int32_t BLOCK_N,
          int32_t BLOCK_K>
__global__ void matmul(const AType* a, const BType* b, CType* c)
{
    constexpr int WAVE_SIZE = 64;
    assert(threadIdx.x < WAVE_SIZE);
    assert(blockDim.x == 1 && blockDim.y == 1 && blockDim.z == 1);

    using AFragT        = vector_type<AType, BLOCK_M * BLOCK_K / WAVE_SIZE>::type;
    using BFragT        = vector_type<BType, BLOCK_K * BLOCK_N / WAVE_SIZE>::type;
    using CFragT        = vector_type<CType, BLOCK_M * BLOCK_N / WAVE_SIZE>::type;
    using AccumFragT    = vector_type<AccType, BLOCK_M * BLOCK_N / WAVE_SIZE>;
    using RawAccumFragT = vector_type<AccType, BLOCK_M * BLOCK_N / WAVE_SIZE>::type;

    // Create frags
    auto fragA   = AFragT{};
    auto fragB   = BFragT{};
    auto fragC   = CFragT{};
    auto fragAcc = AccumFragT{0};

    // Load the inputs.
    // A = col major, BLOCK_M x BLOCK_K
    fragA = load_A_col_major<AType, AFragT, BLOCK_M, BLOCK_K>(a);
    // B = col major, BLOCK_K x BLOCK_N
    fragB = load_B_col_major<BType, BFragT, BLOCK_K, BLOCK_N>(b);

    // Matrix multiply-accumulate using MFMA units
    // Accumulation intermediate = BLOCK_M x BLOCK_N
    mfma_type_selector<AFragT, BFragT, AccumFragT, BLOCK_M, BLOCK_N>{}(fragA, fragB, fragAcc);

    for(int i = 0; i < vectorSize(fragC); ++i)
    {
        fragC[i] = type_convert<CType>(fragAcc.template AsType<RawAccumFragT>()[Number<0>{}][i]);
    }

    auto storeC = store_C_col_major<CType, CFragT, BLOCK_M, BLOCK_N>{};
    storeC(c, fragC);
}

/**
 * @brief Structure to hold dimension parameters for GEMM tensors.
 *
 * M Number of rows in matrix A and matrix C.
 * N Number of columns in matrix B and matrix C.
 * K Number of columns in matrix A and number of rows in matrix B.
 * StrideA Stride (leading dimension) of matrix A.
 * StrideB Stride (leading dimension) of matrix B.
 * StrideC Stride (leading dimension) of matrix C.
 */
struct GemmParams
{
    /**
     * @brief This constructor initializes the parameters for GEMM storage with default values.
     *
     * A[16x128] * B[128x16] = C[16x16], all row major.
     */
    GemmParams() : M(16), N(16), K(128) {}

    ck::index_t M;
    ck::index_t N;
    ck::index_t K;

    ck::index_t StrideA = -1;
    ck::index_t StrideB = -1;
    ck::index_t StrideC = -1;
};

template <typename GemmInstance,
          typename ADataType,
          typename BDataType,
          typename CDataType,
          typename AElementwiseOperation,
          typename BElementwiseOperation,
          typename CElementwiseOperation>
void RunHostGEMM(const Tensor<ADataType>& A,
                 const Tensor<BDataType>& B,
                 Tensor<CDataType>& C,
                 AElementwiseOperation a_element_op,
                 BElementwiseOperation b_element_op,
                 CElementwiseOperation c_element_op)
{
    auto ref_gemm     = GemmInstance{};
    auto ref_invoker  = ref_gemm.MakeInvoker();
    auto ref_argument = ref_gemm.MakeArgument(A, B, C, a_element_op, b_element_op, c_element_op);

    ref_invoker.Run(ref_argument);
}

template <typename KernelType, typename ADataType, typename BDataType, typename CDataType>
bool RunDeviceGEMM(KernelType kernel,
                   const Tensor<ADataType>& A,
                   const Tensor<BDataType>& B,
                   Tensor<CDataType>& C)
{
    DeviceMem a_m_k_device_buf(sizeof(ADataType) * A.mDesc.GetElementSpaceSize());
    DeviceMem b_n_k_device_buf(sizeof(BDataType) * B.mDesc.GetElementSpaceSize());
    DeviceMem c_m_n_device_buf(sizeof(CDataType) * C.mDesc.GetElementSpaceSize());

    a_m_k_device_buf.ToDevice(A.mData.data());
    b_n_k_device_buf.ToDevice(B.mData.data());
    kernel<<<1, 64>>>(static_cast<const ADataType*>(a_m_k_device_buf.GetDeviceBuffer()),
                      static_cast<const BDataType*>(b_n_k_device_buf.GetDeviceBuffer()),
                      static_cast<CDataType*>(c_m_n_device_buf.GetDeviceBuffer()));
    c_m_n_device_buf.FromDevice(C.mData.data());

    return true;
}

template <typename DeviceMFMA,
          typename ADataType,
          typename BDataType,
          typename CDataType,
          typename GPUAccDataType,
          typename CPUAccDataType,
          typename ALayout,
          typename BLayout,
          typename CLayout,
          index_t BLOCK_M,
          index_t BLOCK_N,
          index_t BLOCK_K>
struct TestMFMA
{
    auto PrepareGemmTensors(const GemmParams& params)
    {
        auto f_host_tensor_descriptor =
            [](std::size_t row, std::size_t col, std::size_t stride, auto layout) {
                if(std::is_same<decltype(layout), ck::tensor_layout::gemm::RowMajor>::value)
                {
                    return HostTensorDescriptor(std::vector<std::size_t>({row, col}),
                                                std::vector<std::size_t>({stride, 1}));
                }
                else
                {
                    return HostTensorDescriptor(std::vector<std::size_t>({row, col}),
                                                std::vector<std::size_t>({1, stride}));
                }
            };

        Tensor<ADataType> a_m_k(
            f_host_tensor_descriptor(params.M, params.K, params.StrideA, ALayout{}));
        Tensor<BDataType> b_n_k(
            f_host_tensor_descriptor(params.K, params.N, params.StrideB, BLayout{}));
        Tensor<CDataType> c_m_n_host_result(
            f_host_tensor_descriptor(params.M, params.N, params.StrideC, CLayout{}));
        Tensor<CDataType> c_m_n_device_result(
            f_host_tensor_descriptor(params.M, params.N, params.StrideC, CLayout{}));

        switch(0)
        {
        case 0:
            a_m_k.GenerateTensorValue(GeneratorTensor_1<ADataType>{0.015625f});
            // NOTE: not all numbers are representable in FP8, BF8, etc.
            b_n_k.GenerateTensorValue(GeneratorTensor_Sequential<BDataType, 1>{});
            break;
        case 1:
            a_m_k.GenerateTensorValue(GeneratorTensor_1<ADataType>{1.0f});
            b_n_k.GenerateTensorValue(GeneratorTensor_1<BDataType>{1.0f});
            break;
        case 2:
            // expect small round off errors
            a_m_k.GenerateTensorValue(GeneratorTensor_3<ADataType>{-5, 5});
            b_n_k.GenerateTensorValue(GeneratorTensor_3<BDataType>{-5, 5});
            break;
        case 3:
            // expect small round off errors
            a_m_k.GenerateTensorValue(GeneratorTensor_4<ADataType>(-1, 3));
            b_n_k.GenerateTensorValue(GeneratorTensor_4<BDataType>(1, 3));
            break;
        default:
            a_m_k.GenerateTensorValue(GeneratorTensor_2<ADataType>{-5, 6});
            b_n_k.GenerateTensorValue(GeneratorTensor_2<BDataType>{-5, 6});

            break;
        }

        return std::make_tuple(a_m_k, b_n_k, c_m_n_host_result, c_m_n_device_result);
    }

    auto operator()(const DeviceMFMA& mfma_kernel)
    {
        std::cout << "ALayout = " << ALayout{}.name << ", BLayout = " << BLayout{}.name
                  << ", CLayout = " << CLayout{}.name << std::endl;

        // Arrange
        GemmParams params;
        params.M = BLOCK_M;
        params.N = BLOCK_N;
        params.K = BLOCK_K;

        auto f_get_default_stride = [](std::size_t row,
                                       std::size_t col,
                                       ck::index_t stride,
                                       auto layout) {
            if(stride == -1)
            {
                // give a chance if stride is -1, return a default packed stride
                if constexpr(std::is_same_v<decltype(layout), ck::tensor_layout::gemm::RowMajor>)
                {
                    return static_cast<std::size_t>(col);
                }
                else
                {
                    return static_cast<std::size_t>(row);
                }
            }
            else
                return static_cast<std::size_t>(stride);
        };

        params.StrideA = f_get_default_stride(BLOCK_M, BLOCK_K, params.StrideA, ALayout{});
        params.StrideB = f_get_default_stride(BLOCK_K, BLOCK_N, params.StrideB, BLayout{});
        params.StrideC = f_get_default_stride(BLOCK_M, BLOCK_N, params.StrideC, CLayout{});

        auto host_tensors = PrepareGemmTensors(params);

        const Tensor<ADataType>& a  = std::get<0>(host_tensors);
        const Tensor<BDataType>& b  = std::get<1>(host_tensors);
        Tensor<CDataType>& c_host   = std::get<2>(host_tensors);
        Tensor<CDataType>& c_device = std::get<3>(host_tensors);

        using PassThrough = ck::tensor_operation::element_wise::PassThrough;

        auto a_element_op = PassThrough{};
        auto b_element_op = PassThrough{};
        auto c_element_op = PassThrough{};

        using ReferenceGemmInstance = ck::tensor_operation::host::ReferenceGemm<ADataType,
                                                                                BDataType,
                                                                                CDataType,
                                                                                CPUAccDataType,
                                                                                PassThrough,
                                                                                PassThrough,
                                                                                PassThrough>;

        RunHostGEMM<ReferenceGemmInstance>(a, b, c_host, a_element_op, b_element_op, c_element_op);

        RunDeviceGEMM(mfma_kernel, a, b, c_device);

        bool res = false;
        if constexpr(std::is_same<CDataType, float>::value ||
                     std::is_same<CDataType, half_t>::value)
        {
            res = ck::utils::check_err(c_device.mData, c_host.mData);
            std::cout << (res ? "SUCCESS" : "FAILURE") << std::endl;
        }
        else
        {
            std::cout << "UNSUPPORTED CDataType" << std::endl;
        }

        return res;
    }
};

} // namespace mx_mfma_test
} // namespace ck
