#ifndef CK_GRIDWISE_GEMM_AVX2_HPP
#define CK_GRIDWISE_GEMM_AVX2_HPP

#include "common_header.hpp"
#include "multi_index_transform_helper.hpp"
#include "tensor_descriptor.hpp"
#include "tensor_descriptor_helper.hpp"
#include "blockwise_gemm_avx2.hpp"
#include "threadwise_tensor_slice_transfer_avx2.hpp"
#include "dynamic_buffer_cpu.hpp"

namespace ck {
namespace cpu {

template <typename GridwiseGemm,
          typename FloatA,
          typename FloatB,
          typename FloatC,
          typename AGridDesc,
          typename BGridDesc,
          typename CGridDesc,
          typename AElementwiseOperation,
          typename BElementwiseOperation,
          typename CElementwiseOperation>
void kernel_gemm_avx_mxn(const FloatA* __restrict__ p_a_grid,
                         const FloatB* __restrict__ p_b_grid,
                         FloatC* __restrict__ p_c_grid,
                         const AGridDesc& a_grid_desc,
                         const BGridDesc& b_grid_desc,
                         const CGridDesc& c_grid_desc,
                         const AElementwiseOperation& a_element_op,
                         const BElementwiseOperation& b_element_op,
                         const CElementwiseOperation& c_element_op)
{
    GridwiseGemm::Run(p_a_grid,
                      p_b_grid,
                      p_c_grid,
                      a_grid_desc,
                      b_grid_desc,
                      c_grid_desc,
                      a_element_op,
                      b_element_op,
                      c_element_op);
}

template <typename FloatA,
          typename FloatB,
          typename FloatC,
          typename AccDataType,
          typename AGridDesc,
          typename BGridDesc,
          typename CGridDesc,
          typename AElementwiseOperation,
          typename BElementwiseOperation,
          typename CElementwiseOperation,
          ck::index_t MPerBlock, // block means data are designed to fit in cache (L1/L2/L3)
          ck::index_t NPerBlock,
          ck::index_t KPerBlock,
          typename ThreadwiseGemm_Dispatch,
          typename BlockMNKAccessOrder, // how we accss gemm MNK to better fit in cache
          typename ThreadMNAccessOrder, // how we acces gemm MN to utilize micro kernel
          bool UseCLocalBuffer // if true, will allocate a buffer and write to it in kernel, then
                               // copy back to block buffer. if false, will write to C directly
          >
struct GridwiseGemmAvx2_MxN
{
    static constexpr auto I0 = Number<0>{};
    static constexpr auto I1 = Number<1>{};
    static constexpr auto I2 = Number<2>{};
    static constexpr auto I3 = Number<3>{};
    static constexpr auto I4 = Number<4>{};
    static constexpr auto I5 = Number<5>{};
    static constexpr auto I6 = Number<6>{};
    static constexpr auto I7 = Number<7>{};

    // static constexpr auto Avx2RegisterVector = 8;   // 8 floats
    static constexpr index_t MemAlignmentByte = 32; // 256bit

    static constexpr auto GetABlockDescriptor()
    {
        if constexpr(std::is_same<typename ThreadwiseGemm_Dispatch::MatrixALayout,
                                  ck::tensor_layout::gemm::RowMajor>::value)
        {
            // A : M, K
            constexpr auto a_block_desc_m_k =
                make_naive_tensor_descriptor_packed(make_tuple(MPerBlock, KPerBlock));
            return a_block_desc_m_k;
        }
        else
        {
            // A : K, M
            constexpr auto a_block_desc_k_m = make_naive_tensor_descriptor_packed(
                make_tuple(KPerBlock,
                           math::integer_least_multiple(
                               MPerBlock, ThreadwiseGemm_Dispatch::MatrixAMinVectorSize)));
            return a_block_desc_k_m;
        }
    }

    static constexpr auto GetBBlockDescriptor()
    {
        if constexpr(std::is_same<typename ThreadwiseGemm_Dispatch::MatrixBLayout,
                                  ck::tensor_layout::gemm::RowMajor>::value)
        {
            // B : K, N
            constexpr auto b_block_desc_k_n = make_naive_tensor_descriptor_packed(
                make_tuple(KPerBlock,
                           math::integer_least_multiple(
                               NPerBlock, ThreadwiseGemm_Dispatch::MatrixBMinVectorSize)));
            return b_block_desc_k_n;
        }
        else
        {
            // B : N/8, K, N8
            constexpr auto b_block_desc_n0_k_n1 = make_naive_tensor_descriptor_packed(make_tuple(
                math::integer_divide_ceil(NPerBlock, ThreadwiseGemm_Dispatch::MatrixBMinVectorSize),
                KPerBlock,
                ThreadwiseGemm_Dispatch::MatrixBMinVectorSize));
            return b_block_desc_n0_k_n1;
        }
    }

    static constexpr auto GetABlockSliceLength()
    {
        if constexpr(std::is_same<typename ThreadwiseGemm_Dispatch::MatrixALayout,
                                  ck::tensor_layout::gemm::RowMajor>::value)
        {
            // A : M, K
            return ck::Sequence<MPerBlock, KPerBlock>{};
        }
        else
        {
            // A : K, M
            return ck::Sequence<KPerBlock, MPerBlock>{};
        }
    }

    static constexpr auto GetBBlockSliceLength()
    {
        if constexpr(std::is_same<typename ThreadwiseGemm_Dispatch::MatrixBLayout,
                                  ck::tensor_layout::gemm::RowMajor>::value)
        {
            // B : K, N
            return ck::Sequence<KPerBlock, NPerBlock>{};
        }
        else
        {
            // B : N/8, K, N88;
            return ck::Sequence<NPerBlock / ThreadwiseGemm_Dispatch::MatrixBMinVectorSize,
                                KPerBlock,
                                ThreadwiseGemm_Dispatch::MatrixBMinVectorSize>{};
        }
    }

    static constexpr auto GetABlockDimAccessOrder() { return ck::Sequence<0, 1>{}; }

    static constexpr auto GetBBlockDimAccessOrder()
    {
        if constexpr(std::is_same<typename ThreadwiseGemm_Dispatch::MatrixBLayout,
                                  ck::tensor_layout::gemm::RowMajor>::value)
        {
            // B : K, N
            return ck::Sequence<0, 1>{};
        }
        else
        {
            // B : N/8, K, N88;
            return ck::Sequence<0, 1, 2>{};
        }
    }

    static constexpr auto GetABlockMoveFwdStep()
    {
        if constexpr(std::is_same<typename ThreadwiseGemm_Dispatch::MatrixALayout,
                                  ck::tensor_layout::gemm::RowMajor>::value)
        {
            // A : M, K
            return ck::make_multi_index(0, KPerBlock);
        }
        else
        {
            // A : K, M
            return ck::make_multi_index(KPerBlock, 0);
        }
    }

    static constexpr auto GetBBlockMoveFwdStep()
    {
        if constexpr(std::is_same<typename ThreadwiseGemm_Dispatch::MatrixBLayout,
                                  ck::tensor_layout::gemm::RowMajor>::value)
        {
            // B : K, N
            return ck::make_multi_index(KPerBlock, 0);
        }
        else
        {
            // B : N/8, K, N88;
            return ck::make_multi_index(0, KPerBlock, 0);
        }
    }

#if 0
    static constexpr auto GetAThreadDiscriptor()
    {
        if constexpr (std::is_same<typename ThreadwiseGemm_Dispatch::MatrixALayout, ck::tensor_layout::gemm::RowMajor>::value){
            // A : M, K
            constexpr auto a_thread_desc_m_k = make_naive_tensor_descriptor_packed(make_tuple(ThreadwiseGemm_Dispatch::ThreadMaxMr, KPerBlock));
            return a_thread_desc_m_k;
        } else {
            // A : K, M
            constexpr auto a_thread_desc_k_m = make_naive_tensor_descriptor_packed(make_tuple(KPerBlock, ThreadwiseGemm_Dispatch::ThreadMaxMr));
            return a_thread_desc_k_m;
        }
    }

    static constexpr auto GetBThreadDescriptor()
    {
        if constexpr (std::is_same<typename ThreadwiseGemm_Dispatch::MatrixBLayout, ck::tensor_layout::gemm::RowMajor>::value){
            // B : K, N
            constexpr auto b_thread_desc_k_n = make_naive_tensor_descriptor_packed(make_tuple(KPerBlock, ThreadwiseGemm_Dispatch::ThreadMaxNr));
            return b_thread_desc_k_n;
        } else {
            // B : N/8, K, N8
            constexpr auto b_thread_desc_n_k_n8 = make_naive_tensor_descriptor_packed(make_tuple(math::integer_divide_ceil(ThreadwiseGemm_Dispatch::ThreadMaxNr, ThreadwiseGemm_Dispatch::MatrixBMinVectorSize), KPerBlock, ThreadwiseGemm_Dispatch::MatrixBMinVectorSize));
            return b_thread_desc_n_k_n8;
        }
    }
#endif

    static constexpr auto GetAThreadSliceLength()
    {
        if constexpr(std::is_same<typename ThreadwiseGemm_Dispatch::MatrixALayout,
                                  ck::tensor_layout::gemm::RowMajor>::value)
        {
            // A : M, K
            return ck::Sequence<ThreadwiseGemm_Dispatch::ThreadMaxMr, KPerBlock>{};
        }
        else
        {
            // A : K, M
            return ck::Sequence<KPerBlock, ThreadwiseGemm_Dispatch::ThreadMaxMr>{};
        }
    }

    static constexpr auto GetBThreadSliceLength()
    {
        if constexpr(std::is_same<typename ThreadwiseGemm_Dispatch::MatrixBLayout,
                                  ck::tensor_layout::gemm::RowMajor>::value)
        {
            // B : K, N
            return ck::Sequence<KPerBlock, ThreadwiseGemm_Dispatch::ThreadMaxNr>{};
        }
        else
        {
            // B : N/8, K, N88;
            return ck::Sequence<ThreadwiseGemm_Dispatch::ThreadMaxNr /
                                    ThreadwiseGemm_Dispatch::MatrixBMinVectorSize,
                                KPerBlock,
                                ThreadwiseGemm_Dispatch::MatrixBMinVectorSize>{};
        }
    }

    static constexpr auto GetAThreadMoveFwdStep()
    {
        if constexpr(std::is_same<typename ThreadwiseGemm_Dispatch::MatrixALayout,
                                  ck::tensor_layout::gemm::RowMajor>::value)
        {
            // A : M, K
            return ck::make_multi_index(ThreadwiseGemm_Dispatch::ThreadMaxMr, 0);
        }
        else
        {
            // A : K, M
            return ck::make_multi_index(0, ThreadwiseGemm_Dispatch::ThreadMaxMr);
        }
    }

    static constexpr auto GetBThreadMoveFwdStep()
    {
        if constexpr(std::is_same<typename ThreadwiseGemm_Dispatch::MatrixBLayout,
                                  ck::tensor_layout::gemm::RowMajor>::value)
        {
            // B : K, N
            return ck::make_multi_index(0, ThreadwiseGemm_Dispatch::ThreadMaxNr);
        }
        else
        {
            // B : N/8, K, N88;
            return ck::Sequence<ThreadwiseGemm_Dispatch::ThreadMaxNr /
                                    ThreadwiseGemm_Dispatch::MatrixBMinVectorSize,
                                0,
                                0>{};
        }
    }

    static constexpr ck::index_t GetAThreadLoopOverDim()
    {
        if constexpr(std::is_same<typename ThreadwiseGemm_Dispatch::MatrixALayout,
                                  ck::tensor_layout::gemm::RowMajor>::value)
        {
            // A : M, K
            return 0;
        }
        else
        {
            // A : K, M
            return 1;
        }
    }

    static constexpr ck::index_t GetBThreadLoopOverDim()
    {
        if constexpr(std::is_same<typename ThreadwiseGemm_Dispatch::MatrixBLayout,
                                  ck::tensor_layout::gemm::RowMajor>::value)
        {
            // B : K, N
            return 1;
        }
        else
        {
            // B : N/8, K, N88;
            return 0;
        }
    }

    static constexpr auto GetCBlockDescriptor()
    {
        if constexpr(UseCLocalBuffer)
        {
            return make_naive_tensor_descriptor_packed(make_tuple(MPerBlock, NPerBlock));
        }
        else
        {
            return make_naive_tensor_descriptor_packed(make_tuple(MPerBlock, NPerBlock)); // TODO:
        }
    }

    static constexpr auto GetCBlockSliceLength() { return ck::Sequence<MPerBlock, NPerBlock>{}; }

    static constexpr bool CheckValidity(const AGridDesc& a_grid_desc,
                                        const BGridDesc& b_grid_desc,
                                        const CGridDesc& c_grid_desc)
    {
#if 0
        const auto M  = a_grid_desc_k0_m_k1.GetLength(I1);
        const auto N  = b_grid_desc_k0_n_k1.GetLength(I1);
        const auto K0 = a_grid_desc_k0_m_k1.GetLength(I0);

        if(!(M == c_grid_desc_m_n.GetLength(I0) && N == c_grid_desc_m_n.GetLength(I1) &&
             K0 == b_grid_desc_k0_n_k1.GetLength(I0) && K1 == a_grid_desc_k0_m_k1.GetLength(I2) &&
             K1 == b_grid_desc_k0_n_k1.GetLength(I2)))
            return false;

        if(!(M % MPerBlock == 0 && N % NPerBlock == 0 && K0 % K0PerBlock == 0))
            return false;

        // check NumPrefetch
        if constexpr(NumPrefetch == 1)
        {
            // 1-stage prefetch always supported
        }
        else if constexpr(NumPrefetch == 2)
        {
            // 2-stage prefetch currently only support even number of K0 loop
            // TODO: add support for odd number of K0 loop
            if(!((K0 / K0PerBlock) % 2 == 0))
            {
                return false;
            }
        }
        else
        {
            return false;
        }

        // check M01, N01
        constexpr auto M1 = Number<MPerBlock>{};
        constexpr auto N1 = Number<NPerBlock>{};

        const auto M0 = M / M1;
        const auto N0 = N / N1;

        if(!(M0 % M01 == 0 && N0 % N01 == 0))
            return false;
#endif
        // TODO: also check validity of all components (blockwise-copy, threadwise-copy, etc)
        return true;
    }

    static void Run(const FloatA* __restrict__ p_a_grid,
                    const FloatB* __restrict__ p_b_grid,
                    FloatC* __restrict__ p_c_grid,
                    const AGridDesc& a_grid_desc,
                    const BGridDesc& b_grid_desc,
                    const CGridDesc& c_grid_desc,
                    const AElementwiseOperation& a_element_op,
                    const BElementwiseOperation& b_element_op,
                    const CElementwiseOperation& c_element_op)
    {
        ck::index_t m_per_block;
        ck::index_t n_per_block;
        ck::index_t k_per_block;

        if constexpr(MPerBlock == 0 && NPerBlock == 0 && KPerBlock == 0) {}
        else
        {
            m_per_block = MPerBlock;
            n_per_block = NPerBlock;
            k_per_block = KPerBlock;
        }

        const auto M = a_grid_desc.GetLength(I0);
        const auto N = b_grid_desc.GetLength(I1);
        const auto K = b_grid_desc.GetLength(I0);

        const ck::index_t grid_m = math::integer_divide_ceil(M, m_per_block);
        const ck::index_t grid_n = math::integer_divide_ceil(N, n_per_block);

        const ck::index_t grid_size = grid_m * grid_n;

        constexpr auto a_block_desc           = GetABlockDescriptor();
        constexpr auto a_block_slice_length   = GetABlockSliceLength();
        constexpr auto a_block_copy_dim       = decltype(a_block_slice_length)::Size();
        constexpr auto a_dim_access_order     = GetABlockDimAccessOrder();
        constexpr auto a_block_move_step      = GetABlockMoveFwdStep();
        constexpr auto a_thread_slice_length  = GetAThreadSliceLength();
        constexpr auto a_thread_loop_over_dim = GetAThreadLoopOverDim();

        constexpr auto b_block_desc           = GetBBlockDescriptor();
        constexpr auto b_block_slice_length   = GetBBlockSliceLength();
        constexpr auto b_block_copy_dim       = decltype(b_block_slice_length)::Size();
        constexpr auto b_dim_access_order     = GetBBlockDimAccessOrder();
        constexpr auto b_block_move_step      = GetBBlockMoveFwdStep();
        constexpr auto b_thread_slice_length  = GetBThreadSliceLength();
        constexpr auto b_thread_loop_over_dim = GetBThreadLoopOverDim();

        constexpr auto c_block_desc         = GetCBlockDescriptor();
        constexpr auto c_block_slice_length = GetCBlockSliceLength();
        constexpr auto c_block_move_step    = ck::make_multi_index(0, NPerBlock);

        auto a_threadwise_copy = ck::cpu::ThreadwiseTensorSliceTransferAvx2<
            FloatA,                               // SrcData
            FloatA,                               // DstData
            decltype(a_grid_desc),                // SrcDesc
            decltype(a_block_desc),               // DstDesc
            AElementwiseOperation,                // ElementwiseOperation
            decltype(a_block_slice_length),       // SliceLengths
            decltype(a_dim_access_order),         // DimAccessOrder
            1,                                    // VectorDim
            1,                                    // ScalarPerVector
            ck::InMemoryDataOperationEnum_t::Set, // InMemoryDataOperationEnum_t
            false,                                // SrcResetCoordinateAfterRun
            true                                  // DstResetCoordinateAfterRun
            >(a_grid_desc,
              ck::make_zero_multi_index<a_block_copy_dim>(),
              a_block_desc,
              ck::make_zero_multi_index<a_block_copy_dim>(),
              AElementwiseOperation{});

        auto b_threadwise_copy = ck::cpu::ThreadwiseTensorSliceTransferAvx2<
            FloatB,                               // SrcData
            FloatB,                               // DstData
            decltype(b_grid_desc),                // SrcDesc
            decltype(b_block_desc),               // DstDesc
            BElementwiseOperation,                // ElementwiseOperation
            decltype(b_block_slice_length),       // SliceLengths
            decltype(b_dim_access_order),         // DimAccessOrder
            1,                                    // VectorDim
            1,                                    // ScalarPerVector
            ck::InMemoryDataOperationEnum_t::Set, // InMemoryDataOperationEnum_t
            false,                                // SrcResetCoordinateAfterRun
            true                                  // DstResetCoordinateAfterRun
            >(b_grid_desc,
              ck::make_zero_multi_index<b_block_copy_dim>(),
              b_block_desc,
              ck::make_zero_multi_index<b_block_copy_dim>(),
              BElementwiseOperation{});

        auto c_threadwise_copy = ck::cpu::ThreadwiseTensorSliceTransferAvx2<
            FloatC,                               // SrcData
            FloatC,                               // DstData
            decltype(c_block_desc),               // SrcDesc
            decltype(c_grid_desc),                // DstDesc
            BElementwiseOperation,                // ElementwiseOperation
            ck::Sequence<MPerBlock, NPerBlock>,   // SliceLengths
            ck::Sequence<0, 1>,                   // DimAccessOrder
            1,                                    // VectorDim
            1,                                    // ScalarPerVector
            ck::InMemoryDataOperationEnum_t::Set, // InMemoryDataOperationEnum_t
            true,                                 // SrcResetCoordinateAfterRun
            false                                 // DstResetCoordinateAfterRun
            >(c_block_desc,
              ck::make_zero_multi_index<2>(),
              c_grid_desc,
              ck::make_zero_multi_index<2>(),
              CElementwiseOperation{});

        DeviceAlignedMemCPU a_block_mem(MPerBlock * KPerBlock * sizeof(FloatA), MemAlignmentByte);
        DeviceAlignedMemCPU b_block_mem(KPerBlock * NPerBlock * sizeof(FloatB), MemAlignmentByte);
        DeviceAlignedMemCPU c_block_mem(MPerBlock * NPerBlock * sizeof(FloatC), MemAlignmentByte);

        auto a_grid_buf = ck::cpu::make_dynamic_buffer<ck::AddressSpaceEnum_t::Global>(
            reinterpret_cast<const FloatA*>(p_a_grid), a_grid_desc.GetElementSpaceSize());

        auto b_grid_buf = ck::cpu::make_dynamic_buffer<ck::AddressSpaceEnum_t::Global>(
            reinterpret_cast<const FloatB*>(p_b_grid), b_grid_desc.GetElementSpaceSize());

        auto c_grid_buf = ck::cpu::make_dynamic_buffer<ck::AddressSpaceEnum_t::Global>(
            reinterpret_cast<FloatC*>(p_c_grid), c_grid_desc.GetElementSpaceSize());

        auto a_block_buf = ck::cpu::make_dynamic_buffer<ck::AddressSpaceEnum_t::Global>(
            reinterpret_cast<FloatA*>(a_block_mem.mpDeviceBuf),
            a_block_mem.mMemSize / sizeof(FloatA));

        auto b_block_buf = ck::cpu::make_dynamic_buffer<ck::AddressSpaceEnum_t::Global>(
            reinterpret_cast<FloatB*>(b_block_mem.mpDeviceBuf),
            b_block_mem.mMemSize / sizeof(FloatB));

        auto c_block_buf = ck::cpu::make_dynamic_buffer<ck::AddressSpaceEnum_t::Global>(
            reinterpret_cast<FloatC*>(c_block_mem.mpDeviceBuf),
            c_block_mem.mMemSize / sizeof(FloatC));

        auto blockwise_gemm =
            BlockwiseGemmAvx2_MxN<FloatA,                          // FloatA,
                                  FloatB,                          // FloatB,
                                  FloatC,                          // FloatC,
                                  AccDataType,                     // AccDataType,
                                  decltype(a_block_desc),          // ABlockDesc,
                                  decltype(b_block_desc),          // BBlockDesc,
                                  decltype(c_block_desc),          // CBlockDesc,
                                  decltype(a_block_slice_length),  // ABlockSliceLengths,
                                  decltype(b_block_slice_length),  // BBlockSliceLengths,
                                  decltype(c_block_slice_length),  // CBlockSliceLengths,
                                  decltype(a_thread_slice_length), // AThreadSliceLength,
                                  decltype(b_thread_slice_length), // BThreadSliceLength,
                                  a_thread_loop_over_dim,  // AThreadLoopOverDim,   // thread slice
                                                           // loop over on block slice. 1d is enough
                                                           // for now
                                  b_thread_loop_over_dim,  // BThreadLoopOverDim,
                                  KPerBlock,               // KPerBlock,
                                  ThreadwiseGemm_Dispatch, // ThreadwiseGemm_Dispatch,
                                  ThreadMNAccessOrder>{};  // ThreadMNAccessOrder  // how we acces
                                                           // gemm MN to utilize micro kernel>{};

        // TODO: openmp aware ordering
        if constexpr(std::is_same<BlockMNKAccessOrder, ck::Sequence<0, 1, 2>>::value)
        {
#pragma omp parallel for
            for(ck::index_t gid = 0; gid < grid_size; gid++)
            {
                ck::index_t i_mc = (gid / grid_n) * m_per_block;
                ck::index_t i_nc = (gid % grid_n) * n_per_block;

                ck::index_t mc_size = ck::math::min(M - i_mc, m_per_block);
                ck::index_t nc_size = ck::math::min(N - i_nc, n_per_block);

                // pack_b
                b_threadwise_copy.RunGeneric(b_grid_desc, b_grid_buf, b_block_desc, b_block_buf);
                b_threadwise_copy.MoveSrcSliceWindow(b_grid_desc, b_block_move_step);
                if(i_nc == 0)
                {
                    // pack_a
                    a_threadwise_copy.RunGeneric(
                        a_grid_desc, a_grid_buf, a_block_desc, a_block_buf);
                    a_threadwise_copy.MoveSrcSliceWindow(a_grid_desc, a_block_move_step);
                }

                for(ck::index_t i_kc = 0; i_kc < K; i_kc += k_per_block)
                {
                    ck::index_t kc_size = ck::math::min(K - i_kc, k_per_block);

                    blockwise_gemm.Run(a_block_desc,
                                       a_block_buf,
                                       make_zero_multi_index<a_block_copy_dim>(),
                                       b_block_desc,
                                       b_block_buf,
                                       make_zero_multi_index<b_block_copy_dim>(),
                                       c_block_desc,
                                       c_block_buf,
                                       make_zero_multi_index<2>());
                }

                if constexpr(UseCLocalBuffer)
                {
                    c_threadwise_copy.RunGeneric(
                        c_block_desc, c_block_buf, c_grid_desc, c_grid_buf);
                    c_threadwise_copy.MoveDstSliceWindow(c_grid_desc, c_block_move_step);
                }
            }
        }
    }
};

} // namespace cpu
} // namespace ck

#endif
