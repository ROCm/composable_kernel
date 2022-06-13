#ifndef CK_GRIDWISE_GEMM_AVX2_HPP
#define CK_GRIDWISE_GEMM_AVX2_HPP

#include "common_header.hpp"
#include "multi_index_transform_helper.hpp"
#include "tensor_descriptor.hpp"
#include "tensor_descriptor_helper.hpp"
#include "blockwise_gemm_avx2.hpp"
#include "threadwise_tensor_slice_transfer_avx2.hpp"
#include "threadwise_tensor_slice_transfer_avx2_specialization.hpp"
#include "dynamic_buffer_cpu.hpp"
#include "envvar.hpp"
#include <utility>
#include <unistd.h>
#include <omp.h>
#include <pthread.h>

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
void kernel_gemm_avx_mxn(const GridwiseGemm& gridwise_gemm,
                         const FloatA* __restrict__ p_a_grid,
                         const FloatB* __restrict__ p_b_grid,
                         FloatC* __restrict__ p_c_grid,
                         const AGridDesc& a_grid_desc,
                         const BGridDesc& b_grid_desc,
                         const CGridDesc& c_grid_desc,
                         const AElementwiseOperation& a_element_op,
                         const BElementwiseOperation& b_element_op,
                         const CElementwiseOperation& c_element_op)
{
    gridwise_gemm.Run(p_a_grid,
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
          typename AGridDesc,
          typename BGridDesc,
          typename CGridDesc,
          typename AElementwiseOperation,
          typename BElementwiseOperation,
          typename CElementwiseOperation,
          typename ThreadwiseGemm_Dispatch,
          typename AThreadwiseCopy,
          typename BThreadwiseCopy,
          typename CThreadwiseCopy,
          typename ThreadMNAccessOrder, // how we acces gemm MN to utilize micro kernel
          bool UseALocalBuffer,
          bool UseBLocalBuffer,
          bool UseCLocalBuffer // if true, will allocate a buffer and write to it in kernel, then
                               // copy back to block buffer (need CThreadwiseCopy).
                               // if false, will write to C directly (no need CThreadwiseCopy)
          >
struct GridwiseGemmAvx2_MxN
{
    ck::tensor_operation::cpu::device::DeviceConvFwdDynamicTunable dynamic_tunable;
    static constexpr auto I0 = Number<0>{};
    static constexpr auto I1 = Number<1>{};

    // static constexpr auto Avx2RegisterVector = 8;   // 8 floats
    static constexpr index_t MemAlignmentByte = 32; // 256bit

    GridwiseGemmAvx2_MxN(
        const ck::tensor_operation::cpu::device::DeviceConvFwdDynamicTunable dynamic_tunable_)
        : dynamic_tunable(dynamic_tunable_)
    {
    }

    static auto GetABlockDescriptor(const ck::index_t m_per_blk,
                                    const ck::index_t k_per_blk,
                                    const AGridDesc& a_grid_desc)
    {
        if constexpr(UseALocalBuffer)
        {
            if constexpr(std::is_same<typename ThreadwiseGemm_Dispatch::MatrixALayout,
                                      ck::tensor_layout::gemm::RowMajor>::value)
            {
                // A : M, K
                auto a_block_desc_m_k =
                    make_naive_tensor_descriptor_packed(make_tuple(m_per_blk, k_per_blk));
                return a_block_desc_m_k;
            }
            else
            {
                // A : K, M
                auto a_block_desc_k_m = make_naive_tensor_descriptor_packed(
                    make_tuple(k_per_blk,
                               math::integer_least_multiple(
                                   m_per_blk, ThreadwiseGemm_Dispatch::MatrixAMinVectorSize)));
                return a_block_desc_k_m;
            }
        }
        else
        {
            return a_grid_desc;
        }
    }

    static auto GetBBlockDescriptor(const ck::index_t k_per_blk,
                                    const ck::index_t n_per_blk,
                                    const BGridDesc& b_grid_desc)
    {
        if constexpr(UseBLocalBuffer)
        {
            // n_per_blk should be 8x
            if constexpr(std::is_same<typename ThreadwiseGemm_Dispatch::MatrixBLayout,
                                      ck::tensor_layout::gemm::RowMajor>::value)
            {
                // B : K, N
                auto b_block_desc_k_n =
                    make_naive_tensor_descriptor_packed(make_tuple(k_per_blk, n_per_blk));
                return b_block_desc_k_n;
            }
            else
            {
                // B : N/8, K, N8
                auto b_block_desc_n0_k_n1 = make_naive_tensor_descriptor_packed(
                    make_tuple(math::integer_divide_ceil(
                                   n_per_blk, ThreadwiseGemm_Dispatch::MatrixBMinVectorSize),
                               k_per_blk,
                               ThreadwiseGemm_Dispatch::MatrixBMinVectorSize));
                return b_block_desc_n0_k_n1;
            }
        }
        else
        {
            return b_grid_desc;
        }
    }

    static auto GetCBlockDescriptor(const ck::index_t m_per_blk,
                                    const ck::index_t n_per_blk,
                                    const CGridDesc& c_grid_desc)
    {
        if constexpr(UseCLocalBuffer)
        {
            return make_naive_tensor_descriptor_packed(make_tuple(m_per_blk, n_per_blk));
        }
        else
            return c_grid_desc;
    }

    static auto GetASliceLength(const ck::index_t m_per_blk, const ck::index_t k_per_blk)
    {
        if constexpr(std::is_same<typename ThreadwiseGemm_Dispatch::MatrixALayout,
                                  ck::tensor_layout::gemm::RowMajor>::value)
        {
            // A : M, K
            return ck::make_multi_index(m_per_blk, k_per_blk);
        }
        else
        {
            // A : K, M
            return ck::make_multi_index(
                k_per_blk,
                math::integer_least_multiple(m_per_blk,
                                             ThreadwiseGemm_Dispatch::MatrixAMinVectorSize));
        }
    }

    static auto GetBSliceLength(const ck::index_t k_per_blk, const ck::index_t n_per_blk)
    {
        // n_per_blk should be 8x
        if constexpr(std::is_same<typename ThreadwiseGemm_Dispatch::MatrixBLayout,
                                  ck::tensor_layout::gemm::RowMajor>::value)
        {
            // B : K, N
            return ck::make_multi_index(
                k_per_blk,
                math::integer_least_multiple(n_per_blk,
                                             ThreadwiseGemm_Dispatch::MatrixBMinVectorSize));
        }
        else
        {
            // B : N/8, K, N8
            return ck::make_multi_index(
                math::integer_divide_ceil(n_per_blk, ThreadwiseGemm_Dispatch::MatrixBMinVectorSize),
                k_per_blk,
                ThreadwiseGemm_Dispatch::MatrixBMinVectorSize);
        }
    }

    static auto GetCSliceLength(const ck::index_t m_per_blk, const ck::index_t n_per_blk)
    {
        return ck::make_multi_index(m_per_blk, n_per_blk);
    }

    static auto GetAIndex(const ck::index_t i_m, const ck::index_t i_k)
    {
        if constexpr(std::is_same<typename ThreadwiseGemm_Dispatch::MatrixALayout,
                                  ck::tensor_layout::gemm::RowMajor>::value)
        {
            // A : M, K
            return ck::make_multi_index(i_m, i_k);
        }
        else
        {
            // A : K, M
            return ck::make_multi_index(i_k, i_m);
        }
    }

    static auto GetBIndex(const ck::index_t i_k, const ck::index_t i_n)
    {
        // i_n should be 8x
        if constexpr(std::is_same<typename ThreadwiseGemm_Dispatch::MatrixBLayout,
                                  ck::tensor_layout::gemm::RowMajor>::value)
        {
            // B : K, N
            return ck::make_multi_index(i_k, i_n);
        }
        else
        {
            // B : N/8, K, N8
            return ck::make_multi_index(i_n / ThreadwiseGemm_Dispatch::MatrixBMinVectorSize,
                                        i_k,
                                        i_n % ThreadwiseGemm_Dispatch::MatrixBMinVectorSize);
        }
    }

    static auto GetCIndex(const ck::index_t i_m, const ck::index_t i_n)
    {
        return ck::make_multi_index(i_m, i_n);
    }

    bool CheckValidity(const AGridDesc& a_grid_desc,
                       const BGridDesc& b_grid_desc,
                       const CGridDesc& c_grid_desc)
    {
        // TODO: also check validity of all components (blockwise-copy, threadwise-copy, etc)
        bool is_valid    = true;
        const auto GemmN = c_grid_desc.GetLength(I1);
        if constexpr(UseCLocalBuffer)
        {
            // if(std::is_same<BlockMNKAccessOrder, ck::Sequence<0, 2, 1>>::value &&
            // dynamic_tunable.gemm_n_per_block < GemmN)
            if(dynamic_tunable.loop_over_spec ==
                   ck::tensor_operation::cpu::device::
                       ConvolutionForwardBlockLoopOverSpecialization_t::LoopOver_MKN &&
               dynamic_tunable.n_per_block < GemmN)
                is_valid &= false;
        }
        else
        {
            // TODO: need check c grid is simple transform?
            if(GemmN % 8 != 0)
                is_valid &= false;
        }
        return is_valid;
    }

    void Run(const FloatA* __restrict__ p_a_grid,
             const FloatB* __restrict__ p_b_grid,
             FloatC* __restrict__ p_c_grid,
             const AGridDesc& a_grid_desc,
             const BGridDesc& b_grid_desc,
             const CGridDesc& c_grid_desc,
             const AElementwiseOperation& a_element_op,
             const BElementwiseOperation& b_element_op,
             const CElementwiseOperation& c_element_op) const
    {
        ck::index_t m_per_block = dynamic_tunable.m_per_block;
        ck::index_t n_per_block = dynamic_tunable.n_per_block;
        ck::index_t k_per_block = dynamic_tunable.k_per_block;

        const auto GemmM = c_grid_desc.GetLength(I0);
        const auto GemmN = c_grid_desc.GetLength(I1);
        const auto GemmK = a_grid_desc.GetLength(I1);

        constexpr auto a_block_copy_dim = AGridDesc::GetNumOfDimension();

        constexpr auto b_block_copy_dim = BGridDesc::GetNumOfDimension();

        auto a_grid_buf = ck::cpu::make_dynamic_buffer<ck::AddressSpaceEnum::Global>(
            const_cast<FloatA*>(p_a_grid), a_grid_desc.GetElementSpaceSize());

        auto b_grid_buf = ck::cpu::make_dynamic_buffer<ck::AddressSpaceEnum::Global>(
            const_cast<FloatB*>(p_b_grid), b_grid_desc.GetElementSpaceSize());

        auto c_grid_buf = ck::cpu::make_dynamic_buffer<ck::AddressSpaceEnum::Global>(
            reinterpret_cast<FloatC*>(p_c_grid), c_grid_desc.GetElementSpaceSize());

        auto blockwise_gemm = BlockwiseGemmAvx2_MxN<
            FloatA,                                                               // FloatA,
            FloatB,                                                               // FloatB,
            FloatC,                                                               // FloatC,
            decltype(GetABlockDescriptor(m_per_block, k_per_block, a_grid_desc)), // ABlockDesc,
            decltype(GetBBlockDescriptor(k_per_block, n_per_block, b_grid_desc)), // BBlockDesc,
            decltype(GetCBlockDescriptor(m_per_block, n_per_block, c_grid_desc)), // CBlockDesc,
            ThreadwiseGemm_Dispatch, // ThreadwiseGemm_Dispatch,
            ThreadMNAccessOrder>{};  // ThreadMNAccessOrder  // how we acces
                                     // gemm MN to utilize micro kernel>{};

        int total_threads = omp_get_max_threads();

        if(total_threads > 1 && ck::getenv_int("CK_CPU_BIND_CORE", 0) != 0)
        {
#pragma omp parallel
            {
                int tid = omp_get_thread_num();
                cpu_set_t set;
                CPU_ZERO(&set);

                CPU_SET(tid, &set);

                if(sched_setaffinity(0, sizeof(set), &set) == -1)
                {
                    throw std::runtime_error("wrong! fail to set thread affinity");
                }
            }
        }

        // TODO: openmp aware ordering
        //
        if(dynamic_tunable.loop_over_spec ==
           ck::tensor_operation::cpu::device::ConvolutionForwardBlockLoopOverSpecialization_t::
               LoopOver_MNK)
        {
            auto a_move_k_step = GetAIndex(0, k_per_block);
            auto b_move_k_step = GetBIndex(k_per_block, 0);

            const ck::index_t grid_m    = math::integer_divide_ceil(GemmM, m_per_block);
            const ck::index_t grid_n    = math::integer_divide_ceil(GemmN, n_per_block);
            const ck::index_t grid_size = grid_m * grid_n;
            const ck::index_t grids_per_thread =
                math::integer_divide_ceil(grid_size, total_threads);

// This version does not consider K panel re-usage. simple for openmp
#pragma omp parallel
            {
                auto a_threadwise_copy =
                    AThreadwiseCopy(a_grid_desc,
                                    ck::make_zero_multi_index<a_block_copy_dim>(),
                                    GetABlockDescriptor(m_per_block, k_per_block, a_grid_desc),
                                    ck::make_zero_multi_index<a_block_copy_dim>(),
                                    AElementwiseOperation{});

                auto b_threadwise_copy =
                    BThreadwiseCopy(b_grid_desc,
                                    ck::make_zero_multi_index<b_block_copy_dim>(),
                                    GetBBlockDescriptor(k_per_block, n_per_block, b_grid_desc),
                                    ck::make_zero_multi_index<b_block_copy_dim>(),
                                    BElementwiseOperation{});

                auto c_threadwise_copy =
                    CThreadwiseCopy(GetCBlockDescriptor(m_per_block, n_per_block, c_grid_desc),
                                    ck::make_zero_multi_index<2>(),
                                    c_grid_desc,
                                    ck::make_zero_multi_index<2>(),
                                    CElementwiseOperation{});

                DeviceAlignedMemCPU a_block_mem(
                    UseALocalBuffer ? m_per_block * k_per_block * sizeof(FloatA) : 0,
                    MemAlignmentByte);
                DeviceAlignedMemCPU b_block_mem(
                    UseBLocalBuffer ? k_per_block * n_per_block * sizeof(FloatB) : 0,
                    MemAlignmentByte);
                DeviceAlignedMemCPU c_block_mem(
                    UseCLocalBuffer ? (m_per_block * n_per_block * sizeof(FloatC)) : 0,
                    MemAlignmentByte);

                auto a_block_buf = ck::cpu::make_dynamic_buffer<ck::AddressSpaceEnum::Global>(
                    UseALocalBuffer ? reinterpret_cast<FloatA*>(a_block_mem.mpDeviceBuf)
                                    : const_cast<FloatA*>(p_a_grid),
                    UseALocalBuffer ? a_block_mem.mMemSize / sizeof(FloatA)
                                    : a_grid_desc.GetElementSpaceSize());

                auto b_block_buf = ck::cpu::make_dynamic_buffer<ck::AddressSpaceEnum::Global>(
                    UseBLocalBuffer ? reinterpret_cast<FloatB*>(b_block_mem.mpDeviceBuf)
                                    : const_cast<FloatB*>(p_b_grid),
                    UseBLocalBuffer ? b_block_mem.mMemSize / sizeof(FloatB)
                                    : b_grid_desc.GetElementSpaceSize());

                auto c_block_buf = ck::cpu::make_dynamic_buffer<ck::AddressSpaceEnum::Global>(
                    UseCLocalBuffer ? reinterpret_cast<FloatC*>(c_block_mem.mpDeviceBuf)
                                    : reinterpret_cast<FloatC*>(p_c_grid),
                    UseCLocalBuffer ? c_block_mem.mMemSize / sizeof(FloatC)
                                    : c_grid_desc.GetElementSpaceSize());

                const ck::index_t tid = omp_get_thread_num();

                for(ck::index_t i_gpt = 0; i_gpt < grids_per_thread; i_gpt++)
                {
                    ck::index_t gid = i_gpt * total_threads + tid;
                    if(gid >= grid_size)
                        break;

                    ck::index_t i_mc = (gid / grid_n) * m_per_block;
                    ck::index_t i_nc = (gid % grid_n) * n_per_block;

                    ck::index_t mc_size = ck::math::min(GemmM - i_mc, m_per_block);
                    ck::index_t nc_size =
                        ck::math::min(GemmN - i_nc, n_per_block); // TODO: nc need be 8x
                    nc_size = math::integer_least_multiple(
                        nc_size, ThreadwiseGemm_Dispatch::MatrixBMinVectorSize);

                    a_threadwise_copy.SetSrcSliceOrigin(a_grid_desc, GetAIndex(i_mc, 0));
                    b_threadwise_copy.SetSrcSliceOrigin(b_grid_desc, GetBIndex(0, i_nc));

                    auto c_block_desc = GetCBlockDescriptor(mc_size, nc_size, c_grid_desc);
                    if constexpr(!UseCLocalBuffer)
                    {
                        c_threadwise_copy.SetSrcSliceOrigin(c_block_desc, GetCIndex(i_mc, i_nc));
                        c_threadwise_copy.RunRead(c_grid_desc,
                                                  c_grid_buf,
                                                  c_block_desc,
                                                  c_block_buf,
                                                  GetCSliceLength(mc_size, nc_size));
                    }

                    for(ck::index_t i_kc = 0; i_kc < GemmK; i_kc += k_per_block)
                    {
                        ck::index_t kc_size = ck::math::min(GemmK - i_kc, k_per_block);

                        auto a_block_desc = GetABlockDescriptor(mc_size, kc_size, a_grid_desc);
                        auto b_block_desc = GetBBlockDescriptor(kc_size, nc_size, b_grid_desc);

                        a_threadwise_copy.RunRead(a_grid_desc,
                                                  a_grid_buf,
                                                  a_block_desc,
                                                  a_block_buf,
                                                  GetASliceLength(mc_size, kc_size));
                        b_threadwise_copy.RunRead(b_grid_desc,
                                                  b_grid_buf,
                                                  b_block_desc,
                                                  b_block_buf,
                                                  GetBSliceLength(kc_size, nc_size));

                        blockwise_gemm.Run(a_block_desc,
                                           a_block_buf,
                                           make_zero_multi_index<a_block_copy_dim>(),
                                           GetASliceLength(mc_size, kc_size),

                                           b_block_desc,
                                           b_block_buf,
                                           make_zero_multi_index<b_block_copy_dim>(),
                                           GetBSliceLength(kc_size, nc_size),

                                           c_block_desc,
                                           c_block_buf,
                                           make_zero_multi_index<2>(),
                                           GetCSliceLength(mc_size, nc_size),
                                           i_kc != 0);

                        if((i_kc + k_per_block) < GemmK)
                        {
                            a_threadwise_copy.MoveSrcSliceWindow(a_grid_desc, a_move_k_step);
                            b_threadwise_copy.MoveSrcSliceWindow(b_grid_desc, b_move_k_step);
                        }
                    }

                    c_threadwise_copy.SetDstSliceOrigin(c_grid_desc, GetCIndex(i_mc, i_nc));
                    c_threadwise_copy.RunWrite(c_block_desc,
                                               c_block_buf,
                                               c_grid_desc,
                                               c_grid_buf,
                                               GetCSliceLength(mc_size, nc_size));
                }
            }
        }
        else if(dynamic_tunable.loop_over_spec ==
                ck::tensor_operation::cpu::device::ConvolutionForwardBlockLoopOverSpecialization_t::
                    LoopOver_MKN)
        {
            auto a_move_k_step = GetAIndex(0, k_per_block);
            auto b_move_k_step = GetBIndex(0, n_per_block);

            const ck::index_t grid_m            = math::integer_divide_ceil(GemmM, m_per_block);
            const ck::index_t grid_m_per_thread = math::integer_divide_ceil(grid_m, total_threads);

// only parallel in gemm m dim
#pragma omp parallel
            {
                auto a_threadwise_copy =
                    AThreadwiseCopy(a_grid_desc,
                                    ck::make_zero_multi_index<a_block_copy_dim>(),
                                    GetABlockDescriptor(m_per_block, k_per_block, a_grid_desc),
                                    ck::make_zero_multi_index<a_block_copy_dim>(),
                                    AElementwiseOperation{});

                auto b_threadwise_copy =
                    BThreadwiseCopy(b_grid_desc,
                                    ck::make_zero_multi_index<b_block_copy_dim>(),
                                    GetBBlockDescriptor(k_per_block, n_per_block, b_grid_desc),
                                    ck::make_zero_multi_index<b_block_copy_dim>(),
                                    BElementwiseOperation{});

                auto c_threadwise_copy =
                    CThreadwiseCopy(GetCBlockDescriptor(m_per_block, n_per_block, c_grid_desc),
                                    ck::make_zero_multi_index<2>(),
                                    c_grid_desc,
                                    ck::make_zero_multi_index<2>(),
                                    CElementwiseOperation{});

                DeviceAlignedMemCPU a_block_mem(
                    UseALocalBuffer ? m_per_block * k_per_block * sizeof(FloatA) : 0,
                    MemAlignmentByte);
                DeviceAlignedMemCPU b_block_mem(
                    UseBLocalBuffer ? k_per_block * n_per_block * sizeof(FloatB) : 0,
                    MemAlignmentByte);
                DeviceAlignedMemCPU c_block_mem(
                    UseCLocalBuffer ? (m_per_block * n_per_block * sizeof(FloatC)) : 0,
                    MemAlignmentByte);

                auto a_block_buf = ck::cpu::make_dynamic_buffer<ck::AddressSpaceEnum::Global>(
                    UseALocalBuffer ? reinterpret_cast<FloatA*>(a_block_mem.mpDeviceBuf)
                                    : const_cast<FloatA*>(p_a_grid),
                    UseALocalBuffer ? a_block_mem.mMemSize / sizeof(FloatA)
                                    : a_grid_desc.GetElementSpaceSize());

                auto b_block_buf = ck::cpu::make_dynamic_buffer<ck::AddressSpaceEnum::Global>(
                    UseBLocalBuffer ? reinterpret_cast<FloatB*>(b_block_mem.mpDeviceBuf)
                                    : const_cast<FloatB*>(p_b_grid),
                    UseBLocalBuffer ? b_block_mem.mMemSize / sizeof(FloatB)
                                    : b_grid_desc.GetElementSpaceSize());

                auto c_block_buf = ck::cpu::make_dynamic_buffer<ck::AddressSpaceEnum::Global>(
                    UseCLocalBuffer ? reinterpret_cast<FloatC*>(c_block_mem.mpDeviceBuf)
                                    : reinterpret_cast<FloatC*>(p_c_grid),
                    UseCLocalBuffer ? c_block_mem.mMemSize / sizeof(FloatC)
                                    : c_grid_desc.GetElementSpaceSize());

                const ck::index_t tid = omp_get_thread_num();

                for(ck::index_t i_gmpt = 0; i_gmpt < grid_m_per_thread; i_gmpt++)
                {
                    ck::index_t i_mc = (i_gmpt * total_threads + tid) * m_per_block;
                    if(i_mc >= GemmM)
                        break;
                    ck::index_t mc_size = ck::math::min(GemmM - i_mc, m_per_block);
                    a_threadwise_copy.SetSrcSliceOrigin(a_grid_desc, GetAIndex(i_mc, 0));
                    for(ck::index_t i_kc = 0; i_kc < GemmK; i_kc += k_per_block)
                    {
                        ck::index_t kc_size = ck::math::min(GemmK - i_kc, k_per_block);

                        auto a_block_desc = GetABlockDescriptor(mc_size, kc_size, a_grid_desc);
                        a_threadwise_copy.RunRead(a_grid_desc,
                                                  a_grid_buf,
                                                  a_block_desc,
                                                  a_block_buf,
                                                  GetASliceLength(mc_size, kc_size));

                        b_threadwise_copy.SetSrcSliceOrigin(b_grid_desc, GetBIndex(i_kc, 0));

                        // TODO: if use local C buffer, then this nc loop need to loop only once
                        for(ck::index_t i_nc = 0; i_nc < GemmN; i_nc += n_per_block)
                        {
                            ck::index_t nc_size =
                                ck::math::min(GemmN - i_nc, n_per_block); // TODO: nc need be 8x
                            nc_size = math::integer_least_multiple(
                                nc_size, ThreadwiseGemm_Dispatch::MatrixBMinVectorSize);
                            auto b_block_desc = GetBBlockDescriptor(kc_size, nc_size, b_grid_desc);

                            b_threadwise_copy.RunRead(b_grid_desc,
                                                      b_grid_buf,
                                                      b_block_desc,
                                                      b_block_buf,
                                                      GetBSliceLength(kc_size, nc_size));

                            auto c_block_desc = GetCBlockDescriptor(mc_size, nc_size, c_grid_desc);

                            if constexpr(!UseCLocalBuffer)
                            {
                                c_threadwise_copy.SetSrcSliceOrigin(c_block_desc,
                                                                    GetCIndex(i_mc, i_nc));
                                c_threadwise_copy.RunRead(c_grid_desc,
                                                          c_grid_buf,
                                                          c_block_desc,
                                                          c_block_buf,
                                                          GetCSliceLength(mc_size, nc_size));
                            }

                            blockwise_gemm.Run(a_block_desc,
                                               a_block_buf,
                                               make_zero_multi_index<a_block_copy_dim>(),
                                               GetASliceLength(mc_size, kc_size),

                                               b_block_desc,
                                               b_block_buf,
                                               make_zero_multi_index<b_block_copy_dim>(),
                                               GetBSliceLength(kc_size, nc_size),

                                               c_block_desc,
                                               c_block_buf,
                                               make_zero_multi_index<2>(),
                                               GetCSliceLength(mc_size, nc_size),

                                               i_kc != 0);

                            if((i_nc + n_per_block) < GemmN)
                            {
                                b_threadwise_copy.MoveSrcSliceWindow(b_grid_desc, b_move_k_step);
                            }

                            if constexpr(UseCLocalBuffer)
                            {
                                c_threadwise_copy.SetDstSliceOrigin(c_grid_desc,
                                                                    GetCIndex(i_mc, i_nc));

                                c_threadwise_copy.RunWrite(c_block_desc,
                                                           c_block_buf,
                                                           c_grid_desc,
                                                           c_grid_buf,
                                                           GetCSliceLength(mc_size, nc_size));
                            }
                            else
                            {
                                // only write for last K, since the RunWrite here is just doing
                                // elementwise op from global to global
                                if((i_kc + k_per_block) >= GemmK)
                                {
                                    c_threadwise_copy.SetDstSliceOrigin(c_grid_desc,
                                                                        GetCIndex(i_mc, i_nc));

                                    c_threadwise_copy.RunWrite(c_block_desc,
                                                               c_block_buf,
                                                               c_grid_desc,
                                                               c_grid_buf,
                                                               GetCSliceLength(mc_size, nc_size));
                                }
                            }
                        }

                        if((i_kc + k_per_block) < GemmK)
                            a_threadwise_copy.MoveSrcSliceWindow(a_grid_desc, a_move_k_step);
                    }
                }
            }
        }
    }
};

} // namespace cpu
} // namespace ck

#endif
