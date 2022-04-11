#pragma once

#include "cluster_descriptor.hpp"
#include "data_type.hpp"
#include "threadwise_tensor_slice_transfer.hpp"

namespace ck {

template <typename GridwiseEltwise,
          typename ADataType,
          typename BDataType,
          typename CDataType,
          typename AGridDesc_M_N,
          typename BGridDesc_M_N,
          typename CGridDesc_M_N,
          typename ElementwiseFunctor>
__global__ void kernel_elementwise_2d(const ADataType* __restrict__ p_a_global,
                                      const BDataType* __restrict__ p_b_global,
                                      CDataType* __restrict__ p_c_global,
                                      const AGridDesc_M_N a_grid_desc_m_k,
                                      const BGridDesc_M_N b_grid_desc_m_k,
                                      const CGridDesc_M_N c_grid_desc_m_k,
                                      const ElementwiseFunctor functor)
{
    GridwiseEltwise::Run(p_a_global,
                         p_b_global,
                         p_c_global,
                         a_grid_desc_m_k,
                         b_grid_desc_m_k,
                         c_grid_desc_m_k,
                         functor);
}

template <typename ADataType,
          typename BDataType,
          typename CDataType,
          typename AGridDesc_M_N,
          typename BGridDesc_M_N,
          typename CGridDesc_M_N,
          typename ElementwiseFunctor,
          index_t MThreadTileSize,
          index_t NThreadTileSize>
struct GridwiseElementwise_2D
{
    __device__ static void Run(const ADataType* __restrict__ p_a_global,
                               const BDataType* __restrict__ p_b_global,
                               CDataType* __restrict__ p_c_global,
                               const AGridDesc_M_N a_grid_desc_m_n,
                               const BGridDesc_M_N b_grid_desc_m_n,
                               const CGridDesc_M_N c_grid_desc_m_n,
                               const ElementwiseFunctor functor)
    {
        // const index_t thread_id = get_thread_local_1d_id();
        // const index_t block_id = get_block_1d_id();
        // printf("block_id = %d, thread_id = %d \n", block_id, thread_id);

        const auto a_global_buf = make_dynamic_buffer<AddressSpaceEnum::Global>(
            p_a_global, a_grid_desc_m_n.GetElementSpaceSize());
        const auto b_global_buf = make_dynamic_buffer<AddressSpaceEnum::Global>(
            p_b_global, b_grid_desc_m_n.GetElementSpaceSize());
        const auto c_global_buf = make_dynamic_buffer<AddressSpaceEnum::Global>(
            p_c_global, c_grid_desc_m_n.GetElementSpaceSize());

        StaticBuffer<AddressSpaceEnum::Vgpr, ADataType, MThreadTileSize * NThreadTileSize, true>
            a_thread_buf;
        StaticBuffer<AddressSpaceEnum::Vgpr, BDataType, MThreadTileSize * NThreadTileSize, true>
            b_thread_buf;
        StaticBuffer<AddressSpaceEnum::Vgpr, CDataType, MThreadTileSize * NThreadTileSize, true>
            c_thread_buf;

        // TODO - buffer_load, apply functor, buffer_store
        (void)a_global_buf;
        (void)b_global_buf;
        (void)c_global_buf;
        (void)a_thread_buf;
        (void)b_thread_buf;
        (void)c_thread_buf;
        (void)functor;
    }
};

} // namespace ck
