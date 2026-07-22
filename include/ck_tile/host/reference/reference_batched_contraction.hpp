// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <cstdlib>
#include <thread>

#include "ck_tile/core.hpp"
#include "ck_tile/host/host_tensor.hpp"

namespace ck_tile {

// Helper to apply elementwise operation with variable number of D tensors
template <typename EDataType, typename AccDataType, typename CDEElementWise>
struct ApplyCDEElementWise
{
    template <typename... DValues>
    CK_TILE_HOST_DEVICE static void apply(EDataType& result,
                                          AccDataType sum,
                                          const CDEElementWise& cde_elementwise,
                                          DValues... d_vals)
    {
        if constexpr(sizeof...(DValues) == 0)
        {
            result = static_cast<EDataType>(sum);
        }
        else
        {
            cde_elementwise(
                result, ck_tile::type_convert<float>(sum), ck_tile::type_convert<float>(d_vals)...);
        }
    }
};

// Helper to extract D values at a given offset using index sequence
template <typename DDataType,
          ck_tile::index_t NumDTensor,
          typename Indices = std::make_index_sequence<NumDTensor>>
struct ExtractDValues;

template <typename DDataType, ck_tile::index_t NumDTensor, std::size_t... Is>
struct ExtractDValues<DDataType, NumDTensor, std::index_sequence<Is...>>
{
    template <typename EDataType, typename AccDataType, typename CDEElementWise>
    CK_TILE_HOST static void
    apply_at_offsets(EDataType& result,
                     AccDataType sum,
                     const CDEElementWise& cde_elementwise,
                     const std::array<ck_tile::HostTensor<DDataType>, NumDTensor>& ds_tensors,
                     const std::array<std::size_t, NumDTensor>& d_offsets)
    {
        ApplyCDEElementWise<EDataType, AccDataType, CDEElementWise>::apply(
            result, sum, cde_elementwise, ds_tensors[Is].mData[d_offsets[Is]]...);
    }
};

template <typename TensorDataType,
          ck_tile::index_t NumTensor,
          typename Indices = std::make_index_sequence<NumTensor>>
struct ExtractElementWiseValues;

template <typename TensorDataType, ck_tile::index_t NumTensor, std::size_t... Is>
struct ExtractElementWiseValues<TensorDataType, NumTensor, std::index_sequence<Is...>>
{
    template <typename OutputDataType, typename ElementWise>
    CK_TILE_HOST static void
    apply_at_offsets(OutputDataType& result,
                     const ElementWise& elementwise,
                     const std::array<ck_tile::HostTensor<TensorDataType>, NumTensor>& tensors,
                     const std::array<std::size_t, NumTensor>& offsets)
    {
        elementwise(result, tensors[Is].mData[offsets[Is]]...);
    }
};

CK_TILE_HOST std::size_t decode_strided_offset(const std::vector<std::size_t>& strides,
                                               ck_tile::index_t flat_idx,
                                               const std::vector<ck_tile::index_t>& dim_sizes,
                                               ck_tile::index_t stride_offset = 0)
{
    std::size_t offset    = 0;
    ck_tile::index_t temp = flat_idx;
    const auto num_dims   = static_cast<ck_tile::index_t>(dim_sizes.size());

    for(ck_tile::index_t i = num_dims; i > 0; --i)
    {
        const ck_tile::index_t dim = i - 1;
        offset += (temp % dim_sizes[dim]) * strides[stride_offset + dim];
        temp /= dim_sizes[dim];
    }

    return offset;
}

template <typename ADataType,
          typename BDataType,
          typename DDataType,
          typename EDataType,
          typename AccDataType,
          typename CDEElementWise,
          ck_tile::index_t NumDTensor>

void compute_reference_batched_contraction(
    const ck_tile::HostTensor<ADataType>& a_full_dims,
    const ck_tile::HostTensor<BDataType>& b_full_dims,
    const std::array<ck_tile::HostTensor<DDataType>, NumDTensor>& ds_full_dims_host,
    ck_tile::HostTensor<EDataType>& e_full_dims_host_ref,
    ck_tile::index_t G_total,
    ck_tile::index_t M_total,
    ck_tile::index_t N_total,
    ck_tile::index_t K_total,
    const CDEElementWise& cde_elementwise,
    const std::vector<ck_tile::index_t>& G_dims,
    const std::vector<ck_tile::index_t>& M_dims,
    const std::vector<ck_tile::index_t>& N_dims,
    const std::vector<ck_tile::index_t>& K_dims)
{
    // Extract stride information from tensor descriptors
    const auto a_strides = a_full_dims.get_strides();
    const auto b_strides = b_full_dims.get_strides();
    const auto e_strides = e_full_dims_host_ref.get_strides();

    // Extract D tensor strides
    std::array<std::vector<std::size_t>, NumDTensor> ds_strides;
    for(ck_tile::index_t d = 0; d < NumDTensor; ++d)
    {
        ds_strides[d] = ds_full_dims_host[d].get_strides();
    }

    const ck_tile::index_t num_g_dims = G_dims.size();
    const ck_tile::index_t num_m_dims = M_dims.size();
    const ck_tile::index_t num_n_dims = N_dims.size();

    // Helper lambda to compute linear index from flat indices using strides
    auto compute_a_offset = [&](ck_tile::index_t g_flat,
                                ck_tile::index_t m_flat,
                                ck_tile::index_t k_flat) -> std::size_t {
        return decode_strided_offset(a_strides, g_flat, G_dims) +
               decode_strided_offset(a_strides, m_flat, M_dims, num_g_dims) +
               decode_strided_offset(a_strides, k_flat, K_dims, num_g_dims + num_m_dims);
    };

    auto compute_b_offset = [&](ck_tile::index_t g_flat,
                                ck_tile::index_t n_flat,
                                ck_tile::index_t k_flat) -> std::size_t {
        return decode_strided_offset(b_strides, g_flat, G_dims) +
               decode_strided_offset(b_strides, n_flat, N_dims, num_g_dims) +
               decode_strided_offset(b_strides, k_flat, K_dims, num_g_dims + num_n_dims);
    };

    auto compute_e_offset = [&](ck_tile::index_t g_flat,
                                ck_tile::index_t m_flat,
                                ck_tile::index_t n_flat) -> std::size_t {
        return decode_strided_offset(e_strides, g_flat, G_dims) +
               decode_strided_offset(e_strides, m_flat, M_dims, num_g_dims) +
               decode_strided_offset(e_strides, n_flat, N_dims, num_g_dims + num_m_dims);
    };

    // Helper to compute D tensor offset (D tensors have same shape as E: [G, M, N])
    auto compute_d_offset = [&](ck_tile::index_t g_flat,
                                ck_tile::index_t m_flat,
                                ck_tile::index_t n_flat,
                                ck_tile::index_t d_idx) -> std::size_t {
        const auto& d_strides = ds_strides[d_idx];
        return decode_strided_offset(d_strides, g_flat, G_dims) +
               decode_strided_offset(d_strides, m_flat, M_dims, num_g_dims) +
               decode_strided_offset(d_strides, n_flat, N_dims, num_g_dims + num_m_dims);
    };

    // Parallel computation over G and M dimensions
    auto f_gm = [&](auto g_flat, auto m_flat) {
        for(ck_tile::index_t n_flat = 0; n_flat < N_total; ++n_flat)
        {
            AccDataType sum = 0;

            // Compute dot product over K dimension using stride-aware indexing
            for(ck_tile::index_t k_flat = 0; k_flat < K_total; ++k_flat)
            {
                const std::size_t a_offset = compute_a_offset(g_flat, m_flat, k_flat);
                const std::size_t b_offset = compute_b_offset(g_flat, n_flat, k_flat);

                auto a_val = a_full_dims.mData[a_offset];
                auto b_val = b_full_dims.mData[b_offset];
                sum += static_cast<AccDataType>(a_val) * static_cast<AccDataType>(b_val);
            }

            // Compute output offset using strides
            const std::size_t e_offset = compute_e_offset(g_flat, m_flat, n_flat);

            // Compute individual D tensor offsets using their respective strides
            std::array<std::size_t, NumDTensor> d_offsets;
            for(ck_tile::index_t d = 0; d < NumDTensor; ++d)
            {
                d_offsets[d] = compute_d_offset(g_flat, m_flat, n_flat, d);
            }

            // Apply elementwise operation with D tensors using compile-time dispatch
            EDataType result = static_cast<EDataType>(sum);
            ExtractDValues<DDataType, NumDTensor>::apply_at_offsets(
                result, sum, cde_elementwise, ds_full_dims_host, d_offsets);

            // Store result using stride-aware indexing
            e_full_dims_host_ref.mData[e_offset] = static_cast<EDataType>(result);
        }
    };

    // Execute parallel computation using hardware concurrency
    // Parallelize over G_total and M_total dimensions for optimal CPU utilization
    make_ParallelTensorFunctor(f_gm, G_total, M_total)(std::thread::hardware_concurrency());
}

/// @brief Host reference for batched contraction with multiple A and B tensors.
///
/// Each A tensor has layout [G0,G1,...,M0,M1,...,K0,K1,...] with potentially different strides.
/// Each B tensor has layout [G0,G1,...,N0,N1,...,K0,K1,...] with potentially different strides.
/// D/E tensors have layout [G0,G1,...,M0,M1,...,N0,N1,...].
///
/// Computation:
///   fused_a(g,m,k) = a_element_op(A0(g,m,k), A1(g,m,k), ...)
///   fused_b(g,n,k) = b_element_op(B0(g,n,k), B1(g,n,k), ...)
///   C(g,m,n) = sum_k fused_a(g,m,k) * fused_b(g,n,k)
///   E(g,m,n) = cde_element_op(C(g,m,n), D0(g,m,n), D1(g,m,n), ...)
template <typename AsDataType,
          typename BsDataType,
          typename DsDataType,
          typename AccDataType,
          typename EDataType,
          typename AElementWise,
          typename BElementWise,
          typename CDEElementWise,
          ck_tile::index_t NumATensor = AsDataType::size(),
          ck_tile::index_t NumBTensor = BsDataType::size(),
          ck_tile::index_t NumDTensor = DsDataType::size(),
          typename ADataType = ck_tile::remove_cvref_t<std::tuple_element_t<0, AsDataType>>,
          typename BDataType = ck_tile::remove_cvref_t<std::tuple_element_t<0, BsDataType>>,
          typename DDataType = ck_tile::remove_cvref_t<std::tuple_element_t<0, DsDataType>>>
void compute_reference_batched_contraction_multi_abd(
    const std::array<ck_tile::HostTensor<ADataType>, NumATensor>& as_tensors,
    const std::array<ck_tile::HostTensor<BDataType>, NumBTensor>& bs_tensors,
    const std::array<ck_tile::HostTensor<DDataType>, NumDTensor>& ds_tensors,
    ck_tile::HostTensor<EDataType>& e_tensor,
    ck_tile::index_t G_total,
    ck_tile::index_t M_total,
    ck_tile::index_t N_total,
    ck_tile::index_t K_total,
    const AElementWise& a_elementwise,
    const BElementWise& b_elementwise,
    const CDEElementWise& cde_elementwise,
    const std::vector<ck_tile::index_t>& G_dims,
    const std::vector<ck_tile::index_t>& M_dims,
    const std::vector<ck_tile::index_t>& N_dims,
    const std::vector<ck_tile::index_t>& K_dims)
{
    // Collect per-tensor strides
    std::array<std::vector<std::size_t>, NumATensor> as_strides;
    for(ck_tile::index_t a = 0; a < NumATensor; ++a)
        as_strides[a] = as_tensors[a].get_strides();

    std::array<std::vector<std::size_t>, NumBTensor> bs_strides;
    for(ck_tile::index_t b = 0; b < NumBTensor; ++b)
        bs_strides[b] = bs_tensors[b].get_strides();

    std::array<std::vector<std::size_t>, NumDTensor> ds_strides;
    for(ck_tile::index_t d = 0; d < NumDTensor; ++d)
        ds_strides[d] = ds_tensors[d].get_strides();

    const auto e_strides = e_tensor.get_strides();

    const ck_tile::index_t num_g_dims = G_dims.size();
    const ck_tile::index_t num_m_dims = M_dims.size();
    const ck_tile::index_t num_n_dims = N_dims.size();

    // Offset computation for A tensor (layout: [G, M, K])
    auto compute_a_offset = [&](ck_tile::index_t tensor_idx,
                                ck_tile::index_t g_flat,
                                ck_tile::index_t m_flat,
                                ck_tile::index_t k_flat) -> std::size_t {
        const auto& strides = as_strides[tensor_idx];
        return decode_strided_offset(strides, g_flat, G_dims) +
               decode_strided_offset(strides, m_flat, M_dims, num_g_dims) +
               decode_strided_offset(strides, k_flat, K_dims, num_g_dims + num_m_dims);
    };

    // Offset computation for B tensor (layout: [G, N, K])
    auto compute_b_offset = [&](ck_tile::index_t tensor_idx,
                                ck_tile::index_t g_flat,
                                ck_tile::index_t n_flat,
                                ck_tile::index_t k_flat) -> std::size_t {
        const auto& strides = bs_strides[tensor_idx];
        return decode_strided_offset(strides, g_flat, G_dims) +
               decode_strided_offset(strides, n_flat, N_dims, num_g_dims) +
               decode_strided_offset(strides, k_flat, K_dims, num_g_dims + num_n_dims);
    };

    // Offset computation for E/D tensor (layout: [G, M, N])
    auto compute_e_offset = [&](const std::vector<std::size_t>& strides,
                                ck_tile::index_t g_flat,
                                ck_tile::index_t m_flat,
                                ck_tile::index_t n_flat) -> std::size_t {
        return decode_strided_offset(strides, g_flat, G_dims) +
               decode_strided_offset(strides, m_flat, M_dims, num_g_dims) +
               decode_strided_offset(strides, n_flat, N_dims, num_g_dims + num_m_dims);
    };

    auto f_gm = [&](auto g_flat, auto m_flat) {
        for(ck_tile::index_t n_flat = 0; n_flat < N_total; ++n_flat)
        {
            AccDataType sum = 0;

            for(ck_tile::index_t k_flat = 0; k_flat < K_total; ++k_flat)
            {
                // Fuse all A tensors via a_elementwise
                ADataType fused_a{};
                std::array<std::size_t, NumATensor> a_offsets;
                for(ck_tile::index_t a = 0; a < NumATensor; ++a)
                {
                    a_offsets[a] = compute_a_offset(a, g_flat, m_flat, k_flat);
                }
                ExtractElementWiseValues<ADataType, NumATensor>::apply_at_offsets(
                    fused_a, a_elementwise, as_tensors, a_offsets);

                // Fuse all B tensors via b_elementwise
                BDataType fused_b{};
                std::array<std::size_t, NumBTensor> b_offsets;
                for(ck_tile::index_t b = 0; b < NumBTensor; ++b)
                {
                    b_offsets[b] = compute_b_offset(b, g_flat, n_flat, k_flat);
                }
                ExtractElementWiseValues<BDataType, NumBTensor>::apply_at_offsets(
                    fused_b, b_elementwise, bs_tensors, b_offsets);

                sum += static_cast<AccDataType>(fused_a) * static_cast<AccDataType>(fused_b);
            }

            const std::size_t e_offset = compute_e_offset(e_strides, g_flat, m_flat, n_flat);

            // Apply CDE elementwise with D tensors
            std::array<std::size_t, NumDTensor> d_offsets;
            for(ck_tile::index_t d = 0; d < NumDTensor; ++d)
            {
                d_offsets[d] = compute_e_offset(ds_strides[d], g_flat, m_flat, n_flat);
            }
            EDataType result = static_cast<EDataType>(sum);
            ExtractDValues<DDataType, NumDTensor>::apply_at_offsets(
                result, sum, cde_elementwise, ds_tensors, d_offsets);

            e_tensor.mData[e_offset] = result;
        }
    };

    make_ParallelTensorFunctor(f_gm, G_total, M_total)(std::thread::hardware_concurrency());
}

} // namespace ck_tile
