// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2022, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck/utility/common_header.hpp"
#include "ck/tensor_description/tensor_descriptor.hpp"
#include "ck/tensor_description/tensor_descriptor_helper.hpp"
#include "ck/tensor_operation/gpu/device/gemm_specialization.hpp"

namespace ck {
namespace tensor_operation {
namespace device {

// For padding tensors without batch dimension
template <bool PadM,
          bool PadN,
          typename TensorDesc_MRaw_NRaw,
          typename MPerBlockType,
          typename NPerBlockType,
          enable_if_t<TensorDesc_MRaw_NRaw::GetNumOfVisibleDimension() == 2, bool> = false>
__host__ __device__ constexpr auto
PadTensorDescriptor(const TensorDesc_MRaw_NRaw& tensor_desc_mraw_nraw,
                    MPerBlockType MPerBlock,
                    NPerBlockType NPerBlock)
{
    const auto MRaw = tensor_desc_mraw_nraw.GetLength(Number<0>{});
    const auto NRaw = tensor_desc_mraw_nraw.GetLength(Number<1>{});

    const auto M = math::integer_divide_ceil(MRaw, MPerBlock) * MPerBlock;
    const auto N = math::integer_divide_ceil(NRaw, NPerBlock) * NPerBlock;

    const auto MPad = M - MRaw;
    const auto NPad = N - NRaw;

    const auto MTransform = conditional_expr<PadM>(make_right_pad_transform(MRaw, MPad),
                                                   make_pass_through_transform(MRaw));
    const auto NTransform = conditional_expr<PadN>(make_right_pad_transform(NRaw, NPad),
                                                   make_pass_through_transform(NRaw));

    return transform_tensor_descriptor(tensor_desc_mraw_nraw,
                                       make_tuple(MTransform, NTransform),
                                       make_tuple(Sequence<0>{}, Sequence<1>{}),
                                       make_tuple(Sequence<0>{}, Sequence<1>{}));
}

// For padding tensors with batch dimension
template <bool PadM,
          bool PadN,
          typename TensorDesc_GRaw_MRaw_NRaw,
          typename MPerBlockType,
          typename NPerBlockType,
          enable_if_t<TensorDesc_GRaw_MRaw_NRaw::GetNumOfVisibleDimension() == 3, bool> = false>
__host__ __device__ constexpr auto
PadTensorDescriptor(const TensorDesc_GRaw_MRaw_NRaw& tensor_desc_graw_mraw_nraw,
                    MPerBlockType MPerBlock,
                    NPerBlockType NPerBlock)
{
    const auto GRaw = tensor_desc_graw_mraw_nraw.GetLength(Number<0>{});
    const auto MRaw = tensor_desc_graw_mraw_nraw.GetLength(Number<1>{});
    const auto NRaw = tensor_desc_graw_mraw_nraw.GetLength(Number<2>{});

    const auto M = math::integer_divide_ceil(MRaw, MPerBlock) * MPerBlock;
    const auto N = math::integer_divide_ceil(NRaw, NPerBlock) * NPerBlock;

    const auto MPad = M - MRaw;
    const auto NPad = N - NRaw;

    const auto MTransform = conditional_expr<PadM>(make_right_pad_transform(MRaw, MPad),
                                                   make_pass_through_transform(MRaw));
    const auto NTransform = conditional_expr<PadN>(make_right_pad_transform(NRaw, NPad),
                                                   make_pass_through_transform(NRaw));

    return transform_tensor_descriptor(
        tensor_desc_graw_mraw_nraw,
        make_tuple(make_pass_through_transform(GRaw), MTransform, NTransform),
        make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}),
        make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}));
}

// M/N/K/OPerTileType could be index_t or Number<>
template <GemmSpecialization GemmSpec,
          typename MPerTileType,
          typename NPerTileType,
          typename KPerTileType,
          typename OPerTileType>
struct GemmGemmPadder
{
    // TODO: hard to scale; use mask instead
    static constexpr bool PadM =
        GemmSpec == GemmSpecialization::MPadding || GemmSpec == GemmSpecialization::MNPadding ||
        GemmSpec == GemmSpecialization::MKPadding || GemmSpec == GemmSpecialization::MNKPadding ||
        GemmSpec == GemmSpecialization::MOPadding || GemmSpec == GemmSpecialization::MNOPadding ||
        GemmSpec == GemmSpecialization::MKOPadding || GemmSpec == GemmSpecialization::MNKOPadding;
    static constexpr bool PadN =
        GemmSpec == GemmSpecialization::NPadding || GemmSpec == GemmSpecialization::MNPadding ||
        GemmSpec == GemmSpecialization::NKPadding || GemmSpec == GemmSpecialization::MNKPadding ||
        GemmSpec == GemmSpecialization::NOPadding || GemmSpec == GemmSpecialization::MNOPadding ||
        GemmSpec == GemmSpecialization::NKOPadding || GemmSpec == GemmSpecialization::MNKOPadding;
    static constexpr bool PadK =
        GemmSpec == GemmSpecialization::KPadding || GemmSpec == GemmSpecialization::MKPadding ||
        GemmSpec == GemmSpecialization::NKPadding || GemmSpec == GemmSpecialization::MNKPadding ||
        GemmSpec == GemmSpecialization::KOPadding || GemmSpec == GemmSpecialization::MKOPadding ||
        GemmSpec == GemmSpecialization::NKOPadding || GemmSpec == GemmSpecialization::MNKOPadding;
    static constexpr bool PadO =
        GemmSpec == GemmSpecialization::OPadding || GemmSpec == GemmSpecialization::MOPadding ||
        GemmSpec == GemmSpecialization::NOPadding || GemmSpec == GemmSpecialization::KOPadding ||
        GemmSpec == GemmSpecialization::MNOPadding || GemmSpec == GemmSpecialization::MKOPadding ||
        GemmSpec == GemmSpecialization::NKOPadding || GemmSpec == GemmSpecialization::MNKOPadding;

    // A[M, K]
    template <typename ADesc_MRaw_KRaw>
    __host__ __device__ constexpr auto
    PadADescriptor_M_K(const ADesc_MRaw_KRaw& a_desc_mraw_kraw) const
    {
        return PadTensorDescriptor<PadM, PadK>(a_desc_mraw_kraw, MPerTile_, KPerTile_);
    }

    // B[K, N]
    template <typename BDesc_NRaw_KRaw>
    __host__ __device__ constexpr auto
    PadBDescriptor_N_K(const BDesc_NRaw_KRaw& b_desc_nraw_kraw) const
    {
        return PadTensorDescriptor<PadN, PadK>(b_desc_nraw_kraw, NPerTile_, KPerTile_);
    }

    // B1[Gemm1N, Gemm1K] = B1[O, N]
    template <typename B1Desc_NRaw_KRaw>
    __host__ __device__ constexpr auto
    PadB1Descriptor_N_K(const B1Desc_NRaw_KRaw& b1_desc_nraw_kraw) const
    {
        return PadTensorDescriptor<PadO, PadN>(b1_desc_nraw_kraw, OPerTile_, NPerTile_);
    }

    // C[M, Gemm1N] = C[M, O]
    template <typename CDesc_MRaw_NRaw>
    __host__ __device__ constexpr auto
    PadCDescriptor_M_N(const CDesc_MRaw_NRaw& c_desc_mraw_nraw) const
    {
        return PadTensorDescriptor<PadM, PadO>(c_desc_mraw_nraw, MPerTile_, OPerTile_);
    }

    MPerTileType MPerTile_;
    NPerTileType NPerTile_;
    KPerTileType KPerTile_;
    OPerTileType OPerTile_;
};

// M/N/KPerTileType could be index_t or Number<>
template <GemmSpecialization GemmSpec,
          typename MPerTileType,
          typename NPerTileType,
          typename KPerTileType>
struct GemmPadder
{
    static constexpr bool PadM =
        (GemmSpec == GemmSpecialization::MPadding || GemmSpec == GemmSpecialization::MNPadding ||
         GemmSpec == GemmSpecialization::MKPadding || GemmSpec == GemmSpecialization::MNKPadding);
    static constexpr bool PadN =
        (GemmSpec == GemmSpecialization::NPadding || GemmSpec == GemmSpecialization::MNPadding ||
         GemmSpec == GemmSpecialization::NKPadding || GemmSpec == GemmSpecialization::MNKPadding);
    static constexpr bool PadK =
        (GemmSpec == GemmSpecialization::KPadding || GemmSpec == GemmSpecialization::MKPadding ||
         GemmSpec == GemmSpecialization::NKPadding || GemmSpec == GemmSpecialization::MNKPadding);

    template <typename ADesc_MRaw_KRaw>
    __host__ __device__ constexpr auto
    PadADescriptor_M_K(const ADesc_MRaw_KRaw& a_desc_mraw_kraw) const
    {
        return PadTensorDescriptor<PadM, PadK>(a_desc_mraw_kraw, MPerTile_, KPerTile_);
    }

    template <typename BDesc_NRaw_KRaw>
    __host__ __device__ constexpr auto
    PadBDescriptor_N_K(const BDesc_NRaw_KRaw& b_desc_nraw_kraw) const
    {
        return PadTensorDescriptor<PadN, PadK>(b_desc_nraw_kraw, NPerTile_, KPerTile_);
    }

    template <typename CDesc_MRaw_NRaw>
    __host__ __device__ constexpr auto
    PadCDescriptor_M_N(const CDesc_MRaw_NRaw& c_desc_mraw_nraw) const
    {
        return PadTensorDescriptor<PadM, PadN>(c_desc_mraw_nraw, MPerTile_, NPerTile_);
    }

    MPerTileType MPerTile_;
    NPerTileType NPerTile_;
    KPerTileType KPerTile_;
};

// Alias of GemmPadder; to deprecate
template <GemmSpecialization GemmSpec,
          typename MPerTileType,
          typename NPerTileType,
          typename KPerTileType>
struct MatrixPadder : public GemmPadder<GemmSpec, MPerTileType, NPerTileType, KPerTileType>
{
};

// M/N/KPerTileType could be index_t or Number<>
template <bool PadM,
          bool PadN,
          bool PadK,
          typename MPerTileType,
          typename NPerTileType,
          typename KPerTileType>
struct GemmPadder_v2
{
    static constexpr auto I0 = Number<0>{};
    static constexpr auto I1 = Number<1>{};
    static constexpr auto I2 = Number<2>{};
    static constexpr auto I3 = Number<3>{};

    template <typename ADesc_MRaw_KRaw>
    __host__ __device__ constexpr auto
    PadADescriptor_M_K(const ADesc_MRaw_KRaw& a_desc_mraw_kraw) const
    {
        const auto MRaw = a_desc_mraw_kraw.GetLength(I0);
        const auto KRaw = a_desc_mraw_kraw.GetLength(I1);

        const auto M = math::integer_divide_ceil(MRaw, MPerTile_) * MPerTile_;
        const auto K = math::integer_divide_ceil(KRaw, KPerTile_) * KPerTile_;

        const auto MPad = M - MRaw;
        const auto KPad = K - KRaw;

        if constexpr(PadM && PadK)
        {
            // pad both M and K
            return transform_tensor_descriptor(a_desc_mraw_kraw,
                                               make_tuple(make_right_pad_transform(MRaw, MPad),
                                                          make_right_pad_transform(KRaw, KPad)),
                                               make_tuple(Sequence<0>{}, Sequence<1>{}),
                                               make_tuple(Sequence<0>{}, Sequence<1>{}));
        }
        else if constexpr(PadM && (!PadK))
        {
            // pad M, but not K
            return transform_tensor_descriptor(
                a_desc_mraw_kraw,
                make_tuple(make_right_pad_transform(MRaw, MPad), make_pass_through_transform(KRaw)),
                make_tuple(Sequence<0>{}, Sequence<1>{}),
                make_tuple(Sequence<0>{}, Sequence<1>{}));
        }
        else if constexpr((!PadM) && PadK)
        {
            // pad K, but not M
            return transform_tensor_descriptor(
                a_desc_mraw_kraw,
                make_tuple(make_pass_through_transform(MRaw), make_right_pad_transform(KRaw, KPad)),
                make_tuple(Sequence<0>{}, Sequence<1>{}),
                make_tuple(Sequence<0>{}, Sequence<1>{}));
        }
        else
        {
            // not pad M or K
            return a_desc_mraw_kraw;
        }
    }

    template <typename BDesc_NRaw_KRaw>
    __host__ __device__ constexpr auto
    PadBDescriptor_N_K(const BDesc_NRaw_KRaw& b_desc_nraw_kraw) const
    {
        const auto NRaw = b_desc_nraw_kraw.GetLength(I0);
        const auto KRaw = b_desc_nraw_kraw.GetLength(I1);

        const auto N = math::integer_divide_ceil(NRaw, NPerTile_) * NPerTile_;
        const auto K = math::integer_divide_ceil(KRaw, KPerTile_) * KPerTile_;

        const auto NPad = N - NRaw;
        const auto KPad = K - KRaw;

        if constexpr(PadN && PadK)
        {
            // pad both N and K
            return transform_tensor_descriptor(b_desc_nraw_kraw,
                                               make_tuple(make_right_pad_transform(NRaw, NPad),
                                                          make_right_pad_transform(KRaw, KPad)),
                                               make_tuple(Sequence<0>{}, Sequence<1>{}),
                                               make_tuple(Sequence<0>{}, Sequence<1>{}));
        }
        else if constexpr(PadN && (!PadK))
        {
            // pad N, but not K
            return transform_tensor_descriptor(
                b_desc_nraw_kraw,
                make_tuple(make_right_pad_transform(NRaw, NPad), make_pass_through_transform(KRaw)),
                make_tuple(Sequence<0>{}, Sequence<1>{}),
                make_tuple(Sequence<0>{}, Sequence<1>{}));
        }
        else if constexpr((!PadN) && PadK)
        {
            // pad K, but not N
            return transform_tensor_descriptor(
                b_desc_nraw_kraw,
                make_tuple(make_pass_through_transform(NRaw), make_right_pad_transform(KRaw, KPad)),
                make_tuple(Sequence<0>{}, Sequence<1>{}),
                make_tuple(Sequence<0>{}, Sequence<1>{}));
        }
        else
        {
            // not pad N or K
            return b_desc_nraw_kraw;
        }
    }

    template <typename CDesc_MRaw_NRaw>
    __host__ __device__ constexpr auto
    PadCDescriptor_M_N(const CDesc_MRaw_NRaw& c_desc_mraw_nraw) const
    {
        const auto MRaw = c_desc_mraw_nraw.GetLength(I0);
        const auto NRaw = c_desc_mraw_nraw.GetLength(I1);

        const auto M = math::integer_divide_ceil(MRaw, MPerTile_) * MPerTile_;
        const auto N = math::integer_divide_ceil(NRaw, NPerTile_) * NPerTile_;

        const auto MPad = M - MRaw;
        const auto NPad = N - NRaw;

        if constexpr(PadM && PadN)
        {
            // pad M and N
            return transform_tensor_descriptor(c_desc_mraw_nraw,
                                               make_tuple(make_right_pad_transform(MRaw, MPad),
                                                          make_right_pad_transform(NRaw, NPad)),
                                               make_tuple(Sequence<0>{}, Sequence<1>{}),
                                               make_tuple(Sequence<0>{}, Sequence<1>{}));
        }
        else if constexpr(PadM && (!PadN))
        {
            // pad M, but not N
            return transform_tensor_descriptor(
                c_desc_mraw_nraw,
                make_tuple(make_right_pad_transform(MRaw, MPad), make_pass_through_transform(NRaw)),
                make_tuple(Sequence<0>{}, Sequence<1>{}),
                make_tuple(Sequence<0>{}, Sequence<1>{}));
        }
        else if constexpr((!PadM) && PadN)
        {
            // pad N, but not M
            return transform_tensor_descriptor(
                c_desc_mraw_nraw,
                make_tuple(make_pass_through_transform(MRaw), make_right_pad_transform(NRaw, NPad)),
                make_tuple(Sequence<0>{}, Sequence<1>{}),
                make_tuple(Sequence<0>{}, Sequence<1>{}));
        }
        else
        {
            // not pad M or N
            return c_desc_mraw_nraw;
        }
    }

    MPerTileType MPerTile_;
    NPerTileType NPerTile_;
    KPerTileType KPerTile_;
};

} // namespace device
} // namespace tensor_operation
} // namespace ck
