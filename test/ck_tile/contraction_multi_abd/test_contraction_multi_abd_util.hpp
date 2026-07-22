// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
#pragma once
#include <algorithm>
#include <array>
#include <cstdint>
#include <sstream>
#include <stdexcept>
#include <gtest/gtest.h>

#include "ck_tile/core.hpp"
#include "ck_tile/host.hpp"
#include "ck_tile/host/kernel_launch.hpp"
#include "ck_tile/ops/epilogue.hpp"
#include "ck_tile/ops/gemm.hpp"
#include "ck_tile/ops/batched_contraction_multi_abd.hpp"
#include "ck_tile/ops/elementwise/unary_element_wise_operation.hpp"
#include "ck_tile/host/reference/reference_batched_contraction.hpp"

using AddScale          = ck_tile::element_wise::AddScale;
using ElementWiseAddAdd = ck_tile::element_wise::MultiDAdd;
using MultiplyMultiply  = ck_tile::element_wise::MultiDMultiply;
using PassThrough       = ck_tile::element_wise::PassThrough;

template <std::size_t N>
std::array<ck_tile::index_t, N> to_fixed_dims(const std::vector<ck_tile::index_t>& dims)
{
    if(dims.size() != N)
    {
        throw std::invalid_argument("Unexpected dimension count");
    }

    std::array<ck_tile::index_t, N> result{};
    std::copy(dims.begin(), dims.end(), result.begin());
    return result;
}

struct AddDs
{
    template <typename E, typename C, typename... Ds>
    CK_TILE_HOST_DEVICE auto operator()(E& e, const C& c, const Ds&... ds) const -> void
    {
        const float x0_f =
            ck_tile::type_convert<float>(c) + (ck_tile::type_convert<float>(ds) + ...);
        e = ck_tile::type_convert<E>(x0_f);
    }
};

template <typename T, std::size_t>
struct RepeatType
{
    using type = T;
};

template <typename T, typename Indices>
struct RepeatedTuple;

template <typename T, std::size_t... Is>
struct RepeatedTuple<T, std::index_sequence<Is...>>
{
    using type = ck_tile::tuple<typename RepeatType<T, Is>::type...>;
};

template <typename T, ck_tile::index_t N>
using RepeatedTupleT =
    typename RepeatedTuple<T, std::make_index_sequence<static_cast<std::size_t>(N)>>::type;

template <typename TensorDataType, ck_tile::index_t NumTensor, std::size_t... Is>
auto make_host_tensor_array_impl(const ck_tile::HostTensorDescriptor& desc,
                                 std::index_sequence<Is...>)
{
    return std::array<ck_tile::HostTensor<TensorDataType>, NumTensor>{
        {((void)Is, ck_tile::HostTensor<TensorDataType>(desc))...}};
}

template <typename TensorDataType, ck_tile::index_t NumTensor>
auto make_host_tensor_array(const ck_tile::HostTensorDescriptor& desc)
{
    return make_host_tensor_array_impl<TensorDataType, NumTensor>(
        desc, std::make_index_sequence<static_cast<std::size_t>(NumTensor)>{});
}

template <typename A0DataType_,
          typename B0DataType_,
          typename AccDataType_,
          typename EDataType_,
          typename D0DataType_>
auto calculate_rtol_atol(const ck_tile::index_t K,
                         const ck_tile::index_t kbatch,
                         const float max_accumulated_value)
{
    using ComputeTypeAB =
        std::conditional_t<sizeof(A0DataType_) < sizeof(B0DataType_), A0DataType_, B0DataType_>;
    using ComputeType =
        std::conditional_t<sizeof(ComputeTypeAB) < sizeof(D0DataType_), ComputeTypeAB, D0DataType_>;

    const auto rtol = ck_tile::get_relative_threshold<ComputeType, EDataType_, AccDataType_>(
        ck_tile::integer_divide_ceil(K, kbatch));
    const auto atol = ck_tile::get_absolute_threshold<ComputeType, EDataType_, AccDataType_>(
        max_accumulated_value / kbatch, ck_tile::integer_divide_ceil(K, kbatch));

    const auto rtol_split_k =
        ck_tile::get_relative_threshold<EDataType_, EDataType_, EDataType_>(kbatch);
    const auto atol_split_k = ck_tile::get_absolute_threshold<EDataType_, EDataType_, EDataType_>(
        max_accumulated_value, kbatch);

    return ck_tile::make_tuple(std::max(rtol, rtol_split_k), std::max(atol, atol_split_k));
}

// Test fixture parameterized by a tuple of types
template <typename Tuple>
class TestCkTileContractionMultiABD : public ::testing::Test
{
    protected:
    using AsDataType        = std::tuple_element_t<0, Tuple>;
    using BsDataType        = std::tuple_element_t<1, Tuple>;
    using DsDataType        = std::tuple_element_t<2, Tuple>;
    using AccDataType       = std::tuple_element_t<3, Tuple>;
    using EDataType         = std::tuple_element_t<4, Tuple>;
    using AElementWiseFn    = std::tuple_element_t<5, Tuple>;
    using BElementWiseFn    = std::tuple_element_t<6, Tuple>;
    using CDEElementWiseFn  = std::tuple_element_t<7, Tuple>;
    using UseCshuffleEpilog = std::tuple_element_t<8, Tuple>;

    using Row = ck_tile::tensor_layout::gemm::RowMajor;
    using Col = ck_tile::tensor_layout::gemm::ColumnMajor;

    using ADataType = ck_tile::remove_cvref_t<std::tuple_element_t<0, AsDataType>>;
    using BDataType = ck_tile::remove_cvref_t<std::tuple_element_t<0, BsDataType>>;
    using DDataType = ck_tile::remove_cvref_t<std::tuple_element_t<0, DsDataType>>;

    static constexpr ck_tile::index_t NumATensor = AsDataType::size();
    static constexpr ck_tile::index_t NumBTensor = BsDataType::size();
    static constexpr ck_tile::index_t NumDTensor = DsDataType::size();

    using AsLayout = RepeatedTupleT<Row, NumATensor>;
    using BsLayout = RepeatedTupleT<Col, NumBTensor>;
    using DsLayout = RepeatedTupleT<Row, NumDTensor>;
    using ELayout  = Row;

    template <ck_tile::index_t NumDimG,
              ck_tile::index_t NumDimM,
              ck_tile::index_t NumDimN,
              ck_tile::index_t NumDimK,
              bool kPadM = false,
              bool kPadN = false,
              bool kPadK = false>
    bool invoke_kernel(const ck_tile::BatchedContractionMultiABDHostArgs<NumDimG,
                                                                         NumDimM,
                                                                         NumDimN,
                                                                         NumDimK,
                                                                         NumATensor,
                                                                         NumBTensor,
                                                                         NumDTensor>& args,
                       const ck_tile::stream_config& s)
    {
        constexpr ck_tile::index_t M_Tile = 128;
        constexpr ck_tile::index_t N_Tile = 128;
        constexpr ck_tile::index_t K_Tile = 64;

        constexpr ck_tile::index_t M_Warp = 2;
        constexpr ck_tile::index_t N_Warp = 2;
        constexpr ck_tile::index_t K_Warp = 1;

#if CK_TILE_USE_WMMA
        constexpr ck_tile::index_t M_Warp_Tile = 16;
        constexpr ck_tile::index_t N_Warp_Tile = 16;
        constexpr ck_tile::index_t K_Warp_Tile = 16;
#else
        constexpr ck_tile::index_t M_Warp_Tile = 32;
        constexpr ck_tile::index_t N_Warp_Tile = 32;
        constexpr ck_tile::index_t K_Warp_Tile = 16;
#endif

        constexpr bool DoubleSmemBuffer = false;
        constexpr bool TransposeC       = false;

        constexpr int kBlockPerCu                         = 1;
        constexpr ck_tile::index_t TileParitionerGroupNum = 8;
        constexpr ck_tile::index_t TileParitionerM01      = 4;

        using GemmShape =
            ck_tile::TileGemmShape<ck_tile::sequence<M_Tile, N_Tile, K_Tile>,
                                   ck_tile::sequence<M_Warp, N_Warp, K_Warp>,
                                   ck_tile::sequence<M_Warp_Tile, N_Warp_Tile, K_Warp_Tile>>;
        using TilePartitioner = ck_tile::
            GemmSpatiallyLocalTilePartitioner<GemmShape, TileParitionerGroupNum, TileParitionerM01>;

        using GemmUniversalTraits = ck_tile::TileGemmUniversalTraits<kPadM,
                                                                     kPadN,
                                                                     kPadK,
                                                                     DoubleSmemBuffer,
                                                                     AsLayout,
                                                                     BsLayout,
                                                                     ELayout,
                                                                     TransposeC>;

        using Problem = ck_tile::BatchedContractionMultiABDProblem<AsDataType,
                                                                   BsDataType,
                                                                   DsDataType,
                                                                   EDataType,
                                                                   NumDimG,
                                                                   NumDimM,
                                                                   NumDimN,
                                                                   NumDimK>;

        constexpr auto scheduler = ck_tile::GemmPipelineScheduler::Intrawave;

        using UniversalGemmProblem = ck_tile::UniversalGemmPipelineProblem<AsDataType,
                                                                           BsDataType,
                                                                           AccDataType,
                                                                           GemmShape,
                                                                           GemmUniversalTraits,
                                                                           scheduler,
                                                                           AElementWiseFn,
                                                                           BElementWiseFn>;

        using GemmPipeline = ck_tile::GemmPipelineAgBgCrCompV3<UniversalGemmProblem>;

        using CShuffleGemmEpilogue = ck_tile::CShuffleEpilogue<
            ck_tile::CShuffleEpilogueProblem<AsDataType,
                                             BsDataType,
                                             DsDataType,
                                             AccDataType,
                                             EDataType,
                                             DsLayout,
                                             ELayout,
                                             CDEElementWiseFn,
                                             TilePartitioner::MPerBlock,
                                             TilePartitioner::NPerBlock,
                                             M_Warp,
                                             N_Warp,
                                             M_Warp_Tile,
                                             N_Warp_Tile,
                                             K_Warp_Tile,
                                             UniversalGemmProblem::TransposeC>>;

        using DefaultGemmEpilogue = ck_tile::DefaultGemm2DEpilogue<
            ck_tile::DefaultGemm2DEpilogueProblem<AsDataType,
                                                  BsDataType,
                                                  DsDataType,
                                                  AccDataType,
                                                  EDataType,
                                                  DsLayout,
                                                  ELayout,
                                                  CDEElementWiseFn,
                                                  TilePartitioner::MPerBlock,
                                                  TilePartitioner::NPerBlock,
                                                  kPadM,
                                                  kPadN,
                                                  M_Warp_Tile,
                                                  N_Warp_Tile,
                                                  K_Warp_Tile,
                                                  UniversalGemmProblem::TransposeC,
                                                  true>>;

        using GemmEpilogue =
            std::conditional_t<UseCshuffleEpilog::value, CShuffleGemmEpilogue, DefaultGemmEpilogue>;

        using Kernel = ck_tile::
            BatchedContractionMultiABDKernel<Problem, TilePartitioner, GemmPipeline, GemmEpilogue>;

        auto kargs = Kernel::MakeKernelArgs(args);

        const dim3 grids  = Kernel::GridSize(kargs);
        const dim3 blocks = Kernel::GetBlockSize();

        if(!Kernel::IsSupportedArguments(kargs))
        {
            return false;
        }

        ck_tile::ignore = ck_tile::launch_kernel(
            s, ck_tile::make_kernel<kBlockPerCu>(Kernel{}, grids, blocks, 0, kargs));
        return true;
    }

    public:
    template <ck_tile::index_t NumDimG,
              ck_tile::index_t NumDimM,
              ck_tile::index_t NumDimN,
              ck_tile::index_t NumDimK,
              bool kPadM = false,
              bool kPadN = false,
              bool kPadK = false>
    bool Run(const std::vector<ck_tile::index_t>& G_dims,
             const std::vector<ck_tile::index_t>& M_dims,
             const std::vector<ck_tile::index_t>& N_dims,
             const std::vector<ck_tile::index_t>& K_dims,
             bool use_padded_g_strides = false)
    {
        auto calc_total = [](const std::vector<ck_tile::index_t>& dims) {
            ck_tile::index_t t = 1;
            for(auto d : dims)
                t *= d;
            return t;
        };

        auto concat = [](const std::vector<std::vector<ck_tile::index_t>>& vecs) {
            std::vector<ck_tile::index_t> r;
            for(const auto& v : vecs)
                r.insert(r.end(), v.begin(), v.end());
            return r;
        };

        ck_tile::index_t G_total = calc_total(G_dims);
        ck_tile::index_t M_total = calc_total(M_dims);
        ck_tile::index_t N_total = calc_total(N_dims);
        ck_tile::index_t K_total = calc_total(K_dims);

        std::vector<ck_tile::index_t> A_dims = concat({G_dims, M_dims, K_dims});
        std::vector<ck_tile::index_t> B_dims = concat({G_dims, N_dims, K_dims});
        std::vector<ck_tile::index_t> E_dims = concat({G_dims, M_dims, N_dims});

        auto make_desc = [&](const std::vector<ck_tile::index_t>& dims, std::size_t padding) {
            auto desc    = ck_tile::HostTensorDescriptor(dims);
            auto strides = desc.get_strides();
            if(use_padded_g_strides)
            {
                if constexpr(NumDimG > 1)
                {
                    strides[NumDimG - 1] += padding;
                    for(ck_tile::index_t reverse_i = 1; reverse_i < NumDimG; ++reverse_i)
                    {
                        const ck_tile::index_t i = NumDimG - 1 - reverse_i;
                        strides[i] =
                            static_cast<std::size_t>(dims[i + 1]) * strides[i + 1] + padding;
                    }
                }
            }
            return ck_tile::HostTensorDescriptor(dims, strides);
        };

        const auto a_desc = make_desc(A_dims, 17);
        const auto b_desc = make_desc(B_dims, 29);
        const auto e_desc = make_desc(E_dims, 43);

        auto as_host = make_host_tensor_array<ADataType, NumATensor>(a_desc);
        auto bs_host = make_host_tensor_array<BDataType, NumBTensor>(b_desc);
        auto ds_host = make_host_tensor_array<DDataType, NumDTensor>(e_desc);
        ck_tile::HostTensor<EDataType> e_gpu_host(e_desc);

        constexpr uint32_t seed = 11939;
        for(ck_tile::index_t a = 0; a < NumATensor; ++a)
        {
            ck_tile::FillUniformDistribution<ADataType>{-1.f, 1.f, static_cast<uint32_t>(seed + a)}(
                as_host[a]);
        }
        for(ck_tile::index_t b = 0; b < NumBTensor; ++b)
        {
            ck_tile::FillUniformDistribution<BDataType>{
                -1.f, 1.f, static_cast<uint32_t>(seed + NumATensor + b)}(bs_host[b]);
        }
        for(ck_tile::index_t d = 0; d < NumDTensor; ++d)
        {
            ck_tile::FillUniformDistribution<DDataType>{
                -1.f, 1.f, static_cast<uint32_t>(seed + NumATensor + NumBTensor + d)}(ds_host[d]);
        }

        std::array<ck_tile::DeviceMem, NumATensor> as_dev;
        std::array<ck_tile::DeviceMem, NumBTensor> bs_dev;
        std::array<ck_tile::DeviceMem, NumDTensor> ds_dev;
        ck_tile::DeviceMem e_dev(e_gpu_host.get_element_space_size_in_bytes());

        for(ck_tile::index_t a = 0; a < NumATensor; ++a)
        {
            as_dev[a].Realloc(as_host[a].get_element_space_size_in_bytes());
            as_dev[a].ToDevice(as_host[a].data());
        }
        for(ck_tile::index_t b = 0; b < NumBTensor; ++b)
        {
            bs_dev[b].Realloc(bs_host[b].get_element_space_size_in_bytes());
            bs_dev[b].ToDevice(bs_host[b].data());
        }
        for(ck_tile::index_t d = 0; d < NumDTensor; ++d)
        {
            ds_dev[d].Realloc(ds_host[d].get_element_space_size_in_bytes());
            ds_dev[d].ToDevice(ds_host[d].data());
        }
        e_dev.SetZero();
        e_gpu_host.SetZero();

        auto convert_strides = [](const std::vector<std::size_t>& strides) {
            std::vector<ck_tile::index_t> converted(strides.size());
            std::copy(strides.begin(), strides.end(), converted.begin());
            return converted;
        };

        std::vector<ck_tile::index_t> E_strides = convert_strides(e_gpu_host.get_strides());

        using HostArgs = ck_tile::BatchedContractionMultiABDHostArgs<NumDimG,
                                                                     NumDimM,
                                                                     NumDimN,
                                                                     NumDimK,
                                                                     NumATensor,
                                                                     NumBTensor,
                                                                     NumDTensor>;
        using ADims    = typename HostArgs::ADims;
        using BDims    = typename HostArgs::BDims;
        using EDims    = typename HostArgs::EDims;

        std::array<const void*, NumATensor> as_ptr{};
        std::array<const void*, NumBTensor> bs_ptr{};
        std::array<const void*, NumDTensor> ds_ptr{};
        std::array<ADims, NumATensor> As_dims{};
        std::array<BDims, NumBTensor> Bs_dims{};
        std::array<EDims, NumDTensor> Ds_dims{};
        std::array<ADims, NumATensor> As_strides{};
        std::array<BDims, NumBTensor> Bs_strides{};
        std::array<EDims, NumDTensor> Ds_strides{};
        for(ck_tile::index_t a = 0; a < NumATensor; ++a)
        {
            as_ptr[a]     = as_dev[a].GetDeviceBuffer();
            As_dims[a]    = to_fixed_dims<NumDimG + NumDimM + NumDimK>(A_dims);
            As_strides[a] = to_fixed_dims<NumDimG + NumDimM + NumDimK>(
                convert_strides(as_host[a].get_strides()));
        }
        for(ck_tile::index_t b = 0; b < NumBTensor; ++b)
        {
            bs_ptr[b]     = bs_dev[b].GetDeviceBuffer();
            Bs_dims[b]    = to_fixed_dims<NumDimG + NumDimN + NumDimK>(B_dims);
            Bs_strides[b] = to_fixed_dims<NumDimG + NumDimN + NumDimK>(
                convert_strides(bs_host[b].get_strides()));
        }
        for(ck_tile::index_t d = 0; d < NumDTensor; ++d)
        {
            ds_ptr[d]     = ds_dev[d].GetDeviceBuffer();
            Ds_dims[d]    = to_fixed_dims<NumDimG + NumDimM + NumDimN>(E_dims);
            Ds_strides[d] = to_fixed_dims<NumDimG + NumDimM + NumDimN>(
                convert_strides(ds_host[d].get_strides()));
        }
        HostArgs host_args{as_ptr,
                           bs_ptr,
                           ds_ptr,
                           e_dev.GetDeviceBuffer(),
                           As_dims,
                           Bs_dims,
                           Ds_dims,
                           to_fixed_dims<NumDimG + NumDimM + NumDimN>(E_dims),
                           As_strides,
                           Bs_strides,
                           Ds_strides,
                           to_fixed_dims<NumDimG + NumDimM + NumDimN>(E_strides)};

        const bool supported =
            invoke_kernel<NumDimG, NumDimM, NumDimN, NumDimK, kPadM, kPadN, kPadK>(
                host_args, ck_tile::stream_config{nullptr, false});
        if(!supported)
        {
            std::cout << "G=" << G_total << " M=" << M_total << " N=" << N_total << " K=" << K_total
                      << " unsupported by multi-ABD kernel" << std::endl;
            return false;
        }

        e_dev.FromDevice(e_gpu_host.data());

        ck_tile::HostTensor<EDataType> e_ref_host(e_desc);
        e_ref_host.SetZero();

        ck_tile::compute_reference_batched_contraction_multi_abd<AsDataType,
                                                                 BsDataType,
                                                                 DsDataType,
                                                                 AccDataType,
                                                                 EDataType,
                                                                 AElementWiseFn,
                                                                 BElementWiseFn,
                                                                 CDEElementWiseFn,
                                                                 NumATensor,
                                                                 NumBTensor,
                                                                 NumDTensor>(as_host,
                                                                             bs_host,
                                                                             ds_host,
                                                                             e_ref_host,
                                                                             G_total,
                                                                             M_total,
                                                                             N_total,
                                                                             K_total,
                                                                             AElementWiseFn{},
                                                                             BElementWiseFn{},
                                                                             CDEElementWiseFn{},
                                                                             G_dims,
                                                                             M_dims,
                                                                             N_dims,
                                                                             K_dims);

        const float max_accumulated_value =
            *std::max_element(e_ref_host.mData.begin(), e_ref_host.mData.end());
        const auto rtol_atol =
            calculate_rtol_atol<ADataType, BDataType, AccDataType, EDataType, DDataType>(
                K_total, 1, max_accumulated_value);

        bool pass = ck_tile::check_err(e_gpu_host,
                                       e_ref_host,
                                       "Error: Incorrect results!",
                                       rtol_atol.at(ck_tile::number<0>{}),
                                       rtol_atol.at(ck_tile::number<1>{}));

        std::cout << "G=" << G_total << " M=" << M_total << " N=" << N_total << " K=" << K_total
                  << " rtol=" << rtol_atol.at(ck_tile::number<0>{})
                  << " atol=" << rtol_atol.at(ck_tile::number<1>{}) << " "
                  << (pass ? "PASS" : "FAIL") << std::endl;

        return pass;
    }
};
