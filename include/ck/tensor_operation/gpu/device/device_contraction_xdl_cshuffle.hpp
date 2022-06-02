#pragma once
#include <iostream>
#include <sstream>
#include "device.hpp"
#include "device_contraction.hpp"
#include "common_header.hpp"
#include "tensor_layout.hpp"
#include "tensor_descriptor.hpp"
#include "tensor_descriptor_helper.hpp"
#include "gridwise_gemm_xdl_cshuffle_v1.hpp"
#include "tensor_operation/gpu/device/gemm_specialization.hpp"
#include "device_prop.hpp"

namespace ck {
namespace tensor_operation {
namespace device {

// Assume:
//   A[M0, M1, M2, ..., K0, K1, K2...]
//   B[K0, K1, K2, ..., N0, N1, N2...]
//   C[M0, M1, M2, ..., N0, N1, N2...]
template <index_t NumDimM,
          index_t NumDimN,
          index_t NumDimK,
          typename ADataType,
          typename BDataType,
          typename CDataType,
          typename GemmAccDataType,
          typename CShuffleDataType,
          typename AElementwiseOperation,
          typename BElementwiseOperation,
          typename CElementwiseOperation,
          GemmSpecialization GemmSpec,
          index_t NumGemmKPrefetchStage,
          index_t BlockSize,
          index_t MPerBlock,
          index_t NPerBlock,
          index_t KPerBlock,
          index_t AK1,
          index_t BK1,
          index_t MPerXDL,
          index_t NPerXDL,
          index_t MXdlPerWave,
          index_t NXdlPerWave,
          typename ABlockTransferThreadClusterLengths_AK0_M_AK1,
          typename ABlockTransferThreadClusterArrangeOrder,
          typename ABlockTransferSrcAccessOrder,
          index_t ABlockTransferSrcVectorDim,
          index_t ABlockTransferSrcScalarPerVector,
          index_t ABlockTransferDstScalarPerVector_AK1,
          bool ABlockLdsExtraM,
          typename BBlockTransferThreadClusterLengths_BK0_N_BK1,
          typename BBlockTransferThreadClusterArrangeOrder,
          typename BBlockTransferSrcAccessOrder,
          index_t BBlockTransferSrcVectorDim,
          index_t BBlockTransferSrcScalarPerVector,
          index_t BBlockTransferDstScalarPerVector_BK1,
          bool BBlockLdsExtraN,
          index_t CShuffleMXdlPerWavePerShuffle,
          index_t CShuffleNXdlPerWavePerShuffle,
          typename CShuffleBlockTransferClusterLengths_MBlock_MPerBlock_NBlock_NPerBlock,
          index_t CShuffleBlockTransferScalarPerVector_NPerBlock,
          LoopScheduler LoopSched = make_default_loop_scheduler()>
struct DeviceContraction_Xdl_CShuffle : public DeviceContraction<NumDimM,
                                                                 NumDimN,
                                                                 NumDimK,
                                                                 AElementwiseOperation,
                                                                 BElementwiseOperation,
                                                                 CElementwiseOperation>
{
    using DeviceOp = DeviceContraction_Xdl_CShuffle;

    static constexpr auto I0 = Number<0>{};
    static constexpr auto I1 = Number<1>{};
    static constexpr auto I2 = Number<2>{};

    // assume A[M0, M1, M2, ..., K0, K1, K2...]
    static auto MakeAGridDescriptor_AK0_M_AK1(const std::vector<index_t>& a_lengths_vec,
                                              const std::vector<index_t>& a_strides_vec)
    {
        assert(a_lengths_vec.size() == NumDimM + NumDimK &&
               a_strides_vec.size() == NumDimM + NumDimK);

        const auto to_tuple = [&](auto& vec, auto Num) {
            return generate_tuple([&](auto i) { return vec[i]; }, Num);
        };

        const auto a_lengths = to_tuple(a_lengths_vec, Number<NumDimM + NumDimK>{});
        const auto a_strides = to_tuple(a_strides_vec, Number<NumDimM + NumDimK>{});

        // dimension Ids for M0, M1, ...
        constexpr auto mDimIds = typename arithmetic_sequence_gen<0, NumDimM, 1>::type{};

        // dimension Ids for K0, K1, ...
        constexpr auto kDimIds =
            typename arithmetic_sequence_gen<NumDimM, NumDimM + NumDimK, 1>::type{};

        // lengths for M0, M1, ...
        const auto mLengths = get_container_subset(a_lengths, mDimIds);

        // lengths for K0, K1, ...
        const auto kLengths = get_container_subset(a_lengths, kDimIds);

        // naive tensor A[M0, M1, M2, ..., K0, K1, K2...]
        const auto a_grid_desc_ms_ks = make_naive_tensor_descriptor(a_lengths, a_strides);

        // transformed tensor A[MRaw = M0 * M1 * M2 * ... , KRaw = K0 * K1 * K2 * ...]
        const auto a_grid_desc_mraw_kraw = transform_tensor_descriptor(
            a_grid_desc_ms_ks,
            make_tuple(make_merge_transform(mLengths), make_merge_transform(kLengths)),
            make_tuple(mDimIds, kDimIds),
            make_tuple(Sequence<0>{}, Sequence<1>{}));

        const auto MRaw = a_grid_desc_mraw_kraw.GetLength(I0);
        const auto KRaw = a_grid_desc_mraw_kraw.GetLength(I1);

        const auto M = math::integer_divide_ceil(MRaw, MPerBlock) * MPerBlock;
        const auto K = math::integer_divide_ceil(KRaw, KPerBlock) * KPerBlock;

        const auto MPad = M - MRaw;
        const auto KPad = K - KRaw;

        if constexpr(GemmSpec == GemmSpecialization::MKPadding ||
                     GemmSpec == GemmSpecialization::MNKPadding)
        {
            // pad both M and K
            assert(K % AK1 == 0);

            const auto AK0 = K / AK1;

            const auto a_grid_desc_m_k =
                transform_tensor_descriptor(a_grid_desc_mraw_kraw,
                                            make_tuple(make_right_pad_transform(MRaw, MPad),
                                                       make_right_pad_transform(KRaw, KPad)),
                                            make_tuple(Sequence<0>{}, Sequence<1>{}),
                                            make_tuple(Sequence<0>{}, Sequence<1>{}));

            const auto a_grid_desc_ak0_m_ak1 =
                transform_tensor_descriptor(a_grid_desc_m_k,
                                            make_tuple(make_unmerge_transform(make_tuple(AK0, AK1)),
                                                       make_pass_through_transform(M)),
                                            make_tuple(Sequence<1>{}, Sequence<0>{}),
                                            make_tuple(Sequence<0, 2>{}, Sequence<1>{}));

            return a_grid_desc_ak0_m_ak1;
        }
        else if constexpr(GemmSpec == GemmSpecialization::MPadding ||
                          GemmSpec == GemmSpecialization::MNPadding)
        {
            // pad M, but not K
            assert(KRaw % AK1 == 0);

            const auto AK0 = KRaw / AK1;

            const auto a_grid_desc_ak0_m_ak1 =
                transform_tensor_descriptor(a_grid_desc_mraw_kraw,
                                            make_tuple(make_unmerge_transform(make_tuple(AK0, AK1)),
                                                       make_right_pad_transform(MRaw, MPad)),
                                            make_tuple(Sequence<1>{}, Sequence<0>{}),
                                            make_tuple(Sequence<0, 2>{}, Sequence<1>{}));

            return a_grid_desc_ak0_m_ak1;
        }
        else if constexpr(GemmSpec == GemmSpecialization::KPadding ||
                          GemmSpec == GemmSpecialization::NKPadding)
        {
            // pad K, but not M
            assert(K % AK1 == 0);

            const auto AK0 = K / AK1;

            const auto a_grid_desc_m_k = transform_tensor_descriptor(
                a_grid_desc_mraw_kraw,
                make_tuple(make_pass_through_transform(MRaw), make_right_pad_transform(KRaw, KPad)),
                make_tuple(Sequence<0>{}, Sequence<1>{}),
                make_tuple(Sequence<0>{}, Sequence<1>{}));

            const auto a_grid_desc_ak0_m_ak1 =
                transform_tensor_descriptor(a_grid_desc_m_k,
                                            make_tuple(make_unmerge_transform(make_tuple(AK0, AK1)),
                                                       make_pass_through_transform(MRaw)),
                                            make_tuple(Sequence<1>{}, Sequence<0>{}),
                                            make_tuple(Sequence<0, 2>{}, Sequence<1>{}));

            return a_grid_desc_ak0_m_ak1;
        }
        else
        {
            // not pad M or K
            assert(KRaw % AK1 == 0);

            const auto AK0 = KRaw / AK1;

            const auto a_grid_desc_ak0_m_ak1 =
                transform_tensor_descriptor(a_grid_desc_mraw_kraw,
                                            make_tuple(make_unmerge_transform(make_tuple(AK0, AK1)),
                                                       make_pass_through_transform(MRaw)),
                                            make_tuple(Sequence<1>{}, Sequence<0>{}),
                                            make_tuple(Sequence<0, 2>{}, Sequence<1>{}));

            return a_grid_desc_ak0_m_ak1;
        }
    }

    // assume B[K0, K1, K2, ..., N0, N1, N2...]
    static auto MakeBGridDescriptor_BK0_N_BK1(const std::vector<index_t>& b_lengths_vec,
                                              const std::vector<index_t>& b_strides_vec)
    {
        assert(b_lengths_vec.size() == NumDimN + NumDimK &&
               b_strides_vec.size() == NumDimN + NumDimK);

        const auto to_tuple = [&](auto& vec, auto Num) {
            return generate_tuple([&](auto i) { return vec[i]; }, Num);
        };

        const auto b_lengths = to_tuple(b_lengths_vec, Number<NumDimN + NumDimK>{});
        const auto b_strides = to_tuple(b_strides_vec, Number<NumDimN + NumDimK>{});

        // dimension Ids for K0, K1, ...
        constexpr auto kDimIds = typename arithmetic_sequence_gen<0, NumDimK, 1>::type{};

        // dimension Ids for N0, N1, ...
        constexpr auto nDimIds =
            typename arithmetic_sequence_gen<NumDimK, NumDimK + NumDimN, 1>::type{};

        // lengths for K0, K1, ...
        const auto kLengths = get_container_subset(b_lengths, kDimIds);

        // lengths for N0, N1, ...
        const auto nLengths = get_container_subset(b_lengths, nDimIds);

        // naive tensor B[K0, K1, K2..., N0, N1, N2, ...]
        const auto b_grid_desc_ks_ns = make_naive_tensor_descriptor(b_lengths, b_strides);

        // transformed tensor B[KRaw = K0 * K1 * K2 * ... , NRaw = N0 * N1 * N2 * ... ]
        const auto b_grid_desc_nraw_kraw = transform_tensor_descriptor(
            b_grid_desc_ks_ns,
            make_tuple(make_merge_transform(kLengths), make_merge_transform(nLengths)),
            make_tuple(kDimIds, nDimIds),
            make_tuple(Sequence<1>{}, Sequence<0>{}));

        const auto NRaw = b_grid_desc_nraw_kraw.GetLength(I0);
        const auto KRaw = b_grid_desc_nraw_kraw.GetLength(I1);

        const auto N = math::integer_divide_ceil(NRaw, NPerBlock) * NPerBlock;
        const auto K = math::integer_divide_ceil(KRaw, KPerBlock) * KPerBlock;

        const auto NPad = N - NRaw;
        const auto KPad = K - KRaw;

        if constexpr(GemmSpec == GemmSpecialization::NKPadding ||
                     GemmSpec == GemmSpecialization::MNKPadding)
        {
            // pad both N and K
            assert(K % BK1 == 0);

            const auto BK0 = K / BK1;

            const auto b_grid_desc_n_k =
                transform_tensor_descriptor(b_grid_desc_nraw_kraw,
                                            make_tuple(make_right_pad_transform(NRaw, NPad),
                                                       make_right_pad_transform(KRaw, KPad)),
                                            make_tuple(Sequence<0>{}, Sequence<1>{}),
                                            make_tuple(Sequence<0>{}, Sequence<1>{}));

            const auto b_grid_desc_bk0_n_bk1 =
                transform_tensor_descriptor(b_grid_desc_n_k,
                                            make_tuple(make_unmerge_transform(make_tuple(BK0, BK1)),
                                                       make_pass_through_transform(N)),
                                            make_tuple(Sequence<1>{}, Sequence<0>{}),
                                            make_tuple(Sequence<0, 2>{}, Sequence<1>{}));

            return b_grid_desc_bk0_n_bk1;
        }
        else if constexpr(GemmSpec == GemmSpecialization::NPadding ||
                          GemmSpec == GemmSpecialization::MNPadding)
        {
            // pad N, but not K
            assert(KRaw % BK1 == 0);

            const auto BK0 = KRaw / BK1;

            const auto b_grid_desc_bk0_n_bk1 =
                transform_tensor_descriptor(b_grid_desc_nraw_kraw,
                                            make_tuple(make_unmerge_transform(make_tuple(BK0, BK1)),
                                                       make_right_pad_transform(NRaw, NPad)),
                                            make_tuple(Sequence<1>{}, Sequence<0>{}),
                                            make_tuple(Sequence<0, 2>{}, Sequence<1>{}));

            return b_grid_desc_bk0_n_bk1;
        }
        else if constexpr(GemmSpec == GemmSpecialization::KPadding ||
                          GemmSpec == GemmSpecialization::MKPadding)
        {
            // pad K, but not N
            assert(K % BK1 == 0);

            const auto BK0 = K / BK1;

            const auto b_grid_desc_n_k = transform_tensor_descriptor(
                b_grid_desc_nraw_kraw,
                make_tuple(make_pass_through_transform(NRaw), make_right_pad_transform(KRaw, KPad)),
                make_tuple(Sequence<0>{}, Sequence<1>{}),
                make_tuple(Sequence<0>{}, Sequence<1>{}));

            const auto b_grid_desc_bk0_n_bk1 =
                transform_tensor_descriptor(b_grid_desc_n_k,
                                            make_tuple(make_unmerge_transform(make_tuple(BK0, BK1)),
                                                       make_pass_through_transform(NRaw)),
                                            make_tuple(Sequence<1>{}, Sequence<0>{}),
                                            make_tuple(Sequence<0, 2>{}, Sequence<1>{}));

            return b_grid_desc_bk0_n_bk1;
        }
        else
        {
            // not pad N or K
            assert(KRaw % BK1 == 0);

            const auto BK0 = KRaw / BK1;

            const auto b_grid_desc_bk0_n_bk1 =
                transform_tensor_descriptor(b_grid_desc_nraw_kraw,
                                            make_tuple(make_unmerge_transform(make_tuple(BK0, BK1)),
                                                       make_pass_through_transform(NRaw)),
                                            make_tuple(Sequence<1>{}, Sequence<0>{}),
                                            make_tuple(Sequence<0, 2>{}, Sequence<1>{}));

            return b_grid_desc_bk0_n_bk1;
        }
    }

    // assume C[M0, M1, M2, ..., N0, N1, N2...]
    static auto MakeCGridDescriptor_M_N(const std::vector<index_t>& c_lengths_vec,
                                        const std::vector<index_t>& c_strides_vec)
    {
        assert(c_lengths_vec.size() == NumDimM + NumDimN &&
               c_strides_vec.size() == NumDimM + NumDimN);

        const auto to_tuple = [&](auto& vec, auto Num) {
            return generate_tuple([&](auto i) { return vec[i]; }, Num);
        };

        const auto c_lengths = to_tuple(c_lengths_vec, Number<NumDimM + NumDimN>{});
        const auto c_strides = to_tuple(c_strides_vec, Number<NumDimM + NumDimN>{});

        // dimension Ids for M0, M1, ...
        constexpr auto mDimIds = typename arithmetic_sequence_gen<0, NumDimM, 1>::type{};

        // dimension Ids for N0, N1, ...
        constexpr auto nDimIds =
            typename arithmetic_sequence_gen<NumDimM, NumDimM + NumDimN, 1>::type{};

        // lengths for M0, M1, ...
        const auto mLengths = get_container_subset(c_lengths, mDimIds);

        // lengths for K0, K1, ...
        const auto nLengths = get_container_subset(c_lengths, nDimIds);

        // naive tensor C[M0, M1, M2, ..., N0, N1, N2...]
        const auto c_grid_desc_ms_ns = make_naive_tensor_descriptor(c_lengths, c_strides);

        // transformed tensor C[MRaw = M0 * M1 * M2 * ... , NRaw = N0 * N1 * N2 * ...]
        const auto c_grid_desc_mraw_nraw = transform_tensor_descriptor(
            c_grid_desc_ms_ns,
            make_tuple(make_merge_transform(mLengths), make_merge_transform(nLengths)),
            make_tuple(mDimIds, nDimIds),
            make_tuple(Sequence<0>{}, Sequence<1>{}));

        const auto MRaw = c_grid_desc_mraw_nraw.GetLength(I0);
        const auto NRaw = c_grid_desc_mraw_nraw.GetLength(I1);

        const auto M = math::integer_divide_ceil(MRaw, MPerBlock) * MPerBlock;
        const auto N = math::integer_divide_ceil(NRaw, NPerBlock) * NPerBlock;

        const auto MPad = M - MRaw;
        const auto NPad = N - NRaw;

        if constexpr(GemmSpec == GemmSpecialization::MNPadding ||
                     GemmSpec == GemmSpecialization::MNKPadding)
        {
            // pad M and N
            return transform_tensor_descriptor(c_grid_desc_mraw_nraw,
                                               make_tuple(make_right_pad_transform(MRaw, MPad),
                                                          make_right_pad_transform(NRaw, NPad)),
                                               make_tuple(Sequence<0>{}, Sequence<1>{}),
                                               make_tuple(Sequence<0>{}, Sequence<1>{}));
        }
        else if constexpr(GemmSpec == GemmSpecialization::MPadding ||
                          GemmSpec == GemmSpecialization::MKPadding)
        {
            // pad M, but not N
            return transform_tensor_descriptor(
                c_grid_desc_mraw_nraw,
                make_tuple(make_right_pad_transform(MRaw, MPad), make_pass_through_transform(NRaw)),
                make_tuple(Sequence<0>{}, Sequence<1>{}),
                make_tuple(Sequence<0>{}, Sequence<1>{}));
        }
        else if constexpr(GemmSpec == GemmSpecialization::NPadding ||
                          GemmSpec == GemmSpecialization::NKPadding)
        {
            // pad N, but not M
            return transform_tensor_descriptor(
                c_grid_desc_mraw_nraw,
                make_tuple(make_pass_through_transform(MRaw), make_right_pad_transform(NRaw, NPad)),
                make_tuple(Sequence<0>{}, Sequence<1>{}),
                make_tuple(Sequence<0>{}, Sequence<1>{}));
        }
        else
        {
            // not pad M or N
            return c_grid_desc_mraw_nraw;
        }
    }

    using AGridDesc_AK0_M_AK1 =
        decltype(MakeAGridDescriptor_AK0_M_AK1(std::vector<index_t>{}, std::vector<index_t>{}));
    using BGridDesc_BK0_N_BK1 =
        decltype(MakeBGridDescriptor_BK0_N_BK1(std::vector<index_t>{}, std::vector<index_t>{}));
    using CGridDesc_M_N =
        decltype(MakeCGridDescriptor_M_N(std::vector<index_t>{}, std::vector<index_t>{}));

    // GridwiseGemm
    using GridwiseGemm = GridwiseGemm_k0mk1_k0nk1_mn_xdl_cshuffle_v1<
        ADataType, // TODO: distinguish A/B datatype
        GemmAccDataType,
        CShuffleDataType,
        CDataType,
        AElementwiseOperation,
        BElementwiseOperation,
        CElementwiseOperation,
        InMemoryDataOperationEnum::Set,
        AGridDesc_AK0_M_AK1,
        BGridDesc_BK0_N_BK1,
        CGridDesc_M_N,
        NumGemmKPrefetchStage,
        BlockSize,
        MPerBlock,
        NPerBlock,
        KPerBlock,
        AK1,
        BK1,
        MPerXDL,
        NPerXDL,
        MXdlPerWave,
        NXdlPerWave,
        ABlockTransferThreadClusterLengths_AK0_M_AK1,
        ABlockTransferThreadClusterArrangeOrder,
        ABlockTransferSrcAccessOrder,
        ABlockTransferSrcVectorDim,
        ABlockTransferSrcScalarPerVector,
        ABlockTransferDstScalarPerVector_AK1,
        false,
        ABlockLdsExtraM,
        BBlockTransferThreadClusterLengths_BK0_N_BK1,
        BBlockTransferThreadClusterArrangeOrder,
        BBlockTransferSrcAccessOrder,
        BBlockTransferSrcVectorDim,
        BBlockTransferSrcScalarPerVector,
        BBlockTransferDstScalarPerVector_BK1,
        false,
        BBlockLdsExtraN,
        CShuffleMXdlPerWavePerShuffle,
        CShuffleNXdlPerWavePerShuffle,
        CShuffleBlockTransferClusterLengths_MBlock_MPerBlock_NBlock_NPerBlock,
        CShuffleBlockTransferScalarPerVector_NPerBlock,
        LoopSched>;

    // Argument
    struct Argument : public BaseArgument
    {
        Argument(const ADataType* p_a_grid,
                 const BDataType* p_b_grid,
                 CDataType* p_c_grid,
                 std::vector<index_t> a_lengths,
                 std::vector<index_t> a_strides,
                 std::vector<index_t> b_lengths,
                 std::vector<index_t> b_strides,
                 std::vector<index_t> c_lengths,
                 std::vector<index_t> c_strides,
                 AElementwiseOperation a_element_op,
                 BElementwiseOperation b_element_op,
                 CElementwiseOperation c_element_op)
            : p_a_grid_{p_a_grid},
              p_b_grid_{p_b_grid},
              p_c_grid_{p_c_grid},
              a_element_op_{a_element_op},
              b_element_op_{b_element_op},
              c_element_op_{c_element_op},
              a_grid_desc_ak0_m_ak1_{DeviceOp::MakeAGridDescriptor_AK0_M_AK1(a_lengths, a_strides)},
              b_grid_desc_bk0_n_bk1_{DeviceOp::MakeBGridDescriptor_BK0_N_BK1(b_lengths, b_strides)},
              c_grid_desc_m_n_{DeviceOp::MakeCGridDescriptor_M_N(c_lengths, c_strides)},
              c_grid_desc_mblock_mperblock_nblock_nperblock_{},
              block_2_ctile_map_{GridwiseGemm::MakeDefaultBlock2CTileMap(c_grid_desc_m_n_)},
              a_mz_length_{},
              a_mz_stride_{},
              a_kz_length_{},
              a_kz_stride_{},
              b_nz_length_{},
              b_nz_stride_{},
              b_kz_length_{},
              b_kz_stride_{},
              c_mz_length_{},
              c_mz_stride_{},
              c_nz_length_{},
              c_nz_stride_{}
        {
            if(GridwiseGemm::CheckValidity(a_grid_desc_ak0_m_ak1_,
                                           b_grid_desc_bk0_n_bk1_,
                                           c_grid_desc_m_n_,
                                           block_2_ctile_map_))
            {
                c_grid_desc_mblock_mperblock_nblock_nperblock_ =
                    GridwiseGemm::MakeCGridDescriptor_MBlock_MPerBlock_NBlock_NPerBlock(
                        c_grid_desc_m_n_);
            }

            // for sanity check of vector load/store
            a_mz_length_ = a_lengths[NumDimM - 1];
            a_mz_stride_ = a_strides[NumDimM - 1];

            a_kz_length_ = a_lengths[NumDimM + NumDimK - 1];
            a_kz_stride_ = a_strides[NumDimM + NumDimK - 1];

            b_nz_length_ = b_lengths[NumDimK - 1];
            b_nz_stride_ = b_strides[NumDimK - 1];

            b_kz_length_ = b_lengths[NumDimK + NumDimN - 1];
            b_kz_stride_ = b_strides[NumDimK + NumDimN - 1];

            c_mz_length_ = b_lengths[NumDimM - 1];
            c_mz_stride_ = b_strides[NumDimM - 1];

            c_nz_length_ = b_lengths[NumDimM + NumDimN - 1];
            c_nz_stride_ = b_strides[NumDimM + NumDimN - 1];
        }

        //  private:
        const ADataType* p_a_grid_;
        const BDataType* p_b_grid_;
        CDataType* p_c_grid_;

        AElementwiseOperation a_element_op_;
        BElementwiseOperation b_element_op_;
        CElementwiseOperation c_element_op_;

        AGridDesc_AK0_M_AK1 a_grid_desc_ak0_m_ak1_;
        BGridDesc_BK0_N_BK1 b_grid_desc_bk0_n_bk1_;
        CGridDesc_M_N c_grid_desc_m_n_;
        typename GridwiseGemm::CGridDescriptor_MBlock_MPerBlock_NBlock_NPerBlock
            c_grid_desc_mblock_mperblock_nblock_nperblock_;
        typename GridwiseGemm::DefaultBlock2CTileMap block_2_ctile_map_;

        // for sanity check of vector load/store
        index_t a_mz_length_;
        index_t a_mz_stride_;
        index_t a_kz_length_;
        index_t a_kz_stride_;
        index_t b_nz_length_;
        index_t b_nz_stride_;
        index_t b_kz_length_;
        index_t b_kz_stride_;
        index_t c_mz_length_;
        index_t c_mz_stride_;
        index_t c_nz_length_;
        index_t c_nz_stride_;
    };

    // Invoker
    struct Invoker : public BaseInvoker
    {
        using Argument = DeviceOp::Argument;

        float Run(const Argument& arg, const StreamConfig& stream_config = StreamConfig{})
        {
#if 0
            {
                std::cout << "arg.a_grid_desc_ak0_m_ak1_{"
                          << arg.a_grid_desc_ak0_m_ak1_.GetLength(I0) << ", "
                          << arg.a_grid_desc_ak0_m_ak1_.GetLength(I1) << ", "
                          << arg.a_grid_desc_ak0_m_ak1_.GetLength(I2) << "}" << std::endl;

                std::cout << "arg.b_grid_desc_bk0_n_bk1_{"
                          << arg.b_grid_desc_bk0_n_bk1_.GetLength(I0) << ", "
                          << arg.b_grid_desc_bk0_n_bk1_.GetLength(I1) << ", "
                          << arg.b_grid_desc_bk0_n_bk1_.GetLength(I2) << "}" << std::endl;

                std::cout << "arg.c_grid_desc_m_n_{ " << arg.c_grid_desc_m_n_.GetLength(I0) << ", "
                          << arg.c_grid_desc_m_n_.GetLength(I1) << "}" << std::endl;
            }
#endif

            if(!GridwiseGemm::CheckValidity(arg.a_grid_desc_ak0_m_ak1_,
                                            arg.b_grid_desc_bk0_n_bk1_,
                                            arg.c_grid_desc_m_n_,
                                            arg.block_2_ctile_map_))
            {
                throw std::runtime_error("wrong! GridwiseGemm has invalid setting");
            }

            const index_t grid_size =
                arg.block_2_ctile_map_.CalculateGridSize(arg.c_grid_desc_m_n_);

            const auto K =
                arg.a_grid_desc_ak0_m_ak1_.GetLength(I0) * arg.a_grid_desc_ak0_m_ak1_.GetLength(I2);

            float ave_time = 0;

            if(GridwiseGemm::CalculateHasMainKBlockLoop(K))
            {
                const auto kernel = kernel_gemm_xdl_cshuffle_v1<
                    GridwiseGemm,
                    ADataType, // TODO: distiguish A/B datatype
                    CDataType,
                    AElementwiseOperation,
                    BElementwiseOperation,
                    CElementwiseOperation,
                    DeviceOp::AGridDesc_AK0_M_AK1,
                    DeviceOp::BGridDesc_BK0_N_BK1,
                    typename GridwiseGemm::CGridDescriptor_MBlock_MPerBlock_NBlock_NPerBlock,
                    typename GridwiseGemm::DefaultBlock2CTileMap,
                    true>;

                ave_time =
                    launch_and_time_kernel(stream_config,
                                           kernel,
                                           dim3(grid_size),
                                           dim3(BlockSize),
                                           0,
                                           arg.p_a_grid_,
                                           arg.p_b_grid_,
                                           arg.p_c_grid_,
                                           arg.a_element_op_,
                                           arg.b_element_op_,
                                           arg.c_element_op_,
                                           arg.a_grid_desc_ak0_m_ak1_,
                                           arg.b_grid_desc_bk0_n_bk1_,
                                           arg.c_grid_desc_mblock_mperblock_nblock_nperblock_,
                                           arg.block_2_ctile_map_);
            }
            else
            {
                const auto kernel = kernel_gemm_xdl_cshuffle_v1<
                    GridwiseGemm,
                    ADataType, // TODO: distiguish A/B datatype
                    CDataType,
                    AElementwiseOperation,
                    BElementwiseOperation,
                    CElementwiseOperation,
                    DeviceOp::AGridDesc_AK0_M_AK1,
                    DeviceOp::BGridDesc_BK0_N_BK1,
                    typename GridwiseGemm::CGridDescriptor_MBlock_MPerBlock_NBlock_NPerBlock,
                    typename GridwiseGemm::DefaultBlock2CTileMap,
                    false>;
                ave_time =
                    launch_and_time_kernel(stream_config,
                                           kernel,
                                           dim3(grid_size),
                                           dim3(BlockSize),
                                           0,
                                           arg.p_a_grid_,
                                           arg.p_b_grid_,
                                           arg.p_c_grid_,
                                           arg.a_element_op_,
                                           arg.b_element_op_,
                                           arg.c_element_op_,
                                           arg.a_grid_desc_ak0_m_ak1_,
                                           arg.b_grid_desc_bk0_n_bk1_,
                                           arg.c_grid_desc_mblock_mperblock_nblock_nperblock_,
                                           arg.block_2_ctile_map_);
            }

            return ave_time;
        }

        // polymorphic
        float Run(const BaseArgument* p_arg,
                  const StreamConfig& stream_config = StreamConfig{}) override
        {
            return Run(*dynamic_cast<const Argument*>(p_arg), stream_config);
        }
    };

    static constexpr bool IsValidCompilationParameter()
    {
        // TODO: properly implement this check
        return true;
    }

    static bool IsSupportedArgument(const Argument& arg)
    {
        if(!(ck::get_device_name() == "gfx908" || ck::get_device_name() == "gfx90a"))
        {
            return false;
        }

        if(!GridwiseGemm::CheckValidity(arg.a_grid_desc_ak0_m_ak1_,
                                        arg.b_grid_desc_bk0_n_bk1_,
                                        arg.c_grid_desc_m_n_,
                                        arg.block_2_ctile_map_))
        {
            return false;
        }

        // FIXME check vector load/store

        return true;
    }

    // polymorphic
    bool IsSupportedArgument(const BaseArgument* p_arg) override
    {
        return IsSupportedArgument(*dynamic_cast<const Argument*>(p_arg));
    }

    static auto MakeArgument(const ADataType* p_a,
                             const BDataType* p_b,
                             CDataType* p_c,
                             std::vector<index_t> a_lengths,
                             std::vector<index_t> a_strides,
                             std::vector<index_t> b_lengths,
                             std::vector<index_t> b_strides,
                             std::vector<index_t> c_lengths,
                             std::vector<index_t> c_strides,
                             AElementwiseOperation a_element_op,
                             BElementwiseOperation b_element_op,
                             CElementwiseOperation c_element_op)
    {
        return Argument{p_a,
                        p_b,
                        p_c,
                        a_lengths,
                        a_strides,
                        b_lengths,
                        b_strides,
                        c_lengths,
                        c_strides,
                        a_element_op,
                        b_element_op,
                        c_element_op};
    }

    static auto MakeInvoker() { return Invoker{}; }

    // polymorphic
    std::unique_ptr<BaseArgument> MakeArgumentPointer(const void* p_a,
                                                      const void* p_b,
                                                      void* p_c,
                                                      std::vector<index_t> a_lengths,
                                                      std::vector<index_t> a_strides,
                                                      std::vector<index_t> b_lengths,
                                                      std::vector<index_t> b_strides,
                                                      std::vector<index_t> c_lengths,
                                                      std::vector<index_t> c_strides,
                                                      AElementwiseOperation a_element_op,
                                                      BElementwiseOperation b_element_op,
                                                      CElementwiseOperation c_element_op) override
    {
        return std::make_unique<Argument>(static_cast<const ADataType*>(p_a),
                                          static_cast<const BDataType*>(p_b),
                                          static_cast<CDataType*>(p_c),
                                          a_lengths,
                                          a_strides,
                                          b_lengths,
                                          b_strides,
                                          c_lengths,
                                          c_strides,
                                          a_element_op,
                                          b_element_op,
                                          c_element_op);
    }

    // polymorphic
    std::unique_ptr<BaseInvoker> MakeInvokerPointer() override
    {
        return std::make_unique<Invoker>(Invoker{});
    }

    // polymorphic
    std::string GetTypeString() const override
    {
        auto str = std::stringstream();

        // clang-format off
        str << "DeviceContraction_Xdl_CShuffle"
            << "<"
            << BlockSize << ", "
            << MPerBlock << ", "
            << NPerBlock << ", "
            << KPerBlock << ", "
            << AK1 << ", "
            << BK1
            << ">";
        // clang-format on

        return str.str();
    }
};

} // namespace device
} // namespace tensor_operation
} // namespace ck
