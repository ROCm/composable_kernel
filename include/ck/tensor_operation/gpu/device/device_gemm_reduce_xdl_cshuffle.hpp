#pragma once
#include <iostream>
#include <sstream>
#include "device.hpp"
#include "device_gemm_reduce.hpp"
#include "common_header.hpp"
#include "tensor_layout.hpp"
#include "tensor_descriptor.hpp"
#include "tensor_descriptor_helper.hpp"
#include "gridwise_gemm_reduce_xdl_cshuffle_v1.hpp"

namespace ck {
namespace tensor_operation {
namespace device {

template <typename ALayout,
          typename BLayout,
          typename CLayout,
          typename ADataType,
          typename BDataType,
          typename CDataType,
          typename GemmAccDataType,
          typename CShuffleDataType,
          typename ReduceAccDataType,
          typename DDataType,
          typename AElementwiseOperation,
          typename BElementwiseOperation,
          typename CElementwiseOperation,
          typename DReduceOperation,
          GemmSpecialization_t GemmSpecialization,
          index_t NumGemmKPrefetchStage,
          ck::index_t BlockSize,
          ck::index_t MPerBlock,
          ck::index_t NPerBlock,
          ck::index_t KPerBlock,
          ck::index_t AK1,
          ck::index_t BK1,
          ck::index_t MPerXDL,
          ck::index_t NPerXDL,
          ck::index_t MXdlPerWave,
          ck::index_t NXdlPerWave,
          typename ABlockTransferThreadClusterLengths_AK0_M_AK1,
          typename ABlockTransferThreadClusterArrangeOrder,
          typename ABlockTransferSrcAccessOrder,
          ck::index_t ABlockTransferSrcVectorDim,
          ck::index_t ABlockTransferSrcScalarPerVector,
          ck::index_t ABlockTransferDstScalarPerVector_AK1,
          bool ABlockLdsExtraM,
          typename BBlockTransferThreadClusterLengths_BK0_N_BK1,
          typename BBlockTransferThreadClusterArrangeOrder,
          typename BBlockTransferSrcAccessOrder,
          ck::index_t BBlockTransferSrcVectorDim,
          ck::index_t BBlockTransferSrcScalarPerVector,
          ck::index_t BBlockTransferDstScalarPerVector_BK1,
          bool BBlockLdsExtraN,
          index_t CShuffleMXdlPerWavePerShuffle,
          index_t CShuffleNXdlPerWavePerShuffle,
          typename CShuffleBlockTransferClusterLengths_MBlock_MPerBlock_NBlock_NPerBlock,
          index_t CShuffleBlockTransferScalarPerVector_NPerBlock,
          typename CReduceThreadClusterLengths_MPerBlock_NPerBlock,
          index_t CReduceThreadLds2VGprCopySrcDstScalarPerVector_NPerBlock,
          index_t CReduceThreadVgpr2GlobalCopySrcDstScalarPerVector_MPerBlock>
struct DeviceGemmReduce_Xdl_CShuffle : public DeviceGemmReduce<AElementwiseOperation,
                                                               BElementwiseOperation,
                                                               CElementwiseOperation,
                                                               DReduceOperation>
{
    static constexpr auto I0 = Number<0>{};
    static constexpr auto I1 = Number<1>{};
    static constexpr auto I2 = Number<2>{};

    static auto MakeAGridDescriptor_AK0_M_AK1(index_t MRaw, index_t KRaw, index_t StrideA)
    {
        const auto a_grid_desc_mraw_kraw = [&]() {
            if constexpr(is_same_v<tensor_layout::gemm::RowMajor, ALayout>)
            {
                return make_naive_tensor_descriptor(make_tuple(MRaw, KRaw),
                                                    make_tuple(StrideA, I1));
            }
            else if constexpr(is_same_v<tensor_layout::gemm::ColumnMajor, ALayout>)
            {
                return make_naive_tensor_descriptor(make_tuple(MRaw, KRaw),
                                                    make_tuple(I1, StrideA));
            }
        }();

        const auto M = math::integer_divide_ceil(MRaw, MPerBlock) * MPerBlock;
        const auto K = math::integer_divide_ceil(KRaw, KPerBlock) * KPerBlock;

        const auto MPad = M - MRaw;
        const auto KPad = K - KRaw;

        if constexpr(GemmSpecialization == GemmSpecialization_t::MKPadding ||
                     GemmSpecialization == GemmSpecialization_t::MNKPadding)
        {
            // pad both M and N
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
        else if constexpr(GemmSpecialization == GemmSpecialization_t::MPadding ||
                          GemmSpecialization == GemmSpecialization_t::MNPadding)
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
        else if constexpr(GemmSpecialization == GemmSpecialization_t::KPadding ||
                          GemmSpecialization == GemmSpecialization_t::NKPadding)
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

    static auto MakeBGridDescriptor_BK0_N_BK1(index_t KRaw, index_t NRaw, index_t StrideB)
    {
        const auto b_grid_desc_nraw_kraw = [&]() {
            if constexpr(is_same<tensor_layout::gemm::RowMajor, BLayout>::value)
            {
                return make_naive_tensor_descriptor(make_tuple(NRaw, KRaw),
                                                    make_tuple(I1, StrideB));
            }
            else if constexpr(is_same<tensor_layout::gemm::ColumnMajor, BLayout>::value)
            {
                return make_naive_tensor_descriptor(make_tuple(NRaw, KRaw),
                                                    make_tuple(StrideB, I1));
            }
        }();

        const auto N = math::integer_divide_ceil(NRaw, NPerBlock) * NPerBlock;
        const auto K = math::integer_divide_ceil(KRaw, KPerBlock) * KPerBlock;

        const auto NPad = N - NRaw;
        const auto KPad = K - KRaw;

        if constexpr(GemmSpecialization == GemmSpecialization_t::NKPadding ||
                     GemmSpecialization == GemmSpecialization_t::MNKPadding)
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
        else if constexpr(GemmSpecialization == GemmSpecialization_t::NPadding ||
                          GemmSpecialization == GemmSpecialization_t::MNPadding)
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
        else if constexpr(GemmSpecialization == GemmSpecialization_t::KPadding ||
                          GemmSpecialization == GemmSpecialization_t::MKPadding)
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

    static auto MakeCGridDescriptor_M_N(index_t MRaw, index_t NRaw, index_t StrideC)
    {
        const auto c_grid_desc_mraw_nraw = [&]() {
            if constexpr(is_same<tensor_layout::gemm::RowMajor, CLayout>::value)
            {
                return make_naive_tensor_descriptor(make_tuple(MRaw, NRaw),
                                                    make_tuple(StrideC, I1));
            }
            else if constexpr(is_same<tensor_layout::gemm::ColumnMajor, CLayout>::value)
            {
                return make_naive_tensor_descriptor(make_tuple(MRaw, NRaw),
                                                    make_tuple(I1, StrideC));
            }
        }();

        const auto M = math::integer_divide_ceil(MRaw, MPerBlock) * MPerBlock;
        const auto N = math::integer_divide_ceil(NRaw, NPerBlock) * NPerBlock;

        const auto MPad = M - MRaw;
        const auto NPad = N - NRaw;

        if constexpr(GemmSpecialization == GemmSpecialization_t::MNPadding ||
                     GemmSpecialization == GemmSpecialization_t::MNKPadding)
        {
            // pad M and N
            return transform_tensor_descriptor(c_grid_desc_mraw_nraw,
                                        make_tuple(make_right_pad_transform(MRaw, MPad),
                                                   make_right_pad_transform(NRaw, NPad)),
                                        make_tuple(Sequence<0>{}, Sequence<1>{}),
                                        make_tuple(Sequence<0>{}, Sequence<1>{}));
        }
        else if constexpr(GemmSpecialization == GemmSpecialization_t::MPadding ||
                          GemmSpecialization == GemmSpecialization_t::MKPadding)
        {
            // pad M, but not N
            return transform_tensor_descriptor(
                c_grid_desc_mraw_nraw,
                make_tuple(make_right_pad_transform(MRaw, MPad), make_pass_through_transform(NRaw)),
                make_tuple(Sequence<0>{}, Sequence<1>{}),
                make_tuple(Sequence<0>{}, Sequence<1>{}));
        }
        else if constexpr(GemmSpecialization == GemmSpecialization_t::NPadding ||
                          GemmSpecialization == GemmSpecialization_t::NKPadding)
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

    // assume D is packed tensor
    static auto MakeDGridDescriptor_M(index_t MRaw)
    {
        const auto d_grid_desc_mraw = make_naive_tensor_descriptor_packed(make_tuple(MRaw));

        const auto M    = math::integer_divide_ceil(MRaw, MPerBlock) * MPerBlock;
        const auto MPad = M - MRaw;

        if constexpr(GemmSpecialization == GemmSpecialization_t::MPadding ||
                     GemmSpecialization == GemmSpecialization_t::MNPadding ||
                     GemmSpecialization == GemmSpecialization_t::MKPadding ||
                     GemmSpecialization == GemmSpecialization_t::MNKPadding)
        {
            // pad M
            return transform_tensor_descriptor(d_grid_desc_mraw,
                                               make_tuple(make_right_pad_transform(MRaw, MPad)),
                                               make_tuple(Sequence<0>{}),
                                               make_tuple(Sequence<0>{}));
        }
        else
        {
            // not pad M
            return d_grid_desc_mraw;
        }
    }

    using AGridDesc_AK0_M_AK1 = decltype(MakeAGridDescriptor_AK0_M_AK1(1, 1, 1));
    using BGridDesc_BK0_N_BK1 = decltype(MakeBGridDescriptor_BK0_N_BK1(1, 1, 1));
    using CGridDesc_M_N       = decltype(MakeCGridDescriptor_M_N(1, 1, 1));
    using DGridDesc_M         = decltype(MakeDGridDescriptor_M(1));

    // GridwiseGemm
    using GridwiseGemm = GridwiseGemmReduce_k0mk1_k0nk1_mn_xdl_cshuffle_v1<
        ADataType, // TODO: distinguish A/B datatype
        GemmAccDataType,
        CShuffleDataType,
        CDataType,
        ReduceAccDataType,
        DDataType,
        AElementwiseOperation,
        BElementwiseOperation,
        CElementwiseOperation,
        DReduceOperation,
        InMemoryDataOperationEnum_t::Set,
        InMemoryDataOperationEnum_t::AtomicAdd,
        AGridDesc_AK0_M_AK1,
        BGridDesc_BK0_N_BK1,
        CGridDesc_M_N,
        DGridDesc_M,
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
        CReduceThreadClusterLengths_MPerBlock_NPerBlock,
        CReduceThreadLds2VGprCopySrcDstScalarPerVector_NPerBlock,
        CReduceThreadVgpr2GlobalCopySrcDstScalarPerVector_MPerBlock>;

    // Argument
    struct Argument : public BaseArgument
    {
        Argument(const ADataType* p_a_grid,
                 const BDataType* p_b_grid,
                 CDataType* p_c_grid,
                 DDataType* p_d_grid,
                 index_t MRaw,
                 index_t NRaw,
                 index_t KRaw,
                 index_t StrideA,
                 index_t StrideB,
                 index_t StrideC,
                 AElementwiseOperation a_element_op,
                 BElementwiseOperation b_element_op,
                 CElementwiseOperation c_element_op,
                 DReduceOperation d_reduce_op)
            : p_a_grid_{p_a_grid},
              p_b_grid_{p_b_grid},
              p_c_grid_{p_c_grid},
              p_d_grid_{p_d_grid},
              a_grid_desc_ak0_m_ak1_{},
              b_grid_desc_bk0_n_bk1_{},
              c_grid_desc_m_n_{},
              d_grid_desc_m_{},
              c_grid_desc_mblock_mperblock_nblock_nperblock_{},
              d_grid_desc_mblock_mperblock_{},
              block_2_ctile_map_{},
              a_element_op_{a_element_op},
              b_element_op_{b_element_op},
              c_element_op_{c_element_op},
              d_reduce_op_{d_reduce_op}
        {
            a_grid_desc_ak0_m_ak1_ =
                DeviceGemmReduce_Xdl_CShuffle::MakeAGridDescriptor_AK0_M_AK1(MRaw, KRaw, StrideA);
            b_grid_desc_bk0_n_bk1_ =
                DeviceGemmReduce_Xdl_CShuffle::MakeBGridDescriptor_BK0_N_BK1(KRaw, NRaw, StrideB);
            c_grid_desc_m_n_ =
                DeviceGemmReduce_Xdl_CShuffle::MakeCGridDescriptor_M_N(MRaw, NRaw, StrideC);
            d_grid_desc_m_ = DeviceGemmReduce_Xdl_CShuffle::MakeDGridDescriptor_M(MRaw);

            if(GridwiseGemm::CheckValidity(
                   a_grid_desc_ak0_m_ak1_, b_grid_desc_bk0_n_bk1_, c_grid_desc_m_n_))
            {
                c_grid_desc_mblock_mperblock_nblock_nperblock_ =
                    GridwiseGemm::MakeCGridDescriptor_MBlock_MPerBlock_NBlock_NPerBlock(
                        c_grid_desc_m_n_);

                d_grid_desc_mblock_mperblock_ =
                    GridwiseGemm::MakeDGridDescriptor_MBlock_MPerBlock(d_grid_desc_m_);

                block_2_ctile_map_ = GridwiseGemm::MakeDefaultBlock2CTileMap(c_grid_desc_m_n_);
            }
        }

        //  private:
        const ADataType* p_a_grid_;
        const BDataType* p_b_grid_;
        CDataType* p_c_grid_;
        DDataType* p_d_grid_;
        AGridDesc_AK0_M_AK1 a_grid_desc_ak0_m_ak1_;
        BGridDesc_BK0_N_BK1 b_grid_desc_bk0_n_bk1_;
        CGridDesc_M_N c_grid_desc_m_n_;
        DGridDesc_M d_grid_desc_m_;
        typename GridwiseGemm::CGridDescriptor_MBlock_MPerBlock_NBlock_NPerBlock
            c_grid_desc_mblock_mperblock_nblock_nperblock_;
        typename GridwiseGemm::DGridDescriptor_MBlock_MPerBlock d_grid_desc_mblock_mperblock_;
        typename GridwiseGemm::DefaultBlock2CTileMap block_2_ctile_map_;
        AElementwiseOperation a_element_op_;
        BElementwiseOperation b_element_op_;
        CElementwiseOperation c_element_op_;
        DReduceOperation d_reduce_op_;
    };

    // Invoker
    struct Invoker : public BaseInvoker
    {
        using Argument = DeviceGemmReduce_Xdl_CShuffle::Argument;

        float Run(const Argument& arg, int nrepeat = 1)
        {
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

                std::cout << "arg.d_grid_desc_m_{ " << arg.d_grid_desc_m_.GetLength(I0) << "}"
                          << std::endl;
            }

            if(!GridwiseGemm::CheckValidity(
                   arg.a_grid_desc_ak0_m_ak1_, arg.b_grid_desc_bk0_n_bk1_, arg.c_grid_desc_m_n_))
            {
                throw std::runtime_error(
                    "wrong! GridwiseGemm_km_kn_m0m1n0n1_xdlops_v2r3 has invalid setting");
            }

            const index_t grid_size = GridwiseGemm::CalculateGridSize(arg.c_grid_desc_m_n_);

            const auto K0 = arg.a_grid_desc_ak0_m_ak1_.GetLength(I0);

            const bool has_main_k0_block_loop = GridwiseGemm::CalculateHasMainK0BlockLoop(K0);

            float ave_time = 0;

            if(has_main_k0_block_loop)
            {
                const auto kernel = kernel_gemm_reduce_xdl_cshuffle_v1<
                    GridwiseGemm,
                    ADataType, // TODO: distiguish A/B datatype
                    CDataType,
                    DDataType,
                    AElementwiseOperation,
                    BElementwiseOperation,
                    CElementwiseOperation,
                    DReduceOperation,
                    remove_reference_t<DeviceGemmReduce_Xdl_CShuffle::AGridDesc_AK0_M_AK1>,
                    remove_reference_t<DeviceGemmReduce_Xdl_CShuffle::BGridDesc_BK0_N_BK1>,
                    remove_reference_t<
                        typename GridwiseGemm::CGridDescriptor_MBlock_MPerBlock_NBlock_NPerBlock>,
                    remove_reference_t<typename GridwiseGemm::DGridDescriptor_MBlock_MPerBlock>,
                    remove_reference_t<typename GridwiseGemm::DefaultBlock2CTileMap>,
                    true>;

                ave_time =
                    launch_and_time_kernel(kernel,
                                           nrepeat,
                                           dim3(grid_size),
                                           dim3(BlockSize),
                                           0,
                                           arg.p_a_grid_,
                                           arg.p_b_grid_,
                                           arg.p_c_grid_,
                                           arg.p_d_grid_,
                                           arg.a_element_op_,
                                           arg.b_element_op_,
                                           arg.c_element_op_,
                                           arg.d_reduce_op_,
                                           arg.a_grid_desc_ak0_m_ak1_,
                                           arg.b_grid_desc_bk0_n_bk1_,
                                           arg.c_grid_desc_mblock_mperblock_nblock_nperblock_,
                                           arg.d_grid_desc_mblock_mperblock_,
                                           arg.block_2_ctile_map_);
            }
            else
            {
                const auto kernel = kernel_gemm_reduce_xdl_cshuffle_v1<
                    GridwiseGemm,
                    ADataType, // TODO: distiguish A/B datatype
                    CDataType,
                    DDataType,
                    AElementwiseOperation,
                    BElementwiseOperation,
                    CElementwiseOperation,
                    DReduceOperation,
                    remove_reference_t<DeviceGemmReduce_Xdl_CShuffle::AGridDesc_AK0_M_AK1>,
                    remove_reference_t<DeviceGemmReduce_Xdl_CShuffle::BGridDesc_BK0_N_BK1>,
                    remove_reference_t<
                        typename GridwiseGemm::CGridDescriptor_MBlock_MPerBlock_NBlock_NPerBlock>,
                    remove_reference_t<typename GridwiseGemm::DGridDescriptor_MBlock_MPerBlock>,
                    remove_reference_t<typename GridwiseGemm::DefaultBlock2CTileMap>,
                    false>;

                ave_time =
                    launch_and_time_kernel(kernel,
                                           nrepeat,
                                           dim3(grid_size),
                                           dim3(BlockSize),
                                           0,
                                           arg.p_a_grid_,
                                           arg.p_b_grid_,
                                           arg.p_c_grid_,
                                           arg.p_d_grid_,
                                           arg.a_element_op_,
                                           arg.b_element_op_,
                                           arg.c_element_op_,
                                           arg.d_reduce_op_,
                                           arg.a_grid_desc_ak0_m_ak1_,
                                           arg.b_grid_desc_bk0_n_bk1_,
                                           arg.c_grid_desc_mblock_mperblock_nblock_nperblock_,
                                           arg.d_grid_desc_mblock_mperblock_,
                                           arg.block_2_ctile_map_);
            }

            return ave_time;
        }

        // polymorphic
        float Run(const BaseArgument* p_arg, int nrepeat = 1) override
        {
            return Run(*dynamic_cast<const Argument*>(p_arg), nrepeat);
        }
    };

    static constexpr bool IsValidCompilationParameter()
    {
        // TODO: properly implement this check
        return true;
    }

    static bool IsSupportedArgument(const Argument& arg)
    {
        return GridwiseGemm::CheckValidity(
            arg.a_grid_desc_ak0_m_ak1_, arg.b_grid_desc_bk0_n_bk1_, arg.c_grid_desc_m_n_);
    }

    // polymorphic
    bool IsSupportedArgument(const BaseArgument* p_arg) override
    {
        return IsSupportedArgument(*dynamic_cast<const Argument*>(p_arg));
    }

    static auto MakeArgument(const ADataType* p_a,
                             const BDataType* p_b,
                             CDataType* p_c,
                             DDataType* p_d,
                             index_t MRaw,
                             index_t NRaw,
                             index_t KRaw,
                             index_t StrideA,
                             index_t StrideB,
                             index_t StrideC,
                             AElementwiseOperation a_element_op,
                             BElementwiseOperation b_element_op,
                             CElementwiseOperation c_element_op,
                             DReduceOperation d_reduce_op)
    {
        return Argument{p_a,
                        p_b,
                        p_c,
                        p_d,
                        MRaw,
                        NRaw,
                        KRaw,
                        StrideA,
                        StrideB,
                        StrideC,
                        a_element_op,
                        b_element_op,
                        c_element_op,
                        d_reduce_op};
    }

    static auto MakeInvoker() { return Invoker{}; }

    // polymorphic
    std::unique_ptr<BaseArgument> MakeArgumentPointer(const void* p_a,
                                                      const void* p_b,
                                                      void* p_c,
                                                      void* p_d,
                                                      index_t MRaw,
                                                      index_t NRaw,
                                                      index_t KRaw,
                                                      index_t StrideA,
                                                      index_t StrideB,
                                                      index_t StrideC,
                                                      AElementwiseOperation a_element_op,
                                                      BElementwiseOperation b_element_op,
                                                      CElementwiseOperation c_element_op,
                                                      DReduceOperation d_reduce_op) override
    {
        return std::make_unique<Argument>(static_cast<const ADataType*>(p_a),
                                          static_cast<const BDataType*>(p_b),
                                          static_cast<CDataType*>(p_c),
                                          static_cast<DDataType*>(p_d),
                                          MRaw,
                                          NRaw,
                                          KRaw,
                                          StrideA,
                                          StrideB,
                                          StrideC,
                                          a_element_op,
                                          b_element_op,
                                          c_element_op,
                                          d_reduce_op);
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
        str << "DeviceGemmReduce_Xdl_CShuffle"
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
