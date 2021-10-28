#ifndef DEVICE_GEMM_XDL_HPP
#define DEVICE_GEMM_XDL_HPP

#include "common_header.hpp"
#include "tensor_descriptor.hpp"
#include "tensor_descriptor_helper.hpp"
#include "gridwise_gemm_xdlops_v2r3.hpp"

namespace ck {

namespace tensor_layout {

struct BaseTensorLayout
{
};

struct RowMajor : public BaseTensorLayout
{
};

struct ColumnMajor : public BaseTensorLayout
{
};

} // namespace tensor_layout

namespace tensor_operation {
namespace device {

struct BaseArgument
{
};

struct BaseInvoker
{
    float Run(const BaseArgument&, int = 1)
    {
        throw std::runtime_error(
            "wrong! BaseInvoker::Run(const BaseArgument&), should not get here");
    }

    virtual float Run(const BaseArgument*, int = 1)
    {
        throw std::runtime_error(
            "wrong! BaseInvoker::Run(const BaseArgument*), should not get here");
    }

    virtual ~BaseInvoker() {}
};

template <typename ADataType,
          typename BDataType,
          typename CDataType,
          typename AccDataType,
          typename ALayout,
          typename BLayout,
          typename CLayout,
          ck::index_t BlockSize,
          ck::index_t MPerBlock,
          ck::index_t NPerBlock,
          ck::index_t K0PerBlock,
          ck::index_t K1,
          ck::index_t MPerXDL,
          ck::index_t NPerXDL,
          ck::index_t MXdlPerWave,
          ck::index_t NXdlPerWave,
          typename ABlockTransferThreadSliceLengths_K0_M_K1,
          typename ABlockTransferThreadClusterLengths_K0_M_K1,
          typename ABlockTransferThreadClusterArrangeOrder,
          typename ABlockTransferSrcAccessOrder,
          ck::index_t ABlockTransferSrcVectorDim,
          ck::index_t ABlockTransferSrcScalarPerVector,
          ck::index_t ABlockTransferDstScalarPerVector_K1,
          typename BBlockTransferThreadSliceLengths_K0_N_K1,
          typename BBlockTransferThreadClusterLengths_K0_N_K1,
          typename BBlockTransferThreadClusterArrangeOrder,
          typename BBlockTransferSrcAccessOrder,
          ck::index_t BBlockTransferSrcVectorDim,
          ck::index_t BBlockTransferSrcScalarPerVector,
          ck::index_t BBlockTransferDstScalarPerVector_K1,
          ck::index_t CThreadTransferSrcDstVectorDim,
          ck::index_t CThreadTransferDstScalarPerVector,
          bool ABlockLdsAddExtraM,
          bool BBlockLdsAddExtraN>
struct DeviceGemmXdl
{
    static constexpr auto I0 = Number<0>{};
    static constexpr auto I1 = Number<1>{};
    static constexpr auto I2 = Number<2>{};

    static constexpr auto K1Number = Number<K1>{};

    static auto MakeAK0MK1GridDescriptor(index_t M, index_t K, index_t StrideA)
    {
        assert(K % K1 == 0);

        const index_t K0 = K / K1;

        const auto a_grid_desc_m_k = [&]() {
            if constexpr(is_same<tensor_layout::RowMajor, ALayout>::value)
            {
                return make_naive_tensor_descriptor(make_tuple(M, K), make_tuple(StrideA, I1));
            }
            else if constexpr(is_same<tensor_layout::ColumnMajor, ALayout>::value)
            {
                return make_naive_tensor_descriptor(make_tuple(M, K), make_tuple(I1, StrideA));
            }
        }();

        const auto a_grid_desc_k0_m_k1 =
            transform_tensor_descriptor(a_grid_desc_m_k,
                                        make_tuple(make_unmerge_transform(make_tuple(K0, K1Number)),
                                                   make_pass_through_transform(M)),
                                        make_tuple(Sequence<1>{}, Sequence<0>{}),
                                        make_tuple(Sequence<0, 2>{}, Sequence<1>{}));

        return a_grid_desc_k0_m_k1;
    }

    static auto MakeBK0NK1GridDescriptor(index_t K, index_t N, index_t StrideB)
    {
        assert(K % K1 == 0);

        const index_t K0 = K / K1;

        const auto b_grid_desc_k_n = [&]() {
            if constexpr(is_same<tensor_layout::RowMajor, BLayout>::value)
            {
                return make_naive_tensor_descriptor(make_tuple(K, N), make_tuple(StrideB, I1));
            }
            else if constexpr(is_same<tensor_layout::ColumnMajor, BLayout>::value)
            {
                return make_naive_tensor_descriptor(make_tuple(K, N), make_tuple(I1, StrideB));
            }
        }();

        const auto b_grid_desc_k0_n_k1 =
            transform_tensor_descriptor(b_grid_desc_k_n,
                                        make_tuple(make_unmerge_transform(make_tuple(K0, K1Number)),
                                                   make_pass_through_transform(N)),
                                        make_tuple(Sequence<0>{}, Sequence<1>{}),
                                        make_tuple(Sequence<0, 2>{}, Sequence<1>{}));

        return b_grid_desc_k0_n_k1;
    }

    static auto MakeCMNGridDescriptor(index_t M, index_t N, index_t StrideC)
    {
        if constexpr(is_same<tensor_layout::RowMajor, CLayout>::value)
        {
            return make_naive_tensor_descriptor(make_tuple(M, N), make_tuple(StrideC, I1));
        }
        else if constexpr(is_same<tensor_layout::ColumnMajor, CLayout>::value)
        {
            return make_naive_tensor_descriptor(make_tuple(M, N), make_tuple(I1, StrideC));
        }
    }

    using AGridDesc_K0_M_K1 = decltype(MakeAK0MK1GridDescriptor(1, 1, 1));
    using BGridDesc_K0_N_K1 = decltype(MakeBK0NK1GridDescriptor(1, 1, 1));
    using CGridDesc_M_N     = decltype(MakeCMNGridDescriptor(1, 1, 1));

    // TODO remove these hacks
    static constexpr auto a_k0_m_k1_grid_step_hacks =
        make_tuple(make_tuple(Sequence<0, 0, 0>{},   // 0+: K0
                              Sequence<0, 0, 0>{},   // 1+: M
                              Sequence<0, 0, 0>{}),  // 2+: K1
                   make_tuple(Sequence<0, 0, 0>{},   // 0-: K0
                              Sequence<0, 0, 0>{},   // 1-: M
                              Sequence<0, 0, 0>{})); // 2-: K1

    static constexpr auto b_k0_n_k1_grid_step_hacks =
        make_tuple(make_tuple(Sequence<0, 0, 0>{},   // 0+: K0
                              Sequence<0, 0, 0>{},   // 1+: N
                              Sequence<0, 0, 0>{}),  // 2+: K1
                   make_tuple(Sequence<0, 0, 0>{},   // 0-: K0
                              Sequence<0, 0, 0>{},   // 1-: N
                              Sequence<0, 0, 0>{})); // 2-: K1

    static constexpr auto c_m0_n0_m1_n1_m2_m3_m4_n2_grid_step_hacks =
        make_tuple(make_tuple(Sequence<0, 0, 0, 0, 0, 0, 0, 0, 0>{},   // 0+: M0
                              Sequence<0, 0, 0, 0, 0, 0, 0, 0, 0>{},   // 1+: N0
                              Sequence<0, 0, 0, 0, 0, 0, 0, 0, 0>{},   // 2+: M1
                              Sequence<0, 0, 0, 0, 0, 0, 0, 0, 0>{},   // 3+: N1
                              Sequence<0, 0, 0, 0, 0, 0, 0, 0, 0>{},   // 4+: M2
                              Sequence<0, 0, 0, 0, 0, 0, 0, 0, 0>{},   // 5+: M3
                              Sequence<0, 0, 0, 0, 0, 0, 0, 0, 0>{},   // 6+: M4
                              Sequence<0, 0, 0, 0, 0, 0, 0, 0, 0>{}),  // 7+: N2
                   make_tuple(Sequence<0, 0, 0, 0, 0, 0, 0, 0, 0>{},   // 0-: M0
                              Sequence<0, 0, 0, 0, 0, 0, 0, 0, 0>{},   // 1-: N0
                              Sequence<0, 0, 0, 0, 0, 0, 0, 0, 0>{},   // 2-: M1
                              Sequence<0, 0, 0, 0, 0, 0, 0, 0, 0>{},   // 3-: N1
                              Sequence<0, 0, 0, 0, 0, 0, 0, 0, 0>{},   // 4-: M2
                              Sequence<0, 0, 0, 0, 0, 0, 0, 0, 0>{},   // 5-: M3
                              Sequence<0, 0, 0, 0, 0, 0, 0, 0, 0>{},   // 6-: M4
                              Sequence<0, 0, 0, 0, 0, 0, 0, 0, 0>{})); // 7-: N2

    static constexpr auto a_k0_m_k1_grid_move_slice_window_step_hacks = Sequence<0, 0, 0>{};

    static constexpr auto b_k0_n_k1_grid_move_slice_window_step_hacks = Sequence<0, 0, 0>{};

    // GridwiseGemm
    using GridwiseGemm = GridwiseGemm_k0mk1_k0nk1_mn_xdlops_v2r3<
        BlockSize,
        ADataType, // TODO: distinguish A/B datatype
        AccDataType,
        CDataType,
        InMemoryDataOperationEnum_t::Set,
        AGridDesc_K0_M_K1,
        BGridDesc_K0_N_K1,
        CGridDesc_M_N,
        MPerBlock,
        NPerBlock,
        K0PerBlock,
        MPerXDL,
        NPerXDL,
        K1,
        MXdlPerWave,
        NXdlPerWave,
        ABlockTransferThreadSliceLengths_K0_M_K1,
        ABlockTransferThreadClusterLengths_K0_M_K1,
        ABlockTransferThreadClusterArrangeOrder,
        ABlockTransferSrcAccessOrder,
        ABlockTransferSrcVectorDim,
        ABlockTransferSrcScalarPerVector,
        ABlockTransferDstScalarPerVector_K1,
        false, // AThreadTransferSrcResetCoordinateAfterRun,
        BBlockTransferThreadSliceLengths_K0_N_K1,
        BBlockTransferThreadClusterLengths_K0_N_K1,
        BBlockTransferThreadClusterArrangeOrder,
        BBlockTransferSrcAccessOrder,
        BBlockTransferSrcVectorDim,
        BBlockTransferSrcScalarPerVector,
        BBlockTransferDstScalarPerVector_K1,
        false,                            // BThreadTransferSrcResetCoordinateAfterRun,
        Sequence<0, 2, 4, 5, 6, 1, 3, 7>, // CThreadTransferSrcDstAccessOrder,
        CThreadTransferSrcDstVectorDim,
        CThreadTransferDstScalarPerVector,
        decltype(a_k0_m_k1_grid_step_hacks),                   //  AGridStepHacks,
        decltype(b_k0_n_k1_grid_step_hacks),                   //  BGridStepHacks,
        decltype(c_m0_n0_m1_n1_m2_m3_m4_n2_grid_step_hacks),   //  CGridStepHacks,
        decltype(a_k0_m_k1_grid_move_slice_window_step_hacks), //  AGridMoveSliceWindowStepHacks,
        decltype(b_k0_n_k1_grid_move_slice_window_step_hacks), //  BGridMoveSliceWindowStepHacks,
        false,                                                 // CAccessOrderMRepeatNRepeat,
        ABlockLdsAddExtraM,
        BBlockLdsAddExtraN>;

    using CGridDesc_M0_N0_M1_N1_M2_M3_M4_N2 =
        decltype(GridwiseGemm::MakeCM0N0M1N1M2M3M4N2GridDescriptor(CGridDesc_M_N{}));

    using Block2CTileMap = decltype(GridwiseGemm::MakeCBlockClusterAdaptor(CGridDesc_M_N{}, 1, 1));

    // Argument
    struct Argument : public BaseArgument
    {
        Argument(const ADataType* p_a_grid,
                 const BDataType* p_b_grid,
                 CDataType* p_c_grid,
                 index_t M,
                 index_t N,
                 index_t K,
                 index_t StrideA,
                 index_t StrideB,
                 index_t StrideC,
                 index_t M01,
                 index_t N01)
            : p_a_grid_{p_a_grid},
              p_b_grid_{p_b_grid},
              p_c_grid_{p_c_grid},
              a_grid_desc_k0_m_k1_{DeviceGemmXdl::MakeAK0MK1GridDescriptor(M, K, StrideA)},
              b_grid_desc_k0_n_k1_{DeviceGemmXdl::MakeBK0NK1GridDescriptor(K, N, StrideB)},
              c_grid_desc_m_n_{DeviceGemmXdl::MakeCMNGridDescriptor(M, N, StrideC)},
              c_grid_desc_m0_n0_m1_n1_m2_m3_m4_n2_{
                  GridwiseGemm::MakeCM0N0M1N1M2M3M4N2GridDescriptor(c_grid_desc_m_n_)},
              block_2_c_tile_map_{
                  GridwiseGemm::MakeCBlockClusterAdaptor(c_grid_desc_m_n_, M01, N01)},
              M01_{M01},
              N01_{N01}
        {
        }

        //  private:
        const ADataType* p_a_grid_;
        const BDataType* p_b_grid_;
        CDataType* p_c_grid_;
        AGridDesc_K0_M_K1 a_grid_desc_k0_m_k1_;
        BGridDesc_K0_N_K1 b_grid_desc_k0_n_k1_;
        CGridDesc_M_N c_grid_desc_m_n_;
        CGridDesc_M0_N0_M1_N1_M2_M3_M4_N2 c_grid_desc_m0_n0_m1_n1_m2_m3_m4_n2_;
        Block2CTileMap block_2_c_tile_map_;
        index_t M01_;
        index_t N01_;
    };

    // Invoker
    struct Invoker : public BaseInvoker
    {
        using Argument = DeviceGemmXdl::Argument;

        float Run(const Argument& arg, int nrepeat = 1)
        {
            {
                std::cout << "arg.a_grid_desc_k0_m_k1_{" << arg.a_grid_desc_k0_m_k1_.GetLength(I0)
                          << ", " << arg.a_grid_desc_k0_m_k1_.GetLength(I1) << ", "
                          << arg.a_grid_desc_k0_m_k1_.GetLength(I2) << "}" << std::endl;

                std::cout << "arg.b_grid_desc_k0_n_k1_{" << arg.b_grid_desc_k0_n_k1_.GetLength(I0)
                          << ", " << arg.b_grid_desc_k0_n_k1_.GetLength(I1) << ", "
                          << arg.b_grid_desc_k0_n_k1_.GetLength(I2) << "}" << std::endl;

                std::cout << "arg.c_grid_desc_m_n_{ " << arg.c_grid_desc_m_n_.GetLength(I0) << ", "
                          << arg.c_grid_desc_m_n_.GetLength(I1) << "}" << std::endl;
            }

            if(!GridwiseGemm::CheckValidity(arg.a_grid_desc_k0_m_k1_,
                                            arg.b_grid_desc_k0_n_k1_,
                                            arg.c_grid_desc_m_n_,
                                            arg.M01_,
                                            arg.N01_))
            {
                throw std::runtime_error(
                    "wrong! GridwiseGemm_km_kn_m0m1n0n1_xdlops_v2r3 has invalid setting");
            }

            const index_t grid_size = GridwiseGemm::CalculateGridSize(arg.c_grid_desc_m_n_);

            const auto K0 = arg.a_grid_desc_k0_m_k1_.GetLength(I0);

            const bool has_main_k0_block_loop = GridwiseGemm::CalculateHasMainK0BlockLoop(K0);

            float ave_time = 0;

            if(has_main_k0_block_loop)
            {
                const auto kernel = kernel_gemm_xdlops_v2r3<
                    GridwiseGemm,
                    ADataType, // TODO: distiguish A/B datatype
                    CDataType,
                    remove_reference_t<DeviceGemmXdl::AGridDesc_K0_M_K1>,
                    remove_reference_t<DeviceGemmXdl::BGridDesc_K0_N_K1>,
                    remove_reference_t<DeviceGemmXdl::CGridDesc_M0_N0_M1_N1_M2_M3_M4_N2>,
                    remove_reference_t<DeviceGemmXdl::Block2CTileMap>,
                    true>;

                ave_time = launch_and_time_kernel(kernel,
                                                  nrepeat,
                                                  dim3(grid_size),
                                                  dim3(BlockSize),
                                                  0,
                                                  arg.p_a_grid_,
                                                  arg.p_b_grid_,
                                                  arg.p_c_grid_,
                                                  arg.a_grid_desc_k0_m_k1_,
                                                  arg.b_grid_desc_k0_n_k1_,
                                                  arg.c_grid_desc_m0_n0_m1_n1_m2_m3_m4_n2_,
                                                  arg.block_2_c_tile_map_);
            }
            else
            {
                const auto kernel = kernel_gemm_xdlops_v2r3<
                    GridwiseGemm,
                    ADataType, // TODO: distiguish A/B datatype
                    CDataType,
                    remove_reference_t<DeviceGemmXdl::AGridDesc_K0_M_K1>,
                    remove_reference_t<DeviceGemmXdl::BGridDesc_K0_N_K1>,
                    remove_reference_t<DeviceGemmXdl::CGridDesc_M0_N0_M1_N1_M2_M3_M4_N2>,
                    remove_reference_t<DeviceGemmXdl::Block2CTileMap>,
                    false>;

                ave_time = launch_and_time_kernel(kernel,
                                                  nrepeat,
                                                  dim3(grid_size),
                                                  dim3(BlockSize),
                                                  0,
                                                  arg.p_a_grid_,
                                                  arg.p_b_grid_,
                                                  arg.p_c_grid_,
                                                  arg.a_grid_desc_k0_m_k1_,
                                                  arg.b_grid_desc_k0_n_k1_,
                                                  arg.c_grid_desc_m0_n0_m1_n1_m2_m3_m4_n2_,
                                                  arg.block_2_c_tile_map_);
            }

            return ave_time;
        }

        float Run(const BaseArgument* p_arg) { return *dynamic_cast<const Argument*>(p_arg); }
    };

    static constexpr bool IsValidCompilationParameter()
    {
        // TODO: properly implement this check
        if constexpr(!(is_same<ALayout, tensor_layout::RowMajor>::value &&
                       is_same<BLayout, tensor_layout::ColumnMajor>::value &&
                       is_same<CLayout, tensor_layout::RowMajor>::value))
        {
            return false;
        }

        return true;
    }

    static bool IsSupportedArgument(const Argument& arg)
    {
        return GridwiseGemm::CheckValidity(arg.a_grid_desc_k0_m_k1_,
                                           arg.b_grid_desc_k0_n_k1_,
                                           arg.c_grid_desc_m_n_,
                                           arg.M01_,
                                           arg.N01_);
    }

    static auto MakeArgument(const ADataType* p_a,
                             const BDataType* p_b,
                             CDataType* p_c,
                             index_t M,
                             index_t N,
                             index_t K,
                             index_t StrideA,
                             index_t StrideB,
                             index_t StrideC)
    {
        return Argument{p_a, p_b, p_c, M, N, K, StrideA, StrideB, StrideC, 1, 1};
    }

    static auto MakeInvoker() { return Invoker{}; }
};

} // namespace device
} // namespace tensor_operation
} // namespace ck
#endif
