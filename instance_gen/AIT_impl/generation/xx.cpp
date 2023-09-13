
        #pragma once

        #include <iostream>
        #include <sstream>

        #include "ck/utility/common_header.hpp"
        #include "ck/tensor_description/tensor_descriptor.hpp"
        #include "ck/tensor_description/tensor_descriptor_helper.hpp"
        #include "ck/tensor_operation/gpu/device/tensor_layout.hpp"
        #include "ck/tensor_operation/gpu/device/device_gemm.hpp"
        #include "ck/tensor_operation/gpu/device/gemm_specialization.hpp"
        #include "ck/tensor_operation/gpu/grid/gridwise_gemm_dl_v1r3.hpp"
        #include "ck/host_utility/device_prop.hpp"
        #include "ck/host_utility/kernel_launch.hpp"
        namespace ck {
        namespace tensor_operation {
        namespace device {

        template <
            typename ck::half_t,
            typename ck::half_t,
            typename ck::half_t,
            typename float,
            typename ck::tensor_layout::gemm::ColMajor,
            typename ck::tensor_layout::gemm::RowMajor,
            typename ck::tensor_layout::gemm::RowMajor,
            typename ck::tensor_operation::element_wise::PassThrough,
            typename ck::tensor_operation::element_wise::PassThrough,
            typename ck::tensor_operation::element_wise::PassThrough,
            ck::tensor_operation::device::GemmSpecialization::Default,
            256,
            128,
            128,
            16,
            2,
            4,
            4,
            1,
            typename S<8, 2>,
            typename S<8, 2>,
            typename S<2, 1, 4, 2>,
            typename S<8, 1,  32, 1>,
            typename S<0, 3, 1, 2>,
            typename S<0, 3, 1, 2>,
            typename S<1, 1, 4, 1>,
            typename S<0, 3, 1, 2>,
            typename S<1, 1, 4, 2>,
            typename S<2, 1, 4, 2>,
            typename S<8, 1, 32, 1>,
            typename S<0, 3, 1, 2>,
            typename S<0, 3, 1, 2>,
            typename S<1, 1, 4, 1>,
            typename S<0, 3, 1, 2>,
            typename S<1, 1, 4, 2>,
            typename S<0, 1, 2, 3, 4, 5>,
            5,
            4>
        struct DeviceGemmDl : public DeviceGemm<ck::tensor_layout::gemm::ColMajor,
                                                ck::tensor_layout::gemm::RowMajor,
                                                ck::tensor_layout::gemm::RowMajor,
                                                ck::half_t,
                                                ck::half_t,
                                                ck::half_t,
                                                ck::tensor_operation::element_wise::PassThrough,
                                                ck::tensor_operation::element_wise::PassThrough,
                                                ck::tensor_operation::element_wise::PassThrough>

        {
            static constexpr auto I0 = Number<0>{};
            static constexpr auto I1 = Number<1>{};
            static constexpr auto I2 = Number<2>{};
            static constexpr auto I3 = Number<3>{};
            static constexpr auto I4 = Number<4>{};
            static constexpr auto I5 = Number<5>{};

            static constexpr auto K1Number = Number<2>{};

            static auto MakeAGridDescriptor_K0_M_K1(index_t M, index_t K, index_t StrideA)
            {
                assert(K % 2 == 0);

                const index_t K0 = K / 2;

                const auto a_grid_desc_m_k = [&]() {
                    if constexpr(is_same<tensor_layout::gemm::RowMajor, ck::tensor_layout::gemm::ColMajor>::value)
                    {
                        return make_naive_tensor_descriptor(make_tuple(M, K), make_tuple(StrideA, I1));
                    }
                    else if constexpr(is_same<tensor_layout::gemm::ColumnMajor, ck::tensor_layout::gemm::ColMajor>::value)
                    {
                        return make_naive_tensor_descriptor(make_tuple(M, K), make_tuple(I1, StrideA));
                    }
                }();

                if constexpr(ck::tensor_operation::device::GemmSpecialization::Default == GemmSpecialization::MNPadding)
                {
                    const auto PadM = (128 - M % 128) % 128;

                    return transform_tensor_descriptor(
                        a_grid_desc_m_k,
                        make_tuple(make_unmerge_transform(make_tuple(K0, K1Number)),
                                    make_right_pad_transform(M, PadM)),
                        make_tuple(Sequence<1>{}, Sequence<0>{}),
                        make_tuple(Sequence<0, 2>{}, Sequence<1>{}));
                }
                else
                {
                    return transform_tensor_descriptor(
                        a_grid_desc_m_k,
                        make_tuple(make_unmerge_transform(make_tuple(K0, K1Number)),
                                    make_pass_through_transform(M)),
                        make_tuple(Sequence<1>{}, Sequence<0>{}),
                        make_tuple(Sequence<0, 2>{}, Sequence<1>{}));
                }
            }

            static auto MakeBGridDescriptor_K0_N_K1(index_t K, index_t N, index_t StrideB)
    {
        assert(K % 2 == 0);

        const index_t K0 = K / 2;

        const auto b_grid_desc_k_n = [&]() {
            if constexpr(is_same<tensor_layout::gemm::RowMajor, ck::tensor_layout::gemm::RowMajor>::value)
            {
                return make_naive_tensor_descriptor(make_tuple(K, N), make_tuple(StrideB, I1));
            }
            else if constexpr(is_same<tensor_layout::gemm::ColumnMajor, ${layout_B}>::value)
            {
                return make_naive_tensor_descriptor(make_tuple(K, N), make_tuple(I1, StrideB));
            }
        }();

        if constexpr(ck::tensor_operation::device::GemmSpecialization::Default == GemmSpecialization::MNPadding)
        {
            const auto PadN = (128 - N % 128) % 128;

            return transform_tensor_descriptor(
                b_grid_desc_k_n,
                make_tuple(make_unmerge_transform(make_tuple(K0, K1Number)),
                           make_right_pad_transform(N, PadN)),
                make_tuple(Sequence<0>{}, Sequence<1>{}),
                make_tuple(Sequence<0, 2>{}, Sequence<1>{}));
        }
        else
        {
            return transform_tensor_descriptor(
                b_grid_desc_k_n,
                make_tuple(make_unmerge_transform(make_tuple(K0, K1Number)),
                           make_pass_through_transform(N)),
                make_tuple(Sequence<0>{}, Sequence<1>{}),
                make_tuple(Sequence<0, 2>{}, Sequence<1>{}));
        }
    }

    static auto MakeCGridDescriptor_M_N(index_t M, index_t N, index_t StrideC)
    {
        const auto c_grid_desc_m_n = [&]() {
            if constexpr(is_same<tensor_layout::gemm::RowMajor, ck::tensor_layout::gemm::RowMajor>::value)
            {
                return make_naive_tensor_descriptor(make_tuple(M, N), make_tuple(StrideC, I1));
            }
            else if constexpr(is_same<tensor_layout::gemm::ColumnMajor, ck::tensor_layout::gemm::RowMajor>::value)
            {
                return make_naive_tensor_descriptor(make_tuple(M, N), make_tuple(I1, StrideC));
            }
        }();

        if constexpr(ck::tensor_operation::device::GemmSpecialization::Default == GemmSpecialization::MNPadding)
        {
            const auto PadM = (128 - M % 128) % 128;
            const auto PadN = (128 - N % 128) % 128;

            return transform_tensor_descriptor(
                c_grid_desc_m_n,
                make_tuple(make_right_pad_transform(M, PadM), make_right_pad_transform(N, PadN)),
                make_tuple(Sequence<0>{}, Sequence<1>{}),
                make_tuple(Sequence<0>{}, Sequence<1>{}));
        }
        else
        {

            return transform_tensor_descriptor(
                c_grid_desc_m_n,
                make_tuple(make_pass_through_transform(M), make_pass_through_transform(N)),
                make_tuple(Sequence<0>{}, Sequence<1>{}),
                make_tuple(Sequence<0>{}, Sequence<1>{}));
        }
    }

    using AGridDesc_K0_M_K1 = decltype(MakeAGridDescriptor_K0_M_K1(1, 1, 1));
    using BGridDesc_K0_N_K1 = decltype(MakeBGridDescriptor_K0_N_K1(1, 1, 1));
    using CGridDesc_M_N     = decltype(MakeCGridDescriptor_M_N(1, 1, 1));

        // GridwiseGemm
    using GridwiseGemm =
        GridwiseGemmDl_km_kn_mn_v1r3<BlockSize,
                                     ck::half_t,
                                     float,
                                     ck::half_t,
                                     InMemoryDataOperationEnum::Set,
                                     AGridDesc_K0_M_K1,
                                     BGridDesc_K0_N_K1,
                                     CGridDesc_M_N,
                                     128,
                                     128,
                                     16,
                                     2,
                                     4,
                                     4,
                                     1,
                                     S<8, 2>,
                                     S<8, 2>,
                                     S<2, 1, 4, 2>,
                                     S<8, 1,  32, 1>,
                                     S<0, 3, 1, 2>,
                                     S<0, 3, 1, 2>,
                                     S<1, 1, 4, 1>,
                                     S<0, 3, 1, 2>,
                                     S<1, 1, 4, 2>,
                                     S<2, 1, 4, 2>,
                                     S<8, 1, 32, 1>,
                                     S<0, 3, 1, 2>,
                                     S<0, 3, 1, 2>,
                                     S<1, 1, 4, 1>,
                                     S<0, 3, 1, 2>,
                                     S<1, 1, 4, 2>,
                                     S<0, 1, 2, 3, 4, 5>,
                                     5,
                                     4>;

    using AGridDesc_K0_M0_M1_K1 =
        decltype(GridwiseGemm::MakeAGridDescriptor_K0_M0_M1_K1(AGridDesc_K0_M_K1{}));
    using BGridDesc_K0_N0_N1_K1 =
        decltype(GridwiseGemm::MakeBGridDescriptor_K0_N0_N1_K1(BGridDesc_K0_N_K1{}));
    using CGridDesc_M0_M10_M11_N0_N10_N11 =
        decltype(GridwiseGemm::MakeCGridDescriptor_M0_M10_M11_N0_N10_N11(CGridDesc_M_N{}));
    using DefaultBlock2CTileMap =
        decltype(GridwiseGemm::MakeDefaultBlock2CTileMap(CGridDesc_M_N{}));

    // Argument
    struct Argument : public BaseArgument
    {
        Argument(const ck::half_t* p_a_grid,
                 const ck::half_t* p_b_grid,
                 ck::half_t* p_c_grid,
                 index_t M,
                 index_t N,
                 index_t K,
                 index_t StrideA,
                 index_t StrideB,
                 index_t StrideC,
                 index_t M01,
                 index_t N01,
                 ck::tensor_operation::element_wise::PassThrough a_element_op,
                 ck::tensor_operation::element_wise::PassThrough b_element_op,
                 ck::tensor_operation::element_wise::PassThrough c_element_op)
            : p_a_grid_{p_a_grid},
              p_b_grid_{p_b_grid},
              p_c_grid_{p_c_grid},
              a_grid_desc_k0_m0_m1_k1_{},
              b_grid_desc_k0_n0_n1_k1_{},
              c_grid_desc_m0_m10_m11_n0_n10_n11_{},
              block_2_ctile_map_{},
              M01_{M01},
              N01_{N01},
              a_element_op_{a_element_op},
              b_element_op_{b_element_op},
              c_element_op_{c_element_op}
        {
            a_grid_desc_k0_m_k1_ = DeviceGemmDl::MakeAGridDescriptor_K0_M_K1(M, K, StrideA);
            b_grid_desc_k0_n_k1_ = DeviceGemmDl::MakeBGridDescriptor_K0_N_K1(K, N, StrideB);
            c_grid_desc_m_n_     = DeviceGemmDl::MakeCGridDescriptor_M_N(M, N, StrideC);

            if(GridwiseGemm::CheckValidity(
                   a_grid_desc_k0_m_k1_, b_grid_desc_k0_n_k1_, c_grid_desc_m_n_))
            {
                a_grid_desc_k0_m0_m1_k1_ =
                    GridwiseGemm::MakeAGridDescriptor_K0_M0_M1_K1(a_grid_desc_k0_m_k1_);
                b_grid_desc_k0_n0_n1_k1_ =
                    GridwiseGemm::MakeBGridDescriptor_K0_N0_N1_K1(b_grid_desc_k0_n_k1_);
                c_grid_desc_m0_m10_m11_n0_n10_n11_ =
                    GridwiseGemm::MakeCGridDescriptor_M0_M10_M11_N0_N10_N11(c_grid_desc_m_n_);

                block_2_ctile_map_ = GridwiseGemm::MakeDefaultBlock2CTileMap(c_grid_desc_m_n_);
            }
        }

        //  private:
        const ck::half_t* p_a_grid_;
        const ck::half_t* p_b_grid_;
        ck::half_t* p_c_grid_;

        AGridDesc_K0_M_K1 a_grid_desc_k0_m_k1_;
        BGridDesc_K0_N_K1 b_grid_desc_k0_n_k1_;
        CGridDesc_M_N c_grid_desc_m_n_;

        AGridDesc_K0_M0_M1_K1 a_grid_desc_k0_m0_m1_k1_;
        BGridDesc_K0_N0_N1_K1 b_grid_desc_k0_n0_n1_k1_;
        CGridDesc_M0_M10_M11_N0_N10_N11 c_grid_desc_m0_m10_m11_n0_n10_n11_;

        DefaultBlock2CTileMap block_2_ctile_map_;

        // TODO: unused, but may be useful in future.
        index_t M01_;
        index_t N01_;

        // TODO: unused since gridwise_gemm_dl_v1r3 does NOT support prologue for the time being.
        ck::tensor_operation::element_wise::PassThrough a_element_op_;
        ck::tensor_operation::element_wise::PassThrough b_element_op_;
        ck::tensor_operation::element_wise::PassThrough c_element_op_;
    };

    // Invoker
    struct Invoker : public BaseInvoker
    {
        using Argument = DeviceGemmDl::Argument;

        float Run(const Argument& arg, const StreamConfig& stream_config = StreamConfig{})
        {
            {
                std::cout << "arg.a_grid_desc_k0_m0_m1_k1_{"
                          << arg.a_grid_desc_k0_m_k1_.GetLength(I0) << ", "
                          << arg.a_grid_desc_k0_m_k1_.GetLength(I1) << ", "
                          << arg.a_grid_desc_k0_m_k1_.GetLength(I2) << "}" << std::endl;

                std::cout << "arg.b_grid_desc_k0_n0_n1_k1_{"
                          << arg.b_grid_desc_k0_n_k1_.GetLength(I0) << ", "
                          << arg.b_grid_desc_k0_n_k1_.GetLength(I1) << ", "
                          << arg.b_grid_desc_k0_n_k1_.GetLength(I2) << "}" << std::endl;

                std::cout << "arg.c_grid_desc_m_n_{ " << arg.c_grid_desc_m_n_.GetLength(I0) << ", "
                          << arg.c_grid_desc_m_n_.GetLength(I1) << "}" << std::endl;
            }

            if(!GridwiseGemm::CheckValidity(
                   arg.a_grid_desc_k0_m_k1_, arg.b_grid_desc_k0_n_k1_, arg.c_grid_desc_m_n_))
            {
                throw std::runtime_error(
                    "wrong! GridwiseGemm_k0mk1_k0nk1_mn_xdl_v2r3 has invalid setting");
            }

            const index_t grid_size = GridwiseGemm::CalculateGridSize(
                arg.c_grid_desc_m_n_.GetLength(I0), arg.c_grid_desc_m_n_.GetLength(I1));

            const auto K0                    = arg.a_grid_desc_k0_m0_m1_k1_.GetLength(I0);
            const bool has_main_k_block_loop = GridwiseGemm::CalculateHasMainKBlockLoop(K0);
            const bool has_double_tail_k_block_loop =
                GridwiseGemm::CalculateHasDoubleTailKBlockLoop(K0);

            float ave_time = 0;

            if(has_main_k_block_loop && has_double_tail_k_block_loop)
            {
                const auto kernel =
                    kernel_gemm_dl_v1r3<GridwiseGemm,
                                        ck::half_t,
                                        ck::half_t,
                                        remove_reference_t<AGridDesc_K0_M0_M1_K1>,
                                        remove_reference_t<BGridDesc_K0_N0_N1_K1>,
                                        remove_reference_t<CGridDesc_M0_M10_M11_N0_N10_N11>,
                                        remove_reference_t<DefaultBlock2CTileMap>,
                                        true,
                                        true>;

                ave_time = launch_and_time_kernel(stream_config,
                                                  kernel,
                                                  dim3(grid_size),
                                                  dim3(256),
                                                  0,
                                                  arg.p_a_grid_,
                                                  arg.p_b_grid_,
                                                  arg.p_c_grid_,
                                                  arg.a_grid_desc_k0_m0_m1_k1_,
                                                  arg.b_grid_desc_k0_n0_n1_k1_,
                                                  arg.c_grid_desc_m0_m10_m11_n0_n10_n11_,
                                                  arg.block_2_ctile_map_);
            }
            else if(has_main_k_block_loop && !has_double_tail_k_block_loop)
            {
                const auto kernel =
                    kernel_gemm_dl_v1r3<GridwiseGemm,
                                        ck::half_t,
                                        ck::half_t,
                                        remove_reference_t<AGridDesc_K0_M0_M1_K1>,
                                        remove_reference_t<BGridDesc_K0_N0_N1_K1>,
                                        remove_reference_t<CGridDesc_M0_M10_M11_N0_N10_N11>,
                                        remove_reference_t<DefaultBlock2CTileMap>,
                                        true,
                                        false>;

                ave_time = launch_and_time_kernel(stream_config,
                                                  kernel,
                                                  dim3(grid_size),
                                                  dim3(256),
                                                  0,
                                                  arg.p_a_grid_,
                                                  arg.p_b_grid_,
                                                  arg.p_c_grid_,
                                                  arg.a_grid_desc_k0_m0_m1_k1_,
                                                  arg.b_grid_desc_k0_n0_n1_k1_,
                                                  arg.c_grid_desc_m0_m10_m11_n0_n10_n11_,
                                                  arg.block_2_ctile_map_);
            }
            else if(!has_main_k_block_loop && has_double_tail_k_block_loop)
            {
                const auto kernel =
                    kernel_gemm_dl_v1r3<GridwiseGemm,
                                        ck::half_t,
                                        ck::half_t,
                                        remove_reference_t<AGridDesc_K0_M0_M1_K1>,
                                        remove_reference_t<BGridDesc_K0_N0_N1_K1>,
                                        remove_reference_t<CGridDesc_M0_M10_M11_N0_N10_N11>,
                                        remove_reference_t<DefaultBlock2CTileMap>,
                                        false,
                                        true>;

                ave_time = launch_and_time_kernel(stream_config,
                                                  kernel,
                                                  dim3(grid_size),
                                                  dim3(256),
                                                  0,
                                                  arg.p_a_grid_,
                                                  arg.p_b_grid_,
                                                  arg.p_c_grid_,
                                                  arg.a_grid_desc_k0_m0_m1_k1_,
                                                  arg.b_grid_desc_k0_n0_n1_k1_,
                                                  arg.c_grid_desc_m0_m10_m11_n0_n10_n11_,
                                                  arg.block_2_ctile_map_);
            }
            else
            {
                const auto kernel =
                    kernel_gemm_dl_v1r3<GridwiseGemm,
                                        ck::half_t,
                                        ck::half_t,
                                        remove_reference_t<AGridDesc_K0_M0_M1_K1>,
                                        remove_reference_t<BGridDesc_K0_N0_N1_K1>,
                                        remove_reference_t<CGridDesc_M0_M10_M11_N0_N10_N11>,
                                        remove_reference_t<DefaultBlock2CTileMap>,
                                        false,
                                        false>;

                ave_time = launch_and_time_kernel(stream_config,
                                                  kernel,
                                                  dim3(grid_size),
                                                  dim3(256),
                                                  0,
                                                  arg.p_a_grid_,
                                                  arg.p_b_grid_,
                                                  arg.p_c_grid_,
                                                  arg.a_grid_desc_k0_m0_m1_k1_,
                                                  arg.b_grid_desc_k0_n0_n1_k1_,
                                                  arg.c_grid_desc_m0_m10_m11_n0_n10_n11_,
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
        if(ck::get_device_name() == "gfx906" || ck::get_device_name() == "gfx1030")
        {
            return GridwiseGemm::CheckValidity(
                arg.a_grid_desc_k0_m_k1_, arg.b_grid_desc_k0_n_k1_, arg.c_grid_desc_m_n_);
        }
        else
        {
            return false;
        }
    }

    // polymorphic
    bool IsSupportedArgument(const BaseArgument* p_arg) override
    {
        return IsSupportedArgument(*dynamic_cast<const Argument*>(p_arg));
    }

    static auto MakeArgument(const ck::half_t* p_a,
                             const ck::half_t* p_b,
                             ck::half_t* p_c,
                             index_t M,
                             index_t N,
                             index_t K,
                             index_t StrideA,
                             index_t StrideB,
                             index_t StrideC,
                             ck::tensor_operation::element_wise::PassThrough a_element_op,
                             ck::tensor_operation::element_wise::PassThrough b_element_op,
                             ck::tensor_operation::element_wise::PassThrough c_element_op)
    {
        return Argument{p_a,
                        p_b,
                        p_c,
                        M,
                        N,
                        K,
                        StrideA,
                        StrideB,
                        StrideC,
                        1,
                        1,
                        a_element_op,
                        b_element_op,
                        c_element_op};
    }

    static auto MakeInvoker() { return Invoker{}; }

    // polymorphic
    std::unique_ptr<BaseArgument> MakeArgumentPointer(const void* p_a,
                                                      const void* p_b,
                                                      void* p_c,
                                                      index_t M,
                                                      index_t N,
                                                      index_t K,
                                                      index_t StrideA,
                                                      index_t StrideB,
                                                      index_t StrideC,
                                                      ck::tensor_operation::element_wise::PassThrough a_element_op,
                                                      ck::tensor_operation::element_wise::PassThrough b_element_op,
                                                      ck::tensor_operation::element_wise::PassThrough c_element_op) override
    {
        return std::make_unique<Argument>(static_cast<const ck::half_t*>(p_a),
                                          static_cast<const ck::half_t*>(p_b),
                                          static_cast<ck::half_t*>(p_c),
                                          M,
                                          N,
                                          K,
                                          StrideA,
                                          StrideB,
                                          StrideC,
                                          1,
                                          1,
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
        str << "DeviceGemmDl"
            << "<"
            << 256 << ", "
            << 128 << ", "
            << 128 << ", "
            << 16 << ", "
            << 2 << ", "
            << 4 << ", "
            << 4 << ", "
            << 1
            << ">";
        // clang-format on

        return str.str();
    }
};

} // namespace device
} // namespace tensor_operation
} // namespace ck
    