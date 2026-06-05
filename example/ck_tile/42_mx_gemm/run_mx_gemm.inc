// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// Calculate relative and absolute error thresholds for MX GEMM
template <typename ADataType, typename BDataType, typename AccDataType, typename CDataType>
auto calculate_rtol_atol(const ck_tile::index_t K, const float max_accumulated_value)
{
    using ComputeType =
        std::conditional_t<sizeof(ADataType) < sizeof(BDataType), ADataType, BDataType>;
    // Calculate thresholds
    const auto rtol = ck_tile::get_relative_threshold<ComputeType, CDataType, AccDataType>(K);
    const auto atol = ck_tile::get_absolute_threshold<ComputeType, CDataType, AccDataType>(
        max_accumulated_value, K);
    return ck_tile::make_tuple(rtol, atol);
}

template <typename ADataType,
          typename BDataType,
          typename AccDataType,
          typename GemmConfig,
          bool UsePersistentKernel,
          typename ALayout,
          typename BLayout,
          typename CLayout>
int run_mx_gemm_with_layouts(int argc, char* argv[], ALayout, BLayout, CLayout)
{
    auto [result, arg_parser] = create_args(argc, argv);
    if(!result)
        return -1;

    ck_tile::index_t M        = arg_parser.get_int("m");
    ck_tile::index_t N        = arg_parser.get_int("n");
    ck_tile::index_t K        = arg_parser.get_int("k");
    ck_tile::index_t stride_A = arg_parser.get_int("stride_a");
    ck_tile::index_t stride_B = arg_parser.get_int("stride_b");
    ck_tile::index_t stride_C = arg_parser.get_int("stride_c");
    int validation            = arg_parser.get_int("v");
    int n_warmup              = arg_parser.get_int("warmup");
    int n_repeat              = arg_parser.get_int("repeat");
    int kbatch                = arg_parser.get_int("split_k");
    int init_method           = arg_parser.get_int("init");

    using CDataType = ck_tile::fp16_t;

    // Use get_default_stride helper for automatic leading dimension calculation (only if not
    // explicitly provided)
    if(stride_A == 0)
        stride_A = ck_tile::get_default_stride(M, K, 0, is_row_major(ALayout{}));
    if(stride_B == 0)
        stride_B = ck_tile::get_default_stride(K, N, 0, is_row_major(BLayout{}));
    if(stride_C == 0)
        stride_C = ck_tile::get_default_stride(M, N, 0, is_row_major(CLayout{}));

    ck_tile::HostTensor<ADataType> a_host(
        ck_tile::host_tensor_descriptor(M, K, stride_A, is_row_major(ALayout{})));
    ck_tile::HostTensor<BDataType> b_host(
        ck_tile::host_tensor_descriptor(K, N, stride_B, is_row_major(BLayout{})));
    ck_tile::HostTensor<CDataType> c_host(
        ck_tile::host_tensor_descriptor(M, N, stride_C, is_row_major(CLayout{})));

    // Scale tensors - follow parent matrix layouts for optimal memory access
    // A scales: [M, K/32] with A's layout
    // B scales: [K/32, N] with B's layout
    using ScaleType               = ck_tile::e8m0_t;
    ck_tile::index_t scale_k_size = K / 32;

    // Follow A/BLayout to get the layouts for the scale tensors
    ck_tile::index_t stride_scale_a =
        ck_tile::get_default_stride(M, scale_k_size, 0, is_row_major(ALayout{}));
    ck_tile::index_t stride_scale_b =
        ck_tile::get_default_stride(scale_k_size, N, 0, is_row_major(BLayout{}));

    ck_tile::HostTensor<ScaleType> scale_a_host(
        ck_tile::host_tensor_descriptor(M, scale_k_size, stride_scale_a, is_row_major(ALayout{})));
    ck_tile::HostTensor<ScaleType> scale_b_host(
        ck_tile::host_tensor_descriptor(scale_k_size, N, stride_scale_b, is_row_major(BLayout{})));
    std::mt19937 gen(42);
    std::uniform_int_distribution<std::uint32_t> fill_seed(0, 500);

    auto gen_scales = [&](auto& scales, float range_min, float range_max) {
        // e8m0_t is basically an exponent of float32
        ck_tile::HostTensor<float> pow2(scales.get_lengths());
        ck_tile::FillUniformDistributionIntegerValue<float>{range_min, range_max, fill_seed(gen)}(
            pow2);
        scales.ForEach([&](auto& self, const auto& i) {
            self(i) = static_cast<ScaleType>(std::exp2(pow2(i)));
        });
    };
    switch(init_method)
    {
    case 0:
        // Initialize A, B, and scales to random values
        ck_tile::FillUniformDistribution<ADataType>{-2.f, 2.f, fill_seed(gen)}(a_host);
        ck_tile::FillUniformDistribution<BDataType>{-2.f, 2.f, fill_seed(gen)}(b_host);
        gen_scales(scale_a_host, -2, 2);
        gen_scales(scale_b_host, -2, 2);
        break;
    case 1:
        // Initialize A, B, and scales to 1.0
        ck_tile::FillConstant<ADataType>{ADataType(1.f)}(a_host);
        ck_tile::FillConstant<BDataType>{BDataType(1.f)}(b_host);
        gen_scales(scale_a_host, 0, 0);
        gen_scales(scale_b_host, 0, 0);
        break;
    case 2:
        // Initialize A and B with random values but with constant 1.0 scales
        ck_tile::FillUniformDistribution<ADataType>{-2.f, 2.f, fill_seed(gen)}(a_host);
        ck_tile::FillUniformDistribution<BDataType>{-2.f, 2.f, fill_seed(gen)}(b_host);
        gen_scales(scale_a_host, 0, 0);
        gen_scales(scale_b_host, 0, 0);
        break;
    }

    // Compute effective XdlPack sizes based on GemmConfig tile dimensions
    constexpr ck_tile::index_t MPerXdl_      = GemmConfig::M_Warp_Tile;
    constexpr ck_tile::index_t NPerXdl_      = GemmConfig::N_Warp_Tile;
    constexpr ck_tile::index_t KPerXdl_      = GemmConfig::K_Warp_Tile;
    constexpr ck_tile::index_t MIterPerWarp_ = GemmConfig::M_Tile / (GemmConfig::M_Warp * MPerXdl_);
    constexpr ck_tile::index_t NIterPerWarp_ = GemmConfig::N_Tile / (GemmConfig::N_Warp * NPerXdl_);
    constexpr ck_tile::index_t KIterPerWarp_ = GemmConfig::K_Tile / KPerXdl_;

    constexpr ck_tile::index_t MXdlPackEff = (MIterPerWarp_ >= 2 && MIterPerWarp_ % 2 == 0) ? 2 : 1;
    constexpr ck_tile::index_t NXdlPackEff = (NIterPerWarp_ >= 2 && NIterPerWarp_ % 2 == 0) ? 2 : 1;
    constexpr ck_tile::index_t KXdlPackEff = (KIterPerWarp_ >= 2 && KIterPerWarp_ % 2 == 0) ? 2 : 1;

    // Pack scales: [M, K/32] e8m0_t -> [M/MXdlPackEff, K/32/KXdlPackEff] int32_t
    // Original unpacked tensors are kept for CPU reference validation
    constexpr ck_tile::index_t XdlMNThread = GemmConfig::M_Warp_Tile;
    constexpr ck_tile::index_t XdlKThread  = 64 / XdlMNThread;

    auto scale_a_packed =
        ck_tile::packScalesMNxK<MXdlPackEff, KXdlPackEff, XdlMNThread, XdlKThread>(scale_a_host,
                                                                                   true);
    auto scale_b_packed =
        ck_tile::packScalesMNxK<NXdlPackEff, KXdlPackEff, XdlMNThread, XdlKThread>(scale_b_host,
                                                                                   false);

    const auto b_host_for_device = [&]() {
        if constexpr(GemmConfig::Preshuffle)
            if constexpr(GemmConfig::TiledMMAPermuteN)
                return ck_tile::shuffle_b_permuteN<GemmConfig, BDataType, NXdlPackEff>(b_host);
            else
                return ck_tile::shuffle_b<GemmConfig>(b_host);
        else
            return b_host;
    }();

    const auto scale_a_host_for_device = [&]() {
        if constexpr(GemmConfig::Preshuffle)
            return ck_tile::preShuffleScale<GemmConfig::N_Warp_Tile>(scale_a_host, true);
        else
            return scale_a_packed;
    }();

    constexpr ck_tile::index_t XdlNThread = GemmConfig::N_Warp_Tile;
    constexpr ck_tile::index_t NPerBlock  = GemmConfig::N_Tile;
    constexpr ck_tile::index_t NWarp      = GemmConfig::N_Warp;

    const auto scale_b_host_for_device = [&]() {
        if constexpr(GemmConfig::Preshuffle)
        {
            if constexpr(GemmConfig::TiledMMAPermuteN)
                return ck_tile::preShuffleScalePermuteN<NWarp, NPerBlock, XdlNThread>(scale_b_host,
                                                                                      false);
            else
                return ck_tile::preShuffleScale<GemmConfig::N_Warp_Tile>(scale_b_host, false);
        }
        else
            return scale_b_packed;
    }();

    const auto scale_a_device_bytes = [&]() {
        if constexpr(GemmConfig::Preshuffle)
            return scale_a_host_for_device.get_element_space_size_in_bytes();
        else
            return scale_a_host_for_device.size() * sizeof(int32_t);
    }();

    const auto scale_b_device_bytes = [&]() {
        if constexpr(GemmConfig::Preshuffle)
            return scale_b_host_for_device.get_element_space_size_in_bytes();
        else
            return scale_b_host_for_device.size() * sizeof(int32_t);
    }();

    // Device buffers for A, B, C, and packed scale tensors
    ck_tile::DeviceMem a_dev_buf(a_host.get_element_space_size_in_bytes());
    ck_tile::DeviceMem b_dev_buf(b_host_for_device.get_element_space_size_in_bytes());
    ck_tile::DeviceMem c_dev_buf(c_host.get_element_space_size_in_bytes());
    ck_tile::DeviceMem scale_a_dev_buf(scale_a_device_bytes);
    ck_tile::DeviceMem scale_b_dev_buf(scale_b_device_bytes);

    a_dev_buf.ToDevice(a_host.data());
    b_dev_buf.ToDevice(b_host_for_device.data());
    c_dev_buf.SetZero();
    scale_a_dev_buf.ToDevice(scale_a_host_for_device.data());
    scale_b_dev_buf.ToDevice(scale_b_host_for_device.data());

    // Scale pointers - point to packed int32_t data, kernel reinterprets as int32_t*
    using ScaleM = ck_tile::MXScalePointer<ScaleType, 1, 32>;
    using ScaleN = ck_tile::MXScalePointer<ScaleType, 1, 32>;
    ScaleM scale_m(reinterpret_cast<ScaleType*>(scale_a_dev_buf.GetDeviceBuffer()));
    ScaleN scale_n(reinterpret_cast<ScaleType*>(scale_b_dev_buf.GetDeviceBuffer()));

    float ave_time = invoke_mx_gemm<GemmConfig,
                                    ADataType,
                                    BDataType,
                                    AccDataType,
                                    CDataType,
                                    ALayout,
                                    BLayout,
                                    CLayout,
                                    ScaleM,
                                    ScaleN,
                                    UsePersistentKernel>(a_dev_buf,
                                                         b_dev_buf,
                                                         c_dev_buf,
                                                         M,
                                                         N,
                                                         K,
                                                         stride_A,
                                                         stride_B,
                                                         stride_C,
                                                         kbatch,
                                                         scale_m,
                                                         scale_n,
                                                         n_warmup,
                                                         n_repeat);

    (void)ave_time;

    bool pass = true;
    if(validation > 0)
    {
        // get output data from device
        c_dev_buf.FromDevice(c_host.data());

        // compute reference
        ck_tile::HostTensor<CDataType> c_m_n_host_ref(
            ck_tile::host_tensor_descriptor(M, N, stride_C, is_row_major(CLayout{})));
        c_m_n_host_ref.SetZero();

        ck_tile::
            reference_mx_gemm<ADataType, BDataType, ScaleType, ScaleType, AccDataType, CDataType>(
                a_host, b_host, c_m_n_host_ref, scale_a_host, scale_b_host);

        const float max_accumulated_value =
            *std::max_element(c_m_n_host_ref.mData.begin(), c_m_n_host_ref.mData.end());
        const auto rtol_atol = calculate_rtol_atol<ADataType, BDataType, AccDataType, CDataType>(
            K, max_accumulated_value);
        const double rtol = rtol_atol.at(ck_tile::number<0>{});
        const double atol = rtol_atol.at(ck_tile::number<1>{});
        pass = ck_tile::check_err(c_host, c_m_n_host_ref, "Error: Incorrect results!", rtol, atol);

        std::cout << "Relative error threshold: " << rtol << " Absolute error threshold: " << atol
                  << std::endl;
        std::cout << "The CPU verification result is: " << (pass ? "correct" : "fail") << std::endl;
    }
    return pass ? 0 : -1;
}

template <typename GemmConfig>
int run_mx_gemm_example(int argc, char* argv[])
{
    auto [result, arg_parser] = create_args(argc, argv);
    if(!result)
        return -1;

    using Row = ck_tile::tensor_layout::gemm::RowMajor;
    using Col = ck_tile::tensor_layout::gemm::ColumnMajor;

    std::string mx_prec  = arg_parser.get_str("mx_prec");
    std::string a_layout = arg_parser.get_str("a_layout");
    std::string b_layout = arg_parser.get_str("b_layout");
    int preshuffle       = arg_parser.get_int("preshuffle");

    if(preshuffle != 0 && preshuffle != 1)
        throw std::runtime_error("preshuffle must be 0 or 1.");

    if(a_layout == "R" && b_layout == "C")
    {
        if(mx_prec == "fp4" || mx_prec == "fp4xfp4")
        {
            if(preshuffle == 1)
            {
                return run_mx_gemm_with_layouts<ck_tile::pk_fp4_t,
                                                ck_tile::pk_fp4_t,
                                                float,
                                                MXfp4_GemmConfig16_Preshuffle,
                                                true>(argc, argv, Row{}, Col{}, Row{});
            }
            else
            {
                return run_mx_gemm_with_layouts<ck_tile::pk_fp4_t,
                                                ck_tile::pk_fp4_t,
                                                float,
                                                GemmConfig,
                                                true>(argc, argv, Row{}, Col{}, Row{});
            }
        }
        else if(mx_prec == "fp8" || mx_prec == "fp8xfp8")
        {
            if(preshuffle == 1)
            {
                return run_mx_gemm_with_layouts<ck_tile::fp8_t,
                                                ck_tile::fp8_t,
                                                float,
                                                MXfp8_GemmConfig16_Preshuffle,
                                                true>(argc, argv, Row{}, Col{}, Row{});
            }
            else
            {
                return run_mx_gemm_with_layouts<ck_tile::fp8_t,
                                                ck_tile::fp8_t,
                                                float,
                                                GemmConfig,
                                                true>(argc, argv, Row{}, Col{}, Row{});
            }
        }
        else
        {
            throw std::runtime_error("Only fp4/8 is supported currently!");
        }
    }
    else
    {
        throw std::runtime_error("Only A=Row, B=Col layout is supported currently!");
    }
    return 0;
}
