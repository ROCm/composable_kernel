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

// Pack [MN, K/32] e8m0_t scales into [MN/MNPack, K/32/KPack] int32_t
// Each int32_t contains MNPack * KPack e8m0_t values with byte layout matching
// the GPU tile distribution: values are XdlMNThread apart in M and XdlKThread apart in K.
//   byte[ik * MNPack + imn] = e8m0 at strided (mn, k) position
// kLast=true for A scales (layout [M, K/32]), kLast=false for B scales (layout [K/32, N])
template <ck_tile::index_t MNPack      = 2,
          ck_tile::index_t KPack       = 2,
          ck_tile::index_t XdlMNThread = 16,
          ck_tile::index_t XdlKThread  = 4>
auto packScalesMNxK(const ck_tile::HostTensor<ck_tile::e8m0_t>& src, bool kLast)
{
    auto src_lengths                    = src.get_lengths();
    const ck_tile::index_t MN           = kLast ? src_lengths[0] : src_lengths[1];
    const ck_tile::index_t K_scale      = kLast ? src_lengths[1] : src_lengths[0];
    const ck_tile::index_t MN_packed    = MN / MNPack;
    const ck_tile::index_t K_packed     = K_scale / KPack;
    const ck_tile::index_t total_packed = MN_packed * K_packed;

    // Output as flat vector of int32_t (row-major [MN/MNPack, K/32/KPack])
    std::vector<int32_t> packed(total_packed);

    for(ck_tile::index_t packed_mn = 0; packed_mn < MN_packed; packed_mn++)
    {
        for(ck_tile::index_t packed_k = 0; packed_k < K_packed; packed_k++)
        {
            int32_t val               = 0;
            ck_tile::index_t mn_lane  = packed_mn % XdlMNThread;
            ck_tile::index_t mn_group = packed_mn / XdlMNThread;
            ck_tile::index_t k_lane   = packed_k % XdlKThread;
            ck_tile::index_t k_group  = packed_k / XdlKThread;
            for(ck_tile::index_t ik = 0; ik < KPack; ik++)
            {
                for(ck_tile::index_t imn = 0; imn < MNPack; imn++)
                {
                    ck_tile::index_t byteIdx = ik * MNPack + imn;
                    ck_tile::index_t orig_mn =
                        mn_group * XdlMNThread * MNPack + imn * XdlMNThread + mn_lane;
                    ck_tile::index_t orig_k =
                        k_group * XdlKThread * KPack + ik * XdlKThread + k_lane;

                    ck_tile::e8m0_t v = kLast ? src(orig_mn, orig_k) : src(orig_k, orig_mn);
                    val |= (static_cast<int32_t>(v.get()) << (byteIdx * 8));
                }
            }
            packed[packed_mn * K_packed + packed_k] = val;
        }
    }
    return packed;
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
    int seed = 1234;
    switch(init_method)
    {
    case 0:
        // Initialize A, B, and scales to random values
        ck_tile::FillUniformDistribution<ADataType>{-2.f, 2.f, seed++}(a_host);
        ck_tile::FillUniformDistribution<BDataType>{-2.f, 2.f, seed++}(b_host);
        ck_tile::FillUniformDistribution<ScaleType>{0.001f, 10.f, seed++}(scale_a_host);
        ck_tile::FillUniformDistribution<ScaleType>{0.001f, 10.f, seed++}(scale_b_host);
        break;
    case 1:
        // Initialize A, B, and scales to 1.0
        ck_tile::FillConstant<ADataType>{ADataType(1.f)}(a_host);
        ck_tile::FillConstant<BDataType>{BDataType(1.f)}(b_host);
        ck_tile::FillConstant<ScaleType>{ScaleType(1.f)}(scale_a_host);
        ck_tile::FillConstant<ScaleType>{ScaleType(1.f)}(scale_b_host);
        break;
    case 2:
        // Initialize A and B with random values but with constant 1.0 scales
        ck_tile::FillUniformDistribution<ADataType>{-2.f, 2.f, seed++}(a_host);
        ck_tile::FillUniformDistribution<BDataType>{-2.f, 2.f, seed++}(b_host);
        ck_tile::FillConstant<ScaleType>{ScaleType(0.1f)}(scale_a_host);
        ck_tile::FillConstant<ScaleType>{ScaleType(0.1f)}(scale_b_host);
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

    // Pack scales: [M, K/32] e8m0_t → [M/MXdlPackEff, K/32/KXdlPackEff] int32_t
    // Original unpacked tensors are kept for CPU reference validation
    constexpr ck_tile::index_t XdlMNThread = GemmConfig::M_Warp_Tile;
    constexpr ck_tile::index_t XdlKThread  = 64 / XdlMNThread;

    auto scale_a_packed =
        packScalesMNxK<MXdlPackEff, KXdlPackEff, XdlMNThread, XdlKThread>(scale_a_host, true);
    auto scale_b_packed =
        packScalesMNxK<NXdlPackEff, KXdlPackEff, XdlMNThread, XdlKThread>(scale_b_host, false);

    // Device buffers for A, B, C, and packed scale tensors
    ck_tile::DeviceMem a_dev_buf(a_host.get_element_space_size_in_bytes());
    ck_tile::DeviceMem b_dev_buf(b_host.get_element_space_size_in_bytes());
    ck_tile::DeviceMem c_dev_buf(c_host.get_element_space_size_in_bytes());
    ck_tile::DeviceMem scale_a_dev_buf(scale_a_packed.size() * sizeof(int32_t));
    ck_tile::DeviceMem scale_b_dev_buf(scale_b_packed.size() * sizeof(int32_t));

    a_dev_buf.ToDevice(a_host.data());
    b_dev_buf.ToDevice(b_host.data());
    c_dev_buf.SetZero();
    scale_a_dev_buf.ToDevice(scale_a_packed.data());
    scale_b_dev_buf.ToDevice(scale_b_packed.data());

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

    if(a_layout == "R" && b_layout == "C")
    {
        if(mx_prec == "fp4" || mx_prec == "fp4xfp4")
        {
            return run_mx_gemm_with_layouts<ck_tile::pk_fp4_t,
                                            ck_tile::pk_fp4_t,
                                            float,
                                            MXfp4_GemmConfig16,
                                            true>(argc, argv, Row{}, Col{}, Row{});
        }
        else if(mx_prec == "fp8" || mx_prec == "fp8xfp8")
        {
            return run_mx_gemm_with_layouts<ck_tile::fp8_t,
                                            ck_tile::fp8_t,
                                            float,
                                            MXfp8_GemmConfig16,
                                            true>(argc, argv, Row{}, Col{}, Row{});
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
