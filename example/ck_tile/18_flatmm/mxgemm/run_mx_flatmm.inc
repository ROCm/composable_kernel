// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// Initialization proxy: the legacy ck_tile::pk_fp6x16_t type lacks the
// converters that FillUniformDistribution / type_convert / set_element rely
// on. We therefore initialize through a pk_fp6_t HostTensor and copy the
// raw bytes into the legacy tensor before the kernel and reference calls.
template <typename T>
using mx_flatmm_init_proxy_t =
    std::conditional_t<std::is_same_v<T, ck_tile::pk_fp6x16_t>, ck_tile::pk_fp6_t, T>;

template <typename PrecActType,
          typename PrecWeightType,
          typename CDataType,
          typename MXFlatmmArchTraits,
          bool UsePersistentKernel = false,
          typename ALayout,
          typename BLayout,
          typename CLayout>
int run_mx_flatmm_with_layouts(const ck_tile::ArgParser& arg_parser,
                               const ALayout a_layout = ALayout{},
                               const BLayout b_layout = BLayout{},
                               const CLayout c_layout = CLayout{})
{
    using ADataType   = PrecActType;
    using BDataType   = PrecWeightType;
    using AccDataType = float;

    // Init-side proxy types: collapse to ADataType / BDataType for everything
    // except ck_tile::pk_fp6x16_t (legacy fp6), which is initialized through a
    // byte-equivalent ck_tile::pk_fp6_t buffer and memcpy'd into the real tensor.
    using AInitType = mx_flatmm_init_proxy_t<ADataType>;
    using BInitType = mx_flatmm_init_proxy_t<BDataType>;

    // The proxy and the real type may have different sizeof (e.g. pk_fp6_t is
    // 16 bytes due to uint32x3_t alignment, while pk_fp6x16_t is 12 bytes). We
    // therefore only require they share PackedSize and copy element-wise (see
    // the proxy->real transfer below). For all other A/B types the proxy is
    // identical to the real type.
    static_assert(ck_tile::numeric_traits<AInitType>::PackedSize ==
                      ck_tile::numeric_traits<ADataType>::PackedSize,
                  "init proxy must share PackedSize with ADataType");
    static_assert(ck_tile::numeric_traits<BInitType>::PackedSize ==
                      ck_tile::numeric_traits<BDataType>::PackedSize,
                  "init proxy must share PackedSize with BDataType");

    using ScaleType = ck_tile::e8m0_t;

    constexpr int ScaleGranularityM = 1;
    constexpr int ScaleGranularityN = 1;
    constexpr int ScaleGranularityK = 32;

    const bool verbose = arg_parser.get_int("verbose") == 1;

    ck_tile::index_t M = arg_parser.get_int("m");
    ck_tile::index_t N = arg_parser.get_int("n");
    ck_tile::index_t K = arg_parser.get_int("k");

    ck_tile::index_t stride_A = arg_parser.get_int("stride_a");
    ck_tile::index_t stride_B = arg_parser.get_int("stride_b");
    ck_tile::index_t stride_C = arg_parser.get_int("stride_c");

    ck_tile::index_t init_method = arg_parser.get_int("init");
    ck_tile::index_t n_warmup    = arg_parser.get_int("warmup");
    ck_tile::index_t n_repeat    = arg_parser.get_int("repeat");

    stride_A = ck_tile::get_default_stride(M, K, stride_A, is_row_major(a_layout));
    stride_B = ck_tile::get_default_stride(K, N, stride_B, is_row_major(b_layout));
    stride_C = ck_tile::get_default_stride(M, N, stride_C, is_row_major(c_layout));

    auto scale_stride_A = ck_tile::get_default_stride(
        M / ScaleGranularityM, K / ScaleGranularityK, 0, is_row_major(a_layout));
    auto scale_stride_B = ck_tile::get_default_stride(
        K / ScaleGranularityK, N / ScaleGranularityN, 0, is_row_major(b_layout));

    if(K % ScaleGranularityK != 0)
        throw std::runtime_error("wrong! K must be multiple of ScaleGranularityK.");
    if(K % ck_tile::numeric_traits<ADataType>::PackedSize != 0 ||
       K % ck_tile::numeric_traits<BDataType>::PackedSize != 0)
        throw std::runtime_error("wrong! K must be multiple of packed size.");

    ck_tile::HostTensor<ADataType> a_host(
        ck_tile::host_tensor_descriptor(M, K, stride_A, is_row_major(a_layout)));
    ck_tile::HostTensor<BDataType> b_origin_host(
        ck_tile::host_tensor_descriptor(K, N, stride_B, is_row_major(b_layout)));
    ck_tile::HostTensor<CDataType> c_rslt_host(
        ck_tile::host_tensor_descriptor(M, N, stride_C, is_row_major(CLayout{})));

    // Proxy tensors used for initialization. When the init proxy collapses to
    // the real type (i.e. ADataType / BDataType is not pk_fp6x16_t) these are
    // direct stand-ins; otherwise they are byte-equivalent pk_fp6_t buffers
    // that are memcpy'd into a_host / b_origin_host once init is done.
    ck_tile::HostTensor<AInitType> a_init_host(
        ck_tile::host_tensor_descriptor(M, K, stride_A, is_row_major(a_layout)));
    ck_tile::HostTensor<BInitType> b_init_host(
        ck_tile::host_tensor_descriptor(K, N, stride_B, is_row_major(b_layout)));

    ck_tile::HostTensor<ScaleType> scale_a(ck_tile::host_tensor_descriptor(
        M / ScaleGranularityM, K / ScaleGranularityK, scale_stride_A, is_row_major(a_layout)));
    ck_tile::HostTensor<ScaleType> scale_b(ck_tile::host_tensor_descriptor(
        K / ScaleGranularityK, N / ScaleGranularityN, scale_stride_B, is_row_major(b_layout)));

    if(init_method == 0)
    {
        // Random tensor and scale values
        ck_tile::FillUniformDistribution<>{0.0f, 1.0f}(a_init_host);
        ck_tile::FillUniformDistribution<>{-.5f, .5f}(b_init_host);
        ck_tile::FillUniformDistribution<>{-2.f, 2.f}(scale_a);
        ck_tile::FillUniformDistribution<>{-2.f, 2.f}(scale_b);
    }
    else if(init_method == 1)
    {
        // Constant tensor and scale values
        ck_tile::FillUniformDistribution<>{2.f, 2.f}(a_init_host);
        ck_tile::FillUniformDistribution<>{0.5f, 0.5f}(b_init_host);
        ck_tile::FillUniformDistribution<>{0.5f, 0.5f}(scale_a);
        ck_tile::FillUniformDistribution<>{2.f, 2.f}(scale_b);
    }
    else
    {
        throw std::runtime_error("wrong! Unexpected init_method");
    }

    // Transfer the proxy-initialized data into the real host tensors. When the
    // proxy collapsed to the real type (A/BInitType == A/BDataType) this is a
    // same-type move. Otherwise the proxy is pk_fp6_t (16 bytes incl. vector
    // padding) and the real type is pk_fp6x16_t (12 bytes); both store the same
    // packed layout (3 x 32-bit words holding 16 x fp6 values) so we copy the
    // first sizeof(real-type) bytes per element, dropping the trailing pad.
    auto copy_proxy_to_real = [](auto& dst, const auto& src) {
        using DstT = typename std::remove_reference_t<decltype(dst)>::Data::value_type;
        using SrcT = typename std::remove_reference_t<decltype(src)>::Data::value_type;

        const std::size_t n_elements = src.get_element_space_size();
        auto* dst_bytes              = reinterpret_cast<char*>(dst.data());
        const auto* src_bytes        = reinterpret_cast<const char*>(src.data());

        for(std::size_t i = 0; i < n_elements; ++i)
        {
            std::memcpy(dst_bytes + i * sizeof(DstT), src_bytes + i * sizeof(SrcT), sizeof(DstT));
        }
    };

    if constexpr(std::is_same_v<AInitType, ADataType>)
    {
        a_host = std::move(a_init_host);
    }
    else
    {
        static_assert(std::is_same_v<ADataType, ck_tile::pk_fp6x16_t>,
                      "init proxy mismatch is only expected for pk_fp6x16_t");
        copy_proxy_to_real(a_host, a_init_host);
    }
    if constexpr(std::is_same_v<BInitType, BDataType>)
    {
        b_origin_host = std::move(b_init_host);
    }
    else
    {
        static_assert(std::is_same_v<BDataType, ck_tile::pk_fp6x16_t>,
                      "init proxy mismatch is only expected for pk_fp6x16_t");
        copy_proxy_to_real(b_origin_host, b_init_host);
    }

    const auto b_shuffled_host  = preShuffleWeight<MXFlatmmArchTraits::GetNLane()>(b_origin_host);
    const auto scale_a_shuffled = MXFlatmmArchTraits::template preShuffleScale<true>(scale_a);
    const auto scale_b_shuffled = MXFlatmmArchTraits::template preShuffleScale<false>(scale_b);

    ck_tile::DeviceMem a_dev_buf(a_host.get_element_space_size_in_bytes());
    ck_tile::DeviceMem b_shuffled_dev_buf(b_shuffled_host.get_element_space_size_in_bytes());
    ck_tile::DeviceMem c_dev_buf(c_rslt_host.get_element_space_size_in_bytes());

    ck_tile::DeviceMem scale_a_dev_buf(scale_a_shuffled.get_element_space_size_in_bytes());
    ck_tile::DeviceMem scale_b_dev_buf(scale_b_shuffled.get_element_space_size_in_bytes());

    a_dev_buf.ToDevice(a_host.data());
    b_shuffled_dev_buf.ToDevice(b_shuffled_host.data());
    c_rslt_host.SetZero();
    scale_a_dev_buf.ToDevice(scale_a_shuffled.data());
    scale_b_dev_buf.ToDevice(scale_b_shuffled.data());

    auto scale_a_dev_ptr =
        ck_tile::FlatmmScalePointer<ScaleGranularityM, ScaleGranularityK, ScaleType>{
            static_cast<ScaleType*>(scale_a_dev_buf.GetDeviceBuffer()), M / ScaleGranularityM};
    auto scale_b_dev_ptr =
        ck_tile::FlatmmScalePointer<ScaleGranularityN, ScaleGranularityK, ScaleType>{
            static_cast<ScaleType*>(scale_b_dev_buf.GetDeviceBuffer()), N / ScaleGranularityN};

    if(verbose)
    {
        std::cout << "Computing GEMM on device..." << std::endl << std::endl;
    }

    invoke_mx_flatmm<MXFlatmmArchTraits,
                     ADataType,
                     BDataType,
                     ck_tile::tuple<>,
                     AccDataType,
                     CDataType,
                     ALayout,
                     BLayout,
                     ck_tile::tuple<>,
                     CLayout,
                     decltype(scale_a_dev_ptr),
                     decltype(scale_b_dev_ptr),
                     UsePersistentKernel>(a_dev_buf,
                                          b_shuffled_dev_buf,
                                          c_dev_buf,
                                          M,
                                          N,
                                          K,
                                          stride_A,
                                          stride_B,
                                          stride_C,
                                          scale_a_dev_ptr,
                                          scale_b_dev_ptr,
                                          n_warmup,
                                          n_repeat);

    c_dev_buf.FromDevice(c_rslt_host.data());
    if(verbose)
    {
        std::cout << "Computing GEMM on device... Done.\n" << std::endl;
    }

    bool pass = true; // validation of the results
    if(arg_parser.get_int("v") == 1)
    {
        ck_tile::HostTensor<CDataType> c_m_n_host_ref(
            ck_tile::host_tensor_descriptor(M, N, stride_C, is_row_major(CLayout{})));
        c_m_n_host_ref.SetZero();

        if(verbose)
        {
            std::cout << "Computing GEMM on host..." << std::endl;
        }
        ck_tile::
            reference_mx_gemm<ADataType, BDataType, ScaleType, ScaleType, AccDataType, CDataType>(
                a_host, b_origin_host, c_m_n_host_ref, scale_a, scale_b);

        if(verbose)
        {
            std::cout << "Computing GEMM on host... Done.\n" << std::endl;
            std::cout << "Comparing results..." << std::endl;
        }

        const float rtol = std::is_same_v<ADataType, ck_tile::half_t> ? 1e-3 : 1e-2;
        const float atol = std::is_same_v<ADataType, ck_tile::half_t> ? 1e-3 : 1e-2;

        pass = ck_tile::check_err(
            c_rslt_host, c_m_n_host_ref, "\n\nError: Incorrect results!", rtol, atol);

        std::cout << "\n\nRelative error threshold: " << rtol
                  << " Absolute error threshold: " << atol << std::endl;
        std::cout << "The GPU veification result is: " << (pass ? "correct" : "fail") << std::endl;
    }

    return pass ? 0 : -1;
}
