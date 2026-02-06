// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// Pack 4 consecutive e8m0_t scales in K dimension into int32 for efficient 32-bit loads
// For Scale A: [M, K/32] → [M, K/32/4] with int32 elements
// For Scale B: [K/32, N] → [K/32/4, N] with int32 elements
template <typename ScaleTensor>
auto pack_scales_for_k_dimension(const ScaleTensor& scale_unpacked, 
                                  ck_tile::index_t pack_size = 4)
{
    using ScaleType = typename ScaleTensor::Data::value_type;
    static_assert(sizeof(ScaleType) == 1, "Scale type must be 1 byte (e8m0_t)");
    
    const auto& desc = scale_unpacked.mDesc;
    ck_tile::index_t dim0 = desc.get_lengths()[0];
    ck_tile::index_t dim1 = desc.get_lengths()[1];
    ck_tile::index_t stride1 = desc.get_strides()[1];
    
    // Determine which dimension is K (the one to pack)
    // If stride1 == 1, then dim1 is contiguous (K dimension for row-major scale A)
    // If stride0 == 1, then dim0 is contiguous (K dimension for col-major scale B)
    bool pack_dim1 = (stride1 == 1);
    
    ck_tile::index_t packed_k_dim = pack_dim1 ? (dim1 / pack_size) : (dim0 / pack_size);
    ck_tile::index_t new_dim0 = pack_dim1 ? dim0 : packed_k_dim;
    ck_tile::index_t new_dim1 = pack_dim1 ? packed_k_dim : dim1;
    // Calculate new strides based on which dimension was packed
    ck_tile::index_t new_stride0, new_stride1;
    if (pack_dim1) {
        // Packed dim1 (K dimension for row-major): new shape [dim0, packed_k_dim]
        // If original was row-major [dim0, dim1] with stride [dim1, 1]
        // New should be row-major [dim0, packed_k_dim] with stride [packed_k_dim, 1]
        new_stride0 = packed_k_dim;
        new_stride1 = 1;
    } else {
        // Packed dim0 (K dimension for col-major): new shape [packed_k_dim, dim1]
        // If original was col-major [dim0, dim1] with stride [1, dim0]
        // New should be col-major [packed_k_dim, dim1] with stride [1, packed_k_dim]
        new_stride0 = 1;
        new_stride1 = packed_k_dim;
    }
    
    ck_tile::HostTensor<int32_t> scale_packed(
        ck_tile::HostTensorDescriptor({new_dim0, new_dim1}, {new_stride0, new_stride1}));
    
    // Pack scales: strided packing for K_lane distribution with OpSel
    // Each int32_t packs 4 strided scales (one per kIter at same K_lane position)
    // For K=512: 16 unpacked scales [0-15] -> 4 packed int32s
    //   int32[0] = {scale[0], scale[4], scale[8], scale[12]}  <- K_lane=0, OpSel selects kIter
    //   int32[1] = {scale[1], scale[5], scale[9], scale[13]}  <- K_lane=1, OpSel selects kIter
    //   int32[2] = {scale[2], scale[6], scale[10], scale[14]}  <- K_lane=2, OpSel selects kIter
    //   int32[3] = {scale[3], scale[7], scale[11], scale[15]}  <- K_lane=3, OpSel selects kIter
    // OpSel(kIter) selects byte within thread's int32 for current kIter
    for(ck_tile::index_t i = 0; i < new_dim0; ++i)
    {
        for(ck_tile::index_t j = 0; j < new_dim1; ++j)
        {
            int32_t packed_value = 0;
            for(ck_tile::index_t k = 0; k < pack_size; ++k)
            {
                // Strided packing: byte k corresponds to kIter=k
                // The stride is always pack_size (4), not packed_k_dim!
                // For K=512: 16 unpacked elements [0-15] -> 4 packed int32s
                //   int32[0] = {unpacked[0], unpacked[4], unpacked[8], unpacked[12]}  (stride=4)
                //   int32[1] = {unpacked[1], unpacked[5], unpacked[9], unpacked[13]}  (stride=4)
                // For K=1024: 32 unpacked elements [0-31] -> 8 packed int32s
                //   int32[0] = {unpacked[0], unpacked[4], unpacked[8], unpacked[12]}  (stride=4)
                //   int32[4] = {unpacked[16], unpacked[20], unpacked[24], unpacked[28]}  (stride=4)
                // For row-major (pack_dim1=true): packed index j, byte k -> unpacked[j + k*4]
                // For col-major (pack_dim1=false): packed index i, byte k -> unpacked[i*4 + k*4] = unpacked[(i + k)*4]
                // But we want: packed index i, byte k -> unpacked[i*4 + k] (base i*4, then stride 4)
                // Actually: int32[i] should pack {unpacked[i*4 + 0*4], unpacked[i*4 + 1*4], unpacked[i*4 + 2*4], unpacked[i*4 + 3*4]}
                //          = {unpacked[i*4], unpacked[i*4 + 4], unpacked[i*4 + 8], unpacked[i*4 + 12]}
                ck_tile::index_t src_i = pack_dim1 ? i : (i * pack_size + k * pack_size);
                ck_tile::index_t src_j = pack_dim1 ? (j * pack_size + k * pack_size) : j;
                
                uint8_t scale_byte = *reinterpret_cast<const uint8_t*>(&scale_unpacked(src_i, src_j));
                packed_value |= (static_cast<int32_t>(scale_byte) << (k * 8));
            }
            scale_packed(i, j) = packed_value;
        }
    }
    
    return scale_packed;
}

template <typename ADataType,
          typename BDataType,
          typename AccDataType,
          typename GemmConfig,
          bool UsePersistentKernel,
          typename ALayout,
          typename BLayout,
          typename CLayout>
int run_mx_gemm_with_layouts(int argc,
                             char* argv[],
                             ALayout,
                             BLayout,
                             CLayout)
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

    // Use get_default_stride helper for automatic leading dimension calculation (only if not explicitly provided)
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
    // A scales: [M, K/32] with A's layout → coalescing follows A's pattern
    // B scales: [K/32, N] with B's layout → coalescing follows B's pattern
    using ScaleType = ck_tile::e8m0_t;
    ck_tile::index_t scale_k_size = K / 32;
    
    // Follow A/BLayout to get the layouts for the scale tensors
    ck_tile::index_t stride_scale_a = ck_tile::get_default_stride(M, scale_k_size, 0, is_row_major(ALayout{}));
    ck_tile::index_t stride_scale_b = ck_tile::get_default_stride(scale_k_size, N, 0, is_row_major(BLayout{}));
    
    ck_tile::HostTensor<ScaleType> scale_a_host(
        ck_tile::host_tensor_descriptor(M, scale_k_size, stride_scale_a, is_row_major(ALayout{})));
    ck_tile::HostTensor<ScaleType> scale_b_host(
        ck_tile::host_tensor_descriptor(scale_k_size, N, stride_scale_b, is_row_major(BLayout{})));
    int seed = 1234;
    switch(init_method)
    {
    case 0:
        ck_tile::FillUniformDistribution<ADataType>{-1.f, 1.f, seed++}(a_host);
        ck_tile::FillUniformDistribution<BDataType>{-1.f, 1.f, seed++}(b_host);
        ck_tile::FillUniformDistribution<ScaleType>{0.001f, 10.f, seed++}(scale_a_host);
        ck_tile::FillUniformDistribution<ScaleType>{0.001f, 10.f, seed++}(scale_b_host);
        break;
    case 1:
        ck_tile::FillConstant<ADataType>{ADataType(1.f)}(a_host);
        ck_tile::FillConstant<BDataType>{BDataType(1.f)}(b_host);
        ck_tile::FillConstant<ScaleType>{ScaleType(1.f)}(scale_a_host);
        ck_tile::FillConstant<ScaleType>{ScaleType(1.f)}(scale_b_host);
        break;
    case 2:
        ck_tile::FillUniformDistribution<ADataType>{-1.f, 1.f, seed++}(a_host);
        ck_tile::FillUniformDistribution<BDataType>{-1.f, 1.f, seed++}(b_host);
        ck_tile::FillConstant<ScaleType>{ScaleType(1.f)}(scale_a_host);
        ck_tile::FillConstant<ScaleType>{ScaleType(1.f)}(scale_b_host);
        break;
    case 3:
        // Debug mode: simple power-of-2 pattern for scales (e8m0 format)
        ck_tile::FillConstant<ADataType>{ADataType(1.f)}(a_host);
        ck_tile::FillConstant<BDataType>{BDataType(1.f)}(b_host);
        // Fill scales with power-of-2 pattern: 1.0, 2.0, 4.0, 8.0, 16.0, ...
        // e8m0 is exponent-only, so these give clear distinct values
        // for(std::size_t i = 0; i < scale_a_host.mDesc.get_element_space_size(); ++i)
        // {
        //     float val = std::pow(2.0f, static_cast<float>(i % 16)); // cycle through 2^0 to 2^15
        //     scale_a_host.mData[i] = ScaleType(val);
        // }
        // for(std::size_t i = 0; i < scale_b_host.mDesc.get_element_space_size(); ++i)
        // {
        //     float val = std::pow(2.0f, static_cast<float>(i % 16)); // cycle through 2^0 to 2^15
        //     scale_b_host.mData[i] = ScaleType(val);
        // }
        ck_tile::FillConstant<ScaleType>{ScaleType(1.f)}(scale_a_host);
        ck_tile::FillConstant<ScaleType>{ScaleType(1.f)}(scale_b_host);
        
        // Test data to verify K block loading for K=1024 (2 K blocks)
        // K block 0: K indices 0-511, scale K indices 0-15, packed into K_packed indices 0-3
        // K block 1: K indices 512-1023, scale K indices 16-31, packed into K_packed indices 4-7
        
        // Scale A: [M, K/32] row-major (unpacked K indices in second dim)
        // Strided packing: int32[j] packs unpacked[j], unpacked[j+4], unpacked[j+8], unpacked[j+12]
        // K block 0: K indices 0-511 → unpacked K indices 0-15 → packed int32s 0-3
        //   int32[0] = {unpacked[0], unpacked[4], unpacked[8], unpacked[12]}  (K_Lane=0)
        //   int32[1] = {unpacked[1], unpacked[5], unpacked[9], unpacked[13]}  (K_Lane=1)
        scale_a_host(0, 0) = ScaleType(2.f);    // K block 0, int32[0] byte 0 (unpacked[0])
        scale_a_host(0, 4) = ScaleType(4.f);    // K block 0, int32[0] byte 1 (unpacked[4])
        scale_a_host(0, 8) = ScaleType(8.f);    // K block 0, int32[0] byte 2 (unpacked[8])
        scale_a_host(0, 12) = ScaleType(16.f);   // K block 0, int32[0] byte 3 (unpacked[12])
        scale_a_host(1, 0) = ScaleType(32.f);   // K block 0, int32[1] byte 0 (unpacked[1])
        
        // K block 1: K indices 512-1023 → unpacked K indices 16-31 → packed int32s 4-7
        //   int32[4] = {unpacked[16], unpacked[20], unpacked[24], unpacked[28]}  (K_Lane=0)
        //   int32[5] = {unpacked[17], unpacked[21], unpacked[25], unpacked[29]}  (K_Lane=1)
        scale_a_host(0, 16) = ScaleType(256.f);   // K block 1, int32[4] byte 0 (unpacked[16])
        scale_a_host(0, 20) = ScaleType(512.f);   // K block 1, int32[4] byte 1 (unpacked[20])
        scale_a_host(0, 24) = ScaleType(1024.f); // K block 1, int32[4] byte 2 (unpacked[24])
        scale_a_host(0, 28) = ScaleType(2048.f); // K block 1, int32[4] byte 3 (unpacked[28])
        scale_a_host(1, 16) = ScaleType(4096.f); // K block 1, int32[5] byte 0 (unpacked[17])
        
        // mIter=1: M rows 16-31 (second XDL block)
        scale_a_host(16, 0) = ScaleType(64.f);  // K block 0, unpacked K=0, M=16
        scale_a_host(16, 16) = ScaleType(8192.f); // K block 1, unpacked K=16, M=16
        
        // Scale B: [K/32, N] col-major (unpacked K indices in first dim, N in second dim)
        // Strided packing: int32[i] packs unpacked[i], unpacked[i+8], unpacked[i+16], unpacked[i+24]
        // K block 0: K indices 0-511 → unpacked K indices 0-15 → packed int32s 0-3
        //   int32[0] = {unpacked[0], unpacked[4], unpacked[8], unpacked[12]}  (K_Lane=0)
        //   int32[1] = {unpacked[1], unpacked[5], unpacked[9], unpacked[13]}  (K_Lane=1)
        scale_b_host(0, 0) = ScaleType(2.f);    // K block 0, int32[0] byte 0 (unpacked[0])
        scale_b_host(4, 0) = ScaleType(4.f);    // K block 0, int32[0] byte 1 (unpacked[4])
        scale_b_host(8, 0) = ScaleType(8.f);    // K block 0, int32[0] byte 2 (unpacked[8])
        scale_b_host(12, 0) = ScaleType(16.f);   // K block 0, int32[0] byte 3 (unpacked[12])
        scale_b_host(1, 0) = ScaleType(32.f);   // K block 0, int32[1] byte 0 (unpacked[1])
        
        // K block 1: K indices 512-1023 → unpacked K indices 16-31 → packed int32s 4-7
        //   int32[4] = {unpacked[16], unpacked[20], unpacked[24], unpacked[28]}  (K_Lane=0)
        //   int32[5] = {unpacked[17], unpacked[21], unpacked[25], unpacked[29]}  (K_Lane=1)
        scale_b_host(16, 0) = ScaleType(256.f);   // K block 1, int32[4] byte 0 (unpacked[16])
        scale_b_host(20, 0) = ScaleType(512.f);   // K block 1, int32[4] byte 1 (unpacked[20])
        scale_b_host(24, 0) = ScaleType(1024.f); // K block 1, int32[4] byte 2 (unpacked[24])
        scale_b_host(28, 0) = ScaleType(2048.f); // K block 1, int32[4] byte 3 (unpacked[28])
        scale_b_host(17, 0) = ScaleType(4096.f); // K block 1, int32[5] byte 0 (unpacked[17])
        
        // nIter=1: N rows 16-31 (second XDL block)
        scale_b_host(0, 16) = ScaleType(64.f);  // K block 0, unpacked K=0, N=16
        scale_b_host(16, 16) = ScaleType(8192.f); // K block 1, unpacked K=16, N=16
        break;
    }

    // Pack scales: 4 consecutive e8m0_t in K dimension → 1 int32 for efficient 32-bit loads
    // This enables the GPU to load 4 scales (for 4 K-blocks) with a single 32-bit load
    // Scale A: [M, K/32] → [M, K/128] with int32 elements (since K/32/4 = K/128)
    // Scale B: [K/32, N] → [K/128, N] with int32 elements
    auto scale_a_packed = pack_scales_for_k_dimension(scale_a_host, 4);
    auto scale_b_packed = pack_scales_for_k_dimension(scale_b_host, 4);

    // DEBUG: Print first few packed scale values
    if (true ||init_method == 3)
    {
        std::cout << "Host: ScaleA packed [0,0]: ";
        uint8_t* a_bytes = reinterpret_cast<uint8_t*>(&scale_a_packed(0, 0));
        std::cout << "[" << static_cast<int>(a_bytes[0]) << "," << static_cast<int>(a_bytes[1]) << "," 
                  << static_cast<int>(a_bytes[2]) << "," << static_cast<int>(a_bytes[3]) << "]\n";
        std::cout << "Host: ScaleA packed [0,4]: ";
        uint8_t* a_bytes4 = reinterpret_cast<uint8_t*>(&scale_a_packed(0, 4));
        std::cout << "[" << static_cast<int>(a_bytes4[0]) << "," << static_cast<int>(a_bytes4[1]) << "," 
                  << static_cast<int>(a_bytes4[2]) << "," << static_cast<int>(a_bytes4[3]) << "]\n";
        std::cout << "Host: ScaleB packed [0,0]: ";
        uint8_t* b_bytes = reinterpret_cast<uint8_t*>(&scale_b_packed(0, 0));
        std::cout << "[" << static_cast<int>(b_bytes[0]) << "," << static_cast<int>(b_bytes[1]) << "," 
                  << static_cast<int>(b_bytes[2]) << "," << static_cast<int>(b_bytes[3]) << "]\n";
        std::cout << "Host: ScaleB packed [4,0]: ";
        uint8_t* b_bytes4 = reinterpret_cast<uint8_t*>(&scale_b_packed(4, 0));
        std::cout << "[" << static_cast<int>(b_bytes4[0]) << "," << static_cast<int>(b_bytes4[1]) << "," 
                  << static_cast<int>(b_bytes4[2]) << "," << static_cast<int>(b_bytes4[3]) << "]\n";
        
        // Print unpacked first row/col for reference
        std::cout << "Host: ScaleA unpacked thread 0, every 4th element: [";
        for (int k = 0; k < std::min(32, static_cast<int>(scale_a_host.mDesc.get_lengths()[1])); k += 4)
            std::cout << static_cast<int>(*reinterpret_cast<uint8_t*>(&scale_a_host(0, k))) << ",";
        std::cout << "]\n";
        std::cout << "Host: ScaleB unpacked thread 0, every 4th element: [";
        for (int k = 0; k < std::min(32, static_cast<int>(scale_b_host.mDesc.get_lengths()[0])); k += 4)
            std::cout << static_cast<int>(*reinterpret_cast<uint8_t*>(&scale_b_host(k, 0))) << ",";
        std::cout << "]\n";
        // Threads 0-15: M rows 0-15, K_Lane cycles through 0,1,2,3
        // Thread 16: M row 0 again, but next K_Lane group (K_Lane=1 if cycling, or next K group)
        // Actually, thread 16 goes back to row 0 with a different K index
        std::cout << "Host: ScaleA unpacked thread 16 (row 0, next K group), every 4th element: [";
        for (int k = 1; k < std::min(32, static_cast<int>(scale_a_host.mDesc.get_lengths()[1])); k += 4)
            std::cout << static_cast<int>(*reinterpret_cast<uint8_t*>(&scale_a_host(0, k))) << ",";
        std::cout << "]\n";
        std::cout << "Host: ScaleB unpacked thread 16 (row 0, next K group), every 4th element: [";
        for (int k = 1; k < std::min(32, static_cast<int>(scale_b_host.mDesc.get_lengths()[0])); k += 4)
            std::cout << static_cast<int>(*reinterpret_cast<uint8_t*>(&scale_b_host(k, 0))) << ",";
        std::cout << "]\n";
    }

    ck_tile::DeviceMem a_dev_buf(a_host.get_element_space_size_in_bytes());
    ck_tile::DeviceMem b_dev_buf(b_host.get_element_space_size_in_bytes());
    ck_tile::DeviceMem c_dev_buf(c_host.get_element_space_size_in_bytes());
    ck_tile::DeviceMem scale_a_dev_buf(scale_a_packed.get_element_space_size_in_bytes());
    ck_tile::DeviceMem scale_b_dev_buf(scale_b_packed.get_element_space_size_in_bytes());

    a_dev_buf.ToDevice(a_host.data());
    b_dev_buf.ToDevice(b_host.data());
    scale_a_dev_buf.ToDevice(scale_a_packed.data());
    scale_b_dev_buf.ToDevice(scale_b_packed.data());

    // Scale pointers
    using ScaleM = ck_tile::MXScalePointer<1, 32>; // in blocks of 32 in K
    using ScaleN = ck_tile::MXScalePointer<1, 32>;

    ScaleM scale_m(reinterpret_cast<float*>(scale_a_dev_buf.GetDeviceBuffer()));
    ScaleN scale_n(reinterpret_cast<float*>(scale_b_dev_buf.GetDeviceBuffer()));

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
                                    UsePersistentKernel>(
        a_dev_buf, b_dev_buf, c_dev_buf, M, N, K, stride_A, stride_B, stride_C, kbatch, scale_m, scale_n, n_warmup, n_repeat);
    
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

        ck_tile::reference_mx_gemm<ADataType, BDataType, ScaleType, AccDataType, CDataType>(
            a_host, b_host, c_m_n_host_ref, scale_a_host, scale_b_host);
        // ck_tile::reference_gemm<ADataType, BDataType, AccDataType, CDataType>(
        //     a_host, b_host, c_m_n_host_ref);

        auto calculate_rtol_atol = [&K, &kbatch](const float max_accumulated_value)
        {
            // using ComputeType =
            //     std::conditional_t<sizeof(ADataType) < sizeof(BDataType), ADataType, BDataType>;
            // // Calculate thresholds
            // const auto rtol = ck_tile::get_relative_threshold<ComputeType, CDataType, AccDataType>(
            //     ck_tile::integer_divide_ceil(K, kbatch));
            // const auto atol = ck_tile::get_absolute_threshold<ComputeType, CDataType, AccDataType>(
            //     max_accumulated_value / kbatch, ck_tile::integer_divide_ceil(K, kbatch));
            // // Calculate error due to split_k accumulation
            // const auto rtol_split_k =
            //     ck_tile::get_relative_threshold<CDataType, CDataType, CDataType>(kbatch);
            // const auto atol_split_k = ck_tile::get_absolute_threshold<CDataType, CDataType, CDataType>(
            //     max_accumulated_value, kbatch);
            // // Use higher threshold
            // return ck_tile::make_tuple(std::max(rtol, rtol_split_k), std::max(atol, atol_split_k));
            ck_tile::ignore = K; ck_tile::ignore = kbatch; ck_tile::ignore = max_accumulated_value;
            return ck_tile::make_tuple(0.1, 1.0);
        };

        const float max_accumulated_value =
            *std::max_element(c_m_n_host_ref.mData.begin(), c_m_n_host_ref.mData.end());
        const auto [rtol, atol] = calculate_rtol_atol(max_accumulated_value);

        pass = ck_tile::check_err(
            c_host, c_m_n_host_ref, "Error: Incorrect results!", rtol, atol);

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
                                                false>(argc, argv, Row{}, Col{}, Row{});
        }
        else if(mx_prec == "fp8" || mx_prec == "fp8xfp8")
        {
            return run_mx_gemm_with_layouts<ck_tile::fp8_t,
                                                ck_tile::fp8_t,
                                                float,
                                                MXfp8_GemmConfig16,
                                                false>(argc, argv, Row{}, Col{}, Row{});
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
