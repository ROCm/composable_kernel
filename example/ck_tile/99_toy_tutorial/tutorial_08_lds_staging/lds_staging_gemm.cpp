// SPDX-License-Identifier: MIT
// Copyright (c) 2024, Advanced Micro Devices, Inc. All rights reserved.

/*
 * Tutorial 08: LDS Staging for GEMM
 *
 * Demonstrates the standard GPU GEMM data flow:
 *   Global Memory -> LDS (shared memory) -> Registers -> MFMA Compute
 *
 * KEY INSIGHT: Following 02_gemm pattern
 *   - A is stored as [M x K] in memory
 *   - B is stored as [N x K] in memory (transposed B!)
 *   - GEMM computes: C = A * B^T
 *
 * TWO different distributions:
 *   1. COPY distribution: All threads load cooperatively (no replication)
 *   2. GEMM distribution: Warps read with replication (LDS data sharing)
 *
 * Why LDS enables reuse:
 *   - A tile [M x K]: Loaded ONCE, read by NWarp warps (replication)
 *   - B tile [N x K]: Loaded ONCE, read by MWarp warps (replication)
 *   - Global memory bandwidth reduced by replication factor!
 */

#include <iostream>
#include <vector>
#include <iomanip>
#include <chrono>
#include <limits>
#include "ck_tile/core.hpp"
#include "ck_tile/host.hpp"
#include "ck_tile/ops/common.hpp"
#include "ck_tile/ops/gemm/warp/warp_gemm.hpp"

using namespace ck_tile;

// LDS Staging HGEMM kernel - following 02_gemm pattern
template<typename ADataType, typename BDataType, typename CDataType, typename AccDataType>
struct LdsStagingHgemmKernel
{
    static constexpr index_t kWaveSize = 64;
    static constexpr index_t kWarpM = 16;
    static constexpr index_t kWarpN = 16;
    static constexpr index_t kWarpK = 16;

    // 2x2 warp configuration
    static constexpr index_t MWarp = 2;
    static constexpr index_t NWarp = 2;
    static constexpr index_t kBlockSize = MWarp * NWarp * kWaveSize;  // 256

    // Y-dimension repetition
    static constexpr index_t MIterPerWarp = 2;
    static constexpr index_t NIterPerWarp = 2;

    // K-tile for LDS staging
    static constexpr index_t kKPerBlock = 32;
    static constexpr index_t KIterPerWarp = kKPerBlock / kWarpK;  // 2

    // Block tile dimensions
    static constexpr index_t kMPerBlock = MWarp * MIterPerWarp * kWarpM;  // 64
    static constexpr index_t kNPerBlock = NWarp * NIterPerWarp * kWarpN;  // 64

    using WarpGemm = WarpGemmMfmaF16F16F32M16N16K16;

    // LDS size
    CK_TILE_HOST_DEVICE static constexpr index_t GetStaticLdsSize()
    {
        // A: [M x K], B: [N x K] (transposed layout)
        constexpr index_t a_lds_size = kMPerBlock * kKPerBlock * sizeof(ADataType);
        constexpr index_t b_lds_size = kNPerBlock * kKPerBlock * sizeof(BDataType);
        constexpr index_t a_lds_size_aligned = ((a_lds_size + 15) / 16) * 16;
        return a_lds_size_aligned + b_lds_size;
    }

    // ========================================================================
    // COPY Distributions - For coalesced global memory access
    // All 256 threads load cooperatively, NO replication
    // Following 02_gemm/block_gemm_pipeline_agmem_bgmem_creg_default_policy.hpp
    // ========================================================================

    CK_TILE_HOST_DEVICE static constexpr auto MakeACopyDistribution()
    {
        constexpr index_t K1 = 16 / sizeof(ADataType);  // 8 for half_t
        constexpr index_t K0 = kKPerBlock / K1;         // 32/8 = 4
        constexpr index_t M2 = kWaveSize / K0;          // 64/4 = 16
        constexpr index_t M1 = kBlockSize / kWaveSize;  // 256/64 = 4
        constexpr index_t M0 = kMPerBlock / (M2 * M1);  // 64/(16*4) = 1

        return make_static_tile_distribution(
            tile_distribution_encoding<
                sequence<1>,  // No replication for copy
                tuple<sequence<M0, M1, M2>, sequence<K0, K1>>,
                tuple<sequence<1>, sequence<1, 2>>,
                tuple<sequence<1>, sequence<2, 0>>,
                sequence<1, 2>,
                sequence<0, 1>>{});
    }

    CK_TILE_HOST_DEVICE static constexpr auto MakeBCopyDistribution()
    {
        constexpr index_t K1 = 16 / sizeof(BDataType);  // 8 for half_t
        constexpr index_t K0 = kKPerBlock / K1;         // 32/8 = 4
        constexpr index_t N2 = kWaveSize / K0;          // 64/4 = 16
        constexpr index_t N1 = kBlockSize / kWaveSize;  // 256/64 = 4
        constexpr index_t N0 = kNPerBlock / (N2 * N1);  // 64/(16*4) = 1

        return make_static_tile_distribution(
            tile_distribution_encoding<
                sequence<1>,  // No replication for copy
                tuple<sequence<N0, N1, N2>, sequence<K0, K1>>,
                tuple<sequence<1>, sequence<1, 2>>,
                tuple<sequence<1>, sequence<2, 0>>,
                sequence<1, 2>,
                sequence<0, 1>>{});
    }

    CK_TILE_DEVICE void operator()(const ADataType* a,
                                   const BDataType* b,  // B is stored as [N x K]!
                                   const CDataType* c,
                                   CDataType* d,
                                   index_t M,
                                   index_t N,
                                   index_t K,
                                   index_t lda,  // Leading dim for A [M x K]
                                   index_t ldb,  // Leading dim for B [N x K]
                                   index_t ldc,
                                   index_t ldd,
                                   AccDataType alpha,
                                   AccDataType beta) const
    {
        [[maybe_unused]] const index_t warp_id = get_warp_id();
        [[maybe_unused]] const index_t iMWarp = warp_id / NWarp;
        [[maybe_unused]] const index_t iNWarp = warp_id % NWarp;

        const index_t num_blocks_n = N / kNPerBlock;
        const index_t block_m = get_block_id() / num_blocks_n;
        const index_t block_n = get_block_id() % num_blocks_n;

        const index_t m_block_base = block_m * kMPerBlock;
        const index_t n_block_base = block_n * kNPerBlock;

        if(m_block_base >= M || n_block_base >= N)
            return;

        // ====================================================================
        // LDS Setup
        // ====================================================================

        __shared__ char p_smem_char[GetStaticLdsSize()];

        ADataType* p_a_lds = reinterpret_cast<ADataType*>(p_smem_char);
        constexpr index_t a_lds_size_aligned =
            ((kMPerBlock * kKPerBlock * sizeof(ADataType) + 15) / 16) * 16;
        BDataType* p_b_lds = reinterpret_cast<BDataType*>(p_smem_char + a_lds_size_aligned);

        // LDS descriptors: A [M x K], B [N x K]
        const auto a_lds_desc = make_naive_tensor_descriptor_packed(
            make_tuple(number<kMPerBlock>{}, number<kKPerBlock>{}));
        auto a_lds_view = make_tensor_view<address_space_enum::lds>(p_a_lds, a_lds_desc);

        const auto b_lds_desc = make_naive_tensor_descriptor_packed(
            make_tuple(number<kNPerBlock>{}, number<kKPerBlock>{}));
        auto b_lds_view = make_tensor_view<address_space_enum::lds>(p_b_lds, b_lds_desc);

        // ====================================================================
        // Global Memory Views - A [M x K], B [N x K]
        // ====================================================================

        // A: [M x K] with column-major stride
        const auto a_tensor = make_naive_tensor_view<address_space_enum::global>(
            a, make_tuple(M, K), make_tuple(1, lda), number<1>{}, number<1>{});

        // B: [N x K] - stored as transposed B! Row-major with stride ldb
        const auto b_tensor = make_naive_tensor_view<address_space_enum::global>(
            b, make_tuple(N, K), make_tuple(ldb, 1), number<8>{}, number<1>{});

        const auto c_tensor = make_naive_tensor_view<address_space_enum::global>(
            c, make_tuple(M, N), make_tuple(1, ldc), number<1>{}, number<1>{});

        auto d_tensor = make_naive_tensor_view<address_space_enum::global>(
            d, make_tuple(M, N), make_tuple(1, ldd), number<1>{}, number<1>{});

        // ====================================================================
        // COPY Distributions (no replication - cooperative load)
        // ====================================================================

        constexpr auto a_copy_distribution = MakeACopyDistribution();
        constexpr auto b_copy_distribution = MakeBCopyDistribution();

        // A copy windows: [M x K]
        auto a_copy_global_window = make_tile_window(
            a_tensor,
            make_tuple(number<kMPerBlock>{}, number<kKPerBlock>{}),
            {m_block_base, 0},
            a_copy_distribution);

        auto a_copy_lds_window = make_tile_window(
            a_lds_view,
            make_tuple(number<kMPerBlock>{}, number<kKPerBlock>{}),
            {0, 0},
            a_copy_distribution);

        // B copy windows: [N x K]
        auto b_copy_global_window = make_tile_window(
            b_tensor,
            make_tuple(number<kNPerBlock>{}, number<kKPerBlock>{}),
            {n_block_base, 0},
            b_copy_distribution);

        auto b_copy_lds_window = make_tile_window(
            b_lds_view,
            make_tuple(number<kNPerBlock>{}, number<kKPerBlock>{}),
            {0, 0},
            b_copy_distribution);

        // ====================================================================
        // GEMM Distributions (WITH replication - LDS data sharing!)
        // Following 02_gemm/block_gemm_asmem_bsmem_creg.hpp
        // ====================================================================

        constexpr auto a_warp_dstr_encode = tile_distribution_encoding<
            sequence<>,
            tuple<sequence<16>, sequence<4, 4>>,
            tuple<sequence<2, 1>>,
            tuple<sequence<0, 0>>,
            sequence<2>,
            sequence<1>>{};

        constexpr auto b_warp_dstr_encode = tile_distribution_encoding<
            sequence<>,
            tuple<sequence<4, 4>, sequence<16>>,
            tuple<sequence<1, 2>>,
            tuple<sequence<0, 0>>,
            sequence<1>,
            sequence<1>>{};

        constexpr auto c_warp_dstr_encode = tile_distribution_encoding<
            sequence<>,
            tuple<sequence<4, 4>, sequence<16>>,
            tuple<sequence<1, 2>>,
            tuple<sequence<0, 0>>,
            sequence<1>,
            sequence<1>>{};

        // A block distribution: replicated across NWarp (lines 57-63 in block_gemm_asmem_bsmem_creg.hpp)
        constexpr auto a_block_outer_dstr_encode = tile_distribution_encoding<
            sequence<NWarp>,  // REPLICATION: A shared across N-warps
            tuple<sequence<MIterPerWarp, MWarp>, sequence<KIterPerWarp>>,
            tuple<sequence<1, 0>>,
            tuple<sequence<1, 0>>,
            sequence<1, 2>,
            sequence<0, 0>>{};

        // B block distribution: replicated across MWarp (lines 77-83 in block_gemm_asmem_bsmem_creg.hpp)
        // B is [N x K], so dimensions are [NIterPerWarp, NWarp] x [KIterPerWarp]
        constexpr auto b_block_outer_dstr_encode = tile_distribution_encoding<
            sequence<MWarp>,  // REPLICATION: B shared across M-warps
            tuple<sequence<NIterPerWarp, NWarp>, sequence<KIterPerWarp>>,
            tuple<sequence<0, 1>>,
            tuple<sequence<0, 1>>,
            sequence<1, 2>,
            sequence<0, 0>>{};

        constexpr auto c_block_outer_dstr_encode = tile_distribution_encoding<
            sequence<>,
            tuple<sequence<MIterPerWarp, MWarp>, sequence<NIterPerWarp, NWarp>>,
            tuple<sequence<2, 1>>,
            tuple<sequence<1, 1>>,
            sequence<1, 2>,
            sequence<0, 0>>{};

        constexpr auto a_block_dstr_encode =
            detail::make_embed_tile_distribution_encoding(
                a_block_outer_dstr_encode, a_warp_dstr_encode);

        constexpr auto b_block_dstr_encode =
            detail::make_embed_tile_distribution_encoding(
                b_block_outer_dstr_encode, b_warp_dstr_encode);

        constexpr auto c_block_dstr_encode =
            detail::make_embed_tile_distribution_encoding(
                c_block_outer_dstr_encode, c_warp_dstr_encode);

        constexpr auto a_gemm_distribution = make_static_tile_distribution(a_block_dstr_encode);
        constexpr auto b_gemm_distribution = make_static_tile_distribution(b_block_dstr_encode);
        constexpr auto c_block_distribution = make_static_tile_distribution(c_block_dstr_encode);

        // GEMM windows for reading from LDS
        auto a_gemm_lds_window = make_tile_window(
            a_lds_view,
            make_tuple(number<kMPerBlock>{}, number<kKPerBlock>{}),
            {0, 0},
            a_gemm_distribution);

        auto b_gemm_lds_window = make_tile_window(
            b_lds_view,
            make_tuple(number<kNPerBlock>{}, number<kKPerBlock>{}),
            {0, 0},
            b_gemm_distribution);

        // Y-slicing info
        using AWarpDstr = decltype(make_static_tile_distribution(a_warp_dstr_encode));
        using BWarpDstr = decltype(make_static_tile_distribution(b_warp_dstr_encode));
        using CWarpDstr = decltype(make_static_tile_distribution(c_warp_dstr_encode));

        constexpr auto a_warp_y_lengths = to_sequence(AWarpDstr{}.get_ys_to_d_descriptor().get_lengths());
        constexpr auto b_warp_y_lengths = to_sequence(BWarpDstr{}.get_ys_to_d_descriptor().get_lengths());
        constexpr auto c_warp_y_lengths = to_sequence(CWarpDstr{}.get_ys_to_d_descriptor().get_lengths());

        constexpr auto a_warp_y_index_zeros = uniform_sequence_gen_t<AWarpDstr::NDimY, 0>{};
        constexpr auto b_warp_y_index_zeros = uniform_sequence_gen_t<BWarpDstr::NDimY, 0>{};
        constexpr auto c_warp_y_index_zeros = uniform_sequence_gen_t<CWarpDstr::NDimY, 0>{};

        // ====================================================================
        // Initialize Accumulator
        // ====================================================================

        auto c_block_tile = make_static_distributed_tensor<AccDataType>(c_block_distribution);
        set_tile(c_block_tile, AccDataType{0});

        // ====================================================================
        // Main K-Loop with LDS Staging
        // ====================================================================

        const index_t num_k_loops = K / kKPerBlock;

        for(index_t k_iter = 0; k_iter < num_k_loops; ++k_iter)
        {
            // Phase 1: Global -> Registers (COPY distribution, no replication)
            const auto a_copy_tile = load_tile(a_copy_global_window);
            const auto b_copy_tile = load_tile(b_copy_global_window);

            // Phase 2: Registers -> LDS
            store_tile(a_copy_lds_window, a_copy_tile);
            store_tile(b_copy_lds_window, b_copy_tile);

            // Phase 3: Synchronize
            block_sync_lds();

            // Phase 4: LDS -> Registers (GEMM distribution, WITH replication!)
            const auto a_block_tile = load_tile(a_gemm_lds_window);
            const auto b_block_tile = load_tile(b_gemm_lds_window);

            // Phase 5: Compute with Y-slicing
            static_for<0, KIterPerWarp, 1>{}([&](auto kIter) {
                static_for<0, MIterPerWarp, 1>{}([&](auto mIter) {
                    auto a_warp_tensor = make_static_distributed_tensor<ADataType>(
                        make_static_tile_distribution(a_warp_dstr_encode));

                    a_warp_tensor.get_thread_buffer() = a_block_tile.get_y_sliced_thread_data(
                        merge_sequences(sequence<mIter, kIter>{}, a_warp_y_index_zeros),
                        merge_sequences(sequence<1, 1>{}, a_warp_y_lengths));

                    static_for<0, NIterPerWarp, 1>{}([&](auto nIter) {
                        auto b_warp_tensor = make_static_distributed_tensor<BDataType>(
                            make_static_tile_distribution(b_warp_dstr_encode));

                        // B is [N x K], so Y-slice is (nIter, kIter)
                        b_warp_tensor.get_thread_buffer() = b_block_tile.get_y_sliced_thread_data(
                            merge_sequences(sequence<nIter, kIter>{}, b_warp_y_index_zeros),
                            merge_sequences(sequence<1, 1>{}, b_warp_y_lengths));

                        auto c_warp_tensor = make_static_distributed_tensor<AccDataType>(
                            make_static_tile_distribution(c_warp_dstr_encode));

                        c_warp_tensor.get_thread_buffer() = c_block_tile.get_y_sliced_thread_data(
                            merge_sequences(sequence<mIter, nIter>{}, c_warp_y_index_zeros),
                            merge_sequences(sequence<1, 1>{}, c_warp_y_lengths));

                        WarpGemm{}(c_warp_tensor, a_warp_tensor, b_warp_tensor);

                        c_block_tile.set_y_sliced_thread_data(
                            merge_sequences(sequence<mIter, nIter>{}, c_warp_y_index_zeros),
                            merge_sequences(sequence<1, 1>{}, c_warp_y_lengths),
                            c_warp_tensor.get_thread_buffer());
                    });
                });
            });

            // Phase 6: Move windows
            if(k_iter < num_k_loops - 1) {
                block_sync_lds();
                move_tile_window(a_copy_global_window, {0, kKPerBlock});
                move_tile_window(b_copy_global_window, {0, kKPerBlock});
            }
        }

        // ====================================================================
        // Epilogue
        // ====================================================================

        tile_elementwise_inout([alpha](auto& acc_val) { acc_val *= alpha; }, c_block_tile);

        if(std::abs(beta) > 1e-6f)
        {
            auto c_block_window = make_tile_window(
                c_tensor,
                make_tuple(number<kMPerBlock>{}, number<kNPerBlock>{}),
                {m_block_base, n_block_base},
                c_block_distribution);

            const auto c_input_block_tile = load_tile(c_block_window);

            tile_elementwise_inout(
                [beta](const auto& c_val, auto& acc_val) {
                    acc_val += beta * c_val;
                },
                c_input_block_tile, c_block_tile);
        }

        auto d_block_window = make_tile_window(
            d_tensor,
            make_tuple(number<kMPerBlock>{}, number<kNPerBlock>{}),
            {m_block_base, n_block_base},
            c_block_distribution);

        store_tile(d_block_window, c_block_tile);
    }
};

// CPU reference - computes C = alpha * A * B^T + beta * C
// where A is [M x K] and B is [N x K] (transposed)
template<typename InType, typename AccType>
void reference_gemm_transposed_b(const std::vector<InType>& a,
                                  const std::vector<InType>& b,  // B is [N x K]
                                  const std::vector<AccType>& c,
                                  std::vector<AccType>& d,
                                  index_t M, index_t N, index_t K,
                                  index_t lda, index_t ldb, index_t ldc, index_t ldd,
                                  AccType alpha, AccType beta)
{
    // C[m,n] = sum_k A[m,k] * B[n,k]
    // A is column-major: A[m,k] = a[m + k*lda]
    // B is row-major [N x K]: B[n,k] = b[n*ldb + k]
    for(index_t n = 0; n < N; ++n) {
        for(index_t m = 0; m < M; ++m) {
            AccType sum = 0;
            for(index_t k = 0; k < K; ++k) {
                sum += static_cast<AccType>(a[m + k * lda]) *
                       static_cast<AccType>(b[n * ldb + k]);
            }
            d[m + n * ldd] = alpha * sum + beta * c[m + n * ldc];
        }
    }
}

template<typename T>
void fill_random(std::vector<T>& data, T min_val = -1, T max_val = 1)
{
    for(auto& val : data) {
        val = static_cast<T>(min_val + (max_val - min_val) *
                             static_cast<float>(rand()) / RAND_MAX);
    }
}

int main()
{
    std::cout << "\n==================================================\n";
    std::cout << "Tutorial 08: LDS Staging for GEMM (02_gemm pattern)\n";
    std::cout << "==================================================\n\n";

    std::cout << "Memory layout (following 02_gemm):\n";
    std::cout << "  A: [M x K] column-major\n";
    std::cout << "  B: [N x K] row-major (transposed B!)\n";
    std::cout << "  GEMM computes: C = A * B^T\n\n";

    std::cout << "LDS Reuse pattern:\n";
    std::cout << "  Copy distribution: All threads load cooperatively (no replication)\n";
    std::cout << "  GEMM distribution: Warps read with replication\n";
    std::cout << "    A: replicated across NWarp=2 (2x reuse)\n";
    std::cout << "    B: replicated across MWarp=2 (2x reuse)\n\n";

    constexpr index_t M = 2048;
    constexpr index_t N = 2048;
    constexpr index_t K = 2048;

    // A: [M x K] column-major, lda = M
    // B: [N x K] row-major, ldb = K
    constexpr index_t lda = M;
    constexpr index_t ldb = K;  // B is row-major [N x K]
    constexpr index_t ldc = M;
    constexpr index_t ldd = M;

    using InputType = half_t;
    using AccumType = float;

    constexpr AccumType alpha = 2.0f;
    constexpr AccumType beta = 1.5f;

    using KernelType = LdsStagingHgemmKernel<InputType, InputType, AccumType, AccumType>;
    constexpr index_t lds_size = KernelType::GetStaticLdsSize();

    std::cout << "Problem configuration:\n";
    std::cout << "  M x N x K: " << M << " x " << N << " x " << K << "\n";
    std::cout << "  Block output: " << KernelType::kMPerBlock << " x " << KernelType::kNPerBlock << "\n";
    std::cout << "  kKPerBlock: " << KernelType::kKPerBlock << "\n";
    std::cout << "  KIterPerWarp: " << KernelType::KIterPerWarp << "\n";
    std::cout << "  LDS size: " << lds_size << " bytes\n\n";

    // A: [M x K], B: [N x K]
    std::vector<InputType> h_a(M * K);
    std::vector<InputType> h_b(N * K);  // B is [N x K]!
    std::vector<AccumType> h_c(M * N);
    std::vector<AccumType> h_d(M * N, std::numeric_limits<AccumType>::quiet_NaN());
    std::vector<AccumType> h_d_ref(M * N);

    srand(42);
    fill_random(h_a, InputType(-1), InputType(1));
    fill_random(h_b, InputType(-1), InputType(1));
    fill_random(h_c, AccumType(-1), AccumType(1));

    auto cpu_start = std::chrono::high_resolution_clock::now();
    reference_gemm_transposed_b(h_a, h_b, h_c, h_d_ref, M, N, K, lda, ldb, ldc, ldd, alpha, beta);
    auto cpu_end = std::chrono::high_resolution_clock::now();
    double cpu_time_ms = std::chrono::duration<double, std::milli>(cpu_end - cpu_start).count();

    DeviceMem d_a(M * K * sizeof(InputType));
    DeviceMem d_b(N * K * sizeof(InputType));  // B is [N x K]
    DeviceMem d_c(M * N * sizeof(AccumType));
    DeviceMem d_d(M * N * sizeof(AccumType));

    d_a.ToDevice(h_a.data(), M * K * sizeof(InputType));
    d_b.ToDevice(h_b.data(), N * K * sizeof(InputType));
    d_c.ToDevice(h_c.data(), M * N * sizeof(AccumType));
    d_d.ToDevice(h_d.data(), M * N * sizeof(AccumType));

    constexpr index_t block_size = KernelType::kBlockSize;
    const index_t grid_size = (M / KernelType::kMPerBlock) * (N / KernelType::kNPerBlock);

    std::cout << "Launching kernel:\n";
    std::cout << "  Grid: " << grid_size << " blocks\n";
    std::cout << "  Block: " << block_size << " threads\n\n";

    stream_config stream;

    // Warmup
    for(int i = 0; i < 5; ++i) {
        launch_kernel(stream,
                     make_kernel<block_size>(
                         KernelType{},
                         dim3(grid_size),
                         dim3(block_size),
                         lds_size,
                         static_cast<const InputType*>(d_a.GetDeviceBuffer()),
                         static_cast<const InputType*>(d_b.GetDeviceBuffer()),
                         static_cast<const AccumType*>(d_c.GetDeviceBuffer()),
                         static_cast<AccumType*>(d_d.GetDeviceBuffer()),
                         M, N, K, lda, ldb, ldc, ldd, alpha, beta));
    }
    hip_check_error(hipDeviceSynchronize());

    auto gpu_start = std::chrono::high_resolution_clock::now();

    launch_kernel(stream,
                 make_kernel<block_size>(
                     KernelType{},
                     dim3(grid_size),
                     dim3(block_size),
                     lds_size,
                     static_cast<const InputType*>(d_a.GetDeviceBuffer()),
                     static_cast<const InputType*>(d_b.GetDeviceBuffer()),
                     static_cast<const AccumType*>(d_c.GetDeviceBuffer()),
                     static_cast<AccumType*>(d_d.GetDeviceBuffer()),
                     M, N, K, lda, ldb, ldc, ldd, alpha, beta));

    hip_check_error(hipDeviceSynchronize());

    auto gpu_end = std::chrono::high_resolution_clock::now();
    double gpu_time_ms = std::chrono::duration<double, std::milli>(gpu_end - gpu_start).count();

    d_d.FromDevice(h_d.data(), M * N * sizeof(AccumType));

    bool passed = true;
    float max_error = 0;
    index_t error_count = 0;

    for(index_t i = 0; i < M * N; ++i) {
        float error = std::abs(h_d[i] - h_d_ref[i]);
        max_error = std::max(max_error, error);
        if(error > 1e-2f) {
            if(error_count < 5) {
                index_t m = i % M;
                index_t n = i / M;
                std::cout << "Error at [" << m << "," << n << "]: "
                          << h_d[i] << " vs " << h_d_ref[i]
                          << " (diff=" << error << ")\n";
            }
            error_count++;
        }
    }

    passed = (error_count == 0);

    double gflops = 2.0 * M * N * K / 1e9;
    double gpu_tflops = gflops / (gpu_time_ms / 1000);
    double cpu_gflops = gflops / (cpu_time_ms / 1000);

    std::cout << "Results:\n";
    std::cout << "  Correctness: " << (passed ? "PASSED" : "FAILED") << "\n";
    std::cout << "  Max error: " << max_error << "\n";
    if(!passed) std::cout << "  Error count: " << error_count << "/" << M*N << "\n";
    std::cout << "\n";

    std::cout << "Performance:\n";
    std::cout << "  CPU time: " << cpu_time_ms << " ms (" << cpu_gflops << " GFLOPS)\n";
    std::cout << "  GPU time: " << gpu_time_ms << " ms (" << gpu_tflops << " TFLOPS)\n";
    std::cout << "  Speedup: " << cpu_time_ms / gpu_time_ms << "x\n\n";

    return passed ? 0 : 1;
}
