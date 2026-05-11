// SPDX-License-Identifier: MIT
// Copyright (c) 2024, Advanced Micro Devices, Inc. All rights reserved.

/*
 * Tutorial 13: Production-Style XOR with Register Transpose
 *
 * Fixes B's LDS bank conflicts by using the production GEMM pattern:
 * 1. Load B from global as [K, N] (vector loads along contiguous N)
 * 2. transpose_tile2d in registers: [K, N] → [N, K]
 * 3. Store to LDS as [N, K] with XOR (N is slow dim, K is fast — same as A)
 * 4. GEMM reads from [N, K] LDS with proper distribution
 *
 * This is exactly what production CK GEMM kernels do (see gemm_pipeline_ag_bg_cr_mem.hpp).
 * The XOR pattern works correctly because both A [M,K] and B [N,K] have the same
 * layout convention: slow dimension first, K (vectorized) second.
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

template<typename ADataType, typename BDataType, typename CDataType, typename AccDataType>
struct ProductionXorHgemmKernel
{
    static constexpr index_t kWaveSize = 64;
    static constexpr index_t kWarpM = 16;
    static constexpr index_t kWarpN = 16;
    static constexpr index_t kWarpK = 16;

    static constexpr index_t MWarp = 2;
    static constexpr index_t NWarp = 2;
    static constexpr index_t kBlockSize = MWarp * NWarp * kWaveSize;  // 256

    static constexpr index_t MIterPerWarp = 2;
    static constexpr index_t NIterPerWarp = 2;

    static constexpr index_t kKPerBlock = 32;
    static constexpr index_t KIterPerWarp = kKPerBlock / kWarpK;  // 2

    using WarpGemm = WarpGemmMfmaF16F16F32M16N16K16;

    CK_TILE_HOST_DEVICE static constexpr index_t GetStaticLdsSize()
    {
        constexpr index_t kMPerBlock = MWarp * MIterPerWarp * kWarpM;
        constexpr index_t kNPerBlock = NWarp * NIterPerWarp * kWarpN;
        constexpr index_t a_lds_size = kMPerBlock * kKPerBlock * sizeof(ADataType);
        constexpr index_t b_lds_size = kNPerBlock * kKPerBlock * sizeof(BDataType);
        constexpr index_t a_lds_aligned = ((a_lds_size + 15) / 16) * 16;
        return a_lds_aligned + b_lds_size;
    }

    // A copy distribution: [M, K] (unchanged from Tutorial 10)
    template<typename DataType>
    CK_TILE_HOST_DEVICE static constexpr auto MakeACopyDistribution()
    {
        constexpr index_t kMPerBlock = MWarp * MIterPerWarp * kWarpM;
        constexpr index_t K1 = 16 / sizeof(DataType);
        constexpr index_t K0 = kKPerBlock / K1;
        constexpr index_t M2 = kWaveSize / K0;
        constexpr index_t M1 = kBlockSize / kWaveSize;
        constexpr index_t M0 = kMPerBlock / (M2 * M1);
        return make_static_tile_distribution(
            tile_distribution_encoding<
                sequence<1>,
                tuple<sequence<M0, M1, M2>, sequence<K0, K1>>,
                tuple<sequence<1>, sequence<1, 2>>,
                tuple<sequence<1>, sequence<2, 0>>,
                sequence<1, 2>,
                sequence<0, 1>>{}
        );
    }

    // B copy distribution: [K, N] — loads from global (unchanged from Tutorial 10)
    template<typename DataType>
    CK_TILE_HOST_DEVICE static constexpr auto MakeBCopyDistribution()
    {
        constexpr index_t kNPerBlock = NWarp * NIterPerWarp * kWarpN;
        constexpr index_t N1 = 16 / sizeof(DataType);
        constexpr index_t N0 = kNPerBlock / N1;
        constexpr index_t K2 = kWaveSize / N0;
        constexpr index_t K1 = kBlockSize / kWaveSize;
        constexpr index_t K0 = kKPerBlock / (K2 * K1);
        return make_static_tile_distribution(
            tile_distribution_encoding<
                sequence<1>,
                tuple<sequence<K0, K1, K2>, sequence<N0, N1>>,
                tuple<sequence<1>, sequence<1, 2>>,
                tuple<sequence<1>, sequence<2, 0>>,
                sequence<1, 2>,
                sequence<0, 1>>{}
        );
    }

    // *** NEW: B shuffled distribution for [N, K] LDS store after transpose ***
    // Swapped H dims from MakeBCopyDistribution: H0=N, H1=K.
    // Per-thread Y shape must be (8, 1) — the reverse of the copy's (1, 8) —
    // so that transpose_tile2d can rearrange between them.
    template<typename DataType>
    CK_TILE_HOST_DEVICE static constexpr auto MakeBShuffledDistribution()
    {
        constexpr index_t kNPerBlock = NWarp * NIterPerWarp * kWarpN;  // 64
        constexpr index_t N1 = 16 / sizeof(DataType);  // 8
        constexpr index_t N0 = kNPerBlock / N1;         // 8
        constexpr index_t K2 = kWaveSize / N0;          // 8
        constexpr index_t K1 = kBlockSize / kWaveSize;  // 4
        constexpr index_t K0 = kKPerBlock / (K2 * K1);  // 1

        // H0 = N(N0=8, N1=8), H1 = K(K0=1, K1=4, K2=8)
        // P0 → K1=4 (wave id), P1 → K2=8 × N0=8 = 64 (lane id)
        // Y0 → N1=8, Y1 → K0=1  → per-thread shape (8, 1) = reverse of copy's (1, 8)
        return make_static_tile_distribution(
            tile_distribution_encoding<
                sequence<1>,
                tuple<sequence<N0, N1>, sequence<K0, K1, K2>>,
                tuple<sequence<2>, sequence<2, 1>>,
                tuple<sequence<1>, sequence<2, 0>>,
                sequence<1, 2>,
                sequence<1, 0>>{}
        );
    }

    // A GEMM distribution: [M, K] (unchanged from Tutorial 10)
    CK_TILE_HOST_DEVICE static constexpr auto MakeAGemmDistribution()
    {
        constexpr auto a_warp_dstr_encode = tile_distribution_encoding<
            sequence<>,
            tuple<sequence<16>, sequence<4, 4>>,
            tuple<sequence<2, 1>>,
            tuple<sequence<0, 0>>,
            sequence<2>,
            sequence<1>>{};

        constexpr auto a_block_outer_dstr_encode = tile_distribution_encoding<
            sequence<NWarp>,
            tuple<sequence<MIterPerWarp, MWarp>, sequence<KIterPerWarp>>,
            tuple<sequence<0, 1>>,
            tuple<sequence<0, 1>>,
            sequence<1, 2>,
            sequence<0, 0>>{};

        return make_static_tile_distribution(
            detail::make_embed_tile_distribution_encoding(
                a_block_outer_dstr_encode, a_warp_dstr_encode));
    }

    // *** CHANGED: B GEMM distribution for [N, K] LDS layout ***
    // H0 = N, H1 = K at both warp and block level (matches production)
    CK_TILE_HOST_DEVICE static constexpr auto MakeBGemmDistribution()
    {
        // Warp: H0=N(16), H1=K(4,4) — matches WarpGemm::BWarpDstrEncoding
        constexpr auto b_warp_dstr_encode = tile_distribution_encoding<
            sequence<>,
            tuple<sequence<16>, sequence<4, 4>>,
            tuple<sequence<2, 1>>,
            tuple<sequence<0, 0>>,
            sequence<2>,
            sequence<1>>{};

        // Block: H0=N(NIter,NWarp), H1=K(KIter) — matches production
        constexpr auto b_block_outer_dstr_encode = tile_distribution_encoding<
            sequence<MWarp>,
            tuple<sequence<NIterPerWarp, NWarp>, sequence<KIterPerWarp>>,
            tuple<sequence<1, 0>>,
            tuple<sequence<1, 0>>,
            sequence<1, 2>,
            sequence<0, 0>>{};

        return make_static_tile_distribution(
            detail::make_embed_tile_distribution_encoding(
                b_block_outer_dstr_encode, b_warp_dstr_encode));
    }

    CK_TILE_DEVICE void operator()(const ADataType* a,
                                   const BDataType* b,
                                   const CDataType* c,
                                   CDataType* d,
                                   index_t M, index_t N, index_t K,
                                   index_t lda, index_t ldb, index_t ldc, index_t ldd,
                                   AccDataType alpha, AccDataType beta) const
    {
        extern __shared__ char smem[];
        void* p_smem = static_cast<void*>(smem);
        [[maybe_unused]] const index_t warp_id = get_warp_id();

        constexpr index_t kMPerBlock = MWarp * MIterPerWarp * kWarpM;  // 64
        constexpr index_t kNPerBlock = NWarp * NIterPerWarp * kWarpN;  // 64

        const index_t num_blocks_n = N / kNPerBlock;
        const index_t block_m = get_block_id() / num_blocks_n;
        const index_t block_n = get_block_id() % num_blocks_n;
        const index_t m_block_base = block_m * kMPerBlock;
        const index_t n_block_base = block_n * kNPerBlock;

        if(m_block_base >= M || n_block_base >= N)
            return;

        // Global views (same as Tutorial 10)
        const auto a_tensor = make_naive_tensor_view<address_space_enum::global>(
            a, make_tuple(M, K), make_tuple(1, lda), number<1>{}, number<1>{});
        const auto b_tensor = make_naive_tensor_view<address_space_enum::global>(
            b, make_tuple(K, N), make_tuple(ldb, 1), number<4>{}, number<1>{});
        const auto c_tensor = make_naive_tensor_view<address_space_enum::global>(
            c, make_tuple(M, N), make_tuple(1, ldc), number<1>{}, number<1>{});
        auto d_tensor = make_naive_tensor_view<address_space_enum::global>(
            d, make_tuple(M, N), make_tuple(1, ldd), number<1>{}, number<1>{});

        // ==================================================================
        // A LDS DESCRIPTOR: [M, K] with XOR (unchanged from Tutorial 10)
        // ==================================================================
        static constexpr index_t kKPack = 8;
        constexpr auto DataTypeSize = sizeof(ADataType);
        constexpr auto MLdsLayer =
            (32 * 4 / kKPerBlock / DataTypeSize) < 1 ? 1 : (32 * 4 / kKPerBlock / DataTypeSize);

        constexpr auto a_lds_block_desc_0 = make_naive_tensor_descriptor(
            make_tuple(number<kKPerBlock / kKPack * MLdsLayer>{},
                       number<kMPerBlock / MLdsLayer>{},
                       number<kKPack>{}),
            make_tuple(number<kKPack>{}, number<kKPerBlock * MLdsLayer>{}, number<1>{}),
            number<kKPack>{}, number<1>{});

        constexpr auto a_lds_block_desc_permuted = transform_tensor_descriptor(
            a_lds_block_desc_0,
            make_tuple(make_xor_transform(make_tuple(number<kMPerBlock / MLdsLayer>{},
                                                     number<kKPerBlock / kKPack * MLdsLayer>{})),
                       make_pass_through_transform(number<kKPack>{})),
            make_tuple(sequence<1, 0>{}, sequence<2>{}),
            make_tuple(sequence<1, 0>{}, sequence<2>{}));

        constexpr auto a_lds_block_desc_unmerged = transform_tensor_descriptor(
            a_lds_block_desc_permuted,
            make_tuple(make_unmerge_transform(
                           make_tuple(number<MLdsLayer>{}, number<kKPerBlock / kKPack>{})),
                       make_pass_through_transform(number<kMPerBlock / MLdsLayer>{}),
                       make_pass_through_transform(number<kKPack>{})),
            make_tuple(sequence<0>{}, sequence<1>{}, sequence<2>{}),
            make_tuple(sequence<0, 2>{}, sequence<1>{}, sequence<3>{}));

        constexpr auto a_lds_desc = transform_tensor_descriptor(
            a_lds_block_desc_unmerged,
            make_tuple(
                make_merge_transform(
                    make_tuple(number<kMPerBlock / MLdsLayer>{}, number<MLdsLayer>{})),
                make_merge_transform(make_tuple(number<kKPerBlock / kKPack>{}, number<kKPack>{}))),
            make_tuple(sequence<1, 0>{}, sequence<2, 3>{}),
            make_tuple(sequence<0>{}, sequence<1>{}));

        // ==================================================================
        // *** B LDS DESCRIPTOR: [N, K] with XOR (production pattern) ***
        // ==================================================================
        // Same XOR pattern as A but with N as the slow dimension.
        // This is what production GEMM uses for B after the register transpose.
        constexpr auto NLdsLayer =
            (32 * 4 / kKPerBlock / DataTypeSize) < 1 ? 1 : (32 * 4 / kKPerBlock / DataTypeSize);

        constexpr auto b_lds_block_desc_0 = make_naive_tensor_descriptor(
            make_tuple(number<kKPerBlock / kKPack * NLdsLayer>{},
                       number<kNPerBlock / NLdsLayer>{},
                       number<kKPack>{}),
            make_tuple(number<kKPack>{}, number<kKPerBlock * NLdsLayer>{}, number<1>{}),
            number<kKPack>{}, number<1>{});

        constexpr auto b_lds_block_desc_permuted = transform_tensor_descriptor(
            b_lds_block_desc_0,
            make_tuple(make_xor_transform(make_tuple(number<kNPerBlock / NLdsLayer>{},
                                                     number<kKPerBlock / kKPack * NLdsLayer>{})),
                       make_pass_through_transform(number<kKPack>{})),
            make_tuple(sequence<1, 0>{}, sequence<2>{}),
            make_tuple(sequence<1, 0>{}, sequence<2>{}));

        constexpr auto b_lds_block_desc_unmerged = transform_tensor_descriptor(
            b_lds_block_desc_permuted,
            make_tuple(make_unmerge_transform(
                           make_tuple(number<NLdsLayer>{}, number<kKPerBlock / kKPack>{})),
                       make_pass_through_transform(number<kNPerBlock / NLdsLayer>{}),
                       make_pass_through_transform(number<kKPack>{})),
            make_tuple(sequence<0>{}, sequence<1>{}, sequence<2>{}),
            make_tuple(sequence<0, 2>{}, sequence<1>{}, sequence<3>{}));

        // Output [N, K] — production pattern
        constexpr auto b_lds_desc = transform_tensor_descriptor(
            b_lds_block_desc_unmerged,
            make_tuple(
                make_merge_transform(
                    make_tuple(number<kNPerBlock / NLdsLayer>{}, number<NLdsLayer>{})),
                make_merge_transform(make_tuple(number<kKPerBlock / kKPack>{}, number<kKPack>{}))),
            make_tuple(sequence<1, 0>{}, sequence<2, 3>{}),
            make_tuple(sequence<0>{}, sequence<1>{}));

        // LDS pointers
        ADataType* p_a_lds = static_cast<ADataType*>(p_smem);
        constexpr index_t a_lds_size_aligned =
            ((kMPerBlock * kKPerBlock * sizeof(ADataType) + 15) / 16) * 16;
        BDataType* p_b_lds = static_cast<BDataType*>(
            static_cast<void*>(static_cast<char*>(p_smem) + a_lds_size_aligned));

        auto a_lds_view = make_tensor_view<address_space_enum::lds>(p_a_lds, a_lds_desc);
        auto b_lds_view = make_tensor_view<address_space_enum::lds>(p_b_lds, b_lds_desc);

        // ==================================================================
        // DISTRIBUTIONS AND Y-DIMENSION SETUP
        // ==================================================================
        constexpr auto a_warp_dstr_encode = tile_distribution_encoding<
            sequence<>,
            tuple<sequence<16>, sequence<4, 4>>,
            tuple<sequence<2, 1>>,
            tuple<sequence<0, 0>>,
            sequence<2>,
            sequence<1>>{};

        // *** B warp: [N, K] — matches WarpGemm::BWarpDstrEncoding ***
        constexpr auto b_warp_dstr_encode = tile_distribution_encoding<
            sequence<>,
            tuple<sequence<16>, sequence<4, 4>>,
            tuple<sequence<2, 1>>,
            tuple<sequence<0, 0>>,
            sequence<2>,
            sequence<1>>{};

        constexpr auto c_warp_dstr_encode = tile_distribution_encoding<
            sequence<>,
            tuple<sequence<4, 4>, sequence<16>>,
            tuple<sequence<1, 2>>,
            tuple<sequence<0, 0>>,
            sequence<1>,
            sequence<1>>{};

        constexpr auto c_block_outer_dstr_encode = tile_distribution_encoding<
            sequence<>,
            tuple<sequence<MIterPerWarp, MWarp>, sequence<NIterPerWarp, NWarp>>,
            tuple<sequence<2, 1>>,
            tuple<sequence<1, 1>>,
            sequence<1, 2>,
            sequence<0, 0>>{};

        constexpr auto c_block_dstr_encode =
            detail::make_embed_tile_distribution_encoding(
                c_block_outer_dstr_encode, c_warp_dstr_encode);
        constexpr auto c_block_distribution = make_static_tile_distribution(c_block_dstr_encode);

        using AWarpDstr = decltype(make_static_tile_distribution(a_warp_dstr_encode));
        using BWarpDstr = decltype(make_static_tile_distribution(b_warp_dstr_encode));
        using CWarpDstr = decltype(make_static_tile_distribution(c_warp_dstr_encode));

        constexpr auto a_warp_y_lengths = to_sequence(AWarpDstr{}.get_ys_to_d_descriptor().get_lengths());
        constexpr auto b_warp_y_lengths = to_sequence(BWarpDstr{}.get_ys_to_d_descriptor().get_lengths());
        constexpr auto c_warp_y_lengths = to_sequence(CWarpDstr{}.get_ys_to_d_descriptor().get_lengths());

        constexpr auto a_warp_y_index_zeros = uniform_sequence_gen_t<AWarpDstr::NDimY, 0>{};
        constexpr auto b_warp_y_index_zeros = uniform_sequence_gen_t<BWarpDstr::NDimY, 0>{};
        constexpr auto c_warp_y_index_zeros = uniform_sequence_gen_t<CWarpDstr::NDimY, 0>{};

        // ==================================================================
        // COPY WINDOWS (global → registers)
        // ==================================================================
        auto a_copy_dram_window = make_tile_window(
            a_tensor, make_tuple(number<kMPerBlock>{}, number<kKPerBlock>{}),
            {m_block_base, 0}, MakeACopyDistribution<ADataType>());

        // B loads from global as [K, N] (same as Tutorial 10)
        auto b_copy_dram_window = make_tile_window(
            b_tensor, make_tuple(number<kKPerBlock>{}, number<kNPerBlock>{}),
            {0, n_block_base}, MakeBCopyDistribution<BDataType>());

        // A LDS store window (same as Tutorial 10)
        auto a_copy_lds_window = make_tile_window(
            a_lds_view, make_tuple(number<kMPerBlock>{}, number<kKPerBlock>{}),
            {0, 0}, a_copy_dram_window.get_tile_distribution());

        // *** B LDS store window: [N, K] with shuffled distribution ***
        auto b_shuffle_lds_window = make_tile_window(
            b_lds_view, make_tuple(number<kNPerBlock>{}, number<kKPerBlock>{}),
            {0, 0}, MakeBShuffledDistribution<BDataType>());

        // ==================================================================
        // GEMM WINDOWS (LDS → registers for compute)
        // ==================================================================
        auto a_lds_gemm_window = make_tile_window(
            a_lds_view, make_tuple(number<kMPerBlock>{}, number<kKPerBlock>{}),
            {0, 0}, MakeAGemmDistribution());

        // *** B GEMM window: [N, K] with GEMM distribution ***
        auto b_lds_gemm_window = make_tile_window(
            b_lds_view, make_tuple(number<kNPerBlock>{}, number<kKPerBlock>{}),
            {0, 0}, MakeBGemmDistribution());

        auto c_block_tile = make_static_distributed_tensor<AccDataType>(c_block_distribution);
        set_tile(c_block_tile, AccDataType{0});

        // ==================================================================
        // MAIN K-LOOP
        // ==================================================================
        const index_t num_k_loops = K / kKPerBlock;
        for(index_t k_iter = 0; k_iter < num_k_loops; ++k_iter)
        {
            // Phase 1: Global → Registers
            const auto a_block_tile_copy = load_tile(a_copy_dram_window);
            const auto b_block_tile_copy = load_tile(b_copy_dram_window);

            // Phase 2a: A Registers → LDS (direct, same as Tutorial 10)
            store_tile(a_copy_lds_window, a_block_tile_copy);

            // Phase 2b: *** B Registers → transpose → LDS ***
            // transpose_tile2d rearranges data from [K, N] to [N, K] in registers.
            // Then store to [N, K] LDS descriptor with XOR.
            auto b_shuffle_tmp = make_static_distributed_tensor<BDataType>(
                MakeBShuffledDistribution<BDataType>());
            transpose_tile2d(b_shuffle_tmp, b_block_tile_copy);
            store_tile(b_shuffle_lds_window, b_shuffle_tmp);

            // Phase 3: Sync
            block_sync_lds();

            // Phase 4: LDS → Registers (GEMM distributions)
            const auto a_block_tile = load_tile(a_lds_gemm_window);
            const auto b_block_tile = load_tile(b_lds_gemm_window);

            // Phase 5: Compute with Y-slicing
            // *** B Y-dims are now (NIter, KIter) since B GEMM is [N, K] ***
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

                        // *** B slice: (nIter, kIter) — N first, K second ***
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
                move_tile_window(a_copy_dram_window, {0, kKPerBlock});
                move_tile_window(b_copy_dram_window, {kKPerBlock, 0});
            }
        }

        tile_elementwise_inout([alpha](auto& acc_val) { acc_val *= alpha; }, c_block_tile);

        if(std::abs(beta) > 1e-6f)
        {
            auto c_block_window = make_tile_window(
                c_tensor, make_tuple(number<kMPerBlock>{}, number<kNPerBlock>{}),
                {m_block_base, n_block_base}, c_block_distribution);

            const auto c_input_block_tile = load_tile(c_block_window);
            tile_elementwise_inout(
                [beta](const auto& c_val, auto& acc_val) { acc_val += beta * c_val; },
                c_input_block_tile, c_block_tile);
        }

        auto d_block_window = make_tile_window(
            d_tensor, make_tuple(number<kMPerBlock>{}, number<kNPerBlock>{}),
            {m_block_base, n_block_base}, c_block_distribution);

        store_tile(d_block_window, c_block_tile);
    }
};

template<typename InType, typename AccType>
void reference_gemm_mixed(const std::vector<InType>& a,
                          const std::vector<InType>& b,
                          const std::vector<AccType>& c,
                          std::vector<AccType>& d,
                          index_t M, index_t N, index_t K,
                          index_t lda, index_t ldb, index_t ldc, index_t ldd,
                          AccType alpha, AccType beta)
{
    for(index_t n = 0; n < N; ++n) {
        for(index_t m = 0; m < M; ++m) {
            AccType sum = 0;
            for(index_t k = 0; k < K; ++k) {
                sum += static_cast<AccType>(a[m + k * lda]) *
                       static_cast<AccType>(b[k * ldb + n]);
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

int main(int argc, char* argv[])
{
    std::cout << "\n==================================================\n";
    std::cout << "Tutorial 13: Production XOR with Register Transpose\n";
    std::cout << "==================================================\n\n";

    std::cout << "Key: B loaded as [K,N], transposed in registers to [N,K],\n";
    std::cout << "     stored to LDS as [N,K] with XOR (matching A's pattern).\n\n";

    index_t M = 128;
    index_t N = 128;
    index_t K = 64;

    if(argc >= 4) {
        M = std::atoi(argv[1]);
        N = std::atoi(argv[2]);
        K = std::atoi(argv[3]);
    }

    const index_t lda = M;
    const index_t ldb = N;
    const index_t ldc = M;
    const index_t ldd = M;

    using InputType = half_t;
    using AccumType = float;
    constexpr AccumType alpha = 2.0f;
    constexpr AccumType beta = 1.5f;

    std::cout << "Problem: " << M << "x" << N << "x" << K << "\n";

    double cpu_time_ms = 0;
    bool run_cpu = (M <= 2048 && N <= 2048);

    std::vector<InputType> h_a(M * K);
    std::vector<InputType> h_b(K * N);
    std::vector<AccumType> h_c(M * N);
    std::vector<AccumType> h_d(M * N);
    std::vector<AccumType> h_d_ref;
    if(run_cpu) h_d_ref.resize(M * N);

    srand(42);
    fill_random(h_a, InputType(-1), InputType(1));
    fill_random(h_b, InputType(-1), InputType(1));
    fill_random(h_c, AccumType(-1), AccumType(1));

    if(run_cpu) {
        auto cpu_start = std::chrono::high_resolution_clock::now();
        reference_gemm_mixed(h_a, h_b, h_c, h_d_ref, M, N, K, lda, ldb, ldc, ldd, alpha, beta);
        auto cpu_end = std::chrono::high_resolution_clock::now();
        cpu_time_ms = std::chrono::duration<double, std::milli>(cpu_end - cpu_start).count();
    }

    DeviceMem d_a(M * K * sizeof(InputType));
    DeviceMem d_b(K * N * sizeof(InputType));
    DeviceMem d_c(M * N * sizeof(AccumType));
    DeviceMem d_d(M * N * sizeof(AccumType));

    d_a.ToDevice(h_a.data(), M * K * sizeof(InputType));
    d_b.ToDevice(h_b.data(), K * N * sizeof(InputType));
    d_c.ToDevice(h_c.data(), M * N * sizeof(AccumType));
    d_d.ToDevice(h_d.data(), M * N * sizeof(AccumType));

    constexpr index_t block_size = 256;
    const index_t grid_size = (M / 64) * (N / 64);
    stream_config stream;
    constexpr index_t lds_size = ProductionXorHgemmKernel<InputType, InputType, AccumType, AccumType>::GetStaticLdsSize();

    std::cout << "LDS: " << lds_size << " bytes\n\n";

    for(int i = 0; i < 5; ++i) {
        launch_kernel(stream,
                     make_kernel<block_size>(
                         ProductionXorHgemmKernel<InputType, InputType, AccumType, AccumType>{},
                         dim3(grid_size), dim3(block_size), lds_size,
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
                     ProductionXorHgemmKernel<InputType, InputType, AccumType, AccumType>{},
                     dim3(grid_size), dim3(block_size), lds_size,
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

    if(run_cpu) {
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
    } else {
        std::cout << "Skipping CPU verification for large size\n";
    }

    double gflops = 2.0 * M * N * K / 1e9;
    double gpu_tflops = gflops / (gpu_time_ms / 1000);

    std::cout << "\nResults:\n";
    std::cout << "  Correctness: " << (passed ? "PASSED" : "FAILED") << "\n";
    std::cout << "  Max error: " << max_error << "\n";
    if(!passed) std::cout << "  Error count: " << error_count << "/" << M*N << "\n";
    if(run_cpu) {
        double cpu_gflops = gflops / (cpu_time_ms / 1000);
        std::cout << "  CPU time: " << cpu_time_ms << " ms (" << cpu_gflops << " GFLOPS)\n";
    }
    std::cout << "  GPU time: " << gpu_time_ms << " ms (" << gpu_tflops << " TFLOPS)\n\n";

    std::cout << "Profile bank conflicts:\n";
    std::cout << "  rocprofv3 --pmc SQ_LDS_BANK_CONFLICT,SQ_INSTS_LDS -d /tmp/rocprof_t13 -f csv -- ./bin/aa_tutorial_13_production_xor 4096 4096 4096\n\n";

    return passed ? 0 : 1;
}
