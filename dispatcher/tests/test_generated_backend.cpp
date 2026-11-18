// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

/// Unit tests for the supports() gate of the generated GEMM backends.
/// Note: Uses a stub kernel struct; no kernel is compiled or launched.

#include "ck_tile/ops/epilogue/cshuffle_epilogue.hpp"
#include "ck_tile/ops/gemm/kernel/gemm_kernel.hpp"
#include "ck_tile/dispatcher/backends/generated_kernel_backend.hpp"
#include "ck_tile/dispatcher/backends/generated_tile_backend.hpp"
#include "ck_tile/dispatcher/dispatcher.hpp"
#include "ck_tile/dispatcher/registry.hpp"
#include "test_mock_kernel.hpp"
#include <gtest/gtest.h>

using namespace ck_tile::dispatcher;
using namespace ck_tile::dispatcher::test;

namespace {

// Fully padded fp16 kernel, as the codegen emits for every fixed-width (_vec) kernel.
template <int A, int B, int C>
struct PaddedStubKernel
{
    using ADataType             = ck_tile::half_t;
    using BDataType             = ck_tile::half_t;
    using CDataType             = ck_tile::half_t;
    using AccDataType           = float;
    static constexpr bool kPadM = true;
    static constexpr bool kPadN = true;
    static constexpr bool kPadK = true;
    static constexpr int TileM  = 128;
    static constexpr int TileN  = 128;
    static constexpr int TileK  = 32;
    struct GemmPipeline
    {
        template <bool = false>
        static constexpr int GetVectorSizeA()
        {
            return A;
        }
        template <bool = false>
        static constexpr int GetVectorSizeB()
        {
            return B;
        }
    };
    static constexpr int VectorSizeC = C;
    static float launch(const ck_tile::GemmHostArgs&, const ck_tile::stream_config&) { return 0.f; }
};

template <int A, int B, int C>
using LegacyInstance = backends::GeneratedKernelInstance<PaddedStubKernel<A, B, C>>;
template <int A, int B, int C>
using TileInstance = backends::GeneratedTileKernelInstance<PaddedStubKernel<A, B, C>,
                                                           ck_tile::half_t,
                                                           ck_tile::half_t,
                                                           ck_tile::half_t,
                                                           float>;

KernelKey make_vec_key()
{
    KernelKey key               = make_test_key(128, 128, 32, "gfx950");
    key.algorithm.vector_size_a = 1;
    key.algorithm.vector_size_b = 1;
    key.algorithm.vector_size_c = 8;
    return key;
}

// Padding must not hide the global vector widths: native fp16 rcr needs K % 8,
// the _vec1_1_8 kernel only needs N % 8.
template <template <int, int, int> class Instance>
void expect_vector_gate()
{
    const Instance<8, 8, 8> native(make_test_key(128, 128, 32, "gfx950"), "native");
    const Instance<1, 1, 8> vec(make_vec_key(), "vec");

    EXPECT_TRUE(native.supports(Problem(512, 512, 512)));
    EXPECT_FALSE(native.supports(Problem(512, 512, 257)));
    EXPECT_TRUE(vec.supports(Problem(512, 512, 257)));
    EXPECT_FALSE(vec.supports(Problem(512, 129, 257)));

    // Native keys keep zero widths and no suffix, even when the pipeline's
    // tile distribution loads fewer than 16 bytes or the epilogue stores scalar C.
    const auto key = make_test_key(128, 128, 32, "gfx950");
    const Instance<4, 4, 8> small_tile(key, "native_small_tile");
    EXPECT_TRUE(small_tile.supports(Problem(256, 256, 516)));
    EXPECT_FALSE(small_tile.supports(Problem(256, 256, 514)));
    EXPECT_EQ(small_tile.get_key().encode_identifier().find("_vec"), std::string::npos);

    const Instance<8, 8, 1> default_epilogue(key, "native_default");
    EXPECT_TRUE(default_epilogue.supports(Problem(256, 257, 512)));
    EXPECT_FALSE(default_epilogue.supports(Problem(256, 257, 516)));
}

using StubEpilogueProblem = ck_tile::CShuffleEpilogueProblem<ck_tile::bf16_t,
                                                             ck_tile::bf16_t,
                                                             ck_tile::tuple<>,
                                                             float,
                                                             ck_tile::bf16_t,
                                                             ck_tile::tuple<>,
                                                             ck_tile::tensor_layout::gemm::RowMajor,
                                                             ck_tile::element_wise::PassThrough,
                                                             256,
                                                             256,
                                                             4,
                                                             4,
                                                             16,
                                                             16,
                                                             32,
                                                             false>;

template <int TileM_,
          int TileN_,
          bool PadM_,
          bool PadN_,
          int VecC_,
          typename Epilogue_ = ck_tile::CShuffleEpilogue<StubEpilogueProblem>>
struct TdmStubKernel
{
    using ADataType             = ck_tile::bf16_t;
    using BDataType             = ck_tile::bf16_t;
    using CDataType             = ck_tile::bf16_t;
    using AccDataType           = float;
    static constexpr bool kPadM = PadM_;
    static constexpr bool kPadN = PadN_;
    static constexpr bool kPadK = false;
    static constexpr int TileM  = TileM_;
    static constexpr int TileN  = TileN_;
    static constexpr int TileK  = 64;
    struct GemmPipeline
    {
        static constexpr bool skipCheckValidLaunchParams = true;
        template <bool = false>
        static constexpr int GetVectorSizeA()
        {
            return 1;
        }
        template <bool = false>
        static constexpr int GetVectorSizeB()
        {
            return 1;
        }
    };
    using GemmEpilogue               = Epilogue_;
    static constexpr int VectorSizeC = VecC_;
    static float launch(const ck_tile::GemmHostArgs&, const ck_tile::stream_config&) { return 0.f; }
};

template <typename Kernel>
using TdmTileInstance = backends::
    GeneratedTileKernelInstance<Kernel, ck_tile::bf16_t, ck_tile::bf16_t, ck_tile::bf16_t, float>;

template <typename Kernel>
using TdmKernelInstance = backends::GeneratedKernelInstance<Kernel>;

struct StrideCaptureKernel
{
    using ADataType             = ck_tile::bf16_t;
    using BDataType             = ck_tile::bf16_t;
    using CDataType             = ck_tile::bf16_t;
    using AccDataType           = float;
    static constexpr bool kPadM = false;
    static constexpr bool kPadN = false;
    static constexpr bool kPadK = false;
    static constexpr int TileM  = 256;
    static constexpr int TileN  = 256;
    static constexpr int TileK  = 64;
    struct GemmPipeline
    {
        static constexpr bool skipCheckValidLaunchParams = true;
        template <bool = false>
        static constexpr int GetVectorSizeA()
        {
            return 1;
        }
        template <bool = false>
        static constexpr int GetVectorSizeB()
        {
            return 1;
        }
    };
    using GemmEpilogue               = ck_tile::CShuffleEpilogue<StubEpilogueProblem>;
    static constexpr int VectorSizeC = 8;

    static inline ck_tile::GemmHostArgs last_args{};
    static float launch(const ck_tile::GemmHostArgs& args, const ck_tile::stream_config&)
    {
        last_args = args;
        return 0.f;
    }
};

KernelKey make_tdm_key(std::uint16_t tile_m,
                       std::uint16_t tile_n,
                       Pipeline pipeline,
                       Epilogue epilogue,
                       LayoutTag layout_c = LayoutTag::RowMajor,
                       bool pad_m         = false,
                       bool pad_n         = false)
{
    KernelKey key                 = make_test_key(tile_m, tile_n, 64, "gfx1250");
    key.signature.dtype_a         = DataType::BF16;
    key.signature.dtype_b         = DataType::BF16;
    key.signature.dtype_c         = DataType::BF16;
    key.signature.layout_c        = layout_c;
    key.algorithm.wave_shape      = {4, 4, 1};
    key.algorithm.warp_tile_shape = {16, 16, 32};
    key.algorithm.pipeline        = pipeline;
    key.algorithm.epilogue        = epilogue;
    key.algorithm.pad_m           = pad_m;
    key.algorithm.pad_n           = pad_n;
    key.algorithm.pad_k           = false;
    key.algorithm.block_size      = 512;
    return key;
}

} // anonymous namespace

TEST(GeneratedBackendTest, GeneratedKernelInstanceGatesVectorWidths)
{
    expect_vector_gate<LegacyInstance>();
}

TEST(GeneratedBackendTest, GeneratedTileKernelInstanceGatesVectorWidths)
{
    expect_vector_gate<TileInstance>();
}

// The register_all_kernels.hpp wrappers use GeneratedKernelInstance; first-fit
// must skip the native kernel and pick the narrower one, or return none.
TEST(GeneratedBackendTest, FirstFitFallsThroughToNarrowerKernel)
{
    Registry registry;
    auto native =
        std::make_shared<LegacyInstance<8, 8, 8>>(make_test_key(128, 128, 32, "gfx950"), "native");
    auto vec = std::make_shared<LegacyInstance<1, 1, 8>>(make_vec_key(), "vec");
    ASSERT_TRUE(registry.register_kernel(native));
    ASSERT_TRUE(registry.register_kernel(vec));

    Dispatcher dispatcher(&registry);
    EXPECT_EQ(dispatcher.select_kernel(Problem(512, 512, 257)), vec);
    EXPECT_EQ(dispatcher.select_kernel(Problem(512, 129, 257)), nullptr);
}

template <template <typename> class Instance>
void expect_tdm_cshuffle_and_split_k_gate()
{
    using Unpadded256    = TdmStubKernel<256, 256, false, false, 8>;
    using Unpadded128    = TdmStubKernel<256, 128, false, false, 8>;
    using PaddedCShuffle = TdmStubKernel<256, 256, false, true, 8>;
    using TdmEpi256 =
        TdmStubKernel<256, 256, false, false, 1, ck_tile::TdmEpilogue<StubEpilogueProblem>>;

    auto v3_256 = std::make_shared<Instance<Unpadded256>>(
        make_tdm_key(256, 256, Pipeline::CompTDMV3, Epilogue::CShuffle), "v3_256");
    auto v3_128 = std::make_shared<Instance<Unpadded128>>(
        make_tdm_key(256, 128, Pipeline::CompTDMV3, Epilogue::CShuffle), "v3_128");
    auto pc_cshuffle = std::make_shared<Instance<Unpadded256>>(
        make_tdm_key(256, 256, Pipeline::CompTDMPC, Epilogue::CShuffle), "pc_cshuffle");
    auto pc_tdm_epi = std::make_shared<Instance<TdmEpi256>>(
        make_tdm_key(256, 256, Pipeline::CompTDMPC, Epilogue::Tdm), "pc_tdm_epi");
    auto tdm_epi = std::make_shared<Instance<TdmEpi256>>(
        make_tdm_key(256, 256, Pipeline::CompTDMV1, Epilogue::Tdm), "tdm_epi");

    // Clipped A/B loads accept ragged M and K, while unpadded CShuffle rejects
    // vector-aligned N=384 (not a multiple of TileN=256).
    EXPECT_TRUE(v3_256->supports(Problem(301, 512, 257)));
    EXPECT_FALSE(v3_256->supports(Problem(512, 384, 512)));

    // CompTDMV3 launcher guards k_batch == 1, whereas plain CompTDMPC + CShuffle
    // supports split-K when K divides k_batch * TileK (TileK = 64) and C divides TileN.
    Problem split_k_aligned(301, 512, 256);
    split_k_aligned.k_batch = 2;
    EXPECT_FALSE(v3_256->supports(split_k_aligned));
    EXPECT_FALSE(pc_tdm_epi->supports(split_k_aligned));
    EXPECT_TRUE(pc_cshuffle->supports(split_k_aligned));

    Problem split_k_partial_tile(301, 512, 192);
    split_k_partial_tile.k_batch = 2; // 192 / 2 = 96, not a multiple of TileK=64
    EXPECT_FALSE(pc_cshuffle->supports(split_k_partial_tile));

    Problem split_k_ragged_n(301, 384, 256);
    split_k_ragged_n.k_batch = 2;
    EXPECT_FALSE(pc_cshuffle->supports(split_k_ragged_n));

    // Padded CShuffle accepts partial C tiles that divide VectorSizeC, and
    // ColMajor C gates M against TileM/kPadM instead of N.
    const Instance<PaddedCShuffle> padded(
        make_tdm_key(
            256, 256, Pipeline::CompTDMV3, Epilogue::CShuffle, LayoutTag::RowMajor, false, true),
        "padded_v3");
    EXPECT_TRUE(padded.supports(Problem(301, 384, 257)));
    EXPECT_FALSE(padded.supports(Problem(301, 386, 257)));

    const Instance<Unpadded256> col_c(
        make_tdm_key(256, 256, Pipeline::CompTDMV3, Epilogue::CShuffle, LayoutTag::ColMajor),
        "col_c");
    EXPECT_TRUE(col_c.supports(Problem(512, 301, 257)));
    EXPECT_FALSE(col_c.supports(Problem(384, 512, 512)));

    // TdmEpilogue clips C in hardware, so arbitrary N stays supported.
    EXPECT_TRUE(tdm_epi->supports(Problem(301, 384, 257)));
    EXPECT_TRUE(tdm_epi->supports(Problem(301, 385, 257)));

    // Registry first-fit must skip the 256-wide CShuffle kernel on N=384 and
    // select a compatible fallback instead of throwing at launch.
    Registry registry;
    ASSERT_TRUE(registry.register_kernel(v3_256));
    ASSERT_TRUE(registry.register_kernel(v3_128));

    Dispatcher dispatcher(&registry);
    EXPECT_EQ(dispatcher.select_kernel(Problem(512, 384, 512)), v3_128);
    EXPECT_EQ(dispatcher.select_kernel(Problem(512, 385, 512)), nullptr);

    ASSERT_TRUE(registry.register_kernel(tdm_epi));
    dispatcher.set_heuristic([&](const Problem&) -> std::vector<std::string> {
        return {v3_256->get_key().encode_identifier(),
                v3_128->get_key().encode_identifier(),
                tdm_epi->get_key().encode_identifier()};
    });
    EXPECT_EQ(dispatcher.select_kernel(Problem(301, 512, 257)), v3_256);
    EXPECT_EQ(dispatcher.select_kernel(Problem(512, 384, 512)), v3_128);
    EXPECT_EQ(dispatcher.select_kernel(Problem(301, 385, 257)), tdm_epi);
}

TEST(GeneratedBackendTest, TdmCShuffleEnforcesCTileWhilePreservingClippedAB)
{
    expect_tdm_cshuffle_and_split_k_gate<TdmTileInstance>();
    expect_tdm_cshuffle_and_split_k_gate<TdmKernelInstance>();
}

TEST(GeneratedBackendTest, GeneratedKernelInstanceDerivesStridesFromLayout)
{
    const Problem prob(300, 512, 400);
    auto check_strides = [&](LayoutTag la,
                             LayoutTag lb,
                             LayoutTag lc,
                             ck_tile::index_t want_a,
                             ck_tile::index_t want_b,
                             ck_tile::index_t want_c) {
        KernelKey key = make_tdm_key(256, 256, Pipeline::CompTDMV3, Epilogue::CShuffle, lc);
        key.signature.layout_a = la;
        key.signature.layout_b = lb;
        TdmKernelInstance<StrideCaptureKernel> inst(key, "stride_capture");
        StrideCaptureKernel::last_args = ck_tile::GemmHostArgs{};
        inst.run(nullptr, nullptr, nullptr, nullptr, prob, nullptr);
        EXPECT_EQ(StrideCaptureKernel::last_args.stride_A, want_a);
        EXPECT_EQ(StrideCaptureKernel::last_args.stride_B, want_b);
        EXPECT_EQ(StrideCaptureKernel::last_args.stride_E, want_c);
    };

    check_strides(LayoutTag::RowMajor, LayoutTag::ColMajor, LayoutTag::RowMajor, 400, 400, 512);
    check_strides(LayoutTag::RowMajor, LayoutTag::RowMajor, LayoutTag::RowMajor, 400, 512, 512);
    check_strides(LayoutTag::ColMajor, LayoutTag::RowMajor, LayoutTag::RowMajor, 300, 512, 512);
    check_strides(LayoutTag::ColMajor, LayoutTag::ColMajor, LayoutTag::RowMajor, 300, 400, 512);
    check_strides(LayoutTag::ColMajor, LayoutTag::ColMajor, LayoutTag::ColMajor, 300, 400, 300);
}
