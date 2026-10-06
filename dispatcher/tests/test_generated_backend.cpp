// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

/// Unit tests for the supports() gate of the generated GEMM backends.
/// Note: Uses a stub kernel struct; no kernel is compiled or launched.

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
