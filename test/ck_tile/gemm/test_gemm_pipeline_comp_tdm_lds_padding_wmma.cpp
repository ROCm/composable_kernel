// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// comp_tdm / comp_tdm_v2 tiles where the TDM and the LDS tensor descriptors would place the base
// policy's LDS padding differently: LDS rows that are not a power of two, a pad interval that does
// not fit the TDM descriptor's 3-bit field, and TDM loads that start inside a pad group and run
// past its end. The policy pads those at an interval that divides the LDS row instead; the controls
// keep their padding. Padding that would take the LDS buffers past LDS is dropped.
#include "test_gemm_pipeline_kernel_types.hpp"
#include "test_gemm_pipeline_wmma_base.hpp"
#include "gtest/gtest.h"

#include <array>
#include <string>

using I48   = ck_tile::number<48>;
using I96   = ck_tile::number<96>;
using I160  = ck_tile::number<160>;
using I384  = ck_tile::number<384>;
using I512  = ck_tile::number<512>;
using I1024 = ck_tile::number<1024>;

using Waves2x2 = ck_tile::sequence<2, 2, 1>;
using Waves1x8 = ck_tile::sequence<1, 8, 1>;
using Waves8x1 = ck_tile::sequence<8, 1, 1>;

// Expected LDS padding of A and B under the TDM policy, as ck_tile::sequence<A, B>: kKeep keeps
// the base policy's padding, kDrop drops it, and n >= 0 pads with the base pad amount after every
// 2^(n + 1) dwords (pad_interval n). The base policy pads both operands in every instance below.
constexpr ck_tile::index_t kKeep = -1;
constexpr ck_tile::index_t kDrop = -2;
template <ck_tile::index_t A, ck_tile::index_t B>
using LdsPad   = ck_tile::sequence<A, B>;
using LdsPadAB = LdsPad<kKeep, kKeep>;

// clang-format off
using KernelTypesCompTDMLdsPadding = ::testing::Types<
    //         ALayout, BLayout, CLayout, ADataType, BDataType, AccDataType, CDataType, M_BlockSize, N_BlockSize, K_BlockSize, M_TileSize, N_TileSize, Scheduler, PipelineType,    Persistent,  ClusterLaunch, BlockWarps,     LdsPad
    // M/N-contiguous (transposed) operand whose LDS row is not a power of two
    std::tuple<    Col,     Row,     Row,      BF16,      BF16,         F32,      BF16,         I96,         I64,         I64,        I16,        I16, Intrawave,    CompTDMV1, NonPersistent, ClusterDisable,   Waves2x2,    LdsPad<3, kKeep>>,
    std::tuple<    Col,     Col,     Row,      BF16,      BF16,         F32,      BF16,         I96,         I64,         I64,        I16,        I16, Intrawave,    CompTDMV1, NonPersistent, ClusterDisable,   Waves2x2,    LdsPad<3, kKeep>>,
    std::tuple<    Row,     Row,     Row,      BF16,      BF16,         F32,      BF16,         I64,         I96,         I64,        I16,        I16, Intrawave,    CompTDMV1, NonPersistent, ClusterDisable,   Waves2x2,    LdsPad<kKeep, 3>>,
    std::tuple<    Col,     Row,     Row,      BF16,      BF16,         F32,      BF16,        I192,         I64,         I64,        I16,        I16, Intrawave,    CompTDMV2, NonPersistent, ClusterDisable,   Waves2x2,    LdsPad<4, kKeep>>,
    std::tuple<    Col,     Row,     Row,        F8,        F8,         F32,       F16,         I96,         I64,        I128,        I16,        I16, Intrawave,    CompTDMV1, NonPersistent, ClusterDisable,   Waves2x2,    LdsPad<2, kKeep>>,
    // K-contiguous operands whose LDS row (tile_k) is not a power of two
    std::tuple<    Row,     Col,     Row,      BF16,      BF16,         F32,      BF16,         I64,         I64,        I192,        I16,        I16, Intrawave,    CompTDMV1, NonPersistent, ClusterDisable,   Waves2x2, LdsPad<4, 4>>,
    std::tuple<    Row,     Col,     Row,      BF16,      BF16,         F32,      BF16,         I64,         I64,         I96,        I16,        I16, Intrawave,    CompTDMV2, NonPersistent, ClusterDisable,   Waves2x2, LdsPad<3, 3>>,
    // K-contiguous tile_k 192 with two block-gemm K sub-tiles of 96 elements: the pad interval
    // divides the 48-dword sub-tile (16 dwords), not only the 96-dword row (32 dwords)
    std::tuple<    Row,     Col,     Row,       F16,       F16,         F32,       F16,        I128,        I128,        I192,        I16,        I16, Intrawave,    CompTDMV1, NonPersistent, ClusterDisable,   Waves2x2, LdsPad<3, 3>>,
    std::tuple<    Row,     Col,     Row,       F16,       F16,         F32,       F16,        I128,        I128,        I192,        I16,        I16, Intrawave,    CompTDMV2, NonPersistent, ClusterDisable,   Waves2x2, LdsPad<3, 3>>,
    std::tuple<    Row,     Col,     Row,      BF16,      BF16,         F32,      BF16,        I128,        I128,        I192,        I16,        I16, Intrawave,    CompTDMV1, NonPersistent, ClusterDisable,   Waves2x2, LdsPad<3, 3>>,
    std::tuple<    Row,     Col,     Row,      BF16,      BF16,         F32,      BF16,        I128,        I128,        I192,        I16,        I16, Intrawave,    CompTDMV2, NonPersistent, ClusterDisable,   Waves2x2, LdsPad<3, 3>>,
    // K-contiguous tile_k 96, one sub-tile, 16-dword interval
    std::tuple<    Row,     Col,     Row,       F16,       F16,         F32,       F16,        I128,        I128,         I96,        I16,        I16, Intrawave,    CompTDMV1, NonPersistent, ClusterDisable,   Waves2x2, LdsPad<3, 3>>,
    std::tuple<    Row,     Col,     Row,       F16,       F16,         F32,       F16,        I128,        I128,         I96,        I16,        I16, Intrawave,    CompTDMV2, NonPersistent, ClusterDisable,   Waves2x2, LdsPad<3, 3>>,
    // 8-bit K-contiguous rows that are not a power of two (warp K 64): 48-dword rows of tile_k 192
    // (one sub-tile), 96-dword rows of tile_k 384 split into two 48-dword sub-tiles
    std::tuple<    Row,     Col,     Row,        F8,        F8,         F32,       F16,        I128,        I128,        I192,        I16,        I16, Intrawave,    CompTDMV1, NonPersistent, ClusterDisable,   Waves2x2, LdsPad<3, 3>>,
    std::tuple<    Row,     Col,     Row,        F8,        F8,         F32,       F16,        I128,        I128,        I192,        I16,        I16, Intrawave,    CompTDMV2, NonPersistent, ClusterDisable,   Waves2x2, LdsPad<3, 3>>,
    std::tuple<    Row,     Col,     Row,        F8,        F8,         F32,       F16,        I128,        I128,        I384,        I16,        I16, Intrawave,    CompTDMV1, NonPersistent, ClusterDisable,   Waves2x2, LdsPad<3, 3>>,
    std::tuple<    Row,     Col,     Row,        F8,        F8,         F32,       F16,        I128,        I128,        I384,        I16,        I16, Intrawave,    CompTDMV2, NonPersistent, ClusterDisable,   Waves2x2, LdsPad<3, 3>>,
    std::tuple<    Row,     Col,     Row,       BF8,       BF8,         F32,       F16,        I128,        I128,        I192,        I16,        I16, Intrawave,    CompTDMV1, NonPersistent, ClusterDisable,   Waves2x2, LdsPad<3, 3>>,
    std::tuple<    Row,     Col,     Row,       BF8,       BF8,         F32,       F16,        I128,        I128,        I192,        I16,        I16, Intrawave,    CompTDMV2, NonPersistent, ClusterDisable,   Waves2x2, LdsPad<3, 3>>,
    std::tuple<    Row,     Col,     Row,      INT8,      INT8,       INT32,     INT32,        I128,        I128,        I192,        I16,        I16, Intrawave,    CompTDMV1, NonPersistent, ClusterDisable,   Waves2x2, LdsPad<3, 3>>,
    std::tuple<    Row,     Col,     Row,      INT8,      INT8,       INT32,     INT32,        I128,        I128,        I192,        I16,        I16, Intrawave,    CompTDMV2, NonPersistent, ClusterDisable,   Waves2x2, LdsPad<3, 3>>,
    // fp32 tile_k 96 (warp K 4): 96-dword rows split into two 48-dword sub-tiles
    std::tuple<    Row,     Col,     Row,       F32,       F32,         F32,       F32,        I128,        I128,         I96,        I16,        I16, Intrawave,    CompTDMV1, NonPersistent, ClusterDisable,   Waves2x2, LdsPad<3, 3>>,
    std::tuple<    Row,     Col,     Row,       F32,       F32,         F32,       F32,        I128,        I128,         I96,        I16,        I16, Intrawave,    CompTDMV2, NonPersistent, ClusterDisable,   Waves2x2, LdsPad<3, 3>>,
    // packed fp4 tile_k 384 (gfx125 has fp4 WMMA only for 32x16 / 32x32, warp K 128): 48-dword
    // rows; fp8 x fp4 pads A (96-dword rows) and B (48-dword rows) at different intervals
    std::tuple<    Row,     Col,     Row,        F4,        F4,         F32,       F16,        I128,        I128,        I384,        I32,        I32, Intrawave,    CompTDMV1, NonPersistent, ClusterDisable,   Waves2x2, LdsPad<3, 3>>,
    std::tuple<    Row,     Col,     Row,        F4,        F4,         F32,       F16,        I128,        I128,        I384,        I32,        I32, Intrawave,    CompTDMV2, NonPersistent, ClusterDisable,   Waves2x2, LdsPad<3, 3>>,
    std::tuple<    Row,     Col,     Row,        F8,        F4,         F32,       F16,        I128,        I128,        I384,        I32,        I32, Intrawave,    CompTDMV1, NonPersistent, ClusterDisable,   Waves2x2, LdsPad<4, 3>>,
    std::tuple<    Row,     Col,     Row,        F8,        F4,         F32,       F16,        I128,        I128,        I384,        I32,        I32, Intrawave,    CompTDMV2, NonPersistent, ClusterDisable,   Waves2x2, LdsPad<4, 3>>,
    // K-contiguous tile_k 1024: the pad interval does not fit the TDM descriptor's 3-bit field
    std::tuple<    Row,     Col,     Row,      BF16,      BF16,         F32,      BF16,         I32,         I32,       I1024,        I16,        I16, Intrawave,    CompTDMV1, NonPersistent, ClusterDisable,   Waves2x2, LdsPad<7, 7>>,
    std::tuple<    Row,     Col,     Row,      BF16,      BF16,         F32,      BF16,         I32,         I32,       I1024,        I16,        I16, Intrawave,    CompTDMV2, NonPersistent, ClusterDisable,   Waves2x2, LdsPad<7, 7>>,
    // 8 loading waves: every other 6-row TDM load starts inside a 4-row pad group and runs past
    // its end (A, then B)
    std::tuple<    Row,     Col,     Row,      BF16,      BF16,         F32,      BF16,         I48,        I128,         I32,        I16,        I16, Intrawave,    CompTDMV1, NonPersistent, ClusterDisable,   Waves1x8,    LdsPad<3, kKeep>>,
    std::tuple<    Row,     Col,     Row,      BF16,      BF16,         F32,      BF16,        I128,         I48,         I32,        I16,        I16, Intrawave,    CompTDMV1, NonPersistent, ClusterDisable,   Waves8x1,    LdsPad<kKeep, 3>>,
    // controls: the TDM and the LDS tensor descriptors agree, padding kept (transposed tile_k 1024
    // has 32-element rows, so its pad interval fits the field)
    std::tuple<    Row,     Col,     Row,      BF16,      BF16,         F32,      BF16,         I96,         I96,         I64,        I16,        I16, Intrawave,    CompTDMV1, NonPersistent, ClusterDisable,   Waves2x2,   LdsPadAB>,
    std::tuple<    Col,     Row,     Row,      BF16,      BF16,         F32,      BF16,        I128,         I64,         I64,        I16,        I16, Intrawave,    CompTDMV1, NonPersistent, ClusterDisable,   Waves2x2,   LdsPadAB>,
    std::tuple<    Col,     Row,     Row,      BF16,      BF16,         F32,      BF16,         I32,         I32,       I1024,        I16,        I16, Intrawave,    CompTDMV1, NonPersistent, ClusterDisable,   Waves2x2,   LdsPadAB>,
    // comp_tdm_v2 (one wave loads the whole tile): K-contiguous 2-row pad groups, then transposed
    // pads after every 128-element row (A) and after every two 64-element rows (B)
    std::tuple<    Row,     Col,     Row,      BF16,      BF16,         F32,      BF16,         I64,         I64,         I64,        I16,        I16, Intrawave,    CompTDMV2, NonPersistent, ClusterDisable,   Waves2x2,   LdsPadAB>,
    std::tuple<    Col,     Row,     Row,      BF16,      BF16,         F32,      BF16,        I128,         I64,         I64,        I16,        I16, Intrawave,    CompTDMV2, NonPersistent, ClusterDisable,   Waves2x2,   LdsPadAB>,
    // 8 loading waves: 2-row TDM loads, every other one starting mid pad group, never run past a
    // pad group's end, so no load has a pad inside (A, then B)
    std::tuple<    Row,     Col,     Row,      BF16,      BF16,         F32,      BF16,         I16,        I128,         I32,        I16,        I16, Intrawave,    CompTDMV1, NonPersistent, ClusterDisable,   Waves1x8,   LdsPadAB>,
    std::tuple<    Row,     Col,     Row,      BF16,      BF16,         F32,      BF16,        I128,         I16,         I32,        I16,        I16, Intrawave,    CompTDMV1, NonPersistent, ClusterDisable,   Waves8x1,   LdsPadAB>,
    // LDS budget: padding both operands would take two buffers of A and B past the 320 KB of LDS,
    // so the operand whose padding takes more LDS drops it. Padding the transposed 160-element
    // rows of A every 16 dwords (A drops it), then the base policy's padding, consistent but
    // larger for the 512-row B (B drops it)
    std::tuple<    Col,     Col,     Row,      BF16,      BF16,         F32,      BF16,        I160,        I384,        I128,        I16,        I16, Intrawave,    CompTDMV1, NonPersistent, ClusterDisable,   Waves2x2, LdsPad<kDrop, kKeep>>,
    std::tuple<    Row,     Col,     Row,      BF16,      BF16,         F32,      BF16,         I96,        I512,        I128,        I16,        I16, Intrawave,    CompTDMV1, NonPersistent, ClusterDisable,   Waves2x2, LdsPad<kKeep, kDrop>>
>;
// clang-format on

// Whether pads within the K-contiguous LDS rows of an operand repeat at the same offsets in every
// block-gemm sub-tile: the gfx125 LDS reads (tile_window::load_with_offset) take a sub-tile's
// offsets as those of sub-tile 0 plus a constant, so the pad interval must divide the sub-tile.
template <typename Policy, typename Problem, bool IsA>
__device__ constexpr bool lds_pads_repeat_per_sub_tile()
{
    using DataType =
        std::conditional_t<IsA, typename Problem::ADataType, typename Problem::BDataType>;
    constexpr bool is_tr_load = IsA ? Policy::Base::template is_a_load_tr<Problem>
                                    : Policy::Base::template is_b_load_tr<Problem>;
    if constexpr(is_tr_load || !Policy::template IsKLdsPaddedPerChunk<Problem, IsA>())
    {
        return true;
    }
    else
    {
        constexpr auto config = Policy::template GetLdsPaddingConfig<Problem, IsA>();
        constexpr ck_tile::index_t interval_dwords = 1 << (config[ck_tile::number<2>{}] + 1);
        constexpr ck_tile::index_t sub_tile_dwords =
            Problem::BlockGemmShape::kK / Policy::template GetPipelineSubTileNum<Problem>().value *
            ck_tile::index_t{sizeof(DataType)} / ck_tile::numeric_traits<DataType>::PackedSize /
            ck_tile::index_t{sizeof(int32_t)};
        return sub_tile_dwords % interval_dwords == 0;
    }
}

// The base and the TDM LDS padding configs (padded, pad_amount, pad_interval) of A and B, in that
// order. Both depend on is_a_load_tr / is_b_load_tr and the LDS bank count, the TDM one also on
// the __gfx125__ guard; together these have their gfx1250 values only in the gfx125 device pass,
// so the configs are read back from a kernel.
template <typename Policy, typename Problem>
__global__ void tdm_lds_padding_probe(ck_tile::index_t* out)
{
#if defined(__gfx125__)
    using BasePolicy = typename Policy::Base;
    const auto store = [&](ck_tile::index_t first, auto config) {
        out[first + 0] = config[ck_tile::number<0>{}];
        out[first + 1] = config[ck_tile::number<1>{}];
        out[first + 2] = config[ck_tile::number<2>{}];
    };
    store(0, BasePolicy::template GetLdsPaddingConfig<Problem, true>());
    store(3, BasePolicy::template GetLdsPaddingConfig<Problem, false>());
    store(6, Policy::template GetLdsPaddingConfig<Problem, true>());
    store(9, Policy::template GetLdsPaddingConfig<Problem, false>());

    // the LDS budget's model of an operand's LDS buffer is what its LDS tensor descriptor
    // allocates
    using ALdsType = typename Problem::ADataType;
    using BLdsType = typename Problem::BDataType;
    static_assert(Policy::template GetTDMLdsTileBytes<Problem, true>(
                      Policy::template GetLdsPaddingConfig<Problem, true>()) ==
                  Policy::template MakeALdsBlockDescriptor<Problem>().get_element_space_size() *
                      sizeof(ALdsType) / ck_tile::numeric_traits<ALdsType>::PackedSize);
    static_assert(Policy::template GetTDMLdsTileBytes<Problem, false>(
                      Policy::template GetLdsPaddingConfig<Problem, false>()) ==
                  Policy::template MakeBLdsBlockDescriptor<Problem>().get_element_space_size() *
                      sizeof(BLdsType) / ck_tile::numeric_traits<BLdsType>::PackedSize);
    static_assert(lds_pads_repeat_per_sub_tile<Policy, Problem, true>(),
                  "A: LDS pad interval does not divide the block-gemm K sub-tile");
    static_assert(lds_pads_repeat_per_sub_tile<Policy, Problem, false>(),
                  "B: LDS pad interval does not divide the block-gemm K sub-tile");
#else
    static_cast<void>(out);
#endif
}

// The Policy a GEMM pipeline type was instantiated with (its second template argument).
template <typename Pipeline>
struct pipeline_policy;

template <template <typename, typename> class PipelineTemplate, typename Problem, typename Policy>
struct pipeline_policy<PipelineTemplate<Problem, Policy>>
{
    using type = Policy;
};

template <typename T>
class TestCkTileGemmPipelineCompTDMLdsPadding
    : public TestCkTileGemmPipelineWmmaBase<T, TestCkTileGemmPipelineCompTDMLdsPadding<T>>
{
    using Base = TestCkTileGemmPipelineWmmaBase<T, TestCkTileGemmPipelineCompTDMLdsPadding<T>>;

    protected:
    // These instances use the regular (non-cluster) launch, which also works on
    // asicRevision=0, so the base-class TDM revision skip does not apply here.
    // That also bypasses the base-class data-type skip, and Run() does nothing for an
    // unsupported combination, so every instance here must be supported. The base-class
    // gfx1250 skip is repeated here because this override replaces it.
    void SetUp() override
    {
        static_assert(TestCkTileGemmPipelineCompTDMLdsPadding<T>::check_data_type(),
                      "unsupported data type combination: this case would pass without running");
        CK_TILE_SKIP_UNLESS_GFX125();
        // Dedicated WMMA F4 instructions are not supported on gfx1250 A0. Only the revision is
        // checked: this test is built for gfx125 only, and is_gfx125_supported() is false on
        // devices named gfx1250-strict.
        if constexpr(std::is_same_v<typename Base::ADataType, F4> &&
                     std::is_same_v<typename Base::BDataType, F4>)
        {
            if(ck_tile::get_device_revision() == 0)
            {
                GTEST_SKIP() << "Unsupported data type combination for gemm pipeline test on "
                                "gfx1250 asicRevision=0.";
            }
        }
        this->k_batches_ = {1};
    }
};

TYPED_TEST_SUITE(TestCkTileGemmPipelineCompTDMLdsPadding, KernelTypesCompTDMLdsPadding);

// 4 x 4 output tiles and eight K iterations, so that both LDS buffers are filled more than once.
TYPED_TEST(TestCkTileGemmPipelineCompTDMLdsPadding, Regular)
{
    this->Run(4 * TestFixture::M_Tile, 4 * TestFixture::N_Tile, 8 * TestFixture::K_Tile);
}

// One and two K tiles (no hot loop; tails of one and two) and seven (hot loop, tail of one), with
// M and N edge tiles that are only half inside the matrix.
TYPED_TEST(TestCkTileGemmPipelineCompTDMLdsPadding, TailsAndEdgeTiles)
{
    const int M = 3 * TestFixture::M_Tile + TestFixture::M_Tile / 2;
    const int N = 3 * TestFixture::N_Tile + TestFixture::N_Tile / 2;
    for(const int k_tiles : {1, 2, 7})
    {
        SCOPED_TRACE("K tiles: " + std::to_string(k_tiles));
        this->Run(M, N, k_tiles * TestFixture::K_Tile);
    }
}

// How the TDM policy pads each operand's LDS. The GEMM results cannot show this: an operand that
// lost its padding would still be correct, only slower.
TYPED_TEST(TestCkTileGemmPipelineCompTDMLdsPadding, LdsPaddingDecision)
{
    using Shape = ck_tile::TileGemmShape<
        ck_tile::sequence<TestFixture::M_Tile, TestFixture::N_Tile, TestFixture::K_Tile>,
        typename TestFixture::BlockWarps,
        ck_tile::
            sequence<TestFixture::M_Warp_Tile, TestFixture::N_Warp_Tile, TestFixture::K_Warp_Tile>>;
    // A reduced Problem: GetLdsPaddingConfig (base and TDM override) reads only the A/B data
    // types and layouts, the block tile and the block size, built here as invoke_gemm builds
    // them. The other traits may differ from invoke_gemm's (TransposeC does); if
    // GetLdsPaddingConfig starts reading one of them, build it here as invoke_gemm does.
    using Traits  = ck_tile::TileGemmUniversalTraits<false, // kPadM
                                                     false, // kPadN
                                                     false, // kPadK
                                                     true,  // DoubleSmemBuffer
                                                     typename TestFixture::ALayout,
                                                     typename TestFixture::BLayout,
                                                     typename TestFixture::CLayout>;
    using Problem = ck_tile::UniversalGemmPipelineProblem<typename TestFixture::ADataType,
                                                          typename TestFixture::BDataType,
                                                          typename TestFixture::AccDataType,
                                                          Shape,
                                                          Traits,
                                                          TestFixture::Scheduler>;
    // the Policy of the pipeline invoke_gemm selects for this PipelineType
    using Policy = typename pipeline_policy<
        typename GemmPipelineTypeSelector<TestFixture::PipelineType, Problem>::pipeline>::type;
    using ExpectedLdsPad = std::tuple_element_t<17, TypeParam>;

    ck_tile::DeviceMem config_buf(12 * sizeof(ck_tile::index_t));
    config_buf.SetZero();
    tdm_lds_padding_probe<Policy, Problem>
        <<<1, 1>>>(static_cast<ck_tile::index_t*>(config_buf.GetDeviceBuffer()));
    HIP_CHECK_ERROR(hipGetLastError());
    std::array<ck_tile::index_t, 12> config{};
    config_buf.FromDevice(config.data());

    const std::array<ck_tile::index_t, 2> expected{ExpectedLdsPad::at(ck_tile::number<0>{}),
                                                   ExpectedLdsPad::at(ck_tile::number<1>{})};
    for(int op = 0; op < 2; ++op)
    {
        SCOPED_TRACE(op == 0 ? "operand A" : "operand B");
        const ck_tile::index_t* base = config.data() + 3 * op;
        const ck_tile::index_t* tdm  = config.data() + 6 + 3 * op;
        // every instance here is one the base policy pads (this also catches an empty probe)
        EXPECT_EQ(base[0], 1);
        if(expected[op] == kKeep)
        {
            EXPECT_EQ(tdm[0], base[0]);
            EXPECT_EQ(tdm[1], base[1]);
            EXPECT_EQ(tdm[2], base[2]);
        }
        else if(expected[op] == kDrop)
        {
            EXPECT_EQ(tdm[0], 0);
        }
        else
        {
            EXPECT_EQ(tdm[0], 1);
            EXPECT_EQ(tdm[1], base[1]);
            EXPECT_EQ(tdm[2], expected[op]);
        }
    }
}
