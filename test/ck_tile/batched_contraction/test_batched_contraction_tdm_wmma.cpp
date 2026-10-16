// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// Batched contraction on the TDM (tensor data mover) GEMM pipelines of gfx125x.
// Covers TDM V1 / V2 x fp16 / bf16 with multi-dimensional G/M/N/K groups, ragged tails,
// small K, a non-packed A row pitch, length-1 G/M dims with arbitrary strides, padded G strides,
// and the host-side rejections (split-K, non-packed groups, misaligned pointers, wrong arch).
// The host-side rejection tests run on any device; the GPU tests skip unless on gfx1250.

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <tuple>
#include <type_traits>
#include <vector>

#include "gtest/gtest.h"

#include "ck_tile/core.hpp"
#include "ck_tile/host.hpp"
#include "ck_tile/host/reference/reference_batched_contraction.hpp"
#include "ck_tile/ops/batched_contraction.hpp"
#include "ck_tile/ops/epilogue.hpp"
#include "ck_tile/ops/epilogue/tdm_epilogue.hpp"
#include "ck_tile/ops/gemm.hpp"

namespace {

enum class TdmPipeline
{
    V1,
    V2
};

template <TdmPipeline P, typename Problem>
struct TdmPipelineSelector;

template <typename Problem>
struct TdmPipelineSelector<TdmPipeline::V1, Problem>
{
    using type = ck_tile::GemmPipelineAgBgCrCompTDMV1<Problem>;
};

template <typename Problem>
struct TdmPipelineSelector<TdmPipeline::V2, Problem>
{
    using type = ck_tile::GemmPipelineAgBgCrCompTDMV2<Problem>;
};

using Row = ck_tile::tensor_layout::gemm::RowMajor;
using Col = ck_tile::tensor_layout::gemm::ColumnMajor;

template <typename DataType,
          TdmPipeline P,
          ck_tile::index_t NumDimG_,
          ck_tile::index_t NumDimM_,
          ck_tile::index_t NumDimN_,
          ck_tile::index_t NumDimK_>
struct TdmContractionKernel
{
    using ADataType      = DataType;
    using BDataType      = DataType;
    using EDataType      = DataType;
    using AccDataType    = float;
    using DsDataType     = ck_tile::tuple<>;
    using DsLayout       = ck_tile::tuple<>;
    using CDEElementWise = ck_tile::element_wise::PassThrough;

    // 64x64x32 block tile, 2x2 waves (4 waves, required by TDM V2), 16x16x32 WMMA warp tile.
    static constexpr ck_tile::index_t M_Tile      = 64;
    static constexpr ck_tile::index_t N_Tile      = 64;
    static constexpr ck_tile::index_t K_Tile      = 32;
    static constexpr ck_tile::index_t M_Warp      = 2;
    static constexpr ck_tile::index_t N_Warp      = 2;
    static constexpr ck_tile::index_t K_Warp      = 1;
    static constexpr ck_tile::index_t M_Warp_Tile = 16;
    static constexpr ck_tile::index_t N_Warp_Tile = 16;
    static constexpr ck_tile::index_t K_Warp_Tile = 32;

    static constexpr bool DoubleSmemBuffer = true; // required by the TDM pipelines
#if defined(CK_USE_GFX1250)
    static constexpr bool TransposeC = (M_Warp_Tile == N_Warp_Tile);
#else
    static constexpr bool TransposeC = false;
#endif

    using GemmShape =
        ck_tile::TileGemmShape<ck_tile::sequence<M_Tile, N_Tile, K_Tile>,
                               ck_tile::sequence<M_Warp, N_Warp, K_Warp>,
                               ck_tile::sequence<M_Warp_Tile, N_Warp_Tile, K_Warp_Tile>>;
    using TilePartitioner = ck_tile::GemmSpatiallyLocalTilePartitioner<GemmShape, 8, 4>;

    using Traits = ck_tile::TileGemmUniversalTraits<false, // kPadM
                                                    false, // kPadN
                                                    false, // kPadK
                                                    DoubleSmemBuffer,
                                                    Row,
                                                    Col,
                                                    Row,
                                                    TransposeC>;

    using GemmProblem =
        ck_tile::UniversalGemmPipelineProblem<ADataType,
                                              BDataType,
                                              AccDataType,
                                              GemmShape,
                                              Traits,
                                              ck_tile::GemmPipelineScheduler::Intrawave>;

    using GemmPipeline = typename TdmPipelineSelector<P, GemmProblem>::type;

    using EpilogueProblem = ck_tile::CShuffleEpilogueProblem<ADataType,
                                                             BDataType,
                                                             DsDataType,
                                                             AccDataType,
                                                             EDataType,
                                                             DsLayout,
                                                             Row,
                                                             CDEElementWise,
                                                             TilePartitioner::MPerBlock,
                                                             TilePartitioner::NPerBlock,
                                                             M_Warp,
                                                             N_Warp,
                                                             M_Warp_Tile,
                                                             N_Warp_Tile,
                                                             K_Warp_Tile,
                                                             GemmProblem::TransposeC,
                                                             1,     // kNumWaveGroups
                                                             false, // FixedVectorSize
                                                             1,     // VectorSizeC
                                                             1,     // BlockedXDLN_PerWarp
                                                             DoubleSmemBuffer>;
    using Epilogue        = ck_tile::TdmEpilogue<EpilogueProblem>;

    using ContractionProblem = ck_tile::BatchedContractionProblem<ADataType,
                                                                  BDataType,
                                                                  DsDataType,
                                                                  EDataType,
                                                                  NumDimG_,
                                                                  NumDimM_,
                                                                  NumDimN_,
                                                                  NumDimK_,
                                                                  0>;

    using Kernel = ck_tile::
        BatchedContractionKernel<ContractionProblem, TilePartitioner, GemmPipeline, Epilogue>;

    static_assert(Kernel::kIsTdmPipeline, "TDM trait must be detected");
};

// Problem description: dims are split per group, strides are optional overrides.
struct ContractionCase
{
    std::vector<ck_tile::index_t> G, M, N, K;
    // When non-empty, full [G.., M.., K..] stride vector for A. Otherwise packed.
    std::vector<ck_tile::index_t> A_strides{};
    ck_tile::index_t k_batch = 1;
    // When non-empty, full [G.., N.., K..] / [G.., M.., N..] stride vectors for B / E.
    std::vector<ck_tile::index_t> B_strides{};
    std::vector<ck_tile::index_t> E_strides{};
};

std::vector<ck_tile::index_t> concat(std::initializer_list<std::vector<ck_tile::index_t>> parts)
{
    std::vector<ck_tile::index_t> out;
    for(const auto& p : parts)
        out.insert(out.end(), p.begin(), p.end());
    return out;
}

std::vector<ck_tile::index_t> packed_strides(const std::vector<ck_tile::index_t>& dims)
{
    std::vector<ck_tile::index_t> s(dims.size());
    ck_tile::index_t acc = 1;
    for(int i = static_cast<int>(dims.size()) - 1; i >= 0; --i)
    {
        s[i] = acc;
        acc *= dims[i];
    }
    return s;
}

ck_tile::index_t product(const std::vector<ck_tile::index_t>& v)
{
    ck_tile::index_t p = 1;
    for(auto x : v)
        p *= x;
    return p;
}

std::vector<std::size_t> to_size_t(const std::vector<ck_tile::index_t>& v)
{
    return std::vector<std::size_t>(v.begin(), v.end());
}

// Packed strides of dims, with the stride at index idx replaced by value.
std::vector<ck_tile::index_t> packed_strides_with(const std::vector<ck_tile::index_t>& dims,
                                                  std::size_t idx,
                                                  ck_tile::index_t value)
{
    auto s = packed_strides(dims);
    s[idx] = value;
    return s;
}

// Exact bit-pattern match against the poison sentinel. The intent is "was this element left
// untouched", not a numeric comparison, so compare the raw storage instead of float values.
template <typename T>
bool same_bits(const T& x, const T& y)
{
    return std::memcmp(&x, &y, sizeof(T)) == 0;
}

bool device_is_gfx1250() { return ck_tile::is_gfx125_supported(); }

// Full dims / strides of A, B and E for a case (packed unless the case overrides them).
struct ContractionLayout
{
    std::vector<ck_tile::index_t> A_dims, B_dims, E_dims;
    std::vector<ck_tile::index_t> A_strides, B_strides, E_strides;
};

ContractionLayout make_layout(const ContractionCase& c)
{
    ContractionLayout l;
    l.A_dims    = concat({c.G, c.M, c.K});
    l.B_dims    = concat({c.G, c.N, c.K});
    l.E_dims    = concat({c.G, c.M, c.N});
    l.A_strides = c.A_strides.empty() ? packed_strides(l.A_dims) : c.A_strides;
    l.B_strides = c.B_strides.empty() ? packed_strides(l.B_dims) : c.B_strides;
    l.E_strides = c.E_strides.empty() ? packed_strides(l.E_dims) : c.E_strides;
    return l;
}

// BatchedContractionHostArgs copies the dim / stride vectors, so the layout may be a temporary.
template <typename DataType,
          TdmPipeline P,
          ck_tile::index_t NG,
          ck_tile::index_t NM,
          ck_tile::index_t NN,
          ck_tile::index_t NK>
ck_tile::BatchedContractionHostArgs<0>
make_host_args(const ContractionCase& c, const void* a, const void* b, void* e)
{
    const ContractionLayout l = make_layout(c);
    return ck_tile::BatchedContractionHostArgs<0>(a,
                                                  b,
                                                  {},
                                                  e,
                                                  c.k_batch,
                                                  l.A_dims,
                                                  l.B_dims,
                                                  {},
                                                  l.E_dims,
                                                  l.A_strides,
                                                  l.B_strides,
                                                  {},
                                                  l.E_strides);
}

template <typename DataType,
          TdmPipeline P,
          ck_tile::index_t NG,
          ck_tile::index_t NM,
          ck_tile::index_t NN,
          ck_tile::index_t NK>
void run_case(const ContractionCase& c)
{
    using Cfg    = TdmContractionKernel<DataType, P, NG, NM, NN, NK>;
    using Kernel = typename Cfg::Kernel;

    if(!device_is_gfx1250())
    {
        GTEST_SKIP() << "TDM batched contraction requires a gfx1250 device";
    }

    ASSERT_EQ(c.G.size(), static_cast<std::size_t>(NG));
    ASSERT_EQ(c.M.size(), static_cast<std::size_t>(NM));
    ASSERT_EQ(c.N.size(), static_cast<std::size_t>(NN));
    ASSERT_EQ(c.K.size(), static_cast<std::size_t>(NK));

    const ContractionLayout layout = make_layout(c);
    const auto& A_dims             = layout.A_dims;
    const auto& B_dims             = layout.B_dims;
    const auto& E_dims             = layout.E_dims;
    const auto& A_strides          = layout.A_strides;
    const auto& B_strides          = layout.B_strides;
    const auto& E_strides          = layout.E_strides;

    ck_tile::HostTensor<DataType> a_host(
        ck_tile::HostTensorDescriptor(to_size_t(A_dims), to_size_t(A_strides)));
    ck_tile::HostTensor<DataType> b_host(
        ck_tile::HostTensorDescriptor(to_size_t(B_dims), to_size_t(B_strides)));
    ck_tile::HostTensor<DataType> e_host(
        ck_tile::HostTensorDescriptor(to_size_t(E_dims), to_size_t(E_strides)));
    ck_tile::HostTensor<DataType> e_ref(
        ck_tile::HostTensorDescriptor(to_size_t(E_dims), to_size_t(E_strides)));

    ck_tile::FillUniformDistribution<DataType>{-2.f, 2.f, 11939}(a_host);
    ck_tile::FillUniformDistribution<DataType>{-2.f, 2.f, 11940}(b_host);
    // The reference overwrites every in-bounds element of E. Elements in stride gaps (padded
    // E strides) must stay untouched by the kernel, so both sides start from the same poison.
    const auto poison = ck_tile::type_convert<DataType>(-777.f);
    ck_tile::FillConstant<DataType>{poison}(e_ref);

    ck_tile::DeviceMem a_dev(a_host.get_element_space_size_in_bytes());
    ck_tile::DeviceMem b_dev(b_host.get_element_space_size_in_bytes());
    ck_tile::DeviceMem e_dev(e_host.get_element_space_size_in_bytes());
    a_dev.ToDevice(a_host.data());
    b_dev.ToDevice(b_host.data());
    // Poison E so tiles the kernel fails to write are detected.
    ck_tile::FillConstant<DataType>{poison}(e_host);
    e_dev.ToDevice(e_host.data());

    const auto args = make_host_args<DataType, P, NG, NM, NN, NK>(
        c, a_dev.GetDeviceBuffer(), b_dev.GetDeviceBuffer(), e_dev.GetDeviceBuffer());

    const auto kargs = Kernel::MakeKernelArgs(args);
    ASSERT_TRUE(Kernel::IsSupportedArguments(kargs));

    const dim3 grids  = Kernel::GridSize(kargs);
    const dim3 blocks = Kernel::GetBlockSize();
    ck_tile::launch_kernel(ck_tile::stream_config{nullptr, false, 0},
                           ck_tile::make_kernel<1>(Kernel{}, grids, blocks, 0, kargs));
    e_dev.FromDevice(e_host.data());

    const ck_tile::index_t G_total = product(c.G);
    const ck_tile::index_t M_total = product(c.M);
    const ck_tile::index_t N_total = product(c.N);
    const ck_tile::index_t K_total = product(c.K);

    const std::array<ck_tile::HostTensor<DataType>, 0> ds_host{};
    ck_tile::compute_reference_batched_contraction<DataType,
                                                   DataType,
                                                   DataType,
                                                   DataType,
                                                   float,
                                                   ck_tile::element_wise::PassThrough,
                                                   0>(a_host,
                                                      b_host,
                                                      ds_host,
                                                      e_ref,
                                                      G_total,
                                                      M_total,
                                                      N_total,
                                                      K_total,
                                                      ck_tile::element_wise::PassThrough{},
                                                      c.G,
                                                      c.M,
                                                      c.N,
                                                      c.K);

    float max_abs = 0.f;
    for(const auto& v : e_ref.mData)
        if(!same_bits(v, poison))
            max_abs = std::max(max_abs, std::abs(ck_tile::type_convert<float>(v)));
    const auto rtol = ck_tile::get_relative_threshold<DataType, DataType, float>(K_total);
    const auto atol = ck_tile::get_absolute_threshold<DataType, DataType, float>(max_abs, K_total);

    EXPECT_TRUE(ck_tile::check_err(e_host, e_ref, "Error: incorrect results!", rtol, atol));
}

} // namespace

template <typename Tuple>
class TestCkTileBatchedContractionTdm : public ::testing::Test
{
    public:
    using DataType                    = std::tuple_element_t<0, Tuple>;
    static constexpr TdmPipeline Pipe = std::tuple_element_t<1, Tuple>::value;

    template <ck_tile::index_t NG, ck_tile::index_t NM, ck_tile::index_t NN, ck_tile::index_t NK>
    void Run(const ContractionCase& c)
    {
        run_case<DataType, Pipe, NG, NM, NN, NK>(c);
    }

    template <ck_tile::index_t NG, ck_tile::index_t NM, ck_tile::index_t NN, ck_tile::index_t NK>
    using Kernel = typename TdmContractionKernel<DataType, Pipe, NG, NM, NN, NK>::Kernel;
};

template <TdmPipeline P>
using PipeC = std::integral_constant<TdmPipeline, P>;

using KernelTypesTdm = ::testing::Types<std::tuple<ck_tile::half_t, PipeC<TdmPipeline::V1>>,
                                        std::tuple<ck_tile::half_t, PipeC<TdmPipeline::V2>>,
                                        std::tuple<ck_tile::bf16_t, PipeC<TdmPipeline::V1>>,
                                        std::tuple<ck_tile::bf16_t, PipeC<TdmPipeline::V2>>>;

TYPED_TEST_SUITE(TestCkTileBatchedContractionTdm, KernelTypesTdm);

// Degenerate contraction == plain GEMM.
TYPED_TEST(TestCkTileBatchedContractionTdm, SingleBatchSquare)
{
    this->template Run<1, 1, 1, 1>({{1}, {512}, {512}, {512}});
}

TYPED_TEST(TestCkTileBatchedContractionTdm, MultiDimGroups)
{
    this->template Run<1, 2, 2, 2>({{4}, {4, 32}, {2, 64}, {2, 64}});
}

// K_total = 144 is not a multiple of K_Tile (32): the TDM hardware clips the K tail.
TYPED_TEST(TestCkTileBatchedContractionTdm, RaggedK)
{
    this->template Run<1, 2, 2, 2>({{4}, {4, 32}, {2, 64}, {3, 48}});
}

// M_total = 200, N_total = 136, K_total = 100: ragged against the 64x64x32 tile in every
// dimension.
TYPED_TEST(TestCkTileBatchedContractionTdm, RaggedMNK)
{
    this->template Run<1, 2, 1, 2>({{2}, {4, 50}, {136}, {4, 25}});
}

TYPED_TEST(TestCkTileBatchedContractionTdm, SmallK)
{
    this->template Run<1, 2, 2, 2>({{3}, {4, 32}, {2, 64}, {2, 32}}); // K_total = 64
    this->template Run<1, 2, 2, 2>({{3}, {4, 32}, {2, 64}, {2, 64}}); // K_total = 128
}

TYPED_TEST(TestCkTileBatchedContractionTdm, MultiDimBatch)
{
    this->template Run<2, 2, 2, 2>({{2, 3}, {4, 32}, {2, 64}, {2, 64}});
}

// A rows carry a padded pitch (K_total + 64): the last M stride is not K_total but the
// M group itself stays affinely collapsible, so the case must be accepted and correct.
TYPED_TEST(TestCkTileBatchedContractionTdm, NonPackedARowPitch)
{
    const ck_tile::index_t K_total = 2 * 64;
    const ck_tile::index_t pitch   = K_total + 64;
    const ck_tile::index_t M0 = 4, M1 = 32, G = 2;
    // [G, M0, M1, K0, K1]
    const std::vector<ck_tile::index_t> a_strides = {M0 * M1 * pitch, M1 * pitch, pitch, 64, 1};
    ContractionCase c{{G}, {M0, M1}, {2, 64}, {2, 64}, a_strides};
    this->template Run<1, 2, 2, 2>(c);
}

// Exactly one split is required: TdmEpilogue overwrites E.
TYPED_TEST(TestCkTileBatchedContractionTdm, RejectSplitK)
{
    using Kernel = typename TestFixture::template Kernel<1, 2, 2, 2>;
    ContractionCase c{{2}, {4, 32}, {2, 64}, {2, 64}};
    for(ck_tile::index_t k_batch : {0, -1, 2})
    {
        SCOPED_TRACE(k_batch);
        c.k_batch = k_batch;
        const auto args =
            make_host_args<typename TestFixture::DataType, TestFixture::Pipe, 1, 2, 2, 2>(
                c, nullptr, nullptr, nullptr);
        const auto kargs = Kernel::MakeKernelArgs(args);
        EXPECT_FALSE(Kernel::IsSupportedArguments(kargs));
    }
}

TYPED_TEST(TestCkTileBatchedContractionTdm, RejectNonPositiveK)
{
    using Kernel = typename TestFixture::template Kernel<1, 2, 2, 2>;
    ContractionCase c{{2}, {4, 32}, {2, 64}, {2, 64}};
    const auto args = make_host_args<typename TestFixture::DataType, TestFixture::Pipe, 1, 2, 2, 2>(
        c, nullptr, nullptr, nullptr);
    auto kargs = Kernel::MakeKernelArgs(args);
    for(ck_tile::index_t K : {0, -1})
    {
        SCOPED_TRACE(K);
        kargs.K_total = K;
        EXPECT_FALSE(Kernel::IsSupportedArguments(kargs));
    }
}

// A stride inside the M group that breaks affine collapsibility must be rejected on the host.
TYPED_TEST(TestCkTileBatchedContractionTdm, RejectNonPackedMGroup)
{
    using Kernel                   = typename TestFixture::template Kernel<1, 2, 2, 2>;
    const ck_tile::index_t K_total = 128;
    const ck_tile::index_t M0 = 4, M1 = 32;
    // M0 stride has an extra gap: M0 stride != M1 * M1-stride.
    const std::vector<ck_tile::index_t> a_strides = {
        M0 * (M1 * K_total + 256), M1 * K_total + 256, K_total, 64, 1};
    ContractionCase c{{2}, {M0, M1}, {2, 64}, {2, 64}, a_strides};
    const auto args = make_host_args<typename TestFixture::DataType, TestFixture::Pipe, 1, 2, 2, 2>(
        c, nullptr, nullptr, nullptr);
    EXPECT_THROW(Kernel::MakeKernelArgs(args), std::invalid_argument);
}

// K innermost stride must be 1.
TYPED_TEST(TestCkTileBatchedContractionTdm, RejectNonUnitInnermostK)
{
    using Kernel              = typename TestFixture::template Kernel<1, 2, 2, 2>;
    const ck_tile::index_t M0 = 4, M1 = 32;
    // [G, M0, M1, K0, K1] with K0/K1 swapped in memory (K1 stride 2, K0 stride 1).
    const std::vector<ck_tile::index_t> a_strides = {M0 * M1 * 128, M1 * 128, 128, 1, 2};
    ContractionCase c{{2}, {M0, M1}, {2, 64}, {2, 64}, a_strides};
    const auto args = make_host_args<typename TestFixture::DataType, TestFixture::Pipe, 1, 2, 2, 2>(
        c, nullptr, nullptr, nullptr);
    EXPECT_THROW(Kernel::MakeKernelArgs(args), std::invalid_argument);
}

// A zero (broadcast) stride on a non-unit M dim cannot be addressed by TDM and must be
// rejected on the host.
TYPED_TEST(TestCkTileBatchedContractionTdm, RejectZeroStrideMDim)
{
    using Kernel              = typename TestFixture::template Kernel<1, 2, 2, 2>;
    const ck_tile::index_t M0 = 4, M1 = 32;
    // [G, M0, M1, K0, K1] with A broadcast over M: both M strides are 0, which is affinely
    // consistent (0 == 0 * M1) and must still be rejected.
    const std::vector<ck_tile::index_t> a_strides = {128, 0, 0, 64, 1};
    ContractionCase c{{2}, {M0, M1}, {2, 64}, {2, 64}, a_strides};
    const auto args = make_host_args<typename TestFixture::DataType, TestFixture::Pipe, 1, 2, 2, 2>(
        c, nullptr, nullptr, nullptr);
    EXPECT_THROW(Kernel::MakeKernelArgs(args), std::invalid_argument);
}

// Host-only: with valid arguments, IsSupportedArguments is true exactly on gfx125x devices.
TYPED_TEST(TestCkTileBatchedContractionTdm, SupportedOnlyOnGfx125)
{
    using Kernel = typename TestFixture::template Kernel<1, 2, 2, 2>;
    ContractionCase c{{2}, {4, 32}, {2, 64}, {2, 64}};
    const auto args = make_host_args<typename TestFixture::DataType, TestFixture::Pipe, 1, 2, 2, 2>(
        c, nullptr, nullptr, nullptr);
    const auto kargs = Kernel::MakeKernelArgs(args);
    EXPECT_EQ(Kernel::IsSupportedArguments(kargs), device_is_gfx1250());
}

// Host-only: a base pointer that is not aligned to the element size is rejected.
TYPED_TEST(TestCkTileBatchedContractionTdm, RejectMisalignedPointer)
{
    using DataType = typename TestFixture::DataType;
    using Kernel   = typename TestFixture::template Kernel<1, 2, 2, 2>;
    static_assert(sizeof(DataType) > 1, "test needs a multi-byte element type");
    ContractionCase c{{2}, {4, 32}, {2, 64}, {2, 64}};
    alignas(16) static unsigned char storage[64];
    const void* odd = storage + 1;
    const auto args =
        make_host_args<DataType, TestFixture::Pipe, 1, 2, 2, 2>(c, odd, storage, storage);
    const auto kargs = Kernel::MakeKernelArgs(args);
    EXPECT_FALSE(Kernel::IsSupportedArguments(kargs));
}

// Host-only: two non-unit G dims that do not collapse to one batch stride are rejected.
TYPED_TEST(TestCkTileBatchedContractionTdm, RejectNonPackedGGroup)
{
    using Kernel                          = typename TestFixture::template Kernel<2, 2, 2, 2>;
    const std::vector<ck_tile::index_t> G = {2, 3}, M = {4, 32}, N = {2, 64}, K = {2, 64};
    const ck_tile::index_t MK = 4 * 32 * 128;
    // G0 stride != G1 length * G1 stride.
    ContractionCase c{G, M, N, K, packed_strides_with(concat({G, M, K}), 0, 3 * MK + 8)};
    const auto args = make_host_args<typename TestFixture::DataType, TestFixture::Pipe, 2, 2, 2, 2>(
        c, nullptr, nullptr, nullptr);
    EXPECT_THROW(Kernel::MakeKernelArgs(args), std::invalid_argument);
}

// Host-only: a trailing length-1 G dim with an arbitrary stride must not be used as the batch
// stride; the batch stride comes from the fastest non-unit G dim.
TYPED_TEST(TestCkTileBatchedContractionTdm, BatchStrideSkipsUnitGDims)
{
    using Kernel                          = typename TestFixture::template Kernel<2, 2, 2, 2>;
    const std::vector<ck_tile::index_t> G = {2, 1}, M = {4, 32}, N = {2, 64}, K = {2, 64};
    const ck_tile::index_t MK = 4 * 32 * 128, NK = 128 * 128, MN = 4 * 32 * 128;
    ContractionCase c{G, M, N, K, packed_strides_with(concat({G, M, K}), 1, 5)};
    c.B_strides     = packed_strides_with(concat({G, N, K}), 1, 7);
    c.E_strides     = packed_strides_with(concat({G, M, N}), 1, 3);
    const auto args = make_host_args<typename TestFixture::DataType, TestFixture::Pipe, 2, 2, 2, 2>(
        c, nullptr, nullptr, nullptr);
    const auto kargs = Kernel::MakeKernelArgs(args);
    EXPECT_EQ(kargs.batch_stride_A, MK);
    EXPECT_EQ(kargs.batch_stride_B, NK);
    EXPECT_EQ(kargs.batch_stride_E, MN);
}

// G = {2, 1}: the trailing unit G dim carries an arbitrary stride in A, B and E.
TYPED_TEST(TestCkTileBatchedContractionTdm, MultiDimBatchTrailingUnitG)
{
    const std::vector<ck_tile::index_t> G = {2, 1}, M = {4, 32}, N = {2, 64}, K = {2, 64};
    ContractionCase c{G, M, N, K, packed_strides_with(concat({G, M, K}), 1, 5)};
    c.B_strides = packed_strides_with(concat({G, N, K}), 1, 7);
    c.E_strides = packed_strides_with(concat({G, M, N}), 1, 3);
    this->template Run<2, 2, 2, 2>(c);
}

// G = {1, 3}: the leading unit G dim carries an arbitrary stride in A, B and E.
TYPED_TEST(TestCkTileBatchedContractionTdm, MultiDimBatchLeadingUnitG)
{
    const std::vector<ck_tile::index_t> G = {1, 3}, M = {4, 32}, N = {2, 64}, K = {2, 64};
    ContractionCase c{G, M, N, K, packed_strides_with(concat({G, M, K}), 0, 11)};
    c.B_strides = packed_strides_with(concat({G, N, K}), 0, 13);
    c.E_strides = packed_strides_with(concat({G, M, N}), 0, 17);
    this->template Run<2, 2, 2, 2>(c);
}

// Length-1 G and M dims with arbitrary strides in the same problem.
TYPED_TEST(TestCkTileBatchedContractionTdm, UnitGAndMDimsArbitraryStrides)
{
    const std::vector<ck_tile::index_t> G = {2, 1}, M = {1, 128}, N = {2, 64}, K = {2, 64};
    auto a_strides = packed_strides_with(concat({G, M, K}), 1, 9);
    a_strides[2]   = 19; // M0 (length 1)
    auto e_strides = packed_strides_with(concat({G, M, N}), 1, 21);
    e_strides[2]   = 23; // M0 (length 1)
    ContractionCase c{G, M, N, K, a_strides};
    c.E_strides = e_strides;
    this->template Run<2, 2, 2, 2>(c);
}

// Padded G strides (larger than the packed M*K / N*K / M*N): accepted and correct. E stride
// gaps must keep the poison value.
TYPED_TEST(TestCkTileBatchedContractionTdm, PaddedGStride)
{
    const std::vector<ck_tile::index_t> G = {2, 3}, M = {4, 32}, N = {2, 64}, K = {2, 64};
    const ck_tile::index_t MK = 4 * 32 * 128 + 128, NK = 128 * 128 + 64, MN = 4 * 32 * 128 + 32;
    auto a_strides = packed_strides_with(concat({G, M, K}), 1, MK);
    a_strides[0]   = 3 * MK;
    auto b_strides = packed_strides_with(concat({G, N, K}), 1, NK);
    b_strides[0]   = 3 * NK;
    auto e_strides = packed_strides_with(concat({G, M, N}), 1, MN);
    e_strides[0]   = 3 * MN;
    ContractionCase c{G, M, N, K, a_strides};
    c.B_strides = b_strides;
    c.E_strides = e_strides;
    this->template Run<2, 2, 2, 2>(c);
}
