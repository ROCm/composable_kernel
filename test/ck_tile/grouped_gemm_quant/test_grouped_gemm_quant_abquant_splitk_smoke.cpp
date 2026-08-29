// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// Minimal smoke test for QuantGroupedGemmKernel<..., ABQuantGrouped, ...>
// with split-K. The main ABQuant + split-K code paths (uniform splits,
// runtime-tail dispatch, BPreshuffle, etc.) are exercised in
// test/ck_tile/gemm_block_scale/test_gemm_quant_abquant_splitk_*.cpp using
// the non-grouped kernel; the grouped kernel reuses the same Base::RunGemm.
// What we cover here is grouped-specific:
//
//   1. Compile-time instantiation of the grouped kernel for ABQuantGrouped.
//      If any inner static_assert (e.g. RowMajor BQ + ABQuant) fires this
//      file won't compile.
//   2. Host-side IsSupportedArgument acceptance for valid k_batch and
//      rejection for k_batch <= 0.
//   3. A single end-to-end correctness run with k_batch == 2 on a single
//      group.  This exercises QuantGroupedGemmKernel::Run's per-batch
//      pointer offsetting (a/b/aq/bq) and the aq_group_offset wired into
//      MakeAQBlockWindow -- both of which were latent bugs before this
//      change because every existing grouped test launched with k_batch=1.

#include "ck_tile/host.hpp"
#include "ck_tile/ops/epilogue.hpp"
#include "ck_tile/ops/gemm.hpp"
#include "ck_tile/ops/gemm_quant.hpp"

#include <gtest/gtest.h>

namespace {

using ALayout     = ck_tile::tensor_layout::gemm::RowMajor;
using BLayout     = ck_tile::tensor_layout::gemm::ColumnMajor;
using CLayout     = ck_tile::tensor_layout::gemm::RowMajor;
using AQLayout    = ck_tile::tensor_layout::gemm::RowMajor;
using BQLayout    = ck_tile::tensor_layout::gemm::ColumnMajor;
using ADataType   = ck_tile::fp8_t;
using BDataType   = ck_tile::fp8_t;
using AccDataType = float;
using CDataType   = ck_tile::half_t;
using QDataType   = float;

using ComputeDataType = ADataType;
using AQuantGroupSize = ck_tile::QuantGroupShape<ck_tile::sequence<1, 1, 128>>;
using BQuantGroupSize = ck_tile::QuantGroupShape<ck_tile::sequence<1, 128, 128>>;

constexpr ck_tile::index_t M_Tile      = 128;
constexpr ck_tile::index_t N_Tile      = 128;
constexpr ck_tile::index_t K_Tile      = 128;
constexpr ck_tile::index_t M_Warp      = 1;
constexpr ck_tile::index_t N_Warp      = 4;
constexpr ck_tile::index_t K_Warp      = 1;
constexpr ck_tile::index_t M_Warp_Tile = 16;
constexpr ck_tile::index_t N_Warp_Tile = 16;
#if CK_TILE_USE_WMMA
constexpr ck_tile::index_t K_Warp_Tile = 16;
#else
constexpr ck_tile::index_t K_Warp_Tile = 64;
#endif
constexpr bool kPadM                   = false;
constexpr bool kPadN                   = false;
constexpr bool kPadK                   = false;
constexpr bool TransposeC              = true;
constexpr ck_tile::QuantType QuantMode = ck_tile::QuantType::ABQuantGrouped;

using GemmShape = ck_tile::TileGemmShape<ck_tile::sequence<M_Tile, N_Tile, K_Tile>,
                                         ck_tile::sequence<M_Warp, N_Warp, K_Warp>,
                                         ck_tile::sequence<M_Warp_Tile, N_Warp_Tile, K_Warp_Tile>>;

using TilePartitioner = ck_tile::GemmTile1DPartitioner<GemmShape>;

using GemmTraits = ck_tile::TileGemmQuantTraits<kPadM,
                                                kPadN,
                                                kPadK,
                                                /*APreshuffleQuant=*/false,
                                                /*BPreshuffleQuant=*/false,
                                                /*PreshuffleB=*/false,
                                                ALayout,
                                                BLayout,
                                                CLayout,
                                                QuantMode,
                                                AQLayout,
                                                BQLayout,
                                                TransposeC,
                                                /*DoubleSmemBuffer=*/false>;

// PipelineProblem template used to drive BaseGemmPipeline / TailHandler
// dispatch.  The HasHotLoop / TailNumber template parameters are fixed at
// compile time per-instantiation; TailHandler picks the right one at
// runtime based on (K, k_batch).
template <bool HasHotLoop, ck_tile::TailNumber TailNum>
using AbquantPipelineProblem =
    ck_tile::GemmABQuantPipelineProblem<ADataType,
                                        QDataType, // AQDataType
                                        BDataType,
                                        QDataType, // BQDataType
                                        AccDataType,
                                        GemmShape,
                                        GemmTraits,
                                        AQuantGroupSize,
                                        BQuantGroupSize,
                                        TransposeC,
                                        ComputeDataType,
                                        ck_tile::GemmPipelineScheduler::Intrawave,
                                        HasHotLoop,
                                        TailNum>;

using BasePipelineProblem = ck_tile::GemmPipelineProblemBase<ADataType,
                                                             BDataType,
                                                             AccDataType,
                                                             GemmShape,
                                                             GemmTraits,
                                                             ComputeDataType>;
using BaseGemmPipeline    = ck_tile::BaseGemmPipelineAgBgCrCompV3<BasePipelineProblem>;

// Compile-time instantiation check: a representative set of types must
// build cleanly (this is the minimal "smoke" portion of the test).
using SmokePipeline = ck_tile::ABQuantGemmPipelineAgBgCrCompV3<
    AbquantPipelineProblem<true, ck_tile::TailNumber::Full>>;
using SmokeEpilogue =
    ck_tile::CShuffleEpilogue<ck_tile::CShuffleEpilogueProblem<ADataType,
                                                               BDataType,
                                                               ck_tile::tuple<>,
                                                               AccDataType,
                                                               CDataType,
                                                               ck_tile::tuple<>,
                                                               CLayout,
                                                               ck_tile::element_wise::PassThrough,
                                                               TilePartitioner::MPerBlock,
                                                               TilePartitioner::NPerBlock,
                                                               M_Warp,
                                                               N_Warp,
                                                               M_Warp_Tile,
                                                               N_Warp_Tile,
                                                               K_Warp_Tile,
                                                               TransposeC>>;

using SmokeGroupedKernel =
    ck_tile::QuantGroupedGemmKernel<TilePartitioner, SmokePipeline, SmokeEpilogue, QuantMode>;
using SmokeBaseKernel = typename SmokeGroupedKernel::Base;

static_assert(sizeof(SmokeGroupedKernel) > 0, "QuantGroupedGemmKernel must instantiate");
static_assert(sizeof(SmokeBaseKernel) > 0, "QuantGemmKernel base must instantiate");

ck_tile::QuantGemmKernelArgs MakeKargsForValidation(ck_tile::index_t k_batch)
{
    constexpr ck_tile::index_t M = 128;
    constexpr ck_tile::index_t N = 128;
    constexpr ck_tile::index_t K = 1024;

    ck_tile::QuantGemmKernelArgs kargs{};
    kargs.a_ptr     = nullptr;
    kargs.b_ptr     = nullptr;
    kargs.aq_ptr    = nullptr;
    kargs.bq_ptr    = nullptr;
    kargs.c_ptr     = nullptr;
    kargs.M         = M;
    kargs.N         = N;
    kargs.K         = K;
    kargs.QK_A      = ck_tile::integer_divide_ceil(K, AQuantGroupSize::kK);
    kargs.QK_B      = ck_tile::integer_divide_ceil(K, BQuantGroupSize::kK);
    kargs.stride_A  = K;
    kargs.stride_B  = K;
    kargs.stride_C  = N;
    kargs.stride_AQ = kargs.QK_A;
    kargs.stride_BQ = N;
    kargs.k_batch   = k_batch;
    return kargs;
}

// End-to-end runner.  Builds host tensors, fills them with deterministic
// random data, runs the grouped kernel for a single group, and validates
// against the host reference.  Kept intentionally narrow: one group, fixed
// layouts/types, no preshuffle.  The wider parameter space is covered by
// the non-grouped ABQuant tests.
bool RunSingleGroupABQuantSplitK(ck_tile::index_t M,
                                 ck_tile::index_t N,
                                 ck_tile::index_t K,
                                 ck_tile::index_t k_batch)
{
    auto is_row_major = [](auto layout) {
        return ck_tile::bool_constant<std::is_same_v<ck_tile::remove_cvref_t<decltype(layout)>,
                                                     ck_tile::tensor_layout::gemm::RowMajor>>{};
    };

    const ck_tile::index_t stride_A = ck_tile::get_default_stride(M, K, 0, is_row_major(ALayout{}));
    const ck_tile::index_t stride_B = ck_tile::get_default_stride(K, N, 0, is_row_major(BLayout{}));
    const ck_tile::index_t stride_C = ck_tile::get_default_stride(M, N, 0, is_row_major(CLayout{}));

    const ck_tile::index_t AQK = ck_tile::integer_divide_ceil(K, AQuantGroupSize::kK);
    const ck_tile::index_t BQN = ck_tile::integer_divide_ceil(N, BQuantGroupSize::kN);
    const ck_tile::index_t BQK = ck_tile::integer_divide_ceil(K, BQuantGroupSize::kK);
    const ck_tile::index_t stride_AQ =
        ck_tile::get_default_stride(M, AQK, 0, is_row_major(AQLayout{}));
    const ck_tile::index_t stride_BQ =
        ck_tile::get_default_stride(BQK, BQN, 0, is_row_major(BQLayout{}));

    ck_tile::HostTensor<ADataType> a_m_k(
        ck_tile::host_tensor_descriptor(M, K, stride_A, is_row_major(ALayout{})));
    ck_tile::HostTensor<BDataType> b_k_n(
        ck_tile::host_tensor_descriptor(K, N, stride_B, is_row_major(BLayout{})));
    ck_tile::HostTensor<QDataType> aq_m_aqk(
        ck_tile::host_tensor_descriptor(M, AQK, stride_AQ, is_row_major(AQLayout{})));
    ck_tile::HostTensor<QDataType> bq_bqk_bqn(
        ck_tile::host_tensor_descriptor(BQK, BQN, stride_BQ, is_row_major(BQLayout{})));

    ck_tile::FillUniformDistribution<ADataType>{-2.0f, 3.0f}(a_m_k);
    ck_tile::FillUniformDistribution<BDataType>{-5.0f, 5.0f}(b_k_n);
    ck_tile::FillUniformDistribution<QDataType>{-2.0f, 2.0f}(aq_m_aqk);
    ck_tile::FillUniformDistribution<QDataType>{-2.0f, 2.0f}(bq_bqk_bqn);

    ck_tile::DeviceMem a_m_k_dev_buf(a_m_k.get_element_space_size() * sizeof(ADataType));
    ck_tile::DeviceMem b_k_n_dev_buf(b_k_n.get_element_space_size() * sizeof(BDataType));
    ck_tile::DeviceMem aq_m_aqk_dev_buf(aq_m_aqk.get_element_space_size() * sizeof(QDataType));
    ck_tile::DeviceMem bq_bqk_bqn_dev_buf(bq_bqk_bqn.get_element_space_size() * sizeof(QDataType));
    ck_tile::DeviceMem c_m_n_dev_buf(M * N * sizeof(CDataType));

    a_m_k_dev_buf.ToDevice(a_m_k.data());
    b_k_n_dev_buf.ToDevice(b_k_n.data());
    aq_m_aqk_dev_buf.ToDevice(aq_m_aqk.data());
    bq_bqk_bqn_dev_buf.ToDevice(bq_bqk_bqn.data());

    if(k_batch > 1)
    {
        c_m_n_dev_buf.SetZero();
    }

    std::vector<ck_tile::QuantGroupedGemmHostArgs> gemm_descs;
    gemm_descs.emplace_back(a_m_k_dev_buf.GetDeviceBuffer(),
                            b_k_n_dev_buf.GetDeviceBuffer(),
                            c_m_n_dev_buf.GetDeviceBuffer(),
                            aq_m_aqk_dev_buf.GetDeviceBuffer(),
                            bq_bqk_bqn_dev_buf.GetDeviceBuffer(),
                            k_batch,
                            M,
                            N,
                            K,
                            AQK,
                            BQK,
                            stride_A,
                            stride_B,
                            stride_C,
                            stride_AQ,
                            stride_BQ);

    // Workspace holds the per-group QuantGemmTransKernelArg vector, copied
    // to device before the launch.
    ck_tile::DeviceMem gemm_workspace(gemm_descs.size() * sizeof(ck_tile::QuantGemmTransKernelArg));
    void* kargs_ptr = gemm_workspace.GetDeviceBuffer();

    // Drive TailHandler dispatch with split-K-aware K_split (mirrors
    // run_quant_gemm_impl in the non-grouped fixture).
    constexpr auto K1                  = GemmShape::WarpTile::at(ck_tile::number<2>{});
    const ck_tile::index_t K_split     = (k_batch == 1)
                                             ? ck_tile::integer_least_multiple(K, K_Tile)
                                             : ck_tile::get_splitk_batch_k_read(K, k_batch, K1);
    const ck_tile::index_t num_loop    = TilePartitioner::GetLoopNum(K_split);
    const bool has_hot_loop            = BaseGemmPipeline::BlockHasHotloop(num_loop);
    const ck_tile::TailNumber tail_num = BaseGemmPipeline::GetBlockLoopTailNum(num_loop);

    ck_tile::stream_config s{};

    auto Run = [&](const auto has_hot_loop_, const auto tail_number_) {
        constexpr bool has_hot_loop_v = has_hot_loop_.value;
        constexpr auto tail_number_v  = tail_number_.value;

        using PipelineProblem = AbquantPipelineProblem<has_hot_loop_v, tail_number_v>;
        using GemmPipeline    = ck_tile::ABQuantGemmPipelineAgBgCrCompV3<PipelineProblem>;
        using GemmEpilogue    = ck_tile::CShuffleEpilogue<
               ck_tile::CShuffleEpilogueProblem<ADataType,
                                                BDataType,
                                                ck_tile::tuple<>,
                                                AccDataType,
                                                CDataType,
                                                ck_tile::tuple<>,
                                                CLayout,
                                                ck_tile::element_wise::PassThrough,
                                                TilePartitioner::MPerBlock,
                                                TilePartitioner::NPerBlock,
                                                M_Warp,
                                                N_Warp,
                                                M_Warp_Tile,
                                                N_Warp_Tile,
                                                K_Warp_Tile,
                                                TransposeC>>;
        using Kernel =
            ck_tile::QuantGroupedGemmKernel<TilePartitioner, GemmPipeline, GemmEpilogue, QuantMode>;

        auto kargs = Kernel::MakeKargs(gemm_descs);
        if(!Kernel::IsSupportedArgument(kargs))
        {
            throw std::runtime_error("Grouped ABQuant SplitK args not supported");
        }

        const dim3 grids  = Kernel::GridSize(gemm_descs);
        const dim3 blocks = Kernel::BlockSize();

        HIP_CHECK_ERROR(hipMemcpyWithStream(kargs_ptr,
                                            kargs.data(),
                                            kargs.size() * sizeof(ck_tile::QuantGemmTransKernelArg),
                                            hipMemcpyHostToDevice,
                                            s.stream_id_));

        ck_tile::launch_kernel(
            s,
            ck_tile::make_kernel<1>(Kernel{},
                                    grids,
                                    blocks,
                                    0,
                                    ck_tile::cast_pointer_to_constant_address_space(kargs_ptr),
                                    static_cast<ck_tile::index_t>(gemm_descs.size())));
    };

    BaseGemmPipeline::TailHandler(Run, has_hot_loop, tail_num);

    ck_tile::HostTensor<CDataType> c_m_n_host_ref(
        ck_tile::host_tensor_descriptor(M, N, stride_C, is_row_major(CLayout{})));
    c_m_n_host_ref.SetZero();
    ck_tile::reference_gemm_abquant<ADataType,
                                    QDataType,
                                    BDataType,
                                    QDataType,
                                    AccDataType,
                                    CDataType,
                                    AQuantGroupSize,
                                    BQuantGroupSize>(
        a_m_k, aq_m_aqk, b_k_n, bq_bqk_bqn, c_m_n_host_ref);

    ck_tile::HostTensor<CDataType> c_m_n_dev_result(
        ck_tile::host_tensor_descriptor(M, N, stride_C, is_row_major(CLayout{})));
    c_m_n_dev_buf.FromDevice(c_m_n_dev_result.mData.data());

    const float max_accumulated_value =
        *std::max_element(c_m_n_host_ref.mData.begin(), c_m_n_host_ref.mData.end());
    const auto rtol =
        std::max(ck_tile::get_relative_threshold<ADataType, CDataType, AccDataType>(
                     ck_tile::integer_divide_ceil(K, k_batch)),
                 ck_tile::get_relative_threshold<CDataType, CDataType, CDataType>(k_batch));
    const auto atol =
        std::max(ck_tile::get_absolute_threshold<ADataType, CDataType, AccDataType>(
                     max_accumulated_value / k_batch, ck_tile::integer_divide_ceil(K, k_batch)),
                 ck_tile::get_absolute_threshold<CDataType, CDataType, CDataType>(
                     max_accumulated_value, k_batch));

    return ck_tile::check_err(
        c_m_n_dev_result, c_m_n_host_ref, "Grouped ABQuant SplitK mismatch", rtol, atol);
}

} // namespace

TEST(GroupedABQuantSplitKSmoke, AcceptsKBatch1)
{
    EXPECT_TRUE(SmokeBaseKernel::IsSupportedArgument(MakeKargsForValidation(/*k_batch=*/1)));
}

TEST(GroupedABQuantSplitKSmoke, AcceptsKBatch2)
{
    EXPECT_TRUE(SmokeBaseKernel::IsSupportedArgument(MakeKargsForValidation(/*k_batch=*/2)));
}

TEST(GroupedABQuantSplitKSmoke, RejectsKBatchZero)
{
    EXPECT_FALSE(SmokeBaseKernel::IsSupportedArgument(MakeKargsForValidation(/*k_batch=*/0)));
}

TEST(GroupedABQuantSplitKSmoke, RejectsKBatchNegative)
{
    EXPECT_FALSE(SmokeBaseKernel::IsSupportedArgument(MakeKargsForValidation(/*k_batch=*/-1)));
}

// End-to-end correctness for the grouped ABQuant + split-K path on a
// single group with k_batch=2.  Catches regressions in:
//   - QuantGroupedGemmKernel::Run per-batch a/b/aq/bq pointer offsetting,
//   - aq_group_offset wiring through MakeAQBlockWindow.
// Both were latent (existing grouped tests only used k_batch=1).
TEST(GroupedABQuantSplitKSmoke, EndToEnd_SingleGroup_KBatch2)
{
    EXPECT_TRUE(RunSingleGroupABQuantSplitK(/*M=*/128,
                                            /*N=*/128,
                                            /*K=*/512,
                                            /*k_batch=*/2));
}
