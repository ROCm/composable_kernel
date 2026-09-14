// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <algorithm>
#include <type_traits>
#include <utility>
#include <vector>

#include "test_gemm_quant_base.hpp"
#include "ck_tile/host/permute_pk_int4.hpp"
#include "ck_tile/host/tensor_shuffle_utils.hpp"

struct GemmConfigBase
{
    static constexpr bool kPadM = false;
    static constexpr bool kPadN = false;
    static constexpr bool kPadK = false;

    static constexpr bool PermuteA = false;
    static constexpr bool PermuteB = false;

    static constexpr bool TransposeC            = false;
    static constexpr bool UseStructuredSparsity = false;

    static constexpr int kBlockPerCu                         = 1;
    static constexpr ck_tile::index_t TileParitionerGroupNum = 8;
    static constexpr ck_tile::index_t TileParitionerM01      = 4;
    static constexpr auto Scheduler                 = ck_tile::GemmPipelineScheduler::Intrawave;
    static constexpr ck_tile::index_t NumWaveGroups = 1;
    static constexpr bool APreshuffleQuant          = false;
    static constexpr bool BPreshuffleQuant          = false;
    static constexpr bool PreshuffleB               = false;
    static constexpr bool DoubleSmemBuffer          = false;
    static constexpr bool TiledMMAPermuteN          = false;

    // Default GEMM tile sizes for tests
    static constexpr ck_tile::index_t M_Tile = 16;
    static constexpr ck_tile::index_t N_Tile = 128;
    static constexpr ck_tile::index_t K_Tile = 256;

    static constexpr ck_tile::index_t M_Warp = 1;
    static constexpr ck_tile::index_t N_Warp = 4;
    static constexpr ck_tile::index_t K_Warp = 1;

    static constexpr ck_tile::index_t M_Warp_Tile = 16;
    static constexpr ck_tile::index_t N_Warp_Tile = 16;
    static constexpr ck_tile::index_t K_Warp_Tile = get_k_warp_tile<false>();

    // Opt-in to the 64-bit global load/store path for tensors whose single-dimension
    // byte extent exceeds the 2GB buffer-addressing limit. Off by default.
    static constexpr bool LargeTensors = false;
};

// Enables the large-tensor (64-bit global load/store) code path. Same tile shape as the
// base config; only the LargeTensors opt-in differs.
struct GemmConfigLargeTensor : public GemmConfigBase
{
    static constexpr bool LargeTensors = true;
};

// Large-tensor path with padding enabled so non-tile-multiple M/N/K are accepted
// (IsSupportedArgument gates the divisibility checks on kPad*) and the remainder tiles
// are guarded on the unmasked 64-bit global path.
struct GemmConfigLargeTensorPadded : public GemmConfigBase
{
    static constexpr bool kPadM        = true;
    static constexpr bool kPadN        = true;
    static constexpr bool kPadK        = true;
    static constexpr bool LargeTensors = true;
};

struct GemmConfigDecode : public GemmConfigBase
{
    static constexpr ck_tile::index_t M_Tile      = 16;
    static constexpr ck_tile::index_t N_Tile      = 64;
    static constexpr ck_tile::index_t K_Tile      = 256;
    static constexpr ck_tile::index_t K_Warp_Tile = get_k_warp_tile<true>();
};

struct GemmConfigPrefill : public GemmConfigBase
{
    static constexpr ck_tile::index_t M_Tile      = 128;
    static constexpr ck_tile::index_t N_Tile      = 128;
    static constexpr ck_tile::index_t K_Tile      = 128;
    static constexpr ck_tile::index_t K_Warp_Tile = get_k_warp_tile<true>();
};

struct GemmConfigPrefillIntrawave : public GemmConfigBase
{
    static constexpr ck_tile::index_t M_Tile = 128;
    static constexpr ck_tile::index_t N_Tile = 128;
    static constexpr ck_tile::index_t K_Tile = 128;
    static constexpr auto Scheduler          = ck_tile::GemmPipelineScheduler::Intrawave;
};

struct GemmConfigDecodeIntrawave : public GemmConfigBase
{
    static constexpr ck_tile::index_t M_Tile = 16;
    static constexpr ck_tile::index_t N_Tile = 64;
    static constexpr ck_tile::index_t K_Tile = 256;
    // Workaround for ROCm 7.13 compiler codegen regression on gfx1201 (RDNA4).
    // AQuantGemmPipelineAgBgCrMem always uses the Intrawave pipeline implementation
    // regardless of this value, but the Scheduler enum changes the Problem type identity,
    // causing a different template instantiation. The Intrawave instantiation triggers
    // incorrect ISA generation (wrong global instruction scheduling in the hot loop),
    // producing ~3% wrong values for FP8/BF8 AQuant GEMM with K > K_Tile.
    // Setting Interwave creates a different instantiation that gets correct codegen.
    // Revert to Intrawave once the compiler is fixed.
    static constexpr auto Scheduler = ck_tile::GemmPipelineScheduler::Interwave;
};

struct GemmConfigMx : public GemmConfigBase
{
    static constexpr ck_tile::index_t M_Tile = 128;
    static constexpr ck_tile::index_t N_Tile = 128;
    static constexpr ck_tile::index_t K_Tile = 128;
};

// This configuration uses K_Warp_Tile = 64 on CDNA. In this way, on gfx950 we can use
// LDS load transpose on matrix B (FP4) because the instruction requires each
// lane to load 16 4bits elements
struct GemmConfigMxFP4 : public GemmConfigBase
{
    static constexpr ck_tile::index_t M_Tile      = 128;
    static constexpr ck_tile::index_t N_Tile      = 128;
    static constexpr ck_tile::index_t K_Tile      = 128;
    static constexpr ck_tile::index_t K_Warp_Tile = get_k_warp_tile<true>();
};

struct GemmConfigPreshuffleQuant : public GemmConfigBase
{
    static constexpr bool APreshuffleQuant = true;
};

struct GemmConfigTransposeC : public GemmConfigBase
{
    static constexpr bool TransposeC = true;
};

struct GemmConfigPreshuffleQuantTransposeC : public GemmConfigBase
{
    static constexpr bool APreshuffleQuant = true;
    static constexpr bool TransposeC       = true;
};

struct GemmConfigPadding : public GemmConfigBase
{
    static constexpr bool kPadN = true;
    static constexpr bool kPadK = true;
};

struct GemmConfigPreshuffleBDecode : public GemmConfigDecode
{
    static constexpr bool PreshuffleB      = true;
    static constexpr bool DoubleSmemBuffer = true;
};

struct GemmConfigPreshuffleQuantDecode : public GemmConfigDecode
{
    static constexpr bool BPreshuffleQuant = true;
};

struct GemmConfigPreshuffleBPrefill : public GemmConfigPrefill
{
    static constexpr bool PreshuffleB      = true;
    static constexpr bool DoubleSmemBuffer = true;
};

// Preshuffle-B prefill config with the 64-bit global load/store opt-in enabled, for B tensors
// whose element count exceeds the 2^31 32-bit-offset limit.  Same tile shape as
// GemmConfigPreshuffleBPrefill; only the LargeTensors opt-in differs.
struct GemmConfigPreshuffleBLargeTensor : public GemmConfigPreshuffleBPrefill
{
    static constexpr bool LargeTensors = true;
};

struct GemmConfigPreshuffleBPrefillTransposeC : public GemmConfigPreshuffleBPrefill
{
    static constexpr bool TransposeC = true;
};

struct GemmConfigPreshuffleBPrefillPadded : public GemmConfigPreshuffleBPrefill
{
    static constexpr bool kPadN = true;
    static constexpr bool kPadK = true;
};

struct GemmConfigPreshuffleQuantPrefill : public GemmConfigPrefill
{
    static constexpr bool BPreshuffleQuant = true;
};

struct GemmConfigPreshuffleBPrefillTiledPermuteN : public GemmConfigPreshuffleBPrefill
{
    static constexpr int N_Repeat          = N_Tile / N_Warp_Tile / N_Warp;
    static constexpr bool TiledMMAPermuteN = N_Repeat % 2 == 0;
};

template <bool TransposeC_ = false>
struct GemmConfigPreshuffleBPreshuffleQuantPrefill : public GemmConfigPreshuffleBPrefill
{
    static constexpr bool BPreshuffleQuant = true;
    static constexpr bool TransposeC       = TransposeC_;
};

template <bool TransposeC_ = false>
struct GemmConfigPreshuffleBPreshuffleQuantDecode : public GemmConfigPreshuffleBDecode
{
    static constexpr bool BPreshuffleQuant = true;
    static constexpr bool TransposeC       = TransposeC_;
};

struct GemmConfigPreshuffleB_ABQuant_Prefill : public GemmConfigPreshuffleBPrefill
{
    static constexpr bool TransposeC = true;
};

struct GemmConfigEightWaves : public GemmConfigBase
{
    static constexpr ck_tile::index_t M_Warp = 4;
    static constexpr ck_tile::index_t N_Warp = 2; // NWarps == 2 for ping-pong!
    static constexpr ck_tile::index_t K_Warp = 1;

    static constexpr ck_tile::index_t M_Tile = 192;
    static constexpr ck_tile::index_t N_Tile = 128 * N_Warp;
    static constexpr ck_tile::index_t K_Tile = 128 * K_Warp;

    static constexpr ck_tile::index_t K_Warp_Tile =
        ck_tile::get_k_warp_tile<ck_tile::fp8_t, M_Warp_Tile, true>();

    static constexpr bool kPadK      = false;
    static constexpr bool TransposeC = true;
};

struct GemmConfigEightWaves_PreshuffleB : public GemmConfigEightWaves
{
    static constexpr bool PreshuffleB = true;
};

template <typename Tuple>
class TestCkTileGemmAQuant : public TestCkTileGemmQuantBase<Tuple, TestCkTileGemmAQuant<Tuple>>
{
    using Base = TestCkTileGemmQuantBase<Tuple, TestCkTileGemmAQuant<Tuple>>;
    friend Base;

    public:
    using typename Base::AccDataType;
    using typename Base::ADataType;
    using typename Base::ALayout;
    using typename Base::AQLayout;
    using typename Base::BDataType;
    using typename Base::BLayout;
    using typename Base::CDataType;
    using typename Base::CLayout;
    using typename Base::ComputeDataType;
    using typename Base::QDataType;
    using typename Base::QuantGroupSize;

    static constexpr auto QuantType = Base::QuantType;

    protected:
    void SetUpQuantTypeSpecific() {}
    void TearDownQuantTypeSpecific() {}

    // AQuant-specific data generation
    void run_test_with_validation(ck_tile::index_t M, ck_tile::index_t N, ck_tile::index_t K)
    {
        const ck_tile::index_t stride_A =
            ck_tile::get_default_stride(M, K, 0, this->is_row_major(ALayout{}));
        const ck_tile::index_t stride_B =
            ck_tile::get_default_stride(K, N, 0, this->is_row_major(BLayout{}));
        const ck_tile::index_t stride_C =
            ck_tile::get_default_stride(M, N, 0, this->is_row_major(CLayout{}));

        // AQuant uses grouped quantization for A matrix
        const ck_tile::index_t AQK = ck_tile::integer_divide_ceil(K, QuantGroupSize::kK);
        // AQLayout is parameterized in the test tuple (can be RowMajor or ColumnMajor for AQuant)
        const ck_tile::index_t stride_AQ =
            ck_tile::get_default_stride(M, AQK, 0, this->is_row_major(AQLayout{}));

        // Generate test data
        ck_tile::HostTensor<ADataType> a_m_k(
            ck_tile::host_tensor_descriptor(M, K, stride_A, this->is_row_major(ALayout{})));
        // AQLayout is independently specified for each test case
        ck_tile::HostTensor<QDataType> aq_m_aqk(
            ck_tile::host_tensor_descriptor(M, AQK, stride_AQ, this->is_row_major(AQLayout{})));
        ck_tile::HostTensor<BDataType> b_k_n(
            ck_tile::host_tensor_descriptor(K, N, stride_B, this->is_row_major(BLayout{})));

        // Initialize data with random values
        if constexpr(std::is_same_v<ADataType, ck_tile::pk_int4_t>)
        {
            ck_tile::FillUniformDistribution<ADataType>{-5.0f, 5.0f}(a_m_k);
        }
        else
        {
            ck_tile::FillUniformDistribution<ADataType>{-2.0f, 3.0f}(a_m_k);
        }
        ck_tile::FillUniformDistribution<BDataType>{-5.0f, 5.0f}(b_k_n);
        ck_tile::FillUniformDistribution<QDataType>{-2.0f, 2.0f}(aq_m_aqk);

        // Allocate device memory
        ck_tile::DeviceMem a_m_k_dev_buf(a_m_k.get_element_space_size() * sizeof(ADataType));
        ck_tile::DeviceMem aq_m_aqk_dev_buf(aq_m_aqk.get_element_space_size() * sizeof(QDataType));
        ck_tile::DeviceMem b_k_n_dev_buf(b_k_n.get_element_space_size() * sizeof(BDataType));
        ck_tile::DeviceMem c_m_n_dev_buf(static_cast<size_t>(M) * N * sizeof(CDataType));

        // Copy to device
        if constexpr(std::is_same_v<ADataType, ck_tile::pk_int4_t>)
        {
            // Permute vector pk_i4x4 data for device implementation
            ck_tile::HostTensor<ADataType> temp = a_m_k;
            ck_tile::permute_vectors_i4x4_b(temp);
            a_m_k_dev_buf.ToDevice(temp.data());
        }
        else
        {
            a_m_k_dev_buf.ToDevice(a_m_k.data());
        }
        // aq_m_aqk_dev_buf.ToDevice(aq_m_aqk.data());
        if constexpr(Base::GemmConfig::APreshuffleQuant)
        {
            ck_tile::HostTensor<QDataType> aq_shuffle_host =
                ck_tile::shuffle_aq(&aq_m_aqk, Base::GemmConfig::K_Tile / QuantGroupSize::kK);
            aq_m_aqk_dev_buf.ToDevice(aq_shuffle_host.data());
        }
        else
        {
            aq_m_aqk_dev_buf.ToDevice(aq_m_aqk.data());
        }
        b_k_n_dev_buf.ToDevice(b_k_n.data());

        // Create args for kernel execution
        ck_tile::QuantGemmHostArgs args{
            a_m_k_dev_buf.GetDeviceBuffer(),    // a_ptr
            b_k_n_dev_buf.GetDeviceBuffer(),    // b_ptr
            c_m_n_dev_buf.GetDeviceBuffer(),    // c_ptr
            aq_m_aqk_dev_buf.GetDeviceBuffer(), // aq_ptr (scales)
            nullptr,                            // bq_ptr (not used for AQuant)
            1,                                  // k_batch
            M,
            N,
            K,   // M, N, K
            AQK, // QK_A
            0,   // QK_B (not used for AQuant)
            stride_A,
            stride_B,
            stride_C,
            stride_AQ,
            0 // strides
        };

        // Run the kernel
        ck_tile::stream_config stream_config{};
        this->invoke_quant_gemm(args, stream_config);

        // Validation using reference implementation
        ck_tile::HostTensor<CDataType> c_m_n_host_ref(
            ck_tile::host_tensor_descriptor(M, N, stride_C, this->is_row_major(CLayout{})));
        c_m_n_host_ref.SetZero();

        // Run reference AQuant implementation
        ck_tile::reference_gemm_quant<ADataType,
                                      QDataType,
                                      BDataType,
                                      AccDataType,
                                      CDataType,
                                      QuantGroupSize,
                                      true>(a_m_k, aq_m_aqk, b_k_n, c_m_n_host_ref);

        // Get device result
        ck_tile::HostTensor<CDataType> c_m_n_dev_result(
            ck_tile::host_tensor_descriptor(M, N, stride_C, this->is_row_major(CLayout{})));
        c_m_n_dev_buf.FromDevice(c_m_n_dev_result.mData.data());

        // Calculate error tolerances
        const float max_accumulated_value =
            *std::max_element(c_m_n_host_ref.mData.begin(), c_m_n_host_ref.mData.end());
        const auto rtol_atol =
            this->template calculate_rtol_atol<ADataType, BDataType, AccDataType, CDataType>(
                K, 1, max_accumulated_value);

        // Validate results
        bool pass = ck_tile::check_err(c_m_n_dev_result,
                                       c_m_n_host_ref,
                                       "Error: Incorrect results!",
                                       rtol_atol.at(ck_tile::number<0>{}),
                                       rtol_atol.at(ck_tile::number<1>{}));

        EXPECT_TRUE(pass) << "AQuantGrouped validation failed with M=" << M << ", N=" << N
                          << ", K=" << K;

        if(!pass)
        {
            std::cout << "AQuantGrouped - Relative error threshold: "
                      << rtol_atol.at(ck_tile::number<0>{})
                      << " Absolute error threshold: " << rtol_atol.at(ck_tile::number<1>{})
                      << std::endl;
        }
    }

    private:
    // AQuant-specific pipeline implementation
    template <typename CodegenGemmShape, typename TilePartitioner, typename CodegenGemmTraits>
    void run_quant_gemm_impl(const ck_tile::QuantGemmHostArgs& args,
                             const ck_tile::stream_config& s)
    {
        using GemmPipelineProblem = ck_tile::GemmPipelineProblemBase<ADataType,
                                                                     BDataType,
                                                                     AccDataType,
                                                                     CodegenGemmShape,
                                                                     CodegenGemmTraits,
                                                                     ComputeDataType,
                                                                     ComputeDataType>;

        using BaseGemmPipeline = ck_tile::BaseGemmPipelineAgBgCrCompV3<GemmPipelineProblem>;

        constexpr auto K1 = CodegenGemmShape::WarpTile::at(ck_tile::number<2>{});
        const ck_tile::index_t K_split =
            args.k_batch == 1 ? (args.K + Base::K_Tile - 1) / Base::K_Tile * Base::K_Tile
                              : ck_tile::get_splitk_batch_k_read(args.K, args.k_batch, K1);
        const ck_tile::index_t num_loop    = TilePartitioner::GetLoopNum(K_split);
        const bool has_hot_loop            = BaseGemmPipeline::BlockHasHotloop(num_loop);
        const ck_tile::TailNumber tail_num = BaseGemmPipeline::GetBlockLoopTailNum(num_loop);

        const auto Run = [&](const auto has_hot_loop_, const auto tail_number_) {
            constexpr bool has_hot_loop_v = has_hot_loop_.value;
            constexpr auto tail_number_v  = tail_number_.value;
            constexpr bool transpose_c    = CodegenGemmTraits::TransposeC;

            using PipelineProblem =
                ck_tile::GemmAQuantPipelineProblem<ADataType,
                                                   QDataType,
                                                   BDataType,
                                                   AccDataType,
                                                   CodegenGemmShape,
                                                   CodegenGemmTraits,
                                                   QuantGroupSize,
                                                   transpose_c,
                                                   ComputeDataType,
                                                   ck_tile::GemmPipelineScheduler::Intrawave,
                                                   has_hot_loop_v,
                                                   tail_number_v>;

            using GemmPipeline = ck_tile::AQuantGemmPipelineAgBgCrCompV3<PipelineProblem>;
            using GemmEpilogue = ck_tile::CShuffleEpilogue<
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
                                                 Base::M_Warp,
                                                 Base::N_Warp,
                                                 Base::M_Warp_Tile,
                                                 Base::N_Warp_Tile,
                                                 Base::K_Warp_Tile,
                                                 transpose_c>>;

            using Kernel = ck_tile::QuantGemmKernel<TilePartitioner,
                                                    GemmPipeline,
                                                    GemmEpilogue,
                                                    ck_tile::QuantType::AQuantGrouped>;

            auto kargs        = Kernel::MakeKernelArgs(args);
            const dim3 grids  = Kernel::GridSize(args.M, args.N, args.k_batch);
            const dim3 blocks = Kernel::BlockSize();

            if(!Kernel::IsSupportedArgument(kargs))
            {
                throw std::runtime_error("Arguments not supported for AQuant kernel");
            }

            ck_tile::launch_kernel(s,
                                   ck_tile::make_kernel<GemmConfigBase::kBlockPerCu>(
                                       Kernel{}, grids, blocks, 0, kargs));
        };

        return BaseGemmPipeline::TailHandler(Run, has_hot_loop, tail_num);
    }
};

template <typename Tuple>
class TestCkTileGemmAQuantMem
    : public TestCkTileGemmQuantBase<Tuple, TestCkTileGemmAQuantMem<Tuple>>
{
    using Base = TestCkTileGemmQuantBase<Tuple, TestCkTileGemmAQuantMem<Tuple>>;
    friend Base;

    public:
    using typename Base::AccDataType;
    using typename Base::ADataType;
    using typename Base::ALayout;
    using typename Base::AQLayout;
    using typename Base::BDataType;
    using typename Base::BLayout;
    using typename Base::CDataType;
    using typename Base::CLayout;
    using typename Base::ComputeDataType;
    using typename Base::QDataType;
    using typename Base::QuantGroupSize;
    static constexpr auto QuantType = Base::QuantType;

    protected:
    void SetUpQuantTypeSpecific() {}
    void TearDownQuantTypeSpecific() {}
    // AQuant-specific data generation
    void run_test_with_validation(ck_tile::index_t M, ck_tile::index_t N, ck_tile::index_t K)
    {
        const ck_tile::index_t stride_A =
            ck_tile::get_default_stride(M, K, 0, this->is_row_major(ALayout{}));
        const ck_tile::index_t stride_B =
            ck_tile::get_default_stride(K, N, 0, this->is_row_major(BLayout{}));
        const ck_tile::index_t stride_C =
            ck_tile::get_default_stride(M, N, 0, this->is_row_major(CLayout{}));
        // AQuant uses grouped quantization for A matrix
        const ck_tile::index_t AQK = ck_tile::integer_divide_ceil(K, QuantGroupSize::kK);
        // AQLayout is parameterized in the test tuple (can be RowMajor or ColumnMajor for AQuant)
        const ck_tile::index_t stride_AQ =
            ck_tile::get_default_stride(M, AQK, 0, this->is_row_major(AQLayout{}));
        // Generate test data
        ck_tile::HostTensor<ADataType> a_m_k(
            ck_tile::host_tensor_descriptor(M, K, stride_A, this->is_row_major(ALayout{})));
        // AQLayout is independently specified for each test case
        ck_tile::HostTensor<QDataType> aq_m_aqk(
            ck_tile::host_tensor_descriptor(M, AQK, stride_AQ, this->is_row_major(AQLayout{})));
        ck_tile::HostTensor<BDataType> b_k_n(
            ck_tile::host_tensor_descriptor(K, N, stride_B, this->is_row_major(BLayout{})));
        // Initialize data with random values
        if constexpr(std::is_same_v<ADataType, ck_tile::pk_int4_t>)
        {
            ck_tile::FillUniformDistribution<ADataType>{-5.0f, 5.0f}(a_m_k);
        }
        else
        {
            ck_tile::FillUniformDistribution<ADataType>{-2.0f, 3.0f}(a_m_k);
        }
        ck_tile::FillUniformDistribution<BDataType>{-5.0f, 5.0f}(b_k_n);
        ck_tile::FillUniformDistribution<QDataType>{-2.0f, 2.0f}(aq_m_aqk);
        // Allocate device memory
        ck_tile::DeviceMem a_m_k_dev_buf(a_m_k.get_element_space_size() * sizeof(ADataType));
        ck_tile::DeviceMem aq_m_aqk_dev_buf(aq_m_aqk.get_element_space_size() * sizeof(QDataType));
        ck_tile::DeviceMem b_k_n_dev_buf(b_k_n.get_element_space_size() * sizeof(BDataType));
        ck_tile::DeviceMem c_m_n_dev_buf(static_cast<size_t>(M) * N * sizeof(CDataType));
        // Copy to device
        if constexpr(std::is_same_v<ADataType, ck_tile::pk_int4_t>)
        {
            // Permute vector pk_i4x4 data for device implementation
            ck_tile::HostTensor<ADataType> temp = a_m_k;
            ck_tile::permute_vectors_i4x4_b(temp);
            a_m_k_dev_buf.ToDevice(temp.data());
        }
        else
        {
            a_m_k_dev_buf.ToDevice(a_m_k.data());
        }
        // aq_m_aqk_dev_buf.ToDevice(aq_m_aqk.data());
        if constexpr(Base::GemmConfig::APreshuffleQuant)
        {
            ck_tile::HostTensor<QDataType> aq_shuffle_host =
                ck_tile::shuffle_aq(&aq_m_aqk, Base::GemmConfig::K_Tile / QuantGroupSize::kK);
            aq_m_aqk_dev_buf.ToDevice(aq_shuffle_host.data());
        }
        else
        {
            aq_m_aqk_dev_buf.ToDevice(aq_m_aqk.data());
        }
        b_k_n_dev_buf.ToDevice(b_k_n.data());
        // Create args for kernel execution
        ck_tile::QuantGemmHostArgs args{
            a_m_k_dev_buf.GetDeviceBuffer(),    // a_ptr
            b_k_n_dev_buf.GetDeviceBuffer(),    // b_ptr
            c_m_n_dev_buf.GetDeviceBuffer(),    // c_ptr
            aq_m_aqk_dev_buf.GetDeviceBuffer(), // aq_ptr (scales)
            nullptr,                            // bq_ptr (not used for AQuant)
            1,                                  // k_batch
            M,
            N,
            K,   // M, N, K
            AQK, // QK_A
            0,   // QK_B (not used for AQuant)
            stride_A,
            stride_B,
            stride_C,
            stride_AQ,
            0 // strides
        };
        // Run the kernel
        ck_tile::stream_config stream_config{};
        this->invoke_quant_gemm(args, stream_config);
        // Validation using reference implementation
        ck_tile::HostTensor<CDataType> c_m_n_host_ref(
            ck_tile::host_tensor_descriptor(M, N, stride_C, this->is_row_major(CLayout{})));
        c_m_n_host_ref.SetZero();
        // Run reference AQuant implementation
        ck_tile::reference_gemm_quant<ADataType,
                                      QDataType,
                                      BDataType,
                                      AccDataType,
                                      CDataType,
                                      QuantGroupSize,
                                      true>(a_m_k, aq_m_aqk, b_k_n, c_m_n_host_ref);
        // Get device result
        ck_tile::HostTensor<CDataType> c_m_n_dev_result(
            ck_tile::host_tensor_descriptor(M, N, stride_C, this->is_row_major(CLayout{})));
        c_m_n_dev_buf.FromDevice(c_m_n_dev_result.mData.data());
        // Calculate error tolerances
        const float max_accumulated_value =
            *std::max_element(c_m_n_host_ref.mData.begin(), c_m_n_host_ref.mData.end());
        const auto rtol_atol =
            this->template calculate_rtol_atol<ADataType, BDataType, AccDataType, CDataType>(
                K, 1, max_accumulated_value);
        // Validate results
        bool pass = ck_tile::check_err(c_m_n_dev_result,
                                       c_m_n_host_ref,
                                       "Error: Incorrect results!",
                                       rtol_atol.at(ck_tile::number<0>{}),
                                       rtol_atol.at(ck_tile::number<1>{}));
        EXPECT_TRUE(pass) << "AQuantGrouped validation failed with M=" << M << ", N=" << N
                          << ", K=" << K;
        if(!pass)
        {
            std::cout << "AQuantGrouped - Relative error threshold: "
                      << rtol_atol.at(ck_tile::number<0>{})
                      << " Absolute error threshold: " << rtol_atol.at(ck_tile::number<1>{})
                      << std::endl;
        }
    }

    private:
    // AQuant-specific pipeline implementation
    template <typename CodegenGemmShape, typename TilePartitioner, typename CodegenGemmTraits>
    void run_quant_gemm_impl(const ck_tile::QuantGemmHostArgs& args,
                             const ck_tile::stream_config& s)
    {
        using GemmPipelineProblem = ck_tile::GemmPipelineProblemBase<ADataType,
                                                                     BDataType,
                                                                     AccDataType,
                                                                     CodegenGemmShape,
                                                                     CodegenGemmTraits,
                                                                     ComputeDataType>;
        using BaseGemmPipeline    = ck_tile::BaseGemmPipelineAgBgCrMem<GemmPipelineProblem>;
        constexpr auto K1         = CodegenGemmShape::WarpTile::at(ck_tile::number<2>{});
        const ck_tile::index_t K_split =
            args.k_batch == 1 ? (args.K + Base::K_Tile - 1) / Base::K_Tile * Base::K_Tile
                              : ck_tile::get_splitk_batch_k_read(args.K, args.k_batch, K1);
        const ck_tile::index_t num_loop    = TilePartitioner::GetLoopNum(K_split);
        const bool has_hot_loop            = BaseGemmPipeline::BlockHasHotloop(num_loop);
        const ck_tile::TailNumber tail_num = BaseGemmPipeline::GetBlockLoopTailNum(num_loop);
        const auto Run = [&](const auto has_hot_loop_, const auto tail_number_) {
            constexpr bool has_hot_loop_v = has_hot_loop_.value;
            constexpr auto tail_number_v  = tail_number_.value;
            constexpr bool transpose_c    = CodegenGemmTraits::TransposeC;
            using PipelineProblem         = ck_tile::GemmAQuantPipelineProblem<ADataType,
                                                                               QDataType,
                                                                               BDataType,
                                                                               AccDataType,
                                                                               CodegenGemmShape,
                                                                               CodegenGemmTraits,
                                                                               QuantGroupSize,
                                                                               transpose_c,
                                                                               ComputeDataType,
                                                                               Base::GemmConfig::Scheduler,
                                                                               has_hot_loop_v,
                                                                               tail_number_v>;
            using GemmPipeline            = ck_tile::AQuantGemmPipelineAgBgCrMem<PipelineProblem>;
            using GemmEpilogue            = ck_tile::CShuffleEpilogue<
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
                                                            Base::M_Warp,
                                                            Base::N_Warp,
                                                            Base::M_Warp_Tile,
                                                            Base::N_Warp_Tile,
                                                            Base::K_Warp_Tile,
                                                            transpose_c>>;
            using Kernel      = ck_tile::QuantGemmKernel<TilePartitioner,
                                                         GemmPipeline,
                                                         GemmEpilogue,
                                                         ck_tile::QuantType::AQuantGrouped>;
            auto kargs        = Kernel::MakeKernelArgs(args);
            const dim3 grids  = Kernel::GridSize(args.M, args.N, args.k_batch);
            const dim3 blocks = Kernel::BlockSize();
            if(!Kernel::IsSupportedArgument(kargs))
            {
                throw std::runtime_error("Arguments not supported for AQuant kernel");
            }
            ck_tile::launch_kernel(s,
                                   ck_tile::make_kernel<GemmConfigBase::kBlockPerCu>(
                                       Kernel{}, grids, blocks, 0, kargs));
        };
        return BaseGemmPipeline::TailHandler(Run, has_hot_loop, tail_num);
    }
};

// BQuant-specific test fixture
template <typename Tuple>
class TestCkTileGemmBQuant : public TestCkTileGemmQuantBase<Tuple, TestCkTileGemmBQuant<Tuple>>
{
    using Base = TestCkTileGemmQuantBase<Tuple, TestCkTileGemmBQuant<Tuple>>;
    friend Base;

    public:
    using typename Base::AccDataType;
    using typename Base::ADataType;
    using typename Base::ALayout;
    using typename Base::BDataType;
    using typename Base::BLayout;
    using typename Base::CDataType;
    using typename Base::CLayout;
    using typename Base::ComputeDataType;
    using typename Base::GemmConfig;
    using typename Base::QDataType;
    using typename Base::QuantGroupSize;

    // Re-use AQLayout from tuple parameters as BQLayout
    using BQLayout = typename Base::AQLayout;

    static constexpr auto QuantType        = Base::QuantType;
    static constexpr auto PreshuffleB      = Base::PreshuffleB;
    static constexpr auto TiledMMAPermuteN = Base::TiledMMAPermuteN;

    protected:
    void SetUpQuantTypeSpecific() {}
    void TearDownQuantTypeSpecific() {}

    void run_test_with_validation(ck_tile::index_t M,
                                  ck_tile::index_t N,
                                  ck_tile::index_t K,
                                  ck_tile::index_t k_batch = 1)
    {
        const ck_tile::index_t stride_A = K;
        const ck_tile::index_t stride_B = K;
        const ck_tile::index_t stride_C = N;

        // BQuant uses block/grouped quantization for B matrix
        const ck_tile::index_t BQN       = ck_tile::integer_divide_ceil(N, QuantGroupSize::kN);
        const ck_tile::index_t BQK       = ck_tile::integer_divide_ceil(K, QuantGroupSize::kK);
        const ck_tile::index_t stride_BQ = this->is_row_major(BQLayout{}) ? BQN : BQK;

        // Generate test data
        ck_tile::HostTensor<ADataType> a_m_k(
            ck_tile::host_tensor_descriptor(M, K, stride_A, this->is_row_major(ALayout{})));
        ck_tile::HostTensor<BDataType> b_k_n(
            ck_tile::host_tensor_descriptor(K, N, stride_B, this->is_row_major(BLayout{})));
        ck_tile::HostTensor<QDataType> bq_bqk_bqn(
            ck_tile::host_tensor_descriptor(BQK, BQN, stride_BQ, this->is_row_major(BQLayout{})));

        // Initialize data with random values
        ck_tile::FillUniformDistribution<ADataType>{-0.5f, 0.5f}(a_m_k);
        if constexpr(std::is_same_v<BDataType, ck_tile::pk_fp4_t>)
        {
            ck_tile::FillUniformDistribution<BDataType>{-5.0f, 5.0f}(b_k_n);
        }
        else
        {
            ck_tile::FillUniformDistribution<BDataType>{0.f, 1.f}(b_k_n);
        }

        if constexpr(std::is_same_v<QDataType, ck_tile::e8m0_t>)
        {
            auto gen_scales = [&](auto& scales, float range_min, float range_max) {
                // e8m0_t is basically an exponent of float32
                ck_tile::HostTensor<float> pow2(scales.get_lengths());
                ck_tile::FillUniformDistributionIntegerValue<float>{range_min, range_max}(pow2);
                scales.ForEach([&](auto& self, const auto& i) {
                    self(i) = static_cast<QDataType>(std::exp2(pow2(i)));
                });
            };
            gen_scales(bq_bqk_bqn, -2, 2);
        }
        else
        {
            ck_tile::FillUniformDistribution<QDataType>{-1.0f, 1.0f}(bq_bqk_bqn);
        }

        // Allocate device memory
        ck_tile::DeviceMem a_m_k_dev_buf(a_m_k.get_element_space_size() * sizeof(ADataType));
        ck_tile::DeviceMem b_k_n_dev_buf(b_k_n.get_element_space_size() * sizeof(BDataType));
        ck_tile::DeviceMem bq_bqk_bqn_dev_buf(bq_bqk_bqn.get_element_space_size() *
                                              sizeof(QDataType));
        ck_tile::DeviceMem c_m_n_dev_buf(static_cast<size_t>(M) * N * sizeof(CDataType));

        // Zero C buffer - required for split-K atomic_add accumulation
        c_m_n_dev_buf.SetZero();

        // Copy to device
        a_m_k_dev_buf.ToDevice(a_m_k.data());
        ck_tile::HostTensor<BDataType> b_k_n_dev = b_k_n;
        if constexpr(PreshuffleB)
        {
            if constexpr(TiledMMAPermuteN && QuantGroupSize::kN == 1)
            {
                printf("PreshuffleB with TiledMMAPermuteN\n");
                b_k_n_dev = ck_tile::shuffle_b_permuteN<GemmConfig>(b_k_n);
            }
            else
            {
                printf("PreshuffleB without TiledMMAPermuteN\n");
                b_k_n_dev = ck_tile::shuffle_b_v0<GemmConfig>(b_k_n);
            }
        }
        if constexpr(std::is_same_v<BDataType, ck_tile::pk_int4_t>)
        {
            ck_tile::permute_vectors_i4x4_b(b_k_n_dev);
        }

        b_k_n_dev_buf.ToDevice(b_k_n_dev.data());

        if constexpr(PreshuffleB && TiledMMAPermuteN && QuantGroupSize::kN == 1)
        {
            printf("Preshuffle BQ with TiledMMAPermuteN \n");
            ck_tile::HostTensor<QDataType> bq_shuffle_host =
                ck_tile::bq_permuteN<GemmConfig>(bq_bqk_bqn, QuantGroupSize::kN);
            bq_bqk_bqn_dev_buf.ToDevice(bq_shuffle_host.data());
        }
        else if constexpr(GemmConfig::BPreshuffleQuant)
        {
            ck_tile::HostTensor<QDataType> bq_shuffle_host =
                ck_tile::shuffle_bq(&bq_bqk_bqn, GemmConfig::K_Tile / QuantGroupSize::kK);
            bq_bqk_bqn_dev_buf.ToDevice(bq_shuffle_host.data());
        }
        else
        {
            bq_bqk_bqn_dev_buf.ToDevice(bq_bqk_bqn.data());
        }

        // Create args for kernel execution
        ck_tile::QuantGemmHostArgs args{
            a_m_k_dev_buf.GetDeviceBuffer(),      // a_ptr
            b_k_n_dev_buf.GetDeviceBuffer(),      // b_ptr
            c_m_n_dev_buf.GetDeviceBuffer(),      // c_ptr
            nullptr,                              // aq_ptr (not used for BQuant)
            bq_bqk_bqn_dev_buf.GetDeviceBuffer(), // bq_ptr (scales)
            k_batch,                              // k_batch (split-K)
            M,
            N,
            K,   // M, N, K
            0,   // QK_A (not used for BQuant)
            BQK, // QK_B
            stride_A,
            stride_B,
            stride_C,
            0,
            stride_BQ // strides
        };

        // Run the kernel
        ck_tile::stream_config stream_config{};
        this->invoke_quant_gemm(args, stream_config);

        // Validation using reference implementation
        ck_tile::HostTensor<CDataType> c_m_n_host_ref(
            ck_tile::host_tensor_descriptor(M, N, stride_C, this->is_row_major(CLayout{})));
        c_m_n_host_ref.SetZero();

        // Run reference BQuant implementation
        if constexpr(std::is_same_v<QDataType, ck_tile::e8m0_t>)
            ck_tile::reference_mx_gemm_bquant<ADataType,
                                              QDataType,
                                              BDataType,
                                              AccDataType,
                                              CDataType,
                                              QuantGroupSize,
                                              BLayout,
                                              false>(a_m_k, bq_bqk_bqn, b_k_n, c_m_n_host_ref);
        else
            ck_tile::reference_gemm_quant<ADataType,
                                          QDataType,
                                          BDataType,
                                          AccDataType,
                                          CDataType,
                                          QuantGroupSize,
                                          false>(a_m_k, bq_bqk_bqn, b_k_n, c_m_n_host_ref);

        // Get device result
        ck_tile::HostTensor<CDataType> c_m_n_dev_result(
            ck_tile::host_tensor_descriptor(M, N, stride_C, this->is_row_major(CLayout{})));
        c_m_n_dev_buf.FromDevice(c_m_n_dev_result.mData.data());

        // Calculate error tolerances
        const float max_accumulated_value =
            *std::max_element(c_m_n_host_ref.mData.begin(), c_m_n_host_ref.mData.end());
        const auto rtol_atol =
            this->template calculate_rtol_atol<ADataType, BDataType, AccDataType, CDataType>(
                K, k_batch, max_accumulated_value);

        // Validate results
        bool pass = ck_tile::check_err(c_m_n_dev_result,
                                       c_m_n_host_ref,
                                       "Error: Incorrect results!",
                                       rtol_atol.at(ck_tile::number<0>{}),
                                       rtol_atol.at(ck_tile::number<1>{}));

        EXPECT_TRUE(pass) << "BQuantGrouped validation failed with M=" << M << ", N=" << N
                          << ", K=" << K << ", k_batch=" << k_batch;

        if(!pass)
        {
            std::cout << "BQuantGrouped - Relative error threshold: "
                      << rtol_atol.at(ck_tile::number<0>{})
                      << " Absolute error threshold: " << rtol_atol.at(ck_tile::number<1>{})
                      << std::endl;
        }
    }

    // Boundary correctness check for the BQuant large-tensor (64-bit global load/store) path with
    // PreshuffleB.  A host GEMM reference and a host preshuffle of B are both infeasible at the
    // >2 GiB B scales this path targets (reference_permute over N*K elements int32-overflows its
    // element count).  Instead B is filled directly in raw device-buffer offset order (which the
    // kernel's flat B view interprets as the preshuffled weight) with a value that is a pure step
    // function of the raw element offset: flat[i] = (i >= 2^31) ? V_FAR : V_NEAR.  Because the
    // value depends only on the raw offset, the intra-region (k, n) interleave is irrelevant and
    // no host permute is needed.  A is zero except at K-index hot_k = K-1, where A(m, hot_k) = m%8,
    // with unit B dequant scales, so C(m, n) = (m % 8) * B_flat(hot_k, n).  Because hot_k = K-1 has
    // the maximal flat-K index, B(hot_k, n) lands in the far region (offset >= 2^31) exactly when
    // n >= N - N_Warp_Tile, giving C(m, n) = (m % 8) * (n >= N - N_Warp_Tile ? V_FAR : V_NEAR).
    // A 32-bit offset overflow would wrap a far read into the near region (V_NEAR instead of V_FAR)
    // or fault, so the far spot-checks distinguish correct 64-bit from buggy 32-bit addressing,
    // while the near spot-checks (which never overflow) act as controls.  Assumes a 1-byte B type
    // (flat element offset == byte offset, matching the byte-based 2^31 gate) and N_Warp_Tile == 16
    // (the far region aligns to the last n-block); both are asserted below.
    void run_test_boundary_check_bquant(
        ck_tile::index_t M,
        ck_tile::index_t N,
        ck_tile::index_t K,
        ck_tile::index_t hot_k,
        const std::vector<std::pair<ck_tile::index_t, ck_tile::index_t>>& spot_checks)
    {
        static_assert(sizeof(BDataType) == 1,
                      "run_test_boundary_check_bquant assumes a 1-byte B element type");
        ASSERT_GE(hot_k, 0);
        ASSERT_LT(hot_k, K);
        ASSERT_EQ(hot_k, K - 1) << "far/near expectation assumes hot_k == K-1 (max flat-K index)";
        ASSERT_EQ(static_cast<int>(GemmConfig::N_Warp_Tile), 16)
            << "far/near boundary derivation assumes N_Warp_Tile == 16";

        constexpr float V_NEAR = 1.0f;
        constexpr float V_FAR  = 7.0f;

        const ck_tile::index_t stride_A = K;
        const ck_tile::index_t stride_B = K;
        const ck_tile::index_t stride_C = N;

        const ck_tile::index_t BQN       = ck_tile::integer_divide_ceil(N, QuantGroupSize::kN);
        const ck_tile::index_t BQK       = ck_tile::integer_divide_ceil(K, QuantGroupSize::kK);
        const ck_tile::index_t stride_BQ = this->is_row_major(BQLayout{}) ? BQN : BQK;

        // Device-memory guard: B alone exceeds 2 GiB at the trigger shape.
        {
            size_t free_mem = 0, total_mem = 0;
            ASSERT_EQ(hipMemGetInfo(&free_mem, &total_mem), hipSuccess);
            const size_t required = static_cast<size_t>(M) * N * sizeof(CDataType) +
                                    static_cast<size_t>(M) * K * sizeof(ADataType) +
                                    static_cast<size_t>(K) * N * sizeof(BDataType) +
                                    static_cast<size_t>(BQK) * BQN * sizeof(QDataType);
            if(free_mem < required + (size_t{256} << 20)) // 256 MiB headroom
            {
                GTEST_SKIP() << "Insufficient device memory for BQuant large-tensor boundary "
                             << "check: need ~" << ((required >> 20) + 256) << " MiB, have "
                             << (free_mem >> 20) << " MiB free";
            }
        }

        ck_tile::HostTensor<ADataType> a_m_k(
            ck_tile::host_tensor_descriptor(M, K, stride_A, this->is_row_major(ALayout{})));
        ck_tile::HostTensor<BDataType> b_k_n(
            ck_tile::host_tensor_descriptor(K, N, stride_B, this->is_row_major(BLayout{})));
        ck_tile::HostTensor<QDataType> bq_bqk_bqn(
            ck_tile::host_tensor_descriptor(BQK, BQN, stride_BQ, this->is_row_major(BQLayout{})));

        // A is zero except column hot_k, where A(m, hot_k) = m % 8.
        a_m_k.SetZero();
        for(ck_tile::index_t m = 0; m < M; ++m)
        {
            a_m_k(m, hot_k) = ck_tile::type_convert<ADataType>(static_cast<float>(m % 8));
        }
        // Fill B directly in raw buffer-offset order: near region (offset < 2^31) = V_NEAR, far
        // region (>= 2^31) = V_FAR.  split is in elements and equals 2^31 for the 1-byte B above.
        {
            const size_t total = b_k_n.get_element_space_size();
            const size_t split = (size_t{1} << 31) / sizeof(BDataType);
            auto* bp           = b_k_n.data();
            std::fill(bp, bp + std::min(split, total), ck_tile::type_convert<BDataType>(V_NEAR));
            if(total > split)
            {
                std::fill(bp + split, bp + total, ck_tile::type_convert<BDataType>(V_FAR));
            }
        }
        // Unit dequant scales keep B_dequant == B.
        std::fill(bq_bqk_bqn.begin(), bq_bqk_bqn.end(), ck_tile::type_convert<QDataType>(1.0f));

        ck_tile::DeviceMem a_m_k_dev_buf(a_m_k.get_element_space_size() * sizeof(ADataType));
        ck_tile::DeviceMem b_k_n_dev_buf(b_k_n.get_element_space_size() * sizeof(BDataType));
        ck_tile::DeviceMem bq_bqk_bqn_dev_buf(bq_bqk_bqn.get_element_space_size() *
                                              sizeof(QDataType));
        ck_tile::DeviceMem c_m_n_dev_buf(static_cast<size_t>(M) * N * sizeof(CDataType));

        c_m_n_dev_buf.SetZero();

        a_m_k_dev_buf.ToDevice(a_m_k.data());
        b_k_n_dev_buf.ToDevice(b_k_n.data());
        bq_bqk_bqn_dev_buf.ToDevice(bq_bqk_bqn.data());

        ck_tile::QuantGemmHostArgs args{
            a_m_k_dev_buf.GetDeviceBuffer(),      // a_ptr
            b_k_n_dev_buf.GetDeviceBuffer(),      // b_ptr
            c_m_n_dev_buf.GetDeviceBuffer(),      // c_ptr
            nullptr,                              // aq_ptr (not used for BQuant)
            bq_bqk_bqn_dev_buf.GetDeviceBuffer(), // bq_ptr (scales)
            1,                                    // k_batch
            M,
            N,
            K,   // M, N, K
            0,   // QK_A (not used for BQuant)
            BQK, // QK_B
            stride_A,
            stride_B,
            stride_C,
            0,
            stride_BQ // strides
        };

        ck_tile::stream_config stream_config{};
        this->invoke_quant_gemm(args, stream_config);
        ASSERT_EQ(hipGetLastError(), hipSuccess)
            << "Kernel launch failed for the BQuant large-tensor boundary check (M=" << M
            << ", N=" << N << ", K=" << K << ")";
        ASSERT_EQ(hipStreamSynchronize(stream_config.stream_id_), hipSuccess)
            << "Device-side fault while executing the BQuant large-tensor boundary check (M=" << M
            << ", N=" << N << ", K=" << K << ")";

        // C is small here (M*N), so read back only the requested elements via targeted reads.
        const auto c_desc =
            ck_tile::host_tensor_descriptor(M, N, stride_C, this->is_row_major(CLayout{}));
        for(const auto& [m, n] : spot_checks)
        {
            ASSERT_GE(m, 0);
            ASSERT_LT(m, M);
            ASSERT_GE(n, 0);
            ASSERT_LT(n, N);

            const auto byte_offset = c_desc.GetOffsetFromMultiIndex(m, n) * sizeof(CDataType);

            CDataType actual{};
            ASSERT_EQ(
                hipMemcpy(&actual,
                          static_cast<const char*>(c_m_n_dev_buf.GetDeviceBuffer()) + byte_offset,
                          sizeof(CDataType),
                          hipMemcpyDeviceToHost),
                hipSuccess)
                << "Failed to read back C(" << m << ", " << n << ")";

            // hot_k == K-1 has the maximal flat-K index, so B(hot_k, n) is in the far region
            // (flat offset >= 2^31) exactly when n >= N - N_Warp_Tile.
            const bool is_far = n >= (N - static_cast<ck_tile::index_t>(GemmConfig::N_Warp_Tile));
            const float expected = static_cast<float>(m % 8) * (is_far ? V_FAR : V_NEAR);
            EXPECT_NEAR(ck_tile::type_convert<float>(actual), expected, 1e-3f)
                << "BQuant boundary-check mismatch at C(" << m << ", " << n << ") (M=" << M
                << ", N=" << N << ", K=" << K << ", hot_k=" << hot_k << ", far=" << is_far << ")";
        }
    }

    private:
    // BQuant-specific pipeline implementation
    template <typename CodegenGemmShape, typename TilePartitioner, typename CodegenGemmTraits>
    void run_quant_gemm_impl(const ck_tile::QuantGemmHostArgs& args,
                             const ck_tile::stream_config& s)
    {
        using GemmPipelineProblem = ck_tile::GemmPipelineProblemBase<ADataType,
                                                                     BDataType,
                                                                     AccDataType,
                                                                     CodegenGemmShape,
                                                                     CodegenGemmTraits,
                                                                     ComputeDataType,
                                                                     ComputeDataType>;

        using BaseGemmPipeline = std::conditional_t<
            PreshuffleB == false,
            ck_tile::BaseGemmPipelineAgBgCrCompV3<GemmPipelineProblem>,
            ck_tile::BaseWeightPreshufflePipelineAGmemBGmemCRegV2<GemmPipelineProblem>>;

        constexpr auto K1 = CodegenGemmShape::WarpTile::at(ck_tile::number<2>{});
        const ck_tile::index_t K_split =
            args.k_batch == 1 ? (args.K + Base::K_Tile - 1) / Base::K_Tile * Base::K_Tile
                              : ck_tile::get_splitk_batch_k_read(args.K, args.k_batch, K1);
        const ck_tile::index_t num_loop    = TilePartitioner::GetLoopNum(K_split);
        const bool has_hot_loop            = BaseGemmPipeline::BlockHasHotloop(num_loop);
        const ck_tile::TailNumber tail_num = BaseGemmPipeline::GetBlockLoopTailNum(num_loop);

        const auto Run = [&](const auto has_hot_loop_, const auto tail_number_) {
            constexpr bool has_hot_loop_v  = has_hot_loop_.value;
            constexpr auto tail_number_v   = tail_number_.value;
            constexpr auto b_cast_policy_v = std::is_same_v<ADataType, BDataType>
                                                 ? ck_tile::CastPolicy::BeforeLDSWrite
                                                 : ck_tile::CastPolicy::AfterLDSRead;

            using PipelineProblem =
                ck_tile::GemmBQuantPipelineProblem<ADataType,
                                                   BDataType,
                                                   QDataType,
                                                   AccDataType,
                                                   CodegenGemmShape,
                                                   CodegenGemmTraits,
                                                   QuantGroupSize,
                                                   ComputeDataType,
                                                   ck_tile::GemmPipelineScheduler::Intrawave,
                                                   has_hot_loop_v,
                                                   tail_number_v,
                                                   b_cast_policy_v>;

            using GemmPipeline = std::conditional_t<
                PreshuffleB == false,
                std::conditional_t<std::is_same_v<QDataType, ck_tile::e8m0_t>,
                                   ck_tile::MicroscaleGemmPipelineAgBgCrCompV3<PipelineProblem>,
                                   ck_tile::BQuantGemmPipelineAgBgCrCompV3<PipelineProblem>>,
                ck_tile::WPQuantBPipelineAgBgCrV2<PipelineProblem>>;

            // clang-format off
            using BTypeForEpilogue =
                std::conditional_t<std::is_same_v<BDataType, ck_tile::pk_fp4_t>, ADataType, BDataType>;
            // clang-format on

            using GemmEpilogue = std::conditional_t<
                TiledMMAPermuteN,
                ck_tile::PermuteNEpilogue<
                    ck_tile::PermuteNEpilogueProblem<ADataType,
                                                     BTypeForEpilogue,
                                                     ck_tile::tuple<>,
                                                     AccDataType,
                                                     CDataType,
                                                     ck_tile::tuple<>,
                                                     CLayout,
                                                     ck_tile::element_wise::PassThrough,
                                                     TilePartitioner::MPerBlock,
                                                     TilePartitioner::NPerBlock,
                                                     Base::M_Warp,
                                                     Base::N_Warp,
                                                     Base::M_Warp_Tile,
                                                     Base::N_Warp_Tile,
                                                     Base::K_Warp_Tile,
                                                     false, // transpose_c
                                                     false,
                                                     1>>,
                ck_tile::CShuffleEpilogue<
                    ck_tile::CShuffleEpilogueProblem<ADataType,
                                                     BTypeForEpilogue,
                                                     ck_tile::tuple<>,
                                                     AccDataType,
                                                     CDataType,
                                                     ck_tile::tuple<>,
                                                     CLayout,
                                                     ck_tile::element_wise::PassThrough,
                                                     TilePartitioner::MPerBlock,
                                                     TilePartitioner::NPerBlock,
                                                     Base::M_Warp,
                                                     Base::N_Warp,
                                                     Base::M_Warp_Tile,
                                                     Base::N_Warp_Tile,
                                                     Base::K_Warp_Tile,
                                                     false>>>; // transpose_c

            using Kernel = ck_tile::QuantGemmKernel<TilePartitioner,
                                                    GemmPipeline,
                                                    GemmEpilogue,
                                                    ck_tile::QuantType::BQuantGrouped>;

            auto kargs        = Kernel::MakeKernelArgs(args);
            const dim3 grids  = Kernel::GridSize(args.M, args.N, args.k_batch);
            const dim3 blocks = Kernel::BlockSize();

            if(!Kernel::IsSupportedArgument(kargs))
            {
                throw std::runtime_error("Arguments not supported for BQuant kernel");
            }

            ck_tile::launch_kernel(s,
                                   ck_tile::make_kernel<GemmConfigBase::kBlockPerCu>(
                                       Kernel{}, grids, blocks, 0, kargs));
        };

        return BaseGemmPipeline::TailHandler(Run, has_hot_loop, tail_num);
    }
};

// ABQuant-specific test fixture
template <typename Tuple>
class TestCkTileGemmABQuant : public TestCkTileGemmQuantBase<Tuple, TestCkTileGemmABQuant<Tuple>>
{
    using Base = TestCkTileGemmQuantBase<Tuple, TestCkTileGemmABQuant<Tuple>>;
    friend Base;

    public:
    using typename Base::AccDataType;
    using typename Base::ADataType;
    using typename Base::ALayout;
    using typename Base::AQLayout;
    using typename Base::AQuantGroupSize;
    using typename Base::BDataType;
    using typename Base::BLayout;
    using typename Base::BQuantGroupSize;
    using typename Base::CDataType;
    using typename Base::CLayout;
    using typename Base::ComputeDataType;
    using typename Base::GemmConfig;
    using typename Base::QDataType;
    using BQLayout = ck_tile::tensor_layout::gemm::ColumnMajor;

    static constexpr auto QuantType        = Base::QuantType;
    static constexpr auto PreshuffleB      = Base::PreshuffleB;
    static constexpr auto TiledMMAPermuteN = Base::TiledMMAPermuteN;

    protected:
    // Opts the host B preshuffle into the packing-aware granularity so that it matches the
    // device-side KBPerLoad. Both sides resolve the type through ck_tile::b_shuffle_type_t
    // and the granularity through ck_tile::items_per_128b_access, so neither can drift.
    // K_Warp_Tile is restated because Base overrides GemmConfig's under CK_USE_GFX1250;
    // N_Warp_Tile is restated only for symmetry -- Base takes it from GemmConfig unchanged.
    using BShuffleType = ck_tile::b_shuffle_type_t<ADataType, BDataType>;

    struct BShuffleConfig : public GemmConfig
    {
        static constexpr ck_tile::index_t N_Warp_Tile = Base::N_Warp_Tile;
        static constexpr ck_tile::index_t K_Warp_Tile = Base::K_Warp_Tile;

        static constexpr ck_tile::index_t BContiguousItemsPerAccess =
            ck_tile::items_per_128b_access<BShuffleType>();
    };

    void SetUpQuantTypeSpecific() {}
    void TearDownQuantTypeSpecific() {}

    void run_test_with_validation(ck_tile::index_t M,
                                  ck_tile::index_t N,
                                  ck_tile::index_t K,
                                  ck_tile::index_t k_batch       = 1,
                                  ck_tile::index_t stride_B_pad  = 0,
                                  bool allow_runtime_splitk_tail = false)
    {
        const ck_tile::index_t stride_A =
            ck_tile::get_default_stride(M, K, 0, this->is_row_major(ALayout{}));
        // stride_B_pad lets a test exercise a B tensor whose leading-dim stride is
        // larger than the packed value (e.g. row-aligned padding). The host tensor,
        // device buffer, and kernel args are all built with this padded stride so
        // the kernel must honor the runtime stride to address B correctly.
        const ck_tile::index_t stride_B =
            ck_tile::get_default_stride(K, N, 0, this->is_row_major(BLayout{})) + stride_B_pad;
        const ck_tile::index_t stride_C =
            ck_tile::get_default_stride(M, N, 0, this->is_row_major(CLayout{}));

        // AQuant uses grouped quantization for A matrix
        const ck_tile::index_t AQK = ck_tile::integer_divide_ceil(K, AQuantGroupSize::kK);
        // BQuant uses block/grouped quantization for B matrix
        const ck_tile::index_t BQN = ck_tile::integer_divide_ceil(N, BQuantGroupSize::kN);
        const ck_tile::index_t BQK = ck_tile::integer_divide_ceil(K, BQuantGroupSize::kK);
        const ck_tile::index_t stride_AQ =
            ck_tile::get_default_stride(M, AQK, 0, this->is_row_major(AQLayout{}));
        const ck_tile::index_t stride_BQ =
            ck_tile::get_default_stride(BQK, BQN, 0, this->is_row_major(BQLayout{}));
        // Generate test data
        ck_tile::HostTensor<ADataType> a_m_k(
            ck_tile::host_tensor_descriptor(M, K, stride_A, this->is_row_major(ALayout{})));
        // AQLayout is independently specified for each test case
        ck_tile::HostTensor<QDataType> aq_m_aqk( // AQDataType
            ck_tile::host_tensor_descriptor(M, AQK, stride_AQ, this->is_row_major(AQLayout{})));
        ck_tile::HostTensor<BDataType> b_k_n(
            ck_tile::host_tensor_descriptor(K, N, stride_B, this->is_row_major(BLayout{})));
        ck_tile::HostTensor<QDataType> bq_bqk_bqn(
            ck_tile::host_tensor_descriptor(BQK, BQN, stride_BQ, this->is_row_major(BQLayout{})));

        // Initialize data with random values
        if constexpr(std::is_same_v<ADataType, ck_tile::pk_int4_t>)
        {
            ck_tile::FillUniformDistribution<ADataType>{-5.0f, 5.0f}(a_m_k);
        }
        else
        {
            ck_tile::FillUniformDistribution<ADataType>{-2.0f, 3.0f}(a_m_k);
        }
        ck_tile::FillUniformDistribution<BDataType>{-5.0f, 5.0f}(b_k_n);
        ck_tile::FillUniformDistribution<QDataType>{-2.0f, 2.0f}(aq_m_aqk);
        ck_tile::FillUniformDistribution<QDataType>{-2.0f, 2.0f}(bq_bqk_bqn);
        // Allocate device memory
        ck_tile::DeviceMem a_m_k_dev_buf(a_m_k.get_element_space_size() * sizeof(ADataType));
        ck_tile::DeviceMem aq_m_aqk_dev_buf(aq_m_aqk.get_element_space_size() *
                                            sizeof(QDataType)); // AQDataType
        ck_tile::DeviceMem b_k_n_dev_buf(b_k_n.get_element_space_size() * sizeof(BDataType));
        ck_tile::DeviceMem bq_bqk_bqn_dev_buf(bq_bqk_bqn.get_element_space_size() *
                                              sizeof(QDataType));
        ck_tile::DeviceMem c_m_n_dev_buf(static_cast<size_t>(M) * N * sizeof(CDataType));

        // Copy to device
        if constexpr(std::is_same_v<ADataType, ck_tile::pk_int4_t>)
        {
            // Permute vector pk_i4x4 data for device implementation
            ck_tile::HostTensor<ADataType> temp = a_m_k;
            ck_tile::permute_vectors_i4x4_b(temp);
            a_m_k_dev_buf.ToDevice(temp.data());
        }
        else
        {
            a_m_k_dev_buf.ToDevice(a_m_k.data());
        }
        ck_tile::HostTensor<BDataType> b_k_n_dev = b_k_n;
        if constexpr(PreshuffleB)
        {
            if constexpr(TiledMMAPermuteN && BQuantGroupSize::kN == 1)
            {
                // Not BShuffleConfig: shuffle_b_permuteN's gfx12 path hardcodes its own
                // granularity, and no packed-4-bit config reaches the non-gfx12 path
                // (they all have BQuantGroupSize::kN != 1).
                printf("PreshuffleB with TiledMMAPermuteN\n");
                b_k_n_dev = ck_tile::shuffle_b_permuteN<GemmConfig>(b_k_n);
            }
            else
            {
                printf("PreshuffleB without TiledMMAPermuteN\n");
                b_k_n_dev = ck_tile::shuffle_b_v0<BShuffleConfig>(b_k_n);
            }
        }
        if constexpr(std::is_same_v<BDataType, ck_tile::pk_int4_t>)
        {
            ck_tile::permute_vectors_i4x4_b(b_k_n_dev);
        }

        b_k_n_dev_buf.ToDevice(b_k_n_dev.data());

        if constexpr(Base::GemmConfig::APreshuffleQuant)
        {
            ck_tile::HostTensor<QDataType> aq_shuffle_host =
                ck_tile::shuffle_aq(&aq_m_aqk, Base::GemmConfig::K_Tile / AQuantGroupSize::kK);
            aq_m_aqk_dev_buf.ToDevice(aq_shuffle_host.data());
        }
        else
        {
            aq_m_aqk_dev_buf.ToDevice(aq_m_aqk.data());
        }
        if constexpr(PreshuffleB && TiledMMAPermuteN && BQuantGroupSize::kN == 1)
        {
            printf("Preshuffle BQ with TiledMMAPermuteN \n");
            ck_tile::HostTensor<QDataType> bq_shuffle_host =
                ck_tile::bq_permuteN<GemmConfig>(bq_bqk_bqn, BQuantGroupSize::kN);
            bq_bqk_bqn_dev_buf.ToDevice(bq_shuffle_host.data());
        }
        else if constexpr(GemmConfig::BPreshuffleQuant)
        {
            ck_tile::HostTensor<QDataType> bq_shuffle_host =
                ck_tile::shuffle_bq(&bq_bqk_bqn, GemmConfig::K_Tile / BQuantGroupSize::kK);
            bq_bqk_bqn_dev_buf.ToDevice(bq_shuffle_host.data());
        }
        else
        {
            bq_bqk_bqn_dev_buf.ToDevice(bq_bqk_bqn.data());
        }

        // For split-K (k_batch > 1), the kernel uses atomic_add to accumulate partial results
        // into C. Zero the output buffer before launching so atomic additions start from zero.
        if(k_batch > 1)
        {
            c_m_n_dev_buf.SetZero();
        }

        // Create args for kernel execution
        ck_tile::QuantGemmHostArgs args{
            a_m_k_dev_buf.GetDeviceBuffer(),      // a_ptr
            b_k_n_dev_buf.GetDeviceBuffer(),      // b_ptr
            c_m_n_dev_buf.GetDeviceBuffer(),      // c_ptr
            aq_m_aqk_dev_buf.GetDeviceBuffer(),   // aq_ptr (scales)
            bq_bqk_bqn_dev_buf.GetDeviceBuffer(), // bq_ptr (scales)
            k_batch,                              // k_batch
            M,
            N,
            K,   // M, N, K
            AQK, // QK_A
            BQK, // QK_B
            stride_A,
            stride_B,
            stride_C,
            stride_AQ,
            stride_BQ // strides
        };

        // Run the kernel
        ck_tile::stream_config stream_config{};
        this->invoke_quant_gemm(args, stream_config, allow_runtime_splitk_tail);

        // Validation using reference implementation
        ck_tile::HostTensor<CDataType> c_m_n_host_ref(
            ck_tile::host_tensor_descriptor(M, N, stride_C, this->is_row_major(CLayout{})));
        c_m_n_host_ref.SetZero();

        // Run reference ABQuant implementation
        ck_tile::reference_gemm_abquant<ADataType,
                                        QDataType, // AQDataType
                                        BDataType,
                                        QDataType,
                                        AccDataType,
                                        CDataType,
                                        AQuantGroupSize,
                                        BQuantGroupSize>(
            a_m_k, aq_m_aqk, b_k_n, bq_bqk_bqn, c_m_n_host_ref);

        // Get device result
        ck_tile::HostTensor<CDataType> c_m_n_dev_result(
            ck_tile::host_tensor_descriptor(M, N, stride_C, this->is_row_major(CLayout{})));
        c_m_n_dev_buf.FromDevice(c_m_n_dev_result.mData.data());

        // Calculate error tolerances (adjusted for split-K accumulation error)
        const float max_accumulated_value =
            *std::max_element(c_m_n_host_ref.mData.begin(), c_m_n_host_ref.mData.end());
        const auto rtol_atol =
            this->template calculate_rtol_atol<ADataType, BDataType, AccDataType, CDataType>(
                K, k_batch, max_accumulated_value);

        // Validate results
        bool pass = ck_tile::check_err(c_m_n_dev_result,
                                       c_m_n_host_ref,
                                       "Error: Incorrect results!",
                                       rtol_atol.at(ck_tile::number<0>{}),
                                       rtol_atol.at(ck_tile::number<1>{}));

        EXPECT_TRUE(pass) << "ABQuantGrouped validation failed with M=" << M << ", N=" << N
                          << ", K=" << K << ", k_batch=" << k_batch;

        if(!pass)
        {
            std::cout << "ABQuantGrouped - Relative error threshold: "
                      << rtol_atol.at(ck_tile::number<0>{})
                      << " Absolute error threshold: " << rtol_atol.at(ck_tile::number<1>{})
                      << std::endl;
        }
    }

    public:
    // ABQuant-specific pipeline implementation.  Public so the
    // has_run_quant_gemm_impl_splitk SFINAE trait in
    // test_gemm_quant_base.hpp can detect this 3-arg overload from outside
    // the class (the trait lives in a different namespace and is not a
    // friend of this fixture).
    template <typename CodegenGemmShape, typename TilePartitioner, typename CodegenGemmTraits>
    void run_quant_gemm_impl(const ck_tile::QuantGemmHostArgs& args,
                             const ck_tile::stream_config& s,
                             bool allow_runtime_splitk_tail)
    {

        static_assert(std::is_same_v<CLayout, ck_tile::tensor_layout::gemm::RowMajor>);
        constexpr bool IS_FP8BLOCKSCALE = BQuantGroupSize::kN == 128 &&
                                          (std::is_same_v<ADataType, ck_tile::fp8_t> ||
                                           std::is_same_v<ADataType, ck_tile::bf8_t>) &&
                                          (std::is_same_v<BDataType, ck_tile::fp8_t> ||
                                           std::is_same_v<BDataType, ck_tile::bf8_t>);
        constexpr bool transpose_c = CodegenGemmTraits::TransposeC;
        constexpr bool eight_waves =
#ifdef CK_GFX950_SUPPORT
            IS_FP8BLOCKSCALE &&
            (GemmConfig::M_Warp * GemmConfig::N_Warp * GemmConfig::K_Warp == 8) &&
            GemmConfig::K_Warp_Tile == 128;
#else
            false;
#endif
        using GemmPipelineProblem = ck_tile::GemmPipelineProblemBase<ADataType,
                                                                     BDataType,
                                                                     AccDataType,
                                                                     CodegenGemmShape,
                                                                     CodegenGemmTraits,
                                                                     ComputeDataType,
                                                                     ComputeDataType>;

        constexpr auto base_gemm_pipeline = []() {
            if constexpr(eight_waves)
                return ck_tile::BaseGemmPipelineAgBgCrCompV3<GemmPipelineProblem>{};
            else if constexpr(PreshuffleB)
                return ck_tile::BaseWeightPreshufflePipelineAGmemBGmemCRegV2<GemmPipelineProblem>{};
            else if constexpr(IS_FP8BLOCKSCALE)
                return ck_tile::BaseGemmPipelineAgBgCrCompV3<GemmPipelineProblem>{};
            else
                return ck_tile::BaseGemmPipelineAgBgCrCompV3<GemmPipelineProblem>{};
        }();
        using BaseGemmPipeline = std::decay_t<decltype(base_gemm_pipeline)>;

        constexpr auto K1 = CodegenGemmShape::WarpTile::at(ck_tile::number<2>{});
        const ck_tile::index_t K_split =
            args.k_batch == 1 ? ck_tile::integer_least_multiple(args.K, GemmConfig::K_Tile)
                              : ck_tile::get_splitk_batch_k_read(args.K, args.k_batch, K1);
        const ck_tile::index_t num_loop    = TilePartitioner::GetLoopNum(K_split);
        const bool has_hot_loop            = BaseGemmPipeline::BlockHasHotloop(num_loop);
        const ck_tile::TailNumber tail_num = BaseGemmPipeline::GetBlockLoopTailNum(num_loop);

        const auto Run = [&](const auto has_hot_loop_, const auto tail_number_) {
            constexpr bool has_hot_loop_v = has_hot_loop_.value;
            constexpr auto tail_number_v  = tail_number_.value;

            using PipelineProblem =
                ck_tile::GemmABQuantPipelineProblem<ADataType,
                                                    QDataType, // AQDataType
                                                    BDataType,
                                                    QDataType,
                                                    AccDataType,
                                                    CodegenGemmShape,
                                                    CodegenGemmTraits,
                                                    AQuantGroupSize,
                                                    BQuantGroupSize,
                                                    transpose_c,
                                                    ComputeDataType,
                                                    ck_tile::GemmPipelineScheduler::Intrawave,
                                                    has_hot_loop_v,
                                                    tail_number_v>;

            using GemmPipeline = std::conditional_t<
                eight_waves,
                ck_tile::ABQuantGemmPipelineAgBgCrEightWaves<PipelineProblem>,
                std::conditional_t<PreshuffleB,
                                   ck_tile::WPABQuantBPipelineAgBgCrV2<PipelineProblem>,
                                   ck_tile::ABQuantGemmPipelineAgBgCrCompV3<PipelineProblem>>>;

            using GemmEpilogue = std::conditional_t<
                TiledMMAPermuteN,
                ck_tile::PermuteNEpilogue<
                    ck_tile::PermuteNEpilogueProblem<typename PipelineProblem::AComputeDataType,
                                                     typename PipelineProblem::BComputeDataType,
                                                     ck_tile::tuple<>,
                                                     AccDataType,
                                                     CDataType,
                                                     ck_tile::tuple<>,
                                                     CLayout,
                                                     ck_tile::element_wise::PassThrough,
                                                     TilePartitioner::MPerBlock,
                                                     TilePartitioner::NPerBlock,
                                                     Base::M_Warp,
                                                     Base::N_Warp,
                                                     Base::M_Warp_Tile,
                                                     Base::N_Warp_Tile,
                                                     Base::K_Warp_Tile,
                                                     transpose_c,
                                                     false,
                                                     1>>,
                ck_tile::CShuffleEpilogue<
                    ck_tile::CShuffleEpilogueProblem<typename PipelineProblem::AComputeDataType,
                                                     typename PipelineProblem::BComputeDataType,
                                                     ck_tile::tuple<>,
                                                     AccDataType,
                                                     CDataType,
                                                     ck_tile::tuple<>,
                                                     CLayout,
                                                     ck_tile::element_wise::PassThrough,
                                                     TilePartitioner::MPerBlock,
                                                     TilePartitioner::NPerBlock,
                                                     Base::M_Warp,
                                                     Base::N_Warp,
                                                     Base::M_Warp_Tile,
                                                     Base::N_Warp_Tile,
                                                     Base::K_Warp_Tile,
                                                     transpose_c>>>;

            // TODO: Replace with templated lambda when C++20 is available
            auto LaunchKernel = [&](auto RuntimeSplitKTailTag) {
                constexpr bool RuntimeSplitKTail = decltype(RuntimeSplitKTailTag)::value;
                using Kernel                     = ck_tile::QuantGemmKernel<TilePartitioner,
                                                                            GemmPipeline,
                                                                            GemmEpilogue,
                                                                            ck_tile::QuantType::ABQuantGrouped,
                                                                            RuntimeSplitKTail>;

                auto kargs        = Kernel::MakeKernelArgs(args);
                const dim3 grids  = Kernel::GridSize(args.M, args.N, args.k_batch);
                const dim3 blocks = Kernel::BlockSize();

                if(!Kernel::IsSupportedArgument(kargs))
                {
                    throw std::runtime_error("Arguments not supported for ABQuant kernel");
                }
                using k_attr_t = ck_tile::kernel_attr<eight_waves>;
                ck_tile::launch_kernel(s,
                                       ck_tile::make_kernel<GemmConfigBase::kBlockPerCu, k_attr_t>(
                                           Kernel{}, grids, blocks, 0, kargs));
            };

            if(allow_runtime_splitk_tail)
            {
                LaunchKernel(std::true_type{});
            }
            else
            {
                LaunchKernel(std::false_type{});
            }
        };

        return BaseGemmPipeline::TailHandler(Run, has_hot_loop, tail_num);
    }
};

template <typename Tuple>
class TestCkTileGemmPreshuffleBBQuant : public TestCkTileGemmBQuant<Tuple>
{
};

// RowColQuant-specific test fixture
template <typename Tuple>
class TestCkTileGemmRowColQuant
    : public TestCkTileGemmQuantBase<Tuple, TestCkTileGemmRowColQuant<Tuple>>
{
    using Base = TestCkTileGemmQuantBase<Tuple, TestCkTileGemmRowColQuant<Tuple>>;
    friend Base;

    public:
    using typename Base::AccDataType;
    using typename Base::ADataType;
    using typename Base::ALayout;
    using typename Base::BDataType;
    using typename Base::BLayout;
    using typename Base::CDataType;
    using typename Base::CLayout;
    using typename Base::ComputeDataType;
    using typename Base::DsDataType;
    using typename Base::DsLayout;
    using typename Base::QDataType;

    static constexpr auto QuantType = Base::QuantType;

    static constexpr ck_tile::index_t NumDTensor = DsDataType::size();

    protected:
    void SetUpQuantTypeSpecific() {}
    void TearDownQuantTypeSpecific() {}

    void run_test_with_validation(ck_tile::index_t M, ck_tile::index_t N, ck_tile::index_t K)
    {
        const ck_tile::index_t stride_A =
            ck_tile::get_default_stride(M, K, 0, this->is_row_major(ALayout{}));
        const ck_tile::index_t stride_B =
            ck_tile::get_default_stride(K, N, 0, this->is_row_major(BLayout{}));
        const ck_tile::index_t stride_C =
            ck_tile::get_default_stride(M, N, 0, this->is_row_major(CLayout{}));

        // RowColQuant uses per-row and per-column scales
        const ck_tile::index_t stride_row_scales = 1;
        const ck_tile::index_t stride_col_scales = 1;

        // Generate test data
        ck_tile::HostTensor<ADataType> a_m_k(
            ck_tile::host_tensor_descriptor(M, K, stride_A, this->is_row_major(ALayout{})));
        ck_tile::HostTensor<BDataType> b_k_n(
            ck_tile::host_tensor_descriptor(K, N, stride_B, this->is_row_major(BLayout{})));
        ck_tile::HostTensor<QDataType> row_scales_m(ck_tile::host_tensor_descriptor(
            M, 1, stride_row_scales, ck_tile::bool_constant<true>{}));
        ck_tile::HostTensor<QDataType> col_scales_n(ck_tile::host_tensor_descriptor(
            N, 1, stride_col_scales, ck_tile::bool_constant<true>{}));

        // Initialize data with random values
        ck_tile::FillUniformDistribution<ADataType>{-0.5f, 0.5f}(a_m_k);
        ck_tile::FillUniformDistribution<BDataType>{-0.5f, 0.5f}(b_k_n);
        ck_tile::FillUniformDistribution<QDataType>{0.001f, 0.01f}(row_scales_m);
        ck_tile::FillUniformDistribution<QDataType>{0.001f, 0.01f}(col_scales_n);

        // Allocate device memory
        ck_tile::DeviceMem a_m_k_dev_buf(a_m_k.get_element_space_size() * sizeof(ADataType));
        ck_tile::DeviceMem b_k_n_dev_buf(b_k_n.get_element_space_size() * sizeof(BDataType));
        ck_tile::DeviceMem row_scales_dev_buf(row_scales_m.get_element_space_size() *
                                              sizeof(QDataType));
        ck_tile::DeviceMem col_scales_dev_buf(col_scales_n.get_element_space_size() *
                                              sizeof(QDataType));
        ck_tile::DeviceMem c_m_n_dev_buf(static_cast<size_t>(M) * N * sizeof(CDataType));

        // Copy to device
        a_m_k_dev_buf.ToDevice(a_m_k.data());
        b_k_n_dev_buf.ToDevice(b_k_n.data());
        row_scales_dev_buf.ToDevice(row_scales_m.data());
        col_scales_dev_buf.ToDevice(col_scales_n.data());

        auto ds_m_n = ck_tile::generate_tuple(
            [&](auto i) {
                using DiLayout = ck_tile::remove_cvref_t<std::tuple_element_t<i.value, DsLayout>>;
                using DiDataType =
                    ck_tile::remove_cvref_t<std::tuple_element_t<i.value, DsDataType>>;
                ck_tile::HostTensor<DiDataType> d_m_n(ck_tile::host_tensor_descriptor(
                    M, N, stride_C, this->is_row_major(DiLayout{})));
                ck_tile::FillUniformDistribution<DiDataType>{-10.0f, 10.0f}(d_m_n);
                return d_m_n;
            },
            ck_tile::number<NumDTensor>{});

        std::array<ck_tile::DeviceMem, NumDTensor> ds_m_n_dev_buf;
        std::array<const void*, NumDTensor> ds_ptr_buf;
        std::array<ck_tile::index_t, NumDTensor> stride_Ds;

        ck_tile::static_for<0, NumDTensor, 1>{}([&](auto i) {
            ds_m_n_dev_buf[i].Realloc(ds_m_n[i].get_element_space_size_in_bytes());
            ds_m_n_dev_buf[i].ToDevice(ds_m_n[i].data());
            ds_ptr_buf[i] = ds_m_n_dev_buf[i].GetDeviceBuffer();
            stride_Ds[i]  = stride_C;
        });

        // Create args for kernel execution
        ck_tile::QuantGemmMultiDHostArgs<NumDTensor> args{
            a_m_k_dev_buf.GetDeviceBuffer(),      // a_ptr
            b_k_n_dev_buf.GetDeviceBuffer(),      // b_ptr
            ds_ptr_buf,                           // ds_ptr
            c_m_n_dev_buf.GetDeviceBuffer(),      // c_ptr
            row_scales_dev_buf.GetDeviceBuffer(), // aq_ptr (row scales)
            col_scales_dev_buf.GetDeviceBuffer(), // bq_ptr (col scales)
            1,                                    // k_batch
            M,
            N,
            K, // M, N, K
            1, // QK_A (row scales)
            1, // QK_B (col scales)
            stride_A,
            stride_B,
            stride_Ds,
            stride_C,
            stride_row_scales,
            stride_col_scales // strides
        };

        // Run the kernel
        ck_tile::stream_config stream_config{};
        this->invoke_quant_gemm(args, stream_config);

        // Validation using reference implementation
        ck_tile::HostTensor<AccDataType> c_m_n_host_ref0(
            ck_tile::host_tensor_descriptor(M, N, stride_C, this->is_row_major(CLayout{})));
        c_m_n_host_ref0.SetZero();

        // Run reference RowColQuant implementation
        ck_tile::reference_gemm_rowcol_quant<ADataType,
                                             QDataType,
                                             BDataType,
                                             QDataType,
                                             AccDataType,
                                             AccDataType>(
            a_m_k, row_scales_m, b_k_n, col_scales_n, c_m_n_host_ref0);

        ck_tile::static_for<0, NumDTensor, 1>{}([&](auto i) {
            using DiDataType = ck_tile::remove_cvref_t<std::tuple_element_t<i.value, DsDataType>>;

            auto op = [](const auto& v0, const auto& v1) {
                return v0 + ck_tile::type_convert<AccDataType>(v1);
            };
            ck_tile::
                reference_binary_elementwise<AccDataType, DiDataType, AccDataType, AccDataType>(
                    c_m_n_host_ref0, ds_m_n[i], c_m_n_host_ref0, op);
        });

        auto c_m_n_host_ref = c_m_n_host_ref0.template CopyAsType<CDataType>();

        // Get device result
        ck_tile::HostTensor<CDataType> c_m_n_dev_result(
            ck_tile::host_tensor_descriptor(M, N, stride_C, this->is_row_major(CLayout{})));
        c_m_n_dev_buf.FromDevice(c_m_n_dev_result.mData.data());

        // Calculate error tolerances
        const float max_accumulated_value =
            *std::max_element(c_m_n_host_ref.mData.begin(), c_m_n_host_ref.mData.end());
        const auto rtol_atol =
            this->template calculate_rtol_atol<ADataType, BDataType, AccDataType, CDataType>(
                K, 1, max_accumulated_value);

        // Validate results
        bool pass = ck_tile::check_err(c_m_n_dev_result,
                                       c_m_n_host_ref,
                                       "Error: Incorrect results!",
                                       rtol_atol.at(ck_tile::number<0>{}),
                                       rtol_atol.at(ck_tile::number<1>{}));

        EXPECT_TRUE(pass) << "RowColQuant validation failed with M=" << M << ", N=" << N
                          << ", K=" << K;

        if(!pass)
        {
            std::cout << "RowColQuant - Relative error threshold: "
                      << rtol_atol.at(ck_tile::number<0>{})
                      << " Absolute error threshold: " << rtol_atol.at(ck_tile::number<1>{})
                      << std::endl;
        }
    }

    // Combined A+B+C device-byte requirement shared by every large-tensor
    // memory-availability guard below, so the estimate has one source of truth instead of
    // being duplicated per method.
    static size_t required_device_bytes(ck_tile::index_t M, ck_tile::index_t N, ck_tile::index_t K)
    {
        return static_cast<size_t>(M) * N * sizeof(CDataType) +
               static_cast<size_t>(M) * K * sizeof(ADataType) +
               static_cast<size_t>(K) * N * sizeof(BDataType);
    }

    // Boundary correctness check for the large-tensor (64-bit global load/store) path.
    // A full host GEMM reference is infeasible at the scales this path targets (M*N*K can
    // reach ~1e12), and a crash-only launch would only prove the kernel does not fault -- it
    // cannot catch a silent out-of-bounds-but-still-valid address computation. This uses a
    // sparse, exactly-computable input instead: A and B are zero everywhere except a single
    // K-index `hot_k`, where A(m, hot_k) = m % 8 and B(hot_k, n) = n % 8, with unit row/col
    // scales. Every other K contributes zero, so C(m, n) reduces to the exact integer
    // product (m % 8) * (n % 8) -- representable exactly in FP8/Half, so no floating-point
    // reference implementation or tolerance derivation is needed. The caller supplies
    // `hot_k` and the (m, n) coordinates to spot-check, so each call site can target
    // whichever tensor's addressing crosses the 2^31-byte boundary for its shape. Only the
    // requested elements are read back (one hipMemcpy each), not the full (potentially
    // multi-GiB) C buffer.
    void run_test_boundary_check(
        ck_tile::index_t M,
        ck_tile::index_t N,
        ck_tile::index_t K,
        ck_tile::index_t hot_k,
        const std::vector<std::pair<ck_tile::index_t, ck_tile::index_t>>& spot_checks)
    {
        ASSERT_GE(hot_k, 0);
        ASSERT_LT(hot_k, K);

        const ck_tile::index_t stride_A =
            ck_tile::get_default_stride(M, K, 0, this->is_row_major(ALayout{}));
        const ck_tile::index_t stride_B =
            ck_tile::get_default_stride(K, N, 0, this->is_row_major(BLayout{}));
        const ck_tile::index_t stride_C =
            ck_tile::get_default_stride(M, N, 0, this->is_row_major(CLayout{}));

        const ck_tile::index_t stride_row_scales = 1;
        const ck_tile::index_t stride_col_scales = 1;

        // Same memory-availability guard as the other large-tensor helpers.
        {
            size_t free_mem = 0, total_mem = 0;
            ASSERT_EQ(hipMemGetInfo(&free_mem, &total_mem), hipSuccess);
            const size_t required = required_device_bytes(M, N, K);
            if(free_mem < required + (size_t{256} << 20)) // 256 MiB headroom
            {
                GTEST_SKIP() << "Insufficient device memory for large-tensor boundary-check "
                             << "test: need ~" << ((required >> 20) + 256) << " MiB, have "
                             << (free_mem >> 20) << " MiB free";
            }
        }

        ck_tile::HostTensor<ADataType> a_m_k(
            ck_tile::host_tensor_descriptor(M, K, stride_A, this->is_row_major(ALayout{})));
        ck_tile::HostTensor<BDataType> b_k_n(
            ck_tile::host_tensor_descriptor(K, N, stride_B, this->is_row_major(BLayout{})));
        ck_tile::HostTensor<QDataType> row_scales_m(ck_tile::host_tensor_descriptor(
            M, 1, stride_row_scales, ck_tile::bool_constant<true>{}));
        ck_tile::HostTensor<QDataType> col_scales_n(ck_tile::host_tensor_descriptor(
            N, 1, stride_col_scales, ck_tile::bool_constant<true>{}));

        // A and B are zero everywhere except the single K-index `hot_k`; unit scales make
        // C(m, n) reduce to the exact integer product (m % 8) * (n % 8).
        a_m_k.SetZero();
        b_k_n.SetZero();
        for(ck_tile::index_t m = 0; m < M; ++m)
        {
            a_m_k(m, hot_k) = ck_tile::type_convert<ADataType>(static_cast<float>(m % 8));
        }
        for(ck_tile::index_t n = 0; n < N; ++n)
        {
            b_k_n(hot_k, n) = ck_tile::type_convert<BDataType>(static_cast<float>(n % 8));
        }
        std::fill(row_scales_m.begin(), row_scales_m.end(), ck_tile::type_convert<QDataType>(1.0f));
        std::fill(col_scales_n.begin(), col_scales_n.end(), ck_tile::type_convert<QDataType>(1.0f));

        ck_tile::DeviceMem a_m_k_dev_buf(a_m_k.get_element_space_size() * sizeof(ADataType));
        ck_tile::DeviceMem b_k_n_dev_buf(b_k_n.get_element_space_size() * sizeof(BDataType));
        ck_tile::DeviceMem row_scales_dev_buf(row_scales_m.get_element_space_size() *
                                              sizeof(QDataType));
        ck_tile::DeviceMem col_scales_dev_buf(col_scales_n.get_element_space_size() *
                                              sizeof(QDataType));
        ck_tile::DeviceMem c_m_n_dev_buf(static_cast<size_t>(M) * N * sizeof(CDataType));

        a_m_k_dev_buf.ToDevice(a_m_k.data());
        b_k_n_dev_buf.ToDevice(b_k_n.data());
        row_scales_dev_buf.ToDevice(row_scales_m.data());
        col_scales_dev_buf.ToDevice(col_scales_n.data());

        // No D tensors in these test tuples (NumDTensor == 0 in every use of this helper).
        // Zero-filled (rather than random) so the exact
        // (m % 8) * (n % 8) expectation below still holds if a future tuple adds Ds. Guarded
        // by a static_assert below because HostTensor::SetZero() special-cases e8m0_t to 1.f
        // rather than 0 (host_tensor.hpp), which would silently violate this invariant.
        auto ds_m_n = ck_tile::generate_tuple(
            [&](auto i) {
                using DiLayout = ck_tile::remove_cvref_t<std::tuple_element_t<i.value, DsLayout>>;
                using DiDataType =
                    ck_tile::remove_cvref_t<std::tuple_element_t<i.value, DsDataType>>;
                static_assert(!std::is_same_v<DiDataType, ck_tile::e8m0_t>,
                              "run_test_boundary_check assumes SetZero() produces a value that "
                              "contributes 0 to the sum; e8m0_t's SetZero() fills 1.f instead. "
                              "Use an explicit zero fill for this DiDataType.");
                ck_tile::HostTensor<DiDataType> d_m_n(ck_tile::host_tensor_descriptor(
                    M, N, stride_C, this->is_row_major(DiLayout{})));
                d_m_n.SetZero();
                return d_m_n;
            },
            ck_tile::number<NumDTensor>{});

        std::array<ck_tile::DeviceMem, NumDTensor> ds_m_n_dev_buf;
        std::array<const void*, NumDTensor> ds_ptr_buf;
        std::array<ck_tile::index_t, NumDTensor> stride_Ds;

        ck_tile::static_for<0, NumDTensor, 1>{}([&](auto i) {
            ds_m_n_dev_buf[i].Realloc(ds_m_n[i].get_element_space_size_in_bytes());
            ds_m_n_dev_buf[i].ToDevice(ds_m_n[i].data());
            ds_ptr_buf[i] = ds_m_n_dev_buf[i].GetDeviceBuffer();
            stride_Ds[i]  = stride_C;
        });

        ck_tile::QuantGemmMultiDHostArgs<NumDTensor> args{
            a_m_k_dev_buf.GetDeviceBuffer(),      // a_ptr
            b_k_n_dev_buf.GetDeviceBuffer(),      // b_ptr
            ds_ptr_buf,                           // ds_ptr
            c_m_n_dev_buf.GetDeviceBuffer(),      // c_ptr
            row_scales_dev_buf.GetDeviceBuffer(), // aq_ptr (row scales)
            col_scales_dev_buf.GetDeviceBuffer(), // bq_ptr (col scales)
            1,                                    // k_batch
            M,
            N,
            K, // M, N, K
            1, // QK_A (row scales)
            1, // QK_B (col scales)
            stride_A,
            stride_B,
            stride_Ds,
            stride_C,
            stride_row_scales,
            stride_col_scales // strides
        };

        ck_tile::stream_config stream_config{};
        this->invoke_quant_gemm(args, stream_config);
        ASSERT_EQ(hipStreamSynchronize(stream_config.stream_id_), hipSuccess)
            << "Device-side fault while executing the large-tensor RowColQuant boundary "
            << "check (M=" << M << ", N=" << N << ", K=" << K << ")";

        // Spot-check only the requested elements via targeted single-element reads,
        // instead of pulling the full (potentially multi-GiB) C buffer back to host.
        const auto c_desc =
            ck_tile::host_tensor_descriptor(M, N, stride_C, this->is_row_major(CLayout{}));

        for(const auto& [m, n] : spot_checks)
        {
            ASSERT_GE(m, 0);
            ASSERT_LT(m, M);
            ASSERT_GE(n, 0);
            ASSERT_LT(n, N);

            const auto byte_offset = c_desc.GetOffsetFromMultiIndex(m, n) * sizeof(CDataType);

            CDataType actual{};
            ASSERT_EQ(
                hipMemcpy(&actual,
                          static_cast<const char*>(c_m_n_dev_buf.GetDeviceBuffer()) + byte_offset,
                          sizeof(CDataType),
                          hipMemcpyDeviceToHost),
                hipSuccess)
                << "Failed to read back C(" << m << ", " << n << ")";

            const float expected = static_cast<float>(m % 8) * static_cast<float>(n % 8);
            EXPECT_NEAR(ck_tile::type_convert<float>(actual), expected, 1e-3f)
                << "Boundary-check mismatch at C(" << m << ", " << n << ") (M=" << M << ", N=" << N
                << ", K=" << K << ", hot_k=" << hot_k << ")";
        }
    }

    // Full-output validation for the large-tensor (64-bit global load/store) path. A host
    // reference is intractable at these scales, so the reference is computed on the device by
    // reference_gemm_gpu (naive_gemm_kernel, which uses 64-bit indexing) and compared against
    // the kernel output on the host. Row/column scales are 1 so the RowColQuant result equals a
    // plain A*B GEMM -- exactly what the GPU reference computes; non-unit scale math is already
    // covered by the small validated tests above. A and B are drawn from [-0.25, 0.25] so the K
    // reduction cannot overflow the Half output even at the largest K exercised here (worst case
    // K * 0.25 * 0.25 stays below Half's finite max for every shape in this file).
    void run_test_with_gpu_reference(ck_tile::index_t M, ck_tile::index_t N, ck_tile::index_t K)
    {
        static_assert(NumDTensor == 0,
                      "run_test_with_gpu_reference validates against a plain A*B GPU GEMM, which "
                      "has no multi-D epilogue; use a test tuple without D tensors.");

        const ck_tile::index_t stride_A =
            ck_tile::get_default_stride(M, K, 0, this->is_row_major(ALayout{}));
        const ck_tile::index_t stride_B =
            ck_tile::get_default_stride(K, N, 0, this->is_row_major(BLayout{}));
        const ck_tile::index_t stride_C =
            ck_tile::get_default_stride(M, N, 0, this->is_row_major(CLayout{}));

        const ck_tile::index_t stride_row_scales = 1;
        const ck_tile::index_t stride_col_scales = 1;

        // Device-memory guard: the kernel output C, the reference C and the A/B inputs must all
        // fit at once (required_device_bytes covers A+B+one C; the reference adds a second C).
        {
            size_t free_mem = 0, total_mem = 0;
            ASSERT_EQ(hipMemGetInfo(&free_mem, &total_mem), hipSuccess);
            const size_t required =
                required_device_bytes(M, N, K) + static_cast<size_t>(M) * N * sizeof(CDataType);
            if(free_mem < required + (size_t{256} << 20)) // 256 MiB headroom
            {
                GTEST_SKIP() << "Insufficient device memory for large-tensor GPU-reference test: "
                             << "need ~" << ((required >> 20) + 256) << " MiB, have "
                             << (free_mem >> 20) << " MiB free";
            }
        }

        ck_tile::HostTensor<ADataType> a_m_k(
            ck_tile::host_tensor_descriptor(M, K, stride_A, this->is_row_major(ALayout{})));
        ck_tile::HostTensor<BDataType> b_k_n(
            ck_tile::host_tensor_descriptor(K, N, stride_B, this->is_row_major(BLayout{})));
        ck_tile::HostTensor<QDataType> row_scales_m(ck_tile::host_tensor_descriptor(
            M, 1, stride_row_scales, ck_tile::bool_constant<true>{}));
        ck_tile::HostTensor<QDataType> col_scales_n(ck_tile::host_tensor_descriptor(
            N, 1, stride_col_scales, ck_tile::bool_constant<true>{}));

        // Signed inputs in a small range keep the K reduction inside the Half output range; unit
        // scales make the RowColQuant result equal the plain A*B GEMM the GPU reference computes.
        ck_tile::FillUniformDistribution<ADataType>{-0.25f, 0.25f}(a_m_k);
        ck_tile::FillUniformDistribution<BDataType>{-0.25f, 0.25f}(b_k_n);
        std::fill(row_scales_m.begin(), row_scales_m.end(), ck_tile::type_convert<QDataType>(1.0f));
        std::fill(col_scales_n.begin(), col_scales_n.end(), ck_tile::type_convert<QDataType>(1.0f));

        ck_tile::DeviceMem a_m_k_dev_buf(a_m_k.get_element_space_size() * sizeof(ADataType));
        ck_tile::DeviceMem b_k_n_dev_buf(b_k_n.get_element_space_size() * sizeof(BDataType));
        ck_tile::DeviceMem row_scales_dev_buf(row_scales_m.get_element_space_size() *
                                              sizeof(QDataType));
        ck_tile::DeviceMem col_scales_dev_buf(col_scales_n.get_element_space_size() *
                                              sizeof(QDataType));
        ck_tile::DeviceMem c_m_n_dev_buf(static_cast<size_t>(M) * N * sizeof(CDataType));

        a_m_k_dev_buf.ToDevice(a_m_k.data());
        b_k_n_dev_buf.ToDevice(b_k_n.data());
        row_scales_dev_buf.ToDevice(row_scales_m.data());
        col_scales_dev_buf.ToDevice(col_scales_n.data());

        // NumDTensor == 0 for every tuple that uses this helper (enforced above), so the D
        // pointer and stride arrays are empty.
        std::array<const void*, NumDTensor> ds_ptr_buf;
        std::array<ck_tile::index_t, NumDTensor> stride_Ds;

        ck_tile::QuantGemmMultiDHostArgs<NumDTensor> args{
            a_m_k_dev_buf.GetDeviceBuffer(),      // a_ptr
            b_k_n_dev_buf.GetDeviceBuffer(),      // b_ptr
            ds_ptr_buf,                           // ds_ptr
            c_m_n_dev_buf.GetDeviceBuffer(),      // c_ptr
            row_scales_dev_buf.GetDeviceBuffer(), // aq_ptr (row scales)
            col_scales_dev_buf.GetDeviceBuffer(), // bq_ptr (col scales)
            1,                                    // k_batch
            M,
            N,
            K, // M, N, K
            1, // QK_A (row scales)
            1, // QK_B (col scales)
            stride_A,
            stride_B,
            stride_Ds,
            stride_C,
            stride_row_scales,
            stride_col_scales // strides
        };

        ck_tile::stream_config stream_config{};
        this->invoke_quant_gemm(args, stream_config);
        ASSERT_EQ(hipStreamSynchronize(stream_config.stream_id_), hipSuccess)
            << "Device-side fault while executing the large-tensor RowColQuant kernel "
            << "(M=" << M << ", N=" << N << ", K=" << K << ")";

        ck_tile::HostTensor<CDataType> c_m_n_dev_result(
            ck_tile::host_tensor_descriptor(M, N, stride_C, this->is_row_major(CLayout{})));
        c_m_n_dev_buf.FromDevice(c_m_n_dev_result.mData.data());

        // GPU reference: plain A*B GEMM with 64-bit addressing. reads the same A/B device buffers
        // (they are only read by the kernel above), writes into its own C buffer.
        ck_tile::DeviceMem c_ref_dev_buf(static_cast<size_t>(M) * N * sizeof(CDataType));
        c_ref_dev_buf.SetZero();

        ADataType* a_ref_ptr = static_cast<ADataType*>(a_m_k_dev_buf.GetDeviceBuffer());
        BDataType* b_ref_ptr = static_cast<BDataType*>(b_k_n_dev_buf.GetDeviceBuffer());
        CDataType* c_ref_ptr = static_cast<CDataType*>(c_ref_dev_buf.GetDeviceBuffer());
        ck_tile::reference_gemm_gpu<ADataType,
                                    BDataType,
                                    AccDataType,
                                    CDataType,
                                    ALayout,
                                    BLayout,
                                    CLayout>(
            a_ref_ptr, b_ref_ptr, c_ref_ptr, M, N, K, stride_A, stride_B, stride_C);
        ASSERT_EQ(hipDeviceSynchronize(), hipSuccess)
            << "Device-side fault while executing the GPU reference GEMM " << "(M=" << M
            << ", N=" << N << ", K=" << K << ")";

        ck_tile::HostTensor<CDataType> c_m_n_gpu_ref(
            ck_tile::host_tensor_descriptor(M, N, stride_C, this->is_row_major(CLayout{})));
        c_ref_dev_buf.FromDevice(c_m_n_gpu_ref.mData.data());

        const float max_accumulated_value =
            *std::max_element(c_m_n_gpu_ref.mData.begin(), c_m_n_gpu_ref.mData.end());
        const auto rtol_atol =
            this->template calculate_rtol_atol<ADataType, BDataType, AccDataType, CDataType>(
                K, 1, max_accumulated_value);

        bool pass = ck_tile::check_err(c_m_n_dev_result,
                                       c_m_n_gpu_ref,
                                       "Error: Incorrect results!",
                                       rtol_atol.at(ck_tile::number<0>{}),
                                       rtol_atol.at(ck_tile::number<1>{}));

        EXPECT_TRUE(pass) << "RowColQuant GPU-reference validation failed with M=" << M
                          << ", N=" << N << ", K=" << K;
    }

    private:
    // RowColQuant-specific pipeline implementation
    template <typename CodegenGemmShape, typename TilePartitioner, typename CodegenGemmTraits>
    void run_quant_gemm_impl(const ck_tile::QuantGemmMultiDHostArgs<NumDTensor>& args,
                             const ck_tile::stream_config& s)
    {
        using GemmPipelineProblem = ck_tile::GemmPipelineProblemBase<ADataType,
                                                                     BDataType,
                                                                     AccDataType,
                                                                     CodegenGemmShape,
                                                                     CodegenGemmTraits,
                                                                     ComputeDataType,
                                                                     ComputeDataType>;

        using BaseGemmPipeline = ck_tile::BaseGemmPipelineAgBgCrCompV3<GemmPipelineProblem>;

        constexpr auto K1 = CodegenGemmShape::WarpTile::at(ck_tile::number<2>{});
        const ck_tile::index_t K_split =
            args.k_batch == 1 ? (args.K + Base::K_Tile - 1) / Base::K_Tile * Base::K_Tile
                              : ck_tile::get_splitk_batch_k_read(args.K, args.k_batch, K1);
        const ck_tile::index_t num_loop    = TilePartitioner::GetLoopNum(K_split);
        const bool has_hot_loop            = BaseGemmPipeline::BlockHasHotloop(num_loop);
        const ck_tile::TailNumber tail_num = BaseGemmPipeline::GetBlockLoopTailNum(num_loop);

        const auto Run = [&](const auto has_hot_loop_, const auto tail_number_) {
            constexpr bool has_hot_loop_v = has_hot_loop_.value;
            constexpr auto tail_number_v  = tail_number_.value;
            constexpr bool transpose_c    = CodegenGemmTraits::TransposeC;

            using PipelineProblem = ck_tile::GemmRowColTensorQuantPipelineProblem<
                ADataType,
                BDataType,
                AccDataType,
                AccDataType,
                CodegenGemmShape,
                CodegenGemmTraits,
                transpose_c,
                ComputeDataType,
                ck_tile::GemmPipelineScheduler::Intrawave,
                has_hot_loop_v,
                tail_number_v>;

            using GemmPipeline = ck_tile::GemmPipelineAgBgCrCompV3<PipelineProblem>;
            using GemmEpilogue = ck_tile::CShuffleEpilogue<
                ck_tile::CShuffleEpilogueProblem<ADataType,
                                                 BDataType,
                                                 DsDataType,
                                                 AccDataType,
                                                 CDataType,
                                                 DsLayout,
                                                 CLayout,
                                                 ck_tile::element_wise::MultiDAdd,
                                                 TilePartitioner::MPerBlock,
                                                 TilePartitioner::NPerBlock,
                                                 Base::M_Warp,
                                                 Base::N_Warp,
                                                 Base::M_Warp_Tile,
                                                 Base::N_Warp_Tile,
                                                 Base::K_Warp_Tile,
                                                 transpose_c>>;

            using Kernel = ck_tile::QuantGemmMultiDKernel<TilePartitioner,
                                                          GemmPipeline,
                                                          GemmEpilogue,
                                                          ck_tile::QuantType::RowColQuant>;

            auto kargs        = Kernel::MakeKernelArgs(args);
            const dim3 grids  = Kernel::GridSize(args.M, args.N, args.k_batch);
            const dim3 blocks = Kernel::BlockSize();

            if(!Kernel::IsSupportedArgument(kargs))
            {
                throw std::runtime_error("Arguments not supported for RowColQuant kernel");
            }

            ck_tile::launch_kernel(s,
                                   ck_tile::make_kernel<GemmConfigBase::kBlockPerCu>(
                                       Kernel{}, grids, blocks, 0, kargs));
        };

        return BaseGemmPipeline::TailHandler(Run, has_hot_loop, tail_num);
    }
};

// TensorQuant-specific test fixture
template <typename Tuple>
class TestCkTileGemmTensorQuant
    : public TestCkTileGemmQuantBase<Tuple, TestCkTileGemmTensorQuant<Tuple>>
{
    using Base = TestCkTileGemmQuantBase<Tuple, TestCkTileGemmTensorQuant<Tuple>>;
    friend Base;

    public:
    using typename Base::AccDataType;
    using typename Base::ADataType;
    using typename Base::ALayout;
    using typename Base::BDataType;
    using typename Base::BLayout;
    using typename Base::CDataType;
    using typename Base::CLayout;
    using typename Base::ComputeDataType;
    using typename Base::QDataType;
    using typename Base::QuantGroupSize;

    static constexpr auto QuantType = Base::QuantType;

    protected:
    void SetUpQuantTypeSpecific() {}
    void TearDownQuantTypeSpecific() {}

    void run_test_with_validation(ck_tile::index_t M, ck_tile::index_t N, ck_tile::index_t K)
    {
        const ck_tile::index_t stride_A = K;
        const ck_tile::index_t stride_B = K;
        const ck_tile::index_t stride_C = N;

        // TensorQuant uses single scalar scale for each tensor
        const ck_tile::index_t stride_scale_a = 1;
        const ck_tile::index_t stride_scale_b = 1;

        // Generate test data
        ck_tile::HostTensor<ADataType> a_m_k(
            ck_tile::host_tensor_descriptor(M, K, stride_A, this->is_row_major(ALayout{})));
        ck_tile::HostTensor<BDataType> b_k_n(
            ck_tile::host_tensor_descriptor(K, N, stride_B, this->is_row_major(BLayout{})));
        ck_tile::HostTensor<QDataType> scale_a(
            ck_tile::host_tensor_descriptor(1, 1, stride_scale_a, ck_tile::bool_constant<true>{}));
        ck_tile::HostTensor<QDataType> scale_b(
            ck_tile::host_tensor_descriptor(1, 1, stride_scale_b, ck_tile::bool_constant<true>{}));

        // Initialize data with random values
        ck_tile::FillUniformDistribution<ADataType>{-0.5f, 0.5f}(a_m_k);
        ck_tile::FillUniformDistribution<BDataType>{-0.5f, 0.5f}(b_k_n);
        ck_tile::FillUniformDistribution<QDataType>{0.001f, 0.01f}(scale_a);
        ck_tile::FillUniformDistribution<QDataType>{0.001f, 0.01f}(scale_b);

        // Allocate device memory
        ck_tile::DeviceMem a_m_k_dev_buf(a_m_k.get_element_space_size() * sizeof(ADataType));
        ck_tile::DeviceMem b_k_n_dev_buf(b_k_n.get_element_space_size() * sizeof(BDataType));
        ck_tile::DeviceMem scale_a_dev_buf(scale_a.get_element_space_size() * sizeof(QDataType));
        ck_tile::DeviceMem scale_b_dev_buf(scale_b.get_element_space_size() * sizeof(QDataType));
        ck_tile::DeviceMem c_m_n_dev_buf(static_cast<size_t>(M) * N * sizeof(CDataType));

        // Copy to device
        a_m_k_dev_buf.ToDevice(a_m_k.data());
        b_k_n_dev_buf.ToDevice(b_k_n.data());
        scale_a_dev_buf.ToDevice(scale_a.data());
        scale_b_dev_buf.ToDevice(scale_b.data());

        // Create args for kernel execution
        ck_tile::QuantGemmHostArgs args{
            a_m_k_dev_buf.GetDeviceBuffer(),   // a_ptr
            b_k_n_dev_buf.GetDeviceBuffer(),   // b_ptr
            c_m_n_dev_buf.GetDeviceBuffer(),   // c_ptr
            scale_a_dev_buf.GetDeviceBuffer(), // aq_ptr (scale A)
            scale_b_dev_buf.GetDeviceBuffer(), // bq_ptr (scale B)
            1,                                 // k_batch
            M,
            N,
            K, // M, N, K
            1, // QK_A (tensor scale)
            1, // QK_B (tensor scale)
            stride_A,
            stride_B,
            stride_C,
            stride_scale_a,
            stride_scale_b // strides
        };

        // Run the kernel
        ck_tile::stream_config stream_config{};
        this->invoke_quant_gemm(args, stream_config);

        // Validation using reference implementation
        ck_tile::HostTensor<CDataType> c_m_n_host_ref(
            ck_tile::host_tensor_descriptor(M, N, stride_C, this->is_row_major(CLayout{})));
        c_m_n_host_ref.SetZero();

        // Run reference TensorQuant implementation
        ck_tile::reference_gemm_tensor_quant<ADataType,
                                             QDataType,
                                             BDataType,
                                             QDataType,
                                             AccDataType,
                                             CDataType>(
            a_m_k, scale_a, b_k_n, scale_b, c_m_n_host_ref);

        // Get device result
        ck_tile::HostTensor<CDataType> c_m_n_dev_result(
            ck_tile::host_tensor_descriptor(M, N, stride_C, this->is_row_major(CLayout{})));
        c_m_n_dev_buf.FromDevice(c_m_n_dev_result.mData.data());

        // Calculate error tolerances
        const float max_accumulated_value =
            *std::max_element(c_m_n_host_ref.mData.begin(), c_m_n_host_ref.mData.end());
        const auto rtol_atol =
            this->template calculate_rtol_atol<ADataType, BDataType, AccDataType, CDataType>(
                K, 1, max_accumulated_value);

        // Validate results
        bool pass = ck_tile::check_err(c_m_n_dev_result,
                                       c_m_n_host_ref,
                                       "Error: Incorrect results!",
                                       rtol_atol.at(ck_tile::number<0>{}),
                                       rtol_atol.at(ck_tile::number<1>{}));

        EXPECT_TRUE(pass) << "TensorQuant validation failed with M=" << M << ", N=" << N
                          << ", K=" << K;

        if(!pass)
        {
            std::cout << "TensorQuant - Relative error threshold: "
                      << rtol_atol.at(ck_tile::number<0>{})
                      << " Absolute error threshold: " << rtol_atol.at(ck_tile::number<1>{})
                      << std::endl;
        }
    }

    private:
    // TensorQuant-specific pipeline implementation
    template <typename CodegenGemmShape, typename TilePartitioner, typename CodegenGemmTraits>
    void run_quant_gemm_impl(const ck_tile::QuantGemmHostArgs& args,
                             const ck_tile::stream_config& s)
    {
        using GemmPipelineProblem = ck_tile::GemmPipelineProblemBase<ADataType,
                                                                     BDataType,
                                                                     AccDataType,
                                                                     CodegenGemmShape,
                                                                     CodegenGemmTraits,
                                                                     ComputeDataType,
                                                                     ComputeDataType>;

        using BaseGemmPipeline = ck_tile::BaseGemmPipelineAgBgCrCompV3<GemmPipelineProblem>;

        constexpr auto K1 = CodegenGemmShape::WarpTile::at(ck_tile::number<2>{});
        const ck_tile::index_t K_split =
            args.k_batch == 1 ? (args.K + Base::K_Tile - 1) / Base::K_Tile * Base::K_Tile
                              : ck_tile::get_splitk_batch_k_read(args.K, args.k_batch, K1);
        const ck_tile::index_t num_loop    = TilePartitioner::GetLoopNum(K_split);
        const bool has_hot_loop            = BaseGemmPipeline::BlockHasHotloop(num_loop);
        const ck_tile::TailNumber tail_num = BaseGemmPipeline::GetBlockLoopTailNum(num_loop);

        const auto Run = [&](const auto has_hot_loop_, const auto tail_number_) {
            constexpr bool has_hot_loop_v = has_hot_loop_.value;
            constexpr auto tail_number_v  = tail_number_.value;
            constexpr bool transpose_c    = CodegenGemmTraits::TransposeC;

            using PipelineProblem = ck_tile::GemmRowColTensorQuantPipelineProblem<
                ADataType,
                BDataType,
                AccDataType,
                AccDataType,
                CodegenGemmShape,
                CodegenGemmTraits,
                transpose_c,
                ComputeDataType,
                ck_tile::GemmPipelineScheduler::Intrawave,
                has_hot_loop_v,
                tail_number_v>;

            using GemmPipeline = ck_tile::GemmPipelineAgBgCrCompV3<PipelineProblem>;
            using GemmEpilogue = ck_tile::CShuffleEpilogue<
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
                                                 Base::M_Warp,
                                                 Base::N_Warp,
                                                 Base::M_Warp_Tile,
                                                 Base::N_Warp_Tile,
                                                 Base::K_Warp_Tile,
                                                 transpose_c>>;

            using Kernel = ck_tile::QuantGemmKernel<TilePartitioner,
                                                    GemmPipeline,
                                                    GemmEpilogue,
                                                    ck_tile::QuantType::TensorQuant>;

            auto kargs        = Kernel::MakeKernelArgs(args);
            const dim3 grids  = Kernel::GridSize(args.M, args.N, args.k_batch);
            const dim3 blocks = Kernel::BlockSize();

            if(!Kernel::IsSupportedArgument(kargs))
            {
                throw std::runtime_error("Arguments not supported for TensorQuant kernel");
            }

            ck_tile::launch_kernel(s,
                                   ck_tile::make_kernel<GemmConfigBase::kBlockPerCu>(
                                       Kernel{}, grids, blocks, 0, kargs));
        };

        return BaseGemmPipeline::TailHandler(Run, has_hot_loop, tail_num);
    }
};
