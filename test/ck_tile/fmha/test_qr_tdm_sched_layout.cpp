// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <array>
#include <cstdint>
#include <iostream>
#include <string_view>
#include <vector>

#include "ck_tile/host.hpp"
#include "fmha_test_common.hpp"

namespace {

using namespace ck_tile;
#if defined(CK_TILE_FMHA_TDM_V128_TEST_FP16)
using Data = half_t;
#else
using Data = bf16_t;
#endif

CK_TILE_HOST_DEVICE constexpr std::uint16_t Tag(int row, int col, int width)
{
    return static_cast<std::uint16_t>(0x1000 + row * width + col);
}

template <typename Tensor, typename Visitor>
CK_TILE_DEVICE void VisitElements(Tensor& tensor, const Visitor& visitor)
{
    constexpr auto distribution = typename Tensor::StaticTileDistribution{};
    constexpr auto lengths      = to_sequence(distribution.get_ys_to_d_descriptor().get_lengths());
    const auto partitions       = get_partition_index(distribution);
    static_ford<remove_cvref_t<decltype(lengths)>>{}([&](auto ys) {
        constexpr index_t offset = distribution.get_ys_to_d_descriptor().calculate_offset(ys);
        const auto coord         = make_tensor_adaptor_coordinate(
            distribution.get_ps_ys_to_xs_adaptor(),
            container_concat(partitions, to_array<index_t, ys.size()>(ys)));
        const auto xy = coord.get_bottom_index();
        visitor(tensor.get_thread_buffer().template at<offset>(), xy[number<0>{}], xy[number<1>{}]);
    });
}

struct Args
{
    const Data* q;
    const Data* k;
    const Data* v;
    std::uint8_t* arena_dump;
    std::uint32_t* errors;
    int q_rows;
    int kv_rows;
};

template <typename A, typename B>
constexpr bool SameBlockEncoding()
{
    using AWarp = typename A::WarpGemm;
    using BWarp = typename B::WarpGemm;
    return std::is_same_v<typename AWarp::AWarpDstrEncoding, typename BWarp::AWarpDstrEncoding> &&
           std::is_same_v<typename AWarp::BWarpDstrEncoding, typename BWarp::BWarpDstrEncoding> &&
           std::is_same_v<typename AWarp::CWarpDstrEncoding, typename BWarp::CWarpDstrEncoding> &&
           std::is_same_v<remove_cvref_t<decltype(A::MakeABlockDistributionEncode())>,
                          remove_cvref_t<decltype(B::MakeABlockDistributionEncode())>> &&
           std::is_same_v<remove_cvref_t<decltype(A::MakeBBlockDistributionEncode())>,
                          remove_cvref_t<decltype(B::MakeBBlockDistributionEncode())>> &&
           std::is_same_v<remove_cvref_t<decltype(A::MakeCBlockDistributionEncode())>,
                          remove_cvref_t<decltype(B::MakeCBlockDistributionEncode())>>;
}

template <typename Problem, typename Q, typename K, typename V, typename P>
struct OperandProblem : Problem
{
    using QDataType = Q;
    using KDataType = K;
    using VDataType = V;
    using PDataType = P;
};

template <int Dim, int N = 128, int ValueDim = 128>
constexpr bool CheckDtypeEncoding()
{
    using B = typename ck_tile::test::Model<Dim, false, false, false, bf16_t, N, ValueDim>::Problem;
    using F = typename ck_tile::test::Model<Dim, false, false, false, half_t, N, ValueDim>::Problem;
    using BP     = ck_tile::test::Policy<B>;
    using FP     = ck_tile::test::Policy<F>;
    using BQK    = remove_cvref_t<decltype(BP::template GetQKBlockGemmSu<B>())>;
    using FQK    = remove_cvref_t<decltype(FP::template GetQKBlockGemmSu<F>())>;
    using BPV    = remove_cvref_t<decltype(BP::template GetPVBlockGemm<B>())>;
    using FPV    = remove_cvref_t<decltype(FP::template GetPVBlockGemm<F>())>;
    using BRead  = remove_cvref_t<decltype(BP::template MakeVRegTileDistribution<B>())>;
    using FRead  = remove_cvref_t<decltype(FP::template MakeVRegTileDistribution<F>())>;
    using BQRead = remove_cvref_t<decltype(BP::template MakeQRegTileDistribution<B>())>;
    using FQRead = remove_cvref_t<decltype(FP::template MakeQRegTileDistribution<F>())>;
    using BKRead = remove_cvref_t<decltype(BP::template MakeKSuRegTileDistribution<B>())>;
    using FKRead = remove_cvref_t<decltype(FP::template MakeKSuRegTileDistribution<F>())>;
    return BP::template IsSupportedProblem<B>() && FP::template IsSupportedProblem<F>() &&
           !BP::template IsSupportedProblem<F>() && !FP::template IsSupportedProblem<B>() &&
           !FP::template IsSupportedProblem<OperandProblem<F, bf16_t, half_t, half_t, half_t>>() &&
           !FP::template IsSupportedProblem<OperandProblem<F, half_t, bf16_t, half_t, half_t>>() &&
           !FP::template IsSupportedProblem<OperandProblem<F, half_t, half_t, bf16_t, half_t>>() &&
           !FP::template IsSupportedProblem<OperandProblem<F, half_t, half_t, half_t, bf16_t>>() &&
           SameBlockEncoding<BQK, FQK>() && SameBlockEncoding<BPV, FPV>() &&
           std::is_same_v<typename BRead::DstrEncode, typename FRead::DstrEncode> &&
           std::is_same_v<typename BQRead::DstrEncode, typename FQRead::DstrEncode> &&
           std::is_same_v<typename BKRead::DstrEncode, typename FKRead::DstrEncode> &&
           std::is_same_v<typename BP::OutputFragments, typename FP::OutputFragments> &&
           BP::kKFootprintBytes == FP::kKFootprintBytes &&
           BP::kVFootprintBytes == FP::kVFootprintBytes && BP::kLdsArenaSize == FP::kLdsArenaSize;
}
static_assert(CheckDtypeEncoding<128>() && CheckDtypeEncoding<192>());
static_assert(CheckDtypeEncoding<64, 64, 64>() && CheckDtypeEncoding<192, 64, 128>());

template <int Dim, int N = 128, int ValueDim = 128>
struct LayoutProbe
{
    using Problem =
        typename ck_tile::test::Model<Dim, false, false, false, Data, N, ValueDim>::Problem;
    using Policy                        = ck_tile::test::Policy<Problem>;
    using Geometry                      = typename Policy::Geometry;
    using Pipeline                      = BlockFmhaPipelineQRKSVSTdmSched<Problem, Policy>;
    static constexpr index_t kBlockSize = Geometry::kWaves * Geometry::kWaveSize;
    static constexpr index_t kInputQKStride =
        Geometry::kHeadDimQK + 2 * Geometry::kLdsReadBytes / sizeof(Data);
    static_assert(Policy::template GetSmemSizeQ<Problem>() <= 2 * Policy::kKFootprintBytes);
    static_assert(Geometry::kN != 64 ||
                  Policy::template GetSmemSizeQ<Problem>() > Policy::kKFootprintBytes);
    static_assert(Geometry::kN != 128 ||
                  Policy::template GetSmemSizeQ<Problem>() <= Policy::kKFootprintBytes);
    static_assert(Dim != 128 || N != 128 || !std::is_same_v<Data, bf16_t> ||
                  Policy::template GetSmemSizeQ<Problem>() == 34800);

    CK_TILE_DEVICE void operator()(Args args) const
    {
        __shared__ char arena[Policy::kLdsArenaSize];
        for(index_t i = threadIdx.x; i < Policy::kLdsArenaSize; i += kBlockSize)
            arena[i] = static_cast<char>(0xa5);
        block_sync_lds<0>();
        std::uint32_t errors = 0;
        auto input_q         = make_naive_tensor_view<address_space_enum::global>(
            args.q, make_tuple(args.q_rows, Geometry::kHeadDimQK), make_tuple(kInputQKStride, 1));
        auto input_k = make_naive_tensor_view<address_space_enum::global>(
            args.k, make_tuple(args.kv_rows, Geometry::kHeadDimQK), make_tuple(kInputQKStride, 1));
        auto input_v = make_naive_tensor_view<address_space_enum::global>(
            args.v,
            make_tuple(args.kv_rows, Geometry::kDv),
            make_tuple(Policy::kVPhysicalStride, 1));

        // M128 Q can span both N64 K buffers; every Q reader completes before K is overwritten.
        auto q_view = make_tensor_view<address_space_enum::lds>(
            reinterpret_cast<Data*>(arena), Policy::template MakeQLdsBlockDescriptor<Problem>());
        auto q_write = make_tile_window(
            q_view, make_tuple(number<Geometry::kM>{}, number<Geometry::kHeadDimQK>{}), {0, 0});
        auto q_global =
            make_tile_window(input_q,
                             make_tuple(number<Geometry::kM>{}, number<Geometry::kHeadDimQK>{}),
                             {0, 0},
                             Policy::template MakeQDramTileDistribution<Problem>());
        constexpr auto q_padding = Policy::template GetLdsPaddingConfigQ<Problem>();
        TDMConfig q_config{};
        q_config.pad_enable              = q_padding[number<0>{}];
        q_config.pad_config.pad_amount   = q_padding[number<1>{}];
        q_config.pad_config.pad_interval = q_padding[number<2>{}];
        load_tile_tdm(q_config, q_write, q_global);
        s_wait_tensorcnt_barrier<0>();
        auto q = load_tile(
            make_tile_window(q_view,
                             make_tuple(number<Geometry::kM>{}, number<Geometry::kHeadDimQK>{}),
                             {0, 0},
                             Policy::template MakeQRegTileDistribution<Problem>()));
        using Qk = remove_cvref_t<decltype(Policy::template GetQKBlockGemmSu<Problem>())>;
        static_assert(std::is_same_v<typename decltype(q)::StaticTileDistribution::DstrEncode,
                                     remove_cvref_t<decltype(Qk::MakeABlockDistributionEncode())>>);
        VisitElements(q, [&](auto value, int row, int col) {
            errors += bit_cast<std::uint16_t>(value) !=
                      (row < args.q_rows ? Tag(row, col, Geometry::kHeadDimQK) : 0);
        });
        // Check Q cannot touch V before the later V writers can conceal an overrun.
        for(index_t i = Policy::kLdsOffsetV0 + threadIdx.x; i < Policy::kLdsArenaSize;
            i += kBlockSize)
            errors += static_cast<std::uint8_t>(arena[i]) != 0xa5;
        block_sync_lds<0>();

        static_for<0, 2, 1>{}([&](auto buffer) {
            constexpr int base = buffer == 0 ? Policy::kLdsOffsetK0 : Policy::kLdsOffsetK1;
            auto view          = make_tensor_view<address_space_enum::lds>(
                reinterpret_cast<Data*>(arena + base),
                Policy::template MakeKLdsWriteBlockDescriptor<Problem>());
            auto write = make_tile_window(
                view,
                make_tuple(number<Geometry::kN>{}, number<Policy::kKPhysicalStride>{}),
                {0, 0});
            auto global = make_tile_window(
                input_k,
                make_tuple(number<Geometry::kN>{}, number<Policy::kKPhysicalStride>{}),
                {0, 0},
                Policy::template MakeKDramTileDistribution<Problem>());
            load_tile_tdm(TDMConfig{}, write, global);
            s_wait_tensorcnt_barrier<0>();
            static_for<0, Geometry::kQkStages, 1>{}([&](auto su) {
                auto window = make_tile_window(
                    view,
                    make_tuple(number<Geometry::kQkSuColumns>{}, number<Geometry::kHeadDimQK>{}),
                    {su * Geometry::kQkSuColumns, 0},
                    Policy::template MakeKSuRegTileDistribution<Problem>());
                using Window = decltype(window);
                static_assert(Window::NumAccessPerCoord == Geometry::kKSuLoadCount);
                decltype(load_tile(window)) values;
                static_for<0, Geometry::kKSuLoadCount, 1>{}([&](auto access) {
                    Policy::KLoad::template LoadInstruction<access>(values, window);
                });
                s_wait_dscnt<0>();
                static_assert(
                    std::is_same_v<typename decltype(values)::StaticTileDistribution::DstrEncode,
                                   remove_cvref_t<decltype(Qk::MakeBBlockDistributionEncode())>>);
                VisitElements(values, [&](auto value, int row, int col) {
                    row += su * Geometry::kQkSuColumns;
                    errors += bit_cast<std::uint16_t>(value) !=
                              (row < args.kv_rows ? Tag(row, col, Geometry::kHeadDimQK) : 0);
                });
            });
            block_sync_lds<0>();
        });

        using Pv = remove_cvref_t<decltype(Policy::template GetPVBlockGemm<Problem>())>;
        static_for<0, 2, 1>{}([&](auto buffer) {
            constexpr int base = buffer == 0 ? Policy::kLdsOffsetV0 : Policy::kLdsOffsetV1;
            auto view          = make_tensor_view<address_space_enum::lds>(
                reinterpret_cast<Data*>(arena + base),
                Policy::template MakeVLdsWriteBlockDescriptor<Problem>());
            auto write = make_tile_window(
                view, make_tuple(number<Geometry::kN>{}, number<Geometry::kDv>{}), {0, 0});
            auto global =
                make_tile_window(input_v,
                                 make_tuple(number<Geometry::kN>{}, number<Geometry::kDv>{}),
                                 {0, 0},
                                 Policy::template MakeVDramTileDistribution<Problem>());
            TDMConfig config{};
            config.pad_enable              = true;
            config.pad_config.pad_interval = Policy::kVPadInterval;
            config.pad_config.pad_amount   = Policy::kVPadAmount;
            load_tile_tdm(config, write, global);
            s_wait_tensorcnt_barrier<0>();
            static_for<0, Geometry::kPvStages, 1>{}([&](auto stage) {
                auto window = make_tile_window(
                    view,
                    make_tuple(number<Geometry::kPvStageK>{}, number<Geometry::kDv>{}),
                    {stage * Geometry::kPvStageK, 0},
                    Policy::template MakeVRegTileDistribution<Problem>());
                using Window = decltype(window);
                static_assert(Window::NumAccessPerCoord == Geometry::kVStageLoadCount);
                decltype(load_tile_transpose(window)) values;
                static_for<0, Geometry::kVStageLoadCount, 1>{}([&](auto access) {
                    Policy::VLoad::template LoadAccess<access>(values, window);
                });
                s_wait_dscnt<0>();
                static_assert(
                    std::is_same_v<typename decltype(values)::StaticTileDistribution::DstrEncode,
                                   remove_cvref_t<decltype(Pv::MakeBBlockDistributionEncode())>>);
                VisitElements(values, [&](auto value, int col, int row) {
                    row += stage * Geometry::kPvStageK;
                    errors += bit_cast<std::uint16_t>(value) !=
                              (row < args.kv_rows ? Tag(row, col, Geometry::kDv) : 0);
                });
            });
            block_sync_lds<0>();
        });

        using ScoreGemm = remove_cvref_t<decltype(Policy::template GetQKBlockGemm<Problem>())>;
        auto score      = ScoreGemm::MakeCBlockTile();
        VisitElements(score, [](auto& value, int row, int col) {
            value = type_convert<float>(bit_cast<Data>(Tag(row, col, Geometry::kN)));
        });
        auto p = Pipeline::template MakePForGemm1<Pv>(score);
        static_for<0, Geometry::kPvStages, 1>{}([&](auto stage) {
            auto operand =
                get_slice_tile(p,
                               sequence<0, stage * Geometry::kPvStageK>{},
                               sequence<Geometry::kM, (stage + 1) * Geometry::kPvStageK>{});
            static_assert(
                std::is_same_v<typename decltype(operand)::StaticTileDistribution::DstrEncode,
                               remove_cvref_t<decltype(Pv::MakeABlockDistributionEncode())>>);
            VisitElements(operand, [&](auto value, int row, int col) {
                errors += bit_cast<std::uint16_t>(value) !=
                          Tag(row, col + stage * Geometry::kPvStageK, Geometry::kN);
            });
        });
        args.errors[threadIdx.x] = errors;
        for(index_t i = threadIdx.x; i < Policy::kLdsArenaSize; i += kBlockSize)
            args.arena_dump[i] = static_cast<std::uint8_t>(arena[i]);
    }
};

template <int Dim, int N = 128, int ValueDim = 128>
bool CheckLayout(int q_rows, int kv_rows)
{
    using Probe    = LayoutProbe<Dim, N, ValueDim>;
    using Policy   = typename Probe::Policy;
    using Geometry = typename Probe::Geometry;
    std::vector<std::uint16_t> q(Geometry::kM * Probe::kInputQKStride, 0x7fff),
        k(Geometry::kN * Probe::kInputQKStride, 0x7fff),
        v(Geometry::kN * Policy::kVPhysicalStride, 0x7fff);
    for(int row = 0; row < q_rows; ++row)
    {
        for(int col = 0; col < Geometry::kHeadDimQK; ++col)
            q[row * Probe::kInputQKStride + col] = Tag(row, col, Geometry::kHeadDimQK);
    }
    for(int row = 0; row < kv_rows; ++row)
    {
        for(int col = 0; col < Geometry::kHeadDimQK; ++col)
            k[row * Probe::kInputQKStride + col] = Tag(row, col, Geometry::kHeadDimQK);
        for(int col = 0; col < Geometry::kDv; ++col)
            v[row * Policy::kVPhysicalStride + col] = Tag(row, col, Geometry::kDv);
    }
    DeviceMem dq(q.size() * sizeof(Data)), dk(k.size() * sizeof(Data)), dv(v.size() * sizeof(Data)),
        dump(Policy::kLdsArenaSize), counts(Probe::kBlockSize * sizeof(std::uint32_t));
    dq.ToDevice(q.data());
    dk.ToDevice(k.data());
    dv.ToDevice(v.data());
    const Args args{static_cast<const Data*>(dq.GetDeviceBuffer()),
                    static_cast<const Data*>(dk.GetDeviceBuffer()),
                    static_cast<const Data*>(dv.GetDeviceBuffer()),
                    static_cast<std::uint8_t*>(dump.GetDeviceBuffer()),
                    static_cast<std::uint32_t*>(counts.GetDeviceBuffer()),
                    q_rows,
                    kv_rows};
    launch_kernel(stream_config{nullptr, false, 0},
                  make_kernel<1, gfx125_t>(Probe{}, dim3(1), dim3(Probe::kBlockSize), 0, args));
    hip_check_error(hipDeviceSynchronize());
    std::array<std::uint32_t, Probe::kBlockSize> errors{};
    counts.FromDevice(errors.data());
    std::vector<std::uint8_t> actual(Policy::kLdsArenaSize), expected(actual.size(), 0xa5);
    dump.FromDevice(actual.data());
    for(int region = 0; region < 4; ++region)
    {
        const bool is_k  = region < 2;
        const int base   = std::array<int, 4>{Policy::kLdsOffsetK0,
                                              Policy::kLdsOffsetK1,
                                              Policy::kLdsOffsetV0,
                                              Policy::kLdsOffsetV1}[region];
        const int stride = is_k ? Policy::kKPhysicalStride : Policy::kVPhysicalStride;
        const int width  = is_k ? Geometry::kHeadDimQK : Geometry::kDv;
        for(int row = 0; row < Geometry::kN; ++row)
            for(int col = 0; col < (is_k ? stride : width); ++col)
            {
                const std::uint16_t bits = row < kv_rows && col < width ? Tag(row, col, width) : 0;
                const int offset         = base + sizeof(Data) * (row * stride + col);
                expected[offset]         = static_cast<std::uint8_t>(bits);
                expected[offset + 1]     = static_cast<std::uint8_t>(bits >> 8);
            }
    }
    std::uint64_t failures = 0;
    for(auto count : errors)
        failures += count;
    for(std::size_t i = 0; i < actual.size(); ++i)
        failures += actual[i] != expected[i];
    std::cout << (std::is_same_v<Data, half_t> ? "fp16 " : "bf16 ") << "D" << Dim
              << " N=" << Geometry::kN << " V=" << Geometry::kDv << " q_rows=" << q_rows
              << " kv_rows=" << kv_rows << " arena=" << actual.size()
              << " Q/K/V/P tags + arena mismatches=" << failures << '\n';
    return failures == 0;
}

template <int Dim>
bool CheckLayout(int rows)
{
    return CheckLayout<Dim>(rows, rows);
}

} // namespace

using QrTdmSchedLayout = ck_tile::test::Gfx125FmhaTest;
TEST_F(QrTdmSchedLayout, D128PackedArenaAndTail)
{
    EXPECT_TRUE(CheckLayout<128>(128));
    EXPECT_TRUE(CheckLayout<128>(127));
}
TEST_F(QrTdmSchedLayout, D192PackedArenaAndTail)
{
    EXPECT_TRUE(CheckLayout<192>(128));
    EXPECT_TRUE(CheckLayout<192>(127));
}
TEST_F(QrTdmSchedLayout, D64N64V64PackedArenaAndIndependentTails)
{
    EXPECT_TRUE((CheckLayout<64, 64, 64>(128, 64)));
    EXPECT_TRUE((CheckLayout<64, 64, 64>(127, 64)));
    EXPECT_TRUE((CheckLayout<64, 64, 64>(128, 63)));
    EXPECT_TRUE((CheckLayout<64, 64, 64>(127, 63)));
}
TEST_F(QrTdmSchedLayout, D192N64V128PackedArenaAndIndependentTails)
{
    EXPECT_TRUE((CheckLayout<192, 64, 128>(128, 64)));
    EXPECT_TRUE((CheckLayout<192, 64, 128>(127, 64)));
    EXPECT_TRUE((CheckLayout<192, 64, 128>(128, 63)));
    EXPECT_TRUE((CheckLayout<192, 64, 128>(127, 63)));
}
