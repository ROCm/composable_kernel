#include "ck_tile/host/device_memory.hpp"
#include "ck_tile/host/kernel_launch.hpp"
#include "ck_tile/ops/fmha/pipeline/block_fmha_bwd_pipeline_tdm_policy.hpp"
#include "ck_tile/ops/fmha/pipeline/block_fmha_bwd_pipeline_trload_tdm_policy.hpp"

#include "gtest/gtest.h"

#include <cstring>
#include <vector>

namespace {

constexpr auto disabled =
    ck_tile::detail::make_fmha_bwd_tdm_padding_config<ck_tile::half_t, 96, 0>();
static_assert(!disabled[ck_tile::number<0>{}]);
static_assert(disabled[ck_tile::number<1>{}] == 0);
static_assert(disabled[ck_tile::number<2>{}] == 0);
constexpr auto enabled =
    ck_tile::detail::make_fmha_bwd_tdm_padding_config<ck_tile::half_t, 64, 8>();
static_assert(enabled[ck_tile::number<0>{}]);
static_assert(enabled[ck_tile::number<1>{}] == 3);
static_assert(enabled[ck_tile::number<2>{}] == 4);
constexpr auto largest =
    ck_tile::detail::make_fmha_bwd_tdm_padding_config<ck_tile::half_t, 512, 256>();
static_assert(largest[ck_tile::number<1>{}] == 127);
static_assert(largest[ck_tile::number<2>{}] == 7);

template <typename DataType, ck_tile::index_t Cols>
constexpr bool configurations_match()
{
    constexpr auto regular =
        ck_tile::BlockFmhaBwdPipelineTdmPolicy::GetOperandLdsPaddingConfig<DataType, Cols>();
    constexpr auto decode =
        ck_tile::BlockFmhaBwdPipelineTrLoadTdmPolicy::GetTdmPaddingConfig<DataType, Cols>();
    return regular[ck_tile::number<0>{}] == decode[ck_tile::number<0>{}] &&
           regular[ck_tile::number<1>{}] == decode[ck_tile::number<1>{}] &&
           regular[ck_tile::number<2>{}] == decode[ck_tile::number<2>{}];
}

static_assert(configurations_match<ck_tile::half_t, 64>());
static_assert(configurations_match<ck_tile::bf16_t, 128>());

template <typename DataType,
          ck_tile::index_t Rows,
          ck_tile::index_t Cols,
          bool Decode,
          bool InjectOverrun = false,
          bool ReverseOrder  = false>
struct PaddingProbe
{
    using Policy = std::conditional_t<Decode,
                                      ck_tile::BlockFmhaBwdPipelineTrLoadTdmPolicy,
                                      ck_tile::BlockFmhaBwdPipelineTdmPolicy>;
    struct Problem
    {
        using KDataType = DataType;
        struct BlockFmhaShape
        {
            static constexpr ck_tile::index_t kN0        = Rows;
            static constexpr ck_tile::index_t kQKHeaddim = Cols;
        };
    };
    static constexpr ck_tile::index_t kPad = [] {
        if constexpr(Decode)
            return Policy::kTdmLdsPad;
        else
            return Policy::kOperandLdsPad;
    }();
    static constexpr ck_tile::index_t kBlockSize = 128;
    static constexpr auto kDescriptor = Policy::template MakeKLdsWriteBlockDescriptor<Problem>();
    static constexpr ck_tile::index_t kElements     = kDescriptor.get_element_space_size();
    static constexpr ck_tile::index_t kBytes        = kElements * sizeof(DataType);
    static constexpr ck_tile::index_t kSecondOffset = (kBytes + 15) / 16 * 16;
    static constexpr ck_tile::index_t kArenaBytes   = kSecondOffset + kBytes + 256;

    __device__ void operator()(const DataType* input, unsigned char* output) const
    {
        using namespace ck_tile;
        alignas(256) __shared__ unsigned char arena[kArenaBytes];
        const index_t thread = get_thread_local_1d_id();
        for(index_t offset = thread; offset < kArenaBytes; offset += kBlockSize)
            arena[offset] = 0x5a;
        block_sync_lds();

        constexpr auto descriptor = kDescriptor;
        static_assert(kElements == (Rows - 1) * (Cols + kPad) + Cols);
        auto lds_view = make_tensor_view<address_space_enum::lds>(
            reinterpret_cast<DataType*>(arena), descriptor);
        auto lds_window =
            make_tile_window(lds_view, make_tuple(number<Rows>{}, number<Cols>{}), {0, 0});
        auto second_view = make_tensor_view<address_space_enum::lds>(
            reinterpret_cast<DataType*>(arena + kSecondOffset), descriptor);
        auto second_window =
            make_tile_window(second_view, make_tuple(number<Rows>{}, number<Cols>{}), {0, 0});
        auto input_view = make_naive_tensor_view<address_space_enum::global>(
            input, make_tuple(Rows, Cols), make_tuple(Cols, 1), number<8>{}, number<1>{});
        auto distribution = make_static_tile_distribution(
            tile_distribution_encoding<sequence<>,
                                       tuple<sequence<4, Rows / 4>, sequence<Cols>>,
                                       tuple<sequence<1>>,
                                       tuple<sequence<0>>,
                                       sequence<1, 2>,
                                       sequence<1, 0>>{},
            bool_constant<true>{});
        auto input_window = make_tile_window(
            input_view, make_tuple(number<Rows>{}, number<Cols>{}), {0, 0}, distribution);
        auto second_input_view =
            make_naive_tensor_view<address_space_enum::global>(input + Rows * Cols,
                                                               make_tuple(Rows, Cols),
                                                               make_tuple(Cols, 1),
                                                               number<8>{},
                                                               number<1>{});
        auto second_input_window = make_tile_window(
            second_input_view, make_tuple(number<Rows>{}, number<Cols>{}), {0, 0}, distribution);
        constexpr auto padding = [] {
            if constexpr(Decode)
                return BlockFmhaBwdPipelineTrLoadTdmPolicy::GetTdmPaddingConfig<DataType, Cols>();
            else
                return BlockFmhaBwdPipelineTdmPolicy::GetOperandLdsPaddingConfig<DataType, Cols>();
        }();
        TDMConfig config;
        config.pad_enable              = padding[number<0>{}];
        config.pad_config.pad_amount   = padding[number<1>{}];
        config.pad_config.pad_interval = padding[number<2>{}];
        if constexpr(ReverseOrder)
        {
            load_tile_tdm(config, second_window, second_input_window);
            load_tile_tdm(config, lds_window, input_window);
        }
        else
        {
            load_tile_tdm(config, lds_window, input_window);
            load_tile_tdm(config, second_window, second_input_window);
        }
        s_wait_tensorcnt_barrier<0>();
        block_sync_lds();
        if constexpr(InjectOverrun)
        {
            if(thread == 0)
                arena[kSecondOffset + kBytes] = 0;
            block_sync_lds();
        }
        for(index_t offset = thread; offset < kArenaBytes; offset += kBlockSize)
            output[offset] = arena[offset];
    }
};

template <typename DataType,
          ck_tile::index_t Rows,
          ck_tile::index_t Cols,
          bool Decode,
          bool InjectOverrun = false,
          bool ReverseOrder  = false>
::testing::AssertionResult run_probe()
{
    using Probe = PaddingProbe<DataType, Rows, Cols, Decode, InjectOverrun, ReverseOrder>;
    std::vector<DataType> input(2 * Rows * Cols);
    for(ck_tile::index_t index = 0; index < Rows * Cols; ++index)
    {
        input[index]               = ck_tile::type_convert<DataType>((index * 17) % 127);
        input[Rows * Cols + index] = ck_tile::type_convert<DataType>(-1 - (index * 19) % 127);
    }
    std::vector<unsigned char> expected(Probe::kArenaBytes, 0x5a);
    for(ck_tile::index_t row = 0; row < Rows; ++row)
    {
        std::memcpy(expected.data() + row * (Cols + Probe::kPad) * sizeof(DataType),
                    input.data() + row * Cols,
                    Cols * sizeof(DataType));
        std::memcpy(expected.data() + Probe::kSecondOffset +
                        row * (Cols + Probe::kPad) * sizeof(DataType),
                    input.data() + Rows * Cols + row * Cols,
                    Cols * sizeof(DataType));
    }
    std::vector<unsigned char> output(Probe::kArenaBytes);
    ck_tile::DeviceMem input_device(input.size() * sizeof(DataType));
    ck_tile::DeviceMem output_device(output.size());
    input_device.ToDevice(input.data());
    const ck_tile::stream_config stream{nullptr, false, 0, 0, 1};
    for(int iteration = 0; iteration < 10; ++iteration)
    {
        ck_tile::launch_kernel(
            stream,
            ck_tile::make_kernel(Probe{},
                                 dim3(1),
                                 dim3(Probe::kBlockSize),
                                 0,
                                 static_cast<const DataType*>(input_device.GetDeviceBuffer()),
                                 static_cast<unsigned char*>(output_device.GetDeviceBuffer())));
        output_device.FromDevice(output.data());
        for(std::size_t offset = 0; offset < output.size(); ++offset)
        {
            if(output[offset] != expected[offset])
                return ::testing::AssertionFailure()
                       << "rows=" << Rows << " cols=" << Cols << " decode=" << Decode
                       << " reverse=" << ReverseOrder << " iteration=" << iteration
                       << " byte_offset=" << offset << " second_offset=" << Probe::kSecondOffset
                       << " trailing_guard=" << (offset >= Probe::kSecondOffset + Probe::kBytes)
                       << " expected=" << static_cast<unsigned int>(expected[offset])
                       << " actual=" << static_cast<unsigned int>(output[offset]);
        }
    }
    return ::testing::AssertionSuccess();
}

template <typename DataType, ck_tile::index_t Rows, bool Decode>
void check_widths()
{
    EXPECT_TRUE((run_probe<DataType, Rows, 32, Decode>()));
    EXPECT_TRUE((run_probe<DataType, Rows, 64, Decode>()));
    EXPECT_TRUE((run_probe<DataType, Rows, 128, Decode>()));
    EXPECT_TRUE((run_probe<DataType, Rows, 256, Decode>()));
    EXPECT_TRUE((run_probe<DataType, Rows, 32, Decode, false, true>()));
    EXPECT_TRUE((run_probe<DataType, Rows, 64, Decode, false, true>()));
    EXPECT_TRUE((run_probe<DataType, Rows, 128, Decode, false, true>()));
    EXPECT_TRUE((run_probe<DataType, Rows, 256, Decode, false, true>()));
}

TEST(FmhaBwdTdmPadding, DataAndGuards)
{
    if(!ck_tile::is_gfx125_supported())
        GTEST_SKIP() << "TDM padding requires gfx1250";
    check_widths<ck_tile::half_t, 32, false>();
    check_widths<ck_tile::half_t, 64, false>();
    check_widths<ck_tile::half_t, 128, false>();
    check_widths<ck_tile::half_t, 32, true>();
    check_widths<ck_tile::half_t, 64, true>();
    check_widths<ck_tile::half_t, 128, true>();
    check_widths<ck_tile::bf16_t, 32, false>();
    check_widths<ck_tile::bf16_t, 64, false>();
    check_widths<ck_tile::bf16_t, 128, false>();
    check_widths<ck_tile::bf16_t, 32, true>();
    check_widths<ck_tile::bf16_t, 64, true>();
    check_widths<ck_tile::bf16_t, 128, true>();
}

TEST(FmhaBwdTdmPadding, DetectsTrailingOverwrite)
{
    if(!ck_tile::is_gfx125_supported())
        GTEST_SKIP() << "TDM padding requires gfx1250";
    EXPECT_FALSE((run_probe<ck_tile::half_t, 32, 64, false, true>()));
}

} // namespace
