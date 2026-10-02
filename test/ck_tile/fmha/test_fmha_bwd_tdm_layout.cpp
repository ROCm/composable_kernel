#include "example/ck_tile/01_fmha/fmha_bwd.hpp"
#include "ck_tile/host/device_memory.hpp"

#include <gtest/gtest.h>

namespace {

using namespace ck_tile;

template <typename DataType,
          index_t QueryRows,
          index_t HeadDim,
          bool Masked,
          bool StoreRandval,
          index_t Slots   = 0,
          index_t KeyRows = 64>
using LayoutProblem = BlockFmhaBwdPipelineProblem<
    DataType,
    DataType,
    DataType,
    DataType,
    float,
    float,
    float,
    DataType,
    uint8_t,
    DataType,
    DataType,
    DataType,
    DataType,
    DataType,
    DataType,
    TileFmhaBwdShape<
        sequence<QueryRows, KeyRows, HeadDim, QueryRows, HeadDim, QueryRows, 32, HeadDim, HeadDim>,
        sequence<1, 4, 1>,
        sequence<16, 16, 32>,
        sequence<4, 1, 1>,
        sequence<16, 16, 32>,
        sequence<1, 4, 1>,
        sequence<16, 16, 32>,
        sequence<4, 1, 1>,
        sequence<16, 16, 32>,
        sequence<1, 4, 1>,
        sequence<16, 16, 32>>,
    false,
    false,
    SimplifiedGenericAttentionMask<Masked>,
    BlockDropoutBwd<StoreRandval, false, StoreRandval>,
    true,
    TileFmhaBwdTraits<0, 0, BlockAttentionBiasEnum::NO_BIAS, false, -1, Slots>,
    true,
    false>;

template <typename Problem, typename Policy = BlockFmhaBwdPipelineTdmPolicy>
constexpr bool regular_layout_fits()
{
    constexpr auto staged = Policy::template GetSmemSizeStaged<Problem>();
    constexpr auto total  = Policy::template GetSmemSize<Problem>();
    constexpr auto k_bytes =
        sizeof(typename Problem::KDataType) *
        Policy::template MakeKLdsWriteBlockDescriptor<Problem>().get_element_space_size();
    constexpr auto v_bytes =
        sizeof(typename Problem::VDataType) *
        Policy::template MakeVLdsWriteBlockDescriptor<Problem>().get_element_space_size();
    constexpr auto q_bytes =
        sizeof(typename Problem::QDataType) *
        Policy::template MakeQLdsBlockDescriptor<Problem>().get_element_space_size();
    constexpr auto do_bytes =
        sizeof(typename Problem::OGradDataType) *
        Policy::template MakeOGradLdsBlockDescriptor<Problem>().get_element_space_size();
    constexpr auto lse_bytes =
        sizeof(typename Problem::LSEDataType) *
        Policy::template MakeLSEDLdsWriteBlockDescriptor<Problem>().get_element_space_size();
    constexpr auto d_bytes =
        sizeof(typename Problem::DDataType) *
        Policy::template MakeLSEDLdsWriteBlockDescriptor<Problem>().get_element_space_size();
    constexpr auto ds_bytes =
        sizeof(typename Problem::GemmDataType) *
        Policy::template MakeSGradLdsBlockDescriptor<Problem>().get_element_space_size();
    constexpr auto do_offset = Policy::template GetSmemSizeQT<Problem>();
    constexpr auto q_offset  = do_offset + Policy::template GetSmemSizeOGrad<Problem>() +
                              Policy::template GetSmemSizeOGradT<Problem>();
    constexpr auto lse_offset = q_offset + Policy::template GetSmemSizeQ<Problem>();
    constexpr auto d_offset   = lse_offset + Policy::template GetSmemSizeLSE<Problem>();
    constexpr auto ds_offset  = d_offset + Policy::template GetSmemSizeD<Problem>();
    constexpr auto dv_offset  = Policy::template GetVGradAccSmemOffset<Problem>();
    constexpr auto v_offset   = Policy::template GetVSmemOffset<Problem>();
    constexpr auto dv_bytes   = [] {
        if constexpr(Policy::template kDVInReg<Problem>)
            return index_t{0};
        else
            return static_cast<index_t>(sizeof(typename Problem::AccDataType) *
                                        Policy::template MakeVGradAccLdsBlockDescriptor<Problem>()
                                            .get_element_space_size());
    }();
    if(k_bytes > staged || do_offset + do_bytes > q_offset || q_offset + q_bytes > lse_offset ||
       lse_offset + lse_bytes > d_offset || d_offset + d_bytes > ds_offset ||
       ds_offset + ds_bytes > staged || staged > dv_offset || dv_offset + dv_bytes > v_offset ||
       v_offset + v_bytes > total || Policy::template GetSmemSizeVGradAcc<Problem>() != dv_bytes)
        return false;
    if(do_offset % 16 || q_offset % 16 || lse_offset % 16 || d_offset % 16 || dv_offset % 16 ||
       v_offset % 16)
        return false;
    for(index_t slot = 1; slot < Policy::template GetQDOSlots<Problem>(); ++slot)
    {
        const auto base  = Policy::template GetQDOSlotBase<Problem>(slot);
        const auto end   = base + q_bytes + do_bytes + lse_bytes + d_bytes;
        const auto limit = slot + 1 < Policy::template GetQDOSlots<Problem>()
                               ? Policy::template GetQDOSlotBase<Problem>(slot + 1)
                               : total;
        if(base % 16 || base < v_offset + static_cast<index_t>(v_bytes) ||
           end > static_cast<std::size_t>(limit) ||
           (base + Policy::template GetSmemSizeQ<Problem>()) % 16)
            return false;
    }
    return true;
}

static_assert(regular_layout_fits<LayoutProblem<half_t, 64, 64, false, false>>());
static_assert(regular_layout_fits<LayoutProblem<half_t, 32, 128, false, false>>());
static_assert(regular_layout_fits<LayoutProblem<bf16_t, 32, 256, false, false>>());
static_assert(regular_layout_fits<LayoutProblem<half_t, 64, 128, true, true>>());

template <typename DataType>
void check_regular_layouts()
{
    EXPECT_TRUE((regular_layout_fits<LayoutProblem<DataType, 64, 64, false, false>>()));
    EXPECT_TRUE((regular_layout_fits<LayoutProblem<DataType, 32, 128, false, false>>()));
    EXPECT_TRUE((regular_layout_fits<LayoutProblem<DataType, 32, 256, false, false>>()));
    EXPECT_TRUE((regular_layout_fits<LayoutProblem<DataType, 64, 128, true, true>>()));
    EXPECT_TRUE((regular_layout_fits<LayoutProblem<DataType, 64, 128, true, false>>()));
    EXPECT_TRUE((regular_layout_fits<LayoutProblem<DataType, 64, 128, false, false, 3>>()));
    EXPECT_TRUE((regular_layout_fits<LayoutProblem<DataType, 64, 32, false, false, 0, 128>>()));
    EXPECT_TRUE((regular_layout_fits<LayoutProblem<DataType, 64, 64, false, false, 0, 128>>()));
    EXPECT_TRUE((regular_layout_fits<LayoutProblem<DataType, 64, 128, true, true, 0, 128>>()));
    EXPECT_TRUE((regular_layout_fits<LayoutProblem<DataType, 64, 128, true, false, 0, 128>>()));
}

TEST(FmhaBwdTdmLayout, ActualDescriptorsFitLiveRegions)
{
    check_regular_layouts<half_t>();
    check_regular_layouts<bf16_t>();
}

struct UndersizedQPolicy : BlockFmhaBwdPipelineTdmPolicy
{
    template <typename Problem>
    static constexpr index_t GetSmemSizeQ()
    {
        return BlockFmhaBwdPipelineTdmPolicy::GetSmemSizeQ<Problem>() - 16;
    }
};

struct UndersizedTotalPolicy : BlockFmhaBwdPipelineTdmPolicy
{
    template <typename Problem>
    static constexpr index_t GetSmemSize()
    {
        return BlockFmhaBwdPipelineTdmPolicy::GetSmemSize<Problem>() - 16;
    }
};

TEST(FmhaBwdTdmLayout, RejectsUndersizedRegions)
{
    using Problem = LayoutProblem<half_t, 64, 64, false, false>;
    EXPECT_FALSE((regular_layout_fits<Problem, UndersizedQPolicy>()));
    EXPECT_FALSE((regular_layout_fits<Problem, UndersizedTotalPolicy>()));
}

template <typename DataType, index_t HeadDim>
using DecodeProblem = BlockFmhaBwdPipelineProblem<
    DataType,
    DataType,
    DataType,
    DataType,
    float,
    float,
    float,
    DataType,
    uint8_t,
    DataType,
    DataType,
    DataType,
    DataType,
    DataType,
    DataType,
    TileFmhaBwdShape<sequence<32, 32, HeadDim, 32, HeadDim, 32, 32, HeadDim, HeadDim>,
                     sequence<1, 1, 1>,
                     sequence<16, 16, 32>,
                     sequence<1, 1, 1>,
                     sequence<16, 16, 32>,
                     sequence<1, 1, 1>,
                     sequence<16, 16, 32>,
                     sequence<1, 1, 1>,
                     sequence<16, 16, 32>,
                     sequence<1, 1, 1>,
                     sequence<16, 16, 32>,
                     32>,
    false,
    false,
    SimplifiedGenericAttentionMask<false>,
    BlockDropoutBwd<true, false, true>,
    true,
    TileFmhaBwdTraits<0, 0, BlockAttentionBiasEnum::NO_BIAS, false>,
    false,
    true>;

template <typename Problem>
constexpr bool decode_layout_fits()
{
    using Policy = BlockFmhaBwdPipelineTrLoadTdmPolicy;
    constexpr auto k_bytes =
        sizeof(typename Problem::KDataType) *
        Policy::MakeKLdsWriteBlockDescriptor<Problem>().get_element_space_size();
    constexpr auto v_bytes =
        sizeof(typename Problem::VDataType) *
        Policy::MakeVLdsWriteBlockDescriptor<Problem>().get_element_space_size();
    constexpr auto q_bytes =
        sizeof(typename Problem::QDataType) *
        Policy::MakeQLdsWriteBlockDescriptor<Problem>().get_element_space_size();
    constexpr auto do_bytes =
        sizeof(typename Problem::OGradDataType) *
        Policy::MakeOGradLdsWriteBlockDescriptor<Problem>().get_element_space_size();
    constexpr auto ds_bytes =
        sizeof(typename Problem::GemmDataType) *
        Policy::MakeSGradLdsBlockDescriptor<Problem>().get_element_space_size();
    constexpr auto v_offset   = Policy::GetSmemSizeK<Problem>();
    constexpr auto q_offset   = Policy::GetSmemSizeOGrad<Problem>();
    constexpr auto lse_offset = q_offset + Policy::GetSmemSizeQ<Problem>();
    constexpr auto d_offset   = lse_offset + Policy::GetSmemSizeLSE<Problem>();
    constexpr auto ds_offset  = v_offset + Policy::GetSmemSizeV<Problem>();
    constexpr auto total      = Policy::GetSmemSize<Problem>();
    return k_bytes <= v_offset && v_offset + v_bytes <= ds_offset && do_bytes <= q_offset &&
           q_offset + q_bytes <= lse_offset &&
           d_offset + Policy::GetSmemSizeD<Problem>() <= total && ds_offset + ds_bytes <= total &&
           v_offset % 16 == 0 && q_offset % 16 == 0 && lse_offset % 16 == 0 && d_offset % 16 == 0 &&
           ds_offset % 16 == 0;
}

struct DecodeLayoutProbe
{
    static constexpr index_t kBlockSize = 32;
    [[maybe_unused]] __device__ void operator()(int* output) const
    {
        if(get_thread_local_1d_id() != 0)
            return;
#if defined(__gfx125__)
        static_assert(decode_layout_fits<DecodeProblem<half_t, 64>>());
        static_assert(decode_layout_fits<DecodeProblem<bf16_t, 128>>());
        output[0] = decode_layout_fits<DecodeProblem<half_t, 64>>();
        output[1] = decode_layout_fits<DecodeProblem<half_t, 128>>();
        output[2] = decode_layout_fits<DecodeProblem<bf16_t, 64>>();
        output[3] = decode_layout_fits<DecodeProblem<bf16_t, 128>>();
#else
        output[0] = 0;
#endif
    }
};

TEST(FmhaBwdTdmLayout, DecodeDescriptorsFitLiveRegions)
{
    if(!is_gfx125_supported())
        GTEST_SKIP() << "TDM decode layout requires gfx1250";
    int output[4] = {};
    DeviceMem device(sizeof(output));
    device.ToDevice(output);
    launch_kernel(stream_config{nullptr, false, 0, 0, 1},
                  make_kernel(DecodeLayoutProbe{},
                              dim3(1),
                              dim3(DecodeLayoutProbe::kBlockSize),
                              0,
                              static_cast<int*>(device.GetDeviceBuffer())));
    device.FromDevice(output);
    for(int result : output)
        EXPECT_EQ(result, 1);
}

} // namespace
