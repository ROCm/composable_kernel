// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2025, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#if !defined(__HIPCC_RTC__) || !defined(CK_CODE_GEN_RTC)
#include <string>
#include <sstream>
#include <regex>
#include <optional>

#include "ck/stream_config.hpp"
#endif
#include "ck/utility/get_id.hpp"

namespace ck {
namespace tensor_operation {
namespace device {

#if !defined(__HIPCC_RTC__) || !defined(CK_CODE_GEN_RTC)
#define GET_OBJECT_NAME_IMLP                                                  \
    std::optional<std::string> GetObjectName() const override                 \
    {                                                                         \
        std::string str = __PRETTY_FUNCTION__;                                \
        static std::regex obj_name_expr{"<std::string> (.*)::GetObjectName"}; \
        std::smatch match;                                                    \
        if(!std::regex_search(str, match, obj_name_expr))                     \
        {                                                                     \
            return str;                                                       \
        }                                                                     \
        return std::string(match[1]) + ';';                                   \
    }

#define GET_TEMPLATE_INFO_IMPL                                  \
    std::optional<std::string> GetTemplateInfo() const override \
    {                                                           \
        std::string str = __PRETTY_FUNCTION__;                  \
        static std::regex template_expr{"\\[(.*)\\]"};          \
        std::smatch match;                                      \
        if(!std::regex_search(str, match, template_expr))       \
        {                                                       \
            return std::nullopt;                                \
        }                                                       \
        return std::string(match[1]);                           \
    }

#define REGISTER_EXTRA_PRINTING_METHODS GET_OBJECT_NAME_IMLP GET_TEMPLATE_INFO_IMPL
#endif

template <index_t BlockSize_,
          index_t MPerBlock_,
          index_t NPerBlock_,
          index_t MPerXDL_,
          index_t NPerXDL_,
          index_t MXdlPerWave_,
          bool IsWave64>
static constexpr auto GetNXdlPerWave2()
{
    constexpr index_t Waves  = IsWave64 ? BlockSize_ / 64 : BlockSize_ / 32;
    constexpr index_t MWaves = MPerBlock_ / (MXdlPerWave_ * MPerXDL_);
    static_assert(MWaves > 0);

    constexpr index_t NWaves = Waves / MWaves;
    if constexpr(NWaves == 0)
    {
        return 0;
    }
    else
    {
        if constexpr(NPerBlock_ % (NPerXDL_ * NWaves) == 0)
        {
            return NPerBlock_ / (NWaves * NPerXDL_);
        }
        else
        {
            return 0;
        }
    }
}

#define GET_NXDL_PER_WAVE_IMPL              \
    template <bool IsWave64>                \
    static constexpr auto GetNXdlPerWave()  \
    {                                       \
        return GetNXdlPerWave2<BlockSize,   \
                               MPerBlock,   \
                               NPerBlock,   \
                               MPerXDL,     \
                               NPerXDL,     \
                               MXdlPerWave, \
                               IsWave64>(); \
    }

#define INVOKER_RUN_IMPL                                                               \
    float Run(const Argument& arg, const StreamConfig& stream_config = StreamConfig{}) \
    {                                                                                  \
        if(get_warp_size() == 64)                                                      \
        {                                                                              \
            if constexpr(NXdlPerWave64 > 0)                                            \
            {                                                                          \
                return RunImp<GridwiseGemm64>(arg, stream_config);                     \
            }                                                                          \
        }                                                                              \
        else                                                                           \
        {                                                                              \
            if constexpr(NXdlPerWave32 > 0)                                            \
            {                                                                          \
                return RunImp<GridwiseGemm32>(arg, stream_config);                     \
            }                                                                          \
        }                                                                              \
        return 0;                                                                      \
    }

#define INVOKER_RUN3_IMPL                                                              \
    float Run(const Argument& arg, const StreamConfig& stream_config = StreamConfig{}) \
    {                                                                                  \
        if(get_warp_size() == 64)                                                      \
        {                                                                              \
            if constexpr(NXdlPerWave64 > 0)                                            \
            {                                                                          \
                return RunImp<GridwiseGemm64>(arg, stream_config);                     \
            }                                                                          \
        }                                                                              \
        else                                                                           \
        {                                                                              \
            if constexpr(NXdlPerWave32 > 0)                                            \
            {                                                                          \
                return RunImp<GridwiseGemm32>(                                         \
                    reinterpret_cast<const typename GridwiseGemm32::Argument&>(arg),   \
                    stream_config);                                                    \
            }                                                                          \
        }                                                                              \
        return 0;                                                                      \
    }

template <index_t BlockSize,
          index_t MPerBlock,
          index_t NPerBlock,
          index_t MPerXdl,
          index_t NPerXdl,
          index_t MXdlPerWave,
          index_t NXdlPerWave,
          typename CDataType,
          InMemoryDataOperationEnum CGlobalMemoryDataOperation_ = InMemoryDataOperationEnum::Set>
__device__ static bool constexpr IsValidGemmCompilationParameter()
{
#if defined(__gfx11__) || defined(__gfx12__)
    if constexpr(MPerXdl != 16 || NPerXdl != 16)
    {
        return false;
    }
#endif

#if defined(__gfx11__)
    constexpr bool SupportMemOp = CGlobalMemoryDataOperation_ == InMemoryDataOperationEnum::Set;
#else
    constexpr bool SupportMemOp =
        sizeof(CDataType) >= 2 || (CGlobalMemoryDataOperation_ == InMemoryDataOperationEnum::Set);
#endif
    if constexpr(SupportMemOp == false)
    {
        return false;
    }

    if constexpr(MXdlPerWave > 0 && NXdlPerWave > 0)
    {
        constexpr index_t MWaves = MPerBlock / (MXdlPerWave * MPerXdl);
        constexpr index_t NWaves = NPerBlock / (NXdlPerWave * NPerXdl);
        if constexpr(MWaves > 0 && NWaves > 0)
        {
            constexpr index_t WaveSize = BlockSize / (MWaves * NWaves);
            return WaveSize == get_warp_size();
        }
    }
    return false;
}

#define IS_VALID_COMPILATION_PARAMETER_IMPL(CDataType_)                       \
    template <InMemoryDataOperationEnum CGlobalMemoryDataOperation_ =         \
                  InMemoryDataOperationEnum::Set>                             \
    __device__ static bool constexpr IsValidCompilationParameter()            \
    {                                                                         \
        return ck::tensor_operation::device::IsValidGemmCompilationParameter< \
            BlockSize,                                                        \
            MPerBlock,                                                        \
            NPerBlock,                                                        \
            MPerXdl,                                                          \
            NPerXdl,                                                          \
            MXdlPerWave,                                                      \
            NXdlPerWave,                                                      \
            CDataType_,                                                       \
            CGlobalMemoryDataOperation_>();                                   \
    }

#ifndef CK_CODE_GEN_RTC
struct BaseArgument
{
    BaseArgument()                               = default;
    BaseArgument(const BaseArgument&)            = default;
    BaseArgument& operator=(const BaseArgument&) = default;

    virtual ~BaseArgument() {}

    void* p_workspace_ = nullptr;
};

struct BaseInvoker
{
    BaseInvoker()                              = default;
    BaseInvoker(const BaseInvoker&)            = default;
    BaseInvoker& operator=(const BaseInvoker&) = default;

    virtual float Run(const BaseArgument*, const StreamConfig& = StreamConfig{})
    {
        return float{0};
    }

    virtual ~BaseInvoker() {}
};
#endif

struct BaseOperator
{
    BaseOperator()                               = default;
    BaseOperator(const BaseOperator&)            = default;
    BaseOperator& operator=(const BaseOperator&) = default;
#if !defined(__HIPCC_RTC__) || !defined(CK_CODE_GEN_RTC)
    virtual bool IsSupportedArgument(const BaseArgument*) { return false; }
    virtual std::string GetTypeString() const { return ""; }
    virtual std::string GetInstanceString() const { return ""; }

    virtual std::string GetTypeIdName() const { return typeid(*this).name(); }

    virtual std::optional<std::string> GetObjectName() const { return std::nullopt; }

    virtual std::optional<std::string> GetTemplateInfo() const { return std::nullopt; }

    virtual std::string GetTypeIdHashCode() const
    {
        std::ostringstream oss;

        oss << std::hex << typeid(*this).hash_code();

        return oss.str();
    };

    virtual size_t GetWorkSpaceSize(const BaseArgument*) const { return 0; }

    virtual void SetWorkSpacePointer(BaseArgument* p_arg,
                                     void* p_workspace,
                                     const StreamConfig& = StreamConfig{}) const
    {
        assert(p_arg);
        p_arg->p_workspace_ = p_workspace;
    }
#endif
    virtual ~BaseOperator() {}
};

} // namespace device
} // namespace tensor_operation
} // namespace ck
