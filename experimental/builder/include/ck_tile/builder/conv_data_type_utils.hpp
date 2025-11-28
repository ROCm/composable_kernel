// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/builder/conv_signature_concepts.hpp"
#include "ck_tile/builder/types.hpp"

namespace ck_tile::builder::factory_internal
{

struct CK_half
{
    using type = ck::half_t;
};

struct CK_bhalf
{
    using type = ck::bhalf_t;
};

struct CK_float
{
    using type = float;
};

struct CK_int8
{
    using type = int8_t;
};

struct CK_f8
{
    using type = ck::f8_t;
};

struct CK_int32
{
    using type = int32_t;
};

struct CK_empty_tuple
{
    using type = ck::Tuple<>;
};

template <DataType dt>
consteval auto ConvertDataTypeToCK()
{
    if constexpr (dt == DataType::FP16)
    {
        return CK_half{};
    }
    else if constexpr (dt == DataType::BF16)
    {
        return CK_bhalf{};
    }
    else if constexpr (dt == DataType::FP32)
    {
        return CK_float{};
    }
    else if constexpr (dt == DataType::I8)
    {
        return CK_int8{};
    }
    else if constexpr (dt == DataType::FP8)
    {
        return CK_f8{};
    }
    else
    {
        static_assert(sizeof(UnsupportedEnumValue<dt>) == 0,
                      "Internal error. Unsupported data type conversion to CK.");
    }
}

template <TensorConfig Config, DataType SignatureDataType>
consteval auto GetTensorDataAndComputeTypes()
{
    constexpr auto data_type = Config.data_type;
    constexpr auto compute_type = Config.compute_type;

    if constexpr (data_type == DataType::UNDEFINDED && compute_type == DataType::UNDEFINDED)
    {
        return std::make_pair(ConvertDataTypeToCK<SignatureDataType>(), ConvertDataTypeToCK<SignatureDataType>());
    }
    else if constexpr (data_type == DataType::UNDEFINDED)
    {
        return std::make_pair(ConvertDataTypeToCK<SignatureDataType>(), ConvertDataTypeToCK<compute_type>());
    }
    else if constexpr (compute_type == DataType::UNDEFINDED)
    {
        return std::make_pair(ConvertDataTypeToCK<data_type>(), ConvertDataTypeToCK<SignatureDataType>());
    }
    else 
    {
        return std::make_pair(ConvertDataTypeToCK<data_type>(), ConvertDataTypeToCK<compute_type>());
    }
}

template <DataType SignatureAccDataType, DataType SignatureDataType>
consteval auto GetTensorAccumulationType()
{
    constexpr auto data_type = SignatureAccDataType;
    if constexpr (data_type == DataType::UNDEFINDED)
    {
        return ConvertDataTypeToCK<SignatureDataType>();
    }
    else 
    {
        return ConvertDataTypeToCK<data_type>();
    }
}

template <auto Config, DataType SignatureDataType>
consteval auto GetAuxiliaryTensorDataTypeValue()
{
    constexpr auto data_type = Config.data_type;
    if constexpr (data_type == DataType::UNDEFINDED)
    {
        return ConvertDataTypeToCK<SignatureDataType>();
    }
    else
    {
        return ConvertDataTypeToCK<data_type>();
    }
}

template <auto AuxiliaryTensorConfigsArray, DataType SignatureDataType, size_t... Indices>
consteval auto GetAuxiliaryTensorDataTypeTuple(std::index_sequence<Indices...>)
{
    return ck::Tuple<typename decltype(GetAuxiliaryTensorDataTypeValue<AuxiliaryTensorConfigsArray[Indices], SignatureDataType>())::type...>{};
}

template <auto AuxiliaryTensorConfigsValue, DataType SignatureDataType>
struct AuxiliaryTensorDataTypes
{
    static constexpr auto Size = AuxiliaryTensorConfigsValue.size();
    using type = decltype(GetAuxiliaryTensorDataTypeTuple<AuxiliaryTensorConfigsValue, SignatureDataType>(std::make_index_sequence<Size>{}));
};

// TODO: Currently only the ouput tensor can have auxiliary tensors (e.g., bias).
template <auto Signature>
requires (HasElementwiseOpWithAuxiliaryOperands<decltype(Signature.output)>)
consteval auto GetAuxiliaryTensorDataTypes()
{
    return AuxiliaryTensorDataTypes<Signature.output.operation.auxiliary_operand_configs, Signature.data_type>{};
}

template <auto Signature>
requires (!HasElementwiseOpWithAuxiliaryOperands<decltype(Signature.output)>)
consteval auto GetAuxiliaryTensorDataTypes()
{
    return CK_empty_tuple{};
}

template <auto Signature>
struct FwdConvTensorDataTypes
{
    static constexpr auto input_types =
        GetTensorDataAndComputeTypes<Signature.input.config, Signature.data_type>();
    static constexpr auto weight_types =
        GetTensorDataAndComputeTypes<Signature.weight.config, Signature.data_type>();
    static constexpr auto output_types =
        GetTensorDataAndComputeTypes<Signature.output.config, Signature.data_type>();

    using ADataType    = typename decltype(input_types.first)::type;
    using AComputeType = typename decltype(input_types.second)::type;
    using BDataType    = typename decltype(weight_types.first)::type;
    using BComputeType = typename decltype(weight_types.second)::type;
    using AccDataType  = typename decltype(
        GetTensorAccumulationType<Signature.accumulation_data_type, Signature.data_type>())::type;
    using EDataType    = typename decltype(output_types.first)::type;

    // This is the "compute" type for output.
    using CShuffleDataType    = typename decltype(output_types.second)::type;

    // Data types for the auxiliary tensors (e.g., bias).
    using DsDataType = typename decltype(GetAuxiliaryTensorDataTypes<Signature>())::type;
};

}
