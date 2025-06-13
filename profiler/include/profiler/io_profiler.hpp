// SPDX-License-Identifier: MIT
// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights

#pragma once

#include <iostream>
#include <fstream>
#include <string>
#include <iomanip>
#include <chrono>
#include <sstream>
#include <cctype>
#include <cstdio>
#include <typeinfo>

#include "ck/utility/data_type.hpp"
#include "ck/utility/amd_ck_fp8.hpp"

#include "data_type_enum.hpp"
#include "profiler/argparse_wrapper.hpp"

namespace ck {
namespace profiler {
namespace io {

// Enum for output types
enum class OutputType
{
    CONSOLE, // Standard console output
    JSONL    // JSON Lines file output
};

// Layout enum for type safety
enum class LayoutType
{
    RowMajor,
    ColumnMajor,
    Unknown
};

// ResultEntry represents a single performance result
struct ResultEntry
{
    std::string op_name;
    float time_ms;
    float tflops;
    float gb_per_sec;
    bool is_valid;
    std::unordered_map<std::string, std::string> metadata;

    ResultEntry(
        const std::string& name, float time, float tflops_val, float bandwidth, bool valid = true)
        : op_name(name), time_ms(time), tflops(tflops_val), gb_per_sec(bandwidth), is_valid(valid)
    {
    }

    ResultEntry() : op_name(""), time_ms(0.0f), tflops(0.0f), gb_per_sec(0.0f), is_valid(false) {}
};

// OutputManager - Manages output formatting and destination
class OutputManager
{
    private:
    static OutputManager* instance;
    OutputType current_type;
    std::string jsonl_filename;
    std::ofstream jsonl_file;
    ResultEntry best_result;

    OutputManager() : current_type(OutputType::CONSOLE), jsonl_filename(""), best_result() {}

    std::string EscapeJsonString(const std::string& str)
    {
        std::string escaped;
        escaped.reserve(str.size() * 2);

        for(char c : str)
        {
            switch(c)
            {
            case '"': escaped += "\\\""; break;
            case '\\': escaped += "\\\\"; break;
            case '\b': escaped += "\\b"; break;
            case '\f': escaped += "\\f"; break;
            case '\n': escaped += "\\n"; break;
            case '\r': escaped += "\\r"; break;
            case '\t': escaped += "\\t"; break;
            default:
                if(c >= 0 && c < 0x20)
                {
                    char hex[7];
                    std::snprintf(hex, sizeof(hex), "\\u%04x", static_cast<unsigned char>(c));
                    escaped += hex;
                }
                else
                {
                    escaped += c;
                }
                break;
            }
        }
        return escaped;
    }

    std::string GetCurrentTimestamp()
    {
        auto now    = std::chrono::system_clock::now();
        auto time_t = std::chrono::system_clock::to_time_t(now);
        auto ms =
            std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch()) % 1000;

        std::ostringstream oss;
        oss << std::put_time(std::gmtime(&time_t), "%Y-%m-%dT%H:%M:%S");
        oss << '.' << std::setfill('0') << std::setw(3) << ms.count() << 'Z';
        return oss.str();
    }

    bool IsNumeric(const std::string& str)
    {
        if(str.empty())
            return false;

        bool has_dot = false;
        size_t start = 0;

        if(str[0] == '-')
        {
            if(str.length() == 1)
                return false;
            start = 1;
        }

        for(size_t i = start; i < str.length(); ++i)
        {
            char c = str[i];
            if(std::isdigit(c))
            {
                continue;
            }
            else if(c == '.' && !has_dot)
            {
                has_dot = true;
            }
            else
            {
                return false;
            }
        }
        return true;
    }

    void OutputToConsole(const ResultEntry& result, bool is_best)
    {
        if(is_best)
        {
            std::cout << "Best Perf: " << result.time_ms << " ms, " << result.tflops << " TFlops, "
                      << result.gb_per_sec << " GB/s, " << result.op_name << std::endl;
        }
        else
        {
            std::cout << "Perf: " << std::setw(10) << result.time_ms << " ms, " << result.tflops
                      << " TFlops, " << result.gb_per_sec << " GB/s, " << result.op_name
                      << std::endl;
        }
    }

    void OutputToJsonl(const ResultEntry& result, bool is_best)
    {
        if(!jsonl_file.is_open())
        {
            OutputToConsole(result, is_best);
            return;
        }

        jsonl_file << "{";
        jsonl_file << "\"operation\":\"" << EscapeJsonString(result.op_name) << "\",";
        jsonl_file << "\"time_ms\":" << result.time_ms << ",";
        jsonl_file << "\"tflops\":" << result.tflops << ",";
        jsonl_file << "\"gb_per_sec\":" << result.gb_per_sec << ",";
        jsonl_file << "\"is_best\":" << (is_best ? "true" : "false") << ",";
        jsonl_file << "\"timestamp\":\"" << GetCurrentTimestamp() << "\"";

        if(!result.metadata.empty())
        {
            for(const auto& [key, value] : result.metadata)
            {
                jsonl_file << ",\"" << EscapeJsonString(key) << "\":";

                if(IsNumeric(value))
                {
                    jsonl_file << value;
                }
                else
                {
                    jsonl_file << "\"" << EscapeJsonString(value) << "\"";
                }
            }

            jsonl_file << ",\"operation_params\":{";
            bool first = true;
            for(const auto& [key, value] : result.metadata)
            {
                if(!first)
                    jsonl_file << ",";
                jsonl_file << "\"" << EscapeJsonString(key) << "\":";

                if(IsNumeric(value))
                {
                    jsonl_file << value;
                }
                else
                {
                    jsonl_file << "\"" << EscapeJsonString(value) << "\"";
                }
                first = false;
            }
            jsonl_file << "}";
        }

        jsonl_file << "}" << std::endl;
        jsonl_file.flush();
    }

    public:
    static OutputManager& GetInstance()
    {
        if(instance == nullptr)
        {
            instance = new OutputManager();
        }
        return *instance;
    }

    void SetOutputType(OutputType type, const std::string& filename = "")
    {
        current_type = type;

        if(jsonl_file.is_open())
        {
            jsonl_file.close();
        }

        if(type == OutputType::JSONL && !filename.empty())
        {
            jsonl_filename = filename;
            jsonl_file.open(filename);

            if(!jsonl_file.is_open())
            {
                std::cerr << "Error: Could not open JSONL file: " << filename << std::endl;
                std::cerr << "Falling back to console output." << std::endl;
                current_type = OutputType::CONSOLE;
            }
        }
    }

    OutputType GetOutputType() const { return current_type; }

    void OutputResult(const ResultEntry& result)
    {
        if(!best_result.is_valid || result.tflops > best_result.tflops)
        {
            best_result = result;
        }

        switch(current_type)
        {
        case OutputType::JSONL: OutputToJsonl(result, false); break;
        case OutputType::CONSOLE:
        default: OutputToConsole(result, false); break;
        }
    }

    void OutputBestResult()
    {
        if(!best_result.is_valid)
        {
            return;
        }

        switch(current_type)
        {
        case OutputType::JSONL: OutputToJsonl(best_result, true); break;
        case OutputType::CONSOLE:
        default: OutputToConsole(best_result, true); break;
        }
    }

    ~OutputManager()
    {
        if(jsonl_file.is_open())
        {
            jsonl_file.close();
        }
    }
};

// Utility functions for type introspection and conversion
namespace utils {

// Convert DataTypeEnum to string
inline std::string DataTypeToString(DataTypeEnum data_type)
{
    switch(data_type)
    {
    case DataTypeEnum::Half: return "f16";
    case DataTypeEnum::Float: return "f32";
    case DataTypeEnum::Int32: return "i32";
    case DataTypeEnum::Int8: return "i8";
    case DataTypeEnum::Int8x4: return "i8x4";
    case DataTypeEnum::BFloat16: return "bf16";
    case DataTypeEnum::Double: return "f64";
    case DataTypeEnum::Float8: return "fp8";
    case DataTypeEnum::Unknown:
    default: return "unknown";
    }
}

// Convert LayoutType to string
inline std::string LayoutToString(LayoutType layout)
{
    switch(layout)
    {
    case LayoutType::RowMajor: return "RowMajor";
    case LayoutType::ColumnMajor: return "ColumnMajor";
    case LayoutType::Unknown:
    default: return "unknown";
    }
}

// Template function to automatically detect data type from C++ type
template <typename DataType>
DataTypeEnum GetDataTypeEnum()
{
    if constexpr(std::is_same_v<DataType, float>)
    {
        return DataTypeEnum::Float;
    }
    else if constexpr(std::is_same_v<DataType, double>)
    {
        return DataTypeEnum::Double;
    }
    else if constexpr(std::is_same_v<DataType, int8_t>)
    {
        return DataTypeEnum::Int8;
    }
    else if constexpr(std::is_same_v<DataType, int32_t>)
    {
        return DataTypeEnum::Int32;
    }

    // Add explicit checks for CK-specific types
    else if constexpr(std::is_same_v<DataType, ck::half_t>) {
        return DataTypeEnum::Half;
    } else if constexpr(std::is_same_v<DataType, ck::bhalf_t>) {
        return DataTypeEnum::BFloat16;
    } else if constexpr(std::is_same_v<DataType, ck::f8_t>) {
        return DataTypeEnum::Float8;
    }
    // Handle potential alternative type names that might be used
    else if constexpr(std::is_same_v<DataType, __half>) {
        return DataTypeEnum::Half;
    }
    // Fallback for any 1-byte floating point types that aren't int8_t
    else if constexpr(sizeof(DataType) == 1 && std::is_floating_point_v<DataType>) {
        return DataTypeEnum::Float8;
    }
    // Fallback for 2-byte types that might be half precision
    else if constexpr(sizeof(DataType) == 2 && std::is_floating_point_v<DataType>) {
        // Try to distinguish between fp16 and bf16 using typeid as last resort
        std::string type_name = typeid(DataType).name();
        if (type_name.find("bhalf") != std::string::npos || type_name.find("bf16") != std::string::npos) {
            return DataTypeEnum::BFloat16;
        } else {
            return DataTypeEnum::Half;
        }
    }
    // Additional fallback using typeid for types we might have missed
    else {
        std::string type_name = typeid(DataType).name();
        if (type_name.find("half") != std::string::npos && type_name.find("bhalf") == std::string::npos) {
            return DataTypeEnum::Half;
        }
        else if(type_name.find("bhalf") != std::string::npos)
        {
            return DataTypeEnum::BFloat16;

        } else if (type_name.find("f8") != std::string::npos || type_name.find("fp8") != std::string::npos) {
            return DataTypeEnum::Float8;
        }
        return DataTypeEnum::Unknown;
    }
}

// Template function to automatically detect layout type
template <typename Layout>
LayoutType GetLayoutType()
{
    std::string type_name = typeid(Layout).name();
    if(type_name.find("RowMajor") != std::string::npos)
    {
        return LayoutType::RowMajor;
    }
    else if(type_name.find("ColumnMajor") != std::string::npos)
    {
        return LayoutType::ColumnMajor;
    }
    return LayoutType::Unknown;
}

} // namespace utils

// Template functions for automatic metadata creation - ALL LOGIC STAYS HERE
template <typename ALayout,
          typename BLayout,
          typename CLayout,
          typename ADataType,
          typename BDataType,
          typename CDataType>
inline std::unordered_map<std::string, std::string> CreateGemmMetadata(int M, int N, int K)
{
    std::unordered_map<std::string, std::string> metadata;

    // Automatically detect types using template introspection
    auto a_datatype = utils::GetDataTypeEnum<ADataType>();
    auto b_datatype = utils::GetDataTypeEnum<BDataType>();
    auto c_datatype = utils::GetDataTypeEnum<CDataType>();

    auto a_layout = utils::GetLayoutType<ALayout>();
    auto b_layout = utils::GetLayoutType<BLayout>();
    auto c_layout = utils::GetLayoutType<CLayout>();

    metadata["operation_type"]  = "gemm";
    metadata["datatype"]        = utils::DataTypeToString(c_datatype);
    metadata["input_datatype"]  = utils::DataTypeToString(a_datatype);
    metadata["weight_datatype"] = utils::DataTypeToString(b_datatype);
    metadata["output_datatype"] = utils::DataTypeToString(c_datatype);
    metadata["layout_a"]        = utils::LayoutToString(a_layout);
    metadata["layout_b"]        = utils::LayoutToString(b_layout);
    metadata["layout_c"]        = utils::LayoutToString(c_layout);
    metadata["M"]               = std::to_string(M);
    metadata["N"]               = std::to_string(N);
    metadata["K"]               = std::to_string(K);

    return metadata;
}

// Template function for convolution metadata
template <typename InputDataType, typename WeightDataType, typename OutputDataType>
inline std::unordered_map<std::string, std::string> CreateConvMetadata(
    int N, int C, int H, int W, int K, int Y, int X, const std::string& conv_type = "conv2d")
{

    std::unordered_map<std::string, std::string> metadata;

    auto input_dtype  = utils::GetDataTypeEnum<InputDataType>();
    auto weight_dtype = utils::GetDataTypeEnum<WeightDataType>();
    auto output_dtype = utils::GetDataTypeEnum<OutputDataType>();

    metadata["operation_type"]  = conv_type;
    metadata["input_datatype"]  = utils::DataTypeToString(input_dtype);
    metadata["weight_datatype"] = utils::DataTypeToString(weight_dtype);
    metadata["output_datatype"] = utils::DataTypeToString(output_dtype);
    metadata["N"]               = std::to_string(N);
    metadata["C"]               = std::to_string(C);
    metadata["H"]               = std::to_string(H);
    metadata["W"]               = std::to_string(W);
    metadata["K"]               = std::to_string(K);
    metadata["Y"]               = std::to_string(Y);
    metadata["X"]               = std::to_string(X);

    return metadata;
}

// Template function for normalization metadata
template <typename DataType>
inline std::unordered_map<std::string, std::string>
CreateNormMetadata(const std::vector<int>& dimensions, const std::string& norm_type)
{

    std::unordered_map<std::string, std::string> metadata;

    auto dtype = utils::GetDataTypeEnum<DataType>();

    metadata["operation_type"] = norm_type;
    metadata["datatype"]       = utils::DataTypeToString(dtype);

    for(size_t i = 0; i < dimensions.size(); ++i)
    {
        metadata["dim" + std::to_string(i)] = std::to_string(dimensions[i]);
    }

    return metadata;
}

// Generic operation metadata
inline std::unordered_map<std::string, std::string>
CreateOperationMetadata(const std::string& operation_type,
                        const std::unordered_map<std::string, std::string>& additional_fields = {})
{

    std::unordered_map<std::string, std::string> metadata;
    metadata["operation_type"] = operation_type;

    for(const auto& [key, value] : additional_fields)
    {
        metadata[key] = value;
    }

    return metadata;
}

// Main reporting functions - kernel implementations only call these
inline void SetOutputType(OutputType type, const std::string& filename = "")
{
    OutputManager::GetInstance().SetOutputType(type, filename);
}

inline OutputType GetOutputType() { return OutputManager::GetInstance().GetOutputType(); }

inline void ReportResult(const std::string& op_name,
                         float time_ms,
                         float tflops,
                         float gb_per_sec,
                         const std::unordered_map<std::string, std::string>& metadata)
{
    ResultEntry result(op_name, time_ms, tflops, gb_per_sec);
    result.metadata = metadata;
    OutputManager::GetInstance().OutputResult(result);
}

inline void ReportResult(const std::string& op_name, float time_ms, float tflops, float gb_per_sec)
{
    std::unordered_map<std::string, std::string> empty_metadata;
    ReportResult(op_name, time_ms, tflops, gb_per_sec, empty_metadata);
}

inline void ReportBestResult() { OutputManager::GetInstance().OutputBestResult(); }

inline void AddOutputArguments(argparse::ArgumentParser& parser)
{
    parser.add_argument("-o", "--output")
        .help("Output format (console, jsonl=<filename>)")
        .default_value(std::string("console"));
}

inline void ParseOutputArguments(const argparse::ArgumentParser& parser)
{
    try
    {
        std::string output_format = parser.get<std::string>("--output");

        if(output_format == "console")
        {
            SetOutputType(OutputType::CONSOLE);
        }
        else if(output_format.compare(0, 6, "jsonl=") == 0)
        {
            std::string filename = output_format.substr(6);
            if(!filename.empty())
            {
                SetOutputType(OutputType::JSONL, filename);
            }
            else
            {
                std::cerr
                    << "Error: JSONL output requires a filename (use --output=jsonl=filename.jsonl)"
                    << std::endl;
                SetOutputType(OutputType::CONSOLE);
            }
        }
        else
        {
            std::cerr << "Warning: Unknown output format '" << output_format << "', using console"
                      << std::endl;
            SetOutputType(OutputType::CONSOLE);
        }
    }
    catch(const std::exception& err)
    {
        std::cerr << "Error parsing output arguments: " << err.what() << std::endl;
        SetOutputType(OutputType::CONSOLE);
    }
}

} // namespace io
} // namespace profiler
} // namespace ck
