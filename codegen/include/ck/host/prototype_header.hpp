#pragma once

#include <string>
#include <stdexcept>

namespace ck::host {

enum class DataType
{
    Half,
    Float,
    BF16,
    FP8,
    BF8
};

std::string ToString(DataType type)
{
    if(type == DataType::Half)
    {
        return "ck_tile::half_t";
    }
    else if(type == DataType::Float)
    {
        return "float";
    }
    else if(type == DataType::BF16)
    {
        return "bf16_t";
    }
    else if(type == DataType::FP8)
    {
        return "fp8_t";
    }
    else if(type == DataType::BF8)
    {
        return "bf8_t";
    }
    else
    {
        throw std::runtime_error("Unknown type");
    }
}

enum class Layout
{
    RowMajor,
    ColumnMajor,
};

std::string ToString(Layout layout)
{
    if(layout == Layout::RowMajor)
    {
        return "ck_tile::tensor_layout::gemm::RowMajor";
    }
    else if(layout == Layout::ColumnMajor)
    {
        return "ck_tile::tensor_layout::gemm::ColumnMajor";
    }
    else
    {
        throw std::runtime_error("Unknown layout");
    }
}

enum class Scheduler
{
    Default,
    InterWave,
    IntraWave,
};

std::string ToString(Scheduler scheduler)
{
    if(scheduler == Scheduler::Default)
    {
        return "ck_tile::GemmPipelineScheduler::Default";
    }
    else if(scheduler == Scheduler::InterWave)
    {
        return "ck_tile::GemmPipelineScheduler::Interwave";
    }
    else if(scheduler == Scheduler::IntraWave)
    {
        return "ck_tile::GemmPipelineScheduler::Intrawave";
    }
    else
    {
        throw std::runtime_error("Unknown GemmPipelineScheduler value");
    }
}

enum class Pipeline
{
    Mem,
    V3,
    V4
};

std::string PipelineToBaseGemmPipeline(Pipeline pipeline)
{
    if(pipeline == Pipeline::Mem)
    {
        return "ck_tile::BaseGemmPipelineAgBgCrMem";
    }
    else if(pipeline == Pipeline::V3)
    {
        return "ck_tile::BaseGemmPipelineAgBgCrCompV3";
    }
    else if(pipeline == Pipeline::V4)
    {
        return "ck_tile::BaseGemmPipelineAgBgCrCompV4";
    }
    else
    {
        throw std::runtime_error("Unknown Pipeline value");
    }
}

std::string PipelineToGemmPipeline(Pipeline pipeline)
{
    if(pipeline == Pipeline::Mem)
    {
        return "ck_tile::GemmPipelineAgBgCrMem";
    }
    else if(pipeline == Pipeline::V3)
    {
        return "ck_tile::GemmPipelineAgBgCrCompV3";
    }
    else if(pipeline == Pipeline::V4)
    {
        return "ck_tile::GemmPipelineAgBgCrCompV4";
    }
    else
    {
        throw std::runtime_error("Unknown Pipeline value");
    }
}

enum class Epilogue
{
    Default2D,
    CShuffle
};

std::string ToString(Epilogue epilogue)
{
    if(epilogue == Epilogue::Default2D)
    {
        return "ck_tile::Default2DEpilogueSelector";
    }
    else if(epilogue == Epilogue::CShuffle)
    {
        return "ck_tile::CShuffleEpilogueSelector";
    }
    else
    {
        throw std::runtime_error("Unknown GemmEpilogueType value");
    }
}

std::string ToString(bool val) { return val ? "true" : "false"; }

struct GemmKernelInstanceParams
{
    Pipeline pipeline;
    Scheduler scheduler;
    Epilogue epilogue;
    int tileM;
    int tileN;
    int tileK;
    int warpM;
    int warpN;
    int warpK;
    int warpTileM;
    int warpTileN;
    int warpTileK;
};

} // namespace ck::host
