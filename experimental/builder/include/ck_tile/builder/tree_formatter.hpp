#pragma once

#include <sstream>
#include <string>
#include <type_traits>
#include <vector>

#include <ck_tile/builder/conv_signature.hpp>
#include <ck_tile/builder/conv_traits.hpp>

namespace ck_tile::reflect {

// Convert CK block GEMM pipeline enums to string.
// TODO: Remove this once we hide the pipeline version from reflection.
constexpr std::string_view PipelineToString(ck::BlockGemmPipelineVersion pipeline)
{
    switch(pipeline)
    {
    case ck::BlockGemmPipelineVersion::v1: return "V1";
    case ck::BlockGemmPipelineVersion::v2: return "V2";
    case ck::BlockGemmPipelineVersion::v3: return "V3";
    case ck::BlockGemmPipelineVersion::v4: return "V4";
    case ck::BlockGemmPipelineVersion::v5: return "V5";
    default: return "Unknown";
    }
}
// enum class PipelineVersion;

// // Forward declare PipelineToString (actual definition must be included separately)
// constexpr std::string_view PipelineToString(PipelineVersion);

// Helper class for formatting hierarchical tree structures with proper indentation
// and tree-drawing characters (├─, └─, │, etc.)
class TreeFormatter
{
    public:
    TreeFormatter() = default;

    // Write a line at the specified indentation level (branch continues after this)
    template <typename... Args>
    void writeLine(int indent_level, Args&&... args)
    {
        writeLineImpl(indent_level, false, std::forward<Args>(args)...);
    }

    // Write the last line at the specified indentation level (branch ends)
    template <typename... Args>
    void writeLastLine(int indent_level, Args&&... args)
    {
        writeLineImpl(indent_level, true, std::forward<Args>(args)...);
    }

    // Get the formatted string (removes trailing newline if present)
    std::string getString() const
    {
        std::string result = oss_.str();
        if(!result.empty() && result.back() == '\n')
        {
            result.pop_back();
        }
        return result;
    }

    private:
    std::ostringstream oss_;
    std::vector<bool> is_last_at_level_; // Tracks which levels have ended

    // Helper to format individual arguments with automatic type conversion
    template <typename T>
    void formatArg(const T& arg)
    {
        if constexpr(std::is_same_v<std::remove_cv_t<std::remove_reference_t<T>>,
                                    builder::DataType>)
        {
            oss_ << builder::DataTypeToString(arg);
        }
        else if constexpr(std::is_same_v<std::remove_cv_t<std::remove_reference_t<T>>,
                                         builder::ConvDirection>)
        {
            oss_ << builder::ConvDirectionToString(arg);
        }
        else if constexpr(std::is_same_v<std::remove_cv_t<std::remove_reference_t<T>>,
                                         builder::GroupConvLayout>)
        {
            oss_ << builder::LayoutToString(arg);
        }
        else if constexpr(std::is_same_v<std::remove_cv_t<std::remove_reference_t<T>>,
                                         ck::BlockGemmPipelineVersion>)
        {
            oss_ << PipelineToString(arg);
        }
        else
        {
            oss_ << arg; // Default: just stream it
        }
    }

    // Implementation of line writing with tree symbols
    template <typename... Args>
    void writeLineImpl(int indent_level, bool is_last, Args&&... args)
    {
        // Ensure we have enough tracking space
        if(static_cast<size_t>(indent_level) >= is_last_at_level_.size())
        {
            is_last_at_level_.resize(indent_level + 1, false);
            // Level 0 (root) should always be treated as "last" since it has no tree symbols
            if(is_last_at_level_.size() > 0)
            {
                is_last_at_level_[0] = true;
            }
        }

        // Draw the tree structure
        // Start from level 1 (skip level 0 which is the root with no symbols)
        for(int i = 1; i < indent_level; ++i)
        {
            // For all parent levels, draw vertical line or space based on whether they ended
            oss_ << (is_last_at_level_[i] ? "   " : "│  ");
        }

        // Draw the branch symbol for the current level
        if(indent_level > 0)
        {
            oss_ << (is_last ? "└─ " : "├─ ");
        }

        // Write the content using fold expression with formatArg
        (formatArg(std::forward<Args>(args)), ...);

        oss_ << '\n';

        // Update tracking for this level AFTER writing the line
        // This ensures future lines at deeper levels know if this level ended
        is_last_at_level_[indent_level] = is_last;
    }
};

} // namespace ck_tile::reflect
