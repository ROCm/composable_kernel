#pragma once

#include <string>
#include <utility>
#include <unordered_map>
#include <vector>

namespace ck {
namespace host {

enum class DataType_fe
{
    Half,
    Float,
    Int8,
    Int32
};

std::string To_String(DataType_fe dt);

enum class Layout_fe
{
    Row,
    Column
};

std::string To_String(Layout_fe dl);

enum class GemmType_fe
{
    Default
};

//std::string ToString(GemmType gt);

struct TensorDesc_fe
{
    DataType_fe element;
    Layout_fe layout;
};

//std::string SequenceStr(const std::vector<int>& v);

//std::string MakeTuple(const std::vector<std::string>& v);

/**template <int... xs>
const std::string S = SequenceStr({xs...});

constexpr const char* PassThrough = "ck::tensor_operation::element_wise::PassThrough";
constexpr const char* Bilinear    = "ck::tensor_operation::element_wise::Bilinear";**/

} // namespace host
} // namespace ck
