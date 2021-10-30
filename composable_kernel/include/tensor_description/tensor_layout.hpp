#ifndef TENSOR_LAYOUT_HPP
#define TENSOR_LAYOUT_HPP

namespace ck {
namespace tensor_layout {

struct BaseTensorLayout
{
};

struct RowMajor : public BaseTensorLayout
{
};

struct ColumnMajor : public BaseTensorLayout
{
};

} // namespace tensor_layout
} // namespace ck
#endif
