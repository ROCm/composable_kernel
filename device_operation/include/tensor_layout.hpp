#ifndef TENSOR_LAYOUT_HPP
#define TENSOR_LAYOUT_HPP

namespace ck {
namespace tensor_layout {

struct BaseTensorLayout
{
};

namespace gemm {

struct RowMajor : public BaseTensorLayout
{
};

struct ColumnMajor : public BaseTensorLayout
{
};
} // namespace gemm

namespace convolution {

struct NHWC : public BaseTensorLayout
{
};

struct KYXC : public BaseTensorLayout
{
};

struct NHWK : public BaseTensorLayout
{
};

} // namespace convolution

} // namespace tensor_layout
} // namespace ck
#endif
