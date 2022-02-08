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

// 1D Conv
struct NWC : public BaseTensorLayout
{
};

struct KXC : public BaseTensorLayout
{
};

struct NWK : public BaseTensorLayout
{
};

struct NCW : public BaseTensorLayout
{
};

struct KCX : public BaseTensorLayout
{
};

struct NKW : public BaseTensorLayout
{
};

// 2D Conv
struct NHWC : public BaseTensorLayout
{
};

struct KYXC : public BaseTensorLayout
{
};

struct NHWK : public BaseTensorLayout
{
};

struct NCHW : public BaseTensorLayout
{
};

struct KCYX : public BaseTensorLayout
{
};

struct NKHW : public BaseTensorLayout
{
};

} // namespace convolution

} // namespace tensor_layout
} // namespace ck
#endif
