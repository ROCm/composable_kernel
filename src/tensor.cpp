#include <boost/range/adaptor/transformed.hpp>
#include <cassert>

#include "tensor.hpp"

TensorDescriptor::TensorDescriptor() {}

TensorDescriptor::TensorDescriptor(DataType_t t, std::initializer_list<std::size_t> lens)
    : mLens(lens), mDataType(t)
{
    this->CalculateStrides();
}

TensorDescriptor::TensorDescriptor(DataType_t t,
                                   std::vector<std::size_t> lens,
                                   std::vector<std::size_t> strides)
    : mLens(lens), mStrides(strides), mDataType(t)
{
}

void TensorDescriptor::CalculateStrides()
{
    mStrides.clear();
    mStrides.resize(mLens.size(), 0);
    if(strides.empty())
        return;

    mStrides.back() = 1;
    std::partial_sum(
        mLens.rbegin(), mLens.rend() - 1, mStrides.rbegin() + 1, std::multiplies<std::size_t>());
}

std::size_t TensorDescriptor::GetDimension() const { return mLens.size(); }

std::size_t TensorDescriptor::GetElementSize() const
{
    assert(mLens.size() == mStrides.size());
    return std::accumulate(
        mLens.begin(), mLens.end(), std::size_t{1}, std::multiplies<std::size_t>());
}

std::size_t TensorDescriptor::GetElementSpace() const
{
    auto ls = mLens | boost::adaptor::transformed([](auto v) { return v - 1; });
    return std::inner_product(ls.begin(), ls.end(), mStrides.begin(), std::size_t{0}) + 1;
}
