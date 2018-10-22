#include <boost/range/adaptor/transformed.hpp>
#include <cassert>

#include "tensor.hpp"

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
    if(mStrides.empty())
        return;

    mStrides.back() = 1;
    std::partial_sum(
        mLens.rbegin(), mLens.rend() - 1, mStrides.rbegin() + 1, std::multiplies<std::size_t>());
}

DataType_t TensorDescriptor::GetDataType() const { return mDataType; }

std::size_t TensorDescriptor::GetDimension() const { return mLens.size(); }

std::size_t TensorDescriptor::GetElementSize() const
{
    assert(mLens.size() == mStrides.size());
    return std::accumulate(
        mLens.begin(), mLens.end(), std::size_t{1}, std::multiplies<std::size_t>());
}

std::size_t TensorDescriptor::GetElementSpace() const
{
    auto ls = mLens | boost::adaptors::transformed([](std::size_t v) { return v - 1; });
    return std::inner_product(ls.begin(), ls.end(), mStrides.begin(), std::size_t{0}) + 1;
}

const std::vector<std::size_t>& TensorDescriptor::GetLengths() const { return mLens; }

const std::vector<std::size_t>& TensorDescriptor::GetStrides() const { return mStrides; }
