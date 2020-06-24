#include <boost/range/adaptor/transformed.hpp>
#include <cassert>

#include "host_tensor.hpp"

template <typename X>
HostTensorDescriptor::HostTensorDescriptor(std::vector<X> lens) : mLens(lens)
{
    this->CalculateStrides();
}

template <typename X, typename Y>
HostTensorDescriptor::HostTensorDescriptor(std::vector<X> lens, std::vector<Y> strides)
    : mLens(lens), mStrides(strides)
{
}

void HostTensorDescriptor::CalculateStrides()
{
    mStrides.clear();
    mStrides.resize(mLens.size(), 0);
    if(mStrides.empty())
        return;

    mStrides.back() = 1;
    std::partial_sum(
        mLens.rbegin(), mLens.rend() - 1, mStrides.rbegin() + 1, std::multiplies<std::size_t>());
}

std::size_t HostTensorDescriptor::GetNumOfDimension() const { return mLens.size(); }

std::size_t HostTensorDescriptor::GetElementSize() const
{
    assert(mLens.size() == mStrides.size());
    return std::accumulate(
        mLens.begin(), mLens.end(), std::size_t{1}, std::multiplies<std::size_t>());
}

std::size_t HostTensorDescriptor::GetElementSpace() const
{
    auto ls = mLens | boost::adaptors::transformed([](std::size_t v) { return v - 1; });
    return std::inner_product(ls.begin(), ls.end(), mStrides.begin(), std::size_t{0}) + 1;
}

const std::vector<std::size_t>& HostTensorDescriptor::GetLengths() const { return mLens; }

const std::vector<std::size_t>& HostTensorDescriptor::GetStrides() const { return mStrides; }
