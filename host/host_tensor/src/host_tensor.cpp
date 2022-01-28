#include <cassert>

#include "host_tensor.hpp"

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
    std::size_t space = 1;
    for(int i = 0; i < mLens.size(); ++i)
    {
        space += (mLens[i] - 1) * mStrides[i];
    }
    return space;
}

const std::vector<std::size_t>& HostTensorDescriptor::GetLengths() const { return mLens; }

const std::vector<std::size_t>& HostTensorDescriptor::GetStrides() const { return mStrides; }

std::ostream& operator<<(std::ostream& os, const HostTensorDescriptor& desc)
{
    os << "dim " << desc.GetNumOfDimension() << ", ";

    os << "lengths {";
    LogRange(os, desc.GetLengths(), ", ");
    os << "}, ";

    os << "strides {";
    LogRange(os, desc.GetStrides(), ", ");
    os << "}";

    return os;
}

void ostream_HostTensorDescriptor(const HostTensorDescriptor& desc, std::ostream& os)
{
    os << "dim " << desc.GetNumOfDimension() << ", ";

    os << "lengths {";
    LogRange(os, desc.GetLengths(), ", ");
    os << "}, ";

    os << "strides {";
    LogRange(os, desc.GetStrides(), ", ");
    os << "}" << std::endl;
}

float bf16_to_f32_(ushort src_val)
{
    union
    {
        uint32_t int32;
        float fp32;
    } u = {uint32_t(src_val) << 16};
    return u.fp32;
}
