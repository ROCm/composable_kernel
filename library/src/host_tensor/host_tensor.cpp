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

#if 1
// FIXME: remove
void bf16_to_f32_(const Tensor<ck::bhalf_t>& src, Tensor<float>& dst)
{
    for(int i = 0; i < src.mData.size(); ++i)
        dst.mData[i] = ck::type_convert<float>(src.mData[i]);
}
#endif
