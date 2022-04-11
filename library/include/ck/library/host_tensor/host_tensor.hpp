#ifndef HOST_TENSOR_HPP
#define HOST_TENSOR_HPP

#include <thread>
#include <vector>
#include <numeric>
#include <algorithm>
#include <utility>
#include <cassert>
#include <iostream>
#include "data_type.hpp"

template <typename Range>
std::ostream& LogRange(std::ostream& os, Range&& range, std::string delim)
{
    bool first = true;
    for(auto&& v : range)
    {
        if(first)
            first = false;
        else
            os << delim;
        os << v;
    }
    return os;
}

template <typename T, typename Range>
std::ostream& LogRangeAsType(std::ostream& os, Range&& range, std::string delim)
{
    bool first = true;
    for(auto&& v : range)
    {
        if(first)
            first = false;
        else
            os << delim;
        os << static_cast<T>(v);
    }
    return os;
}

template <typename F, typename T, std::size_t... Is>
auto call_f_unpack_args_impl(F f, T args, std::index_sequence<Is...>)
{
    return f(std::get<Is>(args)...);
}

template <typename F, typename T>
auto call_f_unpack_args(F f, T args)
{
    constexpr std::size_t N = std::tuple_size<T>{};

    return call_f_unpack_args_impl(f, args, std::make_index_sequence<N>{});
}

template <typename F, typename T, std::size_t... Is>
auto construct_f_unpack_args_impl(T args, std::index_sequence<Is...>)
{
    return F(std::get<Is>(args)...);
}

template <typename F, typename T>
auto construct_f_unpack_args(F, T args)
{
    constexpr std::size_t N = std::tuple_size<T>{};

    return construct_f_unpack_args_impl<F>(args, std::make_index_sequence<N>{});
}

struct HostTensorDescriptor
{
    HostTensorDescriptor() = delete;

    template <typename X>
    HostTensorDescriptor(const std::vector<X>& lens);

    template <typename X, typename Y>
    HostTensorDescriptor(const std::vector<X>& lens, const std::vector<Y>& strides);

    void CalculateStrides();

    template <typename Range>
    HostTensorDescriptor(const Range& lens) : mLens(lens.begin(), lens.end())
    {
        this->CalculateStrides();
    }

    template <typename Range1, typename Range2>
    HostTensorDescriptor(const Range1& lens, const Range2& strides)
        : mLens(lens.begin(), lens.end()), mStrides(strides.begin(), strides.end())
    {
    }

    std::size_t GetNumOfDimension() const;
    std::size_t GetElementSize() const;
    std::size_t GetElementSpace() const;

    const std::vector<std::size_t>& GetLengths() const;
    const std::vector<std::size_t>& GetStrides() const;

    template <typename... Is>
    std::size_t GetOffsetFromMultiIndex(Is... is) const
    {
        assert(sizeof...(Is) == this->GetNumOfDimension());
        std::initializer_list<std::size_t> iss{static_cast<std::size_t>(is)...};
        return std::inner_product(iss.begin(), iss.end(), mStrides.begin(), std::size_t{0});
    }

    friend std::ostream& operator<<(std::ostream& os, const HostTensorDescriptor& desc);

    private:
    std::vector<std::size_t> mLens;
    std::vector<std::size_t> mStrides;
};

struct joinable_thread : std::thread
{
    template <typename... Xs>
    joinable_thread(Xs&&... xs) : std::thread(std::forward<Xs>(xs)...)
    {
    }

    joinable_thread(joinable_thread&&) = default;
    joinable_thread& operator=(joinable_thread&&) = default;

    ~joinable_thread()
    {
        if(this->joinable())
            this->join();
    }
};

template <typename F, typename... Xs>
struct ParallelTensorFunctor
{
    F mF;
    static constexpr std::size_t NDIM = sizeof...(Xs);
    std::array<std::size_t, NDIM> mLens;
    std::array<std::size_t, NDIM> mStrides;
    std::size_t mN1d;

    ParallelTensorFunctor(F f, Xs... xs) : mF(f), mLens({static_cast<std::size_t>(xs)...})
    {
        mStrides.back() = 1;
        std::partial_sum(mLens.rbegin(),
                         mLens.rend() - 1,
                         mStrides.rbegin() + 1,
                         std::multiplies<std::size_t>());
        mN1d = mStrides[0] * mLens[0];
    }

    std::array<std::size_t, NDIM> GetNdIndices(std::size_t i) const
    {
        std::array<std::size_t, NDIM> indices;

        for(int idim = 0; idim < NDIM; ++idim)
        {
            indices[idim] = i / mStrides[idim];
            i -= indices[idim] * mStrides[idim];
        }

        return indices;
    }

    void operator()(std::size_t num_thread = 1) const
    {
        std::size_t work_per_thread = (mN1d + num_thread - 1) / num_thread;

        std::vector<joinable_thread> threads(num_thread);

        for(std::size_t it = 0; it < num_thread; ++it)
        {
            std::size_t iw_begin = it * work_per_thread;
            std::size_t iw_end   = std::min((it + 1) * work_per_thread, mN1d);

            auto f = [=] {
                for(std::size_t iw = iw_begin; iw < iw_end; ++iw)
                {
                    call_f_unpack_args(mF, GetNdIndices(iw));
                }
            };
            threads[it] = joinable_thread(f);
        }
    }
};

template <typename F, typename... Xs>
auto make_ParallelTensorFunctor(F f, Xs... xs)
{
    return ParallelTensorFunctor<F, Xs...>(f, xs...);
}

template <typename T>
struct Tensor
{
    template <typename X>
    Tensor(std::initializer_list<X> lens) : mDesc(lens), mData(mDesc.GetElementSpace())
    {
    }

    template <typename X>
    Tensor(std::vector<X> lens) : mDesc(lens), mData(mDesc.GetElementSpace())
    {
    }

    template <typename X, typename Y>
    Tensor(std::vector<X> lens, std::vector<Y> strides)
        : mDesc(lens, strides), mData(mDesc.GetElementSpace())
    {
    }

    Tensor(const HostTensorDescriptor& desc) : mDesc(desc), mData(mDesc.GetElementSpace()) {}

    template <typename G>
    void GenerateTensorValue(G g, std::size_t num_thread = 1)
    {
        switch(mDesc.GetNumOfDimension())
        {
        case 1: {
            auto f = [&](auto i) { (*this)(i) = g(i); };
            make_ParallelTensorFunctor(f, mDesc.GetLengths()[0])(num_thread);
            break;
        }
        case 2: {
            auto f = [&](auto i0, auto i1) { (*this)(i0, i1) = g(i0, i1); };
            make_ParallelTensorFunctor(f, mDesc.GetLengths()[0], mDesc.GetLengths()[1])(num_thread);
            break;
        }
        case 3: {
            auto f = [&](auto i0, auto i1, auto i2) { (*this)(i0, i1, i2) = g(i0, i1, i2); };
            make_ParallelTensorFunctor(
                f, mDesc.GetLengths()[0], mDesc.GetLengths()[1], mDesc.GetLengths()[2])(num_thread);
            break;
        }
        case 4: {
            auto f = [&](auto i0, auto i1, auto i2, auto i3) {
                (*this)(i0, i1, i2, i3) = g(i0, i1, i2, i3);
            };
            make_ParallelTensorFunctor(f,
                                       mDesc.GetLengths()[0],
                                       mDesc.GetLengths()[1],
                                       mDesc.GetLengths()[2],
                                       mDesc.GetLengths()[3])(num_thread);
            break;
        }
        case 5: {
            auto f = [&](auto i0, auto i1, auto i2, auto i3, auto i4) {
                (*this)(i0, i1, i2, i3, i4) = g(i0, i1, i2, i3, i4);
            };
            make_ParallelTensorFunctor(f,
                                       mDesc.GetLengths()[0],
                                       mDesc.GetLengths()[1],
                                       mDesc.GetLengths()[2],
                                       mDesc.GetLengths()[3],
                                       mDesc.GetLengths()[4])(num_thread);
            break;
        }
        default: throw std::runtime_error("unspported dimension");
        }
    }

    template <typename... Is>
    T& operator()(Is... is)
    {
        return mData[mDesc.GetOffsetFromMultiIndex(is...)];
    }

    template <typename... Is>
    const T& operator()(Is... is) const
    {
        return mData[mDesc.GetOffsetFromMultiIndex(is...)];
    }

    typename std::vector<T>::iterator begin() { return mData.begin(); }

    typename std::vector<T>::iterator end() { return mData.end(); }

    typename std::vector<T>::const_iterator begin() const { return mData.begin(); }

    typename std::vector<T>::const_iterator end() const { return mData.end(); }

    HostTensorDescriptor mDesc;
    std::vector<T> mData;
};

template <typename X>
HostTensorDescriptor::HostTensorDescriptor(const std::vector<X>& lens) : mLens(lens)
{
    this->CalculateStrides();
}

template <typename X, typename Y>
HostTensorDescriptor::HostTensorDescriptor(const std::vector<X>& lens,
                                           const std::vector<Y>& strides)
    : mLens(lens), mStrides(strides)
{
}

void ostream_HostTensorDescriptor(const HostTensorDescriptor& desc, std::ostream& os = std::cout);

#if 1
// FIXME: remove
void bf16_to_f32_(const Tensor<ck::bhalf_t>& src, Tensor<float>& dst);
#endif

template <typename T>
float check_error(const Tensor<T>& ref, const Tensor<T>& result)
{
    float l1_error       = 0;
    float linf_error     = -1;
    float linf_rel_error = -1;

    float linf_ref_value = 0, linf_result_value = 0;
    float linf_rel_ref_value = 0, linf_rel_result_value = 0;

    constexpr float eps = 1e-10;

    for(int i = 0; i < ref.mData.size(); ++i)
    {
        float ref_v    = ck::type_convert<float>(ref.mData[i]);
        float result_v = ck::type_convert<float>(result.mData[i]);

        float diff     = std::abs(ref_v - result_v);
        float rel_diff = diff / std::max(std::abs(ref_v), eps);

        l1_error += diff;

        if(linf_error < diff)
        {
            linf_error        = diff;
            linf_ref_value    = ref_v;
            linf_result_value = result_v;
        }

        if(linf_rel_error < rel_diff)
        {
            linf_rel_error        = rel_diff;
            linf_rel_ref_value    = ref_v;
            linf_rel_result_value = result_v;
        }
    }

    std::cout << "Absolute Error L1 Norm (sum of abs diff): " << l1_error << std::endl;
    std::cout << "Absolute Error L-inf Norm (max abs diff): " << linf_error << ", ref "
              << linf_ref_value << ", result " << linf_result_value << std::endl;
    std::cout << "Relative Error L-inf Norm (max relative abs diff): " << linf_rel_error << ", ref "
              << linf_rel_ref_value << ", result " << linf_rel_result_value << std::endl;

    return linf_error;
}

#endif
