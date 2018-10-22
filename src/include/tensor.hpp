#pragma once
#include <thread>
#include <vector>
#include <numeric>
#include <algorithm>
#include <utility>
#include <cassert>
#include <iostream>
#include "cuda_runtime.h"
#include "helper_cuda.h"

template <class Range>
std::ostream& LogRange(std::ostream& os, Range&& r, std::string delim)
{
    bool first = true;
    for(auto&& x : r)
    {
        if(first)
            first = false;
        else
            os << delim;
        os << x;
    }
    return os;
}

typedef enum
{
    Half  = 0,
    Float = 1,
} DataType_t;

template <class T>
struct DataType;

template <>
struct DataType<float> : std::integral_constant<DataType_t, DataType_t::Float>
{
};

template <class F, class T, std::size_t... Is>
auto call_f_unpack_args_impl(F f, T args, std::index_sequence<Is...>)
{
    return f(std::get<Is>(args)...);
}

template <class F, class T>
auto call_f_unpack_args(F f, T args)
{
    constexpr std::size_t N = std::tuple_size<T>::value;

    return call_f_unpack_args_impl(f, args, std::make_index_sequence<N>{});
}

template <class F, class T, std::size_t... Is>
auto construct_f_unpack_args_impl(T args, std::index_sequence<Is...>)
{
    return F(std::get<Is>(args)...);
}

template <class F, class T>
auto construct_f_unpack_args(F, T args)
{
    constexpr std::size_t N = std::tuple_size<T>::value;

    return construct_f_unpack_args_impl<F>(args, std::make_index_sequence<N>{});
}

struct TensorDescriptor
{
    TensorDescriptor() = delete;
    TensorDescriptor(DataType_t t, std::initializer_list<std::size_t> lens);
    TensorDescriptor(DataType_t t,
                     std::initializer_list<std::size_t> lens,
                     std::initializer_list<std::size_t> strides);
    TensorDescriptor(DataType_t t, std::vector<std::size_t> lens, std::vector<std::size_t> strides);

    void CalculateStrides();

    template <class Range>
    TensorDescriptor(DataType_t t, const Range& lens)
        : mLens(lens.begin(), lens.end()), mDataType(t)
    {
        this->CalculateStrides();
    }

    template <class Range1, class Range2>
    TensorDescriptor(DataType_t t, const Range1& lens, const Range2& strides)
        : mLens(lens.begin(), lens.end()), mStrides(strides.begin(), strides.end()), mDataType(t)
    {
    }

    DataType_t GetDataType() const;
    std::size_t GetDimension() const;
    std::size_t GetElementSize() const;
    std::size_t GetElementSpace() const;

    const std::vector<std::size_t>& GetLengths() const;
    const std::vector<std::size_t>& GetStrides() const;

    template <class... Is>
    std::size_t Get1dIndex(Is... is) const
    {
        assert(sizeof...(Is) == this->GetDimension());
        std::initializer_list<std::size_t> iss{static_cast<std::size_t>(is)...};
        return std::inner_product(iss.begin(), iss.end(), mStrides.begin(), std::size_t{0});
    }

    private:
    DataType_t mDataType;
    std::vector<std::size_t> mLens;
    std::vector<std::size_t> mStrides;
};

struct DeviceMem
{
    DeviceMem() = delete;
    DeviceMem(std::size_t mem_size) : mMemSize(mem_size)
    {
        cudaMalloc(static_cast<void**>(&mpDeviceBuf), mMemSize);
    }

    void* GetDeviceBuffer() { return mpDeviceBuf; }

    int ToDevice(const void* p)
    {
        return static_cast<int>(
            cudaMemcpy(mpDeviceBuf, const_cast<void*>(p), mMemSize, cudaMemcpyHostToDevice));
    }

    int FromDevice(void* p)
    {
        return static_cast<int>(cudaMemcpy(p, mpDeviceBuf, mMemSize, cudaMemcpyDeviceToHost));
    }

    ~DeviceMem() { cudaFree(mpDeviceBuf); }

    void* mpDeviceBuf;
    std::size_t mMemSize;
};

struct joinable_thread : std::thread
{
    template <class... Xs>
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

template <class F, class... Xs>
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

    void operator()(std::size_t num_thread) const
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

template <class F, class... Xs>
auto make_ParallelTensorFunctor(F f, Xs... xs)
{
    return ParallelTensorFunctor<F, Xs...>(f, xs...);
}

template <class T>
struct Tensor
{
    template <class X>
    Tensor(std::initializer_list<X> lens)
        : mDesc(DataType<T>{}, lens), mData(mDesc.GetElementSpace())
    {
    }

    template <class X>
    Tensor(std::vector<X> lens) : mDesc(DataType<T>{}, lens), mData(mDesc.GetElementSpace())
    {
    }

    template <class X, class Y>
    Tensor(std::vector<X> lens, std::vector<Y> strides)
        : mDesc(DataType<T>{}, lens, strides), mData(mDesc.GetElementSpace())
    {
    }

    template <class G>
    void GenerateTensorValue(G g, std::size_t num_thread = 1)
    {
        switch(mDesc.GetDimension())
        {
        case 1:
        {
            auto f = [&](auto i) { (*this)(i) = g(i); };
            make_ParallelTensorFunctor(f, mDesc.GetLengths()[0])(num_thread);
            break;
        }
        case 2:
        {
            auto f = [&](auto i0, auto i1) { (*this)(i0, i1) = g(i0, i1); };
            make_ParallelTensorFunctor(f, mDesc.GetLengths()[0], mDesc.GetLengths()[1])(num_thread);
            break;
        }
        case 3:
        {
            auto f = [&](auto i0, auto i1, auto i2) { (*this)(i0, i1, i2) = g(i0, i1, i2); };
            make_ParallelTensorFunctor(
                f, mDesc.GetLengths()[0], mDesc.GetLengths()[1], mDesc.GetLengths()[2])(num_thread);
            break;
        }
        case 4:
        {
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
        default: throw std::runtime_error("unspported dimension");
        }
    }

    template <class... Is>
    T& operator()(Is... is)
    {
        return mData[mDesc.Get1dIndex(is...)];
    }

    template <class... Is>
    const T& operator()(Is... is) const
    {
        return mData[mDesc.Get1dIndex(is...)];
    }

    typename std::vector<T>::iterator begin() { return mData.begin(); }

    typename std::vector<T>::iterator end() { return mData.end(); }

    typename std::vector<T>::const_iterator begin() const { return mData.begin(); }

    typename std::vector<T>::const_iterator end() const { return mData.end(); }

    TensorDescriptor mDesc;
    std::vector<T> mData;
};
