#include <thread>
#include <vector>
#include <numeric>
#include <utility>
#include "cuda_runtime.h"
#include "helper_cuda.h"

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

    std::size_t GetDimension() const;
    std::size_t GetElementSize() const;
    std::size_t GetElementSpace() const;

    const std::vector<std::size_t>& GetLengths() const;
    const std::vector<std::size_t>& GetStrides() const;

    template <class... Xs>
    std::size_t Get1dIndex(Xs... xs) const
    {
        assert(sizeof...(Xs) == this->GetDimension());
        std::initializer_list<std::size_t> is{xs...};
        return std::inner_product(is.begin(), is.end(), mStrides.begin(), std::size_t{0});
    }

    private:
    std::vector<std::size_t> mLens;
    std::vector<std::size_t> mStrides;

    DataType_t mDataType;
};

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
    void GenerateTensorValue(G g)
    {
        // ParallelTensorFunctor([&](Xs... xs) { mData(mDesc.Get1dIndex(xs...)) = g(xs...); },
        // mDesc.mLens)();
        switch(mDesc.GetDimension())
        {
        case 1:
        {
            ParallelTensorFunctor([&](auto i) { mData(mDesc.Get1dIndex(i)) = g(i); },
                                  mDesc.GetLengths()[0])();
            break;
        }
        case 2:
        {
            ParallelTensorFunctor(
                [&](auto i0, auto i1) { mData(mDesc.Get1dIndex(i0, i1)) = g(i0, i1); },
                mDesc.GetLengths()[0],
                mDesc.GetLengths()[1])();
            break;
        }
        case 3:
        {
            ParallelTensorFunctor(
                [&](auto i0, auto i1, auto i2) {
                    mData(mDesc.Get1dIndex(i0, i1, i2)) = g(i0, i1, i2);
                },
                mDesc.GetLengths()[0],
                mDesc.GetLengths()[1],
                mDesc.GetLengths()[2])();
            break;
        }
        case 4:
        {
            ParallelTensorFunctor(
                [&](auto i0, auto i1, auto i2, auto i3) {
                    mData(mDesc.Get1dIndex(i0, i1, i2, i3)) = g(i0, i1, i2, i3);
                },
                mDesc.GetLengths()[0],
                mDesc.GetLengths()[1],
                mDesc.GetLengths()[3],
                mDesc.GetLengths()[4])();
            break;
        }
        default: throw std::runtime_error("unspported dimension");
        }
    }

    T& operator[](std::size_t i) { return mData.at(i); }

    const T& operator[](std::size_t i) const { return mData.at(i); }

    typename std::vector<T>::iterator begin() { return mData.begin(); }

    typename std::vector<T>::iterator end() { return mData.end(); }

    typename std::vector<T>::const_iterator begin() const { return mData.begin(); }

    typename std::vector<T>::const_iterator end() const { return mData.end(); }

    TensorDescriptor mDesc;
    std::vector<T> mData;
};

struct GpuMem
{
    GpuMem() = delete;
    GpuMem(std::size_t size, std::size_t data_size) : mSize(size), mDataSize(data_size)
    {
        cudaMalloc(static_cast<void**>(&mGpuBuf), mDataSize * mSize);
    }

    int ToGpu(void* p)
    {
        return static_cast<int>(cudaMemcpy(mGpuBuf, p, mDataSize * mSize, cudaMemcpyHostToDevice));
    }

    int FromGpu(void* p)
    {
        return static_cast<int>(cudaMemcpy(p, mGpuBuf, mDataSize * mSize, cudaMemcpyDeviceToHost));
    }

    ~GpuMem() { cudaFree(mGpuBuf); }

    void* mGpuBuf;
    std::size_t mSize;
    std::size_t mDataSize;
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
    enum ParallelMethod_t
    {
        Serial   = 0,
        Parallel = 1,
    };

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

    void operator()(std::integral_constant<ParallelMethod_t, ParallelMethod_t::Serial>)
    {
        for(std::size_t i = 0; i < mN1d; ++i)
        {
            call_f_unpack_args(mF, GetNdIndices(i));
        }
    }

    void operator()(std::integral_constant<ParallelMethod_t, ParallelMethod_t::Parallel>,
                    std::size_t num_thread)
    {
        std::size_t work_per_thread = (mN1d + num_thread - 1) / num_thread;

        std::vector<joinable_thread> threads(num_thread);

        for(std::size_t it = 0; it < num_thread; ++it)
        {
            std::size_t iw_begin = it * work_per_thread;
            std::size_t iw_end   = std::min(((it + 1) * work_per_thread, mN1d));

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

template <class F, class T>
auto call_f_unpack_args(F f, T args)
{
    static constexpr std::size_t N = std::tuple_size<T>::value;

    return call_f_unpack_args_impl(f, args, std::make_index_sequence<N>{});
}

template <class F, class T, class... Is>
auto call_f_unpack_args_impl(F f, T args, std::integer_sequence<Is...>)
{
    return f(std::get<Is>(args)...);
}

template <class F, class T, class... Is>
auto construct_f_unpack_args_impl(T args, std::integer_sequence<Is...>)
{
    return F(std::get<Is>(args)...);
}

template <class F, class T>
auto construct_f_unpack_args(F, T args)
{
    static constexpr std::size_t N = std::tuple_size<T>::value;

    return construct_f_unpack_args_impl<F>(args, std::make_index_sequence<N>{});
}
