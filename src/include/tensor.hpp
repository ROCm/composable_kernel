#include <thread>
#include <vector>
#include <numeric>

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

    template<class Range1, class Range2>
    TensorDescriptor(DataType_t t, const Range1& lens, const Range2& strides)
        : mLens(lens.begin(), lens.end()), mStrides(strides.begin(), strides.end()), mDataType(t)
    {}

    std::size_t GetDimension() const;
    std::size_t GetElementSize() const;
    std::size_t GetElementSpace() const;

    template<class... Xs>
    std::size_t GetIndex(Xs... xs) const
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
        parallel_for([&](Xs... xs) { mData(mDesc.GetIndex(xs...)) = g(xs...); }, mDesc.mLens);
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
    GpuMem(std::size_t sz, std::size_t data_sz) : mSz(sz), mDataSz(data_sz)
    {
        cudaMalloc(statci_cast<void**>(&GpuBuf), mDataSize * mSz);
    }

    int ToGpu(void* p)
    {
        return static_cast<int>(cudaMemcpy(mGpuBuf, p, mDataSz * mSz, cudaMemCpyHostToDevice));
    }

    int FromGpu(void* p) { return static_cast<int>(cuadMemCpy(p, mGpuBuf, mDataSz * mSz)); }

    ~GpuMem() { cudaFree(mGpuBuf); }

    void* mGpuBuf;
    std::size_t mSz;
    std::size_t mDataSz;
};

void dummy()
{
    auto f1 = [](int n, int c, int h, int w) { do_f1(n, c, h, w); };
    auto f2 = [](int n, int c, int h, int w) { do_f2(n, c, h, w); };

    auto par_f1 = generate_ParallelTensorFunctor(f1, 3, 3, 3, 3, 3);
    auto par_f2 = generate_ParallelTensorFunctor(f2, 4, 4, 4);

    auto r1 = par_f1();
    auto r2 = par_f2();
}

template <class F, class... Xs>
auto generate_parallel_tensor_functor(F f, Xs... xs)
{
    return ParallelTensorFunctor(f, xs...);
}

template <class F, class... Xs>
struct ParallelTensorFunctor
{
    enum ParallelMethod_t
    {
        Serial   = 0,
        Parallel = 1,
    };

    F mF;
    constexpr std::size_t DIM = sizeof...(Xs);
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

    void operator()(std::integral_constant<ParallelMethod_t, ParallelMethod_t::Serial>)
    {
        for(std::size_t i = 0; i < mN1d; ++i)
        {
            call_f_unpack_indices(mF, GetNdIndices(i));
        }
    }

    void operator()(std::integral_constant<ParallelMethod_t, ParallelMethod_t::Parallel>,
                    std::size_t::num_thread)
    {
        std::size_t work_per_thread = (mN1d + num_thread - 1) / num_thread;

        std::vector<joinable_thread> threads(num_thread);

        for(std::size_t it = 0; it < num_thread; ++it)
        {
            std::size_t iw_begin = it * work_per_thread;
            std::size_t iw_end = std::min(((it+1)*work_per_thread, mN1d));

            auto f = [=] {
                for(std::size_t iw = iw_begin; iw < iw_end; ++iw)
                    call_f_unpack_indices(mF, GetNdIndices(iw);
            };
            threads[it] = joinable_thread(f);
        }
    }
};

struct joinable_thread : std::thread
{
    template <class... Xs>
    joinable_thread(Xs&&... xs) : std::thread(std::forward<Xs>(xs)...)
    {
    }

    ~joinable_thread()
    {
        if(this->joinable())
            this->join;
    }
}

template <class F, class T>
auto call_f_unpack_indices(F f, T indices)
{
    constexpr std::size_t N = std::tuple_size<T>::value;
    using NSeq              = std::make_integer_sequence<std::size_t, N>;

    return call_f_unpack_indices_impl(f, indices, NSeq{});
}

template <class F, class T, class... Is>
auto call_f_unpack_indices_impl(F f, T indices, std::integer_sequence<std::size_t, Is...>)
{
    return f(std::get<Is>(indices)...);
}
