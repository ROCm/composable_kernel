// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2024, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include <algorithm>
#include <array>
#include <cassert>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <numeric>
#include <thread>
#include <utility>
#include <vector>

#include "ck_tile/core.hpp"

namespace ck_tile {

template <typename Range>
CK_TILE_HOST std::ostream& LogRange(std::ostream& os,
                                    Range&& range,
                                    std::string delim,
                                    int precision = std::cout.precision(),
                                    int width     = 0)
{
    bool first = true;
    for(auto&& v : range)
    {
        if(first)
            first = false;
        else
            os << delim;
        os << std::setw(width) << std::setprecision(precision) << v;
    }
    return os;
}

template <typename T, typename Range>
CK_TILE_HOST std::ostream& LogRangeAsType(std::ostream& os,
                                          Range&& range,
                                          std::string delim,
                                          int precision = std::cout.precision(),
                                          int width     = 0)
{
    bool first = true;
    for(auto&& v : range)
    {
        if(first)
            first = false;
        else
            os << delim;
        os << std::setw(width) << std::setprecision(precision) << static_cast<T>(v);
    }
    return os;
}

template <typename F, typename T, std::size_t... Is>
CK_TILE_HOST auto call_f_unpack_args_impl(F f, T args, std::index_sequence<Is...>)
{
    return f(std::get<Is>(args)...);
}

template <typename F, typename T>
CK_TILE_HOST auto call_f_unpack_args(F f, T args)
{
    constexpr std::size_t N = std::tuple_size<T>{};

    return call_f_unpack_args_impl(f, args, std::make_index_sequence<N>{});
}

template <typename F, typename T, std::size_t... Is>
CK_TILE_HOST auto construct_f_unpack_args_impl(T args, std::index_sequence<Is...>)
{
    return F(std::get<Is>(args)...);
}

template <typename F, typename T>
CK_TILE_HOST auto construct_f_unpack_args(F, T args)
{
    constexpr std::size_t N = std::tuple_size<T>{};

    return construct_f_unpack_args_impl<F>(args, std::make_index_sequence<N>{});
}

struct HostTensorDescriptor
{
    HostTensorDescriptor() = default;

    void calculate_strides()
    {
        mStrides.clear();
        mStrides.resize(mLens.size(), 0);
        if(mStrides.empty())
            return;

        mStrides.back() = 1;
        std::partial_sum(mLens.rbegin(),
                         mLens.rend() - 1,
                         mStrides.rbegin() + 1,
                         std::multiplies<std::size_t>());
    }

    template <typename X, typename = std::enable_if_t<std::is_convertible_v<X, std::size_t>>>
    explicit HostTensorDescriptor(const std::initializer_list<X>& lens)
        : mLens(lens.begin(), lens.end())
    {
        this->calculate_strides();
    }

    template <typename X,
              typename Y,
              typename = std::enable_if_t<std::is_convertible_v<X, std::size_t> &&
                                          std::is_convertible_v<Y, std::size_t>>>
    HostTensorDescriptor(const std::initializer_list<X>& lens,
                         const std::initializer_list<Y>& strides)
        : mLens(lens.begin(), lens.end()), mStrides(strides.begin(), strides.end())
    {
        assert(mLens.size() == mStrides.size());
    }

    template <typename Lengths,
              typename = std::enable_if_t<
                  !std::is_base_of_v<HostTensorDescriptor, Lengths> &&
                  std::is_convertible_v<ck_tile::ranges::range_value_t<Lengths>, std::size_t>>>
    explicit HostTensorDescriptor(const Lengths& lens) : mLens(lens.begin(), lens.end())
    {
        this->calculate_strides();
    }

    template <typename Lengths,
              typename Strides,
              typename = std::enable_if_t<
                  std::is_convertible_v<ck_tile::ranges::range_value_t<Lengths>, std::size_t> &&
                  std::is_convertible_v<ck_tile::ranges::range_value_t<Strides>, std::size_t>>>
    HostTensorDescriptor(const Lengths& lens, const Strides& strides)
        : mLens(lens.begin(), lens.end()), mStrides(strides.begin(), strides.end())
    {
        assert(mLens.size() == mStrides.size());
    }

    std::size_t get_num_of_dimension() const { return mLens.size(); }
    std::size_t get_element_size() const
    {
        assert(mLens.size() == mStrides.size());
        return std::accumulate(
            mLens.begin(), mLens.end(), std::size_t{1}, std::multiplies<std::size_t>());
    }
    std::size_t get_element_space_size() const
    {
        std::size_t space = 1;
        for(std::size_t i = 0; i < mLens.size(); ++i)
        {
            if(mLens[i] == 0)
                continue;

            space += (mLens[i] - 1) * mStrides[i];
        }
        return space;
    }

    std::size_t get_length(std::size_t dim) const { return mLens[dim]; }

    auto get_lengths() const
    {
        using iterator = remove_cvref_t<decltype(mLens)>::const_iterator;
        return iterator_range<iterator>(mLens);
    }

    std::size_t get_stride(std::size_t dim) const { return mStrides[dim]; }

    auto get_strides() const
    {
        using iterator = remove_cvref_t<decltype(mStrides)>::const_iterator;
        return iterator_range<iterator>(mStrides);
    }

    template <typename... Is>
    std::enable_if_t<((std::is_integral_v<Is> && std::is_convertible_v<Is, std::size_t>)&&...),
                     std::size_t>
    get_offset_from_multi_index(Is... is) const
    {
        assert(sizeof...(Is) == this->get_num_of_dimension());
        return get_offset_from_multi_index(std::array{static_cast<std::size_t>(is)...});
    }

    std::size_t get_offset_from_multi_index(span<const std::size_t> iss) const
    {
        assert(iss.size() == this->get_num_of_dimension());
        return std::inner_product(iss.begin(), iss.end(), mStrides.begin(), std::size_t{0});
    }

    friend std::ostream& operator<<(std::ostream& os, const HostTensorDescriptor& desc);

    private:
    std::vector<std::size_t> mLens;
    std::vector<std::size_t> mStrides;
};

template <typename New2Old>
CK_TILE_HOST HostTensorDescriptor transpose_host_tensor_descriptor_given_new2old(
    const HostTensorDescriptor& a, const New2Old& new2old)
{
    std::vector<std::size_t> new_lengths(a.get_num_of_dimension());
    std::vector<std::size_t> new_strides(a.get_num_of_dimension());

    for(std::size_t i = 0; i < a.get_num_of_dimension(); i++)
    {
        new_lengths[i] = a.get_length(new2old[i]);
        new_strides[i] = a.get_stride(new2old[i]);
    }

    return HostTensorDescriptor(new_lengths, new_strides);
}

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

    std::array<std::size_t, NDIM> get_nd_indices(std::size_t i) const
    {
        std::array<std::size_t, NDIM> indices;

        for(std::size_t idim = 0; idim < NDIM; ++idim)
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

            auto f = [this, iw_begin, iw_end] {
                for(std::size_t iw = iw_begin; iw < iw_end; ++iw)
                {
                    call_f_unpack_args(this->mF, this->get_nd_indices(iw));
                }
            };
            threads[it] = joinable_thread(f);
        }
    }
};

template <typename F, typename... Xs>
CK_TILE_HOST auto make_ParallelTensorFunctor(F f, Xs... xs)
{
    return ParallelTensorFunctor<F, Xs...>(f, xs...);
}

struct HostTensorSlice
{
    using size_type = std::size_t;

    HostTensorSlice(size_type dim_,
                    std::optional<size_type> start_ = std::nullopt,
                    std::optional<size_type> end_   = std::nullopt)
        : dim(dim_), start(start_), end(end_)
    {
    }

    size_type dim;
    std::optional<size_type> start;
    std::optional<size_type> end;
};

struct HostTensorSlicer
{
    using size_type = std::size_t;

    HostTensorSlicer(size_type length_, size_type start_, size_type end_)
        : length(length_), start(start_), end(end_)
    {
        if(!(0 < length && start < end && end <= length))
        {
            throw std::invalid_argument("invalid slice");
        }
    }

    bool update(size_type new_start, size_type new_end)
    {
        if(!(new_start < new_end && new_end <= get_length()))
        {
            return false;
        }

        end = start + new_end;
        start += new_start;

        return true;
    }

    size_type operator()(size_type idx) const { return start + idx; }

    size_type get_length() const { return end - start; }

    size_type get_start() const { return start; }

    private:
    size_type length;
    size_type start;
    size_type end;
};

namespace detail {
template <typename TensorView>
struct Repeat
{
    using const_reference = typename TensorView::const_reference;
    using size_type       = typename TensorView::size_type;

    static inline constexpr size_type MaxNumDims = TensorView::MaxNumDims;

    template <typename X, typename = std::enable_if_t<std::is_convertible_v<X, size_type>>>
    Repeat(TensorView view, std::initializer_list<X> repeats) : mView(std::move(view))
    {
        assert(mView.get_num_of_dimension() <= MaxNumDims);
        assert(std::size(repeats) <= mView.get_num_of_dimension());

        using std::rbegin, std::rend;
        std::copy(rbegin(repeats), rend(repeats), rbegin(get_repeats()));
    }

    size_type get_num_of_dimension() const { return mView.get_num_of_dimension(); }

    size_type get_length(size_type dim) const { return mView.get_length(dim) * get_repeat(dim); }

    auto get_lengths() const
    {
        return make_transform_range(make_zip_range(mView.get_lengths(), get_repeats()),
                                    [](auto bundle) {
                                        // length * repeat
                                        return std::get<0>(bundle) * std::get<1>(bundle);
                                    });
    }

    template <typename... Is>
    std::enable_if_t<((std::is_integral_v<Is> && std::is_convertible_v<Is, size_type>)&&...),
                     const_reference>
    operator()(Is... is) const
    {
        return (*this)(std::array{static_cast<size_type>(is)...});
    }

    const_reference operator()(span<const size_type> idx) const
    {
        assert(std::size(idx) == get_num_of_dimension());

        std::array<size_type, MaxNumDims> real_idx;
        for(size_type dim = 0; dim < std::size(idx); ++dim)
        {
            real_idx[dim] = idx[dim] / get_repeat(dim);
        }

        return mView(span<const size_type>(std::data(real_idx), std::size(idx)));
    }

    private:
    size_type get_repeat(size_type dim) const { return mRepeats[dim]; }

    auto get_repeats()
    {
        using std::begin, std::next;

        return iterator_range(begin(mRepeats), next(begin(mRepeats), get_num_of_dimension()));
    }

    auto get_repeats() const
    {
        using std::begin, std::next;

        return iterator_range(begin(mRepeats), next(begin(mRepeats), get_num_of_dimension()));
    }

    TensorView mView;
    std::array<size_type, MaxNumDims> mRepeats;
};
} // namespace detail

template <typename T>
struct HostTensorView : private HostTensorDescriptor
{
    using Descriptor      = HostTensorDescriptor;
    using Data            = span<T>;
    using Slicer          = HostTensorSlicer;
    using reference       = typename Data::reference;
    using const_reference = typename Data::const_reference;
    using iterator        = typename Data::iterator;
    using pointer         = typename Data::pointer;
    using size_type       = std::size_t;

    static inline constexpr size_type MaxNumDims = 6;

    protected:
    template <typename X, typename = std::enable_if_t<std::is_convertible_v<X, size_type>>>
    explicit HostTensorView(std::initializer_list<X> lens) : Descriptor(lens)
    {
        assert(get_num_of_dimension() <= MaxNumDims);
    }

    template <typename X,
              typename Y,
              typename = std::enable_if_t<std::is_convertible_v<X, size_type> &&
                                          std::is_convertible_v<Y, size_type>>>
    HostTensorView(std::initializer_list<X> lens, std::initializer_list<Y> strides)
        : Descriptor(lens, strides)
    {
        assert(get_num_of_dimension() <= MaxNumDims);
    }

    template <typename Lengths,
              typename = std::enable_if_t<
                  !std::is_base_of_v<HostTensorDescriptor, Lengths> &&
                  std::is_convertible_v<ck_tile::ranges::range_value_t<Lengths>, size_type>>>
    explicit HostTensorView(const Lengths& lens) : Descriptor(lens)
    {
        assert(get_num_of_dimension() <= MaxNumDims);
    }

    template <typename Lengths,
              typename Strides,
              typename = std::enable_if_t<
                  std::is_convertible_v<ck_tile::ranges::range_value_t<Lengths>, size_type> &&
                  std::is_convertible_v<ck_tile::ranges::range_value_t<Strides>, size_type>>>
    HostTensorView(const Lengths& lens, const Strides& strides) : Descriptor(lens, strides)
    {
        assert(get_num_of_dimension() <= MaxNumDims);
    }

    public:
    HostTensorView(Descriptor desc, Data data) : Descriptor(std::move(desc)), mData(data)
    {
        assert(get_element_space_size() <= mData.size());
        assert(get_num_of_dimension() <= MaxNumDims);
    }

    HostTensorView()                      = delete;
    HostTensorView(const HostTensorView&) = default;
    HostTensorView(HostTensorView&&)      = default;

    ~HostTensorView() = default;

    HostTensorView& operator=(const HostTensorView&) = default;
    HostTensorView& operator=(HostTensorView&&) = default;

    friend struct HostTensorView<std::remove_const_t<T>>;

    operator HostTensorView<std::add_const_t<T>>() const
    {
        using std::begin, std::end;

        HostTensorView<std::add_const_t<T>> view(static_cast<const Descriptor&>(*this), mData);
        std::copy(begin(get_slicers()), end(get_slicers()), begin(view.get_slicers()));

        return view;
    }

    using Descriptor::get_element_size;
    using Descriptor::get_element_space_size;
    using Descriptor::get_length;
    using Descriptor::get_lengths;
    using Descriptor::get_num_of_dimension;
    using Descriptor::get_stride;
    using Descriptor::get_strides;

    size_type get_element_space_size_in_bytes() const
    {
        return sizeof(T) * get_element_space_size();
    }

    void SetZero() { std::fill(mData.begin(), mData.end(), 0); }

    HostTensorView transpose(size_type dim0, size_type dim1) const
    {
        if(get_num_of_dimension() <= dim0 || get_num_of_dimension() <= dim1)
        {
            throw std::invalid_argument("transpose with invalid dim0 or dim1");
        }

        using std::begin, std::end;

        std::vector<size_type> order(get_num_of_dimension());
        std::iota(begin(order), end(order), 0);

        std::swap(order[dim0], order[dim1]);

        auto new_lengths = make_permutation_range(get_lengths(), order);
        auto new_strides = make_permutation_range(get_strides(), order);
        auto new_slicers = make_permutation_range(get_slicers(), order);

        HostTensorView view(Descriptor(new_lengths, new_strides), mData);
        std::copy(begin(new_slicers), end(new_slicers), begin(view.get_slicers()));

        return view;
    }

    HostTensorView index(std::initializer_list<HostTensorSlice> slices) const
    {
        using std::begin, std::end;

        std::vector<std::optional<Slicer>> new_slicers(begin(get_slicers()), end(get_slicers()));

        const auto lengths = get_lengths();
        std::vector<size_type> new_lengths(begin(lengths), end(lengths));

        for(size_type idx = 0; idx < std::size(slices); ++idx)
        {
            const auto& slice = *std::next(begin(slices), idx);
            if(get_num_of_dimension() < slice.dim)
            {
                throw std::invalid_argument("invalid dim for slice");
            }

            const size_type length = lengths[slice.dim];

            const size_type start = (slice.start ? *slice.start : 0);
            const size_type end   = (slice.end ? *slice.end : length);

            auto& slicer = new_slicers[slice.dim];
            if(slicer)
            {
                if(!slicer->update(start, end))
                {
                    throw std::invalid_argument("slice conflict with others");
                }
            }
            else
            {
                slicer.emplace(length, start, end);
            }
            new_lengths[slice.dim] = slicer->get_length();
        }

        HostTensorView view(Descriptor(new_lengths, get_strides()), mData);
        std::copy(begin(new_slicers), end(new_slicers), begin(view.get_slicers()));

        return view;
    }

    HostTensorView squeeze(size_type dim) const
    {
        assert(0 < get_num_of_dimension());

        if(get_num_of_dimension() == 1 || get_num_of_dimension() <= dim || 1 < get_length(dim))
        {
            return *this;
        }

        using std::begin, std::end, std::next;

        // check if the squeezing dimension has the largest stride
        const size_type stride = get_stride(dim);

        const auto strides    = get_strides();
        const auto max_stride = std::max_element(begin(strides), end(strides));
        if(stride < *max_stride)
        {
            return *this;
        }

        // remove length/stride on dim
        const auto lengths = get_lengths();
        std::vector<size_type> new_lengths(begin(lengths), end(lengths));
        std::vector<size_type> new_strides(begin(strides), end(strides));

        new_lengths.erase(next(begin(new_lengths), dim));
        new_strides.erase(next(begin(new_strides), dim));

        auto view = [&]() -> HostTensorView {
            HostTensorDescriptor desc(new_lengths, new_strides);

            auto& slicer = get_slicer(dim);
            if(slicer)
            {
                return {desc, mData.subspan(stride * slicer->get_start())};
            }
            else
            {
                return {desc, mData};
            }
        }();

        auto src  = get_slicers();
        auto dest = view.get_slicers();

        // copy all slicers in [0, dim)
        std::copy(begin(src), next(begin(src), dim), begin(dest));
        // copy all slicers in [dim + 1, get_num_of_dimension())
        std::copy(next(begin(src), dim + 1), end(src), next(begin(dest), dim));

        return view;
    }

    template <typename X, typename = std::enable_if_t<std::is_convertible_v<X, size_type>>>
    auto repeat(std::initializer_list<X> repeats) const
    {
        return detail::Repeat<HostTensorView>(*this, repeats);
    }

    template <typename F>
    void for_each(F&& f)
    {
        std::vector<size_t> idx(get_num_of_dimension(), 0);
        for_each_impl(std::forward<F>(f), idx, size_t(0));
    }

    template <typename F>
    void for_each(const F&& f) const
    {
        std::vector<size_t> idx(get_num_of_dimension(), 0);
        for_each_impl(std::forward<const F>(f), idx, size_t(0));
    }

    template <typename G>
    void GenerateTensorValue(G g, std::size_t num_thread = 1)
    {
        switch(get_num_of_dimension())
        {
        case 1: {
            auto f = [&](auto i) { (*this)(i) = g(i); };
            make_ParallelTensorFunctor(f, get_length(0))(num_thread);
            break;
        }
        case 2: {
            auto f = [&](auto i0, auto i1) { (*this)(i0, i1) = g(i0, i1); };
            make_ParallelTensorFunctor(f, get_length(0), get_length(1))(num_thread);
            break;
        }
        case 3: {
            auto f = [&](auto i0, auto i1, auto i2) { (*this)(i0, i1, i2) = g(i0, i1, i2); };
            make_ParallelTensorFunctor(f, get_length(0), get_length(1), get_length(2))(num_thread);
            break;
        }
        case 4: {
            auto f = [&](auto i0, auto i1, auto i2, auto i3) {
                (*this)(i0, i1, i2, i3) = g(i0, i1, i2, i3);
            };
            make_ParallelTensorFunctor(
                f, get_length(0), get_length(1), get_length(2), get_length(3))(num_thread);
            break;
        }
        case 5: {
            auto f = [&](auto i0, auto i1, auto i2, auto i3, auto i4) {
                (*this)(i0, i1, i2, i3, i4) = g(i0, i1, i2, i3, i4);
            };
            make_ParallelTensorFunctor(
                f, get_length(0), get_length(1), get_length(2), get_length(3), get_length(4))(
                num_thread);
            break;
        }
        case 6: {
            auto f = [&](auto i0, auto i1, auto i2, auto i3, auto i4, auto i5) {
                (*this)(i0, i1, i2, i3, i4, i5) = g(i0, i1, i2, i3, i4, i5);
            };
            make_ParallelTensorFunctor(f,
                                       get_length(0),
                                       get_length(1),
                                       get_length(2),
                                       get_length(3),
                                       get_length(4),
                                       get_length(5))(num_thread);
            break;
        }
        default: throw std::runtime_error("unspported dimension");
        }
    }

    template <typename... Is>
    std::enable_if_t<((std::is_integral_v<Is> && std::is_convertible_v<Is, size_type>)&&...),
                     reference>
    operator()(Is... is)
    {
        return (*this)(std::array{static_cast<size_type>(is)...});
    }

    template <typename... Is>
    std::enable_if_t<((std::is_integral_v<Is> && std::is_convertible_v<Is, size_type>)&&...),
                     const_reference>
    operator()(Is... is) const
    {
        return (*this)(std::array{static_cast<size_type>(is)...});
    }

    reference operator()(span<const size_type> idx) { return get_impl(*this, idx); }

    const_reference operator()(span<const size_type> idx) const { return get_impl(*this, idx); }

    iterator begin() const { return mData.begin(); }

    iterator end() const { return std::next(begin(), size()); }

    pointer data() const { return mData.data(); }

    size_type size() const { return get_element_space_size(); }

    protected:
    void set_data(Data data)
    {
        assert(get_element_space_size() <= data.size());
        mData = data;
    }

    auto& get_slicer(size_type dim) { return mSlicers[dim]; }

    auto& get_slicer(size_type dim) const { return mSlicers[dim]; }

    auto get_slicers()
    {
        using std::begin, std::next;

        return iterator_range(begin(mSlicers), next(begin(mSlicers), get_num_of_dimension()));
    }

    auto get_slicers() const
    {
        using std::begin, std::next;

        return iterator_range(begin(mSlicers), next(begin(mSlicers), get_num_of_dimension()));
    }

    private:
    template <typename Self>
    static decltype(auto) get_impl(Self&& self, span<const size_type> idx)
    {
        assert(std::size(idx) == self.get_num_of_dimension());

        std::array<size_type, MaxNumDims> real_idx;
        for(size_type dim = 0; dim < std::size(idx); ++dim)
        {
            auto& slicer  = self.get_slicer(dim);
            real_idx[dim] = (slicer ? (*slicer)(idx[dim]) : idx[dim]);
        }

        return self.mData[self.get_offset_from_multi_index(
            span<const size_type>(std::data(real_idx), std::size(idx)))];
    }

    template <typename F>
    void for_each_impl(F&& f, std::vector<size_t>& idx, size_t rank)
    {
        if(rank == get_num_of_dimension())
        {
            f(*this, idx);
            return;
        }
        // else
        for(size_t i = 0; i < get_length(rank); i++)
        {
            idx[rank] = i;
            for_each_impl(std::forward<F>(f), idx, rank + 1);
        }
    }

    template <typename F>
    void for_each_impl(const F&& f, std::vector<size_t>& idx, size_t rank) const
    {
        if(rank == get_num_of_dimension())
        {
            f(*this, idx);
            return;
        }
        // else
        for(size_t i = 0; i < get_length(rank); i++)
        {
            idx[rank] = i;
            for_each_impl(std::forward<const F>(f), idx, rank + 1);
        }
    }

    Data mData;
    std::array<std::optional<Slicer>, MaxNumDims> mSlicers;
};

template <typename T>
using tensor_value_t =
    remove_cvref_t<decltype(std::declval<remove_cvref_t<T>&>()(0, 0, 0, 0, 0, 0))>;

template <typename T, typename = void>
struct is_tensor : std::false_type
{
};

template <typename T>
struct is_tensor<T,
                 std::void_t<decltype(std::declval<T&>().get_lengths()),
                             decltype(std::declval<T&>()(0, 0, 0, 0, 0, 0)),
                             decltype(std::declval<T&>()(std::declval<span<const std::size_t>>()))>>
    : std::bool_constant<
          std::is_convertible_v<decltype(*std::begin(std::declval<T&>().get_lengths())),
                                std::size_t> &&
          std::is_lvalue_reference_v<decltype(std::declval<T&>()(0, 0, 0, 0, 0, 0))> &&
          std::is_same_v<decltype(std::declval<T&>()(0, 0, 0, 0, 0, 0)),
                         decltype(std::declval<T&>()(std::declval<span<const std::size_t>>()))>>
{
};

template <typename T>
inline constexpr bool is_tensor_v = is_tensor<T>::value;

template <typename T>
struct HostTensor : HostTensorView<T>
{
    using View = HostTensorView<T>;
    using Data = std::vector<T>;

    template <typename X>
    explicit HostTensor(std::initializer_list<X> lens)
        : View(lens), mData(View::get_element_space_size())
    {
        View::set_data(mData);
    }

    template <typename X, typename Y>
    HostTensor(std::initializer_list<X> lens, std::initializer_list<Y> strides)
        : View(lens, strides), mData(View::get_element_space_size())
    {
        View::set_data(mData);
    }

    template <typename Lengths>
    explicit HostTensor(const Lengths& lens) : View(lens), mData(View::get_element_space_size())
    {
        View::set_data(mData);
    }

    template <typename Lengths, typename Strides>
    HostTensor(const Lengths& lens, const Strides& strides)
        : View(lens, strides), mData(View::get_element_space_size())
    {
        View::set_data(mData);
    }

    explicit HostTensor(const typename View::Descriptor& desc)
        : View(desc), mData(View::get_element_space_size())
    {
        View::set_data(mData);
    }

    template <typename FromT>
    explicit HostTensor(const HostTensor<FromT>& other) : HostTensor(other.template copy_as<T>())
    {
    }

    HostTensor()                  = delete;
    HostTensor(const HostTensor&) = default;
    HostTensor(HostTensor&&)      = default;

    ~HostTensor() = default;

    HostTensor& operator=(const HostTensor&) = default;
    HostTensor& operator=(HostTensor&&) = default;

    template <typename OutT>
    HostTensor<OutT> copy_as() const
    {
        HostTensor<OutT> ret(static_cast<const typename View::Descriptor&>(*this));
        std::transform(mData.cbegin(), mData.cend(), ret.mData.begin(), [](auto value) {
            return ck_tile::type_convert<OutT>(value);
        });
        return ret;
    }

    private:
    Data mData;
};

} // namespace ck_tile
