#pragma once
#include "common.hip.hpp"

template <unsigned NRow_, unsigned NCol_, unsigned RowStride_>
struct ConstantMatrixDescriptor
{
    __host__ __device__ constexpr ConstantMatrixDescriptor()
    {
        static_assert(NCol_ <= RowStride_, "wrong! NCol > RowStride!");
    }

    __host__ __device__ constexpr unsigned NRow() const { return NRow_; }

    __host__ __device__ constexpr unsigned NCol() const { return NCol_; }

    __host__ __device__ constexpr unsigned RowStride() const { return RowStride_; }

    __host__ __device__ constexpr auto GetLengths() const { return Sequence<NRow_, NCol_>{}; }

    __host__ __device__ constexpr unsigned GetElementSize() const { return NRow_ * NCol_; }

    __host__ __device__ constexpr unsigned GetElementSpace() const { return NRow_ * RowStride_; }

    __host__ __device__ unsigned Get1dIndex(unsigned irow, unsigned icol) const
    {
        return irow * RowStride_ + icol;
    }

    template <unsigned SubNRow, unsigned SubNCol>
    __host__ __device__ constexpr auto MakeSubMatrixDescriptor(Number<SubNRow>,
                                                               Number<SubNCol>) const
    {
        return ConstantMatrixDescriptor<SubNRow, SubNCol, RowStride_>{};
    }
};

template <unsigned NRow, unsigned NCol>
__host__ __device__ constexpr auto make_ConstantMatrixDescriptor(Number<NRow>, Number<NCol>)
{
    return ConstantMatrixDescriptor<NRow, NCol, NCol>{};
}

template <unsigned NRow, unsigned NCol, unsigned RowStride>
__host__ __device__ constexpr auto
    make_ConstantMatrixDescriptor(Number<NRow>, Number<NCol>, Number<RowStride>)
{
    return ConstantMatrixDescriptor<NRow, NCol, RowStride>{};
}

template <class TDesc>
__host__ __device__ void print_ConstantMatrixDescriptor(TDesc, const char* s)
{
    const auto desc = TDesc{};

    constexpr auto I0 = Number<0>{};
    constexpr auto I1 = Number<1>{};

    printf("%s NRow %u NCol %u RowStride %u\n", s, desc.NRow(), desc.NCol(), desc.RowStride());
}
