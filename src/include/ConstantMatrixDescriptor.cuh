#pragma once
#include "common.cuh"

template <unsigned NRow, unsigned NCol, unsigned RowStride>
struct ConstantMatrixDescriptor
{
    __host__ __device__ ConstantMatrixDescriptor()
    {
        static_assert(NCol <= RowStride, "wrong! NCol > RowStride!");
    }

    __host__ __device__ constexpr unsigned GetNumberOfRow() const { return NRow; }

    __host__ __device__ constexpr unsigned GetNumberOfColumn() const { return NCol; }

    __host__ __device__ constexpr unsigned GetRowStride() const { return RowStride; }

    __host__ __device__ constexpr unsigned GetElementSize() const { return NRow * NCol; }

    __host__ __device__ constexpr unsigned GetElementSpace() const { return NRow * RowStride; }

    __host__ __device__ unsigned Get1dIndex(unsigned irow, unsigned icol) const
    {
        return irow * RowStride + icol;
    }

    template <unsigned SubNRow, unsigned SubNCol>
    __host__ __device__ constexpr auto MakeSubMatrixDescriptor(Number<SubNRow>,
                                                               Number<SubNCol>) const
    {
        return ConstantMatrixDescriptor<SubNRow, SubNCol, RowStride>{};
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
