#include "ck/utility/common_header.hpp"
#include "ck/tensor_description/tensor_descriptor.hpp"
#include "ck/tensor_description/tensor_descriptor_helper.hpp"
#include "ck/tensor_operation/gpu/device/tensor_layout.hpp"
#include "ck/tensor_operation/gpu/device/convolution_forward_specialization.hpp"
#include "ck/tensor_operation/operator_transform/copy_transform_conv_fwd_to_gemm.hpp"
#include "ck/tensor_operation/gpu/device/gemm_specialization.hpp"
#include "ck/tensor_operation/gpu/device/copy_matrix_padder.hpp"
#include "ck/host_utility/io.hpp"
#include <algorithm>
#include <numeric>
#include <fstream>
#include <variant>
#include <utility>

// functions to return the corresponding structs based on generated template parameters

using layouts = std::variant<ck::tensor_layout::convolution::GNWK,
                             ck::tensor_layout::convolution::GNHWK,
                             ck::tensor_layout::convolution::NHWGK,
                             ck::tensor_layout::convolution::GNDHWK,
                             ck::tensor_layout::convolution::NDHWGK>;
auto layout_type(std::string type)
{
    std::variant<ck::tensor_layout::convolution::GNWK,
                 ck::tensor_layout::convolution::GNHWK,
                 ck::tensor_layout::convolution::NHWGK,
                 ck::tensor_layout::convolution::GNDHWK,
                 ck::tensor_layout::convolution::NDHWGK>
        layouts;
    if(type == "ck::tensor_layout::convolution::GNHWK")
    {
        // layouts = ck::tensor_layout::convolution::GNHWK{};
        // return std::get<ck::tensor_layout::convolution::GNHWK>(layouts);
        // return std::make_pair(2,layouts);
        // return 2;
        return ck::tensor_layout::convolution::GNHWK{};
    }
    /**else if(type == "ck::tensor_layout::convolution::NHWGK")
    {
            layouts = ck::tensor_layout::convolution::NHWGK{};
            //return std::make_pair(3,layouts);
            //return 3;
            return std::get<ck::tensor_layout::convolution::NHWGK>(layouts);
            //return ck::tensor_layout::convolution::NHWGK{};
    }**/
    throw std::runtime_error("Incorrect layout");
}
// return the right gemm spec based on the generated template parameters
ck::tensor_operation::device::GemmSpecialization gemm_type(std::string type)
{
    if(type == "ck::tensor_operation::device::GemmSpecialization::Default")
    {
        return ck::tensor_operation::device::GemmSpecialization::Default;
    }
    else if(type == "ck::tensor_operation::device::GemmSpecialization::MNKPadding")
    {
        return ck::tensor_operation::device::GemmSpecialization::MNKPadding;
    }
    throw std::runtime_error("Incorrect gemm spec");
}

// return the type of convolution
ck::tensor_operation::device::ConvolutionForwardSpecialization conv_type(std::string type)
{
    if(type == "ck::tensor_operation::device::ConvolutionForwardSpecialization::Default")
    {
        return ck::tensor_operation::device::ConvolutionForwardSpecialization::Default;
    }
    else if(type == "ck::tensor_operation::device::ConvolutionForwardSpecialization::Filter1x1Pad0")
    {
        return ck::tensor_operation::device::ConvolutionForwardSpecialization::Filter1x1Pad0;
    }
    else if(type ==
            "ck::tensor_operation::device::ConvolutionForwardSpecialization::Filter1x1Stride1Pad0")
    {
        return ck::tensor_operation::device::ConvolutionForwardSpecialization::Filter1x1Stride1Pad0;
    }
    else if(type == "ck::tensor_operation::device::ConvolutionForwardSpecialization::OddC")
    {
        return ck::tensor_operation::device::ConvolutionForwardSpecialization::OddC;
    }
    throw std::runtime_error("Incorrect conv spec");
}

// Function to call on MatrixPadder via a wrapper struct
template <typename CDesc_MRaw_NRaw>
auto pad(ck::index_t mpb,
         ck::index_t npb,
         ck::index_t kpb,
         ck::tensor_operation::device::GemmSpecialization gemm,
         CDesc_MRaw_NRaw conv)
{
    if(gemm == ck::tensor_operation::device::GemmSpecialization::MNKPadding)
    {
        ck::tensor_operation::device::CopyMatrixPadder<
            ck::tensor_operation::device::GemmSpecialization::MNKPadding,
            ck::index_t,
            ck::index_t,
            ck::index_t>
            a;
        a.MPerTile_ = mpb;
        a.NPerTile_ = npb;
        a.KPerTile_ = kpb;
        auto res    = ck::tensor_operation::device::Padder(a, conv);
        auto tmp    = res.grid_desc(a, conv);
        return tmp;
    }
    throw std::runtime_error("Incorrect template parameters, check gemm spec");
}

// Functions to call on TransformConvFwdToGemm through wrapper: different functions based on num
// dims
auto transform_conv(ck::index_t num_dim,
                    ck::tensor_operation::device::ConvolutionForwardSpecialization spec,
                    layouts e_layout,
                    ck::Array<ck::index_t, 5> out_lengths,
                    ck::Array<ck::index_t, 5> out_strides)
{
    if(num_dim == 2 &&
       spec == ck::tensor_operation::device::ConvolutionForwardSpecialization::Default)
    {
        ck::tensor_operation::TransformConvFwdToGemm<
            2,
            ck::tensor_operation::device::ConvolutionForwardSpecialization::Default>
            conv_fwd;

        auto res =
            ck::tensor_operation::TransformConv(e_layout, out_lengths, out_strides, conv_fwd);
        return res.transform_func(e_layout, out_lengths, out_strides, conv_fwd);
    }
    else if(num_dim == 2 &&
            spec == ck::tensor_operation::device::ConvolutionForwardSpecialization::Filter1x1Pad0)
    {
        ck::tensor_operation::TransformConvFwdToGemm<
            2,
            ck::tensor_operation::device::ConvolutionForwardSpecialization::Filter1x1Pad0>
            conv_fwd;

        auto res =
            ck::tensor_operation::TransformConv(e_layout, out_lengths, out_strides, conv_fwd);
        return res.transform_func(e_layout, out_lengths, out_strides, conv_fwd);
    }
    else if(num_dim == 2 && spec == ck::tensor_operation::device::ConvolutionForwardSpecialization::
                                        Filter1x1Stride1Pad0)
    {
        ck::tensor_operation::TransformConvFwdToGemm<
            2,
            ck::tensor_operation::device::ConvolutionForwardSpecialization::Filter1x1Stride1Pad0>
            conv_fwd;

        auto res =
            ck::tensor_operation::TransformConv(e_layout, out_lengths, out_strides, conv_fwd);
        return res.transform_func(e_layout, out_lengths, out_strides, conv_fwd);
    }
    else if(num_dim == 2 &&
            spec == ck::tensor_operation::device::ConvolutionForwardSpecialization::OddC)
    {
        ck::tensor_operation::TransformConvFwdToGemm<
            2,
            ck::tensor_operation::device::ConvolutionForwardSpecialization::OddC>
            conv_fwd;

        auto res =
            ck::tensor_operation::TransformConv(e_layout, out_lengths, out_strides, conv_fwd);
        return res.transform_func(e_layout, out_lengths, out_strides, conv_fwd);
    }
    throw std::runtime_error("Incorrect conv spec");
}

auto transform_conv_3d(ck::index_t num_dim,
                       ck::tensor_operation::device::ConvolutionForwardSpecialization spec,
                       layouts e_layout,
                       ck::Array<ck::index_t, 6> out_lengths,
                       ck::Array<ck::index_t, 6> out_strides)
{
    if(num_dim == 3 &&
       spec == ck::tensor_operation::device::ConvolutionForwardSpecialization::Default)
    {
        ck::tensor_operation::TransformConvFwdToGemm<
            3,
            ck::tensor_operation::device::ConvolutionForwardSpecialization::Default>
            conv_fwd;

        auto res =
            ck::tensor_operation::TransformConv(e_layout, out_lengths, out_strides, conv_fwd);
        return res.transform_func(e_layout, out_lengths, out_strides, conv_fwd);
    }
    else if(num_dim == 3 &&
            spec == ck::tensor_operation::device::ConvolutionForwardSpecialization::Filter1x1Pad0)
    {
        ck::tensor_operation::TransformConvFwdToGemm<
            3,
            ck::tensor_operation::device::ConvolutionForwardSpecialization::Filter1x1Pad0>
            conv_fwd;

        auto res =
            ck::tensor_operation::TransformConv(e_layout, out_lengths, out_strides, conv_fwd);
        return res.transform_func(e_layout, out_lengths, out_strides, conv_fwd);
    }
    else if(num_dim == 3 && spec == ck::tensor_operation::device::ConvolutionForwardSpecialization::
                                        Filter1x1Stride1Pad0)
    {
        ck::tensor_operation::TransformConvFwdToGemm<
            3,
            ck::tensor_operation::device::ConvolutionForwardSpecialization::Filter1x1Stride1Pad0>
            conv_fwd;

        auto res =
            ck::tensor_operation::TransformConv(e_layout, out_lengths, out_strides, conv_fwd);
        return res.transform_func(e_layout, out_lengths, out_strides, conv_fwd);
    }
    else if(num_dim == 3 &&
            spec == ck::tensor_operation::device::ConvolutionForwardSpecialization::OddC)
    {
        ck::tensor_operation::TransformConvFwdToGemm<
            3,
            ck::tensor_operation::device::ConvolutionForwardSpecialization::OddC>
            conv_fwd;

        auto res =
            ck::tensor_operation::TransformConv(e_layout, out_lengths, out_strides, conv_fwd);
        return res.transform_func(e_layout, out_lengths, out_strides, conv_fwd);
    }
    throw std::runtime_error("Incorrect conv spec");
}
auto transform_conv_1d(ck::index_t num_dim,
                       ck::tensor_operation::device::ConvolutionForwardSpecialization spec,
                       layouts e_layout,
                       ck::Array<ck::index_t, 4> out_lengths,
                       ck::Array<ck::index_t, 4> out_strides)
{
    if(num_dim == 1 &&
       spec == ck::tensor_operation::device::ConvolutionForwardSpecialization::Default)
    {
        ck::tensor_operation::TransformConvFwdToGemm<
            1,
            ck::tensor_operation::device::ConvolutionForwardSpecialization::Default>
            conv_fwd;

        auto res =
            ck::tensor_operation::TransformConv(e_layout, out_lengths, out_strides, conv_fwd);
        return res.transform_func(e_layout, out_lengths, out_strides, conv_fwd);
    }
    else if(num_dim == 1 &&
            spec == ck::tensor_operation::device::ConvolutionForwardSpecialization::Filter1x1Pad0)
    {
        ck::tensor_operation::TransformConvFwdToGemm<
            1,
            ck::tensor_operation::device::ConvolutionForwardSpecialization::Filter1x1Pad0>
            conv_fwd;

        auto res =
            ck::tensor_operation::TransformConv(e_layout, out_lengths, out_strides, conv_fwd);
        return res.transform_func(e_layout, out_lengths, out_strides, conv_fwd);
    }
    else if(num_dim == 1 && spec == ck::tensor_operation::device::ConvolutionForwardSpecialization::
                                        Filter1x1Stride1Pad0)
    {
        ck::tensor_operation::TransformConvFwdToGemm<
            1,
            ck::tensor_operation::device::ConvolutionForwardSpecialization::Filter1x1Stride1Pad0>
            conv_fwd;

        auto res =
            ck::tensor_operation::TransformConv(e_layout, out_lengths, out_strides, conv_fwd);
        return res.transform_func(e_layout, out_lengths, out_strides, conv_fwd);
    }
    else if(num_dim == 1 &&
            spec == ck::tensor_operation::device::ConvolutionForwardSpecialization::OddC)
    {
        ck::tensor_operation::TransformConvFwdToGemm<
            1,
            ck::tensor_operation::device::ConvolutionForwardSpecialization::OddC>
            conv_fwd;

        auto res =
            ck::tensor_operation::TransformConv(e_layout, out_lengths, out_strides, conv_fwd);
        return res.transform_func(e_layout, out_lengths, out_strides, conv_fwd);
    }
    throw std::runtime_error("Incorrect dims or conv spec");
}

template <typename CGridDesc_M_N>
auto block_2_etile(ck::index_t m_per_block, ck::index_t n_per_block, CGridDesc_M_N matrix_padder)
{
    // TODO: update and properly scale
    constexpr ck::Array<ck::index_t, 4> perblock{32, 64, 128, 256};
    int x = 0;
    for(int i = 0; i < 5; i++)
    {
        if(m_per_block == perblock[i])
        {
            x = i;
        }
    }
    /**switch(x){
            case 0: ck::BlockToCTileMap_M00_N0_M01Adapt<128, 256, CGridDesc_M_N> b2e(matrix_padder);
                            return b2e.CalculateGridSize(matrix_padder);

    }**/
    if(m_per_block == 32 && n_per_block == 64)
    {
        ck::BlockToCTileMap_M00_N0_M01Adapt<32, 64, CGridDesc_M_N> b2e(matrix_padder);
        return b2e.CalculateGridSize(matrix_padder);
    }
    else if(m_per_block == 32 && n_per_block == 128)
    {
        ck::BlockToCTileMap_M00_N0_M01Adapt<32, 128, CGridDesc_M_N> b2e(matrix_padder);
        return b2e.CalculateGridSize(matrix_padder);
    }
    else if(m_per_block == 64 && n_per_block == 32)
    {
        ck::BlockToCTileMap_M00_N0_M01Adapt<64, 32, CGridDesc_M_N> b2e(matrix_padder);
        return b2e.CalculateGridSize(matrix_padder);
    }
    else if(m_per_block == 64 && n_per_block == 64)
    {
        ck::BlockToCTileMap_M00_N0_M01Adapt<64, 64, CGridDesc_M_N> b2e(matrix_padder);
        return b2e.CalculateGridSize(matrix_padder);
    }
    else if(m_per_block == 64 && n_per_block == 128)
    {
        ck::BlockToCTileMap_M00_N0_M01Adapt<64, 128, CGridDesc_M_N> b2e(matrix_padder);
        return b2e.CalculateGridSize(matrix_padder);
    }
    else if(m_per_block == 128 && n_per_block == 32)
    {
        ck::BlockToCTileMap_M00_N0_M01Adapt<128, 32, CGridDesc_M_N> b2e(matrix_padder);
        return b2e.CalculateGridSize(matrix_padder);
    }
    else if(m_per_block == 128 && n_per_block == 64)
    {
        ck::BlockToCTileMap_M00_N0_M01Adapt<128, 64, CGridDesc_M_N> b2e(matrix_padder);
        return b2e.CalculateGridSize(matrix_padder);
    }
    else if(m_per_block == 128 && n_per_block == 128)
    {
        ck::BlockToCTileMap_M00_N0_M01Adapt<128, 128, CGridDesc_M_N> b2e(matrix_padder);
        return b2e.CalculateGridSize(matrix_padder);
    }
    else if(m_per_block == 128 && n_per_block == 256)
    {
        ck::BlockToCTileMap_M00_N0_M01Adapt<128, 256, CGridDesc_M_N> b2e(matrix_padder);
        return b2e.CalculateGridSize(matrix_padder);
    }
    else if(m_per_block == 256 && n_per_block == 128)
    {
        ck::BlockToCTileMap_M00_N0_M01Adapt<256, 128, CGridDesc_M_N> b2e(matrix_padder);
        return b2e.CalculateGridSize(matrix_padder);
    }
    // TODO: figure out how to pass parameters properly -> not scalable for this method
    /**if(m_per_block == 128 && n_per_block == 256)
    {
        ck::BlockToCTileMap_M00_N0_M01Adapt<128, 256, CGridDesc_M_N> b2e(matrix_padder);
        return b2e.CalculateGridSize(matrix_padder);
    }**/
    // auto tmp = ck::BlockToCTileMap_M00_N0_M01Adapt<perblock[x], n_per_block,
    // CGridDesc_M_N>(matrix_padder);
    // b2e	= ck::BlockToCTileMap_M00_N0_M01Adapt(m_per_block, n_per_block);
    throw std::runtime_error("Incorrect template parameters");
}

// wrapper functions by dims to get grid size - uses above 3 functions
// TODO: can get around this using templating?
auto get_launch_params_1d(ck::index_t m_per_block,
                          ck::index_t n_per_block,
                          ck::index_t k_per_block,
                          ck::index_t num_dim,
                          ck::tensor_operation::device::ConvolutionForwardSpecialization spec,
                          ck::tensor_operation::device::GemmSpecialization gemm,
                          layouts e_layout,
                          ck::Array<ck::index_t, 4> out_lengths,
                          ck::Array<ck::index_t, 4> out_strides)
{
    auto conv_to_gemm_transformer =
        transform_conv_1d(num_dim, spec, e_layout, out_lengths, out_strides);
    auto matrix_padder = pad(m_per_block, n_per_block, k_per_block, gemm, conv_to_gemm_transformer);
    auto b2e           = block_2_etile(m_per_block, n_per_block, matrix_padder);
    return b2e;
}
auto get_launch_params(ck::index_t m_per_block,
                       ck::index_t n_per_block,
                       ck::index_t k_per_block,
                       ck::index_t num_dim,
                       ck::tensor_operation::device::ConvolutionForwardSpecialization spec,
                       ck::tensor_operation::device::GemmSpecialization gemm,
                       layouts e_layout,
                       ck::Array<ck::index_t, 5> out_lengths,
                       ck::Array<ck::index_t, 5> out_strides)
{
    auto conv_to_gemm_transformer =
        transform_conv(num_dim, spec, e_layout, out_lengths, out_strides);
    auto matrix_padder = pad(m_per_block, n_per_block, k_per_block, gemm, conv_to_gemm_transformer);
    auto b2e           = block_2_etile(m_per_block, n_per_block, matrix_padder);
    return b2e;
}
auto get_launch_params_3d(ck::index_t m_per_block,
                          ck::index_t n_per_block,
                          ck::index_t k_per_block,
                          ck::index_t num_dim,
                          ck::tensor_operation::device::ConvolutionForwardSpecialization spec,
                          ck::tensor_operation::device::GemmSpecialization gemm,
                          layouts e_layout,
                          ck::Array<ck::index_t, 6> out_lengths,
                          ck::Array<ck::index_t, 6> out_strides)
{
    auto conv_to_gemm_transformer =
        transform_conv_3d(num_dim, spec, e_layout, out_lengths, out_strides);
    auto matrix_padder = pad(m_per_block, n_per_block, k_per_block, gemm, conv_to_gemm_transformer);
    auto b2e           = block_2_etile(m_per_block, n_per_block, matrix_padder);
    return b2e;
}
