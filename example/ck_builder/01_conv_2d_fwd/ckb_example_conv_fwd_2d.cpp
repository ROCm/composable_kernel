
#include "ck_tile/builder/conv_builder.hpp"
#include "../utils/types.hpp"

int main() {

    namespace ckb = ck_tile::builder;
    namespace ckb_examples = ck_tile::builder::examples;

    constexpr size_t m_tile = 32;
    constexpr size_t n_tile = 16;
    constexpr size_t k_tile = 64;
    constexpr size_t k0 = 4;
    constexpr size_t k1 = 1;

    constexpr ckb_examples::ConvSignature FwdConvSignature 
    {
        .spatial_dim = 2,
        .direction = ckb::ConvDirection::FORWARD,
        .layout = ckb::GroupConvLayout::CHANNELS_LAST,
        .data_type = ckb::DataType::BF16,
    };
    static_assert(ckb::ValidConvSignature<FwdConvSignature>); 

    constexpr ckb::ThreadBlock FwdThreadBlock
    {
        .block_size = 256, 
        .submatrix = {.m = m_tile, .n = n_tile, .k = k_tile}
    };

    constexpr ckb_examples::BlockTransfer FwdBlockTransfer
    {
        .thread_cluster_dims_a = {.k0 = k0, .m = m_tile, .k1 = k1},
        .thread_cluster_dims_b = {.k0 = k0, .n = n_tile, .k1 = k1},
        .thread_cluster_dims_c = {
            .m_block = 1, .m_wave_per_xdl = 32, .n_block = 1, .n_wave_per_xdl = 8}
    };

    constexpr ckb_examples::ConvAlgorithm FwdConvAlgorithm
    {
        .thread_block = FwdThreadBlock,
        .tuning_params = {.ak1 = 16, .bk1 = 16, .m_xdl_per_wave = 2, .n_xdl_per_wave = 2},
        .block_transfer = FwdBlockTransfer,
        .pipeline_version = ckb::BlockGemmPipelineVersion::V1,
    };

    using Builder = ckb::ConvBuilder<FwdConvSignature, FwdConvAlgorithm>;
    const auto kernel_string = Builder::Instance::TypeString();

    std::cout << "Generated kernel: " << kernel_string << std::endl;

    return 0;
}
