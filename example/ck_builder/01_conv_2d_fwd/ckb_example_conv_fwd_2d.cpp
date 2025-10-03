
#include "ck_tile/builder/conv_builder.hpp"
#include "../utils/types.hpp"

int main() {

    namespace ckb = ck_tile::builder;
    namespace ckb_examples = ck_tile::builder::examples;

    constexpr size_t m_tile = 128;
    constexpr size_t n_tile = 128;
    constexpr size_t k_tile = 32;

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
        .thread_cluster_dims_a = {.k0 = 4, .m = 64, .k1 = 1},
        .thread_cluster_dims_b = {.k0 = 4, .n = 64, .k1 = 1},
        .thread_cluster_dims_c = {
            .m_block = 1, .m_wave_per_xdl = 32, .n_block = 1, .n_wave_per_xdl = 8},
        .vector_transfer_a = {
            .src_vector_dim = 2, .src_scaler_per_vector = 8, .dest_scaler_per_vector_k1 = 8, .add_extra = true},
        .vector_transfer_b = {
            .src_vector_dim = 2, .src_scaler_per_vector = 8, .dest_scaler_per_vector_k1 = 8, .add_extra = true},
        .vector_transfer_c = {
            .m_xdl_per_wave_per_shuffle = 1, .n_xdl_per_wave_per_shuffle = 2, .scaler_per_vector = 8},
        .a_thread_cluster_access_order = {1, 0, 2},
        .b_thread_cluster_access_order = {1, 0, 2},
        .a_source_access_order = {1, 0, 2},
        .b_source_access_order = {1, 0, 2}
    };
    

    constexpr ckb_examples::ConvAlgorithm FwdConvAlgorithm
    {
        .thread_block = FwdThreadBlock,
        .tuning_params = {.ak1 = 8, .bk1 = 8, .m_xdl_per_wave = 1, .n_xdl_per_wave = 4},
        .block_transfer = FwdBlockTransfer,
        .pipeline_version = ckb::BlockGemmPipelineVersion::V1,
    };

    using Builder = ckb::ConvBuilder<FwdConvSignature, FwdConvAlgorithm>;
    const auto kernel_string = Builder::Instance::TypeString();

    std::cout << "Generated kernel: " << kernel_string << std::endl;

    // The invoker is the entrypoint to launch the kernel.
    // Creating the invoker triggers the validation of the builder configuration.
    //auto invoker = Builder::Instance::MakeInvoker();
    //(void)invoker;

    return 0;
}
