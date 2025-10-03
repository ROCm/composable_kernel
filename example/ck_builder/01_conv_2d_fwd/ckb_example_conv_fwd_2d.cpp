
#include "ck_tile/builder/conv_builder.hpp"
#include "../utils/types.hpp"

int main() 
{
    namespace ckb = ck_tile::builder;
    namespace ckb_examples = ck_tile::builder::examples;

    constexpr ckb_examples::ConvSignature FwdConvSignature 
    {
        .spatial_dim = 2,
        .direction = ckb::ConvDirection::FORWARD,
        .layout = ckb::GroupConvLayout::CHANNELS_LAST,
        .data_type = ckb::DataType::BF16,
    };
    static_assert(ckb::ValidConvSignature<FwdConvSignature>); 

    // To get valid configuration parameters, refer to "device_grouped_conv_fwd_xdl_comp_instance.hpp". 
    // This file contains the current instances of the kernel DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3.
    // Currently the build has this kernel hard-coded. 
    // In the future, we may need builders per kernel type since they typically have slightly different parameters. 

    constexpr ckb::ThreadBlock FwdThreadBlock
    {
        .block_size = 256, 
        .submatrix = {.m = 256, .n = 256, .k = 32} // Tile sizes
    };

    constexpr ckb::ConvTuningParams FwdTuningParams
    {
        .ak1 = 8, .bk1 = 8, .m_per_xdl=32, .n_per_xdl = 32, .m_xdl_per_wave = 4, .n_xdl_per_wave = 4
    };

    constexpr ckb_examples::BlockTransfer FwdBlockTransfer
    {
        .thread_cluster_dims_a = {.k0 = 4, .m = 64, .k1 = 1},
        .thread_cluster_dims_b = {.k0 = 4, .n = 64, .k1 = 1},
        .thread_cluster_dims_c = {
            .m_block = 1, .m_wave_per_xdl = 32, .n_block = 1, .n_wave_per_xdl = 8},
        .vector_transfer_a = {
            .src_vector_dim = 2, .src_scaler_per_vector = 2, .dest_scaler_per_vector_k1 = 8, .add_extra = false},
        .vector_transfer_b = {
            .src_vector_dim = 2, .src_scaler_per_vector = 8, .dest_scaler_per_vector_k1 = 8, .add_extra = false},
        .vector_transfer_c = {
            .m_xdl_per_wave_per_shuffle = 1, .n_xdl_per_wave_per_shuffle = 1, .scaler_per_vector = 8},
        .a_thread_cluster_access_order = {1, 0, 2},
        .b_thread_cluster_access_order = {1, 0, 2},
        .a_source_access_order = {1, 0, 2},
        .b_source_access_order = {1, 0, 2}
    };

    constexpr ckb_examples::ConvAlgorithm FwdConvAlgorithm
    {
        .thread_block = FwdThreadBlock,
        .tuning_params = FwdTuningParams,
        .block_transfer = FwdBlockTransfer,
        .pipeline_version = ckb::BlockGemmPipelineVersion::V4,
    };

    using Builder = ckb::ConvBuilder<FwdConvSignature, FwdConvAlgorithm>;
    const auto kernel_string = Builder::Instance::TypeString();
    std::cout << "Generated kernel: " << kernel_string << std::endl;

    // The invoker is the entrypoint to launch the kernel.
    // Creating the invoker triggers the validation of the builder configuration,
    // that is, the combination of all builder parameters is checked at compile time.
    auto invoker = Builder::Instance::MakeInvoker();
    
    // TODO: Prepare actual data and launch the kernel.
    (void)invoker;

    return 0;
}
