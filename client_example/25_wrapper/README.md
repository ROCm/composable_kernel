# CK Wrapper Client Examples
## GEMM tutorial

The tutorial shows how to implement matrix multiplication using CK Wrapper. The tutorial presents the base version of gemm without most of the optimizations. It is worth keeping in mind that CK has kernels with different optimizations . If you want to use these optimizations, you can implement them using the CK wrapper or directly use available instances in CK. You can also refer to [optimized gemm example](https://github.com/ROCm/composable_kernel/blob/develop/client_example/25_wrapper/wrapper_optimized_gemm.cpp) implemented using CK Wrapper based on the [gridwise_gemm_xdlops_v2r3](https://github.com/ROCm/composable_kernel/blob/develop/include/ck/tensor_operation/gpu/grid/gridwise_gemm_xdlops_v2r3.hpp) implementation.

The kernel definition might look like this:

```
template <typename DataType,
          typename GemmTraits,
          ck::index_t scalar_per_vector,
          typename BlockShape,
          typename ThreadLayout>
__global__ void __CK_WRAPPER_LAUNCH_BOUNDS__ DeviceGemm(const void* p_a,
                                                        const void* p_b,
                                                        void* p_c,
                                                        const ck::index_t M,
                                                        const ck::index_t N,
                                                        const ck::index_t K,
                                                        const BlockShape tile_shape,
                                                        const ThreadLayout thread_layout)
```

As arguments, we pass pointers to global memory and matrix dimensions. Additionally, we pass a selected lengths of processed data through each block (`tile_shape`) and thread layout (`thread_layout`).
As compilation time parameters, we define the data type, [traits for the gemm operation](https://github.com/ROCm/composable_kernel/blob/develop/include/ck/wrapper/traits/blockwise_gemm_xdl_traits.hpp) and scalar per vector value during copy.


The first step is to create layouts for global and lds memory:
```
    // Specify layouts for global memory.
    const auto a_global_layout =
        ck::wrapper::make_layout(ck::make_tuple(M, K), ck::make_tuple(K, 1));
    const auto b_global_layout =
        ck::wrapper::make_layout(ck::make_tuple(N, K), ck::make_tuple(K, 1));
    const auto c_global_layout =
        ck::wrapper::make_layout(ck::make_tuple(M, N), ck::make_tuple(N, 1));

    // Specify layouts for tiles.
    constexpr auto a_tile_layout = ck::wrapper::make_layout(
        ck::make_tuple(MPerBlock, KPerBlock), ck::make_tuple(KPerBlock, ck::Number<1>{}));
    constexpr auto b_tile_layout = ck::wrapper::make_layout(
        ck::make_tuple(NPerBlock, KPerBlock), ck::make_tuple(KPerBlock, ck::Number<1>{}));
    constexpr auto c_tile_layout = ck::wrapper::make_layout(
        ck::make_tuple(MPerBlock, NPerBlock), ck::make_tuple(NPerBlock, ck::Number<1>{}));

    // Apply padding for global memory.
    auto a_global_layout_padded = ck::wrapper::pad(a_global_layout, shape(a_tile_layout));
    auto b_global_layout_padded = ck::wrapper::pad(b_global_layout, shape(b_tile_layout));
    auto c_global_layout_padded = ck::wrapper::pad(c_global_layout, shape(c_tile_layout));
```
We pad layouts for global tensors in case of M, N, K are not divisible by MPerBlock, NPerBlock, KPerBlock.

The next step is to create tensors for global and lds memory:

```
    // Make tensors for global memory.
    auto a_global_tensor = ck::wrapper::make_tensor<ck::wrapper::MemoryTypeEnum::Global>(
        static_cast<const DataType*>(p_a), a_global_layout_padded);
    auto b_global_tensor = ck::wrapper::make_tensor<ck::wrapper::MemoryTypeEnum::Global>(
        static_cast<const DataType*>(p_b), b_global_layout_padded);
    auto c_global_tensor = ck::wrapper::make_tensor<ck::wrapper::MemoryTypeEnum::Global>(
        static_cast<DataType*>(p_c), c_global_layout_padded);

    // Allocate lds memory.
    __shared__ DataType lds_a[ck::wrapper::size(a_tile_layout)];
    __shared__ DataType lds_b[ck::wrapper::size(b_tile_layout)];

    // Make tensors for lds memory.
    auto a_lds_tensor = ck::wrapper::make_tensor<ck::wrapper::MemoryTypeEnum::Lds>(
        static_cast<DataType*>(lds_a), a_tile_layout);
    auto b_lds_tensor = ck::wrapper::make_tensor<ck::wrapper::MemoryTypeEnum::Lds>(
        static_cast<DataType*>(lds_b), b_tile_layout);
```
There is also a need to specify parameters for copy and convert block indexes to tuple:
```
    // Specify block index as tuple.
    const auto block_idxs = ck::make_tuple(static_cast<ck::index_t>(blockIdx.x),
                                           static_cast<ck::index_t>(blockIdx.y),
                                           ck::wrapper::slice());
    // Specify access parameters for copy.
    using DimAccessOrder             = ck::Tuple<ck::Number<0>, ck::Number<1>>;
    constexpr ck::index_t vector_dim = 1;
```

We create a local tile (per block) and local partitions (per thread) for the global memory `C`. We also define and clear an output register `c_vgpr_reg` for the accumulation.
```
    auto c_global_local_tile = ck::wrapper::make_local_tile(
        c_global_tensor,
        tile_shape,
        block_idxs,
        make_tuple(ck::Number<1>{}, ck::Number<1>{}, ck::wrapper::slice(KPerBlock)));
    auto c_global_local_partition =
        ck::wrapper::make_blockwise_gemm_xdl_c_local_partition<DataType,
                                                               decltype(a_tile_layout),
                                                               decltype(b_tile_layout),
                                                               ck::wrapper::size(thread_layout),
                                                               GemmTraits>(c_global_local_tile);
    // Create C vgpr to accumulate results.
    auto c_vgpr_reg = ck::wrapper::make_blockwise_gemm_xdl_c_vgpr<DataType,
                                                                  decltype(a_tile_layout),
                                                                  decltype(b_tile_layout),
                                                                  ck::wrapper::size(thread_layout),
                                                                  GemmTraits>();
    // Clear C vgpr.
    ck::wrapper::clear(c_vgpr_reg);
```
Two specific functions for blockwise_gemm are used here: `make_blockwise_gemm_xdl_c_local_partition` and `make_blockwise_gemm_xdl_c_vgpr`. This helps to choose the appropriate partition for the C output and to define tensors with specific layouts for `blockwise_gemm`. In the following, only generic functions for the ck wrapper will be used.

Finally we can create the compute loop:
```
    const ck::index_t num_loop = ck::math::integer_divide_ceil(K, KPerBlock);
    ck::index_t i              = 0;
    do
    {
        // Get KPerBlock slice.
        const auto k_slice           = ck::wrapper::slice(i * KPerBlock, (i + 1) * KPerBlock);
        auto a_global_tensor_k_slice = a_global_tensor(ck::wrapper::slice(), k_slice);
        auto b_global_tensor_k_slice = b_global_tensor(ck::wrapper::slice(), k_slice);
        // Create local tiles for A and B.
        auto a_global_local_tile = ck::wrapper::make_local_tile(
            a_global_tensor_k_slice,
            tile_shape,
            block_idxs,
            make_tuple(ck::Number<1>{}, ck::wrapper::slice(N), ck::Number<1>{}));
        auto b_global_local_tile = ck::wrapper::make_local_tile(
            b_global_tensor_k_slice,
            tile_shape,
            block_idxs,
            make_tuple(ck::wrapper::slice(M), ck::Number<1>{}, ck::Number<1>{}));
        // Copy from global to lds.
        ck::wrapper::blockwise_copy<DimAccessOrder, vector_dim, scalar_per_vector>(
            a_global_local_tile, a_lds_tensor, thread_layout);
        ck::wrapper::blockwise_copy<DimAccessOrder, vector_dim, scalar_per_vector>(
            b_global_local_tile, b_lds_tensor, thread_layout);
        // Synchronize lds.
        ck::block_sync_lds();
        // Execute blockwise gemm.
        ck::wrapper::blockwise_gemm_xdl<DataType, ck::wrapper::size(thread_layout), GemmTraits>(
            a_lds_tensor, b_lds_tensor, c_vgpr_reg);

        ++i;
    } while(i < num_loop);
```
Loop iterate over `K / KPerBlock`. Each time a local tile is created for A and B tensors (tensor per block) then the data is copied from global memory to lds. The `blockwise_gemm` function perform gemm operation on `a_lds_tensor` and `b_lds_tensor` and store results in `c_vgpr_reg`.

At the end result from `c_vgpr_reg` is stored in C local partition (tensor per thread):
```
    ck::wrapper::copy(c_vgpr_reg, c_global_local_partition);
```

For more details, the entire example is available [here](https://github.com/ROCm/composable_kernel/blob/develop/client_example/25_wrapper/wrapper_basic_gemm.cpp).
