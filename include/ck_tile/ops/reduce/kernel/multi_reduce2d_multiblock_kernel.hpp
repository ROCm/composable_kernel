// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2025, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/ops/common.hpp"
#include "ck_tile/ops/reduce/block/block_reduce.hpp"
#include "ck_tile/ops/reduce/pipeline/reduce2d_default_policy.hpp"
#include "ck_tile/core/arch/generic_memory_space_atomic.hpp"

// Multi Reduce2d MultiBlock Kernel:
// =======================================
// This kernel implements multiple 2D reduction operation that reduces data along the specified
// dimensions of a matrix. Reductions happen across multiple thread blocks based on intermediate
// reductions at the thread level.


// TODO
// 1. Modify the example to handle:
//      a. Initialize an output buffer with the identity value of the operation
//      b. Provide the atomic add as input type, necessary for the blockwise reduction (or a tuple of atomic ops, one for each reduction)
//      c. Provide a number of cluster
// 2. Create cluster id and thread cluster id
// 3. Initialize the output buffer (and inter block accumulator) with the operation identity
// 4. Initialize the inter-block LDS buffer (for inter thread, but intra block reduction)
// 5. We probably need an internal buffer for processing within this thread, but variables (on register) will do
// 6. Process the subtile window and perform the in-thread reduction for each reduction
// 7. Perform the intra-block reduction for each reduction
// 8. Write the inter-block reduction: look at PartitionedBlockwiseReduction
// 9. Write back into the output buffer, which is also the inter-block accumulator
// 10. Fix the example, to test

// This is what the CK kernel is doing, step-by-step:
// X: done, V: semi-done, need to verify, S: purposedly skipped, blank: not done
//
// [X] 1. Define the LDS buffer, the size of the block, used by thread within this block --> its the variable smem
// [X] 2. Define the global input buffer in global memory. That going to be used to transfer data --> In CK tile that's given as input, nothing to do here
// [V] 3. (p_reduce_work_buffer:LDS) Define the object pointing to LDS buffer, --> I assume we can reuse smem for this ???? To verify
// [X] 4. (in_thread_buf:Reg) Define the thread local buffer, the size of a sub-tile window. Buffer used to read input data from global memory to thread register --> in CK-tile that's done when we load the tile window (buffer_view, transformed_x_tensor and x_window are handling that)
// [S] 5. (in_thread_buf_tuple:Reg) Define a tuple of thread local buffer, one for each reduction operation. It's used for the elementwise reduction --> It's an add-on feature we can add later. Skipped for now
// [ ] 6. (accu_value_buf_tuple:Reg) Define a tuple of thread local buffer, one for each ops, used for accumulation. The size is the dimension of the kept dim. --> TODO: figure out if we have smem do we need this? 
// [ ] 7. Initialize accu_value_buf_tuple to the identity value of the operation --> TODO: Figure out if we need this
// [X] 8. Figure out which cluster id and which thread id we are based on the cluster size --> That's our cluster_id, block_group_id, etc
// [ ] 9. Calculate the total number of elements accross the multiple sub-tile this thread will process. K is the reduced dim, so this total is K size * number of sub-tiles (i.e. num_k_block_tile_iteration). --> it's out num_n_tile_iteration, however (TODO) we should check if the have the same meaning here!
// [V] 10. (thread_buffer_desc) Create a tensor descriptor the size of the sub-tile (ie. tile window in ck-tile context) --> It's used to load the tile window BUT also used in the elementwise op, to calculate offset. Leave it to undone until we start focusing on the elementwise op
// [X] 11. Define the object that going to pull the input sub-tile data into the thread register --> This is handled by load_tile in CK-tile
// [ ] 12. (in_thread_copy_step) define variable used to move offset on the first element of the sub-tile to read??
// [ ] 13. (reducedTiles) define and initialize the variable used to count the number of sub-tile we process
// [ ] 14. Loop over the number of sub-tiles to process, to process run the operation (but does not perform the "reduction", it's the "map" step in the map-reduce)
// [ ]     14.1 Pull the data from global memory to thread register
// [ ]     14.2 For each reduction operation, do the reduction and store it in in_thread_buf_tuple:Reg
// [ ]     14.3 Then another reduction??? The one above seems to be doing a "elementwise" reduction why this one is the real reduction. Is this one needed it seems to be like multiplying each terms by a factor, squaring each term, etc --> Skipped for now
// [ ]     14.4 move the tile window to the next sub-tile
// [ ] 15. (Main reduction loop) Loop over the number of reduction operation
// [ ]     15.1 Do the blockwise reduction, using the buffer reduce_work_buf:LDS and write in back in accu_value_buf_tuple:Reg
// [ ]     15.2 Loop over the size of the kept dim and if we are thread 0, call acc_elementwise_op_tuple[iR] on accu_value_buf_tuple:Reg
// [V]     15.3 Write back to global memory, using atomic operation defined by OutMemoryDataOperation
//
// Outstanding questions: 
// [ ] do we still need the cross warp sync? I would say yes, but need to verify

namespace ck_tile {

template <typename Problem_, typename Policy_ = Reduce2dDefaultPolicy>
struct MultiReduceMultiblock
{
    using Problem = ck_tile::remove_cvref_t<Problem_>;
    using Policy  = ck_tile::remove_cvref_t<Policy_>;

    using XDataType       = ck_tile::remove_cvref_t<typename Problem::XDataType>;
    using ComputeDataType = ck_tile::remove_cvref_t<typename Problem::ComputeDataType>;
    using YDataType       = ck_tile::remove_cvref_t<typename Problem::YDataType>;
    // using ThreadTileSize = ck_tile::sequence<Problem::BlockShape::Block_M,
    //                                              Problem::BlockShape::Block_N>; // TODO: check if it's the right shape!


    static constexpr index_t kBlockSize = Problem::BlockShape::BlockSize;
    // the block shapes are wrong, it should be the real size of a block, not the poorly named Block_X, which are the size of sub-tile!
    // static constexpr auto thread_cluster_desc = make_cluster_descriptor(ck_tile::sequence<Problem::BlockShape::Block_M, Problem::BlockShape::Block_N>{}); // TODO: order, to handle the transpose case!
    static constexpr auto thread_cluster_desc = make_cluster_descriptor(ck_tile::sequence<Problem::BlockShape::Block_M/Problem::BlockShape::ThreadTile_M, Problem::BlockShape::Block_N/Problem::BlockShape::ThreadTile_N>{}); // TODO: order, to handle the transpose case!
        //make_cluster_descriptor(ThreadClusterLengths_M_K{}, ThreadClusterArrangeOrder{});

    CK_TILE_HOST static constexpr auto BlockSize()
    {
        return is_wave32() ? kBlockSize / 2 : kBlockSize;
    }

    CK_TILE_HOST static void CalculateBlockGroupParams(
        const std::size_t reduce_total_length,
        const std::size_t K_BlockTileSize,
        int& num_block_tile_iterations,
        int& block_group_size)
    {
        // TODO: fix that, it seems to be wrong on bigger sizes in the unit test
        num_block_tile_iterations = std::max(
            1, static_cast<int>(std::ceil((reduce_total_length - 1.0) / (127.0 * K_BlockTileSize))));

        block_group_size = (reduce_total_length + (K_BlockTileSize * num_block_tile_iterations) - 1) /
                        (K_BlockTileSize * num_block_tile_iterations);
    }
        private:
    // Helper function to calculate optimal vector size for input tensor
    template <typename InputShape, typename ReduceDims>
    static constexpr index_t CalculateInputVectorSize()
    {
        using S                              = typename Problem::BlockShape;
        constexpr index_t memory_vector_size = 16 / sizeof(XDataType); // Vectorization
        constexpr index_t thread_tile_vector_size =
            S::ThreadTile_N; // In the continuous dimension, within the tile

        constexpr auto innermost_reduce_dim    = ReduceDims{}.at(number<ReduceDims{}.size() - 1>{});
        constexpr bool is_innermost_contiguous = (innermost_reduce_dim == InputShape{}.size() - 1);

        constexpr index_t stride_based_vector_size =
            is_innermost_contiguous
                ? ck_tile::min(memory_vector_size, thread_tile_vector_size)
                : 1; // Move at "vectorization" steps if continuous otherwise 1 step

        return stride_based_vector_size;
    }

    static constexpr index_t CalculateOutputVectorSize()
    {
        using S                                   = typename Problem::BlockShape;
        constexpr index_t memory_vector_size      = 16 / sizeof(YDataType);
        constexpr index_t thread_tile_vector_size = S::ThreadTile_M; // ?
        constexpr index_t vector_size = ck_tile::min(memory_vector_size, thread_tile_vector_size);

        return vector_size;
    }

    public:
    template <typename InputShape, typename InputStrides, typename KeptDim, typename ReduceDims, typename ElementwiseOps = void> //, typename BlockwiseAccOps> // TODO handle the elementwise ops better (i.e. no void)
    CK_TILE_DEVICE void operator()(const XDataType* p_x,
                                   YDataType* p_y_tuple,
                                   InputShape input_shape,
                                   InputStrides input_strides,
                                   KeptDim kept_dim,
                                   ReduceDims reduce_dims,
                                   index_t output_tensor_offset,
                                   [[maybe_unused]] index_t block_group_size,
                                   [[maybe_unused]] index_t num_block_tile_iterations,
                                   [[maybe_unused]] ElementwiseOps elementwise_ops = ElementwiseOps{}) const
                                //    [[maybe_unused]] BlockwiseAccOps blockwise_acc_ops) const

    {

        // TODO: static checks about the different tuple sizes
        
        using S                      = typename Problem::BlockShape;
        // const auto iM                = get_block_id() * S::Block_M;
        auto reduce_funcs            = typename Problem::ReduceOp{};
        const auto number_operations = reduce_funcs.size();

        static_assert(number_operations > 0,
                      "Error: At least one reduction operation must be specified!");

        static_assert(kept_dim.size() + reduce_dims.size() == InputShape::size(),
                      "Size of kept dimensions + reduced dimensions must equal input tensor rank");

        const auto kept_lens = [&]() {
            return generate_tuple([&](auto I) { return input_shape.at(number<kept_dim.at(I)>{}); },
                                  number<kept_dim.size()>{});
        }();
        const auto reduce_lens = [&]() {
            return generate_tuple(
                [&](auto I) { return input_shape.at(number<reduce_dims.at(I)>{}); },
                number<reduce_dims.size()>{});
        }();

        const auto thread_local_id = get_thread_id();
        const auto block_global_id = get_block_id(); // Hardware block id
        const auto block_group_id = block_global_id / block_group_size; // Logical block group id
        const auto block_local_id = block_global_id % block_group_size; // Logical block id within the block group

        // printf("Block Local ID: %d, Block Group ID: %d, Thread Local ID: %d, Block Global ID: %d\n", block_local_id, block_group_id, thread_local_id, block_global_id);

        const auto thread_cluster_idx =
            thread_cluster_desc.calculate_bottom_index(make_tuple(thread_local_id));
        const auto thread_m_cluster_id = thread_cluster_idx[number<0>{}]; // cluster index in M dimension
        const auto thread_n_cluster_id = thread_cluster_idx[number<1>{}]; // cluster index in N dimension

        const auto kept_merge_transform =
            make_merge_transform(kept_lens); // Dimension(s) not reduced are being flattened
        const auto reduce_merge_transform =
            make_merge_transform(reduce_lens); // Dimension(s) to reduce are being flattened

        const auto custom_padding_values = ck_tile::apply(
            [](auto... args) {
                return ck_tile::make_tuple(
                    type_convert<XDataType>(args.template GetIdentityValue<ComputeDataType>())...);
            },
            reduce_funcs); // Get the identity element for each operation

        constexpr auto x_tensor_vector_size =
            CalculateInputVectorSize<InputShape, ReduceDims>(); // Move at "vectorization" steps if
                                                                // continuous otherwise 1 step

        auto desc = make_naive_tensor_descriptor(input_shape,
                                                 input_strides,
                                                 number<x_tensor_vector_size>{},
                                                 number<1>{}); // Create the tensor descriptor (TODO: check if this buffer is still useful)

        auto buffer_view = make_buffer_view<address_space_enum::global>(
            p_x,
            desc.get_element_space_size(),
            custom_padding_values.get(
                number<0>{})); // Input tensor buffer view, currently using the first operation
                               // identity element as padding?? TODO: check this

        const auto x_tensor = tensor_view<decltype(buffer_view), decltype(desc)>{
            buffer_view, desc}; // Tensor view over the buffer view and tensor descriptor
        const auto transformed_x_tensor = pad_tensor_view(
            transform_tensor_view(x_tensor,
                                  make_tuple(kept_merge_transform, reduce_merge_transform),
                                  make_tuple(kept_dim, reduce_dims),
                                  make_tuple(sequence<0>{}, sequence<1>{})),
            make_tuple(number<S::Block_M>{}, number<S::Block_N>{}),
            sequence<0, 1>{}); // Effective transform the input tensor to 2D (kept, reduced) and pad
                               // to block size

        // const auto kept_strides = [&]() {
        //     return generate_tuple(
        //         [&](auto I) {
        //             index_t stride = 1;
        //             static_for<I + 1, kept_dim.size(), 1>{}(
        //                 [&](auto J) { stride *= kept_lens.at(number<J>{}); });
        //             return stride;
        //         },
        //         number<kept_dim.size()>{});
        // }(); // Compute the strides, for a dimensions (a,b,c), the strides are (b*c, c, 1)

        // constexpr auto y_tensor_vector_size = CalculateOutputVectorSize();

        // auto y_tile_windows = generate_tuple(
        //     [&]([[maybe_unused]] auto i) {
        //         const auto y_tensor_view = make_naive_tensor_view<address_space_enum::global>(
        //             p_y_tuple + (i * output_tensor_offset),
        //             kept_lens,
        //             kept_strides,
        //             number<y_tensor_vector_size>{},
        //             number<1>{});

        //         const auto y_merge = transform_tensor_view(
        //             y_tensor_view,
        //             make_tuple(kept_merge_transform),
        //             make_tuple(typename arithmetic_sequence_gen<0, kept_dim.size(), 1>::type{}),
        //             make_tuple(sequence<0>{}));

        //         return make_tile_window(y_merge, make_tuple(number<S::Block_M>{}), {iM});
        //     },
        //     number<number_operations>{});

        __shared__ char smem[Policy::template GetSmemSize<Problem>()]; // shared memory reused by the different operations

        const auto merged_reduce_len =
            transformed_x_tensor.get_tensor_descriptor().get_lengths().at(
                number<1>{}); // Get the last dimension size (reduced dimension)
        // index_t num_n_tile_iteration = __builtin_amdgcn_readfirstlane(integer_divide_ceil(
        //     merged_reduce_len, S::Block_N)); // Figure out the number of iterations needed to cover
        //                                      // the reduced dimension with Block size N
        index_t num_n_tile_iteration = __builtin_amdgcn_readfirstlane(integer_divide_ceil(
            merged_reduce_len/block_group_size, S::Block_N)); // Figure out the number of iterations needed to cover
                                             // the reduced dimension with Block size N
        // index_t num_n_tile_iteration = num_block_tile_iterations;

        auto block_reduce2d =
            Policy::template GetBlockReduce2d<Problem>(); // Get the block reduction , at thread
                                                          // level, function
        auto block_reduce2d_sync =
            Policy::template GetBlockReduce2dSync<Problem>(); // Get the block sync, at warp level,
                                                              // function
        auto block_reduce2d_cross_warp_sync =
            Policy::template GetBlockReduce2dCrossWarpSync<Problem>(); // Get the block for the
                                                                       // cross warp level sync

        // auto reduce_size_per_block = Problem::BlockShape::ThreadTile_M*Problem::BlockShape::ThreadTile_N*num_block_tile_iterations; // TODO: What about num_n_tile_iteration???  KThreadSliceSize * num_k_block_tile_iteration;

        index_t m_offset = S::Block_M * block_group_id;// + thread_m_cluster_id * Problem::BlockShape::ThreadTile_M; // block group_id corresponds to the logical index, in term of block, in the M dimension.
        // index_t n_offset = S::Block_N * num_n_tile_iteration * block_local_id + thread_n_cluster_id * Problem::BlockShape::ThreadTile_N;
        index_t n_offset = S::Block_N * num_n_tile_iteration * block_local_id; //+ thread_n_cluster_id * Problem::BlockShape::ThreadTile_N;

        // TODO: check if we need to update num_n_tile_iteration here. E.g. if we have iteration set two 2 but we only need two blocks to cover the reduce dim, then the last block needs only to do one iteration

        if(thread_local_id == 0 && block_group_id == 1 && block_local_id == 1) {
        // if(thread_local_id == 0 && block_global_id == 2) {
            printf("Block group ID 1 is active for global block %d, thread %d (%d, %d) || ", block_global_id, thread_local_id, m_offset, n_offset);
        }

        static_for<0, number_operations, 1>{}([&](auto i) { // TODO: remove the -1 to process all operations
            // Compute the starting offset for this thread/block/cluster
            // index_t m_offset = block_group_id * S::Block_M + thread_m_cluster_id * MThreadSliceSize;
            // index_t n_offset = block_local_id * reduceSizePerBlock + thread_k_cluster_id * KThreadSliceSize;

            // index_t m_offset = block_group_id * S::Block_M + thread_m_cluster_id * Problem::BlockShape::ThreadTile_M;
            // index_t n_offset = block_local_id * reduce_size_per_block + thread_k_cluster_id * Problem::BlockShape::ThreadTile_N;

            auto x_window = make_tile_window(
                transformed_x_tensor,
                make_tuple(number<S::Block_M>{}, number<S::Block_N>{}),
                // {iM, 0},
                {m_offset, n_offset},
                Policy::template MakeXBlockTileDistribution<Problem>()); // Input tile windows and
                                                                         // prep the block

            using XTensorType = decltype(load_tile(x_window)); // Load input tile

            auto y_compute = block_reduce2d.template MakeYBlockTile<XTensorType>();

            set_tile(y_compute,
                     reduce_funcs.get(number<i>{}).template GetIdentityValue<ComputeDataType>());

            for(int iN = __builtin_amdgcn_readfirstlane(0); iN < num_n_tile_iteration; ++iN)
            {
                auto x = load_tile(x_window);

                if constexpr (elementwise_ops.size() > 0) {
                    // Apply the elementwise operation before the reduction
                    tile_elementwise_inout(elementwise_ops.get(number<i>{}), x);
                }

                block_reduce2d(x, y_compute, reduce_funcs.get(number<i>{}));

                move_tile_window(x_window, {0, S::Block_N});
            }

            block_reduce2d_sync(y_compute, reduce_funcs.get(number<i>{}));
            block_reduce2d_cross_warp_sync(
                y_compute, static_cast<void*>(smem), reduce_funcs.get(number<i>{}));

            // Store the result back in each element of the tuple
            // store_tile(y_tile_windows.get(number<i>{}), cast_tile<YDataType>(y_compute));

            if( thread_n_cluster_id == 0) {
                // 1. Get the pointer to the output buffer for this tile window
                auto* p_y_tile = p_y_tuple + (i * output_tensor_offset) + S::Block_M * block_group_id + y_compute.get_thread_buffer_size() * thread_m_cluster_id; // TODO shall we include num_n_iteration???
                // auto* p_y_tile = y_tile_windows.get(number<i>{}).bottom_tensor_view_.buf_.p_data_;

                // if (i == 0) {
                //     printf("%d,", S::Block_M * block_group_id + y_compute.get_thread_buffer_size() * thread_m_cluster_id);
                //     // printf("Block Group ID: %d, Block Local ID: %d, Thread M Cluster ID: %d, Thread N Cluster Id = %d, buffer size = %d, offset=%d\n", block_group_id, block_local_id, thread_m_cluster_id, thread_n_cluster_id, y_compute.get_thread_buffer_size()
                //         // , S::Block_M * block_group_id + y_compute.get_thread_buffer_size() * thread_m_cluster_id);
                // }
                // 2. Cast y_compute to thread_buffer<YDataType, N> (N = tile size)
                auto y_thread_buf = cast_tile<YDataType>(y_compute).get_thread_buffer();

                // 3. Atomically add the register tile to DRAM
                // static_assert(std::is_same_v<decltype(y_thread_buf), int>);
                // static_assert(std::is_same_v<decltype(p_y_tile), int>);
                // printf("%f ", y_compute.get_thread_buffer().at(0));

                // TODO: revisit this after we implemented the rest
                // static_for<0, S::ThreadTile_M, 1>{}([&](auto j) { // y_compute.get_thread_buffer_size()
                //    *(p_y_tile+j) = y_thread_buf.at(j); //+ static_cast<YDataType>(j); //static_cast<YDataType>(block_group_id * S::Block_M+j); // static_cast<YDataType>(thread_m_cluster_id+1); //
                // });
                
                // Only atomic add is supported for now as the atomic max operation is neither supporting fp16 nor buffer greated than 1 element
                auto atomic_ops = reduce_funcs.get(number<i>{}).template GetAtomic<YDataType, y_thread_buf.N>(); // TODO: check if we need YDataType
                atomic_ops(p_y_tile, y_thread_buf);
            }

        });
    }

    /// @brief Validates if the given arguments are supported by the 2D multi reduction kernel.
    ///
    /// @param y_continous_dim Size of the continuous dimension of the output tensor.
    ///                        Must be a multiple of ThreadTile_N for proper thread mapping.
    ///
    /// @param input_strides   The stride configuration of the input tensor.
    ///                        The last stride must be 1 to ensure contiguous memory access
    ///                        and enable efficient vectorized loads.
    ///
    /// @return true if the arguments are supported, false otherwise.
    ///         Error messages are logged when CK_TILE_LOGGING is enabled.
    ///
    /// @note Requirements:
    ///       - y_continous_dim % ThreadTile_N == 0 (for proper thread distribution)
    ///       - input_strides[-1] == 1 (for contiguous memory access)
    template <typename InputStrides>
    CK_TILE_HOST static bool IsSupportedArgument(index_t y_continous_dim,
                                                 InputStrides input_strides)
    {
        using S = typename Problem::BlockShape;

        if(y_continous_dim % S::ThreadTile_N != 0)
        {
            if(ck_tile::EnvIsEnabled(CK_TILE_ENV(CK_TILE_LOGGING)))
            {
                CK_TILE_ERROR("Total reduction size should be a multiple of ThreadTile_N!");
            }
            return false;
        }

        if(input_strides.at(number<input_strides.size() - 1>{}) != 1)
        {
            if(ck_tile::EnvIsEnabled(CK_TILE_ENV(CK_TILE_LOGGING)))
            {
                CK_TILE_ERROR(
                    "Input tensor's last stride must be 1 to support correct vector access!");
            }
            return false;
        }

        return true;
    }
};

} // namespace ck_tile
