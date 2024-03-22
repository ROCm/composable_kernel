#include <type_traits>

#include "ck/utility/common_header.hpp"
#include "ck/utility/math.hpp"

#include "ck/library/utility/host_tensor.hpp"
// #include "ck/library/utility/host_tensor_generator.hpp"

// Block tile shape: Static
// Embedding dimension
// Sequence dimensions: Static OR Dynamic

template <ck::index_t _TileM, ck::index_t _TileN>
struct SddmmTileMask {
    static constexpr ck::index_t TileM = _TileM;
    static constexpr ck::index_t TileN = _TileN;

    const ck::index_t M;
    const ck::index_t N;

    const ck::index_t TileMaskM;
    const ck::index_t TileMaskN;

    Tensor<int> mask;

    template <typename generator_func_t>
    SddmmTileMask(ck::index_t m, ck::index_t n, generator_func_t f)
        : M(m), N(n), TileMaskM(ck::math::integer_divide_ceil(M, TileM)),
          TileMaskN(ck::math::integer_divide_ceil(N, TileN)), mask({TileMaskM, TileMaskN})
    {
        auto foreach_wrapper = [&](Tensor<int>& mask_ref, std::vector<size_t>& idx) {
            mask_ref(idx) = f(idx[idx.size()-2], idx[idx.size()-1]);  // We are assuming the mask is 2 dimensions
        };

        mask.ForEach(foreach_wrapper);
    }

    std::tuple<Tensor<ck::index_t>, Tensor<ck::index_t>> to_csr() {
        // Get number of non-zeros in each row
        Tensor<ck::index_t> tile_row_count({TileMaskM});
        auto sum_func = [&](Tensor<ck::index_t>& tile_row_count_ref, std::vector<size_t>& idx) {
            ck::index_t sum = 0;
            for (ck::index_t j = 0; j < TileMaskN; j++) { sum += mask(idx[0], j); }
            tile_row_count_ref(idx) = sum;
        };
        tile_row_count.ForEach(sum_func);

        // Get total number of nonzeros
        ck::index_t nnz = 0;
        for (ck::index_t i = 0; i < TileMaskM; i++) { nnz += tile_row_count(i); }

        // Populate row offsets
        Tensor<ck::index_t> row_offsets({TileMaskM + 1});
        ck::index_t prefix_sum = 0;
        row_offsets(0) = 0;
        for (int i = 0; i < TileMaskM; i++) {
            prefix_sum += tile_row_count(i);
            row_offsets(i + 1) = prefix_sum;
        }

        // Allocate and populate column indices
        Tensor<ck::index_t> col_indices({nnz});
        int col_indices_idx = 0;
        for (auto mask_iter = mask.begin(); mask_iter != mask.end(); mask_iter++) {
            if (*mask_iter == 1) {
                auto data_idx = mask_iter - mask.begin();
                auto col_idx = data_idx % TileMaskN;
                col_indices(col_indices_idx) = col_idx;
                col_indices_idx++;
            }
        }

        assert(nnz == col_indices_idx);  // Verify that all 1s were found

        return {row_offsets, col_indices};
    }
};
