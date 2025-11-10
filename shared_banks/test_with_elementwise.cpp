#include <hip/hip_runtime.h>
#include <ck_tile/core.hpp>
#include <ck_tile/core/utility/debug.hpp>

using ck_tile::number;
using ck_tile::make_naive_tensor_descriptor;
using ck_tile::make_tile_window;
using ck_tile::make_tensor_view;
using ck_tile::make_tuple;
using ck_tile::address_space_enum;
using ck_tile::fp16_t;
using ck_tile::index_t;

using ck_tile::tile_distribution_encoding;
using ck_tile::sequence;
using ck_tile::tuple;
using ck_tile::make_static_distributed_tensor;
using ck_tile::make_static_tile_distribution;
using ck_tile::get_n_lds_banks;
using ck_tile::get_n_words_per_128b;
using ck_tile::make_xor_transform;
using ck_tile::make_pass_through_transform;
using ck_tile::make_unmerge_transform;

constexpr index_t kBlockSize = 256;

template<typename DataType, index_t MPerBlock, index_t KPerBlock>
__device__ constexpr auto make_lds_tensor_descriptor()
{
  constexpr auto DataTypeSize = sizeof(DataType);
  constexpr index_t KPack = 16 / sizeof(DataType);

  constexpr auto MLdsLayer =
      ck_tile::max(1UL, get_n_lds_banks() * get_n_words_per_128b() / KPerBlock / DataTypeSize);

  constexpr auto a_lds_block_desc_0 = make_naive_tensor_descriptor(
      make_tuple(number<KPerBlock / KPack * MLdsLayer>{},
                  number<MPerBlock / MLdsLayer>{},
                  number<KPack>{}),
      make_tuple(number<KPack>{}, number<KPerBlock * MLdsLayer>{}, number<1>{}),
      number<KPack>{},
      number<1>{});

  constexpr auto a_lds_block_desc_permuted = transform_tensor_descriptor(
      a_lds_block_desc_0,
      make_tuple(make_xor_transform(make_tuple(number<MPerBlock / MLdsLayer>{},
                                                number<KPerBlock / KPack * MLdsLayer>{})),
                  make_pass_through_transform(number<KPack>{})),
      make_tuple(sequence<1, 0>{}, sequence<2>{}),
      make_tuple(sequence<1, 0>{}, sequence<2>{}));

  constexpr auto a_lds_block_desc_xk0_mnldslayer_mn_xk1 = transform_tensor_descriptor(
      a_lds_block_desc_permuted,
      make_tuple(make_unmerge_transform(
                      make_tuple(number<MLdsLayer>{}, number<KPerBlock / KPack>{})),
                  make_pass_through_transform(number<MPerBlock / MLdsLayer>{}),
                  make_pass_through_transform(number<KPack>{})),
      make_tuple(sequence<0>{}, sequence<1>{}, sequence<2>{}),
      make_tuple(sequence<0, 2>{}, sequence<1>{}, sequence<3>{}));

  constexpr auto a_lds_block_desc = transform_tensor_descriptor(
      a_lds_block_desc_xk0_mnldslayer_mn_xk1,
      make_tuple(make_merge_transform_v3_division_mod(
                      make_tuple(number<MPerBlock / MLdsLayer>{}, number<MLdsLayer>{})),
                  make_merge_transform_v3_division_mod(
                      make_tuple(number<KPerBlock / KPack>{}, number<KPack>{}))),
      make_tuple(sequence<1, 0>{}, sequence<2, 3>{}),
      make_tuple(sequence<0>{}, sequence<1>{}));

  return a_lds_block_desc;
}

template<typename DType, index_t MPerBlock, index_t KPerBlock>
__device__
constexpr auto make_register_distribution()
{
  using ck_tile::tile_distribution_encoding_pattern_2d;
  using ck_tile::tile_distribution_pattern;
  constexpr auto VecLoadSize = 16 / sizeof(DType);
  using TileEncodingPattern =
                tile_distribution_encoding_pattern_2d<kBlockSize,
                                                      MPerBlock,
                                                      KPerBlock,
                                                      VecLoadSize,
                                                      tile_distribution_pattern::thread_raked>;
  return TileEncodingPattern::make_2d_static_tile_distribution();
}

template<typename DType, index_t MTile, index_t KTile>
__global__ void lds_write_simulator()
{
  __shared__ int buf[160000 / sizeof(int)];

  // lds setup
  auto lds_tensor_descriptor = make_lds_tensor_descriptor<DType, MTile, KTile>();
  auto lds_tensor_view = make_tensor_view<address_space_enum::lds>(reinterpret_cast<DType*>(buf), lds_tensor_descriptor);
  auto lds_write_window = make_tile_window(lds_tensor_view, make_tuple(number<MTile>{}, number<KTile>{}), {0, 0});

  // register setup
  auto reg_tile_dst = make_register_distribution<DType, MTile, KTile>();
  auto reg_tensor = make_static_distributed_tensor<DType>(reg_tile_dst);

  // writeout
  store_tile(lds_write_window, reg_tensor);

  ck_tile::block_sync_lds();
}

__device__
constexpr auto make_lds_read_distribution()
{
  return make_static_tile_distribution(tile_distribution_encoding<
     ck_tile::sequence<2>, 
     ck_tile::tuple<ck_tile::sequence<4, 2, 16>, ck_tile::sequence<4, 4, 4>>, 
     ck_tile::tuple<ck_tile::sequence<1, 0>, ck_tile::sequence<2, 1>>, 
     ck_tile::tuple<ck_tile::sequence<1, 0>, ck_tile::sequence<1, 2>>, 
     ck_tile::sequence<1, 2, 2>, 
     ck_tile::sequence<0, 0, 2>>{});
}

template<typename DType, index_t MTile, index_t KTile>
__global__ void lds_read_simulator()
{
  __shared__ int buf[160000 / sizeof(int)];

  auto lds_tensor_descriptor = make_lds_tensor_descriptor<DType, MTile, KTile>();
  auto lds_tensor_view = make_tensor_view<address_space_enum::lds>(reinterpret_cast<DType*>(buf), lds_tensor_descriptor);

  constexpr auto reg_tile_dst = make_lds_read_distribution();
  auto lds_read_window = make_tile_window(lds_tensor_view, make_tuple(number<MTile>{}, number<KTile>{}), {0, 0}, reg_tile_dst);

  [[maybe_unused]] auto reg_tile = load_tile(lds_read_window);
  ck_tile::block_sync_lds();
}

int main()
{
  constexpr auto kGrid = 1;
  constexpr auto kBlockM = 128;
  constexpr auto kBlockK = 64;
  lds_write_simulator<fp16_t, kBlockM, kBlockK><<<kGrid, kBlockSize>>>();
  lds_read_simulator<fp16_t, kBlockM, kBlockK><<<kGrid, kBlockSize>>>();
  return 0;
}
