# Add kernel instance libraries for ck_tile grouped convolution operations
# These libraries contain factory functions that create instances of specific kernel configurations
# Moving these large template instantiations to separate .cpp files enables parallel compilation


add_library(ck_tile_grouped_conv_bwd_data_bf16_instances STATIC
    tile_grouped_conv_bwd_data_bf16_instances.cpp
)

add_library(ck_tile_grouped_conv_bwd_data_bf16_instances_2 STATIC
    tile_grouped_conv_bwd_data_bf16_instances_2.cpp
)

add_library(ck_tile_grouped_conv_bwd_data_bf16_instances_3 STATIC
    tile_grouped_conv_bwd_data_bf16_instances_3.cpp
)

add_library(ck_tile_grouped_conv_bwd_data_bf16_instances_4 STATIC
    tile_grouped_conv_bwd_data_bf16_instances_4.cpp
)

add_library(ck_tile_grouped_conv_bwd_data_bf16_instances_5 STATIC
    tile_grouped_conv_bwd_data_bf16_instances_5.cpp
)

add_library(ck_tile_grouped_conv_bwd_data_bf16_instances_6 STATIC
    tile_grouped_conv_bwd_data_bf16_instances_6.cpp
)

add_library(ck_tile_grouped_conv_bwd_data_bf16_instances_7 STATIC
    tile_grouped_conv_bwd_data_bf16_instances_7.cpp
)

add_library(ck_tile_grouped_conv_bwd_weight_fp16_instances STATIC
    tile_grouped_conv_bwd_weight_fp16_instances.cpp
)

add_library(ck_tile_grouped_conv_bwd_weight_bf16_instances STATIC
    tile_grouped_conv_bwd_weight_bf16_instances.cpp
)

add_library(ck_tile_grouped_conv_bwd_weight_bf16_instances_opt STATIC
    tile_grouped_conv_bwd_weight_bf16_instances_opt.cpp
)

add_library(ck_tile_grouped_conv_fwd_fp16_instances STATIC
    tile_grouped_conv_fwd_fp16_instances.cpp
)

add_library(ck_tile_grouped_conv_fwd_bf16_instances STATIC
    tile_grouped_conv_fwd_bf16_instances.cpp
)

add_library(ck_tile_grouped_conv_fwd_bf16_instances_2 STATIC
    tile_grouped_conv_fwd_bf16_instances_2.cpp
)

add_library(ck_tile_grouped_conv_fwd_bf16_instances_3 STATIC
    tile_grouped_conv_fwd_bf16_instances_3.cpp
)

add_library(ck_tile_grouped_conv_fwd_bf16_instances_4 STATIC
    tile_grouped_conv_fwd_bf16_instances_4.cpp
)

add_library(ck_tile_grouped_conv_fwd_bf16_instances_5 STATIC
    tile_grouped_conv_fwd_bf16_instances_5.cpp
)

add_library(ck_tile_grouped_conv_fwd_bf16_instances_6 STATIC
    tile_grouped_conv_fwd_bf16_instances_6.cpp
)

target_include_directories(ck_tile_grouped_conv_bwd_data_bf16_instances PRIVATE
    ${PROJECT_SOURCE_DIR}/include
    ${PROJECT_SOURCE_DIR}/library/include
)

target_include_directories(ck_tile_grouped_conv_bwd_data_bf16_instances_2 PRIVATE
    ${PROJECT_SOURCE_DIR}/include
    ${PROJECT_SOURCE_DIR}/library/include
)

target_include_directories(ck_tile_grouped_conv_bwd_data_bf16_instances_3 PRIVATE
    ${PROJECT_SOURCE_DIR}/include
    ${PROJECT_SOURCE_DIR}/library/include
)

target_include_directories(ck_tile_grouped_conv_bwd_data_bf16_instances_4 PRIVATE
    ${PROJECT_SOURCE_DIR}/include
    ${PROJECT_SOURCE_DIR}/library/include
)

target_include_directories(ck_tile_grouped_conv_bwd_data_bf16_instances_5 PRIVATE
    ${PROJECT_SOURCE_DIR}/include
    ${PROJECT_SOURCE_DIR}/library/include
)

target_include_directories(ck_tile_grouped_conv_bwd_data_bf16_instances_6 PRIVATE
    ${PROJECT_SOURCE_DIR}/include
    ${PROJECT_SOURCE_DIR}/library/include
)

target_include_directories(ck_tile_grouped_conv_bwd_data_bf16_instances_7 PRIVATE
    ${PROJECT_SOURCE_DIR}/include
    ${PROJECT_SOURCE_DIR}/library/include
)

target_include_directories(ck_tile_grouped_conv_bwd_weight_fp16_instances PRIVATE
    ${PROJECT_SOURCE_DIR}/include
    ${PROJECT_SOURCE_DIR}/library/include
)

target_include_directories(ck_tile_grouped_conv_bwd_weight_bf16_instances PRIVATE
    ${PROJECT_SOURCE_DIR}/include
    ${PROJECT_SOURCE_DIR}/library/include
)

target_include_directories(ck_tile_grouped_conv_bwd_weight_bf16_instances_opt PRIVATE
    ${PROJECT_SOURCE_DIR}/include
    ${PROJECT_SOURCE_DIR}/library/include
)

target_include_directories(ck_tile_grouped_conv_fwd_fp16_instances PRIVATE
    ${PROJECT_SOURCE_DIR}/include
    ${PROJECT_SOURCE_DIR}/library/include
)

target_include_directories(ck_tile_grouped_conv_fwd_bf16_instances PRIVATE
    ${PROJECT_SOURCE_DIR}/include
    ${PROJECT_SOURCE_DIR}/library/include
)

target_include_directories(ck_tile_grouped_conv_fwd_bf16_instances_2 PRIVATE
    ${PROJECT_SOURCE_DIR}/include
    ${PROJECT_SOURCE_DIR}/library/include
)

target_include_directories(ck_tile_grouped_conv_fwd_bf16_instances_3 PRIVATE
    ${PROJECT_SOURCE_DIR}/include
    ${PROJECT_SOURCE_DIR}/library/include
)

target_include_directories(ck_tile_grouped_conv_fwd_bf16_instances_4 PRIVATE
    ${PROJECT_SOURCE_DIR}/include
    ${PROJECT_SOURCE_DIR}/library/include
)

target_include_directories(ck_tile_grouped_conv_fwd_bf16_instances_5 PRIVATE
    ${PROJECT_SOURCE_DIR}/include
    ${PROJECT_SOURCE_DIR}/library/include
)

target_include_directories(ck_tile_grouped_conv_fwd_bf16_instances_6 PRIVATE
    ${PROJECT_SOURCE_DIR}/include
    ${PROJECT_SOURCE_DIR}/library/include
)

