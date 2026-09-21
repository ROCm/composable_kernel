# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

# ckProfiler
set(CK_PROFILER_OP_FILTER "" CACHE STRING "Filter for the operators to be profiled. Default is to include all")
set(CK_PROFILER_INSTANCE_FILTER "" CACHE STRING "Filter for the kernels instances to be profiled. Default is to be the same as the operator filter")
if (CK_PROFILER_OP_FILTER STREQUAL "")
  set(CK_PROFILER_OP_FILTER ".+")
endif()
if (CK_PROFILER_INSTANCE_FILTER STREQUAL "")
  set(CK_PROFILER_INSTANCE_FILTER ${CK_PROFILER_OP_FILTER})
endif()
message(STATUS "CK_PROFILER_OP_FILTER: ${CK_PROFILER_OP_FILTER}")
message(STATUS "CK_PROFILER_INSTANCE_FILTER: ${CK_PROFILER_INSTANCE_FILTER}")

set(PROFILER_OPS
    profile_gemm.cpp
    profile_reduce.cpp
    profile_groupnorm_bwd_data.cpp
    profile_groupnorm_fwd.cpp
    profile_layernorm_bwd_data.cpp
    profile_layernorm_bwd_gamma_beta.cpp
    profile_groupnorm_bwd_gamma_beta.cpp
    profile_layernorm_fwd.cpp
    profile_max_pool2d_fwd.cpp
    profile_pool3d_fwd.cpp
    profile_avg_pool3d_bwd.cpp
    profile_max_pool3d_bwd.cpp
    profile_avg_pool2d_bwd.cpp
    profile_max_pool2d_bwd.cpp
    profile_softmax.cpp
    profile_batchnorm_fwd.cpp
    profile_batchnorm_bwd.cpp
    profile_batchnorm_infer.cpp
    profile_conv_tensor_rearrange.cpp
    profile_transpose.cpp
    profile_permute_scale.cpp
    profile_gemm_quantization.cpp
)

if(SUPPORTED_GPU_TARGETS MATCHES "gfx9|gfx1250")
  if(DTYPES MATCHES "fp32" OR DTYPES MATCHES "fp64" OR NOT DEFINED DTYPES)
    list(APPEND PROFILER_OPS profile_contraction_bilinear.cpp)
    list(APPEND PROFILER_OPS profile_contraction_scale.cpp)
  endif()
  if(CK_EXPERIMENTAL_BUILDER OR CK_TILE_DISPATCHER)
    list(APPEND PROFILER_OPS profile_grouped_conv_fwd_tile.cpp)
    list(APPEND PROFILER_OPS profile_grouped_conv_bwd_weight_tile.cpp)
    list(APPEND PROFILER_OPS profile_grouped_conv_bwd_data_tile.cpp)
  endif()
endif()

if(SUPPORTED_GPU_TARGETS MATCHES "gfx9|gfx1[12]")  
  if(DTYPES MATCHES "fp16" OR NOT DEFINED DTYPES)
    list(APPEND PROFILER_OPS profile_gemm_reduce.cpp)
    list(APPEND PROFILER_OPS profile_batched_gemm_add_relu_gemm_add.cpp)
    list(APPEND PROFILER_OPS profile_gemm_add.cpp)
    list(APPEND PROFILER_OPS profile_grouped_gemm.cpp)
    list(APPEND PROFILER_OPS profile_gemm_add_relu.cpp)
    list(APPEND PROFILER_OPS profile_gemm_add_relu_add_layernorm.cpp)
    list(APPEND PROFILER_OPS profile_grouped_gemm_fixed_nk.cpp)
    list(APPEND PROFILER_OPS profile_grouped_gemm_fastgelu.cpp)
    list(APPEND PROFILER_OPS profile_grouped_gemm_tile_loop.cpp)
    list(APPEND PROFILER_OPS profile_grouped_gemm_multiply_tile_loop.cpp)
  endif()
  if(SUPPORTED_GPU_TARGETS MATCHES "gfx9[45]|gfx12")
    list(APPEND PROFILER_OPS profile_gemm_multiply_multiply_wp.cpp)
    list(APPEND PROFILER_OPS profile_gemm_ab_scale.cpp)
    list(APPEND PROFILER_OPS profile_gemm_blockscale_wp.cpp)
    list(APPEND PROFILER_OPS profile_gemm_universal_preshuffle.cpp)
  endif()
  if(SUPPORTED_GPU_TARGETS MATCHES "gfx95")
    list(APPEND PROFILER_OPS profile_gemm_mx.cpp)
  endif()
  list(APPEND PROFILER_OPS profile_batched_gemm_reduce.cpp)
  list(APPEND PROFILER_OPS profile_gemm_add_multiply.cpp)
  list(APPEND PROFILER_OPS profile_gemm_add.cpp)
  list(APPEND PROFILER_OPS profile_gemm_bias_add_reduce.cpp)
  list(APPEND PROFILER_OPS profile_gemm_splitk.cpp)
  list(APPEND PROFILER_OPS profile_gemm_universal_batched.cpp)
  list(APPEND PROFILER_OPS profile_conv_fwd_bias_relu.cpp)
  list(APPEND PROFILER_OPS profile_conv_fwd_bias_relu_add.cpp)
  list(APPEND PROFILER_OPS profile_conv_bwd_data.cpp)
  list(APPEND PROFILER_OPS profile_conv_fwd.cpp)
endif()

if((SUPPORTED_GPU_TARGETS MATCHES "gfx9" AND (DTYPES MATCHES "fp16" OR NOT DEFINED DTYPES)) OR
   (SUPPORTED_GPU_TARGETS MATCHES "gfx1[12]"))
  list(APPEND PROFILER_OPS profile_gemm_bilinear.cpp)
endif()
if(SUPPORTED_GPU_TARGETS MATCHES "gfx(9[45]|1[12])")
  list(APPEND PROFILER_OPS profile_gemm_multiply_multiply.cpp)
endif()

if(SUPPORTED_GPU_TARGETS MATCHES "gfx9|gfx1[12]")
  list(APPEND PROFILER_OPS profile_gemm_universal.cpp)
  list(APPEND PROFILER_OPS profile_batched_gemm.cpp)
  list(APPEND PROFILER_OPS profile_batched_gemm_b_scale.cpp)
  list(APPEND PROFILER_OPS profile_gemm_b_scale.cpp)
  list(APPEND PROFILER_OPS profile_gemm_universal_reduce.cpp)
  list(APPEND PROFILER_OPS profile_grouped_conv_fwd.cpp)
  list(APPEND PROFILER_OPS profile_grouped_conv_fwd_bias_clamp.cpp)
  list(APPEND PROFILER_OPS profile_grouped_conv_fwd_bias_bnorm_clamp.cpp)
  list(APPEND PROFILER_OPS profile_grouped_conv_fwd_clamp.cpp)
  list(APPEND PROFILER_OPS profile_grouped_conv_fwd_convscale_add.cpp)
  list(APPEND PROFILER_OPS profile_grouped_conv_bwd_data.cpp)
  list(APPEND PROFILER_OPS profile_grouped_conv_fwd_dynamic_op.cpp)
  list(APPEND PROFILER_OPS profile_grouped_conv_fwd_bilinear.cpp)
  list(APPEND PROFILER_OPS profile_grouped_conv_bwd_weight.cpp)
  list(APPEND PROFILER_OPS profile_grouped_conv_fwd_outelementop.cpp)
  list(APPEND PROFILER_OPS profile_gemm_multi_abd.cpp)
  list(APPEND PROFILER_OPS profile_grouped_conv_fwd_scaleadd_scaleadd_relu.cpp)
  if(DTYPES MATCHES "fp16" OR NOT DEFINED DTYPES)
    list(APPEND PROFILER_OPS profile_gemm_add_multiply.cpp)
    list(APPEND PROFILER_OPS profile_gemm_multiply_add.cpp)
    list(APPEND PROFILER_OPS profile_gemm_add_silu.cpp)
    list(APPEND PROFILER_OPS profile_gemm_fastgelu.cpp)
    list(APPEND PROFILER_OPS profile_gemm_add_fastgelu.cpp)
    list(APPEND PROFILER_OPS profile_gemm_add_add_fastgelu.cpp)
    list(APPEND PROFILER_SOURCES profile_gemm_add.cpp)
  endif()
  list(APPEND PROFILER_OPS profile_batched_gemm_gemm.cpp)
endif()

if(DL_KERNELS)
  list(APPEND PROFILER_OPS profile_batched_gemm_multi_d.cpp)
  list(APPEND PROFILER_OPS profile_grouped_conv_bwd_weight.cpp)
endif()

if(CK_ENABLE_INT8)
  list(APPEND PROFILER_OPS profile_gemm_quantization.cpp)
endif()

set(PROFILER_SOURCES profiler.cpp)
foreach(SOURCE ${PROFILER_OPS})
  string(REGEX REPLACE "profile_(.+)\.cpp" "\\1" OP_NAME ${SOURCE})
  if (OP_NAME STREQUAL "") 
    message(FATAL_ERROR "Unexpected source file name: ${SOURCE}")
  endif()
  if("${OP_NAME}" MATCHES "${CK_PROFILER_OP_FILTER}")
    list(APPEND PROFILER_SOURCES ${SOURCE})
  endif()
endforeach()
message(VERBOSE "ckProfiler sources: ${PROFILER_SOURCES}")

set(PROFILER_EXECUTABLE ckProfiler)

add_executable(${PROFILER_EXECUTABLE} ${PROFILER_SOURCES})
target_compile_options(${PROFILER_EXECUTABLE} PRIVATE -Wno-global-constructors)
# flags to compress the library
if(NOT WIN32 AND ${hip_VERSION_FLAT} GREATER 600241132)
  message(DEBUG "Adding --offload-compress flag for ${PROFILER_EXECUTABLE}")
  target_compile_options(${PROFILER_EXECUTABLE} PRIVATE --offload-compress)
endif()


# =============================================================================
# CK Tile Dispatcher: Generate and compile CK Tile conv kernels
# =============================================================================
if(CK_TILE_DISPATCHER AND SUPPORTED_GPU_TARGETS MATCHES "gfx9|gfx1250")
  find_package(Python3 COMPONENTS Interpreter REQUIRED)

  # Build dispatcher instance libraries (defines ck_dispatcher_grouped_conv_* targets)
  if(NOT TARGET ck_dispatcher_grouped_conv_fwd)
    add_subdirectory(
      ${PROJECT_SOURCE_DIR}/dispatcher/library
      ${CMAKE_CURRENT_BINARY_DIR}/dispatcher_library
    )
  endif()

  # Profiler needs dispatcher API headers and builder test infrastructure
  target_include_directories(${PROFILER_EXECUTABLE} PRIVATE
    ${PROJECT_SOURCE_DIR}/dispatcher/include
    ${PROJECT_SOURCE_DIR}/experimental/builder/include
  )
  target_compile_options(${PROFILER_EXECUTABLE} PRIVATE
    -Wno-undefined-func-template
    -Wno-float-equal
    -Wno-ctad-maybe-unsupported
    -Wno-old-style-cast
    # New clang -Weverything suggestion firing on builder-pattern setters that
    # return *this in the shared dispatcher headers; not a real defect.
    -Wno-lifetime-safety-intra-tu-suggestions
  )

  # Link all available dispatcher instance libraries
  foreach(_VARIANT fwd bwd_weight bwd_data)
    set(_TARGET "ck_dispatcher_grouped_conv_${_VARIANT}")
    if(TARGET ${_TARGET})
      target_link_libraries(${PROFILER_EXECUTABLE} PRIVATE ${_TARGET})
    endif()
  endforeach()
endif()

set(DEVICE_INSTANCES "")
list(APPEND DEVICE_INSTANCES device_gemm_instance)
list(APPEND DEVICE_INSTANCES device_normalization_fwd_instance)
list(APPEND DEVICE_INSTANCES device_normalization_bwd_data_instance)
list(APPEND DEVICE_INSTANCES device_normalization_bwd_gamma_beta_instance)
list(APPEND DEVICE_INSTANCES device_softmax_instance)
list(APPEND DEVICE_INSTANCES device_reduce_instance)
list(APPEND DEVICE_INSTANCES device_batchnorm_instance)
list(APPEND DEVICE_INSTANCES device_pool2d_fwd_instance)
list(APPEND DEVICE_INSTANCES device_pool3d_fwd_instance)
list(APPEND DEVICE_INSTANCES device_avg_pool2d_bwd_instance)
list(APPEND DEVICE_INSTANCES device_avg_pool3d_bwd_instance)
list(APPEND DEVICE_INSTANCES device_max_pool_bwd_instance)
list(APPEND DEVICE_INSTANCES device_image_to_column_instance)
list(APPEND DEVICE_INSTANCES device_column_to_image_instance)
list(APPEND DEVICE_INSTANCES device_transpose_instance)
list(APPEND DEVICE_INSTANCES device_permute_scale_instance)

if(SUPPORTED_GPU_TARGETS MATCHES "gfx9|gfx1[12]")
  if(DTYPES MATCHES "fp32" OR DTYPES MATCHES "fp64" OR NOT DEFINED DTYPES)
    list(APPEND DEVICE_INSTANCES device_contraction_bilinear_instance)
    list(APPEND DEVICE_INSTANCES device_contraction_scale_instance)
  endif()
  if(DTYPES MATCHES "fp16" OR NOT DEFINED DTYPES)
    list(APPEND DEVICE_INSTANCES device_gemm_add_instance)
    list(APPEND DEVICE_INSTANCES device_batched_gemm_gemm_instance)
    list(APPEND DEVICE_INSTANCES device_gemm_add_add_fastgelu_instance)
    list(APPEND DEVICE_INSTANCES device_gemm_fastgelu_instance)
    list(APPEND DEVICE_INSTANCES device_batched_gemm_add_relu_gemm_add_instance)
    list(APPEND DEVICE_INSTANCES device_grouped_gemm_instance)
    list(APPEND DEVICE_INSTANCES device_gemm_add_relu_instance)
    list(APPEND DEVICE_INSTANCES device_gemm_add_relu_add_layernorm_instance)
    list(APPEND DEVICE_INSTANCES device_grouped_gemm_fixed_nk_instance)
    list(APPEND DEVICE_INSTANCES device_grouped_gemm_fastgelu_instance)
    list(APPEND DEVICE_INSTANCES device_grouped_gemm_tile_loop_instance)
  endif()
  list(APPEND DEVICE_INSTANCES device_batched_gemm_reduce_instance)
  if(SUPPORTED_GPU_TARGETS MATCHES "gfx9[45]|gfx12")
    list(APPEND DEVICE_INSTANCES device_gemm_multiply_multiply_wp_instance)
    list(APPEND DEVICE_INSTANCES device_gemm_universal_preshuffle_instance)
  endif()
    if(SUPPORTED_GPU_TARGETS MATCHES "gfx9[45]|gfx1[12]")
    list(APPEND DEVICE_INSTANCES device_gemm_ab_scale_instance)
    list(APPEND DEVICE_INSTANCES device_gemm_blockscale_wp_instance)
  endif()
  if(SUPPORTED_GPU_TARGETS MATCHES "gfx95")
    list(APPEND DEVICE_INSTANCES device_gemm_mx_instance)
  endif()
  list(APPEND DEVICE_INSTANCES device_gemm_splitk_instance)
  list(APPEND DEVICE_INSTANCES device_gemm_universal_batched_instance)
  list(APPEND DEVICE_INSTANCES device_gemm_add_multiply_instance)
  list(APPEND DEVICE_INSTANCES device_gemm_add_instance)
  list(APPEND DEVICE_INSTANCES device_gemm_reduce_instance)
  list(APPEND DEVICE_INSTANCES device_gemm_bias_add_reduce_instance)
  list(APPEND DEVICE_INSTANCES device_conv2d_fwd_instance)
  list(APPEND DEVICE_INSTANCES device_conv2d_fwd_bias_relu_instance)
  list(APPEND DEVICE_INSTANCES device_conv2d_fwd_bias_relu_add_instance)
  list(APPEND DEVICE_INSTANCES device_conv1d_bwd_data_instance)
  list(APPEND DEVICE_INSTANCES device_conv3d_bwd_data_instance)
  list(APPEND DEVICE_INSTANCES device_conv2d_bwd_data_instance)
endif()

if((SUPPORTED_GPU_TARGETS MATCHES "gfx9" AND (DTYPES MATCHES "fp16" OR NOT DEFINED DTYPES)) OR
   (SUPPORTED_GPU_TARGETS MATCHES "gfx1[12]" ))
  list(APPEND DEVICE_INSTANCES device_gemm_bilinear_instance)
endif()
if(SUPPORTED_GPU_TARGETS MATCHES "gfx(9[45]|1[12])")
  list(APPEND DEVICE_INSTANCES device_gemm_multiply_multiply_instance)
endif()

if(SUPPORTED_GPU_TARGETS MATCHES "gfx9|gfx1[12]")
  list(APPEND DEVICE_INSTANCES device_gemm_universal_instance)
  list(APPEND DEVICE_INSTANCES device_batched_gemm_instance)
  list(APPEND DEVICE_INSTANCES device_gemm_b_scale_instance)
  list(APPEND DEVICE_INSTANCES device_gemm_universal_reduce_instance)
  list(APPEND DEVICE_INSTANCES device_batched_gemm_b_scale_instance)
  # 1D convolutions
  list(APPEND DEVICE_INSTANCES device_grouped_conv1d_fwd_gnwc_instance)
  # 2D convolutions - core operations
  list(APPEND DEVICE_INSTANCES device_grouped_conv2d_fwd_nhwgc_instance)
  list(APPEND DEVICE_INSTANCES device_grouped_conv2d_fwd_gnhwc_instance)
  list(APPEND DEVICE_INSTANCES device_grouped_conv2d_fwd_ngchw_instance)
  list(APPEND DEVICE_INSTANCES device_grouped_conv2d_bwd_data_nhwgc_instance)
  list(APPEND DEVICE_INSTANCES device_grouped_conv2d_bwd_data_gnhwc_instance)
  list(APPEND DEVICE_INSTANCES device_grouped_conv2d_bwd_data_ngchw_instance)
  # 2D convolutions - variants
  list(APPEND DEVICE_INSTANCES device_grouped_conv2d_fwd_clamp_nhwgc_instance)
  list(APPEND DEVICE_INSTANCES device_grouped_conv2d_fwd_bias_clamp_nhwgc_instance)
  list(APPEND DEVICE_INSTANCES device_grouped_conv2d_fwd_bias_bnorm_clamp_nhwgc_instance)
  list(APPEND DEVICE_INSTANCES device_grouped_conv2d_fwd_dynamic_op_nhwgc_instance)
  # 3D convolutions - core operations
  list(APPEND DEVICE_INSTANCES device_grouped_conv3d_fwd_ndhwgc_instance)
  list(APPEND DEVICE_INSTANCES device_grouped_conv3d_fwd_gndhwc_instance)
  list(APPEND DEVICE_INSTANCES device_grouped_conv3d_fwd_ngcdhw_instance)
  list(APPEND DEVICE_INSTANCES device_grouped_conv3d_bwd_data_ndhwgc_instance)
  list(APPEND DEVICE_INSTANCES device_grouped_conv3d_bwd_data_gndhwc_instance)
  list(APPEND DEVICE_INSTANCES device_grouped_conv3d_bwd_data_ngcdhw_instance)
  # 3D convolutions - variants
  list(APPEND DEVICE_INSTANCES device_grouped_conv3d_fwd_scale_ndhwgc_instance)
  list(APPEND DEVICE_INSTANCES device_grouped_conv3d_fwd_clamp_ndhwgc_instance)
  list(APPEND DEVICE_INSTANCES device_grouped_conv3d_fwd_bias_clamp_ndhwgc_instance)
  list(APPEND DEVICE_INSTANCES device_grouped_conv3d_fwd_bias_bnorm_clamp_ndhwgc_instance)
  list(APPEND DEVICE_INSTANCES device_grouped_conv3d_fwd_bias_bnorm_clamp_nhwgc_instance)
  list(APPEND DEVICE_INSTANCES device_grouped_conv3d_fwd_bilinear_ndhwgc_instance)
  list(APPEND DEVICE_INSTANCES device_grouped_conv3d_fwd_convscale_ndhwgc_instance)
  list(APPEND DEVICE_INSTANCES device_grouped_conv3d_fwd_convscale_add_ndhwgc_instance)
  list(APPEND DEVICE_INSTANCES device_grouped_conv3d_fwd_convscale_relu_ndhwgc_instance)
  list(APPEND DEVICE_INSTANCES device_grouped_conv3d_fwd_convinvscale_ndhwgc_instance)
  list(APPEND DEVICE_INSTANCES device_grouped_conv3d_fwd_dynamic_op_ndhwgc_instance)
  list(APPEND DEVICE_INSTANCES device_grouped_conv3d_fwd_scaleadd_scaleadd_relu_ndhwgc_instance)
  list(APPEND DEVICE_INSTANCES device_gemm_add_relu_instance)
  list(APPEND DEVICE_INSTANCES device_gemm_multi_abd_instance)
  if(DTYPES MATCHES "fp16" OR NOT DEFINED DTYPES)
    list(APPEND DEVICE_INSTANCES device_gemm_add_multiply_instance)
    list(APPEND DEVICE_INSTANCES device_gemm_multiply_add_instance)
    list(APPEND DEVICE_INSTANCES device_gemm_add_instance)
    list(APPEND DEVICE_INSTANCES device_gemm_add_silu_instance)
    list(APPEND DEVICE_INSTANCES device_gemm_fastgelu_instance)
    list(APPEND DEVICE_INSTANCES device_gemm_add_fastgelu_instance)
    list(APPEND DEVICE_INSTANCES device_gemm_add_add_fastgelu_instance)
  endif()
  list(APPEND DEVICE_INSTANCES device_batched_gemm_gemm_instance)
  # Backward weight convolutions
  list(APPEND DEVICE_INSTANCES device_grouped_conv1d_bwd_weight_gnwc_instance)
  list(APPEND DEVICE_INSTANCES device_grouped_conv1d_bwd_weight_nwgc_instance)
  list(APPEND DEVICE_INSTANCES device_grouped_conv2d_bwd_weight_nhwgc_instance)
  list(APPEND DEVICE_INSTANCES device_grouped_conv2d_bwd_weight_gnhwc_instance)
  list(APPEND DEVICE_INSTANCES device_grouped_conv2d_bwd_weight_ngchw_instance)
  list(APPEND DEVICE_INSTANCES device_grouped_convnd_bwd_weight_instance)
  list(APPEND DEVICE_INSTANCES device_grouped_conv3d_bwd_weight_ndhwgc_instance)
  list(APPEND DEVICE_INSTANCES device_grouped_conv3d_bwd_weight_gndhwc_instance)
  list(APPEND DEVICE_INSTANCES device_grouped_conv3d_bwd_weight_ngcdhw_instance)
endif()

if(DL_KERNELS)
  list(APPEND DEVICE_INSTANCES device_batched_gemm_multi_d_instance)
  list(APPEND DEVICE_INSTANCES device_grouped_conv1d_bwd_weight_gnwc_instance)
  list(APPEND DEVICE_INSTANCES device_grouped_conv1d_bwd_weight_nwgc_instance)
  list(APPEND DEVICE_INSTANCES device_grouped_conv2d_bwd_weight_nhwgc_instance)
  list(APPEND DEVICE_INSTANCES device_grouped_conv2d_bwd_weight_gnhwc_instance)
  list(APPEND DEVICE_INSTANCES device_grouped_conv2d_bwd_weight_ngchw_instance)
  list(APPEND DEVICE_INSTANCES device_grouped_conv3d_bwd_weight_ndhwgc_instance)
  list(APPEND DEVICE_INSTANCES device_grouped_conv3d_bwd_weight_gndhwc_instance)
  list(APPEND DEVICE_INSTANCES device_grouped_conv3d_bwd_weight_ngcdhw_instance)
endif()

if(CK_ENABLE_INT8)
  list(APPEND DEVICE_INSTANCES device_quantization_instance)
endif()

set(PROFILER_LIBS utility getopt::getopt)
foreach(LIB ${DEVICE_INSTANCES})
  string(REGEX REPLACE "device_(.+)_instance" "\\1" INSTANCE_NAME ${LIB})
  if (INSTANCE_NAME STREQUAL "")
    message(FATAL_ERROR "Unexpected kernel instance name: ${LIB}")
  endif()
  if("${INSTANCE_NAME}" MATCHES "${CK_PROFILER_INSTANCE_FILTER}")
    # Only link if the target was actually created
    if(TARGET ${LIB})
      list(APPEND PROFILER_LIBS ${LIB})
    else()
      message(VERBOSE "Skipping ${LIB} - no instances built for current GPU targets")
    endif()
  endif()
endforeach()
message(VERBOSE "ckProfiler libs: ${PROFILER_LIBS}")
target_link_libraries(${PROFILER_EXECUTABLE} PRIVATE ${PROFILER_LIBS})

rocm_install(TARGETS ${PROFILER_EXECUTABLE} COMPONENT profiler)
