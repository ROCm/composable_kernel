
include_directories(AFTER
  ${PROJECT_SOURCE_DIR}/include
  ${PROJECT_SOURCE_DIR}/library/include
  ${PROJECT_SOURCE_DIR}/experimental/builder/include
)

add_subdirectory(01_conv_2d_fwd)
