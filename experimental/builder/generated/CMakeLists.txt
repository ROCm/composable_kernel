# CMakeLists.txt for ConvBuilder generated instantiations

# Use GLOB to collect all generated instance files
file(GLOB CONV_INSTANCE_SOURCES "conv_instances_batch_*.cpp")

# Add all generated instance files as a library
add_library(ckb_generated_instances STATIC ${CONV_INSTANCE_SOURCES})

target_include_directories(ckb_generated_instances PUBLIC
  ${CMAKE_SOURCE_DIR}/include
  ${CMAKE_SOURCE_DIR}/experimental/builder/include
  ${CMAKE_SOURCE_DIR}/experimental/builder/test/impl
)
