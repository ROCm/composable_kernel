# CMakeLists.txt for ConvBuilder generated instantiations

# Use GLOB to collect all generated instance files
file(GLOB CONV_INSTANCE_SOURCES "conv_instances_batch_*.cpp")

# Check if any files were found
if(NOT CONV_INSTANCE_SOURCES)
    message(WARNING "No conv_instances_batch_*.cpp files found in ${CMAKE_CURRENT_SOURCE_DIR}")
endif()

list(APPEND CONV_INSTANCE_SOURCES "instance_registry.cpp")

# Add all generated instance files as a library
add_library(ckb_instances SHARED ${CONV_INSTANCE_SOURCES})

target_include_directories(ckb_instances PUBLIC
  ${PROJECT_SOURCE_DIR}/include
  ${PROJECT_SOURCE_DIR}/experimental/builder/include
  ${CMAKE_CURRENT_SOURCE_DIR}/../test/impl
)

target_compile_features(ckb_instances PUBLIC cxx_std_20)

# Add necessary compile options for the builder
target_compile_options(ckb_instances PRIVATE
    -Wno-global-constructors
    -Wno-c++20-compat
    -ftemplate-backtrace-limit=0
)
