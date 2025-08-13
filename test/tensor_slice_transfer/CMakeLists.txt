set(test_target test_tensor_slice_transfer)

add_executable(${test_target} test_tensor_slice_transfer.cpp)

target_include_directories(${test_target} PRIVATE ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(${test_target} gtest gtest_main hip::device)

target_compile_definitions(${test_target} PRIVATE
    CK_EXPERIMENTAL_BIT_INT_EXTENSION_INT4
)

# Disable warnings that conflict with GTest
target_compile_options(${test_target} PRIVATE
    -Wno-global-constructors
    -Wno-unused-result
    -Wno-shadow
)

set_target_properties(${test_target} PROPERTIES
    CXX_STANDARD 17
    HIP_SEPARABLE_COMPILATION ON
)

# Set GPU architecture for compilation
if(GPU_TARGETS)
    set_property(TARGET ${test_target} PROPERTY HIP_ARCHITECTURES ${GPU_TARGETS})
else()
    set_property(TARGET ${test_target} PROPERTY HIP_ARCHITECTURES "gfx950")
endif()

# Add test to CTest
add_test(NAME ${test_target} COMMAND $<TARGET_FILE:${test_target}>)
