set(CMAKE_CXX_STANDARD 20)

add_executable(gemm_example gemm_example.cpp)
target_include_directories(gemm_example PRIVATE
    "${PROJECT_SOURCE_DIR}/include"
)

include(gtest)

add_executable(test_conv_builder builder.cpp)
target_include_directories(test_conv_builder PRIVATE
    "${PROJECT_SOURCE_DIR}/experimental/builder/include"
    "${PROJECT_SOURCE_DIR}/include"
)
target_compile_options(test_conv_builder PRIVATE -Wno-global-constructors -Wno-c++20-compat)
target_link_libraries(test_conv_builder PRIVATE GTest::gtest GTest::gtest_main)

