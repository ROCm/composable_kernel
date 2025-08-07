set(CMAKE_CXX_STANDARD 20)

add_executable(convolution_example convolution_example.cpp)

set(EXAMPLE_COMPILE_OPTIONS)
list(APPEND EXAMPLE_COMPILE_OPTIONS -g -fverbose-asm --save-temps -Wno-gnu-line-marker)
target_compile_options(convolution_example PRIVATE ${EXAMPLE_COMPILE_OPTIONS})