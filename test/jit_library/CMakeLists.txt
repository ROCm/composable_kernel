add_test_executable(test_jit_library jit_library.cpp)
add_dependencies(test_jit_library jit_library)
target_include_directories(test_jit_library PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../../library/src/jit_library/include>)
target_link_libraries(test_jit_library PRIVATE jit_library)
