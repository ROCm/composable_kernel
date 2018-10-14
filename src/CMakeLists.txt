include_directories(BEFORE include)

set(SOURCE 
    tensor.cpp;
)

add_library(convolution SHARED ${SOURCE})
set_target_properties(convolution PROPERTIES PREFIX "")

# boost.python
target_link_libraries(convolution boost_python3)

# cuda
target_link_libraries(convolution nvToolsExt)
target_compile_features(convolution PUBLIC)
set_target_properties(convolution PROPERTIES POSITION_INDEPENDENT_CODE ON)
set_target_properties(convolution PROPERTIES CUDA_SEPARABLE_COMPILATION OFF)

install(TARGETS convolution LIBRARY DESTINATION lib) 
