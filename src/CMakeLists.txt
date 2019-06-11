configure_file("${PROJECT_SOURCE_DIR}/src/include/config.hpp.in" "${PROJECT_BINARY_DIR}/src/include/config.hpp")

set(TENSOR_SOURCE 
    tensor.cpp;
    device.cpp;
)

add_library(tensor SHARED ${TENSOR_SOURCE})
target_compile_features(tensor PUBLIC)
set_target_properties(tensor PROPERTIES POSITION_INDEPENDENT_CODE ON)


if(DEVICE_BACKEND STREQUAL "CUDA")
    target_link_libraries(tensor nvToolsExt cudart)
endif()

install(TARGETS tensor LIBRARY DESTINATION lib) 
