if(DEVICE_BACKEND STREQUAL "AMD")
    configure_file("${PROJECT_SOURCE_DIR}/include/composable_kernel/utility/config_amd.hpp.in" "${PROJECT_BINARY_DIR}/include/composable_kernel/utility/config.hpp")
elseif(DEVICE_BACKEND STREQUAL "NVIDIA")
    configure_file("${PROJECT_SOURCE_DIR}/include/composable_kernel/utility/config_nvidia.hpp.in" "${PROJECT_BINARY_DIR}/include/composable_kernel/utility/config.hpp")
endif()

set(TENSOR_SOURCE 
    tensor.cpp;
    device.cpp;
)

add_library(tensor SHARED ${TENSOR_SOURCE})
target_compile_features(tensor PUBLIC)
set_target_properties(tensor PROPERTIES POSITION_INDEPENDENT_CODE ON)

if(DEVICE_BACKEND STREQUAL "NVIDIA")
    target_link_libraries(tensor nvToolsExt cudart)
endif()

install(TARGETS tensor LIBRARY DESTINATION lib) 
