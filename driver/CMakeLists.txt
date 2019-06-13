set(TENSOR_SOURCE 
    src/tensor.cpp;
    src/device.cpp;
)

add_library(tensor SHARED ${TENSOR_SOURCE})
target_compile_features(tensor PUBLIC)
set_target_properties(tensor PROPERTIES POSITION_INDEPENDENT_CODE ON)

if(DEVICE_BACKEND STREQUAL "NVIDIA")
    target_link_libraries(tensor nvToolsExt cudart)
endif()

install(TARGETS tensor LIBRARY DESTINATION lib) 


if(DEVICE_BACKEND STREQUAL "AMD")
    set(DRIVER_SOURCE src/driver.cpp)
elseif(DEVICE_BACKEND STREQUAL "NVIDIA")
    set(DRIVER_SOURCE src/driver.cu)
endif()

add_executable(driver ${DRIVER_SOURCE}) 
target_link_libraries(driver PRIVATE tensor)
