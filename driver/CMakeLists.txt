if(DEVICE_BACKEND STREQUAL "AMD")
    set(DRIVER_SOURCE driver.cpp)
elseif(DEVICE_BACKEND STREQUAL "NVIDIA")
    set(DRIVER_SOURCE driver.cu)
endif()

add_executable(driver ${DRIVER_SOURCE}) 
target_link_libraries(driver PRIVATE tensor)
