set(DRIVER_SOURCE driver.cpp)

add_executable(driver ${DRIVER_SOURCE}) 
target_link_libraries(driver PRIVATE tensor)
