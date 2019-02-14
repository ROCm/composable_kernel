set(TENSOR_SOURCE 
    tensor.cpp;
)

add_library(tensor SHARED ${TENSOR_SOURCE})
set_target_properties(tensor PROPERTIES PREFIX "")
target_compile_features(tensor PUBLIC)
set_target_properties(tensor PROPERTIES POSITION_INDEPENDENT_CODE ON)
install(TARGETS tensor LIBRARY DESTINATION lib) 


set(DEVICE_SOURCE 
    device.cu;
)

add_library(device SHARED ${DEVICE_SOURCE})
set_target_properties(device PROPERTIES PREFIX "")
target_compile_features(device PUBLIC)
set_target_properties(device PROPERTIES POSITION_INDEPENDENT_CODE ON)
install(TARGETS device LIBRARY DESTINATION lib) 
target_link_libraries(device nvToolsExt cudart)
