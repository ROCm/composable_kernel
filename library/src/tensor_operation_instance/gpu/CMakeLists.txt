function(add_instance_library INSTANCE_NAME)
    message("adding instance ${INSTANCE_NAME}")
    add_library(${INSTANCE_NAME} OBJECT ${ARGN})
    target_compile_features(${INSTANCE_NAME} PUBLIC)
    set_target_properties(${INSTANCE_NAME} PROPERTIES POSITION_INDEPENDENT_CODE ON)
    clang_tidy_check(${INSTANCE_NAME})
endfunction(add_instance_library INSTANCE_NAME)


file(GLOB dir_list LIST_DIRECTORIES true *)
set(CK_DEVICE_INSTANCES)
FOREACH(subdir_path ${dir_list})
set(target_dir)
IF(IS_DIRECTORY "${subdir_path}")
    set(cmake_instance)
    file(READ "${subdir_path}/CMakeLists.txt" cmake_instance)
    set(add_inst 0)
    if("${cmake_instance}" MATCHES "DTYPES MATCHES \"fp8\" " AND DTYPES MATCHES "fp8")
            #message("fp8 instance found!")
            set(add_inst 1)
    endif()
    if("${cmake_instance}" MATCHES "DTYPES MATCHES \"fp16\"" AND DTYPES MATCHES "fp16")
            #message("fp16 instance found!")
            set(add_inst 1)
    endif()
    if("${cmake_instance}" MATCHES "DTYPES MATCHES \"fp32\"" AND DTYPES MATCHES "fp32")
            #message("fp32 instance found!")
            set(add_inst 1)
    endif()
    if("${cmake_instance}" MATCHES "DTYPES MATCHES \"fp64\"" AND DTYPES MATCHES "fp64")
            #message("fp64 instance found!")
            set(add_inst 1)
    endif()
    if("${cmake_instance}" MATCHES "DTYPES MATCHES \"bf16\"" AND DTYPES MATCHES "bf16")
            #message("bf16 instance found!")
            set(add_inst 1)
    endif()
    if("${cmake_instance}" MATCHES "DTYPES MATCHES \"int8\"" AND DTYPES MATCHES "int8")
            #message("int8 instance found!")
            set(add_inst 1)
    endif()
    if(NOT "${cmake_instance}" MATCHES "DTYPES" OR NOT DEFINED DTYPES)
            #message("instance should be built for all types!")
            set(add_inst 1)
    endif()
    if("${cmake_instance}" MATCHES "ONLY DL_KERNELS" AND NOT DEFINED DL_KERNELS)
	    message("Found only dl instances, but DL_KERNELS is not set. Skipping.")
            set(add_inst 0)
    endif()
    if(add_inst EQUAL 1)
      get_filename_component(target_dir ${subdir_path} NAME)
      add_subdirectory(${target_dir})
      list(APPEND CK_DEVICE_INSTANCES $<TARGET_OBJECTS:device_${target_dir}_instance>)
    endif()
ENDIF()
ENDFOREACH()

add_library(device_operations STATIC ${CK_DEVICE_INSTANCES})
add_library(composablekernels::device_operations ALIAS device_operations)


set(DEV_OPS_INC_DIRS
    ${PROJECT_SOURCE_DIR}/include/ck/
    ${PROJECT_SOURCE_DIR}/library/include/ck/
)

target_compile_features(device_operations PUBLIC)
set_target_properties(device_operations PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_include_directories(device_operations PUBLIC
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/ck>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/ck/utility>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/ck/tensor_description>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/ck/tensor>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/ck/problem_transform>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/ck/tensor_operation/gpu/device>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/ck/tensor_operation/gpu/device/impl>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/ck/tensor_operation/gpu/grid>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/ck/tensor_operation/gpu/block>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/ck/tensor_operation/gpu/warp>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/ck/tensor_operation/gpu/thread>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/ck/tensor_operation/gpu/element>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/ck/library/utility>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/ck/library/tensor_operation_instance>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/ck/library/tensor_operation_instance/gpu>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/ck/library/tensor_operation_instance/gpu/reduce>
)

#once new arches are enabled make this an option on the main cmake file
# and pass down here to be exported
target_compile_options(device_operations PRIVATE
    --offload-arch=gfx908
    --offload-arch=gfx90a
)

# install(TARGETS device_operations LIBRARY DESTINATION lib)
rocm_install(TARGETS device_operations
        EXPORT device_operationsTargets)

rocm_install(DIRECTORY ${DEV_OPS_INC_DIRS} DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/ck)
rocm_install(EXPORT device_operationsTargets
        FILE composable_kerneldevice_operationsTargets.cmake
        NAMESPACE composable_kernel::
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/composable_kernel
)
