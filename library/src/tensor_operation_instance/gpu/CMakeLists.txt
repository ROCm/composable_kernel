function(add_instance_library INSTANCE_NAME)
    message("adding instance ${INSTANCE_NAME}")
    set(result 1)
    if(DEFINED DTYPES)
      foreach(source IN LISTS ARGN)
        set(test 0)
        foreach(type IN LISTS DTYPES)
                if(type MATCHES "fp16")
                        set(type1 "_f16")
                elseif(type MATCHES "fp32")
                        set(type1 "_f32")
                elseif(type MATCHES "fp8")
                        set(type1 "_f8")
                elseif(type MATCHES "bf16")
                        set(type1 "_b16")
                elseif(type MATCHES "fp64")
                        set(type1 "_f64")
                elseif(type MATCHES "int8")
                        set(type1 "_i8")
                endif()
                #make an exception for reduction kernels
                if("${source}" MATCHES "${type}" OR "${source}" MATCHES "${type1}" OR "${source}" MATCHES "device_reduce_instance")
                        #if filename matches any selected type, exit type loop and do no exclude the file from the list
                        set(test 0)
                        break()
                elseif((source MATCHES "fp8" OR source MATCHES "fp32" OR source MATCHES "fp64" OR source MATCHES "bf16" OR source MATCHES "int8" OR source MATCHES "fp16" OR
                  source MATCHES "_f8" OR source MATCHES "_f32" OR source MATCHES "_f64" OR source MATCHES "_i8" OR source MATCHES "_f16" OR source MATCHES "_b16") AND 
                  NOT(source MATCHES type OR source MATCHES type1))
                        #if filename contains a type which doesn't match any selected type, mark it for removal
                        set(test 1)
                endif()
        endforeach()
        if(test EQUAL 1)
                message("removing instance ${source} ")
                list(REMOVE_ITEM ARGN "${source}")
        endif()
      endforeach()
    endif()
    foreach(source IN LISTS ARGN)
        if(NOT DEFINED DL_KERNELS AND source MATCHES "_dl")
                message("removing dl instance ${source} ")
                list(REMOVE_ITEM ARGN "${source}")
        endif()
    endforeach()
    #only continue if there are some source files left on the list
    if(ARGN)
        add_library(${INSTANCE_NAME} OBJECT ${ARGN})
        target_compile_features(${INSTANCE_NAME} PUBLIC)
        set_target_properties(${INSTANCE_NAME} PROPERTIES POSITION_INDEPENDENT_CODE ON)
        clang_tidy_check(${INSTANCE_NAME})
        set(result 0)
    endif()
    #message("add_instance_library returns ${result}")
    return(PROPAGATE result)
endfunction(add_instance_library INSTANCE_NAME)


file(GLOB dir_list LIST_DIRECTORIES true *)
set(CK_DEVICE_INSTANCES)
FOREACH(subdir_path ${dir_list})
set(target_dir)
IF(IS_DIRECTORY "${subdir_path}")
    set(cmake_instance)
    file(READ "${subdir_path}/CMakeLists.txt" cmake_instance)
    set(add_inst 0)
    if(("${cmake_instance}" MATCHES "_fp8" OR "${cmake_instance}" MATCHES "_f8") AND DTYPES MATCHES "fp8")
            message("fp8 instance found!")
            set(add_inst 1)
    endif()
    if(("${cmake_instance}" MATCHES "_fp16" OR "${cmake_instance}" MATCHES "_f16") AND DTYPES MATCHES "fp16")
            message("fp16 instance found!")
            set(add_inst 1)
    endif()
    if(("${cmake_instance}" MATCHES "_fp32" OR "${cmake_instance}" MATCHES "_f32") AND DTYPES MATCHES "fp32")
            message("fp32 instance found!")
            set(add_inst 1)
    endif()
    if(("${cmake_instance}" MATCHES "_fp64" OR "${cmake_instance}" MATCHES "_f64") AND DTYPES MATCHES "fp64")
            message("fp64 instance found!")
            set(add_inst 1)
    endif()
    if("${cmake_instance}" MATCHES "_bf16" AND DTYPES MATCHES "bf16")
            message("bf16 instance found!")
            set(add_inst 1)
    endif()
    if(("${cmake_instance}" MATCHES "_int8" OR "${cmake_instance}" MATCHES "_i8") AND DTYPES MATCHES "int8")
            message("int8 instance found!")
            set(add_inst 1)
    endif()
    if(NOT "${cmake_instance}" MATCHES "_fp8" OR
                   NOT "${cmake_instance}" MATCHES "_f8" OR
                   NOT "${cmake_instance}" MATCHES "_fp16" OR
                   NOT "${cmake_instance}" MATCHES "_f16" OR
                   NOT "${cmake_instance}" MATCHES "_fp32" OR
                   NOT "${cmake_instance}" MATCHES "_f32" OR
                   NOT "${cmake_instance}" MATCHES "_fp64" OR
                   NOT "${cmake_instance}" MATCHES "_f64" OR
                   NOT "${cmake_instance}" MATCHES "_bf16" OR
                   NOT "${cmake_instance}" MATCHES "_int8" OR
                   NOT "${cmake_instance}" MATCHES "_i8" OR
                   NOT "${cmake_instance}" MATCHES "_int4" OR
                   NOT DEFINED DTYPES)
             message("instance should be built for all types!")
             set(add_inst 1)
     endif()
    if("${cmake_instance}" MATCHES "quantization" AND DEFINED DTYPES AND NOT DTYPES MATCHES "int8")
           message("quantization instances will not be built!")
           set(add_inst 0)
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
