
set(CMAKE_CXX_COMPILER /opt/rocm/llvm/bin/clang++)

## for online-compiling of HIP kernels
set(OLC_HIP_COMPILER ${CMAKE_CXX_COMPILER} CACHE PATH "")

## reset to avoid the C++ options from the parent project
set(CMAKE_CXX_FLAGS "")
message("Compiling options for library and kernels: ${CMAKE_CXX_FLAGS}")

# look for and register clang-offload-bundler
if(OLC_HIP_COMPILER MATCHES ".*clang\\+\\+$")
    find_program(OLC_OFFLOADBUNDLER_BIN clang-offload-bundler
        PATH_SUFFIXES bin
        PATHS
	    /opt/rocm/llvm
	    ${CMAKE_INSTALL_PREFIX}/llvm
    )
endif()
if(OLC_OFFLOADBUNDLER_BIN)
    message(STATUS "clang-offload-bundler found: ${OLC_OFFLOADBUNDLER_BIN}")
    set(OLC_OFFLOADBUNDLER_BIN "${OLC_OFFLOADBUNDLER_BIN}")
else()
    # look for and register extractkernel
    message(STATUS "clang-offload-bundler not found")

    find_program(EXTRACTKERNEL_BIN extractkernel
        PATH_SUFFIXES bin
        PATHS
            /opt/rocm/hip
            /opt/rocm/hcc
            /opt/rocm
	    ${CMAKE_INSTALL_PREFIX}/hip
            ${CMAKE_INSTALL_PREFIX}/hcc
            ${CMAKE_INSTALL_PREFIX}

    )
    if(EXTRACTKERNEL_BIN)
        message(STATUS "extractkernel found: ${EXTRACTKERNEL_BIN}")
        set(EXTRACTKERNEL_BIN "${EXTRACTKERNEL_BIN}")
    else()
        message(FATAL_ERROR "extractkernel not found")
    endif()
endif()

option(Boost_USE_STATIC_LIBS "Use boost static libraries" OFF)
set(BOOST_COMPONENTS filesystem)
add_definitions(-DBOOST_ALL_NO_LIB=1)
find_package(Boost REQUIRED COMPONENTS ${BOOST_COMPONENTS})

# HIP is always required
find_package(hip REQUIRED PATHS /opt/rocm)
message(STATUS "Build with HIP ${hip_VERSION}")
target_flags(HIP_COMPILER_FLAGS hip::device)
# Remove cuda arch flags
string(REGEX REPLACE --cuda-gpu-arch=[a-z0-9]+ "" HIP_COMPILER_FLAGS "${HIP_COMPILER_FLAGS}")
string(REGEX REPLACE --offload-arch=[a-z0-9]+ "" HIP_COMPILER_FLAGS "${HIP_COMPILER_FLAGS}")

set(OLC_hip_VERSION_MAJOR "${hip_VERSION_MAJOR}")
set(OLC_hip_VERSION_MINOR "${hip_VERSION_MINOR}")
set(OLC_hip_VERSION_PATCH "${hip_VERSION_PATCH}")

option(ENABLE_DEBUG "Build to enable debugging" ON)
if(ENABLE_DEBUG)
    set(OLC_DEBUG 1)
else()
    set(OLC_DEBUG 0)
endif()

configure_file("${CMAKE_CURRENT_SOURCE_DIR}/olCompiling/include/config.h.in" "${CMAKE_CURRENT_SOURCE_DIR}/olCompiling/include/config.h")

message(STATUS "Hip compiler flags: ${HIP_COMPILER_FLAGS}")

## HIP_COMPILER_FLAGS will be used for on-line compiling of the HIP kernels
add_definitions("-DHIP_COMPILER_FLAGS=${HIP_COMPILER_FLAGS}")

file(GLOB COMPOSABLE_KERNEL_INCLUDE_1 "${PROJECT_SOURCE_DIR}/composable_kernel/include/kernel_algorithm/*.hpp")
file(GLOB COMPOSABLE_KERNEL_INCLUDE_2 "${PROJECT_SOURCE_DIR}/composable_kernel/include/tensor_description/*.hpp")
file(GLOB COMPOSABLE_KERNEL_INCLUDE_3 "${PROJECT_SOURCE_DIR}/composable_kernel/include/tensor_operation/*.hpp")
file(GLOB COMPOSABLE_KERNEL_INCLUDE_4 "${PROJECT_SOURCE_DIR}/composable_kernel/include/utility/*.hpp")
file(GLOB COMPOSABLE_KERNEL_INCLUDE_5 "${PROJECT_BINARY_DIR}/composable_kernel/include/utility/*.hpp") 
file(GLOB COMPOSABLE_KERNEL_INCLUDE_6 "${PROJECT_SOURCE_DIR}/external/rocm/include/bfloat16_dev.hpp")
set(MCONV_KERNEL_INCLUDES
    ${COMPOSABLE_KERNEL_INCLUDE_1}
    ${COMPOSABLE_KERNEL_INCLUDE_2}
    ${COMPOSABLE_KERNEL_INCLUDE_3}
    ${COMPOSABLE_KERNEL_INCLUDE_4}
    ${COMPOSABLE_KERNEL_INCLUDE_5}
    ${COMPOSABLE_KERNEL_INCLUDE_6}
   )

set(MCONV_KERNELS
   ../composable_kernel/src/kernel_wrapper/dynamic_convolution_forward_implicit_gemm_v4r4_nchw_kcyx_nkhw.cpp	
   ../composable_kernel/src/kernel_wrapper/dynamic_convolution_forward_implicit_gemm_v4r5_nchw_kcyx_nkhw.cpp	
   )

add_kernels("olCompiling/" "${MCONV_KERNELS}")
add_kernel_includes("olCompiling/" "${MCONV_KERNEL_INCLUDES}")

set(MCONV_SOURCES
    src/host_tensor.cpp;
    src/device.cpp;
)

set(OLC_HIP_UTILITY_HEADERS
    olCompiling/include/config.h
    olCompiling/include/logger.hpp
    olCompiling/include/stringutils.hpp 
    olCompiling/include/tmp_dir.hpp
    olCompiling/include/write_file.hpp
    olCompiling/include/env.hpp
    olCompiling/include/manage_ptr.hpp  
    olCompiling/include/md5.hpp
    olCompiling/include/simple_hash.hpp
    olCompiling/include/exec_utils.hpp
    olCompiling/include/hipCheck.hpp
    olCompiling/include/target_properties.hpp
    olCompiling/include/handle.hpp
    olCompiling/include/op_kernel_args.hpp  
    olCompiling/include/kernel.hpp
    olCompiling/include/kernel_build_params.hpp
    olCompiling/include/hip_build_utils.hpp
    olCompiling/include/hipoc_program.hpp
    olCompiling/include/hipoc_program_impl.hpp
    olCompiling/include/hipoc_kernel.hpp  
    olCompiling/include/kernel_cache.hpp
    olCompiling/include/binary_cache.hpp  
   )

set(OLC_HIP_UTILITY_CPPS
    olCompiling/hip_utility/logger.cpp
    olCompiling/hip_utility/tmp_dir.cpp
    olCompiling/hip_utility/md5.cpp  
    olCompiling/hip_utility/exec_utils.cpp
    olCompiling/hip_utility/target_properties.cpp  
    olCompiling/hip_utility/handlehip.cpp
    olCompiling/hip_utility/kernel_build_params.cpp  
    olCompiling/hip_utility/hip_build_utils.cpp  
    olCompiling/hip_utility/hipoc_program.cpp  
    olCompiling/hip_utility/hipoc_kernel.cpp  
    olCompiling/hip_utility/kernel_cache.cpp  
    olCompiling/hip_utility/binary_cache.cpp
   )

list(APPEND OLC_SOURCES ${OLC_HIP_UTILITY_CPPS} ${OLC_HIP_UTILITY_HEADERS})

list(INSERT MCONV_SOURCES 0
     ${PROJECT_BINARY_DIR}/kernel.cpp
     ${PROJECT_BINARY_DIR}/kernel_includes.cpp
    )

## addkernels provide the tool to create inlined kernels in one header
add_subdirectory(olCompiling/addkernels)

function(inline_kernels_src KERNELS KERNEL_INCLUDES)
    set(KERNEL_SRC_HPP_FILENAME batch_all.cpp.hpp)
    set(KERNEL_SRC_HPP_PATH ${PROJECT_BINARY_DIR}/inlined_kernels/${KERNEL_SRC_HPP_FILENAME})
    set(KERNEL_SRC_CPP_PATH ${PROJECT_BINARY_DIR}/inlined_kernels/batch_all.cpp)

    add_custom_command(
        OUTPUT ${KERNEL_SRC_HPP_PATH}
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
        DEPENDS addkernels ${KERNELS} ${KERNEL_INCLUDES}
        COMMAND $<TARGET_FILE:addkernels> -target ${KERNEL_SRC_HPP_PATH} -extern -source ${KERNELS}
	COMMENT "Inlining All kernels"
    )
    configure_file(olCompiling/kernels_batch.cpp.in ${KERNEL_SRC_CPP_PATH})
    list(APPEND OLC_SOURCES ${KERNEL_SRC_CPP_PATH} ${KERNEL_SRC_HPP_PATH})

    set(OLC_SOURCES ${OLC_SOURCES} PARENT_SCOPE)
endfunction()

inline_kernels_src("${MCONV_KERNELS}" "${MCONV_KERNEL_INCLUDES}")

list(APPEND MCONV_SOURCES ${OLC_SOURCES} ${PROJECT_BINARY_DIR}/olc_kernel_includes.h)

add_custom_command(
    OUTPUT ${PROJECT_BINARY_DIR}/olc_kernel_includes.h
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    DEPENDS addkernels ${MCONV_KERNEL_INCLUDES}
    COMMAND $<TARGET_FILE:addkernels> -no-recurse -guard GUARD_OLC_KERNEL_INCLUDES_HPP_ -target ${PROJECT_BINARY_DIR}/olc_kernel_includes.h -source ${MCONV_KERNEL_INCLUDES}
    COMMENT "Inlining HIP kernel includes"
  )

## the library target
add_library(modConv SHARED ${MCONV_SOURCES}) 

target_include_directories(modConv PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/olCompiling/include/)
target_include_directories(modConv PRIVATE ${PROJECT_BINARY_DIR})
target_include_directories(modConv PRIVATE ${PROJECT_SOURCE_DIR}/external/half/include/)

target_link_libraries(modConv PRIVATE hip::device)
target_link_libraries(modConv INTERFACE hip::host)
target_link_libraries(modConv PRIVATE Boost::filesystem)

target_compile_options(modConv PRIVATE -mfma)

target_compile_features(modConv PUBLIC)
set_target_properties(modConv PROPERTIES POSITION_INDEPENDENT_CODE ON)

install(TARGETS modConv LIBRARY DESTINATION lib) 
