cmake_minimum_required(VERSION 3.5)

# Check support for CUDA/HIP in Cmake
project(composable_kernel)

list(APPEND CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake")

enable_testing()

find_package(ROCM REQUIRED PATHS /opt/rocm)

include(ROCMInstallTargets)
include(ROCMPackageConfigHelpers)
include(ROCMSetupVersion)
include(ROCMInstallSymlinks)
include(ROCMCreatePackage)
include(CheckCXXCompilerFlag)

rocm_setup_version(VERSION 1.0.0)
include(TargetFlags)
list(APPEND CMAKE_PREFIX_PATH ${CMAKE_INSTALL_PREFIX} ${CMAKE_INSTALL_PREFIX}/llvm ${CMAKE_INSTALL_PREFIX}/hip /opt/rocm /opt/rocm/llvm /opt/rocm/hip)

## C++
enable_language(CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
message("CMAKE_CXX_COMPILER_ID: ${CMAKE_CXX_COMPILER_ID}")

## OpenMP
if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
	# workaround issue hipcc in rocm3.5 cannot find openmp
	set(OpenMP_CXX "${CMAKE_CXX_COMPILER}")
	set(OpenMP_CXX_FLAGS "-fopenmp=libomp -Wno-unused-command-line-argument")
	set(OpenMP_CXX_LIB_NAMES "libomp" "libgomp" "libiomp5")
	set(OpenMP_libomp_LIBRARY ${OpenMP_CXX_LIB_NAMES})
	set(OpenMP_libgomp_LIBRARY ${OpenMP_CXX_LIB_NAMES})
	set(OpenMP_libiomp5_LIBRARY ${OpenMP_CXX_LIB_NAMES})
else()
	find_package(OpenMP REQUIRED)
endif()

message("OpenMP_CXX_LIB_NAMES: ${OpenMP_CXX_LIB_NAMES}")
message("OpenMP_gomp_LIBRARY: ${OpenMP_gomp_LIBRARY}")
message("OpenMP_pthread_LIBRARY: ${OpenMP_pthread_LIBRARY}")
message("OpenMP_CXX_FLAGS: ${OpenMP_CXX_FLAGS}")

# set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
link_libraries(${OpenMP_gomp_LIBRARY})
link_libraries(${OpenMP_pthread_LIBRARY})

## HIP
find_package(HIP REQUIRED)
# Override HIP version in config.h, if necessary.
# The variables set by find_package() can't be overwritten,
# therefore let's use intermediate variables.
set(CK_HIP_VERSION_MAJOR "${HIP_VERSION_MAJOR}")
set(CK_HIP_VERSION_MINOR "${HIP_VERSION_MINOR}")
set(CK_HIP_VERSION_PATCH "${HIP_VERSION_PATCH}")
if( DEFINED CK_OVERRIDE_HIP_VERSION_MAJOR )
    set(CK_HIP_VERSION_MAJOR "${CK_OVERRIDE_HIP_VERSION_MAJOR}")
    message(STATUS "CK_HIP_VERSION_MAJOR overriden with ${CK_OVERRIDE_HIP_VERSION_MAJOR}")
endif()
if( DEFINED CK_OVERRIDE_HIP_VERSION_MINOR )
    set(CK_HIP_VERSION_MINOR "${CK_OVERRIDE_HIP_VERSION_MINOR}")
    message(STATUS "CK_HIP_VERSION_MINOR overriden with ${CK_OVERRIDE_HIP_VERSION_MINOR}")
endif()
if( DEFINED CK_OVERRIDE_HIP_VERSION_PATCH )
    set(CK_HIP_VERSION_PATCH "${CK_OVERRIDE_HIP_VERSION_PATCH}")
    message(STATUS "CK_HIP_VERSION_PATCH overriden with ${CK_OVERRIDE_HIP_VERSION_PATCH}")
endif()
message(STATUS "Build with HIP ${HIP_VERSION}")

## half
#find_path(HALF_INCLUDE_DIR half.hpp)
set(HALF_INCLUDE_DIR "${PROJECT_SOURCE_DIR}/external/half/include")
message("HALF_INCLUDE_DIR: ${HALF_INCLUDE_DIR}")


rocm_create_package(
    NAME CK-${CK_BACKEND}
    DESCRIPTION "High Performance Composable Kernels for AMD GPUs"
    LDCONFIG
)
## tidy
include(EnableCompilerWarnings)
set(CK_TIDY_ERRORS ERRORS * -readability-inconsistent-declaration-parameter-name)
if(CMAKE_CXX_COMPILER MATCHES ".*hcc" OR CMAKE_CXX_COMPILER MATCHES ".*clang\\+\\+")
    set(CK_TIDY_CHECKS -modernize-use-override -readability-non-const-parameter)
# Enable tidy on hip
elseif(CK_BACKEND STREQUAL "HIP" OR CK_BACKEND STREQUAL "HIPNOGPU")
    set(CK_TIDY_ERRORS ALL)
endif()


include(ClangTidy)
enable_clang_tidy(
    CHECKS
        *
        -abseil-*
        -android-cloexec-fopen
        # Yea we shouldn't be using rand()
        -cert-msc30-c
        -bugprone-exception-escape
        -bugprone-macro-parentheses
        -cert-env33-c
        -cert-msc32-c
        -cert-msc50-cpp
        -cert-msc51-cpp
        -cert-dcl37-c
        -cert-dcl51-cpp
        -clang-analyzer-alpha.core.CastToStruct
        -clang-analyzer-optin.performance.Padding
        -clang-diagnostic-deprecated-declarations
        -clang-diagnostic-extern-c-compat
        -clang-diagnostic-unused-command-line-argument
        -cppcoreguidelines-avoid-c-arrays
        -cppcoreguidelines-avoid-magic-numbers
        -cppcoreguidelines-explicit-virtual-functions
        -cppcoreguidelines-init-variables
        -cppcoreguidelines-macro-usage
        -cppcoreguidelines-non-private-member-variables-in-classes
        -cppcoreguidelines-pro-bounds-array-to-pointer-decay
        -cppcoreguidelines-pro-bounds-constant-array-index
        -cppcoreguidelines-pro-bounds-pointer-arithmetic
        -cppcoreguidelines-pro-type-member-init
        -cppcoreguidelines-pro-type-reinterpret-cast
        -cppcoreguidelines-pro-type-union-access
        -cppcoreguidelines-pro-type-vararg
        -cppcoreguidelines-special-member-functions
        -fuchsia-*
        -google-explicit-constructor
        -google-readability-braces-around-statements
        -google-readability-todo
        -google-runtime-int
        -google-runtime-references
        -hicpp-vararg
        -hicpp-braces-around-statements
        -hicpp-explicit-conversions
        -hicpp-named-parameter
        -hicpp-no-array-decay
        # We really shouldn't use bitwise operators with signed integers, but
        # opencl leaves us no choice
        -hicpp-avoid-c-arrays
        -hicpp-signed-bitwise
        -hicpp-special-member-functions
        -hicpp-uppercase-literal-suffix
        -hicpp-use-auto
        -hicpp-use-equals-default
        -hicpp-use-override
        -llvm-header-guard
        -llvm-include-order
        #-llvmlibc-*
        -llvmlibc-restrict-system-libc-headers
        -llvmlibc-callee-namespace
        -llvmlibc-implementation-in-namespace
        -llvm-else-after-return
        -llvm-qualified-auto
        -misc-misplaced-const
        -misc-non-private-member-variables-in-classes
        -misc-no-recursion
        -modernize-avoid-bind
        -modernize-avoid-c-arrays
        -modernize-pass-by-value
        -modernize-use-auto
        -modernize-use-default-member-init
        -modernize-use-equals-default
        -modernize-use-trailing-return-type
        -modernize-use-transparent-functors
        -performance-unnecessary-value-param
        -readability-braces-around-statements
        -readability-else-after-return
        # we are not ready to use it, but very useful
        -readability-function-cognitive-complexity
        -readability-isolate-declaration
        -readability-magic-numbers
        -readability-named-parameter
        -readability-uppercase-literal-suffix
        -readability-convert-member-functions-to-static
        -readability-qualified-auto
        -readability-redundant-string-init
        # too many narrowing conversions in our code
        -bugprone-narrowing-conversions
        -cppcoreguidelines-narrowing-conversions
        -altera-struct-pack-align
        -cppcoreguidelines-prefer-member-initializer

        ${CK_TIDY_CHECKS}
        ${CK_TIDY_ERRORS}
    HEADER_FILTER
        "\.hpp$"
    EXTRA_ARGS
        -DCK_USE_CLANG_TIDY
)

include(CppCheck)
enable_cppcheck(
    CHECKS
        warning
        style
        performance
        portability
    SUPPRESS
        ConfigurationNotChecked
        constStatement
        duplicateCondition
        noExplicitConstructor
        passedByValue
        preprocessorErrorDirective
        shadowVariable
        unusedFunction
        unusedPrivateFunction
        unusedStructMember
        unmatchedSuppression
    FORCE
    SOURCES
        host/host_tensor/src
        host/driver_offline/src
        composable_kernel/src/kernel_wrapper
    INCLUDE
        host/host_tensor/include
        host/device/include
        host/solver/include
        host/driver_offline/include
        composable_kernel/include/*
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${CMAKE_CURRENT_BINARY_DIR}/include
    DEFINE
        CPPCHECK=1
        __linux__=1
)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/lib)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/bin)

file(GLOB_RECURSE COMPOSABLE_KERNEL_HEADERS "composable_kernel/include/*/*.hpp")
file(GLOB_RECURSE DEVICE_OPS_HEADERS "device_operation/include/*.hpp")

file(GLOB_RECURSE DEVICE_OPS_SOURCE "device_operation/*.cpp")

set(CK_HEADERS ${COMPOSABLE_KERNEL_HEADERS} ${DEVICE_OPS_HEADERS})
set(CK_SOURCE ${DEVICE_OPS_SOURCE})
add_library(composable_kernel
    ${CK_SOURCE}
)

target_include_directories(composable_kernel PUBLIC
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/composable_kernel/include>
)
target_include_directories(composable_kernel PUBLIC 
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/device_operation/include>
)
target_include_directories(composable_kernel PUBLIC 
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/host/include>
)
target_include_directories(composable_kernel PUBLIC 
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/host/host_tensor/include>
)
# The following should eventually be removed
target_include_directories(composable_kernel PUBLIC
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/composable_kernel/include/utility>
)
target_include_directories(composable_kernel PUBLIC
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/composable_kernel/include/tensor_operation>
)
target_include_directories(composable_kernel PUBLIC
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/composable_kernel/include/tensor_description>
)
# clang_tidy_check(composable_kernel)
SET(BUILD_DEV ON CACHE BOOL "BUILD_DEV")
if(BUILD_DEV)
    target_compile_options(composable_kernel PRIVATE -Werror)
    target_compile_options(composable_kernel PRIVATE -Weverything)
endif()
message("CMAKE_CXX_FLAGS: ${CMAKE_CXX_FLAGS}")

configure_file("${PROJECT_SOURCE_DIR}/composable_kernel/include/hip_version.hpp.in" "${PROJECT_BINARY_DIR}/composable_kernel/include/hip_version.hpp")

add_subdirectory(host)
add_subdirectory(device_operation)
add_subdirectory(example)
add_subdirectory(profiler)
add_subdirectory(test)
