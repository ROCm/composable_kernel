set(EXAMPLE_POOL_2D "example_pool2d")
message(DEBUG "adding example ${EXAMPLE_POOL_2D}")

add_executable(${EXAMPLE_POOL_2D} EXCLUDE_FROM_ALL pool2d.cpp)
target_include_directories(${EXAMPLE_POOL_2D} PRIVATE ${CMAKE_CURRENT_LIST_DIR})
set(EXAMPLE_POOL_COMPILE_OPTIONS)

list(APPEND EXAMPLE_POOL_COMPILE_OPTIONS -Wno-undefined-func-template -Wno-float-equal)

target_compile_options(${EXAMPLE_POOL_2D} PRIVATE ${EXAMPLE_POOL_COMPILE_OPTIONS})

set(EXAMPLE_POOL_3D "example_pool3d")
message(DEBUG "adding example ${EXAMPLE_POOL_3D}")

add_executable(${EXAMPLE_POOL_3D} EXCLUDE_FROM_ALL pool3d.cpp)
target_include_directories(${EXAMPLE_POOL_3D} PRIVATE ${CMAKE_CURRENT_LIST_DIR})

target_compile_options(${EXAMPLE_POOL_3D} PRIVATE ${EXAMPLE_POOL_COMPILE_OPTIONS})

set_property(GLOBAL PROPERTY RULE_MESSAGES OFF)