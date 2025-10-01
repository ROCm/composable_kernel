set(EXAMPLE_POOL_3D "example_pool3d")
message(DEBUG "adding example ${EXAMPLE_POOL_3D}")

add_executable(${EXAMPLE_POOL_3D} EXCLUDE_FROM_ALL pool3d.cpp)
target_include_directories(${EXAMPLE_POOL_3D} PRIVATE ${CMAKE_CURRENT_LIST_DIR})

target_compile_options(${EXAMPLE_POOL_3D} PRIVATE ${EXAMPLE_POOL_COMPILE_OPTIONS})

set_property(GLOBAL PROPERTY RULE_MESSAGES OFF)