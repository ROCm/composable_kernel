set(EXAMPLE_BUFFER "example_buffer")

message(DEBUG "adding example ${EXAMPLE_BUFFER}")

add_executable(${EXAMPLE_BUFFER} EXCLUDE_FROM_ALL buffer_examples.cpp)
target_include_directories(${EXAMPLE_BUFFER} PRIVATE ${CMAKE_CURRENT_LIST_DIR})
set(EXAMPLE_BUFFER_COMPILE_OPTIONS)

# NOTE: we turn off undefined-func-template to let source compile without explicit declare function specializations
list(APPEND EXAMPLE_BUFFER_COMPILE_OPTIONS -Wno-undefined-func-template -Wno-float-equal)

# Debug and analysis flags
list(APPEND EXAMPLE_BUFFER_COMPILE_OPTIONS 
    -g                          # Debug symbols
    -Wno-unused-variable       # Suppress unused variable warnings
    -Wno-unused-parameter      # Suppress unused parameter warnings
    -Wno-unused-function       # Suppress unused function warnings
)

target_compile_options(${EXAMPLE_BUFFER} PRIVATE ${EXAMPLE_BUFFER_COMPILE_OPTIONS})

set_property(GLOBAL PROPERTY RULE_MESSAGES OFF)

# Add transformations example
set(EXAMPLE_TRANSFORMATIONS "example_transformations")

message(DEBUG "adding example ${EXAMPLE_TRANSFORMATIONS}")

add_executable(${EXAMPLE_TRANSFORMATIONS} EXCLUDE_FROM_ALL transformations.cpp)
target_include_directories(${EXAMPLE_TRANSFORMATIONS} PRIVATE ${CMAKE_CURRENT_LIST_DIR})
set(EXAMPLE_TRANSFORMATIONS_COMPILE_OPTIONS)

# NOTE: we turn off undefined-func-template to let source compile without explicit declare function specializations
list(APPEND EXAMPLE_TRANSFORMATIONS_COMPILE_OPTIONS -Wno-undefined-func-template -Wno-float-equal)

# Debug and analysis flags
list(APPEND EXAMPLE_TRANSFORMATIONS_COMPILE_OPTIONS 
    -g                          # Debug symbols
    -Wno-unused-variable       # Suppress unused variable warnings
    -Wno-unused-parameter      # Suppress unused parameter warnings
    -Wno-unused-function       # Suppress unused function warnings
)

target_compile_options(${EXAMPLE_TRANSFORMATIONS} PRIVATE ${EXAMPLE_TRANSFORMATIONS_COMPILE_OPTIONS})