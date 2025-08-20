
set(GEMM_DATATYPE "fp8;fp16" CACHE STRING "List of datatypes for GEMM (semicolon-separated)")
set(GEMM_LAYOUT "rcr" CACHE STRING "List of layout for GEMM (semicolon-separated)")
option(GEMM_BUILD_INDIVIDUAL "Build individual GEMM targets" ON)
option(GEMM_BUILD_MONOLITHIC "Build monolithic GEMM targets" OFF)

function(build_gemm_for_datatype datatype layout)
    # Filter GPU targets to only gfx90a, gfx942, and gfx950
    set(GEMM_GPU_TARGETS "")
    set(DESIRED_TARGETS "gfx90a;gfx942;gfx950")
    
    foreach(target IN LISTS SUPPORTED_GPU_TARGETS)
        if(target IN_LIST DESIRED_TARGETS)
            list(APPEND GEMM_GPU_TARGETS ${target})
        endif()
    endforeach()
    
    # Skip compilation if no matching targets found
    if(NOT GEMM_GPU_TARGETS)
        message(WARNING "Skipping Tile Engine GEMM compilation: No supported GPU targets (gfx90a, gfx942, gfx950) found in SUPPORTED_GPU_TARGETS: ${SUPPORTED_GPU_TARGETS}")
        return()
    endif()
    
    message(STATUS "Building GEMM for GPU targets: ${GEMM_GPU_TARGETS}")
    
    set(working_path "${CMAKE_CURRENT_BINARY_DIR}/${datatype}/${layout}")

    # Comment this if-else block when using user_provided_config
    if(layout STREQUAL "rcr")
        set(json_blob "${CMAKE_CURRENT_LIST_DIR}/configs/default_config.json")
    else()
        set(json_blob "${CMAKE_CURRENT_LIST_DIR}/configs/custom_ci_config.json")
    endif()

    # uncomment this if you want to use user_provided_config.json
    # set(json_blob "${CMAKE_CURRENT_LIST_DIR}/configs/user_provided_config.json")
    
    # Determine number of workers for parallel generation
    if(DEFINED ENV{CMAKE_BUILD_PARALLEL_LEVEL})
        set(num_workers $ENV{CMAKE_BUILD_PARALLEL_LEVEL})
    else()
        # Use processor count but limit to avoid memory issues
        cmake_host_system_information(RESULT num_cores QUERY NUMBER_OF_LOGICAL_CORES)
        math(EXPR num_workers "${num_cores}")
        if(num_workers GREATER 8)
            set(num_workers 8)
        endif()
    endif()

    # Generate kernel list
    execute_process(
        COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_LIST_DIR}/gemm_instance_builder_parallel.py
                --working_path ${working_path}
                --datatype ${datatype}
                --layout ${layout}
                --config_json ${json_blob}
                --num_workers ${num_workers}
                --list_blobs
        RESULT_VARIABLE ret
    )
    if(NOT ret EQUAL 0)
        message(FATAL_ERROR "Failed to list kernels for ${datatype} ${layout}: ${ret}")
    endif()

    file(STRINGS "${working_path}/gemm_instance_blobs.txt" codegen_blobs)
    file(STRINGS "${working_path}/gemm_instance_blobs_range.txt" codegen_blobs_range)
    
    # Generate the blobs
    add_custom_command(
        OUTPUT ${codegen_blobs}
        COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_LIST_DIR}/gemm_instance_builder_parallel.py
                --working_path "${working_path}"
                --datatype ${datatype}
                --layout ${layout}
                --config_json "${json_blob}"
                --num_workers ${num_workers}
                --gen_blobs
        COMMENT "Generating GEMM instance sources for ${datatype} ${layout} using ${num_workers} workers"
    )
    add_custom_target(gemm_gen_${datatype}_${layout} DEPENDS ${codegen_blobs})

    set(intermediate_libs)
    list(LENGTH codegen_blobs codegen_blobs_len)

    foreach(blob IN LISTS codegen_blobs_range)
        string(STRIP "${blob}" stripped_blob)
        separate_arguments(spilit_blob UNIX_COMMAND "${stripped_blob}")
        # Each line is: <trait_name> <first_index_inclusive> <last_index_exclusive>   
        list(GET spilit_blob 0 name)
        list(GET spilit_blob 1 first)
        list(GET spilit_blob 2 last)
        math(EXPR total_files "${last} - ${first}")
        if(total_files EQUAL 0)
            continue()        # nothing for this trait
        endif()

        # Object libraries (chunked) per trait
        set(sub_intermediate_libs)
        set(chunk_size 3)
        math(EXPR num_chunks "( ${total_files} + ${chunk_size} - 1 ) / ${chunk_size}")
        math(EXPR num_chunks_minus_1 "${num_chunks} - 1")
        
        foreach(i RANGE 0 ${num_chunks_minus_1})
            math(EXPR start "${first} + ${i} * ${chunk_size} ")
            math(EXPR end "${start} + ${chunk_size} - 1")

            set(chunk_files)
            foreach(j RANGE ${start} ${end})
                if(j LESS ${last} AND j LESS ${codegen_blobs_len})
                    list(GET codegen_blobs ${j} f)
                    list(APPEND chunk_files "${f}")
                endif()
            endforeach()

            #list(LENGTH chunk_files chunk_files_len)
            #if(chunk_files_len AND chunk_files_len GREATER 1)
            if(chunk_files)
                set(sub_intermediate_lib_name "gemm_objlib_${name}_${i}_${datatype}_${layout}")
                add_library(${sub_intermediate_lib_name} OBJECT ${chunk_files})
                set_property(TARGET ${sub_intermediate_lib_name} PROPERTY HIP_ARCHITECTURES ${GEMM_GPU_TARGETS})
                list(APPEND sub_intermediate_libs ${sub_intermediate_lib_name})
            endif()

        endforeach()

        # ------------------ Bundle the object libs into one static lib ---------
        #list(LENGTH sub_intermediate_libs sub_intermediate_libs_len)
        #if(sub_intermediate_libs AND sub_intermediate_libs_len GREATER 1)
        if(sub_intermediate_libs)
            set(intermediate_lib_name "gemm_staticlib_${name}_${datatype}_${layout}")
            # Collect the $<TARGET_OBJECTS:...> expressions
            
            set(obj_exprs)
            foreach(objlib IN LISTS sub_intermediate_libs)
                list(APPEND obj_exprs $<TARGET_OBJECTS:${objlib}>)
            endforeach()
            
            add_library(${intermediate_lib_name} STATIC ${obj_exprs})
            add_dependencies(${intermediate_lib_name} gemm_gen_${datatype}_${layout})
            set_property(TARGET ${intermediate_lib_name} PROPERTY HIP_ARCHITECTURES ${GEMM_GPU_TARGETS})
            #foreach(objlib IN LISTS sub_intermediate_libs)
            #    target_sources(${intermediate_lib_name} PRIVATE $<TARGET_OBJECTS:${objlib}>)
            #endforeach()
            list(APPEND intermediate_libs ${intermediate_lib_name})
        endif()

    endforeach()
    
    # Interface library for instances
    add_library(gemm_template_instances_${datatype}_${layout} INTERFACE)
    add_dependencies(gemm_template_instances_${datatype}_${layout} gemm_gen_${datatype}_${layout})
    target_link_libraries(gemm_template_instances_${datatype}_${layout} INTERFACE ${intermediate_libs})
    target_include_directories(gemm_template_instances_${datatype}_${layout} INTERFACE
        ${CMAKE_CURRENT_LIST_DIR}
        "${working_path}"
    )
    set_target_properties(gemm_template_instances_${datatype}_${layout} PROPERTIES LINKER_LANGUAGE CXX)
    
    # Host API interface library
    add_library(gemm_host_api_${datatype}_${layout} INTERFACE)
    target_link_libraries(gemm_host_api_${datatype}_${layout} INTERFACE gemm_template_instances_${datatype}_${layout})
    target_include_directories(gemm_host_api_${datatype}_${layout} INTERFACE
        ${CMAKE_CURRENT_LIST_DIR}
        "${working_path}"
    )
    

    # Executable per datatype
    set(exec_name "benchmark_gemm_${datatype}_${layout}")
    add_executable(${exec_name} benchmark_gemm.cpp)
    set_property(TARGET ${exec_name} PROPERTY HIP_ARCHITECTURES ${GEMM_GPU_TARGETS})
    target_link_libraries(${exec_name} PRIVATE gemm_host_api_${datatype}_${layout})
    target_compile_options(${exec_name} PRIVATE
        -Wno-undefined-func-template
        -Wno-float-equal
        --offload-compress
    )
endfunction()

# Store the directory path for use in functions
set(GEMM_SOURCE_DIR ${CMAKE_CURRENT_LIST_DIR})

# Function to create individual GEMM targets
function(create_individual_gemm_target datatype layout trait tile_config)
    # Use the parent scope GEMM_GPU_TARGETS_INDIVIDUAL variable
    if(NOT GEMM_GPU_TARGETS_INDIVIDUAL)
        message(WARNING "Skipping individual GEMM target ${datatype}_${layout}_${trait}_${tile_config}: No supported GPU targets")
        return()
    endif()
    
    # Parse tile configuration: format is tile_mxtile_nxtile_k_warp_mxwarp_nxwarp_k_warp_tile_mxwarp_tile_nxwarp_tile_k
    # First split by underscore to get three groups
    string(REPLACE "_" ";" config_groups ${tile_config})
    list(GET config_groups 0 tile_dims)      # e.g., 256x256x32
    list(GET config_groups 1 warp_dims)      # e.g., 4x1x1
    list(GET config_groups 2 warp_tile_dims) # e.g., 16x16x16
    
    # Parse tile dimensions
    string(REPLACE "x" ";" tile_parts ${tile_dims})
    list(GET tile_parts 0 tile_m)
    list(GET tile_parts 1 tile_n)
    list(GET tile_parts 2 tile_k)
    
    # Parse warp dimensions
    string(REPLACE "x" ";" warp_parts ${warp_dims})
    list(GET warp_parts 0 warp_m)
    list(GET warp_parts 1 warp_n)
    list(GET warp_parts 2 warp_k)
    
    # Parse warp tile dimensions
    string(REPLACE "x" ";" warp_tile_parts ${warp_tile_dims})
    list(GET warp_tile_parts 0 warp_tile_m)
    list(GET warp_tile_parts 1 warp_tile_n)
    list(GET warp_tile_parts 2 warp_tile_k)
    
    set(target_name "benchmark_gemm_${datatype}_${layout}_${trait}_${tile_config}")
    set(working_path "${CMAKE_CURRENT_BINARY_DIR}/${datatype}/${layout}")
    
    # Generate the single instance header for this kernel
    set(instance_header "${working_path}/gemm_single_${datatype}_${layout}_${trait}_${tile_config}.hpp")
    
    # Create the executable
    add_executable(${target_name} 
        ${GEMM_SOURCE_DIR}/benchmark_gemm_single.cpp
    )
    
    # Set GPU architectures
    set_property(TARGET ${target_name} PROPERTY HIP_ARCHITECTURES ${GEMM_GPU_TARGETS_INDIVIDUAL})
    
    # Set compile definitions
    target_compile_definitions(${target_name} PRIVATE
        GEMM_SINGLE_INSTANCE_HPP="${instance_header}"
    )
    
    # Include directories
    target_include_directories(${target_name} PRIVATE
        ${GEMM_SOURCE_DIR}
        ${working_path}
    )
    
    # Compile options
    target_compile_options(${target_name} PRIVATE
        -Wno-undefined-func-template
        -Wno-float-equal
        --offload-compress
        -include ${instance_header}
    )
    
    # Add to collection targets
    add_dependencies(benchmark_gemm_all ${target_name})
    add_dependencies(benchmark_gemm_${datatype} ${target_name})
    add_dependencies(benchmark_gemm_${layout} ${target_name})
    add_dependencies(benchmark_gemm_${datatype}_${layout} ${target_name})
    
    # Add to trait-specific targets
    string(REPLACE "_" ";" trait_parts ${trait})
    list(GET trait_parts 0 pipeline)
    list(GET trait_parts 1 epilogue)
    list(GET trait_parts 2 scheduler)
    
    add_dependencies(benchmark_gemm_${pipeline} ${target_name})
    add_dependencies(benchmark_gemm_${epilogue} ${target_name})
    add_dependencies(benchmark_gemm_${scheduler} ${target_name})
endfunction()

# Function to generate dispatcher
function(generate_dispatcher)
    # Include the dispatcher generation script
    include(${CMAKE_CURRENT_LIST_DIR}/generate_dispatcher.cmake)
    
    # Create dispatcher library
    add_library(gemm_dispatcher STATIC
        ${CMAKE_CURRENT_BINARY_DIR}/gemm_dispatcher.cpp
    )
    
    target_include_directories(gemm_dispatcher PUBLIC
        ${CMAKE_CURRENT_LIST_DIR}
        ${CMAKE_CURRENT_BINARY_DIR}
    )
    
    # Link with all generated instances
    foreach(dt IN LISTS GEMM_DATATYPE)
        foreach(l IN LISTS GEMM_LAYOUT)
            if(TARGET gemm_template_instances_${dt}_${l})
                target_link_libraries(gemm_dispatcher PUBLIC gemm_template_instances_${dt}_${l})
            endif()
        endforeach()
    endforeach()
endfunction()

# Function to build individual GEMM targets
function(build_individual_gemm_targets datatype layout)
    set(working_path "${CMAKE_CURRENT_BINARY_DIR}/${datatype}/${layout}")
    
    # Choose config file
    if(layout STREQUAL "rcr")
        set(json_blob "${CMAKE_CURRENT_LIST_DIR}/configs/default_config.json")
    else()
        set(json_blob "${CMAKE_CURRENT_LIST_DIR}/configs/custom_ci_config.json")
    endif()
    
    # Determine number of workers for parallel generation
    if(DEFINED ENV{CMAKE_BUILD_PARALLEL_LEVEL})
        set(num_workers $ENV{CMAKE_BUILD_PARALLEL_LEVEL})
    else()
        # Use processor count but limit to avoid memory issues
        cmake_host_system_information(RESULT num_cores QUERY NUMBER_OF_LOGICAL_CORES)
        math(EXPR num_workers "${num_cores}")
        if(num_workers GREATER 8)
            set(num_workers 8)
        endif()
    endif()
    
    # Generate individual kernel files using parallel version
    execute_process(
        COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_LIST_DIR}/gemm_instance_builder_parallel.py
                --working_path ${working_path}
                --datatype ${datatype}
                --layout ${layout}
                --config_json ${json_blob}
                --num_workers ${num_workers}
                --gen_individual
        RESULT_VARIABLE ret
    )
    if(NOT ret EQUAL 0)
        message(FATAL_ERROR "Failed to generate individual kernels for ${datatype} ${layout}: ${ret}")
    endif()
    
    # Include the generated CMake file
    include(${working_path}/gemm_individual_targets.cmake)
endfunction()

# Main build logic
if(GEMM_BUILD_MONOLITHIC)
    # Process each datatype in isolation for monolithic build
    foreach(dt IN LISTS GEMM_DATATYPE)
        foreach(l IN LISTS GEMM_LAYOUT)
            build_gemm_for_datatype(${dt} ${l})
        endforeach()
    endforeach()
endif()

if(GEMM_BUILD_INDIVIDUAL)
    # Filter GPU targets to only gfx90a, gfx942, and gfx950 for individual builds
    set(GEMM_GPU_TARGETS_INDIVIDUAL "")
    set(DESIRED_TARGETS "gfx90a;gfx942;gfx950")
    
    foreach(target IN LISTS SUPPORTED_GPU_TARGETS)
        if(target IN_LIST DESIRED_TARGETS)
            list(APPEND GEMM_GPU_TARGETS_INDIVIDUAL ${target})
        endif()
    endforeach()
    
    # Skip individual build if no matching targets found
    if(NOT GEMM_GPU_TARGETS_INDIVIDUAL)
        message(WARNING "Skipping Tile Engine GEMM individual build: No supported GPU targets (gfx90a, gfx942, gfx950) found in SUPPORTED_GPU_TARGETS: ${SUPPORTED_GPU_TARGETS}")
    else()
        message(STATUS "Building individual GEMM targets for GPU targets: ${GEMM_GPU_TARGETS_INDIVIDUAL}")
    
        # Enable parallel compilation optimizations
        # Set up job pools for better parallel compilation control
        set_property(GLOBAL PROPERTY JOB_POOLS 
            compile_heavy=4    # Limit heavy compilations to prevent OOM
            compile_normal=16  # Allow more parallel normal compilations
        )
    
    # Enable compiler cache if available
    find_program(CCACHE_PROGRAM ccache)
    if(CCACHE_PROGRAM)
        set(CMAKE_CXX_COMPILER_LAUNCHER ${CCACHE_PROGRAM})
        message(STATUS "Using ccache for faster compilation")
    endif()
    
    # Create master collection targets
    add_custom_target(benchmark_gemm_all)
    
    # Create datatype collection targets
    foreach(dt IN LISTS GEMM_DATATYPE)
        add_custom_target(benchmark_gemm_${dt})
    endforeach()
    
    # Create layout collection targets
    foreach(l IN LISTS GEMM_LAYOUT)
        add_custom_target(benchmark_gemm_${l})
    endforeach()
    
    # Create combined collection targets
    foreach(dt IN LISTS GEMM_DATATYPE)
        foreach(l IN LISTS GEMM_LAYOUT)
            add_custom_target(benchmark_gemm_${dt}_${l})
        endforeach()
    endforeach()
    
    # Create trait-based collection targets
    # These are common trait components used across all GEMM kernels
    set(GEMM_PIPELINES "mem;compv3;compv4")
    set(GEMM_EPILOGUES "default;cshuffle")
    set(GEMM_SCHEDULERS "intrawave;interwave")
    
    foreach(pipeline IN LISTS GEMM_PIPELINES)
        add_custom_target(benchmark_gemm_${pipeline})
    endforeach()
    
    foreach(epilogue IN LISTS GEMM_EPILOGUES)
        add_custom_target(benchmark_gemm_${epilogue})
    endforeach()
    
    foreach(scheduler IN LISTS GEMM_SCHEDULERS)
        add_custom_target(benchmark_gemm_${scheduler})
    endforeach()
    
        # Build individual targets for each datatype/layout combination
        foreach(dt IN LISTS GEMM_DATATYPE)
            foreach(l IN LISTS GEMM_LAYOUT)
                build_individual_gemm_targets(${dt} ${l})
            endforeach()
        endforeach()
    endif()
endif()

# Generate dispatcher after all targets are defined
if(GEMM_BUILD_MONOLITHIC OR GEMM_BUILD_INDIVIDUAL)
    generate_dispatcher()
endif()
