set(GEMM_BASIC tile_example_gemm_basic)
set(GEMM_UNIVERSAL tile_example_gemm_universal)

add_example_executable_no_testing(${GEMM_BASIC} gemm_basic.cpp)
# add_example_executable_no_testing(${GEMM_UNIVERSAL} universal_gemm.cpp)

# link options
set(EXAMPLE_GEMM_COMPILE_OPTIONS)
if(CK_USE_OCP_FP8)
  list(APPEND EXAMPLE_GEMM_COMPILE_OPTIONS -DCK_TILE_USE_OCP_FP8)
endif()
list(APPEND EXAMPLE_GEMM_COMPILE_OPTIONS -mllvm -enable-noalias-to-md-conversion=0)
target_compile_options(${GEMM_BASIC} PRIVATE ${EXAMPLE_GEMM_COMPILE_OPTIONS})
# target_compile_options(${GEMM_UNIVERSAL} PRIVATE ${EXAMPLE_GEMM_COMPILE_OPTIONS})

# TODO function should be moved to example/CMakeLists.txt
  
# Defines a function to add a script-based example and test.  
#  
# This function automates the process of:  
#  1. Creating a custom target for the example.  
#  2. Copying a test script to the build output directory at build time.  
#  3. Adding a CTest test that executes the copied script.  
#  4. Setting up dependencies for 'examples' and 'check' targets.  
#  5. Adding an installation rule for the example target.  
#  
# @param TEST_NAME         The name for the test and the custom target (e.g., "gemm_basic_example").  
# @param SCRIPT_NAME            The filename of the script (e.g., "smoke_test_basic.sh").  
# @param SCRIPT_SOURCE_DIR The source directory of the script, relative to CMAKE_CURRENT_SOURCE_DIR.  
# @param SCRIPT_DEST_DIR   The destination directory for the script, relative to CMAKE_RUNTIME_OUTPUT_DIRECTORY.  
#  
function(add_example_with_script TEST_NAME SCRIPT_NAME SCRIPT_SOURCE_DIR SCRIPT_DEST_DIR)  
  
    # Construct the full paths for the source and destination scripts  
    set(SOURCE_SCRIPT_PATH "${CMAKE_CURRENT_SOURCE_DIR}/${SCRIPT_SOURCE_DIR}/${SCRIPT_NAME}")  
    set(DEST_SCRIPT_DIR "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/${SCRIPT_DEST_DIR}")  
    set(DEST_SCRIPT_PATH "${DEST_SCRIPT_DIR}/${SCRIPT_NAME}")  
  
    # Check if the source script exists to prevent configuration errors  
    if(NOT EXISTS "${SOURCE_SCRIPT_PATH}")  
        message(FATAL_ERROR "Script file does not exist: ${SOURCE_SCRIPT_PATH}")  
    endif()  
  
    # 1. TEST_NAME should be an existing target
	if(NOT TARGET ${TEST_NAME})  
		message(FATAL_ERROR "FAILURE: The target '${TEST_NAME}' does not exist.")  
	endif() 
  
    # 2. Add a command that runs before the target is "built".  
    #    This command copies the script to its destination in the build directory.  
    add_custom_command(  
        TARGET ${TEST_NAME}  
        PRE_BUILD  
        # First, ensure the destination directory exists  
        COMMAND ${CMAKE_COMMAND} -E make_directory "${DEST_SCRIPT_DIR}"  
        # Then, copy the script file  
        COMMAND ${CMAKE_COMMAND} -E copy  
                "${SOURCE_SCRIPT_PATH}"  
                "${DEST_SCRIPT_PATH}"  
        COMMENT "Copying script for ${TEST_NAME}"  
    )  
  
    # 3. Add the test case to CTest.  
    #    It runs the script from its location in the build directory.  
    add_test(  
        NAME ${TEST_NAME}  
        COMMAND "${DEST_SCRIPT_PATH}"  
        WORKING_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}  
    )  
  
    # 4. Add dependencies to global targets (assuming they exist).  
    #    This ensures that running 'make examples' or 'make check' builds this test.  
    add_dependencies(examples ${TEST_NAME})  
    add_dependencies(check ${TEST_NAME})  
  
    # 5. Add installation rule using the provided rocm_install command.  
    #    This will install the custom target.  
    rocm_install(TARGETS ${TEST_NAME} COMPONENT examples)  
  
endfunction()  

add_example_with_script(${GEMM_BASIC} smoke_test_basic.sh script script/example/03_gemm)
