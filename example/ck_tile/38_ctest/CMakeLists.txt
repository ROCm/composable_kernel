# Define the path to your script
# Using CMAKE_CURRENT_SOURCE_DIR assumes the script is in the same directory as CMakeLists.txt
set(CTEST_SCRIPT "${CMAKE_CURRENT_SOURCE_DIR}/ctest_script.sh")

# Add a test that runs the bash script
# The COMMAND argument specifies what CTest should execute.
# We pass "success" as an argument to the script to make this test pass.
add_test(
  NAME CBashScriptTest_Success
  COMMAND ${CMAKE_COMMAND} -E chdir ${CMAKE_CURRENT_BINARY_DIR} ${CTEST_SCRIPT} success
  # Alternative if you don't need to change directory or pass specific args, just direct execution:
  # COMMAND ${CTEST_SCRIPT} success
  # Or with arguments:
  # COMMAND ${CTEST_SCRIPT} arg1 arg2
)

# Add another test that runs the script and expects it to fail
add_test(
  NAME CBashScriptTest_Failure
  COMMAND ${CMAKE_COMMAND} -E chdir ${CMAKE_CURRENT_BINARY_DIR} ${CTEST_SCRIPT} failure
)

# Optional: Add a custom command to copy the script to the build directory
# This ensures the script is available where CTest runs it, especially useful
# if you are defining your test COMMAND relative to CMAKE_CURRENT_BINARY_DIR.
add_custom_command(
    OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/ctest_script.sh
    COMMAND ${CMAKE_COMMAND} -E copy_if_different ${CTEST_SCRIPT} ${CMAKE_CURRENT_BINARY_DIR}/ctest_script.sh
    DEPENDS ${CTEST_SCRIPT}
    COMMENT "Copying ctest_script.sh to build directory"
)

# Add a custom target that depends on the script being copied
# This ensures the copy command runs before the tests are configured/run.
add_custom_target(prepare_scripts ALL DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/ctest_script.sh)

# Ensure tests depend on the script preparation
add_dependencies(CBashScriptTest_Success prepare_scripts)
add_dependencies(CBashScriptTest_Failure prepare_scripts)
