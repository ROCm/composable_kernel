function(AssureOutOfSourceBuilds)
  get_filename_component(srcdir "${CMAKE_SOURCE_DIR}" REALPATH)
  get_filename_component(bindir "${CMAKE_BINARY_DIR}" REALPATH)

  if("${srcdir}" STREQUAL "${bindir}")
    message("Warning: in-source builds are disabled")
    message("Please create a separate build directory and run cmake from there")
    message(FATAL_ERROR "Quitting configuration")
  endif()
endfunction()

assureOutOfSourceBuilds()
