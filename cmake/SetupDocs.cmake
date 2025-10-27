function(setup_documentation)
    # Find Python executable
    find_package(Python3 REQUIRED COMPONENTS Interpreter)
    
    # Check for existing virtual environment in ~/python-env
    set(HOME_VENV "$ENV{HOME}/python-env/venv-1")
    set(BUILD_VENV "${CMAKE_BINARY_DIR}/python-env")
    
    if(EXISTS "${HOME_VENV}/bin/python3")
        set(PYTHON_VENV "${HOME_VENV}")
        set(PYTHON_EXECUTABLE "${HOME_VENV}/bin/python3")
        set(VENV_IS_NEW FALSE)
        message(STATUS "Using existing Python virtual environment: ${HOME_VENV}")
    else()
        set(PYTHON_VENV "${BUILD_VENV}")
        set(PYTHON_EXECUTABLE "${BUILD_VENV}/bin/python3")
        set(VENV_IS_NEW TRUE)
        message(STATUS "Will create Python virtual environment: ${BUILD_VENV}")
    endif()
    
    # Create virtual environment if it doesn't exist
    if(NOT EXISTS "${PYTHON_EXECUTABLE}")
        message(STATUS "Creating Python virtual environment...")
        execute_process(
            COMMAND ${Python3_EXECUTABLE} -m venv ${PYTHON_VENV}
            RESULT_VARIABLE VENV_RESULT
        )
        if(NOT VENV_RESULT EQUAL 0)
            message(FATAL_ERROR "Failed to create Python virtual environment")
        endif()
        set(VENV_IS_NEW TRUE)
    endif()
    
    # Set requirements file path
    set(REQUIREMENTS_FILE "${CMAKE_SOURCE_DIR}/docs/sphinx/requirements.txt")
    
    # Only install packages if we created a new virtual environment
    if(VENV_IS_NEW)
        message(STATUS "Installing Python documentation requirements in new virtual environment...")
        execute_process(
            COMMAND ${PYTHON_EXECUTABLE} -m pip install --upgrade pip
            RESULT_VARIABLE PIP_RESULT
        )
        if(NOT PIP_RESULT EQUAL 0)
            message(FATAL_ERROR "Failed to upgrade pip")
        endif()
        
        execute_process(
            COMMAND ${PYTHON_EXECUTABLE} -m pip install -r ${REQUIREMENTS_FILE}
            RESULT_VARIABLE PIP_RESULT
        )
        if(NOT PIP_RESULT EQUAL 0)
            message(FATAL_ERROR "Failed to install Python documentation requirements")
        endif()
        
        message(STATUS "Python documentation requirements installed successfully")
    endif()
    
    # Create dummy target (no build-time package installation needed)
    add_custom_target(docs_venv
        COMMENT "Python virtual environment ready"
    )
    
    # Set up Doxygen
    find_package(Doxygen REQUIRED)
    
    set(DOXYGEN_OUTPUT_DIR "${CMAKE_BINARY_DIR}/docs/doxygen")
    set(DOXYGEN_CONFIG_FILE "${CMAKE_SOURCE_DIR}/docs/doxygen/Doxyfile")
    
    add_custom_command(
        OUTPUT "${DOXYGEN_OUTPUT_DIR}/xml/index.xml"
        COMMAND ${CMAKE_COMMAND} -E make_directory ${DOXYGEN_OUTPUT_DIR}
        COMMAND ${DOXYGEN_EXECUTABLE} ${DOXYGEN_CONFIG_FILE}
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/docs/doxygen
        DEPENDS ${DOXYGEN_CONFIG_FILE}
        COMMENT "Building Doxygen XML documentation"
    )
    
    add_custom_target(docs_doxygen
        DEPENDS "${DOXYGEN_OUTPUT_DIR}/xml/index.xml"
    )
    
    # Set up Sphinx
    set(SPHINX_OUTPUT_DIR "${CMAKE_BINARY_DIR}/docs/html")
    set(SPHINX_SOURCE_DIR "${CMAKE_SOURCE_DIR}/docs")
    
    # Set number of parallel jobs for Sphinx.
    # Use 8 as a reasonable default for modern multi-core machines if not set by the user.
    set(SPHINX_DEFAULT_PARALLEL_JOBS 8)
    if(NOT CMAKE_BUILD_PARALLEL_LEVEL)
        set(CMAKE_BUILD_PARALLEL_LEVEL ${SPHINX_DEFAULT_PARALLEL_JOBS})
    endif()
    
    add_custom_target(docs
        COMMAND ${PYTHON_EXECUTABLE} -m sphinx -b html -j ${CMAKE_BUILD_PARALLEL_LEVEL} ${SPHINX_SOURCE_DIR} ${SPHINX_OUTPUT_DIR}
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
        DEPENDS docs_venv docs_doxygen
        COMMENT "Building HTML documentation with Sphinx"
    )
    
    # Add clean target for docs
    add_custom_target(docs_clean
        COMMAND ${CMAKE_COMMAND} -E remove_directory ${CMAKE_BINARY_DIR}/docs
        COMMENT "Cleaning documentation build files"
    )
    
    message(STATUS "Documentation build configured:")
    message(STATUS "  Build command: cmake --build . --target docs")
    message(STATUS "  Clean command: cmake --build . --target docs_clean")
    message(STATUS "  Output directory: ${SPHINX_OUTPUT_DIR}")
endfunction()