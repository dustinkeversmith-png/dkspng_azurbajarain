cmake_minimum_required(VERSION 3.20)
project(forward_analysis CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# 1. Locate Eigen3 package (works if installed via vcpkg, brew, apt, or system)
find_package(Eigen3 3.3 QUIET NO_MODULE)

if(NOT Eigen3_FOUND)
    include(FetchContent)
    FetchContent_Declare(
        eigen3_headers
        URL https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.tar.gz
        SOURCE_SUBDIR cmake_header_only_noop
        DOWNLOAD_EXTRACT_TIMESTAMP TRUE
    )
    FetchContent_MakeAvailable(eigen3_headers)
    set(EIGEN3_INCLUDE_DIRS "${eigen3_headers_SOURCE_DIR}")
    if(NOT TARGET Eigen3::Eigen)
        add_library(Eigen3::Eigen INTERFACE IMPORTED)
        set_target_properties(Eigen3::Eigen PROPERTIES
            INTERFACE_INCLUDE_DIRECTORIES "${EIGEN3_INCLUDE_DIRS}"
        )
    endif()
endif()

# 2. Locate nlohmann/json for importing LLM-generated JSONL training datasets
find_package(nlohmann_json 3.2.0 QUIET)

if(NOT nlohmann_json_FOUND)
    include(FetchContent)
    set(JSON_Install OFF CACHE INTERNAL "")
    set(JSON_BuildTests OFF CACHE INTERNAL "")
    FetchContent_Declare(
        nlohmann_json
        URL https://github.com/nlohmann/json/releases/download/v3.11.3/json.tar.xz
        DOWNLOAD_EXTRACT_TIMESTAMP TRUE
    )
    FetchContent_MakeAvailable(nlohmann_json)
endif()

# 3. OpenCL + CLBlast acceleration for decoder training and evaluation (native C++, runtime-compiled OpenCL C kernels)
option(FORWARD_ANALYSIS_ENABLE_OPENCL "Build the OpenCL/CLBlast device path for decoder training" ON)

if(FORWARD_ANALYSIS_ENABLE_OPENCL)
    include(FetchContent)
    FetchContent_Declare(
        opencl_headers
        URL https://github.com/KhronosGroup/OpenCL-Headers/archive/refs/tags/v2024.10.24.tar.gz
        DOWNLOAD_EXTRACT_TIMESTAMP TRUE
    )
    FetchContent_MakeAvailable(opencl_headers)

    set(OPENCL_ICD_LOADER_BUILD_TESTING OFF CACHE BOOL "" FORCE)
    set(BUILD_TESTING OFF CACHE BOOL "" FORCE)
    FetchContent_Declare(
        opencl_icd_loader
        URL https://github.com/KhronosGroup/OpenCL-ICD-Loader/archive/refs/tags/v2024.10.24.tar.gz
        DOWNLOAD_EXTRACT_TIMESTAMP TRUE
    )
    FetchContent_MakeAvailable(opencl_icd_loader)

    # CLBlast's FindOpenCL keeps these when already set: headers and the ICD loader target fetched above.
    set(OPENCL_INCLUDE_DIRS "$<BUILD_INTERFACE:${opencl_headers_SOURCE_DIR}>" CACHE STRING "" FORCE)
    set(OPENCL_LIBRARIES OpenCL CACHE STRING "" FORCE)
    set(OPENCL ON CACHE BOOL "" FORCE)
    set(CUDA OFF CACHE BOOL "" FORCE)
    set(TUNERS OFF CACHE BOOL "" FORCE)
    set(CLIENTS OFF CACHE BOOL "" FORCE)
    set(SAMPLES OFF CACHE BOOL "" FORCE)
    set(TESTS OFF CACHE BOOL "" FORCE)
    set(NETLIB OFF CACHE BOOL "" FORCE)
    FetchContent_Declare(
        clblast
        URL https://github.com/CNugteren/CLBlast/archive/refs/tags/1.6.3.tar.gz
        DOWNLOAD_EXTRACT_TIMESTAMP TRUE
    )
    set(CMAKE_POLICY_VERSION_MINIMUM 3.5)
    FetchContent_MakeAvailable(clblast)
    unset(CMAKE_POLICY_VERSION_MINIMUM)
endif()

add_library(forward_analysis INTERFACE)

# 3. Add include paths for both forward_analysis and Eigen
target_include_directories(forward_analysis INTERFACE
    "${CMAKE_CURRENT_SOURCE_DIR}/include"
    ${EIGEN3_INCLUDE_DIRS}
)

# 4. Link the Eigen and nlohmann/json targets
target_link_libraries(forward_analysis INTERFACE
    Eigen3::Eigen
    nlohmann_json::nlohmann_json
)

if(FORWARD_ANALYSIS_ENABLE_OPENCL)
    target_link_libraries(forward_analysis INTERFACE clblast OpenCL::OpenCL)
    target_compile_definitions(forward_analysis INTERFACE FORWARD_ANALYSIS_WITH_OPENCL=1 CL_TARGET_OPENCL_VERSION=120)
endif()

# 5. AVX2 + FMA code generation for the CPU-side Eigen math (encoder, heads); the binaries then need a CPU with AVX2.
option(FORWARD_ANALYSIS_ENABLE_AVX2 "Compile the CPU math with AVX2 and FMA instructions" ON)

if(FORWARD_ANALYSIS_ENABLE_AVX2)
    if(MSVC)
        target_compile_options(forward_analysis INTERFACE /arch:AVX2)
    else()
        target_compile_options(forward_analysis INTERFACE -mavx2 -mfma)
    endif()
endif()

set(FORWARD_ANALYSIS_DATA_DIR "${CMAKE_CURRENT_SOURCE_DIR}/data")
set(FORWARD_ANALYSIS_ARTIFACT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/artifacts")

function(forward_analysis_add_executable target source)
    add_executable(${target} ${source})
    target_link_libraries(${target} PRIVATE forward_analysis)
    target_compile_definitions(${target} PRIVATE
        FORWARD_ANALYSIS_DATA_DIR="${FORWARD_ANALYSIS_DATA_DIR}"
        FORWARD_ANALYSIS_ARTIFACT_DIR="${FORWARD_ANALYSIS_ARTIFACT_DIR}"
    )
    if(MSVC)
        target_compile_options(${target} PRIVATE /O2 /EHsc /W4 /bigobj)
    else()
        target_compile_options(${target} PRIVATE -O2 -Wall -Wextra -Wpedantic)
    endif()
    if(WIN32 AND FORWARD_ANALYSIS_ENABLE_OPENCL)
        # clblast.dll and OpenCL.dll (ICD loader) next to the executable.
        add_custom_command(TARGET ${target} POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different $<TARGET_RUNTIME_DLLS:${target}> $<TARGET_FILE_DIR:${target}>
            COMMAND_EXPAND_LISTS)
    endif()
endfunction()

option(FORWARD_ANALYSIS_BUILD_TESTS "Build forward_analysis standalone verification suite" ON)
option(FORWARD_ANALYSIS_BUILD_DEMOS "Build forward_analysis end-to-end pipeline demos" ON)

if(FORWARD_ANALYSIS_BUILD_TESTS)
    enable_testing()
    forward_analysis_add_executable(test_forward_analysis tests/test_forward_analysis.cpp)
    target_compile_definitions(test_forward_analysis PRIVATE
        FORWARD_ANALYSIS_FIXTURE_DIR="${CMAKE_CURRENT_SOURCE_DIR}/tests/fixtures")
    add_test(NAME test_forward_analysis COMMAND test_forward_analysis)
endif()

if(FORWARD_ANALYSIS_BUILD_DEMOS)
    forward_analysis_add_executable(forward_pipeline_demo demos/forward_pipeline_demo.cpp)
    forward_analysis_add_executable(forward_experiment experiments/forward_experiment.cpp)
endif()
