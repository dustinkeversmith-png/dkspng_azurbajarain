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
        target_compile_options(${target} PRIVATE /O2 /EHsc /W4)
    else()
        target_compile_options(${target} PRIVATE -O2 -Wall -Wextra -Wpedantic)
    endif()
endfunction()

option(FORWARD_ANALYSIS_BUILD_TESTS "Build forward_analysis standalone verification suite" ON)
option(FORWARD_ANALYSIS_BUILD_DEMOS "Build forward_analysis end-to-end pipeline demos" ON)

if(FORWARD_ANALYSIS_BUILD_TESTS)
    enable_testing()
    forward_analysis_add_executable(test_forward_analysis tests/test_forward_analysis.cpp)
    add_test(NAME test_forward_analysis COMMAND test_forward_analysis)
endif()

if(FORWARD_ANALYSIS_BUILD_DEMOS)
    forward_analysis_add_executable(forward_pipeline_demo demos/forward_pipeline_demo.cpp)
endif()
