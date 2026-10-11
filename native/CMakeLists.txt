cmake_minimum_required(VERSION 3.18)
project(native_engine LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

find_package(Python3 COMPONENTS Interpreter Development.Module REQUIRED)
find_package(pybind11 CONFIG REQUIRED)

set(NATIVE_INCLUDE "${CMAKE_CURRENT_SOURCE_DIR}/include")
set(NATIVE_ENGINE_OUTPUT "${CMAKE_CURRENT_SOURCE_DIR}/../build/native")

function(add_native_module target source)
    pybind11_add_module(${target} ${source})
    target_include_directories(${target} PRIVATE "${NATIVE_INCLUDE}")
    if(WIN32)
        target_compile_definitions(${target} PRIVATE NOMINMAX _hypot=hypot)
    endif()
    if(MSVC)
        target_compile_options(${target} PRIVATE /O2 /EHsc /W3)
    else()
        target_compile_options(${target} PRIVATE -O3 -Wall -Wextra -DNDEBUG)
    endif()
    if(MINGW)
        target_link_options(${target} PRIVATE -static)
    elseif(NOT WIN32)
        target_link_libraries(${target} PRIVATE Threads::Threads)
    endif()
    set_target_properties(${target} PROPERTIES
        OUTPUT_NAME ${target}
        RUNTIME_OUTPUT_DIRECTORY "${NATIVE_ENGINE_OUTPUT}"
        RUNTIME_OUTPUT_DIRECTORY_RELEASE "${NATIVE_ENGINE_OUTPUT}"
        RUNTIME_OUTPUT_DIRECTORY_DEBUG "${NATIVE_ENGINE_OUTPUT}"
        LIBRARY_OUTPUT_DIRECTORY "${NATIVE_ENGINE_OUTPUT}"
        LIBRARY_OUTPUT_DIRECTORY_RELEASE "${NATIVE_ENGINE_OUTPUT}"
        LIBRARY_OUTPUT_DIRECTORY_DEBUG "${NATIVE_ENGINE_OUTPUT}"
    )
endfunction()

if(NOT WIN32)
    find_package(Threads REQUIRED)
endif()

add_native_module(native_engine bindings/bind_processor.cpp)
add_native_module(tensor_engine bindings/bind_tensor_engine.cpp)
if(WIN32)
    target_link_libraries(tensor_engine PRIVATE winhttp bcrypt crypt32)
endif()
