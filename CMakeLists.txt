cmake_minimum_required(VERSION 3.14)
project(ClassicML LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Header-only core exposed as a shared library for the Python bridge.
add_library(ml_core SHARED src/classicml_lib.cpp)
target_include_directories(ml_core PUBLIC ${CMAKE_SOURCE_DIR})

# Demo executables (auto-discovered)
file(GLOB DEMO_FILES "${CMAKE_SOURCE_DIR}/demos/*_demo.cpp")
foreach(DEMO ${DEMO_FILES})
    get_filename_component(DEMO_NAME ${DEMO} NAME_WE)
    add_executable(${DEMO_NAME} ${DEMO})
    target_include_directories(${DEMO_NAME} PRIVATE ${CMAKE_SOURCE_DIR} ${CMAKE_SOURCE_DIR}/demos)
    target_link_libraries(${DEMO_NAME} PRIVATE ml_core)
endforeach()

# Optional pybind11 Python module
find_package(pybind11 CONFIG QUIET)
if(pybind11_FOUND)
    pybind11_add_module(classicml_py python/bindings.cpp)
    target_include_directories(classicml_py PRIVATE ${CMAKE_SOURCE_DIR})
    target_link_libraries(classicml_py PRIVATE ml_core)
endif()
