cmake_minimum_required(VERSION 3.14)
project(ClassicML LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_library(ml_core STATIC
    utils.cpp
    ino.cpp
    models.cpp
)
target_include_directories(ml_core PUBLIC ${CMAKE_SOURCE_DIR})
target_compile_options(ml_core PRIVATE -Wall -Wextra)

file(GLOB DEMO_FILES "${CMAKE_SOURCE_DIR}/demos/*_demo.cpp")
foreach(DEMO ${DEMO_FILES})
    get_filename_component(DEMO_NAME ${DEMO} NAME_WE)
    add_executable(${DEMO_NAME} ${DEMO})
    target_include_directories(${DEMO_NAME} PRIVATE ${CMAKE_SOURCE_DIR} ${CMAKE_SOURCE_DIR}/demos)
    target_link_libraries(${DEMO_NAME} PRIVATE ml_core)
    target_compile_options(${DEMO_NAME} PRIVATE -Wall -Wextra)
endforeach()
