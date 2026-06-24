cmake_minimum_required(VERSION 3.16)
project(ROE)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Top-level source layout: the existing sample renderer lives in 3DModels/main.cpp
# and uses Externals/Glad/glad.c. Adjust sources as you modularize the codebase.
set(SOURCES
    3DModels/main.cpp
    Externals/Glad/glad.c
)

add_executable(app ${SOURCES})

target_include_directories(app PRIVATE
    ${CMAKE_SOURCE_DIR}/Externals/Glad
    ${CMAKE_SOURCE_DIR}/Externals
)

# Link against system libraries; prefer configuring via find_package in future.
target_link_libraries(app PRIVATE
    glfw
    GL
    dl
    assimp
)

message(STATUS "Configured app target. Modify CMakeLists as you split modules.")
