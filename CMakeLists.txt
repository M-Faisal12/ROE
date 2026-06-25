cmake_minimum_required(VERSION 3.16)
project(ROE)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# =============================================================================
# SHARED CONFIGURATION
# =============================================================================

# Define root directory for accessing assets and resources from anywhere
set(ROE_ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}")
set(ROE_ASSETS_DIR "${ROE_ROOT_DIR}/assets")
set(ROE_RESOURCES_DIR "${ROE_ROOT_DIR}/engine/resources")
set(ROE_EXTERNALS_DIR "${ROE_ROOT_DIR}/Externals")

# Make these available to all subdirectories via preprocessor defines
# Export as quoted string literals so source can use them as C++ string literals
add_definitions(-DROE_ASSETS_PATH=\"${ROE_ASSETS_DIR}\")
add_definitions(-DROE_RESOURCES_PATH=\"${ROE_RESOURCES_DIR}\")

# =============================================================================
# GLOBAL DEPENDENCIES
# =============================================================================

find_package(glfw3 REQUIRED)
find_package(assimp REQUIRED)
find_package(glm REQUIRED)

# =============================================================================
# GLOBAL INCLUDE DIRECTORIES (accessible to all targets)
# =============================================================================

include_directories(
    ${ROE_EXTERNALS_DIR}
    ${ROE_EXTERNALS_DIR}/Glad
    ${ROE_EXTERNALS_DIR}/NoiseAlgo
)

# =============================================================================
# ADD ENGINE SUBDIRECTORIES
# =============================================================================

add_subdirectory(engine)
