# =============================================================================
# Engine Renderer Module - Graphics rendering, terrain, models
# =============================================================================

set(RENDERER_SOURCES
    src/Heightgrid.cpp
    src/Mapgrid.cpp
    src/Texturecache.cpp
    src/Shaders.cpp
    src/Modelnormalize.cpp
    src/main.cpp
    ${ROE_EXTERNALS_DIR}/Glad/glad.c
)

# Create renderer library
add_library(engine_renderer STATIC ${RENDERER_SOURCES})

# Include paths for this module
target_include_directories(engine_renderer PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${ROE_EXTERNALS_DIR}
    ${ROE_EXTERNALS_DIR}/Glad
    ${ROE_EXTERNALS_DIR}/NoiseAlgo
)

# Link external dependencies
target_link_libraries(engine_renderer PUBLIC
    glfw
    assimp::assimp
    glm::glm
)

# Ensure glad is compiled with the right flags
target_compile_options(engine_renderer PRIVATE)

# Make assets and resources available as defines
target_compile_definitions(engine_renderer PRIVATE
    ASSETS_PATH="${ROE_ASSETS_DIR}"
    RESOURCES_PATH="${ROE_RESOURCES_DIR}"
)

# =============================================================================
# MAIN EXECUTABLE - Renderer Demo
# =============================================================================

# Create the main terrain renderer executable
add_executable(TerrainRenderer
    ${RENDERER_SOURCES}
)

target_include_directories(TerrainRenderer PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${ROE_EXTERNALS_DIR}
    ${ROE_EXTERNALS_DIR}/Glad
    ${ROE_EXTERNALS_DIR}/NoiseAlgo
)

target_link_libraries(TerrainRenderer PRIVATE
    glfw
    assimp::assimp
    glm::glm
)

target_compile_definitions(TerrainRenderer PRIVATE
    ASSETS_PATH="${ROE_ASSETS_DIR}"
    RESOURCES_PATH="${ROE_RESOURCES_DIR}"
)

# =============================================================================
# NOTE: Run the executable from repo root so relative paths to assets work:
#   cd /home/faisal/PROJECT/1/ROE
#   ./build/TerrainRenderer
# =============================================================================
