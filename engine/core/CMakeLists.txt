# =============================================================================
# Engine Core Module - Map generation and world logic
# =============================================================================

set(CORE_SOURCES
    src/Mapgenerator.cpp
    src/main.cpp
)

set(CORE_HEADERS
    ${CMAKE_CURRENT_SOURCE_DIR}/../include
)

# Create core library
add_library(engine_core STATIC ${CORE_SOURCES})

# Include paths for this module
target_include_directories(engine_core PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${ROE_EXTERNALS_DIR}
    ${ROE_EXTERNALS_DIR}/NoiseAlgo
)

# Link external dependencies
target_link_libraries(engine_core PUBLIC
    glm::glm
)

# Add header-only includes from parent include directory
target_include_directories(engine_core PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/../include
)
