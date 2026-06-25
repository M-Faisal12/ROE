# =============================================================================
# Engine Module - Organizes core and renderer submodules
# =============================================================================

# Add all engine submodules
add_subdirectory(core)
add_subdirectory(renderer)

# Optional: Create an engine library that others can link against
# This allows game/ and tools/ to easily depend on engine components
add_library(engine INTERFACE)

target_link_libraries(engine INTERFACE
    engine_core
    engine_renderer
)

target_include_directories(engine INTERFACE
    ${CMAKE_CURRENT_SOURCE_DIR}/core/include
    ${CMAKE_CURRENT_SOURCE_DIR}/renderer/include
)
