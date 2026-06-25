// =============================================================================
// HeightGrid.h   (ADDED FOR TERRAIN HEIGHT SYSTEM)
//
//   Lightweight 2D height-field representation of the terrain mesh, sampled
//   in WORLD SPACE (i.e. after the normalization matrix -- scale + ground
//   snap -- has been applied to the raw vertex positions). This lets
//   getTerrainHeight() operate in the same coordinate space the model is
//   actually rendered in, without needing to touch the render loop.
//
//   height[x][z] stores the MAXIMUM world-space Y found in that cell, per
//   the task spec (highest terrain point wins -- conservative for grounding
//   on top of overlapping terrain layers).
//
//   CURRENT SCOPE: BuildHeightGrid() populates g_TerrainHeightGrid from a
//   single model's scene + world matrix (the "origin model" in main.cpp).
//   This is the file to hand to an AI assistant when extending to a
//   multi-chunk / whole-map global grid -- see the header comment on
//   BuildHeightGrid() below for the current single-model contract that
//   any extension needs to preserve or explicitly replace.
// =============================================================================
#pragma once

#include <glm/glm.hpp>
#include <assimp/scene.h>
#include <vector>

// =============================================================================
// HeightGrid
// =============================================================================
struct HeightGrid
{
    static const int GRID_RESOLUTION = 256;

    std::vector<std::vector<float>> height; // height[x][z]

    float minX = 0.0f, maxX = 0.0f;
    float minZ = 0.0f, maxZ = 0.0f;
    float cellSizeX = 1.0f;
    float cellSizeZ = 1.0f;
    bool valid = false;

    HeightGrid()
    {
        height.assign(
            GRID_RESOLUTION,
            std::vector<float>(GRID_RESOLUTION, 0.0f));
    }
};

// Global instance -- mirrors the style of g_TextureCache (single global
// resource). Populated once after model load, read every frame by
// getTerrainHeight().
extern HeightGrid g_TerrainHeightGrid;

// -----------------------------------------------------------------------------
// BuildHeightGrid
//
//   Samples terrain-classified meshes (via IsLikelyTerrainMesh, declared in
//   ModelNormalize.h) from `scene` into `outGrid`, applying `worldMatrix`
//   (the same normalization matrix used as the render "model" uniform) to
//   each raw vertex so the grid lives in true world space.
//
//   For each vertex: compute its world-space XZ, find which grid cell it
//   falls into, and keep the MAX world-space Y seen for that cell.
//   Cells with no samples fall back to the lowest recorded height (0 after
//   ground-snap) so getTerrainHeight() never returns an uninitialized value.
//
//   NOTE: this currently builds the grid from ONE scene/worldMatrix pair
//   (intended for one "origin" model). It does not yet aggregate multiple
//   placed chunks into one global grid.
// -----------------------------------------------------------------------------
void BuildHeightGrid(
    const aiScene *scene,
    const glm::mat4 &worldMatrix,
    HeightGrid &outGrid);

// -----------------------------------------------------------------------------
// getTerrainHeight
//
//   Converts world (X, Z) into the height grid's local cell space and
//   returns a bilinearly-interpolated terrain Y. Works against whichever
//   transform was baked into g_TerrainHeightGrid by BuildHeightGrid.
//
//   Returns 0.0f if the grid was never successfully built (e.g. no terrain
//   mesh identified), which matches the default ground-snap plane.
// -----------------------------------------------------------------------------
float getTerrainHeight(float worldX, float worldZ);