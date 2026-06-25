// =============================================================================
// HeightGrid.cpp   (ADDED FOR TERRAIN HEIGHT SYSTEM)
//
//   Implementation of BuildHeightGrid() and getTerrainHeight().
//   See HeightGrid.h for the system-level contract and current
//   single-model scope.
// =============================================================================
#include "HeightGrid.h"
#include "ModelNormalize.h" // IsLikelyTerrainMesh
#include "Logger.h"

#include <algorithm>
#include <cmath>

HeightGrid g_TerrainHeightGrid;

// -----------------------------------------------------------------------------
// BuildHeightGrid
// -----------------------------------------------------------------------------
void BuildHeightGrid(
    const aiScene *scene,
    const glm::mat4 &worldMatrix,
    HeightGrid &outGrid)
{
    glm::vec3 worldMin(1e30f, 1e30f, 1e30f);
    glm::vec3 worldMax(-1e30f, -1e30f, -1e30f);

    bool anyTerrainMeshFound = false;
    unsigned int terrainMeshCount = 0;

    // First pass: world-space AABB over terrain-classified meshes only.
    for (unsigned int mi = 0; mi < scene->mNumMeshes; ++mi)
    {
        const aiMesh *mesh = scene->mMeshes[mi];
        if (!IsLikelyTerrainMesh(mesh))
            continue;

        anyTerrainMeshFound = true;
        terrainMeshCount++;

        for (unsigned int vi = 0; vi < mesh->mNumVertices; ++vi)
        {
            const aiVector3D &v = mesh->mVertices[vi];
            glm::vec4 worldPos = worldMatrix * glm::vec4(v.x, v.y, v.z, 1.0f);

            worldMin.x = std::min(worldMin.x, worldPos.x);
            worldMin.y = std::min(worldMin.y, worldPos.y);
            worldMin.z = std::min(worldMin.z, worldPos.z);
            worldMax.x = std::max(worldMax.x, worldPos.x);
            worldMax.y = std::max(worldMax.y, worldPos.y);
            worldMax.z = std::max(worldMax.z, worldPos.z);
        }
    }

    if (!anyTerrainMeshFound)
    {
        Log("TERRAIN", "No terrain-like mesh identified - height grid left invalid.");
        outGrid.valid = false;
        return;
    }

    outGrid.minX = worldMin.x;
    outGrid.maxX = worldMax.x;
    outGrid.minZ = worldMin.z;
    outGrid.maxZ = worldMax.z;

    float spanX = std::max(outGrid.maxX - outGrid.minX, 0.0001f);
    float spanZ = std::max(outGrid.maxZ - outGrid.minZ, 0.0001f);

    outGrid.cellSizeX = spanX / static_cast<float>(HeightGrid::GRID_RESOLUTION - 1);
    outGrid.cellSizeZ = spanZ / static_cast<float>(HeightGrid::GRID_RESOLUTION - 1);

    // Initialize all cells to the lowest known terrain height (ground-snap
    // means this is typically ~0) so unsampled cells stay sane.
    float fallbackHeight = worldMin.y;
    for (int x = 0; x < HeightGrid::GRID_RESOLUTION; ++x)
        for (int z = 0; z < HeightGrid::GRID_RESOLUTION; ++z)
            outGrid.height[x][z] = fallbackHeight;

    std::vector<std::vector<bool>> sampled(
        HeightGrid::GRID_RESOLUTION,
        std::vector<bool>(HeightGrid::GRID_RESOLUTION, false));

    // Second pass: rasterize terrain vertices into the grid, MAX Y per cell.
    for (unsigned int mi = 0; mi < scene->mNumMeshes; ++mi)
    {
        const aiMesh *mesh = scene->mMeshes[mi];
        if (!IsLikelyTerrainMesh(mesh))
            continue;

        for (unsigned int vi = 0; vi < mesh->mNumVertices; ++vi)
        {
            const aiVector3D &v = mesh->mVertices[vi];
            glm::vec4 worldPos = worldMatrix * glm::vec4(v.x, v.y, v.z, 1.0f);

            int gridX = static_cast<int>((worldPos.x - outGrid.minX) / outGrid.cellSizeX);
            int gridZ = static_cast<int>((worldPos.z - outGrid.minZ) / outGrid.cellSizeZ);

            gridX = std::min(std::max(gridX, 0), HeightGrid::GRID_RESOLUTION - 1);
            gridZ = std::min(std::max(gridZ, 0), HeightGrid::GRID_RESOLUTION - 1);

            if (!sampled[gridX][gridZ] || worldPos.y > outGrid.height[gridX][gridZ])
                outGrid.height[gridX][gridZ] = worldPos.y;

            sampled[gridX][gridZ] = true;
        }
    }

    outGrid.valid = true;

    Log("TERRAIN", "Height grid built from " + std::to_string(terrainMeshCount) + " terrain mesh(es).  Resolution=" + std::to_string(HeightGrid::GRID_RESOLUTION) + "x" + std::to_string(HeightGrid::GRID_RESOLUTION));
    Log("TERRAIN", "  World bounds X[" + std::to_string(outGrid.minX) + ", " + std::to_string(outGrid.maxX) + "]  Z[" + std::to_string(outGrid.minZ) + ", " + std::to_string(outGrid.maxZ) + "]");
}

// -----------------------------------------------------------------------------
// getTerrainHeight
// -----------------------------------------------------------------------------
float getTerrainHeight(float worldX, float worldZ)
{
    const HeightGrid &grid = g_TerrainHeightGrid;

    if (!grid.valid)
        return 0.0f;

    // Convert world coords -> local grid (fractional cell) coordinates.
    float localX = (worldX - grid.minX) / grid.cellSizeX;
    float localZ = (worldZ - grid.minZ) / grid.cellSizeZ;

    // Clamp into valid sampling range.
    localX = std::min(std::max(localX, 0.0f),
                      static_cast<float>(HeightGrid::GRID_RESOLUTION - 1));
    localZ = std::min(std::max(localZ, 0.0f),
                      static_cast<float>(HeightGrid::GRID_RESOLUTION - 1));

    int x0 = static_cast<int>(std::floor(localX));
    int z0 = static_cast<int>(std::floor(localZ));
    int x1 = std::min(x0 + 1, HeightGrid::GRID_RESOLUTION - 1);
    int z1 = std::min(z0 + 1, HeightGrid::GRID_RESOLUTION - 1);

    float tx = localX - static_cast<float>(x0);
    float tz = localZ - static_cast<float>(z0);

    float h00 = grid.height[x0][z0];
    float h10 = grid.height[x1][z0];
    float h01 = grid.height[x0][z1];
    float h11 = grid.height[x1][z1];

    float heightAlongX0 = h00 + (h10 - h00) * tx;
    float heightAlongX1 = h01 + (h11 - h01) * tx;

    return heightAlongX0 + (heightAlongX1 - heightAlongX0) * tz;
}