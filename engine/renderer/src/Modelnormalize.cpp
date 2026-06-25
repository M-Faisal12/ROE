// =============================================================================
// ModelNormalize.cpp
//
//   Implementation of ComputeNormalizationMatrix() and IsLikelyTerrainMesh().
//   See ModelNormalize.h for the system-level explanation.
// =============================================================================
#include "ModelNormalize.h"
#include "Logger.h"

#include <glm/gtc/matrix_transform.hpp>

#include <algorithm>
#include <cctype>
#include <string>
#include <vector>

// -----------------------------------------------------------------------------
// ComputeNormalizationMatrix
//
//   Because aiProcess_PreTransformVertices is active, all node transforms
//   (rotation, scale, translation) are already baked into vertex positions
//   by Assimp before this code runs. A flat loop over scene->mMeshes
//   therefore gives the correct world-space AABB -- no tree walk needed.
//
//   Two corrections applied:
//     1. SCALE  -- normalize the largest dimension to targetSize.
//     2. GROUND -- snap the lowest Y vertex (after scaling) to Y = 0.
// -----------------------------------------------------------------------------
glm::mat4 ComputeNormalizationMatrix(
    const aiScene *scene,
    float targetSize,
    BoundingBox &outBox)
{
    // ---- Step 0: flat AABB -- correct because aiProcess_PreTransformVertices
    //             has already baked all node transforms into vertex data. ------
    outBox = BoundingBox{};

    for (unsigned int mi = 0; mi < scene->mNumMeshes; ++mi)
    {
        const aiMesh *mesh = scene->mMeshes[mi];
        for (unsigned int vi = 0; vi < mesh->mNumVertices; ++vi)
        {
            const aiVector3D &v = mesh->mVertices[vi];
            outBox.min.x = std::min(outBox.min.x, v.x);
            outBox.min.y = std::min(outBox.min.y, v.y);
            outBox.min.z = std::min(outBox.min.z, v.z);
            outBox.max.x = std::max(outBox.max.x, v.x);
            outBox.max.y = std::max(outBox.max.y, v.y);
            outBox.max.z = std::max(outBox.max.z, v.z);
        }
    }

    outBox.size = outBox.max - outBox.min;

    Log("NORM", "World-space AABB (post-PreTransformVertices):");
    Log("NORM", "  min  = (" + std::to_string(outBox.min.x) + ", " + std::to_string(outBox.min.y) + ", " + std::to_string(outBox.min.z) + ")");
    Log("NORM", "  max  = (" + std::to_string(outBox.max.x) + ", " + std::to_string(outBox.max.y) + ", " + std::to_string(outBox.max.z) + ")");
    Log("NORM", "  size = (" + std::to_string(outBox.size.x) + ", " + std::to_string(outBox.size.y) + ", " + std::to_string(outBox.size.z) + ")");

    // ---- Step 1: SCALE -------------------------------------------------------
    float largestDim = std::max({outBox.size.x, outBox.size.y, outBox.size.z});
    if (largestDim < 1e-6f)
        largestDim = 1.0f;

    float scaleFactor = targetSize / largestDim;

    glm::mat4 scaleMatrix = glm::scale(
        glm::mat4(1.0f),
        glm::vec3(scaleFactor));

    Log("NORM", "  largest dim  = " + std::to_string(largestDim));
    Log("NORM", "  scale factor = " + std::to_string(scaleFactor));
    Log("NORM", "  target size  = " + std::to_string(targetSize) + " units");

    // ---- Step 2: GROUND ------------------------------------------------------
    float scaledMinY = outBox.min.y * scaleFactor;

    glm::mat4 groundPlacement = glm::translate(
        glm::mat4(1.0f),
        glm::vec3(0.0f, -scaledMinY, 0.0f));

    Log("NORM", "  scaled minY  = " + std::to_string(scaledMinY) + "  (translated to Y=0)");

    // ---- Combine -------------------------------------------------------------
    glm::mat4 normMatrix = groundPlacement * scaleMatrix;

    Log("NORM", "Normalization matrix built (scale + ground-snap, no centering, no rotation).");
    return normMatrix;
}

// -----------------------------------------------------------------------------
// IsLikelyTerrainMesh
//
//   Conservative heuristic to skip trees/rocks/props/leaves and keep only
//   ground-like meshes, without requiring a scene-graph/node-name walk
//   (node names are collapsed by aiProcess_PreTransformVertices, but each
//   aiMesh still carries its own mName from the exporter in most GLB/GLTF
//   files, so we check that first).
//
//   Rules (first match wins):
//     1. Name contains an exclude keyword (tree, rock, prop, leaf, leaves,
//        foliage, grass, bush, stone, debris) -> NOT terrain.
//     2. Name contains an include keyword (terrain, ground, floor, land,
//        landscape) -> IS terrain.
//     3. No name signal -> fall back to "large flat footprint" heuristic:
//        a mesh is considered terrain-like if its horizontal (X/Z) footprint
//        is large relative to its vertical extent, which is true for ground
//        meshes and false for most vertical props/trees/rocks clusters.
// -----------------------------------------------------------------------------
bool IsLikelyTerrainMesh(const aiMesh *mesh)
{
    std::string name = mesh->mName.C_Str();
    for (auto &c : name)
        c = static_cast<char>(std::tolower(c));

    static const std::vector<std::string> excludeKeywords = {
        "tree", "rock", "prop", "leaf", "leaves",
        "foliage", "grass", "bush", "stone", "debris"};
    static const std::vector<std::string> includeKeywords = {
        "terrain", "ground", "floor", "land", "landscape"};

    for (const auto &kw : excludeKeywords)
        if (!name.empty() && name.find(kw) != std::string::npos)
            return false;

    for (const auto &kw : includeKeywords)
        if (!name.empty() && name.find(kw) != std::string::npos)
            return true;

    // Fallback: large flat footprint test.
    glm::vec3 localMin(1e30f, 1e30f, 1e30f);
    glm::vec3 localMax(-1e30f, -1e30f, -1e30f);

    for (unsigned int vi = 0; vi < mesh->mNumVertices; ++vi)
    {
        const aiVector3D &v = mesh->mVertices[vi];
        localMin.x = std::min(localMin.x, v.x);
        localMin.y = std::min(localMin.y, v.y);
        localMin.z = std::min(localMin.z, v.z);
        localMax.x = std::max(localMax.x, v.x);
        localMax.y = std::max(localMax.y, v.y);
        localMax.z = std::max(localMax.z, v.z);
    }

    glm::vec3 extent = localMax - localMin;
    float horizontalFootprint = extent.x * extent.z;
    float verticalExtent = std::max(extent.y, 0.0001f);

    // Heuristic threshold: ground meshes are wide & flat, so footprint
    // dwarfs vertical extent. Tuned loosely on purpose (no extra config).
    return (horizontalFootprint / verticalExtent) > 50.0f;
}