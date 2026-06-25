// =============================================================================
// ModelNormalize.h
//
// MODEL NORMALIZATION SYSTEM
// =============================================================================
//
//   Problem:
//     Sketchfab / random community models (glTF, FBX, OBJ, ...) come with
//     wildly inconsistent scales -- a character might be 0.01 units tall, a
//     building might be 50 000 units wide -- and may float above or sink
//     below the ground plane because their author never zeroed the pivot
//     vertically.
//
//   Solution -- a pure GLM model-matrix correction, zero mesh edits:
//
//   Step 1 -- Flat AABB over scene->mMeshes
//     After aiProcess_PreTransformVertices bakes all node transforms into
//     vertex data, every mesh in scene->mMeshes already lives in world space.
//     A simple flat loop over all vertices is sufficient and correct.
//
//   Step 2 -- ComputeNormalizationMatrix()
//     From those bounds it derives one glm::mat4 that:
//       a) Scales the model so its largest dimension (X, Y or Z) equals
//          targetSize. Aspect ratio is preserved.
//       b) Translates down so the lowest Y vertex sits exactly on Y = 0
//          (the XZ ground plane). Applied AFTER scaling.
//
//     NOTE: this version intentionally does NOT recenter the model on the
//     X/Z origin, and never adds/removes rotation.
//
//   Step 3 -- Render loop
//     Pass the normalization matrix as the "model" uniform. The GPU does
//     all the work; original vertex data on the CPU / GPU is never touched.
//
//   Also includes IsLikelyTerrainMesh(), the ground-mesh classification
//   heuristic shared with the terrain height system (HeightGrid.h), since
//   both ComputeNormalizationMatrix and BuildHeightGrid operate over the
//   same raw aiScene vertex data.
// =============================================================================
#pragma once

#include "Meshe.h"
#include <assimp/scene.h>
#include <glm/glm.hpp>

// Builds a single glm::mat4 that corrects scale and vertical (ground)
// placement for community GLB models, without modifying any vertex data.
// Writes the computed flat world-space AABB into outBox.
glm::mat4 ComputeNormalizationMatrix(
    const aiScene *scene,
    float targetSize,
    BoundingBox &outBox);

// Conservative heuristic to skip trees/rocks/props/leaves and keep only
// ground-like meshes. See ModelNormalize.cpp for the full rule order.
bool IsLikelyTerrainMesh(const aiMesh *mesh);