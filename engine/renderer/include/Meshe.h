// =============================================================================
// Mesh.h
//
//   Plain GPU-side mesh handle (VAO/VBO/EBO + texture + index count) and
//   the BoundingBox helper struct used throughout normalization and
//   terrain-height code.
//
//   Vertex buffer layout per vertex  (8 floats = 32 bytes):
//     [ x  y  z ]  [ nx  ny  nz ]  [ u  v ]
//      attr 0         attr 1          attr 2
// =============================================================================
#pragma once

#include "../Externals/Glad/glad.h"
#include <glm/glm.hpp>

// =============================================================================
// Mesh
// =============================================================================
struct Mesh
{
    GLuint VAO;
    GLuint VBO;
    GLuint EBO;
    GLuint textureID;
    unsigned int indexCount;
};

// =============================================================================
// BoundingBox
// =============================================================================
struct BoundingBox
{
    glm::vec3 min{1e30f, 1e30f, 1e30f};
    glm::vec3 max{-1e30f, -1e30f, -1e30f};
    glm::vec3 size{0.0f};
};