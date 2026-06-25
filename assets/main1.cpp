// =============================================================================
// main.cpp  –  Terrain Renderer  +  Universal Model Normalizer
//
// Rendering pipeline (in order of execution):
//   1.  Load mesh geometry + normals + UV coords from a glTF/GLB/FBX/OBJ file
//       via Assimp
//   2.  For each mesh, resolve its diffuse texture path from the material
//       └─ Handles both embedded (.glb) and external (.gltf) textures
//       └─ Check the texture cache first; load from disk only on a cache miss
//   3.  Upload interleaved vertex data  (pos · normal · uv)  to the GPU
//   4.  Vertex shader  – transforms positions, corrects normals, passes UVs
//   5.  Fragment shader – samples the diffuse texture, applies diffuse lighting
//
// Vertex buffer layout per vertex  (8 floats = 32 bytes):
//   [ x  y  z ]  [ nx  ny  nz ]  [ u  v ]
//    attr 0         attr 1          attr 2
//
// Memory strategy – texture cache:
//   All loaded textures live in a single  unordered_map<path, GLuint>.
//   Multiple meshes that reference the same file (or embedded index) share
//   one GPU texture object. The cache also drives all log output so you can
//   see reuse vs. new loads.
//
// .GLB notes:
//   A .glb file is a self-contained binary glTF bundle.  Textures are stored
//   inside the file itself rather than as separate image files on disk.
//   Assimp references them with an asterisk-prefixed index string ("*0", "*1"
//   ...) instead of a file path.  This code detects that prefix and routes to
//   LoadEmbeddedTexture() which decodes the compressed bytes directly from
//   the aiScene via stbi_load_from_memory.
//
// =============================================================================
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
//          TARGET_SIZE (default 10 units).  Aspect ratio is preserved.
//       b) Translates down so the lowest Y vertex sits exactly on Y = 0
//          (the XZ ground plane).  Applied AFTER scaling.
//
//     NOTE: this version intentionally does NOT recenter the model on the
//     X/Z origin.
//
//   Step 3 -- Render loop
//     Pass the normalization matrix as the "model" uniform.  The GPU does
//     all the work; original vertex data on the CPU / GPU is never touched.
//
//   Tuning:
//     TARGET_SIZE (float, default 10.0f) -- change to make every model
//     occupy a different canonical size.  All other math is automatic.
//
// =============================================================================
//
// Camera auto-fit:
//   After normalization the model always sits at a known size (TARGET_SIZE)
//   on its largest axis and on the ground plane (Y = 0).
// =============================================================================
//
// =============================================================================
// MAP-FROM-BINARY SYSTEM   (ADDED FOR MAP LOADER)
// =============================================================================
//
//   The companion "ChunkTerrainSystem demo" program (a separate main, built
//   from Mapgenerator.h/.cpp) writes a single file:
//
//       3DModels/Map.bin
//
//   containing two back-to-back sections, written by its WriteMapBin():
//
//     Section 1 -- chunk coordinates
//       size_t                 coordCount
//       coordCount * { int x; int y; }
//
//     Section 2 -- model file names, index-aligned with Section 1
//       size_t                 nameCount        (== coordCount)
//       nameCount * {
//         size_t   len            (byte length INCLUDING the null
//                                   terminator; 0 means "no model / empty
//                                   cell", i.e. the slot was a nullptr)
//         char[len] bytes         (only present if len > 0, already
//                                   null-terminated by the writer)
//       }
//
//   coords[i] and names[i] describe the SAME cell: names[i] is the model
//   placed at chunk (coords[i].first, coords[i].second), or "no model" if
//   coords[i] has no corresponding non-empty name.
//
//   This program does NOT know in advance how many records are in the file
//   or what coordinate range they span, so the loading strategy is:
//
//     1. Build an empty 2D map of the requested width x height up front
//        (every cell initialised to "" / no model), exactly as the user
//        would see it before any data has arrived.
//     2. Stream records out of Map.bin ONE AT A TIME.
//     3. For each (x, y, modelName) record read, if (x, y) falls inside the
//        map's bounds, write modelName into that cell immediately -- i.e.
//        the map is filled in incrementally as the file is read, not
//        bulk-loaded then copied in one shot.
//     4. Records whose coordinates fall outside the configured map bounds
//        are skipped (logged, not fatal) -- the generator may have produced
//        a larger or differently-centred range than what was requested here.
//
//   Coordinate convention:
//     Chunk coordinates from the generator can be negative (they're centred
//     on an arbitrary origin chunk). This map is indexed [row][col] with
//     row 0 == the map's minZ and col 0 == the map's minX, both supplied
//     on the command line (or defaulted), so negative chunk coordinates
//     are handled by an offset, not by changing the storage type.
//
// =============================================================================

#include "../Externals/Glad/glad.h"
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <assimp/Importer.hpp>
#include <assimp/scene.h>
#include <assimp/postprocess.h>

// stb_image -- single-header image loader (PNG, JPG, BMP, TGA ...)
// STB_IMAGE_IMPLEMENTATION must be defined exactly once in the project.
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#include <cctype>    // ADDED FOR TERRAIN HEIGHT SYSTEM -- std::tolower
#include <algorithm> // std::min / std::max
#include <cmath>     // std::sin / std::cos
#include <iostream>
#include <string>
#include <unordered_map>
#include <vector>
#include <string.h>
#include <fstream>
#include <memory>
#include <cstdint>
#include <sstream>
#include <iomanip>

// =============================================================================
// Logging
// =============================================================================
static void Log(const std::string &tag, const std::string &message)
{
    std::cout << "[" << tag << "] " << message << "\n";
}

// =============================================================================
// MapGrid   (ADDED FOR MAP LOADER)
//
//   Simple 2D grid of model-name strings. Empty string == "no model placed
//   yet" (this is the initial state of every cell). Indexed [row][col],
//   row == Z axis, col == X axis, both offset by (originX, originZ) so
//   negative chunk coordinates from the generator map into valid indices.
// =============================================================================
struct MapGrid
{
    int width = 0;   // number of columns (X)
    int height = 0;  // number of rows    (Z)
    int originX = 0; // chunk X that maps to column 0
    int originZ = 0; // chunk Z that maps to row 0

    std::vector<std::vector<std::string>> cells; // cells[row][col]

    MapGrid() = default;

    MapGrid(int w, int h, int originXIn, int originZIn)
        : width(w), height(h), originX(originXIn), originZ(originZIn)
    {
        // Every cell starts empty -- "initially the map is empty" per spec.
        cells.assign(
            static_cast<size_t>(height),
            std::vector<std::string>(static_cast<size_t>(width), std::string()));
    }

    bool InBounds(int chunkX, int chunkZ, int &outCol, int &outRow) const
    {
        outCol = chunkX - originX;
        outRow = chunkZ - originZ;
        return outCol >= 0 && outCol < width &&
               outRow >= 0 && outRow < height;
    }

    // Fills a single cell in-place. Returns false (and leaves the map
    // untouched) if the coordinate is outside the configured bounds.
    bool Set(int chunkX, int chunkZ, const std::string &modelName)
    {
        int col = 0, row = 0;
        if (!InBounds(chunkX, chunkZ, col, row))
            return false;

        cells[static_cast<size_t>(row)][static_cast<size_t>(col)] = modelName;
        return true;
    }

    void Print() const
    {
        for (int row = 0; row < height; ++row)
        {
            for (int col = 0; col < width; ++col)
            {
                const std::string &cell = cells[static_cast<size_t>(row)][static_cast<size_t>(col)];
                std::cout << (cell.empty() ? std::string("..") : cell) << "\t";
            }
            std::cout << "\n";
        }
    }
};

// =============================================================================
// ReadMapBinStreaming   (ADDED FOR MAP LOADER)
//
//   Mirrors WriteMapBin() from the generator program field-for-field.
//   Reads Section 1 (coords) and Section 2 (names) and, for every record i,
//   immediately calls onRecord(x, y, name) so the caller can fill its map
//   one cell at a time as the file streams in -- rather than materializing
//   both vectors fully and copying them into the grid afterward.
//
//   Returns false on any I/O error or size mismatch between the two
//   sections (which should never happen if the file was produced by
//   WriteMapBin, but the reader doesn't trust that blindly).
// =============================================================================
template <typename RecordCallback>
static bool ReadMapBinStreaming(const std::string &filename, RecordCallback onRecord)
{
    std::ifstream inFile(filename, std::ios::in | std::ios::binary);
    if (!inFile)
    {
        Log("ERROR", "Could not open map file: " + filename);
        return false;
    }

    // ---- Section 1: vector<pair<int,int>> coords ----
    size_t coordCount = 0;
    inFile.read(reinterpret_cast<char *>(&coordCount), sizeof(coordCount));
    if (!inFile)
    {
        Log("ERROR", "Failed reading coord count from: " + filename);
        return false;
    }

    std::vector<std::pair<int, int>> coords;
    coords.reserve(coordCount);

    for (size_t i = 0; i < coordCount; ++i)
    {
        int x = 0, z = 0;
        inFile.read(reinterpret_cast<char *>(&x), sizeof(x));
        inFile.read(reinterpret_cast<char *>(&z), sizeof(z));
        if (!inFile)
        {
            Log("ERROR", "Truncated coord section at record " + std::to_string(i));
            return false;
        }
        coords.emplace_back(x, z);
    }

    // ---- Section 2: vector<char*> model file names ----
    size_t nameCount = 0;
    inFile.read(reinterpret_cast<char *>(&nameCount), sizeof(nameCount));
    if (!inFile)
    {
        Log("ERROR", "Failed reading name count from: " + filename);
        return false;
    }

    if (nameCount != coordCount)
    {
        Log("ERROR", "Map file corrupt: coordCount=" + std::to_string(coordCount) +
                          " but nameCount=" + std::to_string(nameCount));
        return false;
    }

    for (size_t i = 0; i < nameCount; ++i)
    {
        size_t len = 0;
        inFile.read(reinterpret_cast<char *>(&len), sizeof(len));
        if (!inFile)
        {
            Log("ERROR", "Truncated name length at record " + std::to_string(i));
            return false;
        }

        std::string name; // stays empty for len == 0 (nullptr slot in writer)

        if (len > 0)
        {
            std::vector<char> buffer(len);
            inFile.read(buffer.data(), static_cast<std::streamsize>(len));
            if (!inFile)
            {
                Log("ERROR", "Truncated name bytes at record " + std::to_string(i));
                return false;
            }
            // buffer already includes the writer's null terminator.
            name.assign(buffer.data());
        }

        // Stream this single record straight to the caller -- this is the
        // "fill the map as it reads" step.
        onRecord(coords[i].first, coords[i].second, name);
    }

    return true;
}

// =============================================================================
// LoadMapFromBinary   (ADDED FOR MAP LOADER)
//
//   1. Builds an empty MapGrid of (width x height), centred so that chunk
//      (centerX, centerZ) lands in the middle of the grid -- matching the
//      same minX/minZ convention the generator's own main() uses
//      (chunkX - rangeW/2 .. chunkX + rangeW/2, etc).
//   2. Streams 3DModels/Map.bin one record at a time, filling matching
//      cells in place via grid.Set().
// =============================================================================
static MapGrid LoadMapFromBinary(
    const std::string &filename,
    int width, int height,
    int centerX, int centerZ)
{
    int originX = centerX - width / 2;
    int originZ = centerZ - height / 2;

    MapGrid grid(width, height, originX, originZ);

    Log("MAP", "Empty map created: " + std::to_string(width) + "x" + std::to_string(height) +
                   "  origin=(" + std::to_string(originX) + ", " + std::to_string(originZ) + ")");

    size_t recordsApplied = 0;
    size_t recordsSkipped = 0;

    bool ok = ReadMapBinStreaming(filename,
        [&](int x, int z, const std::string &modelName)
        {
            if (grid.Set(x, z, modelName))
            {
                ++recordsApplied;
                Log("MAP", "  filled (" + std::to_string(x) + ", " + std::to_string(z) + ") = " +
                               (modelName.empty() ? "<empty>" : modelName));
            }
            else
            {
                ++recordsSkipped;
            }
        });

    if (!ok)
    {
        Log("ERROR", "Map load failed or was incomplete -- grid may be partially filled.");
    }

    Log("MAP", "Done. Applied " + std::to_string(recordsApplied) +
                   " record(s), skipped " + std::to_string(recordsSkipped) +
                   " (outside map bounds).");

    return grid;
}

// =============================================================================
// TextureCache
// =============================================================================
static std::unordered_map<std::string, GLuint> g_TextureCache;

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
// ADDED FOR TERRAIN HEIGHT SYSTEM
// =============================================================================
// HeightGrid
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
// =============================================================================
struct HeightGrid
{
    static const int GRID_RESOLUTION = 256; // ADDED FOR TERRAIN HEIGHT SYSTEM

    std::vector<std::vector<float>> height; // ADDED FOR TERRAIN HEIGHT SYSTEM -- height[x][z]

    float minX = 0.0f, maxX = 0.0f; // ADDED FOR TERRAIN HEIGHT SYSTEM
    float minZ = 0.0f, maxZ = 0.0f; // ADDED FOR TERRAIN HEIGHT SYSTEM
    float cellSizeX = 1.0f;         // ADDED FOR TERRAIN HEIGHT SYSTEM
    float cellSizeZ = 1.0f;         // ADDED FOR TERRAIN HEIGHT SYSTEM
    bool valid = false;             // ADDED FOR TERRAIN HEIGHT SYSTEM

    HeightGrid() // ADDED FOR TERRAIN HEIGHT SYSTEM
    {
        height.assign(
            GRID_RESOLUTION,
            std::vector<float>(GRID_RESOLUTION, 0.0f));
    }
};

// ADDED FOR TERRAIN HEIGHT SYSTEM
// Global instance -- mirrors the style of g_TextureCache (single global
// resource owned by main.cpp). Populated once after model load.
static HeightGrid g_TerrainHeightGrid;
// =============================================================================
// Vertex Shader
// =============================================================================
const char *vertexShaderSource = R"(
#version 330 core

layout(location = 0) in vec3 aPos;
layout(location = 1) in vec3 aNormal;
layout(location = 2) in vec2 aTexCoord;

out vec3 FragPos;
out vec3 Normal;
out vec2 TexCoord;

uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;

void main()
{
    FragPos = vec3(model * vec4(aPos, 1.0));
    Normal = mat3(transpose(inverse(model))) * aNormal;
    TexCoord = aTexCoord;
    gl_Position = projection * view * vec4(FragPos, 1.0);
}
)";

// =============================================================================
// Fragment Shader
// =============================================================================
const char *fragmentShaderSource = R"(
#version 330 core

in vec3 FragPos;
in vec3 Normal;
in vec2 TexCoord;

out vec4 FragColor;

uniform sampler2D uDiffuseMap;

void main()
{
    vec3 lightDirection = normalize(vec3(-1.0, -1.0, -1.0));
    float diffuse = max(dot(normalize(Normal), -lightDirection), 0.0);
    float ambientStrength = 0.2;
    vec4 textureColor = texture(uDiffuseMap, TexCoord);
    vec3 finalColor   = textureColor.rgb * (ambientStrength + diffuse);
    FragColor = vec4(finalColor, textureColor.a);
}
)";

// =============================================================================
// LoadTextureFromFile
// =============================================================================
GLuint LoadTextureFromFile(const std::string &filePath)
{
    stbi_set_flip_vertically_on_load(true);

    int imageWidth = 0;
    int imageHeight = 0;
    int channelCount = 0;

    unsigned char *pixelData =
        stbi_load(filePath.c_str(), &imageWidth, &imageHeight, &channelCount, 4);

    if (!pixelData)
    {
        Log("ERROR", "stb_image could not open: " + filePath);
        Log("ERROR", std::string("  reason: ") + stbi_failure_reason());
        return 0;
    }

    GLuint textureHandle = 0;
    glGenTextures(1, &textureHandle);
    glBindTexture(GL_TEXTURE_2D, textureHandle);

    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA,
                 imageWidth, imageHeight, 0,
                 GL_RGBA, GL_UNSIGNED_BYTE, pixelData);
    glGenerateMipmap(GL_TEXTURE_2D);
    stbi_image_free(pixelData);

    Log("TEXTURE", "Uploaded to GPU: " + filePath + "  (" + std::to_string(imageWidth) + "x" + std::to_string(imageHeight) + ")" + "  handle=" + std::to_string(textureHandle));

    return textureHandle;
}

// =============================================================================
// LoadEmbeddedTexture
// =============================================================================
GLuint LoadEmbeddedTexture(const aiScene *scene, const std::string &embeddedRef)
{
    int texIndex = std::stoi(embeddedRef.substr(1));

    if (texIndex < 0 || static_cast<unsigned int>(texIndex) >= scene->mNumTextures)
    {
        Log("ERROR", "Embedded texture index out of range: " + embeddedRef);
        return 0;
    }

    const aiTexture *tex = scene->mTextures[texIndex];

    int w = 0, h = 0, channels = 0;
    unsigned char *pixelData = nullptr;
    bool ownsPixelData = true;

    if (tex->mHeight == 0)
    {
        stbi_set_flip_vertically_on_load(true);

        pixelData = stbi_load_from_memory(
            reinterpret_cast<const unsigned char *>(tex->pcData),
            static_cast<int>(tex->mWidth),
            &w, &h, &channels, 4);

        if (!pixelData)
        {
            Log("ERROR", "stbi failed to decode embedded texture: " + embeddedRef);
            Log("ERROR", std::string("  reason: ") + stbi_failure_reason());
            return 0;
        }
    }
    else
    {
        w = static_cast<int>(tex->mWidth);
        h = static_cast<int>(tex->mHeight);
        ownsPixelData = false;

        std::vector<unsigned char> rgba(static_cast<size_t>(w) * h * 4);
        for (int i = 0; i < w * h; ++i)
        {
            const aiTexel &texel = tex->pcData[i];
            rgba[i * 4 + 0] = texel.r;
            rgba[i * 4 + 1] = texel.g;
            rgba[i * 4 + 2] = texel.b;
            rgba[i * 4 + 3] = texel.a;
        }

        GLuint textureHandle = 0;
        glGenTextures(1, &textureHandle);
        glBindTexture(GL_TEXTURE_2D, textureHandle);

        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA,
                     w, h, 0,
                     GL_RGBA, GL_UNSIGNED_BYTE, rgba.data());
        glGenerateMipmap(GL_TEXTURE_2D);

        Log("TEXTURE", "Uploaded embedded raw texture " + embeddedRef + "  (" + std::to_string(w) + "x" + std::to_string(h) + ")" + "  handle=" + std::to_string(textureHandle));

        return textureHandle;
    }

    GLuint textureHandle = 0;
    glGenTextures(1, &textureHandle);
    glBindTexture(GL_TEXTURE_2D, textureHandle);

    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA,
                 w, h, 0,
                 GL_RGBA, GL_UNSIGNED_BYTE, pixelData);
    glGenerateMipmap(GL_TEXTURE_2D);

    if (ownsPixelData)
        stbi_image_free(pixelData);

    Log("TEXTURE", "Uploaded embedded texture " + embeddedRef + "  (" + std::to_string(w) + "x" + std::to_string(h) + ")" + "  handle=" + std::to_string(textureHandle));

    return textureHandle;
}

// Compute a simple content hash for an embedded texture. Uses FNV-1a
// 64-bit over the raw image bytes (for compressed images) or the
// expanded RGBA texel bytes (for uncompressed images). Returns a
// hex string suitable for use as a cache key.
static std::string HashEmbeddedTexture(const aiTexture *tex)
{
    const uint64_t FNV_OFFSET = 14695981039346656037ULL;
    const uint64_t FNV_PRIME = 1099511628211ULL;

    uint64_t hash = FNV_OFFSET;

    if (!tex)
        return std::string("0000000000000000");

    if (tex->mHeight == 0)
    {
        const unsigned char *bytes = reinterpret_cast<const unsigned char *>(tex->pcData);
        size_t len = static_cast<size_t>(tex->mWidth);
        for (size_t i = 0; i < len; ++i)
        {
            hash ^= static_cast<uint64_t>(bytes[i]);
            hash *= FNV_PRIME;
        }
    }
    else
    {
        int w = static_cast<int>(tex->mWidth);
        int h = static_cast<int>(tex->mHeight);
        const aiTexel *pixels = tex->pcData;
        for (int i = 0; i < w * h; ++i)
        {
            hash ^= static_cast<uint64_t>(pixels[i].r);
            hash *= FNV_PRIME;
            hash ^= static_cast<uint64_t>(pixels[i].g);
            hash *= FNV_PRIME;
            hash ^= static_cast<uint64_t>(pixels[i].b);
            hash *= FNV_PRIME;
            hash ^= static_cast<uint64_t>(pixels[i].a);
            hash *= FNV_PRIME;
        }
    }

    std::ostringstream oss;
    oss << std::hex << std::setfill('0') << std::setw(16) << hash;
    return oss.str();
}

// =============================================================================
// GetOrLoadTexture
// =============================================================================
GLuint GetOrLoadTexture(const std::string &filePath)
{
    auto cacheEntry = g_TextureCache.find(filePath);

    if (cacheEntry != g_TextureCache.end())
    {
        Log("CACHE", "HIT  - reusing handle " + std::to_string(cacheEntry->second) + "  for: " + filePath);
        return cacheEntry->second;
    }

    Log("CACHE", "MISS - loading from disk: " + filePath);
    GLuint newHandle = LoadTextureFromFile(filePath);
    g_TextureCache[filePath] = newHandle;
    return newHandle;
}

// =============================================================================
// GetOrLoadEmbeddedTexture
// =============================================================================
GLuint GetOrLoadEmbeddedTexture(const aiScene *scene, const std::string &embeddedRef)
{
    // Compute a content-based key for this embedded texture so identical
    // image data across different files (or different embedded indices)
    // reuse the same GPU texture object.
    int texIndex = 0;
    try
    {
        texIndex = std::stoi(embeddedRef.substr(1));
    }
    catch (...)
    {
        Log("ERROR", "Invalid embedded texture reference: " + embeddedRef);
        return 0;
    }

    if (texIndex < 0 || static_cast<unsigned int>(texIndex) >= scene->mNumTextures)
    {
        Log("ERROR", "Embedded texture index out of range: " + embeddedRef);
        return 0;
    }

    const aiTexture *tex = scene->mTextures[texIndex];
    std::string key = std::string("*embedded_") + HashEmbeddedTexture(tex);

    auto cacheEntry = g_TextureCache.find(key);
    if (cacheEntry != g_TextureCache.end())
    {
        Log("CACHE", "HIT  - reusing handle " + std::to_string(cacheEntry->second) + "  for embedded hash: " + key);
        return cacheEntry->second;
    }

    Log("CACHE", "MISS - decoding embedded texture: " + embeddedRef + "  hash=" + key);
    GLuint newHandle = LoadEmbeddedTexture(scene, embeddedRef);
    if (newHandle != 0)
        g_TextureCache[key] = newHandle;
    return newHandle;
}

// =============================================================================
// ResolveMeshTexturePath
// =============================================================================
std::string ResolveMeshTexturePath(
    const aiScene *scene,
    unsigned int meshIndex,
    const std::string &sceneDirectory,
    bool &outIsEmbedded)
{
    outIsEmbedded = false;

    const aiMesh *sourceMesh = scene->mMeshes[meshIndex];
    const aiMaterial *material = scene->mMaterials[sourceMesh->mMaterialIndex];

    aiString relativeTexturePath;

    if (material->GetTexture(
            aiTextureType_DIFFUSE, 0, &relativeTexturePath) != AI_SUCCESS)
        return "";

    const char *rawPath = relativeTexturePath.C_Str();

    if (rawPath[0] == '*')
    {
        outIsEmbedded = true;
        return std::string(rawPath);
    }

    return sceneDirectory + "/" + rawPath;
}

// =============================================================================
// CreateShaderProgram
// =============================================================================
GLuint CreateShaderProgram()
{
    Log("INIT", "Compiling vertex shader ...");

    GLuint vertexShader = glCreateShader(GL_VERTEX_SHADER);
    glShaderSource(vertexShader, 1, &vertexShaderSource, nullptr);
    glCompileShader(vertexShader);

    GLint vertexCompileSuccess = 0;
    glGetShaderiv(vertexShader, GL_COMPILE_STATUS, &vertexCompileSuccess);
    if (!vertexCompileSuccess)
    {
        char infoLog[512];
        glGetShaderInfoLog(vertexShader, 512, nullptr, infoLog);
        Log("ERROR", std::string("Vertex shader compile failed:\n") + infoLog);
    }
    else
    {
        Log("INIT", "Vertex shader compiled OK.");
    }

    Log("INIT", "Compiling fragment shader ...");

    GLuint fragmentShader = glCreateShader(GL_FRAGMENT_SHADER);
    glShaderSource(fragmentShader, 1, &fragmentShaderSource, nullptr);
    glCompileShader(fragmentShader);

    GLint fragmentCompileSuccess = 0;
    glGetShaderiv(fragmentShader, GL_COMPILE_STATUS, &fragmentCompileSuccess);
    if (!fragmentCompileSuccess)
    {
        char infoLog[512];
        glGetShaderInfoLog(fragmentShader, 512, nullptr, infoLog);
        Log("ERROR", std::string("Fragment shader compile failed:\n") + infoLog);
    }
    else
    {
        Log("INIT", "Fragment shader compiled OK.");
    }

    Log("INIT", "Linking shader program ...");

    GLuint program = glCreateProgram();
    glAttachShader(program, vertexShader);
    glAttachShader(program, fragmentShader);
    glLinkProgram(program);

    GLint linkSuccess = 0;
    glGetProgramiv(program, GL_LINK_STATUS, &linkSuccess);
    if (!linkSuccess)
    {
        char infoLog[512];
        glGetProgramInfoLog(program, 512, nullptr, infoLog);
        Log("ERROR", std::string("Shader program link failed:\n") + infoLog);
    }
    else
    {
        Log("INIT", "Shader program linked OK.  handle=" + std::to_string(program));
    }

    glDeleteShader(vertexShader);
    glDeleteShader(fragmentShader);
    return program;
}

// =============================================================================
// =============================================================================
//
//   MODEL NORMALIZATION SYSTEM
//
// =============================================================================
// =============================================================================

// -----------------------------------------------------------------------------
// ComputeNormalizationMatrix
//
//   Builds a single glm::mat4 that corrects scale and vertical (ground)
//   placement for community GLB models, without modifying any vertex data.
//
//   Because aiProcess_PreTransformVertices is active, all node transforms
//   (rotation, scale, translation) are already baked into vertex positions
//   by Assimp before this code runs.  A flat loop over scene->mMeshes
//   therefore gives the correct world-space AABB -- no tree walk needed.
//
//   Two corrections applied:
//     1. SCALE  -- normalize the largest dimension to targetSize.
//     2. GROUND -- snap the lowest Y vertex (after scaling) to Y = 0.
//
//   The model's authored XZ position passes through unchanged.
//   No rotation is ever added or removed by this function.
// -----------------------------------------------------------------------------
static glm::mat4 ComputeNormalizationMatrix(
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

// ADDED FOR TERRAIN HEIGHT SYSTEM
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
static bool IsLikelyTerrainMesh(const aiMesh *mesh)
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

// ADDED FOR TERRAIN HEIGHT SYSTEM
// -----------------------------------------------------------------------------
// BuildHeightGrid
//
//   Samples terrain-classified meshes from `scene` into `outGrid`, applying
//   `worldMatrix` (the same normalization matrix used as the render "model"
//   uniform) to each raw vertex so the grid lives in true world space.
//
//   For each vertex: compute its world-space XZ, find which grid cell it
//   falls into, and keep the MAX world-space Y seen for that cell.
//   Cells with no samples fall back to the lowest recorded height (0 after
//   ground-snap) so getTerrainHeight() never returns an uninitialized value.
// -----------------------------------------------------------------------------
static void BuildHeightGrid(
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

// ADDED FOR TERRAIN HEIGHT SYSTEM
// -----------------------------------------------------------------------------
// getTerrainHeight
//
//   Converts world (X, Z) into the height grid's local cell space and
//   returns a bilinearly-interpolated terrain Y. Works against whichever
//   transform was baked into the grid by BuildHeightGrid (currently the
//   model's normalization matrix -- scale + ground-snap -- matching what's
//   passed to the "model" shader uniform, so results line up with what's
//   on screen).
//
//   Returns 0.0f if the grid was never successfully built (e.g. no terrain
//   mesh identified), which matches the default ground-snap plane.
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

// =============================================================================
// main
//
//   Usage:  ./renderer [height] [width] [centerX] [centerZ]
//
//   height/width  -- size of the 2D map grid to allocate (defaults below)
//   centerX/Z     -- chunk coordinate the map is centred on (defaults 0,0)
//
//   NOTE: argv[1]/argv[2] preserve this program's original (height, width)
//   argument order from before the map loader was added.
// =============================================================================
int main(int argc, char *argv[])
{
    int mapHeight = 20;
    int mapWidth = 20;
    int mapCenterX = 0;
    int mapCenterZ = 0;

    if (argc > 1)
        mapHeight = std::atoi(argv[1]);
    if (argc > 2)
        mapWidth = std::atoi(argv[2]);
    if (argc > 3)
        mapCenterX = std::atoi(argv[3]);
    if (argc > 4)
        mapCenterZ = std::atoi(argv[4]);

    if (mapHeight <= 0 || mapWidth <= 0)
    {
        Log("ERROR", "height and width must both be positive.");
        return -1;
    }

    // -------------------------------------------------------------------------
    // MAP LOADER  (ADDED FOR MAP LOADER)
    //
    //   1. Allocate the empty mapWidth x mapHeight grid.
    //   2. Stream 3DModels/Map.bin record-by-record, filling matching cells
    //      as each record arrives.
    // -------------------------------------------------------------------------
    Log("MAP", "Building empty " + std::to_string(mapWidth) + "x" + std::to_string(mapHeight) + " map ...");

    MapGrid worldMap = LoadMapFromBinary(
        "3DModels/Map.bin", mapWidth, mapHeight, mapCenterX, mapCenterZ);

    Log("MAP", "Final map contents:");
    worldMap.Print();

    // -------------------------------------------------------------------------
    // GLFW + window
    // -------------------------------------------------------------------------
    Log("INIT", "Initialising GLFW ...");

    if (!glfwInit())
    {
        Log("ERROR", "glfwInit() failed.");
        return -1;
    }

    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

    Log("INIT", "Creating window (1280x720) ...");

    GLFWwindow *window = glfwCreateWindow(
        1280, 720, "Render Full Model", nullptr, nullptr);

    if (!window)
    {
        Log("ERROR", "glfwCreateWindow() failed.");
        glfwTerminate();
        return -1;
    }

    glfwMakeContextCurrent(window);

    // -------------------------------------------------------------------------
    // GLAD
    // -------------------------------------------------------------------------
    Log("INIT", "Loading OpenGL function pointers via GLAD ...");

    if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress))
    {
        Log("ERROR", "gladLoadGLLoader() failed.");
        return -1;
    }

    Log("INIT", "OpenGL context ready.");
    glViewport(0, 0, 1280, 720);

    // -------------------------------------------------------------------------
    // Shader program
    // -------------------------------------------------------------------------
    GLuint shaderProgram = CreateShaderProgram();

    glUseProgram(shaderProgram);
    glUniform1i(glGetUniformLocation(shaderProgram, "uDiffuseMap"), 0);
    Log("INIT", "uDiffuseMap uniform bound to texture unit 0.");

    // -------------------------------------------------------------------------
    // Scene loading (Assimp) — multi-model mode driven by `worldMap`.
    //  - Build one GPU upload per unique model filename found in the map.
    //  - Pick an origin chunk (center if non-empty, otherwise first non-empty)
    //    and place all other models relative to that origin by integer
    //    multiples of `CHUNK_SPACING`.
    // -------------------------------------------------------------------------
    const std::string sceneDirectory = "3DModels";

    // Collect unique model names and a list of placed records (chunk coords)
    struct MapRecord { int chunkX; int chunkZ; std::string name; };
    std::vector<MapRecord> placedRecords;
    std::unordered_map<std::string, bool> uniqueNames;

    for (int row = 0; row < worldMap.height; ++row)
    {
        for (int col = 0; col < worldMap.width; ++col)
        {
            const std::string &cell = worldMap.cells[static_cast<size_t>(row)][static_cast<size_t>(col)];
            if (cell.empty())
                continue;

            int chunkX = worldMap.originX + col;
            int chunkZ = worldMap.originZ + row;
            placedRecords.push_back({chunkX, chunkZ, cell});
            uniqueNames[cell] = true;
        }
    }

    if (placedRecords.empty())
    {
        Log("MODEL", "No models placed in the map — nothing to load.");
    }

    // Decide origin chunk: prefer the configured center; fallback to first
    int originChunkX = mapCenterX;
    int originChunkZ = mapCenterZ;

    auto centerCellName = std::string();
    int centerCol = mapCenterX - worldMap.originX;
    int centerRow = mapCenterZ - worldMap.originZ;
    if (centerCol >= 0 && centerCol < worldMap.width && centerRow >= 0 && centerRow < worldMap.height)
        centerCellName = worldMap.cells[static_cast<size_t>(centerRow)][static_cast<size_t>(centerCol)];

    if (centerCellName.empty())
    {
        if (!placedRecords.empty())
        {
            originChunkX = placedRecords[0].chunkX;
            originChunkZ = placedRecords[0].chunkZ;
        }
    }

    Log("MODEL", "Origin chunk chosen: (" + std::to_string(originChunkX) + ", " + std::to_string(originChunkZ) + ")");

    // Load and upload each unique model once, keep data in a registry.
    struct UploadedModel
    {
        std::vector<Mesh> meshes;
        glm::mat4 normMatrix;
        BoundingBox box;
    };

    std::unordered_map<std::string, UploadedModel> registry;
    std::vector<std::unique_ptr<Assimp::Importer>> importers;

    const float TARGET_SIZE = 10.0f; // same normalization target as before
    const float SPACING_FACTOR = 1.0f; // spacing between chunk-aligned instances
    const float CHUNK_SPACING = TARGET_SIZE * SPACING_FACTOR;

    for (const auto &entry : uniqueNames)
    {
        const std::string modelName = entry.first;
        std::string modelPath = sceneDirectory + "/" + modelName;

        Log("MODEL", "Loading scene: " + modelPath);

        auto importerPtr = std::make_unique<Assimp::Importer>();
        const aiScene *scene = importerPtr->ReadFile(
            modelPath,
            aiProcess_Triangulate |
                aiProcess_FlipUVs |
                aiProcess_GenNormals |
                aiProcess_PreTransformVertices);

        if (!scene || !scene->mRootNode)
        {
            Log("ERROR", "Assimp failed to load scene: " + modelPath + "  reason: " + importerPtr->GetErrorString());
            continue;
        }

        Log("MODEL", "Scene loaded.  Sub-meshes: " + std::to_string(scene->mNumMeshes) + "  Materials: " + std::to_string(scene->mNumMaterials) + "  Textures (embedded): " + std::to_string(scene->mNumTextures));

        BoundingBox rawBox;
        glm::mat4 norm = ComputeNormalizationMatrix(scene, TARGET_SIZE, rawBox);

        // If this is the origin model, build the global height grid from it
        // so getTerrainHeight() behaves sensibly for placement queries.
        if (!placedRecords.empty())
        {
            // check if any placed record matches origin and this filename
            for (const auto &rec : placedRecords)
            {
                if (rec.chunkX == originChunkX && rec.chunkZ == originChunkZ && rec.name == modelName)
                {
                    BuildHeightGrid(scene, norm, g_TerrainHeightGrid);
                    if (g_TerrainHeightGrid.valid)
                        Log("TERRAIN", "Height grid built from origin model: " + modelName);
                    break;
                }
            }
        }

        UploadedModel uploaded;
        uploaded.normMatrix = norm;
        uploaded.box = rawBox;

        // Upload meshes for this scene
        unsigned int totalVerticesLocal = 0;
        for (unsigned int meshIndex = 0; meshIndex < scene->mNumMeshes; meshIndex++)
        {
            aiMesh *sourceMesh = scene->mMeshes[meshIndex];

            std::vector<float> interleavedVertices;
            std::vector<unsigned int> indices;

            interleavedVertices.reserve(sourceMesh->mNumVertices * 8);
            indices.reserve(sourceMesh->mNumFaces * 3);

            for (unsigned int vertexIndex = 0; vertexIndex < sourceMesh->mNumVertices; vertexIndex++)
            {
                interleavedVertices.push_back(sourceMesh->mVertices[vertexIndex].x);
                interleavedVertices.push_back(sourceMesh->mVertices[vertexIndex].y);
                interleavedVertices.push_back(sourceMesh->mVertices[vertexIndex].z);

                if (sourceMesh->HasNormals())
                {
                    interleavedVertices.push_back(sourceMesh->mNormals[vertexIndex].x);
                    interleavedVertices.push_back(sourceMesh->mNormals[vertexIndex].y);
                    interleavedVertices.push_back(sourceMesh->mNormals[vertexIndex].z);
                }
                else
                {
                    interleavedVertices.push_back(0.0f);
                    interleavedVertices.push_back(1.0f);
                    interleavedVertices.push_back(0.0f);
                }

                if (sourceMesh->HasTextureCoords(0))
                {
                    interleavedVertices.push_back(sourceMesh->mTextureCoords[0][vertexIndex].x);
                    interleavedVertices.push_back(sourceMesh->mTextureCoords[0][vertexIndex].y);
                }
                else
                {
                    interleavedVertices.push_back(0.0f);
                    interleavedVertices.push_back(0.0f);
                }
            }

            for (unsigned int faceIndex = 0; faceIndex < sourceMesh->mNumFaces; faceIndex++)
            {
                const aiFace &face = sourceMesh->mFaces[faceIndex];
                for (unsigned int indexSlot = 0; indexSlot < face.mNumIndices; indexSlot++)
                    indices.push_back(face.mIndices[indexSlot]);
            }

            totalVerticesLocal += sourceMesh->mNumVertices;

            Mesh gpuMesh;
            glGenVertexArrays(1, &gpuMesh.VAO);
            glGenBuffers(1, &gpuMesh.VBO);
            glGenBuffers(1, &gpuMesh.EBO);

            glBindVertexArray(gpuMesh.VAO);

            glBindBuffer(GL_ARRAY_BUFFER, gpuMesh.VBO);
            glBufferData(GL_ARRAY_BUFFER, interleavedVertices.size() * sizeof(float), interleavedVertices.data(), GL_STATIC_DRAW);

            glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, gpuMesh.EBO);
            glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.size() * sizeof(unsigned int), indices.data(), GL_STATIC_DRAW);

            const GLsizei vertexStride = 8 * sizeof(float);
            glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, vertexStride, (void *)0);
            glEnableVertexAttribArray(0);
            glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, vertexStride, (void *)(3 * sizeof(float)));
            glEnableVertexAttribArray(1);
            glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, vertexStride, (void *)(6 * sizeof(float)));
            glEnableVertexAttribArray(2);

            gpuMesh.indexCount = static_cast<unsigned int>(indices.size());

            bool isEmbedded = false;
            std::string texturePath = ResolveMeshTexturePath(scene, meshIndex, sceneDirectory, isEmbedded);

            if (texturePath.empty())
            {
                gpuMesh.textureID = 0;
            }
            else if (isEmbedded)
                gpuMesh.textureID = GetOrLoadEmbeddedTexture(scene, texturePath);
            else
                gpuMesh.textureID = GetOrLoadTexture(texturePath);

            uploaded.meshes.push_back(gpuMesh);
        }

        Log("MODEL", "Uploaded model " + modelName + "  meshes=" + std::to_string(uploaded.meshes.size()) + "  verts=" + std::to_string(totalVerticesLocal) + "  textures=" + std::to_string(g_TextureCache.size()));

        // Keep importer alive for the lifetime of the program
        importers.push_back(std::move(importerPtr));
        registry[modelName] = std::move(uploaded);
    }

    Log("MODEL", "All unique models uploaded: " + std::to_string(registry.size()));

    // Determine origin model (name) and pick its bounding box / norm matrix
    BoundingBox rawBox;
    glm::mat4 modelMatrix = glm::mat4(1.0f);
    std::string originModelName;

    for (const auto &rec : placedRecords)
    {
        if (rec.chunkX == originChunkX && rec.chunkZ == originChunkZ)
        {
            originModelName = rec.name;
            break;
        }
    }

    if (originModelName.empty() && !placedRecords.empty())
        originModelName = placedRecords[0].name;

    if (!originModelName.empty())
    {
        auto it = registry.find(originModelName);
        if (it != registry.end())
        {
            rawBox = it->second.box;
            modelMatrix = it->second.normMatrix;
        }
    }
    else if (!registry.empty())
    {
        // fallback to any loaded model
        auto it = registry.begin();
        rawBox = it->second.box;
        modelMatrix = it->second.normMatrix;
    }

    // =========================================================================
    // Camera auto-fit
    // =========================================================================
    const float CAM_PADDING_FACTOR = 1.35f;
    const float CAM_ELEVATION_DEG = 35.0f;
    const float CAM_FOV_DEG = 45.0f;
    const float CAM_FAR_PLANE_MULTIPLIER = 8.0f;

    float rawLargestDim = std::max({rawBox.size.x, rawBox.size.y, rawBox.size.z});
    if (rawLargestDim < 1e-6f)
        rawLargestDim = 1.0f;
    float normScaleFactor = TARGET_SIZE / rawLargestDim;

    glm::vec3 normalizedHalfExtents = (rawBox.size * normScaleFactor) * 0.5f;

    float boundingRadius = glm::length(normalizedHalfExtents);

    float fovYRadians = glm::radians(CAM_FOV_DEG);
    float camDist = (boundingRadius / std::sin(fovYRadians * 0.5f)) * CAM_PADDING_FACTOR;

    glm::vec3 scaledCenterXZAndHeight = glm::vec3(
        (rawBox.min.x + rawBox.max.x) * 0.5f * normScaleFactor,
        normalizedHalfExtents.y,
        (rawBox.min.z + rawBox.max.z) * 0.5f * normScaleFactor);

    glm::vec3 lookAtTarget = scaledCenterXZAndHeight;

    float elevationRad = glm::radians(CAM_ELEVATION_DEG);
    float camHorizDist = camDist * std::cos(elevationRad);
    float camHeight = camDist * std::sin(elevationRad);

    glm::vec3 cameraPos = lookAtTarget + glm::vec3(
                                             camHorizDist * 0.70710678f,
                                             camHeight,
                                             camHorizDist * 0.70710678f);

    glm::mat4 viewMatrix = glm::lookAt(
        cameraPos,
        lookAtTarget,
        glm::vec3(0.0f, 1.0f, 0.0f));

    glm::mat4 projectionMatrix = glm::perspective(
        fovYRadians,
        1280.0f / 720.0f,
        0.05f,
        camDist * CAM_FAR_PLANE_MULTIPLIER);

    Log("INIT", "Bounding radius (normalized) = " + std::to_string(boundingRadius));
    Log("INIT", "Camera distance computed     = " + std::to_string(camDist));
    Log("INIT", "Camera position (" + std::to_string(cameraPos.x) + ", " + std::to_string(cameraPos.y) + ", " + std::to_string(cameraPos.z) + ")" + "  looking at (" + std::to_string(lookAtTarget.x) + ", " + std::to_string(lookAtTarget.y) + ", " + std::to_string(lookAtTarget.z) + ")");
    Log("INIT", "Transformation matrices constructed.");

    glEnable(GL_DEPTH_TEST);

    // =========================================================================
    // Render loop
    // =========================================================================
    Log("RENDER", "Entering render loop.");

    while (!glfwWindowShouldClose(window))
    {
        glfwPollEvents();

        glClearColor(0.1f, 0.1f, 0.15f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        glUseProgram(shaderProgram);

        // Render every placed record as an instance of its uploaded model,
        // positioned relative to the chosen origin chunk.
        glUniformMatrix4fv(
            glGetUniformLocation(shaderProgram, "view"),
            1, GL_FALSE, glm::value_ptr(viewMatrix));

        glUniformMatrix4fv(
            glGetUniformLocation(shaderProgram, "projection"),
            1, GL_FALSE, glm::value_ptr(projectionMatrix));

        for (const auto &rec : placedRecords)
        {
            auto it = registry.find(rec.name);
            if (it == registry.end())
                continue;

            float dx = static_cast<float>(rec.chunkX - originChunkX);
            float dz = static_cast<float>(rec.chunkZ - originChunkZ);

            glm::mat4 translation = glm::translate(glm::mat4(1.0f), glm::vec3(dx * CHUNK_SPACING, 0.0f, dz * CHUNK_SPACING));
            glm::mat4 finalModel = translation * it->second.normMatrix;

            glUniformMatrix4fv(
                glGetUniformLocation(shaderProgram, "model"),
                1, GL_FALSE, glm::value_ptr(finalModel));

            for (const Mesh &mesh : it->second.meshes)
            {
                glActiveTexture(GL_TEXTURE0);
                glBindTexture(GL_TEXTURE_2D, mesh.textureID);

                glBindVertexArray(mesh.VAO);

                glDrawElements(
                    GL_TRIANGLES,
                    mesh.indexCount,
                    GL_UNSIGNED_INT,
                    0);
            }
        }

        glfwSwapBuffers(window);
    }

    Log("RENDER", "Window closed - exiting render loop.");
    Log("RENDER", "Textures in cache at shutdown: " + std::to_string(g_TextureCache.size()));

    glfwTerminate();
    return 0;
}