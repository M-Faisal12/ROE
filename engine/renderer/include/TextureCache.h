// =============================================================================
// TextureCache.h
//
//   Texture loading (file-based + embedded GLB textures) and the shared
//   GPU texture cache.
//
//   All loaded textures live in a single unordered_map<key, GLuint>.
//   Multiple meshes that reference the same file (or identical embedded
//   image content) share one GPU texture object. The cache also drives
//   all log output so you can see reuse vs. new loads.
//
//   .GLB notes:
//     A .glb file is a self-contained binary glTF bundle. Textures are
//     stored inside the file itself rather than as separate image files
//     on disk. Assimp references them with an asterisk-prefixed index
//     string ("*0", "*1" ...) instead of a file path. GetOrLoadEmbeddedTexture
//     detects that prefix and decodes the compressed bytes directly from
//     the aiScene via stbi_load_from_memory.
// =============================================================================
#pragma once

#include "../Externals/Glad/glad.h"
#include <assimp/scene.h>

#include <string>
#include <unordered_map>

// Shared GPU texture cache: key -> GL texture handle.
// Key is either a resolved file path, or "*embedded_<hash>" for embedded
// textures (content-hashed so identical embedded images across different
// files/indices share one GPU texture).
extern std::unordered_map<std::string, GLuint> g_TextureCache;

// Loads an image file from disk straight to a new GPU texture. Does NOT
// check or update the cache -- callers should go through GetOrLoadTexture.
GLuint LoadTextureFromFile(const std::string &filePath);

// Decodes one embedded aiTexture (compressed or raw) from `scene` and
// uploads it to a new GPU texture. Does NOT check or update the cache --
// callers should go through GetOrLoadEmbeddedTexture.
GLuint LoadEmbeddedTexture(const aiScene *scene, const std::string &embeddedRef);

// Cache-checked file texture load. Returns existing handle on a cache hit,
// otherwise loads from disk and stores the result.
GLuint GetOrLoadTexture(const std::string &filePath);

// Cache-checked embedded texture load (content-hash keyed). Returns
// existing handle on a cache hit, otherwise decodes and stores the result.
GLuint GetOrLoadEmbeddedTexture(const aiScene *scene, const std::string &embeddedRef);

// Resolves mesh `meshIndex`'s diffuse texture reference from its material.
// Returns "" if the mesh has no diffuse texture. Sets outIsEmbedded=true
// and returns the raw "*N" reference for embedded (.glb) textures;
// otherwise returns sceneDirectory + "/" + relative path.
std::string ResolveMeshTexturePath(
    const aiScene *scene,
    unsigned int meshIndex,
    const std::string &sceneDirectory,
    bool &outIsEmbedded);