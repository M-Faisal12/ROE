// =============================================================================
// TextureCache.cpp
//
//   Implementation of texture loading + the shared GPU texture cache.
//   See TextureCache.h for the contract each function follows.
// =============================================================================
#include "TextureCache.h"
#include "Logger.h"

#include "stb_image.h"

#include <sstream>
#include <iomanip>
#include <cstdint>
#include <vector>

// =============================================================================
// TextureCache (definition of the extern declared in the header)
// =============================================================================
std::unordered_map<std::string, GLuint> g_TextureCache;

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