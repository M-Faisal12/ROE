// =============================================================================
// main.cpp  –  Terrain Renderer  +  Universal Model Normalizer
//
//   This file now only contains: window/GL setup, the Assimp multi-model
//   loading loop driven by the map (placedRecords/registry), camera
//   auto-fit, and the render loop.
//
//   Everything else has been split out:
//     Logger.h              -- shared Log()
//     MapGrid.h/.cpp         -- MapGrid struct + Map.bin streaming loader
//     TextureCache.h/.cpp     -- texture cache + file/embedded texture loading
//     Mesh.h                  -- Mesh + BoundingBox structs
//     Shaders.h/.cpp          -- shader source + CreateShaderProgram()
//     ModelNormalize.h/.cpp   -- ComputeNormalizationMatrix + IsLikelyTerrainMesh
//     HeightGrid.h/.cpp       -- HeightGrid struct + BuildHeightGrid + getTerrainHeight
//
//   See each header for the detailed contract of its system. Hand the AI
//   assistant only the specific .h/.cpp pair you're changing, plus this
//   file only if the change touches loading/placement/render-loop logic.
//
// Rendering pipeline (in order of execution):
//   1.  Load mesh geometry + normals + UV coords from a glTF/GLB/FBX/OBJ file
//       via Assimp
//   2.  For each mesh, resolve its diffuse texture path from the material
//   3.  Upload interleaved vertex data  (pos · normal · uv)  to the GPU
//   4.  Vertex shader  – transforms positions, corrects normals, passes UVs
//   5.  Fragment shader – samples the diffuse texture, applies diffuse lighting
//
// MAP-FROM-BINARY SYSTEM   (ADDED FOR MAP LOADER)
//   See MapGrid.h for the full file-format contract. In short: this program
//   builds an empty (width x height) grid up front, then streams
//   3DModels/Map.bin record-by-record, filling matching cells as each
//   record arrives. Records outside the configured bounds are skipped
//   (logged, not fatal).
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
// STB_IMAGE_IMPLEMENTATION must be defined exactly once in the whole
// program. This is that one place; TextureCache.cpp includes stb_image.h
// without defining this macro.
#define STB_IMAGE_IMPLEMENTATION
#include "../Externals/stb_image.h"

#include "Logger.h"
#include "MapGrid.h"
#include "TextureCache.h"
#include "Meshe.h"
#include "Shaders.h"
#include "ModelNormalize.h"
#include "HeightGrid.h"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <string>
#include <unordered_map>
#include <vector>
#include <memory>
#include <filesystem>

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
    // -------------------------------------------------------------------------
    Log("MAP", "Building empty " + std::to_string(mapWidth) + "x" + std::to_string(mapHeight) + " map ...");

    // Locate Map.bin: try a small set of likely locations and the compile-time
    // resources path (if available). This avoids failing when the program is
    // launched from a different working directory.
    namespace fs = std::filesystem;
    auto find_map_bin = [&]() -> std::string {
        std::vector<std::string> candidates = {
            std::string("./engine/resources/Map.bin"),
            std::string("engine/resources/Map.bin"),
            std::string("./resources/Map.bin"),
            std::string("resources/Map.bin"),
            std::string("./Map.bin"),
            std::string("Map.bin")
        };

#ifdef ROE_RESOURCES_PATH
        candidates.insert(candidates.begin(), std::string(ROE_RESOURCES_PATH) + "/Map.bin");
#endif

        for (const auto &c : candidates)
        {
            try {
                if (fs::exists(c))
                    return c;
            } catch (...) {}
        }
        // fallback: return first candidate (so caller sees the same error message)
        return candidates.front();
    };

    std::string mapPath = find_map_bin();
    Log("MAP", "Attempting to load map from: " + mapPath);

    MapGrid worldMap = LoadMapFromBinary(mapPath, mapWidth, mapHeight, mapCenterX, mapCenterZ);

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
    const std::string sceneDirectory = "assets";

    // Helper: find asset file by trying likely candidate locations. This
    // allows running the binary from build/ or repo root without failing.
    auto find_asset_file = [&](const std::string &name) -> std::string {
        std::vector<std::string> candidates;

#ifdef ROE_ASSETS_PATH
        candidates.push_back(std::string(ROE_ASSETS_PATH) + "/" + name);
#endif
        candidates.push_back("./" + sceneDirectory + "/" + name);
        candidates.push_back(sceneDirectory + "/" + name);
        candidates.push_back("../" + sceneDirectory + "/" + name);
        candidates.push_back("./" + name);
        candidates.push_back(name);

        for (const auto &c : candidates)
        {
            try {
                if (std::filesystem::exists(c))
                    return c;
            } catch (...) {}
        }

        // fallback: prefer the runtime-relative sceneDirectory path
        return sceneDirectory + "/" + name;
    };

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
        std::string modelPath = find_asset_file(modelName);

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
    //
    //   After normalization the model always sits at a known size
    //   (TARGET_SIZE) on its largest axis and on the ground plane (Y = 0).
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