#pragma once
// ─────────────────────────────────────────────────────────────────────────────
//  ChunkTerrainSystem.h
//
//  Chunk-based terrain CLASSIFICATION and GLB model selection.
//  Generates ONLY metadata (biome type + model filename) per chunk.
//  NO collision grids, NO vertex meshes, NO physics data.
//
//  Pipeline (per batch):
//    1. Sample 6 noise layers per chunk (height, mountain, moisture,
//       variation, temperature, continent) — same as before.
//    2. DISTRIBUTION-DRIVEN QUANTIZATION: sort chunks by each relevant
//       noise axis and assign biomes so the actual counts hit the target
//       fractions exactly, replacing the old threshold + post-remap approach.
//    3. CONTINENT GATE: every assignment is filtered through
//       continentAllowsBiome() so climatically impossible biomes never appear.
//    4. ADJACENCY VALIDATION (8 directions): scan every chunk against its
//       8 neighbours. Any pair whose biomes are on the "impossible transition"
//       list is flagged.
//    5. DETERMINISTIC REPAIR: flagged chunks are resolved via a seeded
//       m_repairNoise lookup that picks a valid intermediate biome. The same
//       seed always produces the same repair, so the map is fully deterministic.
//
//  DETERMINISM: same (chunkX, chunkZ, seed) → always identical output.
// ─────────────────────────────────────────────────────────────────────────────

#include "FastNoiseLite.h"
#include <string>
#include <vector>
#include <array>
#include <unordered_map>
#include <utility>

// ─────────────────────────────────────────────────────────────────────────────
//  ContinentType
// ─────────────────────────────────────────────────────────────────────────────
enum class ContinentType : int
{
    Tropical  = 0,
    Temperate = 1,
    Polar     = 2,
    Desert    = 3,
    COUNT     = 4
};
static constexpr int CONTINENT_COUNT = static_cast<int>(ContinentType::COUNT);

// ─────────────────────────────────────────────────────────────────────────────
//  BiomeType
// ─────────────────────────────────────────────────────────────────────────────
enum class BiomeType : int
{
    Plains        = 0,
    Sand          = 1,
    SandDunes     = 2,
    Water         = 3,
    Mountains     = 4,
    Snow          = 5,
    SnowMountains = 6,
    FrozenWater   = 7,
    COUNT         = 8
};
static constexpr int BIOME_COUNT = static_cast<int>(BiomeType::COUNT);

// ─────────────────────────────────────────────────────────────────────────────
//  ChunkDescriptor
// ─────────────────────────────────────────────────────────────────────────────
struct ChunkDescriptor
{
    int chunkX = 0;
    int chunkZ = 0;

    float avgHeight      = 0.f;
    float roughness      = 0.f;
    float moisture       = 0.f;
    float mountainFactor = 0.f;
    float temperature    = 0.f;

    ContinentType continent    = ContinentType::Temperate;
    float         continentScore = 0.f;

    BiomeType   biome;
    std::string selectedModel;

    // Set during adjacency validation; true if this chunk was repaired.
    bool wasRepaired = false;
};

// ─────────────────────────────────────────────────────────────────────────────
//  BiomeDistribution – target fraction for each biome (must sum ≤ 1.0)
// ─────────────────────────────────────────────────────────────────────────────
struct BiomeDistribution
{
    float plains        = 0.15f;
    float sand          = 0.15f;
    float snow          = 0.15f;
    float sandDunes     = 0.10f;
    float mountains     = 0.10f;
    float snowMountains = 0.10f;
    float water         = 0.10f;
    float frozenWater   = 0.10f;
    // Remainder (if sum < 1) falls naturally to the continent's dominant biome.
};

// ─────────────────────────────────────────────────────────────────────────────
//  TerrainSystemConfig
// ─────────────────────────────────────────────────────────────────────────────
struct TerrainSystemConfig
{
    int   seed       = 1337;
    float chunkSize  = 64.f;
    int   sampleGrid = 5;

    float heightFreq     = 0.018f;
    float mountainFreq   = 0.035f;
    float moistureFreq   = 0.022f;
    float variationFreq  = 0.055f;
    float tempFreq       = 0.012f;
    float continentFreq  = 0.0045f;

    BiomeDistribution distribution;

    // Model pools per biome (using available assets from assets/ folder)
    std::vector<std::string> modelsPlains        = {"t27.glb"};
    std::vector<std::string> modelsSand          = {"dunes_01.glb"};
    std::vector<std::string> modelsSandDunes     = {"dunes_01.glb"};
    std::vector<std::string> modelsWater         = {"water_01.glb"};
    std::vector<std::string> modelsMountains     = {"mountain_01.glb"};
    std::vector<std::string> modelsSnow          = {"Snow01.glb"};
    std::vector<std::string> modelsSnowMountains = {"snow_mtn_01.glb"};
    std::vector<std::string> modelsFrozenWater   = {"ice_01.glb","ice_03.glb","ice_04.glb","ice_05.glb"};
};

// ─────────────────────────────────────────────────────────────────────────────
//  ChunkTerrainSystem
// ─────────────────────────────────────────────────────────────────────────────
class ChunkTerrainSystem
{
public:
    explicit ChunkTerrainSystem(const TerrainSystemConfig& cfg = {});

    // Single-chunk query (pure noise, no distribution enforcement).
    ChunkDescriptor getChunk(int chunkX, int chunkZ) const;

    // Batch query with full pipeline:
    //   noise → distribution-driven quantization → adjacency validation → repair.
    std::vector<ChunkDescriptor> getChunksInRange(int minX, int maxX,
                                                   int minZ, int maxZ);

    void printChunk(const ChunkDescriptor& d) const;
    std::vector<char*>* printRange(int minX, int maxX, int minZ, int maxZ);

    const TerrainSystemConfig& config() const { return m_cfg; }

    static const char* biomeName(BiomeType b);
    static char        biomeChar(BiomeType b);
    static char* biomeANSI(BiomeType b);
    static bool        transitionIsValid(BiomeType a, BiomeType b);

private:
    TerrainSystemConfig m_cfg;

    FastNoiseLite m_heightNoise;
    FastNoiseLite m_mountainNoise;
    FastNoiseLite m_moistureNoise;
    FastNoiseLite m_variationNoise;
    FastNoiseLite m_tempNoise;
    FastNoiseLite m_warpNoise;
    FastNoiseLite m_continentNoise;
    // Used exclusively in the deterministic repair pass.
    FastNoiseLite m_repairNoise;

    void setupNoise();

    struct RawSample
    {
        float height, mountain, moisture, variation, temperature, continent;
    };
    RawSample sampleAt(float wx, float wz) const;

    ContinentType classifyContinent(float continentScore) const;

    void buildDescriptor(int chunkX, int chunkZ, ChunkDescriptor& out) const;

    // ── Distribution-driven quantization ─────────────────────────────────────
    // Assigns biomes to the whole batch so actual counts match target fractions.
    // Each biome "slot" is filled by the best-fitting chunks (scored via a
    // noise-axis ranking specific to that biome), respecting continent gates.
    void assignBiomesByDistribution(std::vector<ChunkDescriptor>& batch) const;

    // Score how naturally a descriptor fits a given biome (lower = better fit).
    float naturalFitScore(const ChunkDescriptor& d, BiomeType target) const;

    // ── Adjacency validation & repair ────────────────────────────────────────
    // Chooses a climatically valid bridge biome deterministically from noise.
    BiomeType repairBiome(const ChunkDescriptor& chunk,
                          BiomeType neighbour) const;

    // Full 8-direction pass over a rectangular batch.
    void validateAndRepairAdjacency(std::vector<ChunkDescriptor>& batch,
                                    int cols, int rows) const;

    // Model selection
    std::string selectModel(const ChunkDescriptor& d) const;

    static float remap01(float v) { return v * 0.5f + 0.5f; }
};