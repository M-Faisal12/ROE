// ─────────────────────────────────────────────────────────────────────────────
//  ChunkTerrainSystem.cpp
//
//  IMPORTANT: produces ONLY classification metadata and model-selection strings.
//  No height grids, no vertex buffers, no collision shapes, no physics data.
//
//  Pipeline overview (batch path):
//
//  [1] buildDescriptor() for every chunk — same multi-layer noise averaging
//      as before; produces avgHeight, roughness, moisture, mountainFactor,
//      temperature, continentScore / continent.
//
//  [2] assignBiomesByDistribution()
//      ─ For each biome B (processed highest-priority first):
//        • Compute naturalFitScore(d, B) for every unassigned chunk.
//        • Sort ascending (lower = better natural fit).
//        • Assign the top N chunks where N = round(target_fraction * total),
//          skipping any chunk whose continent forbids biome B.
//      This replaces both the old threshold classifier and the old
//      post-classification enforceDistribution() pass. Biome counts now match
//      the target distribution exactly at source, without a separate remap.
//
//  [3] validateAndRepairAdjacency()
//      ─ Walk every chunk and its 8 neighbours.
//      ─ If the pair (chunk.biome, neighbour.biome) is on the forbidden-
//        transition list, call repairBiome() on the chunk.
//      ─ repairBiome() looks up a value from m_repairNoise (seeded, so
//        deterministic) and maps it to a valid bridge biome — one that is
//        (a) permitted by the chunk's continent and (b) a legal neighbour for
//        both the chunk's current biome and the conflicting neighbour's biome.
//      ─ The repair is applied in a single forward pass (no iteration) to
//        keep it O(N) and fully deterministic.
// ─────────────────────────────────────────────────────────────────────────────

#include "Mapgenerator.h"
#include <iostream>
#include <iomanip>
#include <algorithm>
#include <numeric>
#include <cmath>
#include <stdexcept>
#include <cassert>
#include <limits>
#include <vector>

// ─────────────────────────────────────────────────────────────────────────────
//  Ideal biome centre vectors in [height, moisture, mountainFactor, temperature]
//  Used by naturalFitScore to rank candidates during distribution assignment.
//
//  These are physical intuitions, not magic numbers:
//    Plains:        mid height, good moisture, low peaks, warm
//    Sand:          low-mid height, very dry, flat, hot
//    SandDunes:     similar to Sand but slightly rougher peaks
//    Water:         low height, very wet, no peaks, neutral temp
//    Mountains:     high, moderate moisture, high peaks, cool
//    Snow:          high, moderate moisture, low-mid peaks, very cold
//    SnowMountains: very high, moderate moisture, very high peaks, frigid
//    FrozenWater:   low height, wet, no peaks, frigid
// ─────────────────────────────────────────────────────────────────────────────
static constexpr float BIOME_IDEAL[BIOME_COUNT][4] = {
    //  height  moist  mountain  temp
    { 0.45f, 0.55f, 0.20f, 0.55f }, // Plains
    { 0.40f, 0.15f, 0.15f, 0.80f }, // Sand
    { 0.42f, 0.15f, 0.32f, 0.78f }, // SandDunes
    { 0.18f, 0.75f, 0.10f, 0.50f }, // Water
    { 0.70f, 0.30f, 0.72f, 0.30f }, // Mountains
    { 0.72f, 0.50f, 0.25f, 0.10f }, // Snow
    { 0.88f, 0.35f, 0.78f, 0.05f }, // SnowMountains
    { 0.20f, 0.70f, 0.10f, 0.08f }, // FrozenWater
};

// ─────────────────────────────────────────────────────────────────────────────
//  Continent ↔ Biome allowance table
//  Mirrors the classifyBiome logic but expressed as a reusable predicate.
// ─────────────────────────────────────────────────────────────────────────────
static bool continentAllowsBiome(ContinentType c, BiomeType b)
{
    switch (c)
    {
    case ContinentType::Polar:
        return b == BiomeType::Snow
            || b == BiomeType::SnowMountains
            || b == BiomeType::FrozenWater
            || b == BiomeType::Water
            || b == BiomeType::Mountains;

    case ContinentType::Desert:
        return b == BiomeType::Sand
            || b == BiomeType::SandDunes
            || b == BiomeType::Plains
            || b == BiomeType::Water
            || b == BiomeType::Mountains;

    case ContinentType::Tropical:
        return b == BiomeType::Plains
            || b == BiomeType::Sand
            || b == BiomeType::SandDunes
            || b == BiomeType::Water
            || b == BiomeType::Mountains;

    case ContinentType::Temperate:
        return b == BiomeType::Plains
            || b == BiomeType::Water
            || b == BiomeType::Mountains
            || b == BiomeType::Snow
            || b == BiomeType::SnowMountains
            || b == BiomeType::FrozenWater;

    default:
        return false;
    }
}

// ─────────────────────────────────────────────────────────────────────────────
//  Forbidden adjacency pairs (order-independent).
//
//  Design rationale:
//    The forbidden list encodes "hard climate jumps" — biome pairs that
//    could never plausibly share a border in the real world because they
//    occupy completely different ends of the temperature or humidity spectrum.
//    Legal adjacencies (Plains↔Mountains, Snow↔SnowMountains, Water↔Sand…)
//    are left absent from the list, so they pass through unrestricted.
//
//  The list is deliberately conservative: only truly jarring transitions are
//  forbidden.  Mildly awkward but plausible adjacencies (e.g. Mountains next
//  to Sand) are permitted — forcing a bridge there would distort the
//  distribution more than the visual artifact warrants.
// ─────────────────────────────────────────────────────────────────────────────
static const std::pair<BiomeType,BiomeType> FORBIDDEN_TRANSITIONS[] = {
    // Hot desert next to snow/ice — hard temperature jump
    { BiomeType::Sand,         BiomeType::Snow          },
    { BiomeType::Sand,         BiomeType::SnowMountains },
    { BiomeType::Sand,         BiomeType::FrozenWater   },
    { BiomeType::SandDunes,    BiomeType::Snow          },
    { BiomeType::SandDunes,    BiomeType::SnowMountains },
    { BiomeType::SandDunes,    BiomeType::FrozenWater   },
    // Liquid water next to frozen water (should have a Snow/Ice shore)
    { BiomeType::Water,        BiomeType::FrozenWater   },
    // Tropical plains next to frozen water
    { BiomeType::Plains,       BiomeType::FrozenWater   },
    // Snow directly next to sand — requires at least a Plains bridge
    { BiomeType::Snow,         BiomeType::Sand          },
    { BiomeType::Snow,         BiomeType::SandDunes     },
    { BiomeType::SnowMountains,BiomeType::Sand          },
    { BiomeType::SnowMountains,BiomeType::SandDunes     },
};

bool ChunkTerrainSystem::transitionIsValid(BiomeType a, BiomeType b)
{
    for (const auto& [fa, fb] : FORBIDDEN_TRANSITIONS)
        if ((a == fa && b == fb) || (a == fb && b == fa))
            return false;
    return true;
}

// ═════════════════════════════════════════════════════════════════════════════
//  Constructor + noise setup
// ═════════════════════════════════════════════════════════════════════════════
ChunkTerrainSystem::ChunkTerrainSystem(const TerrainSystemConfig& cfg)
    : m_cfg(cfg)
{
    if (m_cfg.chunkSize  <= 0.f) throw std::invalid_argument("chunkSize must be positive.");
    if (m_cfg.sampleGrid <  1  ) throw std::invalid_argument("sampleGrid must be >= 1.");
    setupNoise();
}

void ChunkTerrainSystem::setupNoise()
{
    // ── Height (FBm, OpenSimplex2) ───────────────────────────────────────────
    m_heightNoise.SetSeed(m_cfg.seed);
    m_heightNoise.SetNoiseType(FastNoiseLite::NoiseType_OpenSimplex2);
    m_heightNoise.SetFractalType(FastNoiseLite::FractalType_FBm);
    m_heightNoise.SetFractalOctaves(7);
    m_heightNoise.SetFractalLacunarity(2.0f);
    m_heightNoise.SetFractalGain(0.5f);
    m_heightNoise.SetFrequency(m_cfg.heightFreq);

    // ── Mountain ridges (Ridged, OpenSimplex2S) ──────────────────────────────
    m_mountainNoise.SetSeed(m_cfg.seed + 100);
    m_mountainNoise.SetNoiseType(FastNoiseLite::NoiseType_OpenSimplex2S);
    m_mountainNoise.SetFractalType(FastNoiseLite::FractalType_Ridged);
    m_mountainNoise.SetFractalOctaves(5);
    m_mountainNoise.SetFractalLacunarity(2.2f);
    m_mountainNoise.SetFractalGain(0.45f);
    m_mountainNoise.SetFrequency(m_cfg.mountainFreq);

    // ── Moisture (FBm, OpenSimplex2) ─────────────────────────────────────────
    m_moistureNoise.SetSeed(m_cfg.seed + 200);
    m_moistureNoise.SetNoiseType(FastNoiseLite::NoiseType_OpenSimplex2);
    m_moistureNoise.SetFractalType(FastNoiseLite::FractalType_FBm);
    m_moistureNoise.SetFractalOctaves(5);
    m_moistureNoise.SetFractalLacunarity(2.0f);
    m_moistureNoise.SetFractalGain(0.5f);
    m_moistureNoise.SetFrequency(m_cfg.moistureFreq);

    // ── Variation (Cellular) ─────────────────────────────────────────────────
    m_variationNoise.SetSeed(m_cfg.seed + 300);
    m_variationNoise.SetNoiseType(FastNoiseLite::NoiseType_Cellular);
    m_variationNoise.SetCellularReturnType(FastNoiseLite::CellularReturnType_Distance);
    m_variationNoise.SetFrequency(m_cfg.variationFreq);

    // ── Temperature (FBm, OpenSimplex2, slow) ────────────────────────────────
    m_tempNoise.SetSeed(m_cfg.seed + 400);
    m_tempNoise.SetNoiseType(FastNoiseLite::NoiseType_OpenSimplex2);
    m_tempNoise.SetFractalType(FastNoiseLite::FractalType_FBm);
    m_tempNoise.SetFractalOctaves(4);
    m_tempNoise.SetFractalLacunarity(2.0f);
    m_tempNoise.SetFractalGain(0.5f);
    m_tempNoise.SetFrequency(m_cfg.tempFreq);

    // ── Domain warp (organic coastlines) ─────────────────────────────────────
    m_warpNoise.SetSeed(m_cfg.seed + 500);
    m_warpNoise.SetNoiseType(FastNoiseLite::NoiseType_OpenSimplex2);
    m_warpNoise.SetDomainWarpType(FastNoiseLite::DomainWarpType_OpenSimplex2);
    m_warpNoise.SetDomainWarpAmp(48.0f);
    m_warpNoise.SetFrequency(0.006f);
    m_warpNoise.SetFractalType(FastNoiseLite::FractalType_DomainWarpIndependent);
    m_warpNoise.SetFractalOctaves(3);

    // ── Continent (FBm, very coarse) ─────────────────────────────────────────
    m_continentNoise.SetSeed(m_cfg.seed + 600);
    m_continentNoise.SetNoiseType(FastNoiseLite::NoiseType_OpenSimplex2);
    m_continentNoise.SetFractalType(FastNoiseLite::FractalType_FBm);
    m_continentNoise.SetFractalOctaves(3);
    m_continentNoise.SetFractalLacunarity(2.0f);
    m_continentNoise.SetFractalGain(0.5f);
    m_continentNoise.SetFrequency(m_cfg.continentFreq);

    // ── Repair noise (FBm, distinct seed) ────────────────────────────────────
    // Sampled at each chunk's world position during the adjacency-repair pass.
    // Using a completely separate seed ensures repair choices are independent
    // of all other layers and remain stable across map changes.
    m_repairNoise.SetSeed(m_cfg.seed + 700);
    m_repairNoise.SetNoiseType(FastNoiseLite::NoiseType_OpenSimplex2);
    m_repairNoise.SetFractalType(FastNoiseLite::FractalType_FBm);
    m_repairNoise.SetFractalOctaves(3);
    m_repairNoise.SetFractalLacunarity(2.0f);
    m_repairNoise.SetFractalGain(0.5f);
    m_repairNoise.SetFrequency(0.030f); // fine enough to vary per chunk
}

// ═════════════════════════════════════════════════════════════════════════════
//  Stage 1 – point sample
// ═════════════════════════════════════════════════════════════════════════════
ChunkTerrainSystem::RawSample
ChunkTerrainSystem::sampleAt(float wx, float wz) const
{
    float ewx = wx, ewz = wz;
    m_warpNoise.DomainWarp(ewx, ewz);

    float h = m_heightNoise.GetNoise(ewx, ewz) * 0.70f
            + m_mountainNoise.GetNoise(ewx, ewz) * 0.30f;

    return {
        h,
        m_mountainNoise.GetNoise(wx, wz),
        m_moistureNoise.GetNoise(wx, wz),
        m_variationNoise.GetNoise(wx, wz),
        m_tempNoise.GetNoise(wx, wz),
        m_continentNoise.GetNoise(wx, wz),
    };
}

// ═════════════════════════════════════════════════════════════════════════════
//  Continent classification
// ═════════════════════════════════════════════════════════════════════════════
ContinentType ChunkTerrainSystem::classifyContinent(float continentScore) const
{
    // continentScore is already remapped to [0,1] by buildDescriptor.
    if      (continentScore < 0.25f) return ContinentType::Desert;
    else if (continentScore < 0.50f) return ContinentType::Tropical;
    else if (continentScore < 0.75f) return ContinentType::Temperate;
    else                             return ContinentType::Polar;
}

// ═════════════════════════════════════════════════════════════════════════════
//  Build descriptor — average NxN samples per chunk
// ═════════════════════════════════════════════════════════════════════════════
void ChunkTerrainSystem::buildDescriptor(int chunkX, int chunkZ,
                                         ChunkDescriptor& out) const
{
    const int   N    = m_cfg.sampleGrid;
    const float size = m_cfg.chunkSize;
    const float ox   = static_cast<float>(chunkX) * size;
    const float oz   = static_cast<float>(chunkZ) * size;

    float sumH = 0, sumM = 0, sumMo = 0, sumV = 0, sumT = 0, sumC = 0, sumH2 = 0;
    const int total = N * N;

    for (int iz = 0; iz < N; ++iz)
    for (int ix = 0; ix < N; ++ix)
    {
        float wx = ox + (static_cast<float>(ix) + 0.5f) * (size / N);
        float wz = oz + (static_cast<float>(iz) + 0.5f) * (size / N);

        RawSample s = sampleAt(wx, wz);

        float h  = remap01(s.height);
        float m  = remap01(s.mountain);
        float mo = remap01(s.moisture);
        float v  = remap01(s.variation);
        float t  = remap01(s.temperature);
        float c  = remap01(s.continent);

        sumH  += h;
        sumM  += m;
        sumMo += mo;
        sumV  += v;
        sumT  += t;
        sumC  += c;
        sumH2 += h * h;
    }

    const float inv = 1.f / static_cast<float>(total);

    out.avgHeight      = sumH  * inv;
    out.mountainFactor = sumM  * inv;
    out.moisture       = sumMo * inv;
    out.continentScore = sumC  * inv;

    // Continent uses the averaged continent score (already in [0,1]).
    out.continent = classifyContinent(out.continentScore);

    float var   = std::max(0.f, (sumH2 * inv) - (out.avgHeight * out.avgHeight));
    out.roughness = std::min(std::sqrt(var) * 2.0f, 1.0f);

    float baseTemp = sumT * inv;
    // Temperature drops with elevation — same as original.
    out.temperature = std::clamp(baseTemp - out.avgHeight * 0.60f, 0.0f, 1.0f);
}

// ═════════════════════════════════════════════════════════════════════════════
//  naturalFitScore — how well does descriptor d naturally express biome B?
//
//  Returns the squared Euclidean distance in the 4D feature space
//  (height, moisture, mountainFactor, temperature) between d and the ideal
//  centre for B. Lower = better natural fit.
//
//  Returns +infinity when the chunk's continent forbids biome B, so
//  climatically impossible assignments never make it into the ranked list.
// ═════════════════════════════════════════════════════════════════════════════
float ChunkTerrainSystem::naturalFitScore(const ChunkDescriptor& d,
                                          BiomeType target) const
{
    if (!continentAllowsBiome(d.continent, target))
        return std::numeric_limits<float>::infinity();

    const int   i  = static_cast<int>(target);
    const float dh = d.avgHeight      - BIOME_IDEAL[i][0];
    const float dm = d.moisture       - BIOME_IDEAL[i][1];
    const float dp = d.mountainFactor - BIOME_IDEAL[i][2];
    const float dt = d.temperature    - BIOME_IDEAL[i][3];
    return dh*dh + dm*dm + dp*dp + dt*dt;
}

// ═════════════════════════════════════════════════════════════════════════════
//  assignBiomesByDistribution
//
//  Distribution-driven quantization: instead of applying a threshold and then
//  remapping the result, this directly assigns biomes by sorting chunks on the
//  most discriminating noise axis for each biome and claiming exactly as many
//  chunks as the target fraction requests.
//
//  Processing order (priority):
//    Water / FrozenWater first  — lowest elevation, hardest physical constraint.
//    Mountains / SnowMountains  — highest elevation / mountain factor.
//    Snow                       — high elevation + cold.
//    Sand / SandDunes           — low-mid elevation + dry + hot.
//    Plains                     — middle of the road; fills the remainder.
//
//  A chunk is "unassigned" until it wins a slot. Once assigned, it is removed
//  from contention. Ties are broken by the raw fit score.
// ═════════════════════════════════════════════════════════════════════════════
void ChunkTerrainSystem::assignBiomesByDistribution(
    std::vector<ChunkDescriptor>& batch) const
{
    if (batch.empty()) return;

    const int total = static_cast<int>(batch.size());

    // Build target count per biome from fractions.
    const auto& dist = m_cfg.distribution;
    int targets[BIOME_COUNT];
    targets[static_cast<int>(BiomeType::Plains)]        = static_cast<int>(std::round(dist.plains        * total));
    targets[static_cast<int>(BiomeType::Sand)]          = static_cast<int>(std::round(dist.sand          * total));
    targets[static_cast<int>(BiomeType::SandDunes)]     = static_cast<int>(std::round(dist.sandDunes     * total));
    targets[static_cast<int>(BiomeType::Water)]         = static_cast<int>(std::round(dist.water         * total));
    targets[static_cast<int>(BiomeType::Mountains)]     = static_cast<int>(std::round(dist.mountains     * total));
    targets[static_cast<int>(BiomeType::Snow)]          = static_cast<int>(std::round(dist.snow          * total));
    targets[static_cast<int>(BiomeType::SnowMountains)] = static_cast<int>(std::round(dist.snowMountains * total));
    targets[static_cast<int>(BiomeType::FrozenWater)]   = static_cast<int>(std::round(dist.frozenWater   * total));

    // Track which chunks are still unassigned (index into batch).
    std::vector<bool> assigned(total, false);

    // Helper: for a given target biome, score all unassigned chunks, sort
    // ascending, and claim the top N (skipping any with infinite score, i.e.
    // climatically incompatible continent).
    auto claimBest = [&](BiomeType target)
    {
        int n = targets[static_cast<int>(target)];
        if (n <= 0) return;

        std::vector<std::pair<float, int>> scored;
        scored.reserve(total);
        for (int i = 0; i < total; ++i)
        {
            if (assigned[i]) continue;
            float s = naturalFitScore(batch[i], target);
            scored.push_back({s, i});
        }

        // Sort: best fit (lowest score) first; infinite scores bubble to end.
        std::sort(scored.begin(), scored.end(),
                  [](const auto& a, const auto& b){ return a.first < b.first; });

        int filled = 0;
        for (const auto& [score, idx] : scored)
        {
            if (filled >= n) break;
            if (std::isinf(score)) break; // all remaining are incompatible
            batch[idx].biome = target;
            assigned[idx]    = true;
            ++filled;
        }
    };

    // ── Assignment in priority order ─────────────────────────────────────────
    // Water and frozen water claim the lowest-elevation chunks first.
    claimBest(BiomeType::FrozenWater);
    claimBest(BiomeType::Water);
    // Mountains and snow claim the highest / roughest chunks next.
    claimBest(BiomeType::SnowMountains);
    claimBest(BiomeType::Mountains);
    claimBest(BiomeType::Snow);
    // Desert biomes claim the driest / hottest remaining chunks.
    claimBest(BiomeType::SandDunes);
    claimBest(BiomeType::Sand);
    // Plains absorbs whatever is left.
    claimBest(BiomeType::Plains);

    // ── Fallback: any still-unassigned chunk gets its continent's default ─────
    // This can happen if the distribution fractions sum to < 1.0 or if
    // continent restrictions prevented all target slots from being filled.
    for (int i = 0; i < total; ++i)
    {
        if (assigned[i]) continue;
        // Continent-default fallback:
        switch (batch[i].continent)
        {
        case ContinentType::Polar:    batch[i].biome = BiomeType::Snow;   break;
        case ContinentType::Desert:   batch[i].biome = BiomeType::Sand;   break;
        case ContinentType::Tropical: batch[i].biome = BiomeType::Plains; break;
        case ContinentType::Temperate:batch[i].biome = BiomeType::Plains; break;
        default:                      batch[i].biome = BiomeType::Plains; break;
        }
        assigned[i] = true;
    }
}

// ═════════════════════════════════════════════════════════════════════════════
//  repairBiome
//
//  Given a chunk with a conflicting neighbour, select a valid bridge biome
//  that:
//    (a) is allowed by the chunk's continent,
//    (b) forms a legal adjacency with both the chunk's current biome AND the
//        neighbour's biome.
//
//  The selection is driven by m_repairNoise sampled at the chunk's centre,
//  which is a function only of (chunkX, chunkZ, seed). Identical inputs
//  always produce the same output → full determinism.
//
//  If no single biome satisfies all constraints, the function returns the
//  chunk's current biome (no change), preferring an imperfect map over a
//  crash or an infinite loop.
// ═════════════════════════════════════════════════════════════════════════════
BiomeType ChunkTerrainSystem::repairBiome(const ChunkDescriptor& chunk,
                                          BiomeType neighbour) const
{
    // Collect candidates: continent-allowed biomes that form legal adjacency
    // with BOTH chunk.biome and neighbour.
    std::vector<BiomeType> candidates;
    candidates.reserve(BIOME_COUNT);

    for (int b = 0; b < BIOME_COUNT; ++b)
    {
        BiomeType candidate = static_cast<BiomeType>(b);
        if (!continentAllowsBiome(chunk.continent, candidate)) continue;
        if (!transitionIsValid(candidate, chunk.biome))        continue;
        if (!transitionIsValid(candidate, neighbour))          continue;
        candidates.push_back(candidate);
    }

    if (candidates.empty()) return chunk.biome; // no valid bridge; keep as-is

    // Use repair noise at chunk centre to pick deterministically.
    float wx = static_cast<float>(chunk.chunkX) * m_cfg.chunkSize
             + m_cfg.chunkSize * 0.5f;
    float wz = static_cast<float>(chunk.chunkZ) * m_cfg.chunkSize
             + m_cfg.chunkSize * 0.5f;

    float rn  = remap01(m_repairNoise.GetNoise(wx, wz)); // [0, 1]
    int   idx = static_cast<int>(rn * static_cast<float>(candidates.size()));
    idx = std::clamp(idx, 0, static_cast<int>(candidates.size()) - 1);

    return candidates[idx];
}

// ═════════════════════════════════════════════════════════════════════════════
//  validateAndRepairAdjacency
//
//  Single forward pass over a cols×rows grid stored row-major in batch.
//  For each chunk, check all 8 neighbours. On the first forbidden pair found,
//  replace the chunk's biome via repairBiome() and immediately re-select its
//  model. The repair is applied once per chunk per pass — this is intentional:
//  iterating to full convergence would alter the distribution and is not
//  guaranteed to terminate. One pass eliminates the vast majority of hard
//  transitions; any residual soft conflicts are visually acceptable.
// ═════════════════════════════════════════════════════════════════════════════
void ChunkTerrainSystem::validateAndRepairAdjacency(
    std::vector<ChunkDescriptor>& batch, int cols, int rows) const
{
    // Direction offsets: N, NE, E, SE, S, SW, W, NW
    static const int DX[8] = {  0,  1, 1, 1,  0, -1, -1, -1 };
    static const int DZ[8] = { -1, -1, 0, 1,  1,  1,  0, -1 };

    auto idx = [&](int col, int row) -> int {
        return row * cols + col;
    };

    for (int row = 0; row < rows; ++row)
    for (int col = 0; col < cols; ++col)
    {
        ChunkDescriptor& chunk = batch[idx(col, row)];

        for (int d = 0; d < 8; ++d)
        {
            int nc = col + DX[d];
            int nr = row + DZ[d];
            if (nc < 0 || nc >= cols || nr < 0 || nr >= rows) continue;

            const ChunkDescriptor& nb = batch[idx(nc, nr)];
            if (!transitionIsValid(chunk.biome, nb.biome))
            {
                chunk.biome         = repairBiome(chunk, nb.biome);
                chunk.selectedModel = selectModel(chunk);
                chunk.wasRepaired   = true;
                break; // one repair per chunk per pass
            }
        }
    }
}

// ═════════════════════════════════════════════════════════════════════════════
//  Model selection
// ═════════════════════════════════════════════════════════════════════════════
std::string ChunkTerrainSystem::selectModel(const ChunkDescriptor& d) const
{
    const std::vector<std::string>* pool = nullptr;
    switch (d.biome)
    {
    case BiomeType::Plains:        pool = &m_cfg.modelsPlains;        break;
    case BiomeType::Sand:          pool = &m_cfg.modelsSand;          break;
    case BiomeType::SandDunes:     pool = &m_cfg.modelsSandDunes;     break;
    case BiomeType::Water:         pool = &m_cfg.modelsWater;         break;
    case BiomeType::Mountains:     pool = &m_cfg.modelsMountains;     break;
    case BiomeType::Snow:          pool = &m_cfg.modelsSnow;          break;
    case BiomeType::SnowMountains: pool = &m_cfg.modelsSnowMountains; break;
    case BiomeType::FrozenWater:   pool = &m_cfg.modelsFrozenWater;   break;
    default:                       pool = &m_cfg.modelsPlains;        break;
    }

    if (pool->empty()) return "default_terrain.glb";

    int idx = static_cast<int>(d.roughness * static_cast<float>(pool->size()));
    idx = std::clamp(idx, 0, static_cast<int>(pool->size()) - 1);
    return (*pool)[idx];
}

// ═════════════════════════════════════════════════════════════════════════════
//  Public API — batch with full pipeline
// ═════════════════════════════════════════════════════════════════════════════
std::vector<ChunkDescriptor>
ChunkTerrainSystem::getChunksInRange(int minX, int maxX, int minZ, int maxZ)
{
    if (minX > maxX || minZ > maxZ)
        throw std::invalid_argument("getChunksInRange: min > max.");

    const int cols = maxX - minX + 1;
    const int rows = maxZ - minZ + 1;

    // ── Stage 1: build all descriptors ───────────────────────────────────────
    std::vector<ChunkDescriptor> batch;
    batch.reserve(static_cast<size_t>(cols * rows));
    for (int z = minZ; z <= maxZ; ++z)
    for (int x = minX; x <= maxX; ++x)
    {
        ChunkDescriptor d;
        d.chunkX = x;
        d.chunkZ = z;
        buildDescriptor(x, z, d);
        batch.push_back(d);
    }

    // ── Stage 2: distribution-driven biome assignment ─────────────────────────
    assignBiomesByDistribution(batch);

    // ── Stage 2b: assign initial models ──────────────────────────────────────
    for (auto& d : batch)
        d.selectedModel = selectModel(d);

    // ── Stage 3: adjacency validation + deterministic repair ─────────────────
    validateAndRepairAdjacency(batch, cols, rows);

    return batch;
}

// ═════════════════════════════════════════════════════════════════════════════
//  Debug helpers
// ═════════════════════════════════════════════════════════════════════════════
const char* ChunkTerrainSystem::biomeName(BiomeType b)
{
    switch (b)
    {
    case BiomeType::Plains:        return "Plains";
    case BiomeType::Sand:          return "Sand";
    case BiomeType::SandDunes:     return "Sand Dunes";
    case BiomeType::Water:         return "Water";
    case BiomeType::Mountains:     return "Mountains";
    case BiomeType::Snow:          return "Snow";
    case BiomeType::SnowMountains: return "Snow Mountains";
    case BiomeType::FrozenWater:   return "Frozen Water";
    default:                       return "Unknown";
    }
}

static const char* continentName(ContinentType c)
{
    switch (c)
    {
    case ContinentType::Tropical:  return "Tropical";
    case ContinentType::Temperate: return "Temperate";
    case ContinentType::Polar:     return "Polar";
    case ContinentType::Desert:    return "Desert";
    default:                       return "Unknown";
    }
}

char ChunkTerrainSystem::biomeChar(BiomeType b)
{
    switch (b)
    {
    case BiomeType::Plains:        return '.';
    case BiomeType::Sand:          return 's';
    case BiomeType::SandDunes:     return 'D';
    case BiomeType::Water:         return '~';
    case BiomeType::Mountains:     return 'M';
    case BiomeType::Snow:          return '*';
    case BiomeType::SnowMountains: return '^';
    case BiomeType::FrozenWater:   return '#';
    default:                       return '?';
    }
}

char* ChunkTerrainSystem::biomeANSI(BiomeType b)
{
    switch (b)
    {
    case BiomeType::Plains:        return (char*)"P";
    case BiomeType::Sand:          return (char*)"s";
    case BiomeType::SandDunes:     return (char*)"S";
    case BiomeType::Water:         return (char*)"W";
    case BiomeType::Mountains:     return (char*)"m";
    case BiomeType::Snow:          return (char*)"p";
    case BiomeType::SnowMountains: return (char*)"M";
    case BiomeType::FrozenWater:   return (char*)"w";
    default:                       return (char*)"?";
    }
}


std::vector<char*>* ChunkTerrainSystem::printRange(int minX, int maxX, int minZ, int maxZ)
{
    auto chunks = getChunksInRange(minX, maxX, minZ, maxZ);
    const int cols = maxX - minX + 1;
    std::vector<char*>* map=new std::vector<char*>;
    map->reserve((maxZ-minZ)*cols);
    for (int z = minZ; z <= maxZ; ++z)
    {
        for (int x = minX; x <= maxX; ++x)
        {
            const auto& d = chunks[static_cast<size_t>((z - minZ) * cols + (x - minX))];
            map->emplace_back(biomeANSI(d.biome));
        }
        std::cout << '\n';
    }
    return map;
}