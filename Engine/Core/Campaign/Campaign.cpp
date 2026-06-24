/*
    =============================================================
    PROCEDURAL WORLD GENERATOR  –  2.5D OpenGL Renderer
    =============================================================

    BUILD (Linux / macOS):
        g++ -std=c++17 worldgen.cpp -o worldgen \
            -lGL -lGLU -lglfw -lm \
            -I<path-to-FastNoiseLite-header>

    BUILD (Windows / MSVC):
        Adjust include / lib paths for GLFW + OpenGL accordingly.
        FastNoiseLite is header-only; just point -I at the folder.

    DEPENDENCY:
        FastNoiseLite (single header):
        https://github.com/Auburn/FastNoiseLite/blob/master/Cpp/FastNoiseLite.h

    CONTROLS:
        W/A/S/D or Arrow keys : pan camera
        Q / E                 : zoom out / in
        R                     : regenerate new random world
        ESC                   : quit
    =============================================================

    PIPELINE OVERVIEW
    -----------------
    1.  DOMAIN WARP            – distort UV coords so everything
                                 feels organic, not grid-aligned.

    2.  HEIGHT MAP             – continent base + fBm hills +
                                 ridged-multifractal mountains.
                                 All sampled at warped coords.

    3.  THERMAL EROSION        – iterative slope-relaxation pass
                                 that rounds off spike noise into
                                 believable terrain shapes.

    4.  MOISTURE MAP           – independent fBm noise, biased
                                 slightly toward rivers / coasts
                                 (lower tiles get a moisture bump).

    5.  TEMPERATURE MAP        – latitude gradient + noise offset;
                                 high tiles get a cold penalty.

    6.  BIOME CLASSIFICATION   – Whittaker-style lookup:
                                   height  →  water vs land
                                   moisture + temperature → biome

    7.  RIVER SIMULATION       – greedy downhill walk from high
                                 peaks; tolerates small climbs so
                                 rivers don't die in potholes.

    8.  OPENGL 2.5D RENDER     – each tile drawn as a quad at a
                                 height-offset screen position
                                 (isometric projection).  Side
                                 faces rendered for cliff depth.
                                 Per-vertex shading from a fixed
                                 directional light approximates
                                 hillshading / ambient occlusion.
*/

#include <iostream>
#include <vector>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <algorithm>
#include <string>
#include <functional>

#include <GLFW/glfw3.h>

/* ── FastNoiseLite ─────────────────────────────────────────── */
/* Adjust this path to wherever you placed FastNoiseLite.h      */
#include "../../../Externals/NoiseAlgo/FastNoiseLite.h"

using namespace std;


/* ════════════════════════════════════════════════════════════
   CONSTANTS
   ════════════════════════════════════════════════════════════ */

static const int   MAP_W      = 150;   // tiles wide
static const int   MAP_H      = 150;   // tiles tall

// 2.5D isometric tile dimensions (screen pixels)
static const float ISO_TW     = 16.0f; // tile half-width  (diamond)
static const float ISO_TH     = 8.0f;  // tile half-height (diamond)
static const float ISO_DEPTH  = 4.0f;  // pixels per unit of height for side face

// Height thresholds
static const float H_DEEP_OCEAN  =  2.0f;
static const float H_OCEAN       =  8.0f;
static const float H_SHORE       = 12.0f;
static const float H_PLAIN       = 18.0f;
static const float H_HILL        = 32.0f;
static const float H_MOUNTAIN    = 55.0f;
static const float H_SNOW        = 70.0f;


/* ════════════════════════════════════════════════════════════
   BIOME ENUM
   ════════════════════════════════════════════════════════════ */

enum class Biome
{
    DeepOcean,
    Ocean,
    Shore,
    Beach,
    Desert,
    Savanna,
    Grassland,
    Forest,
    TropicalRainforest,
    Taiga,
    Tundra,
    Hill,
    Mountain,
    SnowPeak,
    River
};


/* ════════════════════════════════════════════════════════════
   TILE STRUCT  – everything needed for render + simulation
   ════════════════════════════════════════════════════════════ */

struct Tile
{
    // Grid coordinates
    int   gx = 0;
    int   gy = 0;

    // Raw scalar fields (all 0-1 normalised after generation)
    float height      = 0.0f;  // 0 = lowest depression, 1 = highest peak
    float moisture    = 0.0f;  // 0 = arid, 1 = wet
    float temperature = 0.0f;  // 0 = arctic, 1 = tropical

    // Classification
    Biome biome = Biome::Grassland;
    bool  isRiver = false;
    bool  isWater = false;   // ocean/river combined flag for flow sim

    // Lighting / shading (0-1), computed from neighbour height delta
    float normalX = 0.0f;
    float normalY = 1.0f;
    float shade   = 1.0f;   // 0 = fully shadowed, 1 = fully lit

    // Screen-space isometric position (top-left of top face)
    float screenX = 0.0f;
    float screenY = 0.0f;

    // Height in screen pixels (for stacking side faces)
    float screenH = 0.0f;

    // RGBA colour for top face (filled during biome assignment)
    float r = 0.5f, g = 0.5f, b = 0.5f, a = 1.0f;

    // Side-face colour (slightly darkened version of top)
    float sr = 0.3f, sg = 0.3f, sb = 0.3f;
};


/* ════════════════════════════════════════════════════════════
   UTILITY
   ════════════════════════════════════════════════════════════ */

static float clamp01(float x)
{
    return x < 0.f ? 0.f : (x > 1.f ? 1.f : x);
}

static float lerp(float a, float b, float t)
{
    return a + (b - a) * clamp01(t);
}

static float smoothstep(float a, float b, float x)
{
    x = clamp01((x - a) / (b - a));
    return x * x * (3.f - 2.f * x);
}

/* remap noise from [-1,1] to [0,1] */
static float n01(float n) { return (n + 1.f) * 0.5f; }

/* darken a colour by factor f */
static void darken(float r, float g, float b,
                   float f,
                   float& or_, float& og_, float& ob_)
{
    or_ = clamp01(r * f);
    og_ = clamp01(g * f);
    ob_ = clamp01(b * f);
}


/* ════════════════════════════════════════════════════════════
   THERMAL EROSION
   ════════════════════════════════════════════════════════════ */

static void thermalErode(
    vector<vector<float>>& hm,
    int W, int H,
    int   iterations,
    float talus,
    float rate)
{
    vector<vector<float>> delta(W, vector<float>(H, 0.f));

    for(int it = 0; it < iterations; ++it)
    {
        for(auto& row : delta) fill(row.begin(), row.end(), 0.f);

        for(int x = 0; x < W; ++x)
        for(int y = 0; y < H; ++y)
        {
            float cur = hm[x][y];
            float totalDiff = 0.f;

            struct N { int nx, ny; float diff; };
            vector<N> lows;

            for(int dx = -1; dx <= 1; ++dx)
            for(int dy = -1; dy <= 1; ++dy)
            {
                if(!dx && !dy) continue;
                int nx = x+dx, ny = y+dy;
                if(nx<0||nx>=W||ny<0||ny>=H) continue;
                float d = cur - hm[nx][ny];
                if(d > talus) { lows.push_back({nx,ny,d}); totalDiff += d; }
            }

            if(lows.empty()) continue;

            float budget = rate * min(totalDiff, cur);

            for(auto& n : lows)
            {
                float mv = (n.diff / totalDiff) * budget;
                delta[x][y]       -= mv;
                delta[n.nx][n.ny] += mv;
            }
        }

        for(int x=0;x<W;++x)
        for(int y=0;y<H;++y)
            hm[x][y] += delta[x][y];
    }
}


/* ════════════════════════════════════════════════════════════
   BIOME COLOUR TABLE
   ════════════════════════════════════════════════════════════ */

struct RGB { float r,g,b; };

static RGB biomeColor(Biome b, float height, float shade)
{
    RGB c {};

    switch(b)
    {
        // Water bodies – depth-shaded blue
        case Biome::DeepOcean:  c = {0.06f, 0.16f, 0.42f}; break;
        case Biome::Ocean:      c = {0.10f, 0.28f, 0.60f}; break;
        case Biome::Shore:
        case Biome::Beach:      c = {0.76f, 0.70f, 0.50f}; break;  // sandy
        case Biome::River:      c = {0.20f, 0.50f, 0.80f}; break;

        // Dry biomes
        case Biome::Desert:     c = {0.87f, 0.78f, 0.46f}; break;
        case Biome::Savanna:    c = {0.70f, 0.72f, 0.30f}; break;

        // Temperate
        case Biome::Grassland:  c = {0.34f, 0.62f, 0.22f}; break;
        case Biome::Forest:     c = {0.13f, 0.45f, 0.15f}; break;
        case Biome::TropicalRainforest: c = {0.06f, 0.35f, 0.10f}; break;

        // Cold biomes
        case Biome::Taiga:      c = {0.20f, 0.40f, 0.28f}; break;
        case Biome::Tundra:     c = {0.58f, 0.62f, 0.55f}; break;

        // Elevation
        case Biome::Hill:       c = {0.45f, 0.42f, 0.35f}; break;
        case Biome::Mountain:   c = {0.52f, 0.50f, 0.48f}; break;
        case Biome::SnowPeak:   c = {0.92f, 0.93f, 0.97f}; break;
    }

    // Apply directional shading
    c.r *= shade;
    c.g *= shade;
    c.b *= shade;

    // Subtle height tint: slightly lighten peaks, darken valleys
    float htint = 0.85f + 0.3f * height;
    c.r = clamp01(c.r * htint);
    c.g = clamp01(c.g * htint);
    c.b = clamp01(c.b * htint);

    return c;
}


/* ════════════════════════════════════════════════════════════
   BIOME CLASSIFICATION

   Uses a simplified Whittaker diagram approach:
     height   → eliminates water / forces snow at peaks
     temperature + moisture → land biome
   ════════════════════════════════════════════════════════════ */

static Biome classifyBiome(float h, float m, float t)
{
    // ── Water ──────────────────────────────────────────────
    if(h < H_DEEP_OCEAN / 80.f) return Biome::DeepOcean;
    if(h < H_OCEAN      / 80.f) return Biome::Ocean;
    if(h < H_SHORE      / 80.f) return Biome::Shore;

    // ── Snow cap ───────────────────────────────────────────
    if(h > H_SNOW / 80.f)       return Biome::SnowPeak;

    // ── Rock mountain ──────────────────────────────────────
    if(h > H_MOUNTAIN / 80.f)   return Biome::Mountain;

    // ── Hill ───────────────────────────────────────────────
    if(h > H_HILL / 80.f)       return Biome::Hill;

    // ── Beach / shore transition ───────────────────────────
    if(h < H_PLAIN / 80.f && m < 0.35f) return Biome::Beach;

    // ── Land biomes from (temperature, moisture) ───────────
    // cold
    if(t < 0.2f)
        return (m < 0.4f) ? Biome::Tundra : Biome::Taiga;

    // sub-polar
    if(t < 0.35f)
        return (m < 0.5f) ? Biome::Tundra : Biome::Taiga;

    // temperate
    if(t < 0.6f)
    {
        if(m < 0.25f) return Biome::Desert;
        if(m < 0.50f) return Biome::Grassland;
        return Biome::Forest;
    }

    // warm / tropical
    if(m < 0.20f) return Biome::Desert;
    if(m < 0.45f) return Biome::Savanna;
    if(m < 0.70f) return Biome::Grassland;
    return Biome::TropicalRainforest;
}


/* ════════════════════════════════════════════════════════════
   WORLD GENERATION
   ════════════════════════════════════════════════════════════ */

static vector<vector<Tile>> generateWorld()
{
    vector<vector<Tile>> grid(MAP_W, vector<Tile>(MAP_H));

    /* ── Noise sources ──────────────────────────────────── */

    // Domain warp pair
    FastNoiseLite wX, wY;
    wX.SetNoiseType(FastNoiseLite::NoiseType_OpenSimplex2); wX.SetSeed(rand());
    wY.SetNoiseType(FastNoiseLite::NoiseType_OpenSimplex2); wY.SetSeed(rand());

    // Continent mask (very low frequency)
    FastNoiseLite contNoise;
    contNoise.SetNoiseType(FastNoiseLite::NoiseType_OpenSimplex2);
    contNoise.SetSeed(rand());

    // fBm for rolling hills / detail
    FastNoiseLite fbm;
    fbm.SetNoiseType(FastNoiseLite::NoiseType_OpenSimplex2);
    fbm.SetFractalType(FastNoiseLite::FractalType_FBm);
    fbm.SetFractalOctaves(6);
    fbm.SetFractalLacunarity(2.0f);
    fbm.SetFractalGain(0.5f);
    fbm.SetSeed(rand());

    // Ridged noise for mountain ridgelines
    FastNoiseLite ridge;
    ridge.SetNoiseType(FastNoiseLite::NoiseType_OpenSimplex2);
    ridge.SetFractalType(FastNoiseLite::FractalType_Ridged);
    ridge.SetFractalOctaves(5);
    ridge.SetFractalLacunarity(2.1f);
    ridge.SetFractalGain(0.45f);
    ridge.SetSeed(rand());

    // Cellular noise for plateau / mesa detail inside mountains
    FastNoiseLite cell;
    cell.SetNoiseType(FastNoiseLite::NoiseType_Cellular);
    cell.SetCellularReturnType(FastNoiseLite::CellularReturnType_Distance2Div);
    cell.SetSeed(rand());

    // Moisture (independent fBm)
    FastNoiseLite moist;
    moist.SetNoiseType(FastNoiseLite::NoiseType_OpenSimplex2);
    moist.SetFractalType(FastNoiseLite::FractalType_FBm);
    moist.SetFractalOctaves(4);
    moist.SetSeed(rand());

    // Temperature perturbation (adds local anomalies to lat gradient)
    FastNoiseLite tempNoise;
    tempNoise.SetNoiseType(FastNoiseLite::NoiseType_OpenSimplex2);
    tempNoise.SetSeed(rand());

    /* ── Raw height generation ──────────────────────────── */

    vector<vector<float>> hm(MAP_W, vector<float>(MAP_H, 0.f));
    vector<vector<float>> mm(MAP_W, vector<float>(MAP_H, 0.f));
    vector<vector<float>> tm(MAP_W, vector<float>(MAP_H, 0.f));

    float hMin =  1e9f, hMax = -1e9f;

    for(int x = 0; x < MAP_W; ++x)
    for(int y = 0; y < MAP_H; ++y)
    {
        float xf = (float)x, yf = (float)y;

        /* 1. Domain warp */
        const float WS = 14.f;
        float wx = xf + wX.GetNoise(xf*0.012f, yf*0.012f) * WS;
        float wy = yf + wY.GetNoise(xf*0.012f, yf*0.012f) * WS;

        /* 2. Continent mask */
        float cont = n01(contNoise.GetNoise(wx*0.018f, wy*0.018f));

        // Radial falloff so ocean tends to appear at edges
        float cx = (x / (float)MAP_W) - 0.5f;
        float cy = (y / (float)MAP_H) - 0.5f;
        float radial = 1.f - clamp01(sqrtf(cx*cx + cy*cy) * 2.0f);
        cont = clamp01(cont * 0.7f + radial * 0.3f);

        /* 3. Hill detail (fBm) */
        float hills = n01(fbm.GetNoise(wx*0.07f, wy*0.07f));

        /* 4. Mountain ridges */
        float ridgeV = n01(ridge.GetNoise(wx*0.032f, wy*0.032f));

        /* 5. Cellular mesa detail blended into high areas */
        float cellV  = n01(cell.GetNoise(wx*0.05f, wy*0.05f));

        /* 6. Mountain mask – only where continent is high */
        float mtnMask = smoothstep(0.42f, 0.72f, cont);

        /* 7. Combine layers */
        float h =
              5.0f                           // flat baseline
            + hills  * 8.0f                 // rolling hills
            + ridgeV * 70.0f * mtnMask      // sharp ridgelines
            + cellV  * 6.0f  * mtnMask      // mesa / plateau texture
            + cont   * 10.0f;               // broad continent lift

        hm[x][y] = h;
        hMin = min(hMin, h);
        hMax = max(hMax, h);

        /* 8. Moisture – slightly wetter near low-lying terrain */
        mm[x][y] = n01(moist.GetNoise(xf*0.06f, yf*0.06f));

        /* 9. Temperature – latitude gradient (warm equator) + noise */
        float latGrad = 1.f - fabsf(cy) * 2.f;   // 1 at mid, 0 at top/bot
        float tNoise  = n01(tempNoise.GetNoise(xf*0.04f, yf*0.04f)) * 0.35f;
        tm[x][y] = clamp01(latGrad * 0.65f + tNoise);
    }

    /* ── Normalise height to [0,1] then scale to [0,80] ── */
    float hRange = hMax - hMin + 1e-6f;
    for(int x=0;x<MAP_W;++x)
    for(int y=0;y<MAP_H;++y)
        hm[x][y] = ((hm[x][y] - hMin) / hRange) * 80.f;

    /* ── Thermal erosion ──────────────────────────────────── */
    thermalErode(hm, MAP_W, MAP_H, 25, 3.5f, 0.12f);

    /* ── Moisture coastline bias ──────────────────────────── */
    // Tiles near water get moisture boost (they're wetter in reality)
    for(int x=0;x<MAP_W;++x)
    for(int y=0;y<MAP_H;++y)
    {
        if(hm[x][y] < H_SHORE)
            for(int dx=-3;dx<=3;++dx)
            for(int dy=-3;dy<=3;++dy)
            {
                int nx=x+dx, ny=y+dy;
                if(nx<0||nx>=MAP_W||ny<0||ny>=MAP_H) continue;
                float dist = sqrtf((float)(dx*dx+dy*dy));
                mm[nx][ny] = clamp01(mm[nx][ny] + 0.12f / (dist+1.f));
            }
    }

    /* ── Temperature cold penalty at altitude ─────────────── */
    for(int x=0;x<MAP_W;++x)
    for(int y=0;y<MAP_H;++y)
    {
        float h = hm[x][y] / 80.f;
        tm[x][y] = clamp01(tm[x][y] - h * 0.6f);
    }

    /* ── River simulation ─────────────────────────────────── */
    // Collect high-altitude starting points
    vector<pair<int,int>> peaks;
    for(int x=0;x<MAP_W;++x)
    for(int y=0;y<MAP_H;++y)
        if(hm[x][y] >= 50.f)
            peaks.push_back({x,y});

    // Spawn several rivers from random peaks
    int numRivers = 8;
    for(int r = 0; r < numRivers && !peaks.empty(); ++r)
    {
        int idx   = rand() % (int)peaks.size();
        int cx    = peaks[idx].first;
        int cy    = peaks[idx].second;
        peaks.erase(peaks.begin() + idx);

        vector<vector<bool>> visited(MAP_W, vector<bool>(MAP_H, false));
        float lowestReached = hm[cx][cy];
        const float maxClimb = 3.5f;
        const int   maxSteps = MAP_W * MAP_H;

        for(int step = 0; step < maxSteps; ++step)
        {
            grid[cx][cy].isRiver = true;
            visited[cx][cy] = true;
            lowestReached = min(lowestReached, hm[cx][cy]);

            // Stop when we reach ocean
            if(hm[cx][cy] < H_OCEAN) break;

            // Find lowest unvisited neighbour
            int bx = -1, by = -1;
            float bh = 1e9f;
            for(int dx=-1;dx<=1;++dx)
            for(int dy=-1;dy<=1;++dy)
            {
                if(!dx&&!dy) continue;
                int nx=cx+dx, ny=cy+dy;
                if(nx<0||nx>=MAP_W||ny<0||ny>=MAP_H) continue;
                if(visited[nx][ny]) continue;
                if(hm[nx][ny] < bh) { bh=hm[nx][ny]; bx=nx; by=ny; }
            }

            if(bx < 0) break;
            if(bh > lowestReached + maxClimb) break;

            cx = bx; cy = by;
        }
    }

    /* ── Compute normals for shading ──────────────────────── */
    // Simple central-difference surface normal, collapse to shade value
    for(int x=0;x<MAP_W;++x)
    for(int y=0;y<MAP_H;++y)
    {
        int xl = max(x-1, 0),   xr = min(x+1, MAP_W-1);
        int yd = max(y-1, 0),   yu = min(y+1, MAP_H-1);
        float dhdx = (hm[xr][y]  - hm[xl][y])  / 2.f;
        float dhdy = (hm[x][yu]  - hm[x][yd])  / 2.f;

        // Normalised slope -> shade (0.4 ambient min)
        float slope = sqrtf(dhdx*dhdx + dhdy*dhdy);
        // Light direction: roughly from upper-left
        float lightDot = 1.f - clamp01((-dhdx - dhdy * 0.5f) * 0.08f);
        grid[x][y].shade = clamp01(lerp(0.42f, 1.0f, lightDot));
    }

    /* ── Bake everything into Tile grid ───────────────────── */
    for(int x=0;x<MAP_W;++x)
    for(int y=0;y<MAP_H;++y)
    {
        Tile& t  = grid[x][y];
        t.gx     = x;
        t.gy     = y;
        t.height      = hm[x][y] / 80.f;   // normalised
        t.moisture    = mm[x][y];
        t.temperature = tm[x][y];

        Biome b = classifyBiome(t.height, t.moisture, t.temperature);

        // River overrides land biome (but not ocean)
        if(t.isRiver && b != Biome::DeepOcean && b != Biome::Ocean)
            b = Biome::River;

        t.biome   = b;
        t.isWater = (b == Biome::DeepOcean || b == Biome::Ocean ||
                     b == Biome::River);

        // Screen-space height (exaggerated for 2.5D feel)
        t.screenH = t.height * ISO_DEPTH * 8.f;

        // Isometric projection
        // World tile (x,y) → screen (sx,sy)  (centre of map at screen origin)
        float ox = ((float)MAP_W * ISO_TW * 0.5f);
        float oy = ((float)MAP_H * ISO_TH * 0.5f);
        t.screenX =  (x - y) * ISO_TW - ox;
        t.screenY =  (x + y) * ISO_TH - oy - t.screenH;

        // Colours
        RGB c = biomeColor(b, t.height, t.shade);
        t.r = c.r; t.g = c.g; t.b = c.b; t.a = 1.f;
        darken(c.r, c.g, c.b, 0.55f, t.sr, t.sg, t.sb);
    }

    return grid;
}


/* ════════════════════════════════════════════════════════════
   OPENGL RENDERING
   ════════════════════════════════════════════════════════════ */

static void drawTile(const Tile& t, float camX, float camY, float zoom)
{
    // Skip fully off-screen tiles (rough cull – good enough for 150²)
    float sx = (t.screenX - camX) * zoom;
    float sy = (t.screenY - camY) * zoom;
    float tw = ISO_TW * zoom;
    float th = ISO_TH * zoom;
    float sh = t.screenH * zoom + 2.f;  // +2 so thin tiles still appear

    // --- Draw left side face (south-west) ---
    if(t.screenH > 0.5f && !t.isWater)
    {
        glColor3f(t.sr, t.sg, t.sb);
        glBegin(GL_QUADS);
            // top-left of face = bottom-left of top diamond
            glVertex2f(sx,        sy + th);
            glVertex2f(sx - tw,   sy);
            glVertex2f(sx - tw,   sy + sh);
            glVertex2f(sx,        sy + th + sh);
        glEnd();
    }

    // --- Draw right side face (south-east) ---
    if(t.screenH > 0.5f && !t.isWater)
    {
        float dr = clamp01(t.sr * 0.75f);
        float dg = clamp01(t.sg * 0.75f);
        float db = clamp01(t.sb * 0.75f);
        glColor3f(dr, dg, db);
        glBegin(GL_QUADS);
            glVertex2f(sx,        sy + th);
            glVertex2f(sx + tw,   sy);
            glVertex2f(sx + tw,   sy + sh);
            glVertex2f(sx,        sy + th + sh);
        glEnd();
    }

    // --- Draw top diamond face ---
    // Use vertex colours for a smooth gradient effect
    float r0 = clamp01(t.r * 1.12f);
    float g0 = clamp01(t.g * 1.12f);
    float b0 = clamp01(t.b * 1.12f);

    // Water gets a subtle animated shimmer tint (static here; could be time-based)
    if(t.isWater)
    {
        r0 *= 0.9f; g0 *= 0.9f; b0 = clamp01(b0 * 1.05f);
    }

    glBegin(GL_QUADS);
        glColor3f(r0, g0, b0);
        glVertex2f(sx,      sy);           // top    (north)
        glColor3f(t.r * 0.95f, t.g * 0.95f, t.b * 0.95f);
        glVertex2f(sx - tw, sy + th);      // left   (west)
        glColor3f(t.r * 0.88f, t.g * 0.88f, t.b * 0.88f);
        glVertex2f(sx,      sy + th*2);    // bottom (south)
        glColor3f(t.r * 0.93f, t.g * 0.93f, t.b * 0.93f);
        glVertex2f(sx + tw, sy + th);      // right  (east)
    glEnd();

    // Thin outline so tiles don't blur into each other at low zoom
    if(zoom > 1.4f)
    {
        glColor4f(0.f, 0.f, 0.f, 0.18f);
        glBegin(GL_LINE_LOOP);
            glVertex2f(sx,      sy);
            glVertex2f(sx - tw, sy + th);
            glVertex2f(sx,      sy + th*2);
            glVertex2f(sx + tw, sy + th);
        glEnd();
    }
}


/* ════════════════════════════════════════════════════════════
   MAIN
   ════════════════════════════════════════════════════════════ */

int main()
{
    srand((unsigned int)time(nullptr));

    if(!glfwInit())
    {
        cerr << "GLFW init failed\n";
        return 1;
    }

    int winW = 1280, winH = 720;
    GLFWwindow* window = glfwCreateWindow(winW, winH,
        "Procedural World Generator – 2.5D", nullptr, nullptr);
    if(!window)
    {
        cerr << "Window creation failed\n";
        glfwTerminate();
        return 1;
    }

    glfwMakeContextCurrent(window);
    glfwSwapInterval(1); // vsync

    // OpenGL 2D setup
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glOrtho(-winW/2, winW/2, winH/2, -winH/2, -1, 1);
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();

    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    glClearColor(0.05f, 0.05f, 0.08f, 1.f);

    // Generate initial world
    cout << "Generating world...\n";
    auto grid = generateWorld();
    cout << "Done. Controls: WASD/arrows=pan, Q/E=zoom, R=regenerate, ESC=quit\n";

    // Camera state
    float camX = 0.f, camY = 0.f;
    float zoom = 1.8f;

    while(!glfwWindowShouldClose(window))
    {
        glfwPollEvents();

        // ── Input ──────────────────────────────────────────
        float speed = 12.f / zoom;

        if(glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS)
            glfwSetWindowShouldClose(window, true);

        if(glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS ||
           glfwGetKey(window, GLFW_KEY_UP) == GLFW_PRESS)    camY -= speed;
        if(glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS ||
           glfwGetKey(window, GLFW_KEY_DOWN) == GLFW_PRESS)  camY += speed;
        if(glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS ||
           glfwGetKey(window, GLFW_KEY_LEFT) == GLFW_PRESS)  camX -= speed;
        if(glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS ||
           glfwGetKey(window, GLFW_KEY_RIGHT) == GLFW_PRESS) camX += speed;
        if(glfwGetKey(window, GLFW_KEY_E) == GLFW_PRESS)     zoom = min(zoom * 1.02f, 8.f);
        if(glfwGetKey(window, GLFW_KEY_Q) == GLFW_PRESS)     zoom = max(zoom * 0.98f, 0.3f);

        static bool rPrev = false;
        bool rNow = glfwGetKey(window, GLFW_KEY_R) == GLFW_PRESS;
        if(rNow && !rPrev)
        {
            cout << "Regenerating...\n";
            grid = generateWorld();
            cout << "Done.\n";
        }
        rPrev = rNow;

        // ── Render ─────────────────────────────────────────
        glClear(GL_COLOR_BUFFER_BIT);
        glLoadIdentity();

        /*
            Isometric painter's order: draw tiles from back (top-left)
            to front (bottom-right) so taller tiles correctly occlude
            tiles behind them.  In isometric space that means iterating
            y first (rows) and x second (columns).
        */
        for(int y = 0; y < MAP_H; ++y)
        for(int x = 0; x < MAP_W; ++x)
            drawTile(grid[x][y], camX, camY, zoom);

        // ── HUD (minimal legend) ───────────────────────────
        // (OpenGL 1.x has no built-in text; skip or use a bitmap font lib)

        glfwSwapBuffers(window);
    }

    glfwDestroyWindow(window);
    glfwTerminate();
    return 0;
}