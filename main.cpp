// ─────────────────────────────────────────────────────────────────────────────
//  main.cpp  –  ChunkTerrainSystem demo
//
//  Usage:  ./terrain_demo [chunkX] [chunkZ] [rangeW] [rangeH] [seed]
//
//  Examples:
//    ./terrain_demo                    → 20×20 map around origin, seed 1337
//    ./terrain_demo 5 -3               → single chunk (5,-3) + 20×20 map
//    ./terrain_demo 0 0 32 32 9999     → 32×32 map with a different seed
// ─────────────────────────────────────────────────────────────────────────────

#include "Mapgenerator.h"
#include <iostream>
#include <iomanip>
#include <cstdlib>
#include <string.h>
#include <fstream>
#include <stdexcept>
static bool WriteMapBin(
    const std::string &filename,
    const std::vector<std::pair<int, int>> &coords,
    const std::vector<char *> &names)
{
    if (coords.size() != names.size())
    {
        return false;
    }

    std::ofstream outFile(filename, std::ios::out | std::ios::binary);

    if (!outFile)
    {
        return false;
    }

    // ---- Section 1: vector<pair<int,int>> ----
    size_t coordCount = coords.size();
    outFile.write(reinterpret_cast<const char *>(&coordCount), sizeof(coordCount));

    for (const auto &c : coords)
    {
        int x = c.first;
        int y = c.second;
        outFile.write(reinterpret_cast<const char *>(&x), sizeof(x));
        outFile.write(reinterpret_cast<const char *>(&y), sizeof(y));
    }

    // ---- Section 2: vector<char*> model file names ----
    size_t nameCount = names.size();
    outFile.write(reinterpret_cast<const char *>(&nameCount), sizeof(nameCount));

    for (char *name : names)
    {
        if (name == nullptr)
        {
            size_t len = 0;
            outFile.write(reinterpret_cast<const char *>(&len), sizeof(len));
            continue;
        }

        size_t len = strlen(name) + 1; // +1 for null terminator, matches reader
        outFile.write(reinterpret_cast<const char *>(&len), sizeof(len));
        outFile.write(name, static_cast<std::streamsize>(len));
    }

    if (!outFile)
    {
        return false;
    }

    return true;
}
std::vector<char *> *readVector(const char *filename)
{
    std::ifstream file(filename, std::ios::binary);

    if (!file)
        return nullptr;

    size_t count;

    file.read(reinterpret_cast<char *>(&count),
              sizeof(count));

    auto *vec = new std::vector<char *>;

    for (size_t i = 0; i < count; i++)
    {
        size_t len;

        file.read(reinterpret_cast<char *>(&len),
                  sizeof(len));

        if (len == 0)
        {
            vec->push_back(nullptr);
            continue;
        }

        char *str = new char[len];

        file.read(str, len);

        vec->push_back(str);
    }

    return vec;
}
void freeVector(std::vector<char *> *vec)
{
    for (char *str : *vec)
    {
        delete[] str;
    }

    delete vec;
}
void writeVector(const char *filename,
                 const std::vector<char *> *vec)
{
    std::ofstream file(filename, std::ios::binary);

    if (!file)
        return;

    size_t count = vec->size();

    file.write(reinterpret_cast<char *>(&count),
               sizeof(count));

    for (size_t i = 0; i < count; i++)
    {
        char *str = (*vec)[i];

        size_t len = (str == nullptr)
                         ? 0
                         : strlen(str) + 1;

        file.write(reinterpret_cast<char *>(&len),
                   sizeof(len));

        if (len > 0)
        {
            file.write(str, len);
        }
    }
}
static void printUsage(const char *prog)
{
    std::cerr
        << "\nUsage: " << prog
        << " [chunkX] [chunkZ] [rangeW] [rangeH] [seed]\n"
        << "  chunkX/Z – centre chunk for single-chunk query  (default 0,0)\n"
        << "  rangeW/H – map display size in chunks           (default 20,20)\n"
        << "  seed     – noise seed; same seed = same map     (default 1337)\n\n";
}

int main(int argc, char *argv[])
{
    int chunkX = 0, chunkZ = 0;
    int rangeW = 5, rangeH = 1;
    int seed = 1337;

    if (argc > 1)
        chunkX = std::atoi(argv[1]);
    if (argc > 2)
        chunkZ = std::atoi(argv[2]);
    if (argc > 3)
        rangeW = std::atoi(argv[3]);
    if (argc > 4)
        rangeH = std::atoi(argv[4]);
    if (argc > 5)
        seed = std::atoi(argv[5]);

    if (rangeW <= 0 || rangeH <= 0)
    {
        std::cerr << "Error: rangeW and rangeH must be positive.\n";
        printUsage(argv[0]);
        return EXIT_FAILURE;
    }

    // ── Configure ────────────────────────────────────────────────────────────
    TerrainSystemConfig cfg;
    cfg.seed = seed;
    cfg.chunkSize = 64.f;
    cfg.sampleGrid = 5;

    cfg.heightFreq = 0.018f;
    cfg.mountainFreq = 0.038f;
    cfg.moistureFreq = 0.024f;
    cfg.variationFreq = 0.060f;
    cfg.tempFreq = 0.011f;
    cfg.continentFreq = 0.0045f;

    // Target distribution — applied exactly over batch queries.
    cfg.distribution.plains = 0.15f;
    cfg.distribution.sand = 0.15f;
    cfg.distribution.sandDunes = 0.10f;
    cfg.distribution.water = 0.10f;
    cfg.distribution.mountains = 0.10f;
    cfg.distribution.snow = 0.15f;
    cfg.distribution.snowMountains = 0.10f;
    cfg.distribution.frozenWater = 0.10f;
    // sum = 0.95 → 5% fallback to continent default

    try
    {
        ChunkTerrainSystem terrain(cfg);

        // ── 1. Single-chunk query ─────────────────────────────────────────────
        std::cout << "\n── Single chunk query ─────────────────────────────────\n";
        int minX = chunkX - rangeW / 2;
        int maxX = minX + rangeW - 1;
        int minZ = chunkZ - rangeH / 2;
        int maxZ = minZ + rangeH - 1;
        const int cols = maxX - minX + 1;
        auto map = terrain.getChunksInRange(minX, maxX, minZ, maxZ);
        for (int z = minZ; z <= maxZ; ++z)
        {
            for (int x = minX; x <= maxX; ++x)
            {
                const auto &d = map[static_cast<size_t>((z - minZ) * cols + (x - minX))];
                std::cout << d.selectedModel << std::endl;
            }
            std::cout << '\n';
        }
        std::vector<char *> *viz = terrain.printRange(minX, maxX, minZ, maxZ);
        const char *filename = "3DModels/models.bin";
        const char *mySentence = "3DModels/snow_mtn_01.glb";
        writeVector("3DModels/vector.bin", viz);
        delete viz;
        auto *data = readVector("3DModels/vector.bin");
        for (char *s : *data)
        {
            std::cout << s << '\n';
        }
        std::vector<std::pair<int, int>> coords = {{0, 0}, {1, 0}, {0, 1},{0, -1}, {-1, 0}};
        std::vector<char *> names = {
            const_cast<char *>("ice_01.glb"),
            const_cast<char *>("dunes_01.glb"),
            const_cast<char *>("mountain_01.glb"),
            const_cast<char *>("ice_01.glb"),
            const_cast<char *>("dunes_01.glb"),};

        if(!WriteMapBin("3DModels/Map.bin", coords, names))std::cout<<"Failed";
    }
    catch (const std::exception &e)
    {
        std::cerr << "Fatal: " << e.what() << "\n";
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}