// =============================================================================
// MapGrid.cpp   (ADDED FOR MAP LOADER)
//
//   Implementation of MapGrid + the streaming binary-file reader/loader.
//   See MapGrid.h for the file-format contract this mirrors.
// =============================================================================
#include "MapGrid.h"
#include "Logger.h"

#include <fstream>

// =============================================================================
// MapGrid:: implementation
// =============================================================================
MapGrid::MapGrid(int w, int h, int originXIn, int originZIn)
    : width(w), height(h), originX(originXIn), originZ(originZIn)
{
    // Every cell starts empty -- "initially the map is empty" per spec.
    cells.assign(
        static_cast<size_t>(height),
        std::vector<std::string>(static_cast<size_t>(width), std::string()));
}

bool MapGrid::InBounds(int chunkX, int chunkZ, int &outCol, int &outRow) const
{
    outCol = chunkX - originX;
    outRow = chunkZ - originZ;
    return outCol >= 0 && outCol < width &&
           outRow >= 0 && outRow < height;
}

bool MapGrid::Set(int chunkX, int chunkZ, const std::string &modelName)
{
    int col = 0, row = 0;
    if (!InBounds(chunkX, chunkZ, col, row))
        return false;

    cells[static_cast<size_t>(row)][static_cast<size_t>(col)] = modelName;
    return true;
}

void MapGrid::Print() const
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
// =============================================================================
MapGrid LoadMapFromBinary(
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