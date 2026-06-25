// =============================================================================
// MapGrid.h   (ADDED FOR MAP LOADER)
//
//   Simple 2D grid of model-name strings, plus the binary map-file loader
//   that populates it from the companion ChunkTerrainSystem generator's
//   Map.bin output.
//
//   Empty string in a cell == "no model placed yet" (the initial state of
//   every cell). Indexed [row][col], row == Z axis, col == X axis, both
//   offset by (originX, originZ) so negative chunk coordinates from the
//   generator map into valid indices.
//
//   File format written by the generator's WriteMapBin() (mirrored exactly
//   by ReadMapBinStreaming() in MapGrid.cpp):
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
// =============================================================================
#pragma once

#include <string>
#include <vector>

// =============================================================================
// MapGrid
// =============================================================================
struct MapGrid
{
    int width = 0;   // number of columns (X)
    int height = 0;  // number of rows    (Z)
    int originX = 0; // chunk X that maps to column 0
    int originZ = 0; // chunk Z that maps to row 0

    std::vector<std::vector<std::string>> cells; // cells[row][col]

    MapGrid() = default;
    MapGrid(int w, int h, int originXIn, int originZIn);

    bool InBounds(int chunkX, int chunkZ, int &outCol, int &outRow) const;

    // Fills a single cell in-place. Returns false (and leaves the map
    // untouched) if the coordinate is outside the configured bounds.
    bool Set(int chunkX, int chunkZ, const std::string &modelName);

    void Print() const;
};

// =============================================================================
// LoadMapFromBinary
//
//   1. Builds an empty MapGrid of (width x height), centred so that chunk
//      (centerX, centerZ) lands in the middle of the grid.
//   2. Streams 3DModels/Map.bin one record at a time, filling matching
//      cells in place via grid.Set().
// =============================================================================
MapGrid LoadMapFromBinary(
    const std::string &filename,
    int width, int height,
    int centerX, int centerZ);