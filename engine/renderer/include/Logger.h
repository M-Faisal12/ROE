// =============================================================================
// Logger.h
//
//   Tiny shared logging helper. Every other file includes this instead of
//   redefining its own Log() function.
// =============================================================================
#pragma once

#include <iostream>
#include <string>

inline void Log(const std::string &tag, const std::string &message)
{
    std::cout << "[" << tag << "] " << message << "\n";
}