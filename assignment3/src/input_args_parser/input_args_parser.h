#pragma once

#include <string>
#include "../utils/filter_utils.h"

// ============================================================
// Parsed command-line arguments
// ============================================================
struct InputArgs {
    // ---- Input source (unchanged) ----
    std::string inputSource  = "webcam";
    std::string inputPath    = "test_image.jpg";
    std::string syntheticPat = "checkerboard";
    int         deviceId     = 0;

    // ---- Filter (extended) ----
    std::string filterType   = "blur";
    int         kernelSize   = 3;
    float       sigma        = 1.0f;
    float       intensity    = 1.0f;

    // ---- Display ----
    bool        preview      = false;

    // ---- HDR-specific (new) ----
    std::string toneMappingAlg = "reinhard";   // reinhard | drago | local
    float       exposure       = 1.0f;
    float       gamma          = 2.2f;
    float       saturation     = 1.2f;
    float       whitePoint     = 4.0f;
    float       dragoBias      = 0.85f;
    int         localRadius    = 15;
    float       localSigmaS    = 10.0f;
    float       localSigmaR    = 0.15f;

    // ---- Benchmarking (new) ----
    bool        benchmark      = false;   // print GPU vs CPU timing

    // Convert HDR string params to typed structs
    HdrParams toHdrParams() const;
};

// Returns true on success; on --help prints usage and returns false.
bool parseArgs(int argc, char** argv, InputArgs& out);
