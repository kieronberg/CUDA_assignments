#pragma once

#include <opencv2/opencv.hpp>
#include <string>
#include "../utils/filter_utils.h"

// ============================================================
// Source type (unchanged from template)
// ============================================================
enum class InputSource { WEBCAM, IMAGE, VIDEO, SYNTHETIC };
enum class SyntheticPattern { CHECKERBOARD, GRADIENT, NOISE };

InputSource     stringToInputSource(const std::string& s);
SyntheticPattern stringToSyntheticPattern(const std::string& s);

// ============================================================
// GPU buffer bundle for the HDR pipeline.
// Allocated once (resize on resolution change), reused per frame.
// ============================================================
struct HdrDeviceBuffers {
    uint8_t* d_input    = nullptr;  // uint8 BGR in  (H×W×3)
    uint8_t* d_output   = nullptr;  // uint8 BGR out (H×W×3)
    float*   d_linearF  = nullptr;  // float BGR     (H×W×3)
    float*   d_lum      = nullptr;  // float lum     (H×W)
    float*   d_scratch  = nullptr;  // float scratch (H×W) for local TM

    int width = 0, height = 0;

    void allocate(int w, int h);
    void free();
    ~HdrDeviceBuffers() { free(); }
};

// ============================================================
// Process one frame through the full HDR pipeline.
// Returns the tone-mapped BGR frame.
// ============================================================
cv::Mat applyHdrToneMapping(
    const cv::Mat&     frame,
    HdrDeviceBuffers&  bufs,
    const HdrParams&   params);

// ============================================================
// CPU reference implementation (for performance comparison)
// ============================================================
cv::Mat applyHdrToneMappingCPU(
    const cv::Mat&   frame,
    const HdrParams& params);

// ============================================================
// Existing helpers (unchanged from template)
// ============================================================
cv::Mat generateSyntheticFrame(SyntheticPattern pattern, int width, int height);
