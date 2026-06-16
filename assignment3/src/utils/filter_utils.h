#pragma once

#include <string>
#include <vector>

// ============================================================
// FilterType enum — extend this to add new filter modes.
// ============================================================
enum class FilterType {
    BLUR,
    SHARPEN,
    EDGE,
    EMBOSS,
    HDR_TONEMAPPING   // <-- NEW: HDR tone-mapping filter
};

// ============================================================
// Tone-mapping algorithm selector
// ============================================================
enum class ToneMappingAlgorithm {
    REINHARD,   // Global Reinhard (simple, fast)
    DRAGO,      // Drago logarithmic (better detail in darks)
    LOCAL       // Local / bilateral tone mapping (advanced)
};

// ============================================================
// HDR parameters — passed to the CUDA kernels
// ============================================================
struct HdrParams {
    float exposure    = 1.0f;   // Linear EV multiplier before tone-mapping
    float gamma       = 2.2f;   // Display gamma (applied after tone-map)
    float saturation  = 1.2f;   // Chroma boost factor
    ToneMappingAlgorithm algorithm = ToneMappingAlgorithm::REINHARD;

    // Reinhard / Drago tuning
    float whitePoint  = 4.0f;   // Reinhard extended white point
    float bias        = 0.85f;  // Drago bias (0-1)

    // Local tone-mapping tuning
    int   localRadius = 15;     // Bilateral filter spatial radius
    float localSigmaS = 10.0f;  // Bilateral spatial sigma
    float localSigmaR = 0.15f;  // Bilateral range sigma (luminance)
};

// ============================================================
// Convolution filter kernel (unchanged from template)
// ============================================================
struct FilterKernel {
    std::vector<float> data;
    int size = 0;
};

// ============================================================
// String helpers
// ============================================================
FilterType          stringToFilterType(const std::string& s);
ToneMappingAlgorithm stringToToneMappingAlgorithm(const std::string& s);

// Creates a convolution kernel matrix for non-HDR filters
FilterKernel createFilterKernel(FilterType type, int kernelSize, float sigma = 1.0f, float intensity = 1.0f);
