#include "filter_utils.h"
#include <stdexcept>
#include <cmath>

// ============================================================
// String → FilterType
// ============================================================
FilterType stringToFilterType(const std::string& s) {
    if (s == "blur")            return FilterType::BLUR;
    if (s == "sharpen")         return FilterType::SHARPEN;
    if (s == "edge")            return FilterType::EDGE;
    if (s == "emboss")          return FilterType::EMBOSS;
    if (s == "hdr")             return FilterType::HDR_TONEMAPPING;  // NEW
    throw std::invalid_argument("Unknown filter type: " + s +
        ". Valid options: blur, sharpen, edge, emboss, hdr");
}

// ============================================================
// String → ToneMappingAlgorithm
// ============================================================
ToneMappingAlgorithm stringToToneMappingAlgorithm(const std::string& s) {
    if (s == "reinhard")    return ToneMappingAlgorithm::REINHARD;
    if (s == "drago")       return ToneMappingAlgorithm::DRAGO;
    if (s == "local")       return ToneMappingAlgorithm::LOCAL;
    throw std::invalid_argument("Unknown tone-mapping algorithm: " + s +
        ". Valid options: reinhard, drago, local");
}

// ============================================================
// Convolution kernel factory (unchanged filters + guard for HDR)
// ============================================================
FilterKernel createFilterKernel(FilterType type, int kernelSize,
                                float sigma, float intensity) {
    // HDR does not use a convolution matrix — guard here so callers
    // can check before allocating constant memory on the GPU.
    if (type == FilterType::HDR_TONEMAPPING) {
        return FilterKernel{};  // empty — caller must use HDR path
    }

    // Ensure odd kernel size
    if (kernelSize % 2 == 0) kernelSize++;
    const int half = kernelSize / 2;
    FilterKernel kernel;
    kernel.size = kernelSize;
    kernel.data.resize(kernelSize * kernelSize, 0.0f);

    auto idx = [&](int r, int c) { return r * kernelSize + c; };

    switch (type) {
    // ---- Gaussian blur -------------------------------------------
    case FilterType::BLUR: {
        float sum = 0.0f;
        for (int r = -half; r <= half; ++r)
            for (int c = -half; c <= half; ++c) {
                float v = std::exp(-(r*r + c*c) / (2.0f * sigma * sigma));
                kernel.data[idx(r+half, c+half)] = v;
                sum += v;
            }
        for (auto& v : kernel.data) v /= sum;
        break;
    }
    // ---- Sharpen -------------------------------------------------
    case FilterType::SHARPEN: {
        // Identity + Laplacian scaled by intensity
        for (auto& v : kernel.data) v = 0.0f;
        kernel.data[idx(half, half)] = 1.0f + 4.0f * intensity;
        if (half > 0) {
            kernel.data[idx(half-1, half)] = -intensity;
            kernel.data[idx(half+1, half)] = -intensity;
            kernel.data[idx(half, half-1)] = -intensity;
            kernel.data[idx(half, half+1)] = -intensity;
        }
        break;
    }
    // ---- Edge detection (Laplacian) ------------------------------
    case FilterType::EDGE: {
        for (auto& v : kernel.data) v = -1.0f * intensity;
        kernel.data[idx(half, half)] = (kernelSize * kernelSize - 1) * intensity;
        break;
    }
    // ---- Emboss --------------------------------------------------
    case FilterType::EMBOSS: {
        for (int r = 0; r < kernelSize; ++r)
            for (int c = 0; c < kernelSize; ++c) {
                int dr = r - half, dc = c - half;
                if (dr < 0 && dc < 0)      kernel.data[idx(r,c)] = -intensity;
                else if (dr > 0 && dc > 0) kernel.data[idx(r,c)] =  intensity;
                // diagonal: leave 0 for centre
            }
        kernel.data[idx(half,half)] = 1.0f;
        break;
    }
    default:
        break;
    }
    return kernel;
}
