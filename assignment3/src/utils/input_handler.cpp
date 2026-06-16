#include "input_handler.h"
#include "../kernels/kernels.h"
#include <cuda_runtime.h>
#include <stdexcept>
#include <cmath>
#include <algorithm>

// ------------------------------------------------------------------
// String helpers (unchanged from template)
// ------------------------------------------------------------------
InputSource stringToInputSource(const std::string& s) {
    if (s == "webcam")    return InputSource::WEBCAM;
    if (s == "image")     return InputSource::IMAGE;
    if (s == "video")     return InputSource::VIDEO;
    if (s == "synthetic") return InputSource::SYNTHETIC;
    throw std::invalid_argument("Unknown input source: " + s);
}

SyntheticPattern stringToSyntheticPattern(const std::string& s) {
    if (s == "checkerboard") return SyntheticPattern::CHECKERBOARD;
    if (s == "gradient")     return SyntheticPattern::GRADIENT;
    if (s == "noise")        return SyntheticPattern::NOISE;
    throw std::invalid_argument("Unknown synthetic pattern: " + s);
}

// ------------------------------------------------------------------
// Synthetic frame generator (unchanged from template)
// ------------------------------------------------------------------
cv::Mat generateSyntheticFrame(SyntheticPattern p, int w, int h) {
    cv::Mat frame(h, w, CV_8UC3);
    switch (p) {
    case SyntheticPattern::CHECKERBOARD:
        for (int y = 0; y < h; ++y)
            for (int x = 0; x < w; ++x) {
                uint8_t v = ((x/32 + y/32) % 2) ? 255 : 0;
                frame.at<cv::Vec3b>(y,x) = {v,v,v};
            }
        break;
    case SyntheticPattern::GRADIENT:
        for (int y = 0; y < h; ++y)
            for (int x = 0; x < w; ++x) {
                uint8_t bv = (uint8_t)(255.0f * x / (w-1));
                uint8_t gv = (uint8_t)(255.0f * y / (h-1));
                frame.at<cv::Vec3b>(y,x) = {bv, gv, 128};
            }
        break;
    case SyntheticPattern::NOISE:
        cv::randu(frame, cv::Scalar::all(0), cv::Scalar::all(255));
        break;
    }
    return frame;
}

// ==================================================================
// HdrDeviceBuffers — allocation / free
// ==================================================================
void HdrDeviceBuffers::allocate(int w, int h) {
    if (w == width && h == height) return;   // already allocated
    free();
    width = w; height = h;
    size_t pix = (size_t)w * h;

    auto cudaAlloc = [](void** ptr, size_t bytes) {
        cudaError_t e = cudaMalloc(ptr, bytes);
        if (e != cudaSuccess)
            throw std::runtime_error(std::string("cudaMalloc failed: ") +
                                     cudaGetErrorString(e));
    };
    cudaAlloc((void**)&d_input,   pix * 3 * sizeof(uint8_t));
    cudaAlloc((void**)&d_output,  pix * 3 * sizeof(uint8_t));
    cudaAlloc((void**)&d_linearF, pix * 3 * sizeof(float));
    cudaAlloc((void**)&d_lum,     pix *     sizeof(float));
    cudaAlloc((void**)&d_scratch, pix *     sizeof(float));
}

void HdrDeviceBuffers::free() {
    if (d_input)   { cudaFree(d_input);   d_input   = nullptr; }
    if (d_output)  { cudaFree(d_output);  d_output  = nullptr; }
    if (d_linearF) { cudaFree(d_linearF); d_linearF = nullptr; }
    if (d_lum)     { cudaFree(d_lum);     d_lum     = nullptr; }
    if (d_scratch) { cudaFree(d_scratch); d_scratch = nullptr; }
    width = height = 0;
}

// ==================================================================
// GPU HDR pipeline  (one frame)
// ==================================================================
cv::Mat applyHdrToneMapping(
    const cv::Mat&    frame,
    HdrDeviceBuffers& bufs,
    const HdrParams&  p)
{
    CV_Assert(frame.type() == CV_8UC3);
    const int W = frame.cols, H = frame.rows;

    bufs.allocate(W, H);

    // --- Upload frame to device ---
    cudaMemcpy(bufs.d_input, frame.ptr<uint8_t>(),
               (size_t)W * H * 3, cudaMemcpyHostToDevice);

    // --- Stage 1: BGR uint8 → float + lum ---
    launchRGBtoLinear(bufs.d_input, bufs.d_linearF, bufs.d_lum,
                      W, H, p.exposure);

    // --- Compute luminance stats on CPU ---
    float avgLogLum, maxLum;
    computeLuminanceStats(bufs.d_lum, W, H, avgLogLum, maxLum);

    // --- Stage 2: tone-mapping ---
    switch (p.algorithm) {
    case ToneMappingAlgorithm::REINHARD:
        launchReinhardGlobal(bufs.d_linearF, bufs.d_lum,
                             avgLogLum, p.whitePoint, W, H);
        break;
    case ToneMappingAlgorithm::DRAGO:
        launchDragoToneMap(bufs.d_linearF, bufs.d_lum,
                           maxLum, p.bias, W, H);
        break;
    case ToneMappingAlgorithm::LOCAL:
        launchLocalToneMap(bufs.d_linearF, bufs.d_lum, bufs.d_scratch,
                           p.localRadius, p.localSigmaS, p.localSigmaR,
                           W, H);
        break;
    }

    // --- Stage 3: gamma + saturation + float→uint8 ---
    launchLinearToRGB(bufs.d_linearF, bufs.d_output,
                      p.saturation, p.gamma, W, H);

    // --- Download result ---
    cv::Mat result(H, W, CV_8UC3);
    cudaMemcpy(result.ptr<uint8_t>(), bufs.d_output,
               (size_t)W * H * 3, cudaMemcpyDeviceToHost);
    return result;
}

// ==================================================================
// CPU reference implementation  (Reinhard global for comparison)
// ==================================================================
cv::Mat applyHdrToneMappingCPU(const cv::Mat& frame, const HdrParams& p) {
    CV_Assert(frame.type() == CV_8UC3);
    const int W = frame.cols, H = frame.rows;
    const int N = W * H;

    // Convert to linear float
    std::vector<float> lin(N * 3), lum(N);
    const uint8_t* src = frame.ptr<uint8_t>();
    for (int i = 0; i < N; ++i) {
        float b = std::pow(src[i*3+0] / 255.0f, 2.2f) * p.exposure;
        float g = std::pow(src[i*3+1] / 255.0f, 2.2f) * p.exposure;
        float r = std::pow(src[i*3+2] / 255.0f, 2.2f) * p.exposure;
        lin[i*3+0] = b; lin[i*3+1] = g; lin[i*3+2] = r;
        lum[i] = 0.2126f*r + 0.7152f*g + 0.0722f*b;
    }

    // Log-average luminance
    double logSum = 0.0; float maxL = 0.0f;
    for (int i = 0; i < N; ++i) {
        float L = std::max(lum[i], 1e-6f);
        logSum += std::log(L);
        if (L > maxL) maxL = L;
    }
    float avgLogLum = std::exp((float)(logSum / N));

    // Reinhard global (only algorithm implemented on CPU for benchmark)
    const float key = 0.18f;
    float Lw2 = p.whitePoint * p.whitePoint;
    for (int i = 0; i < N; ++i) {
        float L  = std::max(lum[i], 1e-6f);
        float Ls = (key / (avgLogLum + 1e-6f)) * L;
        float Ld = Ls * (1.0f + Ls / Lw2) / (1.0f + Ls);
        float sc = Ld / L;
        lin[i*3+0] *= sc; lin[i*3+1] *= sc; lin[i*3+2] *= sc;
    }

    // Saturation + gamma + convert back to uint8
    cv::Mat result(H, W, CV_8UC3);
    uint8_t* dst = result.ptr<uint8_t>();
    float invGamma = 1.0f / p.gamma;
    for (int i = 0; i < N; ++i) {
        float b = lin[i*3+0], g = lin[i*3+1], r = lin[i*3+2];
        float Y  =  0.299f*r + 0.587f*g + 0.114f*b;
        float Cb = (-0.169f*r - 0.331f*g + 0.500f*b) * p.saturation;
        float Cr = ( 0.500f*r - 0.419f*g - 0.081f*b) * p.saturation;
        r = Y + 1.402f*Cr;
        g = Y - 0.344f*Cb - 0.714f*Cr;
        b = Y + 1.772f*Cb;

        auto enc = [&](float v) -> uint8_t {
            v = std::min(std::max(v, 0.0f), 1.0f);
            return (uint8_t)(std::pow(v, invGamma) * 255.0f + 0.5f);
        };
        dst[i*3+0] = enc(b); dst[i*3+1] = enc(g); dst[i*3+2] = enc(r);
    }
    return result;
}
