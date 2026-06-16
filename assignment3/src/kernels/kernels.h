#pragma once

#include <cstdint>
#include "../utils/filter_utils.h"

// ============================================================
// Existing convolution kernel launcher (unchanged interface)
// ============================================================
void launchConvolutionKernel(
    const uint8_t* d_input,
    uint8_t*       d_output,
    int            width,
    int            height,
    int            channels,
    const float*   d_kernel,
    int            kernelSize);

// ============================================================
// HDR tone-mapping kernel launchers
// ============================================================

/**
 * Stage 1 — RGB → Linear float + luminance extraction.
 *
 * Converts uint8 BGR (OpenCV default) to linear-light float and
 * writes per-pixel luminance into a separate float buffer.
 * Uses shared memory to amortise global-memory traffic.
 *
 * @param d_bgr      Input  uint8 BGR  (H×W×3)
 * @param d_linearF  Output float BGR  (H×W×3) — linear light [0,∞)
 * @param d_lum      Output float lum  (H×W)   — Y from Rec.709
 * @param width      Frame width  in pixels
 * @param height     Frame height in pixels
 * @param exposure   Linear EV multiplier
 */
void launchRGBtoLinear(
    const uint8_t* d_bgr,
    float*         d_linearF,
    float*         d_lum,
    int            width,
    int            height,
    float          exposure);

/**
 * Stage 2a — Global Reinhard tone-mapping.
 *
 * Maps each luminance value through the Reinhard extended operator:
 *   L_d = L_w * (1 + L_w / Lw²) / (1 + L_w)
 * and scales colour channels proportionally.
 *
 * @param d_linearF  In/out float BGR (H×W×3)
 * @param d_lum      Luminance buffer (H×W)
 * @param avgLogLum  Log-average luminance (computed on CPU from d_lum)
 * @param whitePoint Scene-referent white-point value
 * @param width / height
 */
void launchReinhardGlobal(
    float*       d_linearF,
    const float* d_lum,
    float        avgLogLum,
    float        whitePoint,
    int          width,
    int          height);

/**
 * Stage 2b — Drago logarithmic tone-mapping.
 *
 * L_d = (Ldmax / log10(Lw_max+1)) * log(L+1) / log(2 + 8*(L/Lw_max)^(log(b)/log(0.5)))
 *
 * @param d_linearF  In/out float BGR
 * @param d_lum      Luminance buffer
 * @param maxLum     Maximum luminance in the scene
 * @param bias       Drago bias parameter (0-1, default 0.85)
 * @param width / height
 */
void launchDragoToneMap(
    float*       d_linearF,
    const float* d_lum,
    float        maxLum,
    float        bias,
    int          width,
    int          height);

/**
 * Stage 2c — Local (bilateral) tone-mapping.
 *
 * Uses a joint bilateral filter in the log-luminance domain:
 * 1. Separates log-luminance into base (bilateral-smooth) + detail layers.
 * 2. Compresses the base layer; preserves the detail layer.
 * 3. Reconstruct and apply per-pixel scale to colour channels.
 *
 * Uses shared memory for the spatial neighbourhood window.
 *
 * @param d_linearF   In/out float BGR
 * @param d_lum       Luminance buffer (input)
 * @param d_scratch   Temporary float buffer H×W (caller-allocated on device)
 * @param radius      Bilateral filter spatial radius
 * @param sigmaS      Bilateral spatial sigma
 * @param sigmaR      Bilateral range sigma (log-lum domain)
 * @param width / height
 */
void launchLocalToneMap(
    float*       d_linearF,
    const float* d_lum,
    float*       d_scratch,
    int          radius,
    float        sigmaS,
    float        sigmaR,
    int          width,
    int          height);

/**
 * Stage 3 — Gamma correction + saturation boost + float→uint8.
 *
 * Applies:
 *   1. Saturation in YCbCr-ish space (chroma scaling).
 *   2. Display gamma:  out = clamp(in, 0, 1) ^ (1/gamma) * 255
 *   3. Writes back to uint8 BGR.
 *
 * @param d_linearF   Input float BGR  (tone-mapped, [0,1] range)
 * @param d_bgr       Output uint8 BGR
 * @param saturation  Chroma multiplier  (1.0 = unchanged)
 * @param gamma       Display gamma      (2.2 typical)
 * @param width / height
 */
void launchLinearToRGB(
    const float* d_linearF,
    uint8_t*     d_bgr,
    float        saturation,
    float        gamma,
    int          width,
    int          height);

// ============================================================
// Host-side helpers  (run on CPU, called between CUDA stages)
// ============================================================

/**
 * Compute log-average and max luminance from device luminance buffer.
 * Uses cudaMemcpy to pull lum data to host; fine for 1 call/frame.
 */
void computeLuminanceStats(
    const float* d_lum,
    int          width,
    int          height,
    float&       outAvgLogLum,
    float&       outMaxLum);
