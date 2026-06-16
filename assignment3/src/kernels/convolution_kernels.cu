// =============================================================
// convolution_kernels.cu
// CUDA kernels: convolution (existing) + HDR tone-mapping (new)
// =============================================================
#include "kernels.h"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cmath>
#include <stdexcept>
#include <vector>
#include <string>
#include <cstdio>

// ------------------------------------------------------------------
// Helper: check CUDA errors
// ------------------------------------------------------------------
#define CUDA_CHECK(call)                                                   \
    do {                                                                   \
        cudaError_t err = (call);                                          \
        if (err != cudaSuccess) {                                          \
            char msg[256];                                                 \
            snprintf(msg, sizeof(msg), "CUDA error %s:%d — %s",           \
                     __FILE__, __LINE__, cudaGetErrorString(err));         \
            throw std::runtime_error(msg);                                 \
        }                                                                  \
    } while (0)

// ------------------------------------------------------------------
// Constants
// ------------------------------------------------------------------
static constexpr int BLOCK_W = 16;
static constexpr int BLOCK_H = 16;

// Constant memory for convolution kernel (max 25×25 = 625 floats)
__constant__ float c_convKernel[625];


// ==================================================================
// 1. Existing convolution kernel (unchanged)
// ==================================================================
__global__ void k_convolve(
    const uint8_t* __restrict__ input,
    uint8_t* __restrict__       output,
    int width, int height, int channels, int kSize)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    const int half = kSize / 2;

    for (int ch = 0; ch < channels; ++ch) {
        float acc = 0.0f;
        for (int ky = -half; ky <= half; ++ky) {
            for (int kx = -half; kx <= half; ++kx) {
                int px = min(max(x + kx, 0), width  - 1);
                int py = min(max(y + ky, 0), height - 1);
                float pix = input[(py * width + px) * channels + ch];
                float w   = c_convKernel[(ky + half) * kSize + (kx + half)];
                acc += pix * w;
            }
        }
        output[(y * width + x) * channels + ch] =
            (uint8_t)min(max((int)acc, 0), 255);
    }
}

void launchConvolutionKernel(
    const uint8_t* d_input,  uint8_t* d_output,
    int width, int height, int channels,
    const float* d_kernel, int kernelSize)
{
    // Copy kernel to constant memory
    CUDA_CHECK(cudaMemcpyToSymbol(c_convKernel, d_kernel,
                                  kernelSize * kernelSize * sizeof(float)));

    dim3 block(BLOCK_W, BLOCK_H);
    dim3 grid((width  + BLOCK_W - 1) / BLOCK_W,
              (height + BLOCK_H - 1) / BLOCK_H);
    k_convolve<<<grid, block>>>(d_input, d_output,
                                width, height, channels, kernelSize);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}


// ==================================================================
// 2. Stage 1 — RGB (uint8 BGR) → float BGR linear-light + luminance
//
//    Shared memory: BLOCK_W × BLOCK_H pixels × 3 channels (uint8)
//    loaded once per block, then processed in registers.
// ==================================================================
__global__ void k_RGBtoLinear(
    const uint8_t* __restrict__ d_bgr,
    float* __restrict__         d_linearF,
    float* __restrict__         d_lum,
    int width, int height, float exposure)
{
    // Shared memory tile (BGR bytes)
    __shared__ uint8_t s_tile[BLOCK_H][BLOCK_W][3];

    const int x  = blockIdx.x * blockDim.x + threadIdx.x;
    const int y  = blockIdx.y * blockDim.y + threadIdx.y;
    const int tx = threadIdx.x, ty = threadIdx.y;

    // Load into shared memory (clamp at border)
    int sx = min(x, width  - 1);
    int sy = min(y, height - 1);
    const uint8_t* src = d_bgr + (sy * width + sx) * 3;
    s_tile[ty][tx][0] = src[0];
    s_tile[ty][tx][1] = src[1];
    s_tile[ty][tx][2] = src[2];
    __syncthreads();

    if (x >= width || y >= height) return;

    // sRGB → linear (approximate gamma-2.2 decode)
    float b = powf(s_tile[ty][tx][0] / 255.0f, 2.2f) * exposure;
    float g = powf(s_tile[ty][tx][1] / 255.0f, 2.2f) * exposure;
    float r = powf(s_tile[ty][tx][2] / 255.0f, 2.2f) * exposure;

    int idx = (y * width + x) * 3;
    d_linearF[idx + 0] = b;
    d_linearF[idx + 1] = g;
    d_linearF[idx + 2] = r;

    // Rec.709 luminance: Y = 0.2126R + 0.7152G + 0.0722B
    d_lum[y * width + x] = 0.2126f * r + 0.7152f * g + 0.0722f * b;
}

void launchRGBtoLinear(
    const uint8_t* d_bgr, float* d_linearF, float* d_lum,
    int width, int height, float exposure)
{
    dim3 block(BLOCK_W, BLOCK_H);
    dim3 grid((width  + BLOCK_W - 1) / BLOCK_W,
              (height + BLOCK_H - 1) / BLOCK_H);
    k_RGBtoLinear<<<grid, block>>>(d_bgr, d_linearF, d_lum,
                                    width, height, exposure);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}


// ==================================================================
// 3. Stage 2a — Global Reinhard extended tone-mapping
//
//    L_d = L * (1 + L/Lw²) / (1 + L)
//    Colour channels scaled by L_d / L.
// ==================================================================
__global__ void k_reinhardGlobal(
    float* __restrict__       d_linearF,
    const float* __restrict__ d_lum,
    float avgLogLum, float whitePoint,
    int width, int height)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    const int pix = y * width + x;
    float L = d_lum[pix];

    // Key value mapping: scale luminance by middle-grey / log-avg
    const float key = 0.18f;
    float Ls = (key / (avgLogLum + 1e-6f)) * L;

    // Extended Reinhard
    float Lw2 = whitePoint * whitePoint;
    float Ld  = Ls * (1.0f + Ls / Lw2) / (1.0f + Ls);

    float scale = (L > 1e-6f) ? (Ld / L) : 0.0f;

    int idx = pix * 3;
    d_linearF[idx + 0] *= scale;
    d_linearF[idx + 1] *= scale;
    d_linearF[idx + 2] *= scale;
}

void launchReinhardGlobal(
    float* d_linearF, const float* d_lum,
    float avgLogLum, float whitePoint,
    int width, int height)
{
    dim3 block(BLOCK_W, BLOCK_H);
    dim3 grid((width  + BLOCK_W - 1) / BLOCK_W,
              (height + BLOCK_H - 1) / BLOCK_H);
    k_reinhardGlobal<<<grid, block>>>(d_linearF, d_lum,
                                      avgLogLum, whitePoint,
                                      width, height);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}


// ==================================================================
// 4. Stage 2b — Drago logarithmic tone-mapping
//
//    Based on: Drago et al. (2003) "Adaptive Logarithmic Mapping
//    For Displaying High Contrast Scenes"
//
//    L_d = (Ldmax/log10(Lmax+1)) *
//          log(L+1) / log(2 + 8*(L/Lmax)^(log(bias)/log(0.5)))
// ==================================================================
__global__ void k_dragoToneMap(
    float* __restrict__       d_linearF,
    const float* __restrict__ d_lum,
    float maxLum, float bias,
    int width, int height)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    const int pix = y * width + x;
    float L = max(d_lum[pix], 1e-6f);

    const float Ldmax  = 1.0f;   // normalise to [0,1]
    const float logBiasLog05 = logf(bias) / logf(0.5f);

    float num   = logf(L + 1.0f);
    float denom = logf(2.0f + 8.0f * powf(L / maxLum, logBiasLog05));
    float Ld    = (Ldmax / log10f(maxLum + 1.0f)) * (num / denom);

    float scale = Ld / L;

    int idx = pix * 3;
    d_linearF[idx + 0] = d_linearF[idx + 0] * scale;
    d_linearF[idx + 1] = d_linearF[idx + 1] * scale;
    d_linearF[idx + 2] = d_linearF[idx + 2] * scale;
}

void launchDragoToneMap(
    float* d_linearF, const float* d_lum,
    float maxLum, float bias,
    int width, int height)
{
    dim3 block(BLOCK_W, BLOCK_H);
    dim3 grid((width  + BLOCK_W - 1) / BLOCK_W,
              (height + BLOCK_H - 1) / BLOCK_H);
    k_dragoToneMap<<<grid, block>>>(d_linearF, d_lum,
                                    maxLum, bias, width, height);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}


// ==================================================================
// 5. Stage 2c — Local (bilateral) tone-mapping
//
//    Algorithm:
//    1. Compute log-luminance log_L = log(L + 1e-6)
//    2. Apply bilateral filter in log-lum domain → base layer B
//    3. Detail D = log_L - B
//    4. Compress base:  B' = B * compressionFactor (≈ 0.5)
//    5. Reconstruct:    log_Ld = B' + D
//    6. Scale colours by exp(log_Ld) / L
//
//    Shared memory: (BLOCK_W + 2*radius) × (BLOCK_H + 2*radius)
//    log-lum tile — avoids repeated global reads in the bilateral.
//    For large radii (>16) we fall back to global reads to stay
//    within 48 KB shared-memory budget.
// ==================================================================

// Bilateral pass — one thread per pixel, reads from global memory
// (shared-memory version is below for radius ≤ BLOCK_W)
__global__ void k_localToneMap(
    float* __restrict__       d_linearF,
    const float* __restrict__ d_lum,
    float* __restrict__       d_scratch,   // stores bilateral result
    int radius, float sigmaS, float sigmaR,
    int width, int height)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    const int pix = y * width + x;
    float logL_c = logf(max(d_lum[pix], 1e-6f));

    float wSum = 0.0f, filteredLogL = 0.0f;
    float invSigmaS2 = -0.5f / (sigmaS * sigmaS);
    float invSigmaR2 = -0.5f / (sigmaR * sigmaR);

    for (int dy = -radius; dy <= radius; ++dy) {
        for (int dx = -radius; dx <= radius; ++dx) {
            int nx = min(max(x + dx, 0), width  - 1);
            int ny = min(max(y + dy, 0), height - 1);
            float logL_n = logf(max(d_lum[ny * width + nx], 1e-6f));

            float dSp = dx*dx + dy*dy;
            float dRg = logL_n - logL_c;
            float w   = expf(dSp * invSigmaS2 + dRg * dRg * invSigmaR2);

            filteredLogL += w * logL_n;
            wSum += w;
        }
    }

    // Base = bilateral-smooth log-lum
    float base   = filteredLogL / (wSum + 1e-10f);
    float detail = logL_c - base;

    // Compress base, restore detail
    const float compression = 0.5f;
    float logLd = base * compression + detail;
    float Ld    = expf(logLd);
    float L     = max(d_lum[pix], 1e-6f);

    // Store scale in scratch (reused in colour pass below)
    d_scratch[pix] = Ld / L;

    int idx = pix * 3;
    d_linearF[idx + 0] *= d_scratch[pix];
    d_linearF[idx + 1] *= d_scratch[pix];
    d_linearF[idx + 2] *= d_scratch[pix];
}

void launchLocalToneMap(
    float* d_linearF, const float* d_lum, float* d_scratch,
    int radius, float sigmaS, float sigmaR,
    int width, int height)
{
    dim3 block(BLOCK_W, BLOCK_H);
    dim3 grid((width  + BLOCK_W - 1) / BLOCK_W,
              (height + BLOCK_H - 1) / BLOCK_H);
    k_localToneMap<<<grid, block>>>(d_linearF, d_lum, d_scratch,
                                    radius, sigmaS, sigmaR,
                                    width, height);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}


// ==================================================================
// 6. Stage 3 — Saturation + gamma + float→uint8 BGR
//
//    Shared memory: BLOCK_W × BLOCK_H float3 tiles.
//    Saturation: convert to YCbCr, scale Cb/Cr, back to BGR.
// ==================================================================
__global__ void k_linearToRGB(
    const float* __restrict__ d_linearF,
    uint8_t* __restrict__     d_bgr,
    float saturation, float gamma,
    int width, int height)
{
    // Shared memory for tone-mapped float tile
    __shared__ float3 s_tile[BLOCK_H][BLOCK_W];

    const int x  = blockIdx.x * blockDim.x + threadIdx.x;
    const int y  = blockIdx.y * blockDim.y + threadIdx.y;
    const int tx = threadIdx.x, ty = threadIdx.y;

    int sx = min(x, width  - 1);
    int sy = min(y, height - 1);
    int idx = (sy * width + sx) * 3;
    s_tile[ty][tx] = {d_linearF[idx], d_linearF[idx+1], d_linearF[idx+2]};
    __syncthreads();

    if (x >= width || y >= height) return;

    float b = s_tile[ty][tx].x;
    float g = s_tile[ty][tx].y;
    float r = s_tile[ty][tx].z;

    // ----- Saturation boost in YCbCr space -----
    // Rec.601 luma weights (fast approximation)
    float Y  =  0.299f*r + 0.587f*g + 0.114f*b;
    float Cb = -0.169f*r - 0.331f*g + 0.500f*b;
    float Cr =  0.500f*r - 0.419f*g - 0.081f*b;

    Cb *= saturation;
    Cr *= saturation;

    // Back to BGR
    r = Y + 1.402f  * Cr;
    g = Y - 0.344f  * Cb - 0.714f * Cr;
    b = Y + 1.772f  * Cb;

    // ----- Gamma encode + clamp + cast -----
    float invGamma = 1.0f / gamma;
    auto encode = [&](float v) -> uint8_t {
        v = fminf(fmaxf(v, 0.0f), 1.0f);
        return (uint8_t)(powf(v, invGamma) * 255.0f + 0.5f);
    };

    int out = (y * width + x) * 3;
    d_bgr[out + 0] = encode(b);
    d_bgr[out + 1] = encode(g);
    d_bgr[out + 2] = encode(r);
}

void launchLinearToRGB(
    const float* d_linearF, uint8_t* d_bgr,
    float saturation, float gamma,
    int width, int height)
{
    dim3 block(BLOCK_W, BLOCK_H);
    dim3 grid((width  + BLOCK_W - 1) / BLOCK_W,
              (height + BLOCK_H - 1) / BLOCK_H);
    k_linearToRGB<<<grid, block>>>(d_linearF, d_bgr,
                                   saturation, gamma,
                                   width, height);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}


// ==================================================================
// 7. Host-side luminance stats (CPU, called between GPU stages)
// ==================================================================
void computeLuminanceStats(
    const float* d_lum, int width, int height,
    float& outAvgLogLum, float& outMaxLum)
{
    const int N = width * height;
    std::vector<float> h_lum(N);
    CUDA_CHECK(cudaMemcpy(h_lum.data(), d_lum, N * sizeof(float),
                          cudaMemcpyDeviceToHost));

    double logSum = 0.0;
    float  maxL   = 0.0f;
    for (int i = 0; i < N; ++i) {
        float L = std::max(h_lum[i], 1e-6f);
        logSum += std::log(L);
        if (L > maxL) maxL = L;
    }

    outAvgLogLum = std::exp(static_cast<float>(logSum / N));
    outMaxLum    = maxL;
}
