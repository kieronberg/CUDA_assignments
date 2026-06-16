// =============================================================
// main.cpp — CUDA webcam filter with HDR tone-mapping
// =============================================================
#include <opencv2/opencv.hpp>
#include <cuda_runtime.h>
#include <iostream>
#include <stdexcept>
#include <chrono>

#include "utils/filter_utils.h"
#include "utils/input_handler.h"
#include "kernels/kernels.h"
#include "input_args_parser/input_args_parser.h"

// ------------------------------------------------------------------
// Timing helper
// ------------------------------------------------------------------
using Clock = std::chrono::high_resolution_clock;
static double msElapsed(Clock::time_point t0) {
    return std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
}

// ------------------------------------------------------------------
// main
// ------------------------------------------------------------------
int main(int argc, char** argv) {
    InputArgs args;
    if (!parseArgs(argc, argv, args)) return 0;

    // Validate GPU
    int devCount = 0;
    cudaGetDeviceCount(&devCount);
    if (devCount == 0) {
        std::cerr << "No CUDA-capable GPU found.\n";
        return 1;
    }

    // ---- Open input source ----
    cv::VideoCapture cap;
    cv::Mat staticFrame;
    bool isStaticImage = false;

    InputSource src = stringToInputSource(args.inputSource);

    switch (src) {
    case InputSource::WEBCAM:
        cap.open(args.deviceId);
        if (!cap.isOpened()) {
            std::cerr << "Cannot open camera " << args.deviceId << "\n";
            return 1;
        }
        break;
    case InputSource::VIDEO:
        cap.open(args.inputPath);
        if (!cap.isOpened()) {
            std::cerr << "Cannot open video: " << args.inputPath << "\n";
            return 1;
        }
        break;
    case InputSource::IMAGE:
        staticFrame = cv::imread(args.inputPath);
        if (staticFrame.empty()) {
            std::cerr << "Cannot read image: " << args.inputPath << "\n";
            return 1;
        }
        isStaticImage = true;
        break;
    case InputSource::SYNTHETIC:
        staticFrame = generateSyntheticFrame(
            stringToSyntheticPattern(args.syntheticPat), 1280, 720);
        isStaticImage = true;
        break;
    }

    // ---- Prepare filter ----
    FilterType ft = stringToFilterType(args.filterType);
    bool isHdr = (ft == FilterType::HDR_TONEMAPPING);

    // GPU buffers (convolution path)
    uint8_t* d_convIn  = nullptr;
    uint8_t* d_convOut = nullptr;
    float*   d_kernel  = nullptr;
    FilterKernel fk;

    // GPU buffers (HDR path — persistent across frames for zero-alloc loop)
    HdrDeviceBuffers hdrBufs;
    HdrParams        hdrParams;

    if (isHdr) {
        hdrParams = args.toHdrParams();
        std::cout << "HDR tone-mapping enabled\n"
                  << "  algorithm : " << args.toneMappingAlg << "\n"
                  << "  exposure  : " << args.exposure       << "\n"
                  << "  gamma     : " << args.gamma          << "\n"
                  << "  saturation: " << args.saturation     << "\n";
    } else {
        fk = createFilterKernel(ft, args.kernelSize, args.sigma, args.intensity);
        cudaMalloc(&d_kernel, fk.size * fk.size * sizeof(float));
        cudaMemcpy(d_kernel, fk.data.data(),
                   fk.size * fk.size * sizeof(float), cudaMemcpyHostToDevice);
    }

    std::cout << "Press ESC to exit.\n";
    cv::Mat frame, filtered;
    int frameCount = 0;
    double gpuTotal = 0.0, cpuTotal = 0.0;

    while (true) {
        // ---- Grab frame ----
        if (isStaticImage) {
            frame = staticFrame.clone();
        } else {
            cap >> frame;
            if (frame.empty()) break;
        }

        auto t0 = Clock::now();

        // ---- Apply filter ----
        if (isHdr) {
            filtered = applyHdrToneMapping(frame, hdrBufs, hdrParams);
            double gpuMs = msElapsed(t0);
            gpuTotal += gpuMs;

            if (args.benchmark) {
                auto t1 = Clock::now();
                cv::Mat cpuResult = applyHdrToneMappingCPU(frame, hdrParams);
                double cpuMs = msElapsed(t1);
                cpuTotal += cpuMs;
                ++frameCount;
                std::printf("[Frame %4d] GPU: %6.2f ms  CPU: %6.2f ms  speedup: %.1fx\n",
                            frameCount, gpuMs, cpuMs, cpuMs / (gpuMs + 0.001));
            }
        } else {
            // Allocate / reallocate convolution buffers on size change
            int W = frame.cols, H = frame.rows;
            cudaMalloc(&d_convIn,  (size_t)W * H * 3);
            cudaMalloc(&d_convOut, (size_t)W * H * 3);
            cudaMemcpy(d_convIn, frame.ptr<uint8_t>(),
                       (size_t)W * H * 3, cudaMemcpyHostToDevice);
            launchConvolutionKernel(d_convIn, d_convOut,
                                    W, H, 3, d_kernel, fk.size);
            filtered.create(H, W, CV_8UC3);
            cudaMemcpy(filtered.ptr<uint8_t>(), d_convOut,
                       (size_t)W * H * 3, cudaMemcpyDeviceToHost);
            cudaFree(d_convIn);  d_convIn  = nullptr;
            cudaFree(d_convOut); d_convOut = nullptr;
        }

        // ---- Display ----
        if (args.preview) {
            cv::Mat side;
            cv::hconcat(frame, filtered, side);
            cv::imshow("Original | Filtered", side);
        } else {
            cv::imshow("CUDA Filter", filtered);
        }

        if (cv::waitKey(1) == 27) break;   // ESC
        if (isStaticImage) cv::waitKey(0); // pause on static
    }

    // ---- Summary ----
    if (args.benchmark && frameCount > 0) {
        std::printf("\nAverage over %d frames:\n"
                    "  GPU: %.2f ms/frame\n"
                    "  CPU: %.2f ms/frame\n"
                    "  Speedup: %.1fx\n",
                    frameCount,
                    gpuTotal / frameCount,
                    cpuTotal / frameCount,
                    cpuTotal / (gpuTotal + 0.001));
    }

    // ---- Cleanup ----
    if (d_kernel)  cudaFree(d_kernel);
    if (d_convIn)  cudaFree(d_convIn);
    if (d_convOut) cudaFree(d_convOut);
    cap.release();
    cv::destroyAllWindows();
    return 0;
}
