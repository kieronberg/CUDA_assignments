#pragma once
#include <cuda_runtime.h>
#include <vector>
#include <memory>
#include <algorithm>

enum class FilterType { Blur, Sharpen, EdgeDetect, Emboss, WipeTransition };

struct FilterConfig {
    FilterType type;
    float param1; // Used for Sigma or Transition Progress
};

class FilterPipeline {
private:
    int width, height;
    size_t frameSize;
    uchar3* d_bufferA = nullptr;
    uchar3* d_bufferB = nullptr;
    cudaStream_t streamProcess;
    std::vector<FilterConfig> stages;

public:
    FilterPipeline(int w, int h) : width(w), height(h) {
        frameSize = width * height * sizeof(uchar3);
        cudaMalloc(&d_bufferA, frameSize);
        cudaMalloc(&d_bufferB, frameSize);
        cudaStreamCreate(&streamProcess);
    }

    ~FilterPipeline() {
        cudaFree(d_bufferA);
        cudaFree(d_bufferB);
        cudaStreamDestroy(streamProcess);
    }

    void addFilter(FilterType type, float p1 = 0.0f) { stages.push_back({type, p1}); }
    void clear() { stages.clear(); }
    
    // Process frame using ping-pong buffers
    uchar3* execute(const uchar3* d_input);
};