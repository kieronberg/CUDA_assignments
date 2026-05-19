// File: basic_cuda_matrix.cu
#include <iostream>
#include <cuda_runtime.h>
#include <cmath>
#include <chrono>

void transposeCPU(const float* in, float* out, int width, int height) {
    for(int y = 0; y < height; y++) {
        for(int x = 0; x < width; x++) {
            out[x * height + y] = in[y * width + x];
        }
    }
}

__global__ void transposeNaive(const float* in, float* out, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        out[x * height + y] = in[y * width + x];
    }
}

bool verifyCorrectness(const float* expected, const float* actual, int size) {
    const float TOLERANCE = 1e-5;
    for (int i = 0; i < size; i++) {
        if (std::abs(expected[i] - actual[i]) > TOLERANCE) {
            return false;
        }
    }
    return true;
}

int main() {
    int width = 4096;
    int height = 4096;
    int size = width * height;
    size_t bytes = size * sizeof(float);

    float* h_in = new float[size];
    float* h_outGPU = new float[size];
    float* h_outCPU = new float[size];

    for (int i = 0; i < size; i++) {
        h_in[i] = static_cast<float>(i);
    }

    float *d_in, *d_out;
    cudaMalloc((void**)&d_in, bytes);
    cudaMalloc((void**)&d_out, bytes);
    cudaMemcpy(d_in, h_in, bytes, cudaMemcpyHostToDevice);

    dim3 dimBlock(32, 32, 1);
    dim3 dimGrid((width + dimBlock.x - 1) / dimBlock.x, 
                 (height + dimBlock.y - 1) / dimBlock.y, 1);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Warm-up kernel (optional but good for accurate timing)
    transposeNaive<<<dimGrid, dimBlock>>>(d_in, d_out, width, height);
    cudaDeviceSynchronize();

    // Timing GPU Naive
    cudaEventRecord(start);
    transposeNaive<<<dimGrid, dimBlock>>>(d_in, d_out, width, height);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float gpu_ms = 0;
    cudaEventElapsedTime(&gpu_ms, start, stop);

    cudaMemcpy(h_outGPU, d_out, bytes, cudaMemcpyDeviceToHost);

    // Timing CPU
    auto cpu_start = std::chrono::high_resolution_clock::now();
    transposeCPU(h_in, h_outCPU, width, height);
    auto cpu_stop = std::chrono::high_resolution_clock::now();
    std::chrono::duration<float, std::milli> cpu_duration = cpu_stop - cpu_start;

    if (verifyCorrectness(h_outCPU, h_outGPU, size)) {
        std::cout << "Basic Implementation Correct!" << std::endl;
        std::cout << "CPU Time:        " << cpu_duration.count() << " ms" << std::endl;
        std::cout << "Naive GPU Time:  " << gpu_ms << " ms" << std::endl;
    } else {
        std::cout << "Mismatch detected!" << std::endl;
    }

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_in);
    cudaFree(d_out);
    delete[] h_in;
    delete[] h_outGPU;
    delete[] h_outCPU;

    return 0;
}