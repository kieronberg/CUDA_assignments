// File: optimized_cuda_matrix.cu
#include <iostream>
#include <cuda_runtime.h>
#include <cmath>
#include <chrono>

#define TILE_DIM 32
#define BLOCK_ROWS 8

void transposeCPU(const float* in, float* out, int width, int height) {
    for(int y = 0; y < height; y++) {
        for(int x = 0; x < width; x++) {
            out[x * height + y] = in[y * width + x];
        }
    }
}

__global__ void transposeOptimized(const float* in, float* out, int width, int height) {
    __shared__ float tile[TILE_DIM][TILE_DIM + 1];

    int x = blockIdx.x * TILE_DIM + threadIdx.x;
    int y = blockIdx.y * TILE_DIM + threadIdx.y;

    for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS) {
        if (x < width && (y + j) < height) {
            tile[threadIdx.y + j][threadIdx.x] = in[(y + j) * width + x];
        }
    }

    __syncthreads();

    x = blockIdx.y * TILE_DIM + threadIdx.x;
    y = blockIdx.x * TILE_DIM + threadIdx.y;

    for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS) {
        if (x < height && (y + j) < width) {
            out[(y + j) * height + x] = tile[threadIdx.x][threadIdx.y + j];
        }
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

    float *in, *outOpt, *outCPU;

    cudaMallocManaged(&in, bytes);
    cudaMallocManaged(&outOpt, bytes);
    outCPU = new float[size];

    for (int i = 0; i < size; i++) {
        in[i] = static_cast<float>(i);
    }

    int device = -1;
    cudaGetDevice(&device);
    cudaMemPrefetchAsync(in, bytes, device, NULL);

    dim3 dimBlock(TILE_DIM, BLOCK_ROWS, 1);
    dim3 dimGrid((width + TILE_DIM - 1) / TILE_DIM, (height + TILE_DIM - 1) / TILE_DIM, 1);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Warm-up kernel
    transposeOptimized<<<dimGrid, dimBlock>>>(in, outOpt, width, height);
    cudaDeviceSynchronize();

    // Timing GPU Optimized
    cudaEventRecord(start);
    transposeOptimized<<<dimGrid, dimBlock>>>(in, outOpt, width, height);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float gpu_ms = 0;
    cudaEventElapsedTime(&gpu_ms, start, stop);

    // Timing CPU
    auto cpu_start = std::chrono::high_resolution_clock::now();
    transposeCPU(in, outCPU, width, height);
    auto cpu_stop = std::chrono::high_resolution_clock::now();
    std::chrono::duration<float, std::milli> cpu_duration = cpu_stop - cpu_start;

    if(verifyCorrectness(outCPU, outOpt, size)) {
        std::cout << "Optimized Implementation Correct!" << std::endl;
        std::cout << "CPU Time:            " << cpu_duration.count() << " ms" << std::endl;
        std::cout << "Optimized GPU Time:  " << gpu_ms << " ms" << std::endl;
    } else {
        std::cout << "Mismatch detected!" << std::endl;
    }

    cudaEventDestroy(start);
    cudaFree(in);
    cudaFree(outOpt);
    delete[] outCPU;

    return 0;
}