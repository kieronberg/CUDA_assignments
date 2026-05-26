#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cuda_runtime.h>
#include <math.h>

// --- Filters ---
float boxBlur3x3[9] = {
    1 / 9.0f, 1 / 9.0f, 1 / 9.0f,
    1 / 9.0f, 1 / 9.0f, 1 / 9.0f,
    1 / 9.0f, 1 / 9.0f, 1 / 9.0f};

float gaussianBlur5x5[25] = {
    1 / 273.0f, 4 / 273.0f, 7 / 273.0f, 4 / 273.0f, 1 / 273.0f,
    4 / 273.0f, 16 / 273.0f, 26 / 273.0f, 16 / 273.0f, 4 / 273.0f,
    7 / 273.0f, 26 / 273.0f, 41 / 273.0f, 26 / 273.0f, 7 / 273.0f,
    4 / 273.0f, 16 / 273.0f, 26 / 273.0f, 16 / 273.0f, 4 / 273.0f,
    1 / 273.0f, 4 / 273.0f, 7 / 273.0f, 4 / 273.0f, 1 / 273.0f};

// Utility to check for CUDA errors
#define CHECK_CUDA_ERROR(call)                                                \
    {                                                                         \
        cudaError_t err = call;                                               \
        if (err != cudaSuccess)                                               \
        {                                                                     \
            fprintf(stderr, "CUDA Error: %s at line %d in file %s\n",         \
                    cudaGetErrorString(err), __LINE__, __FILE__);             \
            exit(EXIT_FAILURE);                                               \
        }                                                                     \
    }

#define CLAMP(x, min_val, max_val) ((x) < (min_val) ? (min_val) : ((x) > (max_val) ? (max_val) : (x)))

typedef struct
{
    unsigned char *data;
    int width;
    int height;
    int channels; 
} Image;

// --- CPU implementation ---
void convolutionCPU(const Image *input, Image *output, const float *filter, int filterWidth)
{
    int r = filterWidth / 2;
    for (int y = 0; y < input->height; ++y) {
        for (int x = 0; x < input->width; ++x) {
            for (int c = 0; c < input->channels; ++c) {
                float sum = 0.0f;
                for (int fy = -r; fy <= r; ++fy) {
                    for (int fx = -r; fx <= r; ++fx) {
                        int iy = CLAMP(y + fy, 0, input->height - 1);
                        int ix = CLAMP(x + fx, 0, input->width - 1);

                        float fval = filter[(fy + r) * filterWidth + (fx + r)];
                        unsigned char pval = input->data[(iy * input->width + ix) * input->channels + c];
                        sum += fval * pval;
                    }
                }
                int outIdx = (y * input->width + x) * input->channels + c;
                output->data[outIdx] = (unsigned char)CLAMP((int)sum, 0, 255);
            }
        }
    }
}

// --- Basic/Naive GPU implementation ---
// Each thread computes one output pixel across all channels reading directly from Global Memory.
__global__ void convolutionKernelNaive(unsigned char *input, unsigned char *output,
                                       float *filter, int filterWidth,
                                       int width, int height, int channels)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        int r = filterWidth / 2;
        for (int c = 0; c < channels; ++c) {
            float sum = 0.0f;
            for (int fy = -r; fy <= r; ++fy) {
                for (int fx = -r; fx <= r; ++fx) {
                    int iy = min(max(y + fy, 0), height - 1);
                    int ix = min(max(x + fx, 0), width - 1);
                    float fval = filter[(fy + r) * filterWidth + (fx + r)];
                    unsigned char pval = input[(iy * width + ix) * channels + c];
                    sum += fval * pval;
                }
            }
            output[(y * width + x) * channels + c] = (unsigned char)min(max((int)sum, 0), 255);
        }
    }
}

// --- Constant Memory Filter ---
__constant__ float d_filter[81]; // Max 9x9 filter

// --- Optimized GPU implementation ---
// Uses Shared Memory for the image tile + halo, and Constant Memory for the filter.
__global__ void convolutionKernelShared(unsigned char *input, unsigned char *output,
                                        int filterWidth, int width, int height, int channels)
{
    extern __shared__ unsigned char s_mem[]; 

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int bx = blockIdx.x * blockDim.x;
    int by = blockIdx.y * blockDim.y;
    int x = bx + tx;
    int y = by + ty;

    int r = filterWidth / 2;
    int s_w = blockDim.x + 2 * r;
    int s_h = blockDim.y + 2 * r;
    int smem_elements = s_w * s_h * channels;

    // Collaborative loading of tile + halo into shared memory
    int tid = ty * blockDim.x + tx;
    int numThreads = blockDim.x * blockDim.y;

    for (int i = tid; i < smem_elements; i += numThreads) {
        int c = i % channels;
        int sx = (i / channels) % s_w;
        int sy = (i / channels) / s_w;

        int gx = bx + sx - r;
        int gy = by + sy - r;

        gx = min(max(gx, 0), width - 1);
        gy = min(max(gy, 0), height - 1);

        s_mem[sy * s_w * channels + sx * channels + c] = input[(gy * width + gx) * channels + c];
    }
    
    __syncthreads();

    // Compute convolution
    if (x < width && y < height) {
        for (int c = 0; c < channels; ++c) {
            float sum = 0.0f;
            for (int fy = -r; fy <= r; ++fy) {
                for (int fx = -r; fx <= r; ++fx) {
                    int sx = tx + r + fx;
                    int sy = ty + r + fy;
                    
                    float fval = d_filter[(fy + r) * filterWidth + (fx + r)];
                    unsigned char pval = s_mem[(sy * s_w + sx) * channels + c];
                    sum += fval * pval;
                }
            }
            output[(y * width + x) * channels + c] = (unsigned char)min(max((int)sum, 0), 255);
        }
    }
}

// --- Main execution ---
int main(int argc, char **argv)
{
    // 1. Generate a synthetic image for testing (e.g., 2048x2048 RGB)
    int imgWidth = 2048;
    int imgHeight = 2048;
    int imgChannels = 3;
    int imgSize = imgWidth * imgHeight * imgChannels * sizeof(unsigned char);

    Image h_input, h_outputCPU, h_outputNaive, h_outputShared;
    h_input.width = h_outputCPU.width = h_outputNaive.width = h_outputShared.width = imgWidth;
    h_input.height = h_outputCPU.height = h_outputNaive.height = h_outputShared.height = imgHeight;
    h_input.channels = h_outputCPU.channels = h_outputNaive.channels = h_outputShared.channels = imgChannels;

    h_input.data = (unsigned char *)malloc(imgSize);
    h_outputCPU.data = (unsigned char *)malloc(imgSize);
    h_outputNaive.data = (unsigned char *)malloc(imgSize);
    h_outputShared.data = (unsigned char *)malloc(imgSize);

    for (int i = 0; i < imgWidth * imgHeight * imgChannels; ++i) {
        h_input.data[i] = rand() % 256;
    }

    // 2. Define convolution filter (using 5x5 Gaussian)
    float *h_filter = gaussianBlur5x5;
    int filterWidth = 5;
    int filterSize = filterWidth * filterWidth * sizeof(float);

    // 3. Device memory allocation
    unsigned char *d_input, *d_outputNaive, *d_outputShared;
    float *d_filterNaive;

    CHECK_CUDA_ERROR(cudaMalloc(&d_input, imgSize));
    CHECK_CUDA_ERROR(cudaMalloc(&d_outputNaive, imgSize));
    CHECK_CUDA_ERROR(cudaMalloc(&d_outputShared, imgSize));
    CHECK_CUDA_ERROR(cudaMalloc(&d_filterNaive, filterSize));

    CHECK_CUDA_ERROR(cudaMemcpy(d_input, h_input.data, imgSize, cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_filterNaive, h_filter, filterSize, cudaMemcpyHostToDevice));
    
    // Copy filter to constant memory for optimized kernel
    CHECK_CUDA_ERROR(cudaMemcpyToSymbol(d_filter, h_filter, filterSize));

    // 4. Timing utilities
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float ms_cpu, ms_naive, ms_shared;

    // --- Run CPU implementation ---
    printf("Running CPU Convolution...\n");
    cudaEventRecord(start);
    convolutionCPU(&h_input, &h_outputCPU, h_filter, filterWidth);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ms_cpu, start, stop);

    // Setup launch config
    dim3 block(16, 16);
    dim3 grid((imgWidth + block.x - 1) / block.x, (imgHeight + block.y - 1) / block.y);

    // --- Run Naive GPU implementation ---
    printf("Running Naive GPU Convolution...\n");
    cudaEventRecord(start);
    convolutionKernelNaive<<<grid, block>>>(d_input, d_outputNaive, d_filterNaive, filterWidth, imgWidth, imgHeight, imgChannels);
    CHECK_CUDA_ERROR(cudaPeekAtLastError());
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ms_naive, start, stop);
    CHECK_CUDA_ERROR(cudaMemcpy(h_outputNaive.data, d_outputNaive, imgSize, cudaMemcpyDeviceToHost));

    // --- Run Shared Memory GPU implementation ---
    printf("Running Shared Memory GPU Convolution...\n");
    int r = filterWidth / 2;
    int smem_size = (block.x + 2 * r) * (block.y + 2 * r) * imgChannels * sizeof(unsigned char);
    
    cudaEventRecord(start);
    convolutionKernelShared<<<grid, block, smem_size>>>(d_input, d_outputShared, filterWidth, imgWidth, imgHeight, imgChannels);
    CHECK_CUDA_ERROR(cudaPeekAtLastError());
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ms_shared, start, stop);
    CHECK_CUDA_ERROR(cudaMemcpy(h_outputShared.data, d_outputShared, imgSize, cudaMemcpyDeviceToHost));

    // 5. Compare Results & Performance
    printf("\n--- Performance Results ---\n");
    printf("CPU Time:        %8.2f ms\n", ms_cpu);
    printf("Naive GPU Time:  %8.2f ms (Speedup: %.2fx)\n", ms_naive, ms_cpu / ms_naive);
    printf("Shared GPU Time: %8.2f ms (Speedup: %.2fx vs CPU, %.2fx vs Naive)\n", ms_shared, ms_cpu / ms_shared, ms_naive / ms_shared);

    // Verify Correctness (Compare against CPU)
    int errorsNaive = 0, errorsShared = 0;
    for (int i = 0; i < imgWidth * imgHeight * imgChannels; ++i) {
        if (abs(h_outputCPU.data[i] - h_outputNaive.data[i]) > 1) errorsNaive++;
        if (abs(h_outputCPU.data[i] - h_outputShared.data[i]) > 1) errorsShared++;
    }
    
    printf("\n--- Verification ---\n");
    printf("Naive Errors:  %d\n", errorsNaive);
    printf("Shared Errors: %d\n", errorsShared);

    // Cleanup
    free(h_input.data); free(h_outputCPU.data); free(h_outputNaive.data); free(h_outputShared.data);
    cudaFree(d_input); cudaFree(d_outputNaive); cudaFree(d_outputShared); cudaFree(d_filterNaive);

    return 0;
}