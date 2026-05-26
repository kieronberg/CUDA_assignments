# 🚀 CUDA Convolution Performance Report

## 1. Implementation Summary

The project compared three distinct approaches to 2D image convolution:

* **Sequential CPU:** A triple-nested loop ($y, x, channels$) using clamping for boundary conditions. It is computationally expensive ($\mathcal{O}(N^2 \cdot K^2)$ complexity) and lacks parallelism.
* **Naive GPU Kernel:** Parallelized across pixels. Each thread is responsible for one output pixel. While fast, it suffers from **redundant global memory access** (neighboring threads fetch the same pixels repeatedly).
* **Optimized Shared Memory Kernel:** Utilizes `__shared__` memory to load an image tile once. All threads in a block then access the local tile. Additionally, the filter is stored in `__constant__` memory to leverage the constant cache.

---

## 2. Performance Analysis

The following results were obtained using a $2048 \times 2048$ RGB image and a $5 \times 5$ Gaussian filter.

### Execution Time Comparison

| Implementation | Execution Time (ms) | Speedup (vs CPU) | Speedup (vs Naive) |
| --- | --- | --- | --- |
| **CPU Sequential** | $3079.69$ | $1.0\times$ | — |
| **Naive GPU** | $3.61$ | $853.17\times$ | $1.0\times$ |
| **Shared Memory GPU** | **$2.05$** | **$1505.47\times$** | **$1.76\times$** |

### Performance Visualization (Log Scale Concept)

> **Note:** The CPU time is so high that on a linear chart, the GPU bars would be invisible (less than $0.1\%$ of the height).

```text
CPU     |############################################################| 3079.7ms
Naive   |#                                                           | 3.6ms
Shared  |                                                            | 2.1ms

```

---

## 3. Deep Dive: Why the Speedup?

### The CPU Bottleneck

The CPU implementation is limited by **instruction latency** and **memory bandwidth**. For every single pixel, the CPU must calculate 25 memory addresses, fetch them from RAM (or L3 cache if lucky), and perform the FMA (Fused Multiply-Add) operations sequentially. At $2048^2 \times 3$ channels, this is over **314 million** inner-loop iterations.

### The Naive vs. Shared Memory Gap

The $1.76\times$ improvement from Naive to Shared memory is significant.

1. **Memory Coalescing:** The shared memory version loads data in a coalesced fashion.
2. **Reduction in Global Loads:** In the Naive version, for a $5 \times 5$ filter, each pixel is read from global memory up to **25 times** by different threads. In the Shared version, each pixel is read from global memory **exactly once** (ignoring the halo overhead), then served from high-speed shared memory.

---

## 4. Identifying Bottlenecks (Nsight Systems/Compute)

If we were to look at this in **Nvidia Nsight**, we would likely identify the following:

* **Naive Kernel Bottleneck: Memory Bound.**
* The "Memory Throughput" would be high, but the "Compute Throughput" would be low. The SMs (Streaming Multiprocessors) are frequently idling while waiting for the Global Memory controller to fulfill redundant requests.


* **Shared Memory Kernel Bottleneck: Instruction/Latency Bound.**
* The bottleneck shifts away from memory bandwidth. It may now be limited by **Shared Memory Bank Conflicts** (if multiple threads in a warp access the same bank) or the overhead of `__syncthreads()`.


* **Occupancy:** With large shared memory tiles, occupancy might drop (fewer blocks can run on one SM at once), but the trade-off for memory speed usually results in a net gain.

---

## 5. Further Optimizations

To squeeze even more performance out of this kernel, we could explore:

1. **Separable Convolution:** A $5 \times 5$ Gaussian blur can be broken into two $1D$ passes ($1 \times 5$ and $5 \times 1$). This reduces the complexity from $\mathcal{O}(K^2)$ to $\mathcal{O}(2K)$.

$$\text{Operations per pixel: } 25 \rightarrow 10$$


2. **Texture Memory:** Using CUDA's `cudaTextureObject_t` would allow us to use the hardware's built-in 2D spatial cache and automatic boundary handling (clamping/wrapping), potentially outperforming manual shared memory management for some filter sizes.
3. **Vectorized Loads (`float4`):**
Instead of reading `unsigned char` (1 byte), we can read 4 or 16 bytes at a time using vectorized types, maximizing the bus width utilization between VRAM and the SM.
4. **Register Pressure:**
If the filter is very small ($3 \times 3$), unrolling the loops entirely can help the compiler keep values in registers rather than fetching them from any memory at all.

---

**Verdict:** Your optimized implementation successfully cut the GPU execution time nearly in half compared to the naive approach, proving that **memory hierarchy management** is the king of CUDA optimization.