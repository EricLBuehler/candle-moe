#pragma once

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <mma.h>
#include <cstdint>

using namespace nvcuda;

// =============================================================================
// Architecture Detection
// =============================================================================

#if defined(__CUDA_ARCH__)
    #define CUDA_ARCH __CUDA_ARCH__
#else
    #define CUDA_ARCH 750
#endif

#define IS_SM75_OR_HIGHER (CUDA_ARCH >= 750)
#define IS_SM80_OR_HIGHER (CUDA_ARCH >= 800)
#define IS_SM90_OR_HIGHER (CUDA_ARCH >= 900)

// =============================================================================
// Common Constants
// =============================================================================

#define WARP_SIZE 32
#define MAX_WARPS_PER_BLOCK 32
#define WMMA_M 16
#define WMMA_N 16
#define WMMA_K 16

// =============================================================================
// GEMM Tile Configurations
// =============================================================================

namespace gemv {
    constexpr int BLOCK_SIZE = 256;
    constexpr int WARPS_PER_BLOCK = 8;
    constexpr int K_UNROLL = 8;
    constexpr int VECTOR_SIZE = 8;
}

namespace gemm_small {
    constexpr int BLOCK_M = 32;
    constexpr int BLOCK_N = 64;
    constexpr int BLOCK_K = 32;
    constexpr int THREADS = 128;
    constexpr int WARPS_M = 2;
    constexpr int WARPS_N = 2;
    constexpr int WARP_TILE_M = BLOCK_M / WARPS_M;
    constexpr int WARP_TILE_N = BLOCK_N / WARPS_N;
    constexpr int SMEM_A = BLOCK_M * BLOCK_K;
    constexpr int SMEM_B = BLOCK_K * BLOCK_N;
    constexpr int SMEM_C = BLOCK_M * BLOCK_N;
}

namespace gemm_medium {
    constexpr int BLOCK_M = 64;
    constexpr int BLOCK_N = 64;
    constexpr int BLOCK_K = 32;
    constexpr int THREADS = 128;
    constexpr int WARPS_M = 2;
    constexpr int WARPS_N = 2;
    constexpr int WARP_TILE_M = BLOCK_M / WARPS_M;
    constexpr int WARP_TILE_N = BLOCK_N / WARPS_N;
    constexpr int SMEM_A = BLOCK_M * BLOCK_K;
    constexpr int SMEM_B = BLOCK_K * BLOCK_N;
    constexpr int SMEM_C = BLOCK_M * BLOCK_N;
}

namespace gemm_large {
    constexpr int BLOCK_M = 128;
    constexpr int BLOCK_N = 128;
    constexpr int BLOCK_K = 32;
    constexpr int THREADS = 256;
    constexpr int WARPS_M = 4;
    constexpr int WARPS_N = 2;
    constexpr int WARP_TILE_M = BLOCK_M / WARPS_M;
    constexpr int WARP_TILE_N = BLOCK_N / WARPS_N;
    constexpr int SMEM_A = BLOCK_M * BLOCK_K;
    constexpr int SMEM_B = BLOCK_K * BLOCK_N;
    constexpr int SMEM_C = BLOCK_M * BLOCK_N;
}

namespace gemm_async {
    constexpr int BLOCK_M = 64;
    constexpr int BLOCK_N = 64;
    constexpr int BLOCK_K = 32;
    constexpr int THREADS = 128;
    constexpr int STAGES = 2;
    constexpr int WARPS_M = 2;
    constexpr int WARPS_N = 2;
    constexpr int WARP_TILE_M = BLOCK_M / WARPS_M;
    constexpr int WARP_TILE_N = BLOCK_N / WARPS_N;
}

namespace thresholds {
    constexpr int DIRECT_MAX_TOKENS = 32;
    constexpr int SMALL_GEMM_MAX_TOKENS = 64;
    constexpr int MEDIUM_GEMM_MAX_TOKENS = 256;
}

namespace direct_config {
    constexpr int BLOCK_SIZE = 256;
    constexpr int MAX_TOKENS_PER_EXPERT = 32;
    constexpr int VECTOR_SIZE = 8;
}

// =============================================================================
// Activation Functions
// =============================================================================

__device__ __forceinline__ float silu_f32(float x) {
    return x / (1.0f + __expf(-x));
}

__device__ __forceinline__ float gelu_f32(float x) {
    const float sqrt_2_over_pi = 0.7978845608f;
    const float coef = 0.044715f;
    float x3 = x * x * x;
    return 0.5f * x * (1.0f + tanhf(sqrt_2_over_pi * (x + coef * x3)));
}

__device__ __forceinline__ float relu_f32(float x) {
    return fmaxf(0.0f, x);
}

__device__ __forceinline__ float apply_activation(float x, int act_type) {
    switch (act_type) {
        case 0: return silu_f32(x);
        case 1: return gelu_f32(x);
        case 2: return relu_f32(x);
        default: return silu_f32(x);
    }
}

__device__ __forceinline__ half silu_half(half x) {
    return __float2half(silu_f32(__half2float(x)));
}

__device__ __forceinline__ half gelu_half(half x) {
    return __float2half(gelu_f32(__half2float(x)));
}

__device__ __forceinline__ half relu_half(half x) {
    return __float2half(relu_f32(__half2float(x)));
}

__device__ __forceinline__ half apply_activation_half(half x, int act_type) {
    return __float2half(apply_activation(__half2float(x), act_type));
}

__device__ __forceinline__ __nv_bfloat16 silu_bf16(__nv_bfloat16 x) {
    return __float2bfloat16(silu_f32(__bfloat162float(x)));
}

__device__ __forceinline__ __nv_bfloat16 gelu_bf16(__nv_bfloat16 x) {
    return __float2bfloat16(gelu_f32(__bfloat162float(x)));
}

__device__ __forceinline__ __nv_bfloat16 relu_bf16(__nv_bfloat16 x) {
    return __float2bfloat16(relu_f32(__bfloat162float(x)));
}

__device__ __forceinline__ __nv_bfloat16 apply_activation_bf16(__nv_bfloat16 x, int act_type) {
    return __float2bfloat16(apply_activation(__bfloat162float(x), act_type));
}

// =============================================================================
// Vectorized Load/Store Helpers
// =============================================================================

__device__ __forceinline__ float4 load_float4(const half* ptr) {
    return *reinterpret_cast<const float4*>(ptr);
}

__device__ __forceinline__ void store_float4(half* ptr, float4 val) {
    *reinterpret_cast<float4*>(ptr) = val;
}

__device__ __forceinline__ float2 load_float2(const half* ptr) {
    return *reinterpret_cast<const float2*>(ptr);
}

__device__ __forceinline__ void store_float2(half* ptr, float2 val) {
    *reinterpret_cast<float2*>(ptr) = val;
}

__device__ __forceinline__ half2 load_half2(const half* ptr) {
    return *reinterpret_cast<const half2*>(ptr);
}

__device__ __forceinline__ void store_half2(half* ptr, half2 val) {
    *reinterpret_cast<half2*>(ptr) = val;
}

__device__ __forceinline__ float4 load_float4_bf16(const __nv_bfloat16* ptr) {
    return *reinterpret_cast<const float4*>(ptr);
}

__device__ __forceinline__ void store_float4_bf16(__nv_bfloat16* ptr, float4 val) {
    *reinterpret_cast<float4*>(ptr) = val;
}

__device__ __forceinline__ float2 load_float2_bf16(const __nv_bfloat16* ptr) {
    return *reinterpret_cast<const float2*>(ptr);
}

__device__ __forceinline__ void store_float2_bf16(__nv_bfloat16* ptr, float2 val) {
    *reinterpret_cast<float2*>(ptr) = val;
}

__device__ __forceinline__ __nv_bfloat162 load_bf162(const __nv_bfloat16* ptr) {
    return *reinterpret_cast<const __nv_bfloat162*>(ptr);
}

__device__ __forceinline__ void store_bf162(__nv_bfloat16* ptr, __nv_bfloat162 val) {
    *reinterpret_cast<__nv_bfloat162*>(ptr) = val;
}

// =============================================================================
// Reduction Functions
// =============================================================================

__device__ __forceinline__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        val += __shfl_xor_sync(0xffffffff, val, offset);
    }
    return val;
}

__device__ __forceinline__ float warp_reduce_max(float val) {
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        val = fmaxf(val, __shfl_xor_sync(0xffffffff, val, offset));
    }
    return val;
}

template<int THREADS>
__device__ __forceinline__ float block_reduce_sum(float val) {
    __shared__ float shared[THREADS / WARP_SIZE];
    const int lane = threadIdx.x % WARP_SIZE;
    const int warp_id = threadIdx.x / WARP_SIZE;

    val = warp_reduce_sum(val);
    if (lane == 0) shared[warp_id] = val;
    __syncthreads();

    if (warp_id == 0) {
        val = (lane < THREADS / WARP_SIZE) ? shared[lane] : 0.0f;
        val = warp_reduce_sum(val);
    }
    return val;
}

// =============================================================================
// Atomic Operations
// =============================================================================

__device__ __forceinline__ void atomic_add_half(half* addr, half val) {
#if __CUDA_ARCH__ >= 700
    atomicAdd(addr, val);
#else
    unsigned int* addr_as_uint = (unsigned int*)((char*)addr - ((size_t)addr & 2));
    unsigned int old = *addr_as_uint;
    unsigned int assumed;
    do {
        assumed = old;
        unsigned int new_val = assumed;
        half* as_half = (half*)&new_val;
        if ((size_t)addr & 2) {
            as_half[1] = __hadd(as_half[1], val);
        } else {
            as_half[0] = __hadd(as_half[0], val);
        }
        old = atomicCAS(addr_as_uint, assumed, new_val);
    } while (assumed != old);
#endif
}

__device__ __forceinline__ void atomic_add_bf16(__nv_bfloat16* addr, __nv_bfloat16 val) {
#if __CUDA_ARCH__ >= 800
    atomicAdd(addr, val);
#else
    unsigned int* addr_as_uint = (unsigned int*)((char*)addr - ((size_t)addr & 2));
    unsigned int old = *addr_as_uint;
    unsigned int assumed;
    do {
        assumed = old;
        unsigned int new_val = assumed;
        __nv_bfloat16* as_bf16 = (__nv_bfloat16*)&new_val;
        if ((size_t)addr & 2) {
            as_bf16[1] = __float2bfloat16(__bfloat162float(as_bf16[1]) + __bfloat162float(val));
        } else {
            as_bf16[0] = __float2bfloat16(__bfloat162float(as_bf16[0]) + __bfloat162float(val));
        }
        old = atomicCAS(addr_as_uint, assumed, new_val);
    } while (assumed != old);
#endif
}

// =============================================================================
// Utility Functions
// =============================================================================

__device__ __forceinline__ int smem_index_no_conflict(int row, int col, int row_stride) {
    return row * (row_stride + 1) + col;
}

__device__ __forceinline__ size_t expert_weight_offset(int expert_id, int dim1, int dim2) {
    return (size_t)expert_id * dim1 * dim2;
}

#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        printf("CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
    } \
} while(0)

struct KernelConfig {
    dim3 grid;
    dim3 block;
    size_t smem_size;
};

inline KernelConfig get_expert_parallel_config(
    int num_tokens_per_expert, int output_dim, int block_m, int block_n,
    int threads, size_t smem_per_block, int num_experts
) {
    KernelConfig config;
    config.grid = dim3((output_dim + block_n - 1) / block_n,
                       (num_tokens_per_expert + block_m - 1) / block_m,
                       num_experts);
    config.block = dim3(threads);
    config.smem_size = smem_per_block;
    return config;
}

inline size_t dtype_size(uint32_t dtype) {
    switch (dtype) {
        case 0: return sizeof(half);
        case 1: return sizeof(__nv_bfloat16);
        case 2: return sizeof(float);
        default: return sizeof(half);
    }
}

// =============================================================================
// SM80+ Async Copy (Ampere and newer)
// =============================================================================

#ifdef SM80_OR_HIGHER
#include <cuda_pipeline.h>

__device__ __forceinline__ void cp_async_cg(void* dst, const void* src) {
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" :: "r"(static_cast<uint32_t>(__cvta_generic_to_shared(dst))), "l"(src));
}

__device__ __forceinline__ void cp_async_ca(void* dst, const void* src) {
    asm volatile("cp.async.ca.shared.global [%0], [%1], 16;\n" :: "r"(static_cast<uint32_t>(__cvta_generic_to_shared(dst))), "l"(src));
}

__device__ __forceinline__ void cp_async_commit() {
    asm volatile("cp.async.commit_group;\n");
}

template<int N>
__device__ __forceinline__ void cp_async_wait() {
    asm volatile("cp.async.wait_group %0;\n" :: "n"(N));
}

__device__ __forceinline__ void cp_async_wait_all() {
    asm volatile("cp.async.wait_all;\n");
}
#endif // SM80_OR_HIGHER

// =============================================================================
// SM90+ Warpgroup MMA (Hopper and newer)
// =============================================================================

#ifdef SM90_OR_HIGHER

#define WARPGROUP_SIZE 128
#define WARPS_PER_WARPGROUP 4

namespace wgmma_tiles {
    constexpr int M = 64;
    constexpr int N = 128;
    constexpr int K = 16;
}

namespace sm90_gemm {
    constexpr int BLOCK_M = 128;
    constexpr int BLOCK_N = 128;
    constexpr int BLOCK_K = 64;
    constexpr int THREADS = 128;
    constexpr int STAGES = 3;
}

// Warpgroup synchronization
__device__ __forceinline__ void warpgroup_arrive() {
    asm volatile("wgmma.fence.sync.aligned;\n");
}

__device__ __forceinline__ void warpgroup_commit() {
    asm volatile("wgmma.commit_group.sync.aligned;\n");
}

__device__ __forceinline__ void wgmma_commit() {
    asm volatile("wgmma.commit_group.sync.aligned;\n");
}

template<int N>
__device__ __forceinline__ void warpgroup_wait() {
    static_assert(N >= 0 && N <= 7, "wgmma wait group must be 0-7");
    asm volatile("wgmma.wait_group.sync.aligned %0;\n" :: "n"(N));
}

template<int N>
__device__ __forceinline__ void wgmma_wait() {
    static_assert(N >= 0 && N <= 7, "wgmma wait group must be 0-7");
    asm volatile("wgmma.wait_group.sync.aligned %0;\n" :: "n"(N));
}

// Matrix descriptor for shared memory operand
__device__ __forceinline__ uint64_t make_smem_desc(const void* ptr) {
    uint32_t addr = static_cast<uint32_t>(__cvta_generic_to_shared(ptr));
    uint64_t desc = (addr >> 4);
    desc |= (uint64_t(64) << 16);  // Leading dimension in bytes >> 4
    desc |= (uint64_t(1) << 62);   // Valid bit
    return desc;
}

// Swizzled shared memory index to avoid bank conflicts
__device__ __forceinline__ int swizzle_offset(int row, int col, int stride) {
    int swizzle = (row ^ (col >> 3)) & 0x7;
    return row * stride + (col ^ (swizzle << 3));
}

// Accumulator fragment for wgmma
template<int M, int N>
struct WgmmaAccumulator {
    static constexpr int NUM_ELEMENTS = (M * N) / WARPGROUP_SIZE;
    float data[NUM_ELEMENTS];

    __device__ __forceinline__ void clear() {
        #pragma unroll
        for (int i = 0; i < NUM_ELEMENTS; i++) {
            data[i] = 0.0f;
        }
    }
};

// wgmma M=64, N=128, K=16 for FP16
__device__ __forceinline__ void wgmma_m64n128k16_f16(
    float* acc,
    uint64_t desc_a,
    uint64_t desc_b
) {
    asm volatile(
        "{\n"
        ".reg .pred p;\n"
        "setp.ne.b32 p, %66, 0;\n"
        "wgmma.mma_async.sync.aligned.m64n128k16.f32.f16.f16 "
        "{%0,  %1,  %2,  %3,  %4,  %5,  %6,  %7,  "
        " %8,  %9,  %10, %11, %12, %13, %14, %15, "
        " %16, %17, %18, %19, %20, %21, %22, %23, "
        " %24, %25, %26, %27, %28, %29, %30, %31, "
        " %32, %33, %34, %35, %36, %37, %38, %39, "
        " %40, %41, %42, %43, %44, %45, %46, %47, "
        " %48, %49, %50, %51, %52, %53, %54, %55, "
        " %56, %57, %58, %59, %60, %61, %62, %63}, "
        "%64, %65, p, 1, 1, 0, 0;\n"
        "}\n"
        : "+f"(acc[0]),  "+f"(acc[1]),  "+f"(acc[2]),  "+f"(acc[3]),
          "+f"(acc[4]),  "+f"(acc[5]),  "+f"(acc[6]),  "+f"(acc[7]),
          "+f"(acc[8]),  "+f"(acc[9]),  "+f"(acc[10]), "+f"(acc[11]),
          "+f"(acc[12]), "+f"(acc[13]), "+f"(acc[14]), "+f"(acc[15]),
          "+f"(acc[16]), "+f"(acc[17]), "+f"(acc[18]), "+f"(acc[19]),
          "+f"(acc[20]), "+f"(acc[21]), "+f"(acc[22]), "+f"(acc[23]),
          "+f"(acc[24]), "+f"(acc[25]), "+f"(acc[26]), "+f"(acc[27]),
          "+f"(acc[28]), "+f"(acc[29]), "+f"(acc[30]), "+f"(acc[31]),
          "+f"(acc[32]), "+f"(acc[33]), "+f"(acc[34]), "+f"(acc[35]),
          "+f"(acc[36]), "+f"(acc[37]), "+f"(acc[38]), "+f"(acc[39]),
          "+f"(acc[40]), "+f"(acc[41]), "+f"(acc[42]), "+f"(acc[43]),
          "+f"(acc[44]), "+f"(acc[45]), "+f"(acc[46]), "+f"(acc[47]),
          "+f"(acc[48]), "+f"(acc[49]), "+f"(acc[50]), "+f"(acc[51]),
          "+f"(acc[52]), "+f"(acc[53]), "+f"(acc[54]), "+f"(acc[55]),
          "+f"(acc[56]), "+f"(acc[57]), "+f"(acc[58]), "+f"(acc[59]),
          "+f"(acc[60]), "+f"(acc[61]), "+f"(acc[62]), "+f"(acc[63])
        : "l"(desc_a), "l"(desc_b), "n"(1)
    );
}

// wgmma M=64, N=128, K=16 for BF16
__device__ __forceinline__ void wgmma_m64n128k16_bf16(
    float* acc,
    uint64_t desc_a,
    uint64_t desc_b
) {
    asm volatile(
        "{\n"
        ".reg .pred p;\n"
        "setp.ne.b32 p, %66, 0;\n"
        "wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 "
        "{%0,  %1,  %2,  %3,  %4,  %5,  %6,  %7,  "
        " %8,  %9,  %10, %11, %12, %13, %14, %15, "
        " %16, %17, %18, %19, %20, %21, %22, %23, "
        " %24, %25, %26, %27, %28, %29, %30, %31, "
        " %32, %33, %34, %35, %36, %37, %38, %39, "
        " %40, %41, %42, %43, %44, %45, %46, %47, "
        " %48, %49, %50, %51, %52, %53, %54, %55, "
        " %56, %57, %58, %59, %60, %61, %62, %63}, "
        "%64, %65, p, 1, 1, 0, 0;\n"
        "}\n"
        : "+f"(acc[0]),  "+f"(acc[1]),  "+f"(acc[2]),  "+f"(acc[3]),
          "+f"(acc[4]),  "+f"(acc[5]),  "+f"(acc[6]),  "+f"(acc[7]),
          "+f"(acc[8]),  "+f"(acc[9]),  "+f"(acc[10]), "+f"(acc[11]),
          "+f"(acc[12]), "+f"(acc[13]), "+f"(acc[14]), "+f"(acc[15]),
          "+f"(acc[16]), "+f"(acc[17]), "+f"(acc[18]), "+f"(acc[19]),
          "+f"(acc[20]), "+f"(acc[21]), "+f"(acc[22]), "+f"(acc[23]),
          "+f"(acc[24]), "+f"(acc[25]), "+f"(acc[26]), "+f"(acc[27]),
          "+f"(acc[28]), "+f"(acc[29]), "+f"(acc[30]), "+f"(acc[31]),
          "+f"(acc[32]), "+f"(acc[33]), "+f"(acc[34]), "+f"(acc[35]),
          "+f"(acc[36]), "+f"(acc[37]), "+f"(acc[38]), "+f"(acc[39]),
          "+f"(acc[40]), "+f"(acc[41]), "+f"(acc[42]), "+f"(acc[43]),
          "+f"(acc[44]), "+f"(acc[45]), "+f"(acc[46]), "+f"(acc[47]),
          "+f"(acc[48]), "+f"(acc[49]), "+f"(acc[50]), "+f"(acc[51]),
          "+f"(acc[52]), "+f"(acc[53]), "+f"(acc[54]), "+f"(acc[55]),
          "+f"(acc[56]), "+f"(acc[57]), "+f"(acc[58]), "+f"(acc[59]),
          "+f"(acc[60]), "+f"(acc[61]), "+f"(acc[62]), "+f"(acc[63])
        : "l"(desc_a), "l"(desc_b), "n"(1)
    );
}

// wgmma M=64, N=64, K=16 for FP16
__device__ __forceinline__ void wgmma_m64n64k16_f16(
    float* acc,
    uint64_t desc_a,
    uint64_t desc_b
) {
    asm volatile(
        "{\n"
        ".reg .pred p;\n"
        "setp.ne.b32 p, %34, 0;\n"
        "wgmma.mma_async.sync.aligned.m64n64k16.f32.f16.f16 "
        "{%0,  %1,  %2,  %3,  %4,  %5,  %6,  %7,  "
        " %8,  %9,  %10, %11, %12, %13, %14, %15, "
        " %16, %17, %18, %19, %20, %21, %22, %23, "
        " %24, %25, %26, %27, %28, %29, %30, %31}, "
        "%32, %33, p, 1, 1, 0, 0;\n"
        "}\n"
        : "+f"(acc[0]),  "+f"(acc[1]),  "+f"(acc[2]),  "+f"(acc[3]),
          "+f"(acc[4]),  "+f"(acc[5]),  "+f"(acc[6]),  "+f"(acc[7]),
          "+f"(acc[8]),  "+f"(acc[9]),  "+f"(acc[10]), "+f"(acc[11]),
          "+f"(acc[12]), "+f"(acc[13]), "+f"(acc[14]), "+f"(acc[15]),
          "+f"(acc[16]), "+f"(acc[17]), "+f"(acc[18]), "+f"(acc[19]),
          "+f"(acc[20]), "+f"(acc[21]), "+f"(acc[22]), "+f"(acc[23]),
          "+f"(acc[24]), "+f"(acc[25]), "+f"(acc[26]), "+f"(acc[27]),
          "+f"(acc[28]), "+f"(acc[29]), "+f"(acc[30]), "+f"(acc[31])
        : "l"(desc_a), "l"(desc_b), "n"(1)
    );
}

// wgmma M=64, N=64, K=16 for BF16
__device__ __forceinline__ void wgmma_m64n64k16_bf16(
    float* acc,
    uint64_t desc_a,
    uint64_t desc_b
) {
    asm volatile(
        "{\n"
        ".reg .pred p;\n"
        "setp.ne.b32 p, %34, 0;\n"
        "wgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 "
        "{%0,  %1,  %2,  %3,  %4,  %5,  %6,  %7,  "
        " %8,  %9,  %10, %11, %12, %13, %14, %15, "
        " %16, %17, %18, %19, %20, %21, %22, %23, "
        " %24, %25, %26, %27, %28, %29, %30, %31}, "
        "%32, %33, p, 1, 1, 0, 0;\n"
        "}\n"
        : "+f"(acc[0]),  "+f"(acc[1]),  "+f"(acc[2]),  "+f"(acc[3]),
          "+f"(acc[4]),  "+f"(acc[5]),  "+f"(acc[6]),  "+f"(acc[7]),
          "+f"(acc[8]),  "+f"(acc[9]),  "+f"(acc[10]), "+f"(acc[11]),
          "+f"(acc[12]), "+f"(acc[13]), "+f"(acc[14]), "+f"(acc[15]),
          "+f"(acc[16]), "+f"(acc[17]), "+f"(acc[18]), "+f"(acc[19]),
          "+f"(acc[20]), "+f"(acc[21]), "+f"(acc[22]), "+f"(acc[23]),
          "+f"(acc[24]), "+f"(acc[25]), "+f"(acc[26]), "+f"(acc[27]),
          "+f"(acc[28]), "+f"(acc[29]), "+f"(acc[30]), "+f"(acc[31])
        : "l"(desc_a), "l"(desc_b), "n"(1)
    );
}

// Async bulk copy from global to shared memory
__device__ __forceinline__ void cp_async_bulk_global_to_shared(
    void* smem_ptr,
    const void* gmem_ptr,
    int bytes
) {
    uint32_t smem_addr = static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
    asm volatile(
        "cp.async.bulk.shared.global [%0], [%1], %2;\n"
        :: "r"(smem_addr), "l"(gmem_ptr), "r"(bytes)
    );
}

// Named barrier for warpgroup synchronization
__device__ __forceinline__ void warpgroup_barrier_sync(int barrier_id) {
    asm volatile("bar.sync %0, 128;\n" :: "r"(barrier_id));
}

__device__ __forceinline__ void warpgroup_barrier_arrive(int barrier_id) {
    asm volatile("bar.arrive %0, 128;\n" :: "r"(barrier_id));
}

// Store accumulator to shared memory
template<int M, int N>
__device__ __forceinline__ void store_accumulator_to_smem(
    float* smem,
    const float* acc,
    int stride
) {
    const int tid = threadIdx.x;
    const int warp_id = tid / 32;
    const int lane_id = tid % 32;

    constexpr int ELEMS_PER_THREAD = (M * N) / WARPGROUP_SIZE;

    #pragma unroll
    for (int i = 0; i < ELEMS_PER_THREAD; i++) {
        int row = (warp_id * 16 + lane_id / 4 + (i / (N/4)) * 8) % M;
        int col = (lane_id % 4) * 2 + (i % (N/4)) * 8;
        if (col < N) {
            smem[row * stride + col] = acc[i];
        }
    }
}

#endif // SM90_OR_HIGHER
