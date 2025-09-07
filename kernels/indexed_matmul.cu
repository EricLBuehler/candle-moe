#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

#include <cstdint>
#include <stdio.h>

#define CUDA_CHECK(call)                                                       \
  do {                                                                         \
    cudaError_t err = call;                                                    \
    if (err != cudaSuccess) {                                                  \
      fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,         \
              cudaGetErrorString(err));                                        \
      exit(err);                                                               \
    }                                                                          \
  } while (0)

// CUDA kernel for indexed matrix multiplication (MoE)
// This performs: C[indices[i]] += A[i] * B[expert_id] * weights[i]
// where B contains all expert weights concatenated

template<typename T>
__global__ void indexed_matmul_kernel(
    const T* input,           // Input tensor [num_tokens, hidden_dim]
    const T* expert_weights,  // All expert weights [num_experts, in_dim, out_dim]
    const float* routing_weights, // Routing weights [num_tokens, num_selected_experts]
    const uint32_t* expert_indices, // Expert indices [num_tokens, num_selected_experts]
    T* output,                // Output tensor [num_tokens, out_dim]
    int num_tokens,
    int hidden_dim,
    int out_dim,
    int num_selected_experts,
    int expert_stride        // in_dim * out_dim
) {
    const int token_idx = blockIdx.x;
    const int out_idx = blockIdx.y * blockDim.x + threadIdx.x;

    if (token_idx >= num_tokens || out_idx >= out_dim) {
        return;
    }

    float sum = 0.0f;

    // Process each selected expert for this token
    for (int k = 0; k < num_selected_experts; k++) {
        int expert_id = expert_indices[token_idx * num_selected_experts + k];
        float weight = routing_weights[token_idx * num_selected_experts + k];

        // Compute dot product for this output element
        float dot_product = 0.0f;
        const T* expert_matrix = expert_weights + expert_id * expert_stride + out_idx;

        for (int h = 0; h < hidden_dim; h++) {
            dot_product += float(input[token_idx * hidden_dim + h]) * 
                          float(expert_matrix[h * out_dim]);
        }

        sum += dot_product * weight;
    }

    output[token_idx * out_dim + out_idx] = T(sum);
}

// Optimized version using shared memory for better memory access patterns
template<typename T>
__global__ void indexed_matmul_kernel_optimized(
    const T* input,
    const T* expert_weights,
    const float* routing_weights,
    const uint32_t* expert_indices,
    T* output,
    int num_tokens,
    int hidden_dim,
    int out_dim,
    int num_selected_experts,
    int expert_stride
) {
    extern __shared__ char shared_mem[];
    T* shared_input = (T*)shared_mem;

    const int token_idx = blockIdx.x;
    const int tid = threadIdx.x;
    const int block_size = blockDim.x;

    if (token_idx >= num_tokens) {
        return;
    }

    // Load input to shared memory
    for (int i = tid; i < hidden_dim; i += block_size) {
        shared_input[i] = input[token_idx * hidden_dim + i];
    }
    __syncthreads();

    // Each thread computes multiple output elements
    for (int out_idx = tid; out_idx < out_dim; out_idx += block_size) {
        float sum = 0.0f;

        for (int k = 0; k < num_selected_experts; k++) {
            int expert_id = expert_indices[token_idx * num_selected_experts + k];
            float weight = routing_weights[token_idx * num_selected_experts + k];

            float dot_product = 0.0f;
            const T* expert_matrix = expert_weights + expert_id * expert_stride + out_idx;

            for (int h = 0; h < hidden_dim; h++) {
                dot_product += float(shared_input[h]) * float(expert_matrix[h * out_dim]);
            }

            sum += dot_product * weight;
        }

        output[token_idx * out_dim + out_idx] = T(sum);
    }
}

#define CALL_INDEXED_MATMUL(T)                               \
  indexed_matmul_kernel<T><<<blocks, threads, 0, stream>>>(  \
    reinterpret_cast<T*>(input),                             \
    reinterpret_cast<T*>(expert_weights),                    \
    routing_weights,                                         \
    expert_indices,                                          \
    reinterpret_cast<T*>(output),                            \
    num_tokens,                                              \
    hidden_dim,                                              \
    out_dim,                                                 \
    num_selected_experts,                                    \
    expert_stride                                            \
  );

#define CALL_INDEXED_MATMUL_OPTIMIZED(T)                               \
  indexed_matmul_kernel_optimized<T><<<blocks, threads, 0, stream>>>(  \
    reinterpret_cast<T*>(input),                                       \
    reinterpret_cast<T*>(expert_weights),                              \
    routing_weights,                                                   \
    expert_indices,                                                    \
    reinterpret_cast<T*>(output),                                      \
    num_tokens,                                                        \
    hidden_dim,                                                        \
    out_dim,                                                           \
    num_selected_experts,                                              \
    expert_stride                                                      \
  );

// C interface
extern "C" {

void indexed_matmul(
    void* input,
    void* expert_weights,
    float* routing_weights,
    uint32_t* expert_indices,
    void* output,
    int num_tokens,
    int hidden_dim,
    int out_dim,
    int num_selected_experts,
    uint32_t dtype           // 0 => f16; 1 => bf16; 2 => f32
) {
    const cudaStream_t stream = 0;

    const int threads = 256;
    dim3 blocks(num_tokens, (out_dim + threads - 1) / threads);
    int expert_stride = hidden_dim * out_dim;

    if (dtype == 0) {
        CALL_INDEXED_MATMUL(half);
    } else if (dtype == 1) {
        CALL_INDEXED_MATMUL(__nv_bfloat16);
    } else {
        CALL_INDEXED_MATMUL(float);
    }
}

void indexed_matmul_optimized(
    void* input,
    void* expert_weights,
    float* routing_weights,
    uint32_t* expert_indices,
    void* output,
    int num_tokens,
    int hidden_dim,
    int out_dim,
    int num_selected_experts,
    uint32_t dtype           // 0 => f16; 1 => bf16; 2 => f32
) {
    const cudaStream_t stream = 0;

    const int threads = 256;
    dim3 blocks(num_tokens, (out_dim + threads - 1) / threads);
    int expert_stride = hidden_dim * out_dim;

    if (dtype == 0) {
        CALL_INDEXED_MATMUL_OPTIMIZED(half);
    } else if (dtype == 1) {
        CALL_INDEXED_MATMUL_OPTIMIZED(__nv_bfloat16);
    } else {
        CALL_INDEXED_MATMUL_OPTIMIZED(float);
    }
}

} // extern "C"
