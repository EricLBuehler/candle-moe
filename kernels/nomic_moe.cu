#include "common.cuh"

// Fused direct kernel for tiny batches - skips preprocessing entirely
// Nomic MoE: output = act(input @ gate) @ up.T
// Each block handles one (token, expert_slot) pair
__global__ void __launch_bounds__(direct_config::BLOCK_SIZE)
nomic_direct_fused_vec_kernel(
    const half* __restrict__ input,
    const half* __restrict__ gate_weights,    // [num_experts, hidden_dim, intermediate_dim]
    const half* __restrict__ up_weights,      // [num_experts, hidden_dim, intermediate_dim]
    const float* __restrict__ routing_weights,
    const uint32_t* __restrict__ expert_indices,
    half* __restrict__ output,
    int num_tokens,
    int hidden_dim,
    int intermediate_dim,
    int top_k,
    int activation_type
) {
    using namespace direct_config;

    const int block_idx = blockIdx.x;
    const int token_idx = block_idx / top_k;
    const int expert_slot = block_idx % top_k;

    if (token_idx >= num_tokens) return;

    const int expert_id = expert_indices[token_idx * top_k + expert_slot];
    const float routing_weight = routing_weights[token_idx * top_k + expert_slot];

    const int tid = threadIdx.x;
    const int warp_id = tid / WARP_SIZE;
    const int lane_id = tid % WARP_SIZE;
    const int num_warps = BLOCK_SIZE / WARP_SIZE;

    const half* gate_w = gate_weights + (size_t)expert_id * hidden_dim * intermediate_dim;
    const half* up_w = up_weights + (size_t)expert_id * hidden_dim * intermediate_dim;
    const half* input_row = input + (size_t)token_idx * hidden_dim;

    extern __shared__ char smem[];
    float* s_intermediate = reinterpret_cast<float*>(smem);  // [intermediate_dim]

    // Phase 1: Gate projection with activation
    // intermediate = act(input @ gate)
    for (int out_base = warp_id; out_base < intermediate_dim; out_base += num_warps) {
        float gate_sum = 0.0f;

        for (int k = lane_id; k < hidden_dim; k += WARP_SIZE) {
            float in_val = __half2float(input_row[k]);
            gate_sum += in_val * __half2float(gate_w[k * intermediate_dim + out_base]);
        }

        // Warp reduce
        #pragma unroll
        for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
            gate_sum += __shfl_down_sync(0xffffffff, gate_sum, offset);
        }

        if (lane_id == 0) {
            s_intermediate[out_base] = apply_activation(gate_sum, activation_type);
        }
    }
    __syncthreads();

    // Phase 2: Up projection (transposed) - output = intermediate @ up.T
    // up is [hidden_dim, intermediate_dim], so up.T is [intermediate_dim, hidden_dim]
    // We want: output[h] = sum_i(intermediate[i] * up[h, i])
    for (int out_base = warp_id; out_base < hidden_dim; out_base += num_warps) {
        float sum = 0.0f;

        for (int k = lane_id; k < intermediate_dim; k += WARP_SIZE) {
            // up[out_base, k] = up_w[out_base * intermediate_dim + k]
            sum += s_intermediate[k] * __half2float(up_w[out_base * intermediate_dim + k]);
        }

        // Warp reduce
        #pragma unroll
        for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
            sum += __shfl_down_sync(0xffffffff, sum, offset);
        }

        if (lane_id == 0) {
            half result = __float2half(sum * routing_weight);
            if (top_k == 1) {
                output[token_idx * hidden_dim + out_base] = result;
            } else {
                atomicAdd(&output[token_idx * hidden_dim + out_base], result);
            }
        }
    }
}

#ifndef NO_BF16_KERNEL
// BF16 version of direct fused kernel for Nomic MoE
__global__ void __launch_bounds__(direct_config::BLOCK_SIZE)
nomic_direct_fused_vec_bf16_kernel(
    const __nv_bfloat16* __restrict__ input,
    const __nv_bfloat16* __restrict__ gate_weights,
    const __nv_bfloat16* __restrict__ up_weights,
    const float* __restrict__ routing_weights,
    const uint32_t* __restrict__ expert_indices,
    __nv_bfloat16* __restrict__ output,
    int num_tokens,
    int hidden_dim,
    int intermediate_dim,
    int top_k,
    int activation_type
) {
    using namespace direct_config;

    const int block_idx = blockIdx.x;
    const int token_idx = block_idx / top_k;
    const int expert_slot = block_idx % top_k;

    if (token_idx >= num_tokens) return;

    const int expert_id = expert_indices[token_idx * top_k + expert_slot];
    const float routing_weight = routing_weights[token_idx * top_k + expert_slot];

    const int tid = threadIdx.x;
    const int warp_id = tid / WARP_SIZE;
    const int lane_id = tid % WARP_SIZE;
    const int num_warps = BLOCK_SIZE / WARP_SIZE;

    const __nv_bfloat16* gate_w = gate_weights + (size_t)expert_id * hidden_dim * intermediate_dim;
    const __nv_bfloat16* up_w = up_weights + (size_t)expert_id * hidden_dim * intermediate_dim;
    const __nv_bfloat16* input_row = input + (size_t)token_idx * hidden_dim;

    extern __shared__ char smem[];
    float* s_intermediate = reinterpret_cast<float*>(smem);

    // Phase 1: Gate projection with activation
    for (int out_base = warp_id; out_base < intermediate_dim; out_base += num_warps) {
        float gate_sum = 0.0f;

        for (int k = lane_id; k < hidden_dim; k += WARP_SIZE) {
            float in_val = __bfloat162float(input_row[k]);
            gate_sum += in_val * __bfloat162float(gate_w[k * intermediate_dim + out_base]);
        }

        #pragma unroll
        for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
            gate_sum += __shfl_down_sync(0xffffffff, gate_sum, offset);
        }

        if (lane_id == 0) {
            s_intermediate[out_base] = apply_activation(gate_sum, activation_type);
        }
    }
    __syncthreads();

    // Phase 2: Up projection (transposed)
    for (int out_base = warp_id; out_base < hidden_dim; out_base += num_warps) {
        float sum = 0.0f;

        for (int k = lane_id; k < intermediate_dim; k += WARP_SIZE) {
            sum += s_intermediate[k] * __bfloat162float(up_w[out_base * intermediate_dim + k]);
        }

        #pragma unroll
        for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
            sum += __shfl_down_sync(0xffffffff, sum, offset);
        }

        if (lane_id == 0) {
            __nv_bfloat16 result = __float2bfloat16(sum * routing_weight);
            if (top_k == 1) {
                output[token_idx * hidden_dim + out_base] = result;
            } else {
                atomicAdd(&output[token_idx * hidden_dim + out_base], result);
            }
        }
    }
}
#endif // NO_BF16_KERNEL

// FP32 version of direct fused kernel for Nomic MoE
__global__ void __launch_bounds__(direct_config::BLOCK_SIZE)
nomic_direct_fused_vec_f32_kernel(
    const float* __restrict__ input,
    const float* __restrict__ gate_weights,
    const float* __restrict__ up_weights,
    const float* __restrict__ routing_weights,
    const uint32_t* __restrict__ expert_indices,
    float* __restrict__ output,
    int num_tokens,
    int hidden_dim,
    int intermediate_dim,
    int top_k,
    int activation_type
) {
    using namespace direct_config;

    const int block_idx = blockIdx.x;
    const int token_idx = block_idx / top_k;
    const int expert_slot = block_idx % top_k;

    if (token_idx >= num_tokens) return;

    const int expert_id = expert_indices[token_idx * top_k + expert_slot];
    const float routing_weight = routing_weights[token_idx * top_k + expert_slot];

    const int tid = threadIdx.x;
    const int warp_id = tid / WARP_SIZE;
    const int lane_id = tid % WARP_SIZE;
    const int num_warps = BLOCK_SIZE / WARP_SIZE;

    const float* gate_w = gate_weights + (size_t)expert_id * hidden_dim * intermediate_dim;
    const float* up_w = up_weights + (size_t)expert_id * hidden_dim * intermediate_dim;
    const float* input_row = input + (size_t)token_idx * hidden_dim;

    extern __shared__ char smem[];
    float* s_intermediate = reinterpret_cast<float*>(smem);

    // Phase 1: Gate projection with activation
    for (int out_base = warp_id; out_base < intermediate_dim; out_base += num_warps) {
        float gate_sum = 0.0f;

        for (int k = lane_id; k < hidden_dim; k += WARP_SIZE) {
            float in_val = input_row[k];
            gate_sum += in_val * gate_w[k * intermediate_dim + out_base];
        }

        #pragma unroll
        for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
            gate_sum += __shfl_down_sync(0xffffffff, gate_sum, offset);
        }

        if (lane_id == 0) {
            s_intermediate[out_base] = apply_activation(gate_sum, activation_type);
        }
    }
    __syncthreads();

    // Phase 2: Up projection (transposed)
    for (int out_base = warp_id; out_base < hidden_dim; out_base += num_warps) {
        float sum = 0.0f;

        for (int k = lane_id; k < intermediate_dim; k += WARP_SIZE) {
            sum += s_intermediate[k] * up_w[out_base * intermediate_dim + k];
        }

        #pragma unroll
        for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
            sum += __shfl_down_sync(0xffffffff, sum, offset);
        }

        if (lane_id == 0) {
            float result = sum * routing_weight;
            if (top_k == 1) {
                output[token_idx * hidden_dim + out_base] = result;
            } else {
                atomicAdd(&output[token_idx * hidden_dim + out_base], result);
            }
        }
    }
}

__global__ void __launch_bounds__(gemm_small::THREADS)
nomic_gate_gemm_small_kernel(
    const half* __restrict__ input,
    const half* __restrict__ gate_weights,
    const int* __restrict__ sorted_token_ids,
    const int* __restrict__ expert_offsets,
    half* __restrict__ intermediate,
    int hidden_dim,
    int intermediate_dim,
    int activation_type
) {
    using namespace gemm_small;

    const int expert_id = blockIdx.z;
    const int expert_start = expert_offsets[expert_id];
    const int expert_end = expert_offsets[expert_id + 1];
    const int M = expert_end - expert_start;

    if (M == 0) return;

    const int block_m = blockIdx.y * BLOCK_M;
    const int block_n = blockIdx.x * BLOCK_N;

    if (block_m >= M || block_n >= intermediate_dim) return;

    const int tid = threadIdx.x;
    const int warp_id = tid / WARP_SIZE;
    const int warp_m = warp_id / WARPS_N;
    const int warp_n = warp_id % WARPS_N;

    extern __shared__ char smem[];
    half* s_input = reinterpret_cast<half*>(smem);
    half* s_gate = s_input + SMEM_A;

    const half* gate_w = gate_weights + (size_t)expert_id * hidden_dim * intermediate_dim;

    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> frag_a;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> frag_b[2];
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> frag_c[2];

    wmma::fill_fragment(frag_c[0], 0.0f);
    wmma::fill_fragment(frag_c[1], 0.0f);

    for (int k = 0; k < hidden_dim; k += BLOCK_K) {
        for (int i = tid; i < BLOCK_M * BLOCK_K / 8; i += THREADS) {
            int m = i / (BLOCK_K / 8);
            int kk = (i % (BLOCK_K / 8)) * 8;
            int global_m = block_m + m;
            int global_k = k + kk;

            float4 val = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
            if (global_m < M && global_k + 7 < hidden_dim) {
                int token_id = sorted_token_ids[expert_start + global_m];
                val = load_float4(&input[token_id * hidden_dim + global_k]);
            } else if (global_m < M) {
                half* ptr = reinterpret_cast<half*>(&val);
                int token_id = sorted_token_ids[expert_start + global_m];
                for (int j = 0; j < 8 && global_k + j < hidden_dim; j++) {
                    ptr[j] = input[token_id * hidden_dim + global_k + j];
                }
            }
            store_float4(&s_input[m * BLOCK_K + kk], val);
        }

        for (int i = tid; i < BLOCK_K * BLOCK_N / 8; i += THREADS) {
            int kk = i / (BLOCK_N / 8);
            int n = (i % (BLOCK_N / 8)) * 8;
            int global_k = k + kk;
            int global_n = block_n + n;

            float4 val = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
            if (global_k < hidden_dim && global_n + 7 < intermediate_dim) {
                val = load_float4(&gate_w[global_k * intermediate_dim + global_n]);
            }
            store_float4(&s_gate[kk * BLOCK_N + n], val);
        }
        __syncthreads();

        const int warp_row = warp_m * WMMA_M;
        const int warp_col = warp_n * WMMA_N * 2;

        #pragma unroll
        for (int kk = 0; kk < BLOCK_K; kk += WMMA_K) {
            wmma::load_matrix_sync(frag_a, &s_input[warp_row * BLOCK_K + kk], BLOCK_K);

            #pragma unroll
            for (int ni = 0; ni < 2; ni++) {
                int b_col = warp_col + ni * WMMA_N;
                wmma::load_matrix_sync(frag_b[ni], &s_gate[kk * BLOCK_N + b_col], BLOCK_N);
                wmma::mma_sync(frag_c[ni], frag_a, frag_b[ni], frag_c[ni]);
            }
        }
        __syncthreads();
    }

    float* s_out = reinterpret_cast<float*>(smem);

    const int warp_row = warp_m * WMMA_M;
    const int warp_col = warp_n * WMMA_N * 2;

    #pragma unroll
    for (int ni = 0; ni < 2; ni++) {
        int out_col = warp_col + ni * WMMA_N;
        wmma::store_matrix_sync(&s_out[warp_row * BLOCK_N + out_col], frag_c[ni], BLOCK_N, wmma::mem_row_major);
    }
    __syncthreads();

    for (int i = tid; i < BLOCK_M * BLOCK_N; i += THREADS) {
        int m = i / BLOCK_N;
        int n = i % BLOCK_N;
        int global_m = block_m + m;
        int global_n = block_n + n;

        if (global_m < M && global_n < intermediate_dim) {
            float val = apply_activation(s_out[m * BLOCK_N + n], activation_type);
            intermediate[(expert_start + global_m) * intermediate_dim + global_n] = __float2half(val);
        }
    }
}

__global__ void __launch_bounds__(gemm_small::THREADS)
nomic_up_gemm_small_kernel(
    const half* __restrict__ intermediate,
    const half* __restrict__ up_weights,
    const int* __restrict__ sorted_token_ids,
    const float* __restrict__ sorted_weights,
    const int* __restrict__ expert_offsets,
    half* __restrict__ output,
    int hidden_dim,
    int intermediate_dim,
    int top_k
) {
    using namespace gemm_small;

    const int expert_id = blockIdx.z;
    const int expert_start = expert_offsets[expert_id];
    const int expert_end = expert_offsets[expert_id + 1];
    const int M = expert_end - expert_start;

    if (M == 0) return;

    const int block_m = blockIdx.y * BLOCK_M;
    const int block_n = blockIdx.x * BLOCK_N;

    if (block_m >= M || block_n >= hidden_dim) return;

    const int tid = threadIdx.x;
    const int warp_id = tid / WARP_SIZE;
    const int warp_m = warp_id / WARPS_N;
    const int warp_n = warp_id % WARPS_N;

    extern __shared__ char smem[];
    // Data area - used for both load phase (s_inter, s_up) and output phase (s_out)
    half* s_inter = reinterpret_cast<half*>(smem);
    half* s_up = s_inter + SMEM_A;
    // Metadata area - placed AFTER output area to avoid overlap when s_out reuses smem
    // s_out needs SMEM_C floats = BLOCK_M * BLOCK_N * sizeof(float) bytes
    constexpr int DATA_AREA_BYTES = SMEM_C * sizeof(float);
    int* s_token_ids = reinterpret_cast<int*>(smem + DATA_AREA_BYTES);
    float* s_routing = reinterpret_cast<float*>(s_token_ids + BLOCK_M);

    const half* up_w = up_weights + (size_t)expert_id * hidden_dim * intermediate_dim;

    for (int i = tid; i < BLOCK_M; i += THREADS) {
        int global_m = block_m + i;
        if (global_m < M) {
            s_token_ids[i] = sorted_token_ids[expert_start + global_m];
            s_routing[i] = sorted_weights[expert_start + global_m];
        } else {
            s_token_ids[i] = 0;
            s_routing[i] = 0.0f;
        }
    }

    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> frag_a;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> frag_b[2];
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> frag_c[2];

    wmma::fill_fragment(frag_c[0], 0.0f);
    wmma::fill_fragment(frag_c[1], 0.0f);
    __syncthreads();

    for (int k = 0; k < intermediate_dim; k += BLOCK_K) {
        for (int i = tid; i < BLOCK_M * BLOCK_K / 8; i += THREADS) {
            int m = i / (BLOCK_K / 8);
            int kk = (i % (BLOCK_K / 8)) * 8;
            int global_m = block_m + m;
            int global_k = k + kk;

            float4 val = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
            if (global_m < M && global_k + 7 < intermediate_dim) {
                val = load_float4(&intermediate[(expert_start + global_m) * intermediate_dim + global_k]);
            }
            store_float4(&s_inter[m * BLOCK_K + kk], val);
        }

        for (int i = tid; i < BLOCK_K * BLOCK_N; i += THREADS) {
            int kk = i / BLOCK_N;
            int n = i % BLOCK_N;
            int global_k = k + kk;
            int global_n = block_n + n;

            half val = __float2half(0.0f);
            if (global_k < intermediate_dim && global_n < hidden_dim) {
                val = up_w[global_n * intermediate_dim + global_k];
            }
            s_up[kk * BLOCK_N + n] = val;
        }
        __syncthreads();

        const int warp_row = warp_m * WMMA_M;
        const int warp_col = warp_n * WMMA_N * 2;

        #pragma unroll
        for (int kk = 0; kk < BLOCK_K; kk += WMMA_K) {
            wmma::load_matrix_sync(frag_a, &s_inter[warp_row * BLOCK_K + kk], BLOCK_K);

            #pragma unroll
            for (int ni = 0; ni < 2; ni++) {
                int b_col = warp_col + ni * WMMA_N;
                wmma::load_matrix_sync(frag_b[ni], &s_up[kk * BLOCK_N + b_col], BLOCK_N);
                wmma::mma_sync(frag_c[ni], frag_a, frag_b[ni], frag_c[ni]);
            }
        }
        __syncthreads();
    }

    float* s_out = reinterpret_cast<float*>(smem);

    const int warp_row = warp_m * WMMA_M;
    const int warp_col = warp_n * WMMA_N * 2;

    #pragma unroll
    for (int ni = 0; ni < 2; ni++) {
        int out_col = warp_col + ni * WMMA_N;
        wmma::store_matrix_sync(&s_out[warp_row * BLOCK_N + out_col], frag_c[ni], BLOCK_N, wmma::mem_row_major);
    }
    __syncthreads();

    for (int i = tid; i < BLOCK_M * BLOCK_N; i += THREADS) {
        int m = i / BLOCK_N;
        int n = i % BLOCK_N;
        int global_m = block_m + m;
        int global_n = block_n + n;

        if (global_m < M && global_n < hidden_dim) {
            int token_id = s_token_ids[m];
            float weight = s_routing[m];
            float val = s_out[m * BLOCK_N + n] * weight;
            if (top_k == 1) {
                output[token_id * hidden_dim + global_n] = __float2half(val);
            } else {
                atomic_add_half(&output[token_id * hidden_dim + global_n], __float2half(val));
            }
        }
    }
}

__global__ void __launch_bounds__(gemm_large::THREADS)
nomic_gate_gemm_large_kernel(
    const half* __restrict__ input,
    const half* __restrict__ gate_weights,
    const int* __restrict__ sorted_token_ids,
    const int* __restrict__ expert_offsets,
    half* __restrict__ intermediate,
    int hidden_dim,
    int intermediate_dim,
    int activation_type
) {
    using namespace gemm_large;

    const int expert_id = blockIdx.z;
    const int expert_start = expert_offsets[expert_id];
    const int expert_end = expert_offsets[expert_id + 1];
    const int M = expert_end - expert_start;

    if (M == 0) return;

    const int block_m = blockIdx.y * BLOCK_M;
    const int block_n = blockIdx.x * BLOCK_N;

    if (block_m >= M || block_n >= intermediate_dim) return;

    const int tid = threadIdx.x;
    const int warp_id = tid / WARP_SIZE;
    const int warp_m = warp_id / WARPS_N;
    const int warp_n = warp_id % WARPS_N;

    extern __shared__ char smem[];
    half* s_input = reinterpret_cast<half*>(smem);
    half* s_gate = s_input + SMEM_A;

    const half* gate_w = gate_weights + (size_t)expert_id * hidden_dim * intermediate_dim;

    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> frag_a[2];
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> frag_b[4];
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> frag_c[2][4];

    #pragma unroll
    for (int mi = 0; mi < 2; mi++) {
        #pragma unroll
        for (int ni = 0; ni < 4; ni++) {
            wmma::fill_fragment(frag_c[mi][ni], 0.0f);
        }
    }

    for (int k = 0; k < hidden_dim; k += BLOCK_K) {
        for (int i = tid; i < BLOCK_M * BLOCK_K / 8; i += THREADS) {
            int m = i / (BLOCK_K / 8);
            int kk = (i % (BLOCK_K / 8)) * 8;
            int global_m = block_m + m;
            int global_k = k + kk;

            float4 val = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
            if (global_m < M && global_k + 7 < hidden_dim) {
                int token_id = sorted_token_ids[expert_start + global_m];
                val = load_float4(&input[token_id * hidden_dim + global_k]);
            }
            store_float4(&s_input[m * BLOCK_K + kk], val);
        }

        for (int i = tid; i < BLOCK_K * BLOCK_N / 8; i += THREADS) {
            int kk = i / (BLOCK_N / 8);
            int n = (i % (BLOCK_N / 8)) * 8;
            int global_k = k + kk;
            int global_n = block_n + n;

            float4 val = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
            if (global_k < hidden_dim && global_n + 7 < intermediate_dim) {
                val = load_float4(&gate_w[global_k * intermediate_dim + global_n]);
            }
            store_float4(&s_gate[kk * BLOCK_N + n], val);
        }
        __syncthreads();

        const int warp_row = warp_m * WARP_TILE_M;
        const int warp_col = warp_n * WARP_TILE_N;

        #pragma unroll
        for (int kk = 0; kk < BLOCK_K; kk += WMMA_K) {
            #pragma unroll
            for (int mi = 0; mi < 2; mi++) {
                int a_row = warp_row + mi * WMMA_M;
                wmma::load_matrix_sync(frag_a[mi], &s_input[a_row * BLOCK_K + kk], BLOCK_K);
            }

            #pragma unroll
            for (int ni = 0; ni < 4; ni++) {
                int b_col = warp_col + ni * WMMA_N;
                wmma::load_matrix_sync(frag_b[ni], &s_gate[kk * BLOCK_N + b_col], BLOCK_N);
            }

            #pragma unroll
            for (int mi = 0; mi < 2; mi++) {
                #pragma unroll
                for (int ni = 0; ni < 4; ni++) {
                    wmma::mma_sync(frag_c[mi][ni], frag_a[mi], frag_b[ni], frag_c[mi][ni]);
                }
            }
        }
        __syncthreads();
    }

    float* s_out = reinterpret_cast<float*>(smem);

    const int warp_row = warp_m * WARP_TILE_M;
    const int warp_col = warp_n * WARP_TILE_N;

    #pragma unroll
    for (int mi = 0; mi < 2; mi++) {
        #pragma unroll
        for (int ni = 0; ni < 4; ni++) {
            int out_row = warp_row + mi * WMMA_M;
            int out_col = warp_col + ni * WMMA_N;
            wmma::store_matrix_sync(&s_out[out_row * BLOCK_N + out_col], frag_c[mi][ni], BLOCK_N, wmma::mem_row_major);
        }
    }
    __syncthreads();

    for (int i = tid; i < BLOCK_M * BLOCK_N; i += THREADS) {
        int m = i / BLOCK_N;
        int n = i % BLOCK_N;
        int global_m = block_m + m;
        int global_n = block_n + n;

        if (global_m < M && global_n < intermediate_dim) {
            float val = apply_activation(s_out[m * BLOCK_N + n], activation_type);
            intermediate[(expert_start + global_m) * intermediate_dim + global_n] = __float2half(val);
        }
    }
}

__global__ void __launch_bounds__(gemm_large::THREADS)
nomic_up_gemm_large_kernel(
    const half* __restrict__ intermediate,
    const half* __restrict__ up_weights,
    const int* __restrict__ sorted_token_ids,
    const float* __restrict__ sorted_weights,
    const int* __restrict__ expert_offsets,
    half* __restrict__ output,
    int hidden_dim,
    int intermediate_dim,
    int top_k
) {
    using namespace gemm_large;

    const int expert_id = blockIdx.z;
    const int expert_start = expert_offsets[expert_id];
    const int expert_end = expert_offsets[expert_id + 1];
    const int M = expert_end - expert_start;

    if (M == 0) return;

    const int block_m = blockIdx.y * BLOCK_M;
    const int block_n = blockIdx.x * BLOCK_N;

    if (block_m >= M || block_n >= hidden_dim) return;

    const int tid = threadIdx.x;
    const int warp_id = tid / WARP_SIZE;
    const int warp_m = warp_id / WARPS_N;
    const int warp_n = warp_id % WARPS_N;

    extern __shared__ char smem[];
    // Data area - used for both load phase (s_inter, s_up) and output phase (s_out)
    half* s_inter = reinterpret_cast<half*>(smem);
    half* s_up = s_inter + SMEM_A;
    // Metadata area - placed AFTER output area to avoid overlap when s_out reuses smem
    // s_out needs SMEM_C floats = BLOCK_M * BLOCK_N * sizeof(float) bytes
    constexpr int DATA_AREA_BYTES = SMEM_C * sizeof(float);
    int* s_token_ids = reinterpret_cast<int*>(smem + DATA_AREA_BYTES);
    float* s_routing = reinterpret_cast<float*>(s_token_ids + BLOCK_M);

    const half* up_w = up_weights + (size_t)expert_id * hidden_dim * intermediate_dim;

    for (int i = tid; i < BLOCK_M; i += THREADS) {
        int global_m = block_m + i;
        if (global_m < M) {
            s_token_ids[i] = sorted_token_ids[expert_start + global_m];
            s_routing[i] = sorted_weights[expert_start + global_m];
        } else {
            s_token_ids[i] = 0;
            s_routing[i] = 0.0f;
        }
    }

    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> frag_a[2];
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> frag_b[4];
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> frag_c[2][4];

    #pragma unroll
    for (int mi = 0; mi < 2; mi++) {
        #pragma unroll
        for (int ni = 0; ni < 4; ni++) {
            wmma::fill_fragment(frag_c[mi][ni], 0.0f);
        }
    }
    __syncthreads();

    for (int k = 0; k < intermediate_dim; k += BLOCK_K) {
        for (int i = tid; i < BLOCK_M * BLOCK_K / 8; i += THREADS) {
            int m = i / (BLOCK_K / 8);
            int kk = (i % (BLOCK_K / 8)) * 8;
            int global_m = block_m + m;
            int global_k = k + kk;

            float4 val = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
            if (global_m < M && global_k + 7 < intermediate_dim) {
                val = load_float4(&intermediate[(expert_start + global_m) * intermediate_dim + global_k]);
            }
            store_float4(&s_inter[m * BLOCK_K + kk], val);
        }

        for (int i = tid; i < BLOCK_K * BLOCK_N; i += THREADS) {
            int kk = i / BLOCK_N;
            int n = i % BLOCK_N;
            int global_k = k + kk;
            int global_n = block_n + n;

            half val = __float2half(0.0f);
            if (global_k < intermediate_dim && global_n < hidden_dim) {
                val = up_w[global_n * intermediate_dim + global_k];
            }
            s_up[kk * BLOCK_N + n] = val;
        }
        __syncthreads();

        const int warp_row = warp_m * WARP_TILE_M;
        const int warp_col = warp_n * WARP_TILE_N;

        #pragma unroll
        for (int kk = 0; kk < BLOCK_K; kk += WMMA_K) {
            #pragma unroll
            for (int mi = 0; mi < 2; mi++) {
                int a_row = warp_row + mi * WMMA_M;
                wmma::load_matrix_sync(frag_a[mi], &s_inter[a_row * BLOCK_K + kk], BLOCK_K);
            }

            #pragma unroll
            for (int ni = 0; ni < 4; ni++) {
                int b_col = warp_col + ni * WMMA_N;
                wmma::load_matrix_sync(frag_b[ni], &s_up[kk * BLOCK_N + b_col], BLOCK_N);
            }

            #pragma unroll
            for (int mi = 0; mi < 2; mi++) {
                #pragma unroll
                for (int ni = 0; ni < 4; ni++) {
                    wmma::mma_sync(frag_c[mi][ni], frag_a[mi], frag_b[ni], frag_c[mi][ni]);
                }
            }
        }
        __syncthreads();
    }

    float* s_out = reinterpret_cast<float*>(smem);

    const int warp_row = warp_m * WARP_TILE_M;
    const int warp_col = warp_n * WARP_TILE_N;

    #pragma unroll
    for (int mi = 0; mi < 2; mi++) {
        #pragma unroll
        for (int ni = 0; ni < 4; ni++) {
            int out_row = warp_row + mi * WMMA_M;
            int out_col = warp_col + ni * WMMA_N;
            wmma::store_matrix_sync(&s_out[out_row * BLOCK_N + out_col], frag_c[mi][ni], BLOCK_N, wmma::mem_row_major);
        }
    }
    __syncthreads();

    for (int i = tid; i < BLOCK_M * BLOCK_N; i += THREADS) {
        int m = i / BLOCK_N;
        int n = i % BLOCK_N;
        int global_m = block_m + m;
        int global_n = block_n + n;

        if (global_m < M && global_n < hidden_dim) {
            int token_id = s_token_ids[m];
            float weight = s_routing[m];
            float val = s_out[m * BLOCK_N + n] * weight;
            if (top_k == 1) {
                output[token_id * hidden_dim + global_n] = __float2half(val);
            } else {
                atomic_add_half(&output[token_id * hidden_dim + global_n], __float2half(val));
            }
        }
    }
}

#ifdef SM80_OR_HIGHER
__global__ void __launch_bounds__(gemm_async::THREADS)
nomic_gate_gemm_async_kernel(
    const half* __restrict__ input,
    const half* __restrict__ gate_weights,
    const int* __restrict__ sorted_token_ids,
    const int* __restrict__ expert_offsets,
    half* __restrict__ intermediate,
    int hidden_dim,
    int intermediate_dim,
    int activation_type
) {
    using namespace gemm_async;

    const int expert_id = blockIdx.z;
    const int expert_start = expert_offsets[expert_id];
    const int expert_end = expert_offsets[expert_id + 1];
    const int M = expert_end - expert_start;

    if (M == 0) return;

    const int block_m = blockIdx.y * BLOCK_M;
    const int block_n = blockIdx.x * BLOCK_N;

    if (block_m >= M || block_n >= intermediate_dim) return;

    const int tid = threadIdx.x;
    const int warp_id = tid / WARP_SIZE;
    const int warp_m = warp_id / WARPS_N;
    const int warp_n = warp_id % WARPS_N;

    extern __shared__ char smem[];
    constexpr int SMEM_A = BLOCK_M * BLOCK_K;
    constexpr int SMEM_B = BLOCK_K * BLOCK_N;

    half* s_input[STAGES];
    half* s_gate[STAGES];

    s_input[0] = reinterpret_cast<half*>(smem);
    s_gate[0] = s_input[0] + SMEM_A;
    s_input[1] = s_gate[0] + SMEM_B;
    s_gate[1] = s_input[1] + SMEM_A;

    const half* gate_w = gate_weights + (size_t)expert_id * hidden_dim * intermediate_dim;

    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> frag_a;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> frag_b[2];
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> frag_c[2];

    wmma::fill_fragment(frag_c[0], 0.0f);
    wmma::fill_fragment(frag_c[1], 0.0f);

    auto load_tile_async = [&](int stage, int k) {
        for (int i = tid; i < SMEM_A / 8; i += THREADS) {
            int m = i / (BLOCK_K / 8);
            int kk = (i % (BLOCK_K / 8)) * 8;
            int global_m = block_m + m;
            int global_k = k + kk;

            if (global_m < M && global_k < hidden_dim) {
                int token_id = sorted_token_ids[expert_start + global_m];
                cp_async_cg(&s_input[stage][m * BLOCK_K + kk], &input[token_id * hidden_dim + global_k]);
            }
        }

        for (int i = tid; i < SMEM_B / 8; i += THREADS) {
            int kk = i / (BLOCK_N / 8);
            int n = (i % (BLOCK_N / 8)) * 8;
            int global_k = k + kk;
            int global_n = block_n + n;

            if (global_k < hidden_dim && global_n < intermediate_dim) {
                cp_async_cg(&s_gate[stage][kk * BLOCK_N + n], &gate_w[global_k * intermediate_dim + global_n]);
            }
        }
        cp_async_commit();
    };

    int num_k_tiles = (hidden_dim + BLOCK_K - 1) / BLOCK_K;

    load_tile_async(0, 0);
    if (num_k_tiles > 1) {
        load_tile_async(1, BLOCK_K);
    }

    for (int k_tile = 0; k_tile < num_k_tiles; k_tile++) {
        int stage = k_tile % STAGES;
        int next_stage = (k_tile + 1) % STAGES;

        cp_async_wait<STAGES - 1>();
        __syncthreads();

        if (k_tile + STAGES < num_k_tiles) {
            load_tile_async(next_stage, (k_tile + STAGES) * BLOCK_K);
        }

        const int warp_row = warp_m * WMMA_M;
        const int warp_col = warp_n * WMMA_N * 2;

        #pragma unroll
        for (int kk = 0; kk < BLOCK_K; kk += WMMA_K) {
            wmma::load_matrix_sync(frag_a, &s_input[stage][warp_row * BLOCK_K + kk], BLOCK_K);

            #pragma unroll
            for (int ni = 0; ni < 2; ni++) {
                int b_col = warp_col + ni * WMMA_N;
                wmma::load_matrix_sync(frag_b[ni], &s_gate[stage][kk * BLOCK_N + b_col], BLOCK_N);
                wmma::mma_sync(frag_c[ni], frag_a, frag_b[ni], frag_c[ni]);
            }
        }
        __syncthreads();
    }

    float* s_out = reinterpret_cast<float*>(smem);

    const int warp_row = warp_m * WMMA_M;
    const int warp_col = warp_n * WMMA_N * 2;

    #pragma unroll
    for (int ni = 0; ni < 2; ni++) {
        int out_col = warp_col + ni * WMMA_N;
        wmma::store_matrix_sync(&s_out[warp_row * BLOCK_N + out_col], frag_c[ni], BLOCK_N, wmma::mem_row_major);
    }
    __syncthreads();

    for (int i = tid; i < BLOCK_M * BLOCK_N; i += THREADS) {
        int m = i / BLOCK_N;
        int n = i % BLOCK_N;
        int global_m = block_m + m;
        int global_n = block_n + n;

        if (global_m < M && global_n < intermediate_dim) {
            float val = apply_activation(s_out[m * BLOCK_N + n], activation_type);
            intermediate[(expert_start + global_m) * intermediate_dim + global_n] = __float2half(val);
        }
    }
}

__global__ void __launch_bounds__(gemm_async::THREADS)
nomic_up_gemm_async_kernel(
    const half* __restrict__ intermediate,
    const half* __restrict__ up_weights,
    const int* __restrict__ sorted_token_ids,
    const float* __restrict__ sorted_weights,
    const int* __restrict__ expert_offsets,
    half* __restrict__ output,
    int hidden_dim,
    int intermediate_dim,
    int top_k
) {
    using namespace gemm_async;

    const int expert_id = blockIdx.z;
    const int expert_start = expert_offsets[expert_id];
    const int expert_end = expert_offsets[expert_id + 1];
    const int M = expert_end - expert_start;

    if (M == 0) return;

    const int block_m = blockIdx.y * BLOCK_M;
    const int block_n = blockIdx.x * BLOCK_N;

    if (block_m >= M || block_n >= hidden_dim) return;

    const int tid = threadIdx.x;
    const int warp_id = tid / WARP_SIZE;
    const int warp_m = warp_id / WARPS_N;
    const int warp_n = warp_id % WARPS_N;

    extern __shared__ char smem[];
    constexpr int SMEM_A = BLOCK_M * BLOCK_K;
    constexpr int SMEM_B = BLOCK_K * BLOCK_N;
    constexpr int SMEM_OUT = BLOCK_M * BLOCK_N;  // in floats

    half* s_inter[STAGES];
    half* s_up[STAGES];

    s_inter[0] = reinterpret_cast<half*>(smem);
    s_up[0] = s_inter[0] + SMEM_A;
    s_inter[1] = s_up[0] + SMEM_B;
    s_up[1] = s_inter[1] + SMEM_A;

    // Metadata area - placed AFTER max(staging_area, output_area) to avoid overlap
    // staging_area = STAGES * (SMEM_A + SMEM_B) halfs, output_area = SMEM_OUT floats
    constexpr int STAGING_BYTES = STAGES * (SMEM_A + SMEM_B) * sizeof(half);
    constexpr int OUTPUT_BYTES = SMEM_OUT * sizeof(float);
    constexpr int DATA_AREA_BYTES = (STAGING_BYTES > OUTPUT_BYTES) ? STAGING_BYTES : OUTPUT_BYTES;
    int* s_token_ids = reinterpret_cast<int*>(smem + DATA_AREA_BYTES);
    float* s_routing = reinterpret_cast<float*>(s_token_ids + BLOCK_M);

    const half* up_w = up_weights + (size_t)expert_id * hidden_dim * intermediate_dim;

    for (int i = tid; i < BLOCK_M; i += THREADS) {
        int global_m = block_m + i;
        if (global_m < M) {
            s_token_ids[i] = sorted_token_ids[expert_start + global_m];
            s_routing[i] = sorted_weights[expert_start + global_m];
        } else {
            s_token_ids[i] = 0;
            s_routing[i] = 0.0f;
        }
    }

    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> frag_a;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> frag_b[2];
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> frag_c[2];

    wmma::fill_fragment(frag_c[0], 0.0f);
    wmma::fill_fragment(frag_c[1], 0.0f);
    __syncthreads();

    auto load_tile_async = [&](int stage, int k) {
        for (int i = tid; i < SMEM_A / 8; i += THREADS) {
            int m = i / (BLOCK_K / 8);
            int kk = (i % (BLOCK_K / 8)) * 8;
            int global_m = block_m + m;
            int global_k = k + kk;

            if (global_m < M && global_k < intermediate_dim) {
                cp_async_cg(&s_inter[stage][m * BLOCK_K + kk],
                           &intermediate[(expert_start + global_m) * intermediate_dim + global_k]);
            }
        }

        for (int i = tid; i < SMEM_B; i += THREADS) {
            int kk = i / BLOCK_N;
            int n = i % BLOCK_N;
            int global_k = k + kk;
            int global_n = block_n + n;

            if (global_k < intermediate_dim && global_n < hidden_dim) {
                s_up[stage][kk * BLOCK_N + n] = up_w[global_n * intermediate_dim + global_k];
            }
        }
        cp_async_commit();
    };

    int num_k_tiles = (intermediate_dim + BLOCK_K - 1) / BLOCK_K;

    load_tile_async(0, 0);
    if (num_k_tiles > 1) {
        load_tile_async(1, BLOCK_K);
    }

    for (int k_tile = 0; k_tile < num_k_tiles; k_tile++) {
        int stage = k_tile % STAGES;
        int next_stage = (k_tile + 1) % STAGES;

        cp_async_wait<STAGES - 1>();
        __syncthreads();

        if (k_tile + STAGES < num_k_tiles) {
            load_tile_async(next_stage, (k_tile + STAGES) * BLOCK_K);
        }

        const int warp_row = warp_m * WMMA_M;
        const int warp_col = warp_n * WMMA_N * 2;

        #pragma unroll
        for (int kk = 0; kk < BLOCK_K; kk += WMMA_K) {
            wmma::load_matrix_sync(frag_a, &s_inter[stage][warp_row * BLOCK_K + kk], BLOCK_K);

            #pragma unroll
            for (int ni = 0; ni < 2; ni++) {
                int b_col = warp_col + ni * WMMA_N;
                wmma::load_matrix_sync(frag_b[ni], &s_up[stage][kk * BLOCK_N + b_col], BLOCK_N);
                wmma::mma_sync(frag_c[ni], frag_a, frag_b[ni], frag_c[ni]);
            }
        }
        __syncthreads();
    }

    float* s_out = reinterpret_cast<float*>(smem);

    const int warp_row = warp_m * WMMA_M;
    const int warp_col = warp_n * WMMA_N * 2;

    #pragma unroll
    for (int ni = 0; ni < 2; ni++) {
        int out_col = warp_col + ni * WMMA_N;
        wmma::store_matrix_sync(&s_out[warp_row * BLOCK_N + out_col], frag_c[ni], BLOCK_N, wmma::mem_row_major);
    }
    __syncthreads();

    for (int i = tid; i < BLOCK_M * BLOCK_N; i += THREADS) {
        int m = i / BLOCK_N;
        int n = i % BLOCK_N;
        int global_m = block_m + m;
        int global_n = block_n + n;

        if (global_m < M && global_n < hidden_dim) {
            int token_id = s_token_ids[m];
            float weight = s_routing[m];
            float val = s_out[m * BLOCK_N + n] * weight;
            if (top_k == 1) {
                output[token_id * hidden_dim + global_n] = __float2half(val);
            } else {
                atomic_add_half(&output[token_id * hidden_dim + global_n], __float2half(val));
            }
        }
    }
}
#endif

#ifndef NO_BF16_KERNEL
__global__ void __launch_bounds__(gemm_small::THREADS)
nomic_gate_gemm_small_bf16_kernel(
    const __nv_bfloat16* __restrict__ input,
    const __nv_bfloat16* __restrict__ gate_weights,
    const int* __restrict__ sorted_token_ids,
    const int* __restrict__ expert_offsets,
    __nv_bfloat16* __restrict__ intermediate,
    int hidden_dim,
    int intermediate_dim,
    int activation_type
) {
    using namespace gemm_small;

    const int expert_id = blockIdx.z;
    const int expert_start = expert_offsets[expert_id];
    const int expert_end = expert_offsets[expert_id + 1];
    const int M = expert_end - expert_start;

    if (M == 0) return;

    const int block_m = blockIdx.y * BLOCK_M;
    const int block_n = blockIdx.x * BLOCK_N;

    if (block_m >= M || block_n >= intermediate_dim) return;

    const int tid = threadIdx.x;
    const int warp_id = tid / WARP_SIZE;
    const int warp_m = warp_id / WARPS_N;
    const int warp_n = warp_id % WARPS_N;

    extern __shared__ char smem[];
    __nv_bfloat16* s_input = reinterpret_cast<__nv_bfloat16*>(smem);
    __nv_bfloat16* s_gate = s_input + SMEM_A;

    const __nv_bfloat16* gate_w = gate_weights + (size_t)expert_id * hidden_dim * intermediate_dim;

    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, __nv_bfloat16, wmma::row_major> frag_a;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, __nv_bfloat16, wmma::row_major> frag_b[2];
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> frag_c[2];

    wmma::fill_fragment(frag_c[0], 0.0f);
    wmma::fill_fragment(frag_c[1], 0.0f);

    for (int k = 0; k < hidden_dim; k += BLOCK_K) {
        for (int i = tid; i < BLOCK_M * BLOCK_K / 8; i += THREADS) {
            int m = i / (BLOCK_K / 8);
            int kk = (i % (BLOCK_K / 8)) * 8;
            int global_m = block_m + m;
            int global_k = k + kk;

            float4 val = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
            if (global_m < M && global_k + 7 < hidden_dim) {
                int token_id = sorted_token_ids[expert_start + global_m];
                val = load_float4_bf16(&input[token_id * hidden_dim + global_k]);
            } else if (global_m < M) {
                __nv_bfloat16* ptr = reinterpret_cast<__nv_bfloat16*>(&val);
                int token_id = sorted_token_ids[expert_start + global_m];
                for (int j = 0; j < 8 && global_k + j < hidden_dim; j++) {
                    ptr[j] = input[token_id * hidden_dim + global_k + j];
                }
            }
            store_float4_bf16(&s_input[m * BLOCK_K + kk], val);
        }

        for (int i = tid; i < BLOCK_K * BLOCK_N / 8; i += THREADS) {
            int kk = i / (BLOCK_N / 8);
            int n = (i % (BLOCK_N / 8)) * 8;
            int global_k = k + kk;
            int global_n = block_n + n;

            float4 val = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
            if (global_k < hidden_dim && global_n + 7 < intermediate_dim) {
                val = load_float4_bf16(&gate_w[global_k * intermediate_dim + global_n]);
            }
            store_float4_bf16(&s_gate[kk * BLOCK_N + n], val);
        }
        __syncthreads();

        const int warp_row = warp_m * WMMA_M;
        const int warp_col = warp_n * WMMA_N * 2;

        #pragma unroll
        for (int kk = 0; kk < BLOCK_K; kk += WMMA_K) {
            wmma::load_matrix_sync(frag_a, &s_input[warp_row * BLOCK_K + kk], BLOCK_K);

            #pragma unroll
            for (int ni = 0; ni < 2; ni++) {
                int b_col = warp_col + ni * WMMA_N;
                wmma::load_matrix_sync(frag_b[ni], &s_gate[kk * BLOCK_N + b_col], BLOCK_N);
                wmma::mma_sync(frag_c[ni], frag_a, frag_b[ni], frag_c[ni]);
            }
        }
        __syncthreads();
    }

    float* s_out = reinterpret_cast<float*>(smem);

    const int warp_row = warp_m * WMMA_M;
    const int warp_col = warp_n * WMMA_N * 2;

    #pragma unroll
    for (int ni = 0; ni < 2; ni++) {
        int out_col = warp_col + ni * WMMA_N;
        wmma::store_matrix_sync(&s_out[warp_row * BLOCK_N + out_col], frag_c[ni], BLOCK_N, wmma::mem_row_major);
    }
    __syncthreads();

    for (int i = tid; i < BLOCK_M * BLOCK_N; i += THREADS) {
        int m = i / BLOCK_N;
        int n = i % BLOCK_N;
        int global_m = block_m + m;
        int global_n = block_n + n;

        if (global_m < M && global_n < intermediate_dim) {
            float val = apply_activation(s_out[m * BLOCK_N + n], activation_type);
            intermediate[(expert_start + global_m) * intermediate_dim + global_n] = __float2bfloat16(val);
        }
    }
}

__global__ void __launch_bounds__(gemm_small::THREADS)
nomic_up_gemm_small_bf16_kernel(
    const __nv_bfloat16* __restrict__ intermediate,
    const __nv_bfloat16* __restrict__ up_weights,
    const int* __restrict__ sorted_token_ids,
    const float* __restrict__ sorted_weights,
    const int* __restrict__ expert_offsets,
    __nv_bfloat16* __restrict__ output,
    int hidden_dim,
    int intermediate_dim,
    int top_k
) {
    using namespace gemm_small;

    const int expert_id = blockIdx.z;
    const int expert_start = expert_offsets[expert_id];
    const int expert_end = expert_offsets[expert_id + 1];
    const int M = expert_end - expert_start;

    if (M == 0) return;

    const int block_m = blockIdx.y * BLOCK_M;
    const int block_n = blockIdx.x * BLOCK_N;

    if (block_m >= M || block_n >= hidden_dim) return;

    const int tid = threadIdx.x;
    const int warp_id = tid / WARP_SIZE;
    const int warp_m = warp_id / WARPS_N;
    const int warp_n = warp_id % WARPS_N;

    extern __shared__ char smem[];
    // Data area - used for both load phase (s_inter, s_up) and output phase (s_out)
    __nv_bfloat16* s_inter = reinterpret_cast<__nv_bfloat16*>(smem);
    __nv_bfloat16* s_up = s_inter + SMEM_A;
    // Metadata area - placed AFTER output area to avoid overlap when s_out reuses smem
    // s_out needs SMEM_C floats = BLOCK_M * BLOCK_N * sizeof(float) bytes
    constexpr int DATA_AREA_BYTES = SMEM_C * sizeof(float);
    int* s_token_ids = reinterpret_cast<int*>(smem + DATA_AREA_BYTES);
    float* s_routing = reinterpret_cast<float*>(s_token_ids + BLOCK_M);

    const __nv_bfloat16* up_w = up_weights + (size_t)expert_id * hidden_dim * intermediate_dim;

    for (int i = tid; i < BLOCK_M; i += THREADS) {
        int global_m = block_m + i;
        if (global_m < M) {
            s_token_ids[i] = sorted_token_ids[expert_start + global_m];
            s_routing[i] = sorted_weights[expert_start + global_m];
        } else {
            s_token_ids[i] = 0;
            s_routing[i] = 0.0f;
        }
    }

    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, __nv_bfloat16, wmma::row_major> frag_a;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, __nv_bfloat16, wmma::row_major> frag_b[2];
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> frag_c[2];

    wmma::fill_fragment(frag_c[0], 0.0f);
    wmma::fill_fragment(frag_c[1], 0.0f);
    __syncthreads();

    for (int k = 0; k < intermediate_dim; k += BLOCK_K) {
        for (int i = tid; i < BLOCK_M * BLOCK_K / 8; i += THREADS) {
            int m = i / (BLOCK_K / 8);
            int kk = (i % (BLOCK_K / 8)) * 8;
            int global_m = block_m + m;
            int global_k = k + kk;

            float4 val = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
            if (global_m < M && global_k + 7 < intermediate_dim) {
                val = load_float4_bf16(&intermediate[(expert_start + global_m) * intermediate_dim + global_k]);
            }
            store_float4_bf16(&s_inter[m * BLOCK_K + kk], val);
        }

        for (int i = tid; i < BLOCK_K * BLOCK_N; i += THREADS) {
            int kk = i / BLOCK_N;
            int n = i % BLOCK_N;
            int global_k = k + kk;
            int global_n = block_n + n;

            __nv_bfloat16 val = __float2bfloat16(0.0f);
            if (global_k < intermediate_dim && global_n < hidden_dim) {
                val = up_w[global_n * intermediate_dim + global_k];
            }
            s_up[kk * BLOCK_N + n] = val;
        }
        __syncthreads();

        const int warp_row = warp_m * WMMA_M;
        const int warp_col = warp_n * WMMA_N * 2;

        #pragma unroll
        for (int kk = 0; kk < BLOCK_K; kk += WMMA_K) {
            wmma::load_matrix_sync(frag_a, &s_inter[warp_row * BLOCK_K + kk], BLOCK_K);

            #pragma unroll
            for (int ni = 0; ni < 2; ni++) {
                int b_col = warp_col + ni * WMMA_N;
                wmma::load_matrix_sync(frag_b[ni], &s_up[kk * BLOCK_N + b_col], BLOCK_N);
                wmma::mma_sync(frag_c[ni], frag_a, frag_b[ni], frag_c[ni]);
            }
        }
        __syncthreads();
    }

    float* s_out = reinterpret_cast<float*>(smem);

    const int warp_row = warp_m * WMMA_M;
    const int warp_col = warp_n * WMMA_N * 2;

    #pragma unroll
    for (int ni = 0; ni < 2; ni++) {
        int out_col = warp_col + ni * WMMA_N;
        wmma::store_matrix_sync(&s_out[warp_row * BLOCK_N + out_col], frag_c[ni], BLOCK_N, wmma::mem_row_major);
    }
    __syncthreads();

    for (int i = tid; i < BLOCK_M * BLOCK_N; i += THREADS) {
        int m = i / BLOCK_N;
        int n = i % BLOCK_N;
        int global_m = block_m + m;
        int global_n = block_n + n;

        if (global_m < M && global_n < hidden_dim) {
            int token_id = s_token_ids[m];
            float weight = s_routing[m];
            float val = s_out[m * BLOCK_N + n] * weight;
            if (top_k == 1) {
                output[token_id * hidden_dim + global_n] = __float2bfloat16(val);
            } else {
                atomic_add_bf16(&output[token_id * hidden_dim + global_n], __float2bfloat16(val));
            }
        }
    }
}

__global__ void __launch_bounds__(gemm_large::THREADS)
nomic_gate_gemm_large_bf16_kernel(
    const __nv_bfloat16* __restrict__ input,
    const __nv_bfloat16* __restrict__ gate_weights,
    const int* __restrict__ sorted_token_ids,
    const int* __restrict__ expert_offsets,
    __nv_bfloat16* __restrict__ intermediate,
    int hidden_dim,
    int intermediate_dim,
    int activation_type
) {
    using namespace gemm_large;

    const int expert_id = blockIdx.z;
    const int expert_start = expert_offsets[expert_id];
    const int expert_end = expert_offsets[expert_id + 1];
    const int M = expert_end - expert_start;

    if (M == 0) return;

    const int block_m = blockIdx.y * BLOCK_M;
    const int block_n = blockIdx.x * BLOCK_N;

    if (block_m >= M || block_n >= intermediate_dim) return;

    const int tid = threadIdx.x;
    const int warp_id = tid / WARP_SIZE;
    const int warp_m = warp_id / WARPS_N;
    const int warp_n = warp_id % WARPS_N;

    extern __shared__ char smem[];
    __nv_bfloat16* s_input = reinterpret_cast<__nv_bfloat16*>(smem);
    __nv_bfloat16* s_gate = s_input + SMEM_A;

    const __nv_bfloat16* gate_w = gate_weights + (size_t)expert_id * hidden_dim * intermediate_dim;

    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, __nv_bfloat16, wmma::row_major> frag_a[2];
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, __nv_bfloat16, wmma::row_major> frag_b[4];
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> frag_c[2][4];

    #pragma unroll
    for (int mi = 0; mi < 2; mi++) {
        #pragma unroll
        for (int ni = 0; ni < 4; ni++) {
            wmma::fill_fragment(frag_c[mi][ni], 0.0f);
        }
    }

    for (int k = 0; k < hidden_dim; k += BLOCK_K) {
        for (int i = tid; i < BLOCK_M * BLOCK_K / 8; i += THREADS) {
            int m = i / (BLOCK_K / 8);
            int kk = (i % (BLOCK_K / 8)) * 8;
            int global_m = block_m + m;
            int global_k = k + kk;

            float4 val = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
            if (global_m < M && global_k + 7 < hidden_dim) {
                int token_id = sorted_token_ids[expert_start + global_m];
                val = load_float4_bf16(&input[token_id * hidden_dim + global_k]);
            }
            store_float4_bf16(&s_input[m * BLOCK_K + kk], val);
        }

        for (int i = tid; i < BLOCK_K * BLOCK_N / 8; i += THREADS) {
            int kk = i / (BLOCK_N / 8);
            int n = (i % (BLOCK_N / 8)) * 8;
            int global_k = k + kk;
            int global_n = block_n + n;

            float4 val = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
            if (global_k < hidden_dim && global_n + 7 < intermediate_dim) {
                val = load_float4_bf16(&gate_w[global_k * intermediate_dim + global_n]);
            }
            store_float4_bf16(&s_gate[kk * BLOCK_N + n], val);
        }
        __syncthreads();

        const int warp_row = warp_m * WARP_TILE_M;
        const int warp_col = warp_n * WARP_TILE_N;

        #pragma unroll
        for (int kk = 0; kk < BLOCK_K; kk += WMMA_K) {
            #pragma unroll
            for (int mi = 0; mi < 2; mi++) {
                int a_row = warp_row + mi * WMMA_M;
                wmma::load_matrix_sync(frag_a[mi], &s_input[a_row * BLOCK_K + kk], BLOCK_K);
            }

            #pragma unroll
            for (int ni = 0; ni < 4; ni++) {
                int b_col = warp_col + ni * WMMA_N;
                wmma::load_matrix_sync(frag_b[ni], &s_gate[kk * BLOCK_N + b_col], BLOCK_N);
            }

            #pragma unroll
            for (int mi = 0; mi < 2; mi++) {
                #pragma unroll
                for (int ni = 0; ni < 4; ni++) {
                    wmma::mma_sync(frag_c[mi][ni], frag_a[mi], frag_b[ni], frag_c[mi][ni]);
                }
            }
        }
        __syncthreads();
    }

    float* s_out = reinterpret_cast<float*>(smem);

    const int warp_row = warp_m * WARP_TILE_M;
    const int warp_col = warp_n * WARP_TILE_N;

    #pragma unroll
    for (int mi = 0; mi < 2; mi++) {
        #pragma unroll
        for (int ni = 0; ni < 4; ni++) {
            int out_row = warp_row + mi * WMMA_M;
            int out_col = warp_col + ni * WMMA_N;
            wmma::store_matrix_sync(&s_out[out_row * BLOCK_N + out_col], frag_c[mi][ni], BLOCK_N, wmma::mem_row_major);
        }
    }
    __syncthreads();

    for (int i = tid; i < BLOCK_M * BLOCK_N; i += THREADS) {
        int m = i / BLOCK_N;
        int n = i % BLOCK_N;
        int global_m = block_m + m;
        int global_n = block_n + n;

        if (global_m < M && global_n < intermediate_dim) {
            float val = apply_activation(s_out[m * BLOCK_N + n], activation_type);
            intermediate[(expert_start + global_m) * intermediate_dim + global_n] = __float2bfloat16(val);
        }
    }
}

__global__ void __launch_bounds__(gemm_large::THREADS)
nomic_up_gemm_large_bf16_kernel(
    const __nv_bfloat16* __restrict__ intermediate,
    const __nv_bfloat16* __restrict__ up_weights,
    const int* __restrict__ sorted_token_ids,
    const float* __restrict__ sorted_weights,
    const int* __restrict__ expert_offsets,
    __nv_bfloat16* __restrict__ output,
    int hidden_dim,
    int intermediate_dim,
    int top_k
) {
    using namespace gemm_large;

    const int expert_id = blockIdx.z;
    const int expert_start = expert_offsets[expert_id];
    const int expert_end = expert_offsets[expert_id + 1];
    const int M = expert_end - expert_start;

    if (M == 0) return;

    const int block_m = blockIdx.y * BLOCK_M;
    const int block_n = blockIdx.x * BLOCK_N;

    if (block_m >= M || block_n >= hidden_dim) return;

    const int tid = threadIdx.x;
    const int warp_id = tid / WARP_SIZE;
    const int warp_m = warp_id / WARPS_N;
    const int warp_n = warp_id % WARPS_N;

    extern __shared__ char smem[];
    // Data area - used for both load phase (s_inter, s_up) and output phase (s_out)
    __nv_bfloat16* s_inter = reinterpret_cast<__nv_bfloat16*>(smem);
    __nv_bfloat16* s_up = s_inter + SMEM_A;
    // Metadata area - placed AFTER output area to avoid overlap when s_out reuses smem
    // s_out needs SMEM_C floats = BLOCK_M * BLOCK_N * sizeof(float) bytes
    constexpr int DATA_AREA_BYTES = SMEM_C * sizeof(float);
    int* s_token_ids = reinterpret_cast<int*>(smem + DATA_AREA_BYTES);
    float* s_routing = reinterpret_cast<float*>(s_token_ids + BLOCK_M);

    const __nv_bfloat16* up_w = up_weights + (size_t)expert_id * hidden_dim * intermediate_dim;

    for (int i = tid; i < BLOCK_M; i += THREADS) {
        int global_m = block_m + i;
        if (global_m < M) {
            s_token_ids[i] = sorted_token_ids[expert_start + global_m];
            s_routing[i] = sorted_weights[expert_start + global_m];
        } else {
            s_token_ids[i] = 0;
            s_routing[i] = 0.0f;
        }
    }

    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, __nv_bfloat16, wmma::row_major> frag_a[2];
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, __nv_bfloat16, wmma::row_major> frag_b[4];
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> frag_c[2][4];

    #pragma unroll
    for (int mi = 0; mi < 2; mi++) {
        #pragma unroll
        for (int ni = 0; ni < 4; ni++) {
            wmma::fill_fragment(frag_c[mi][ni], 0.0f);
        }
    }
    __syncthreads();

    for (int k = 0; k < intermediate_dim; k += BLOCK_K) {
        for (int i = tid; i < BLOCK_M * BLOCK_K / 8; i += THREADS) {
            int m = i / (BLOCK_K / 8);
            int kk = (i % (BLOCK_K / 8)) * 8;
            int global_m = block_m + m;
            int global_k = k + kk;

            float4 val = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
            if (global_m < M && global_k + 7 < intermediate_dim) {
                val = load_float4_bf16(&intermediate[(expert_start + global_m) * intermediate_dim + global_k]);
            }
            store_float4_bf16(&s_inter[m * BLOCK_K + kk], val);
        }

        for (int i = tid; i < BLOCK_K * BLOCK_N; i += THREADS) {
            int kk = i / BLOCK_N;
            int n = i % BLOCK_N;
            int global_k = k + kk;
            int global_n = block_n + n;

            __nv_bfloat16 val = __float2bfloat16(0.0f);
            if (global_k < intermediate_dim && global_n < hidden_dim) {
                val = up_w[global_n * intermediate_dim + global_k];
            }
            s_up[kk * BLOCK_N + n] = val;
        }
        __syncthreads();

        const int warp_row = warp_m * WARP_TILE_M;
        const int warp_col = warp_n * WARP_TILE_N;

        #pragma unroll
        for (int kk = 0; kk < BLOCK_K; kk += WMMA_K) {
            #pragma unroll
            for (int mi = 0; mi < 2; mi++) {
                int a_row = warp_row + mi * WMMA_M;
                wmma::load_matrix_sync(frag_a[mi], &s_inter[a_row * BLOCK_K + kk], BLOCK_K);
            }

            #pragma unroll
            for (int ni = 0; ni < 4; ni++) {
                int b_col = warp_col + ni * WMMA_N;
                wmma::load_matrix_sync(frag_b[ni], &s_up[kk * BLOCK_N + b_col], BLOCK_N);
            }

            #pragma unroll
            for (int mi = 0; mi < 2; mi++) {
                #pragma unroll
                for (int ni = 0; ni < 4; ni++) {
                    wmma::mma_sync(frag_c[mi][ni], frag_a[mi], frag_b[ni], frag_c[mi][ni]);
                }
            }
        }
        __syncthreads();
    }

    float* s_out = reinterpret_cast<float*>(smem);

    const int warp_row = warp_m * WARP_TILE_M;
    const int warp_col = warp_n * WARP_TILE_N;

    #pragma unroll
    for (int mi = 0; mi < 2; mi++) {
        #pragma unroll
        for (int ni = 0; ni < 4; ni++) {
            int out_row = warp_row + mi * WMMA_M;
            int out_col = warp_col + ni * WMMA_N;
            wmma::store_matrix_sync(&s_out[out_row * BLOCK_N + out_col], frag_c[mi][ni], BLOCK_N, wmma::mem_row_major);
        }
    }
    __syncthreads();

    for (int i = tid; i < BLOCK_M * BLOCK_N; i += THREADS) {
        int m = i / BLOCK_N;
        int n = i % BLOCK_N;
        int global_m = block_m + m;
        int global_n = block_n + n;

        if (global_m < M && global_n < hidden_dim) {
            int token_id = s_token_ids[m];
            float weight = s_routing[m];
            float val = s_out[m * BLOCK_N + n] * weight;
            if (top_k == 1) {
                output[token_id * hidden_dim + global_n] = __float2bfloat16(val);
            } else {
                atomic_add_bf16(&output[token_id * hidden_dim + global_n], __float2bfloat16(val));
            }
        }
    }
}

extern "C" void nomic_moe_forward_bf16(
    const __nv_bfloat16* input,
    const __nv_bfloat16* gate_weights,
    const __nv_bfloat16* up_weights,
    const int* sorted_token_ids,
    const float* sorted_weights,
    const int* expert_offsets,
    __nv_bfloat16* intermediate,
    __nv_bfloat16* output,
    int num_tokens,
    int hidden_dim,
    int intermediate_dim,
    int num_experts,
    int max_tokens_per_expert,
    int top_k,
    int activation_type,
    cudaStream_t stream
) {
    if (max_tokens_per_expert <= thresholds::SMALL_GEMM_MAX_TOKENS) {
        using namespace gemm_small;
        int m_tiles = (max_tokens_per_expert + BLOCK_M - 1) / BLOCK_M;
        int n_tiles_inter = (intermediate_dim + BLOCK_N - 1) / BLOCK_N;
        int n_tiles_out = (hidden_dim + BLOCK_N - 1) / BLOCK_N;

        size_t gate_smem = max((size_t)(SMEM_A + SMEM_B) * sizeof(__nv_bfloat16),
                               (size_t)SMEM_C * sizeof(float));

        // up kernel: metadata (token_ids, routing) placed AFTER max(load_data, output_data)
        size_t up_smem = max((size_t)(SMEM_A + SMEM_B) * sizeof(__nv_bfloat16), (size_t)SMEM_C * sizeof(float));
        up_smem += BLOCK_M * (sizeof(int) + sizeof(float));

        dim3 grid_gate(n_tiles_inter, m_tiles, num_experts);
        nomic_gate_gemm_small_bf16_kernel<<<grid_gate, THREADS, gate_smem, stream>>>(
            input, gate_weights, sorted_token_ids, expert_offsets,
            intermediate, hidden_dim, intermediate_dim, activation_type
        );

        dim3 grid_up(n_tiles_out, m_tiles, num_experts);
        nomic_up_gemm_small_bf16_kernel<<<grid_up, THREADS, up_smem, stream>>>(
            intermediate, up_weights, sorted_token_ids, sorted_weights, expert_offsets,
            output, hidden_dim, intermediate_dim, top_k
        );
    } else {
        using namespace gemm_large;
        int m_tiles = (max_tokens_per_expert + BLOCK_M - 1) / BLOCK_M;
        int n_tiles_inter = (intermediate_dim + BLOCK_N - 1) / BLOCK_N;
        int n_tiles_out = (hidden_dim + BLOCK_N - 1) / BLOCK_N;

        size_t gate_smem = max((size_t)(SMEM_A + SMEM_B) * sizeof(__nv_bfloat16),
                               (size_t)SMEM_C * sizeof(float));

        // up kernel: metadata (token_ids, routing) placed AFTER max(load_data, output_data)
        size_t up_smem = max((size_t)(SMEM_A + SMEM_B) * sizeof(__nv_bfloat16), (size_t)SMEM_C * sizeof(float));
        up_smem += BLOCK_M * (sizeof(int) + sizeof(float));

        dim3 grid_gate(n_tiles_inter, m_tiles, num_experts);
        nomic_gate_gemm_large_bf16_kernel<<<grid_gate, THREADS, gate_smem, stream>>>(
            input, gate_weights, sorted_token_ids, expert_offsets,
            intermediate, hidden_dim, intermediate_dim, activation_type
        );

        dim3 grid_up(n_tiles_out, m_tiles, num_experts);
        nomic_up_gemm_large_bf16_kernel<<<grid_up, THREADS, up_smem, stream>>>(
            intermediate, up_weights, sorted_token_ids, sorted_weights, expert_offsets,
            output, hidden_dim, intermediate_dim, top_k
        );
    }
}

#endif // NO_BF16_KERNEL

#ifdef SM90_OR_HIGHER

// SM90+ wgmma kernel for gate projection (with activation)
__global__ void __launch_bounds__(128)
nomic_gate_gemm_sm90_kernel(
    const half* __restrict__ input,
    const half* __restrict__ gate_weights,
    const int* __restrict__ sorted_token_ids,
    const int* __restrict__ expert_offsets,
    half* __restrict__ intermediate,
    int hidden_dim,
    int intermediate_dim,
    int activation_type
) {
    using namespace sm90_gemm;

    const int expert_id = blockIdx.z;
    const int expert_start = expert_offsets[expert_id];
    const int expert_end = expert_offsets[expert_id + 1];
    const int M = expert_end - expert_start;

    if (M == 0) return;

    const int block_m = blockIdx.y * BLOCK_M;
    const int block_n = blockIdx.x * BLOCK_N;

    if (block_m >= M || block_n >= intermediate_dim) return;

    const int tid = threadIdx.x;

    extern __shared__ char smem[];
    constexpr int SMEM_A_SIZE = BLOCK_M * BLOCK_K;
    constexpr int SMEM_B_SIZE = BLOCK_K * BLOCK_N;

    half* s_input = reinterpret_cast<half*>(smem);
    half* s_gate = s_input + SMEM_A_SIZE * STAGES;

    const half* gate_w = gate_weights + (size_t)expert_id * hidden_dim * intermediate_dim;

    float acc[32];
    #pragma unroll
    for (int i = 0; i < 32; i++) {
        acc[i] = 0.0f;
    }

    const int num_k_tiles = (hidden_dim + BLOCK_K - 1) / BLOCK_K;

    // Prologue
    for (int stage = 0; stage < min(STAGES, num_k_tiles); stage++) {
        int k = stage * BLOCK_K;

        for (int i = tid; i < SMEM_A_SIZE / 8; i += THREADS) {
            int m = i / (BLOCK_K / 8);
            int kk = (i % (BLOCK_K / 8)) * 8;
            int global_m = block_m + m;
            int global_k = k + kk;

            float4 val = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
            if (global_m < M && global_k + 7 < hidden_dim) {
                int token_id = sorted_token_ids[expert_start + global_m];
                val = *reinterpret_cast<const float4*>(&input[token_id * hidden_dim + global_k]);
            }
            *reinterpret_cast<float4*>(&s_input[stage * SMEM_A_SIZE + m * BLOCK_K + kk]) = val;
        }

        for (int i = tid; i < SMEM_B_SIZE / 8; i += THREADS) {
            int kk = i / (BLOCK_N / 8);
            int n = (i % (BLOCK_N / 8)) * 8;
            int global_k = k + kk;
            int global_n = block_n + n;

            float4 val = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
            if (global_k < hidden_dim && global_n + 7 < intermediate_dim) {
                val = *reinterpret_cast<const float4*>(&gate_w[global_k * intermediate_dim + global_n]);
            }
            *reinterpret_cast<float4*>(&s_gate[stage * SMEM_B_SIZE + kk * BLOCK_N + n]) = val;
        }
        cp_async_commit();
    }

    // Main loop
    for (int k_tile = 0; k_tile < num_k_tiles; k_tile++) {
        int stage = k_tile % STAGES;

        cp_async_wait<STAGES - 1>();
        __syncthreads();

        if (k_tile + STAGES < num_k_tiles) {
            int next_stage = (k_tile + STAGES) % STAGES;
            int k = (k_tile + STAGES) * BLOCK_K;

            for (int i = tid; i < SMEM_A_SIZE / 8; i += THREADS) {
                int m = i / (BLOCK_K / 8);
                int kk = (i % (BLOCK_K / 8)) * 8;
                int global_m = block_m + m;
                int global_k = k + kk;

                float4 val = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
                if (global_m < M && global_k + 7 < hidden_dim) {
                    int token_id = sorted_token_ids[expert_start + global_m];
                    val = *reinterpret_cast<const float4*>(&input[token_id * hidden_dim + global_k]);
                }
                *reinterpret_cast<float4*>(&s_input[next_stage * SMEM_A_SIZE + m * BLOCK_K + kk]) = val;
            }

            for (int i = tid; i < SMEM_B_SIZE / 8; i += THREADS) {
                int kk = i / (BLOCK_N / 8);
                int n = (i % (BLOCK_N / 8)) * 8;
                int global_k = k + kk;
                int global_n = block_n + n;

                float4 val = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
                if (global_k < hidden_dim && global_n + 7 < intermediate_dim) {
                    val = *reinterpret_cast<const float4*>(&gate_w[global_k * intermediate_dim + global_n]);
                }
                *reinterpret_cast<float4*>(&s_gate[next_stage * SMEM_B_SIZE + kk * BLOCK_N + n]) = val;
            }
            cp_async_commit();
        }

        warpgroup_arrive();

        #pragma unroll
        for (int kk = 0; kk < BLOCK_K; kk += 16) {
            uint64_t desc_a = make_smem_desc(&s_input[stage * SMEM_A_SIZE + kk]);
            uint64_t desc_b = make_smem_desc(&s_gate[stage * SMEM_B_SIZE + kk * BLOCK_N]);
            wgmma_m64n64k16_f16(acc, desc_a, desc_b);
        }

        wgmma_commit();
        wgmma_wait<0>();

        __syncthreads();
    }

    // Store with activation
    float* s_out = reinterpret_cast<float*>(smem);
    store_accumulator_to_smem<64, 64>(s_out, acc, BLOCK_N);
    __syncthreads();

    for (int i = tid; i < BLOCK_M * BLOCK_N; i += THREADS) {
        int m = i / BLOCK_N;
        int n = i % BLOCK_N;
        int global_m = block_m + m;
        int global_n = block_n + n;

        if (global_m < M && global_n < intermediate_dim) {
            float result = apply_activation(s_out[m * BLOCK_N + n], activation_type);
            intermediate[(expert_start + global_m) * intermediate_dim + global_n] = __float2half(result);
        }
    }
}

// SM90+ wgmma kernel for up projection (output to hidden)
__global__ void __launch_bounds__(128)
nomic_up_gemm_sm90_kernel(
    const half* __restrict__ intermediate,
    const half* __restrict__ up_weights,
    const int* __restrict__ sorted_token_ids,
    const float* __restrict__ sorted_weights,
    const int* __restrict__ expert_offsets,
    half* __restrict__ output,
    int hidden_dim,
    int intermediate_dim,
    int top_k
) {
    using namespace sm90_gemm;

    const int expert_id = blockIdx.z;
    const int expert_start = expert_offsets[expert_id];
    const int expert_end = expert_offsets[expert_id + 1];
    const int M = expert_end - expert_start;

    if (M == 0) return;

    const int block_m = blockIdx.y * BLOCK_M;
    const int block_n = blockIdx.x * BLOCK_N;

    if (block_m >= M || block_n >= hidden_dim) return;

    const int tid = threadIdx.x;

    extern __shared__ char smem[];
    constexpr int SMEM_A_SIZE = BLOCK_M * BLOCK_K;
    constexpr int SMEM_B_SIZE = BLOCK_K * BLOCK_N;

    half* s_inter = reinterpret_cast<half*>(smem);
    half* s_up = s_inter + SMEM_A_SIZE * STAGES;

    int* s_token_ids = reinterpret_cast<int*>(s_up + SMEM_B_SIZE * STAGES);
    float* s_routing = reinterpret_cast<float*>(s_token_ids + BLOCK_M);

    const half* up_w = up_weights + (size_t)expert_id * hidden_dim * intermediate_dim;

    // Load token IDs and routing weights
    for (int i = tid; i < BLOCK_M; i += THREADS) {
        int global_m = block_m + i;
        if (global_m < M) {
            s_token_ids[i] = sorted_token_ids[expert_start + global_m];
            s_routing[i] = sorted_weights[expert_start + global_m];
        } else {
            s_token_ids[i] = 0;
            s_routing[i] = 0.0f;
        }
    }

    float acc[32];
    #pragma unroll
    for (int i = 0; i < 32; i++) {
        acc[i] = 0.0f;
    }

    const int num_k_tiles = (intermediate_dim + BLOCK_K - 1) / BLOCK_K;

    // Prologue
    for (int stage = 0; stage < min(STAGES, num_k_tiles); stage++) {
        int k = stage * BLOCK_K;

        for (int i = tid; i < SMEM_A_SIZE / 8; i += THREADS) {
            int m = i / (BLOCK_K / 8);
            int kk = (i % (BLOCK_K / 8)) * 8;
            int global_m = block_m + m;
            int global_k = k + kk;

            float4 val = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
            if (global_m < M && global_k + 7 < intermediate_dim) {
                val = *reinterpret_cast<const float4*>(&intermediate[(expert_start + global_m) * intermediate_dim + global_k]);
            }
            *reinterpret_cast<float4*>(&s_inter[stage * SMEM_A_SIZE + m * BLOCK_K + kk]) = val;
        }

        for (int i = tid; i < SMEM_B_SIZE / 8; i += THREADS) {
            int kk = i / (BLOCK_N / 8);
            int n = (i % (BLOCK_N / 8)) * 8;
            int global_k = k + kk;
            int global_n = block_n + n;

            float4 val = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
            if (global_k < intermediate_dim && global_n + 7 < hidden_dim) {
                // up_weights layout: [hidden_dim, intermediate_dim] accessed as up_w[n * intermediate_dim + k]
                val = *reinterpret_cast<const float4*>(&up_w[global_n * intermediate_dim + global_k]);
            }
            *reinterpret_cast<float4*>(&s_up[stage * SMEM_B_SIZE + kk * BLOCK_N + n]) = val;
        }
        cp_async_commit();
    }

    // Main loop
    for (int k_tile = 0; k_tile < num_k_tiles; k_tile++) {
        int stage = k_tile % STAGES;

        cp_async_wait<STAGES - 1>();
        __syncthreads();

        if (k_tile + STAGES < num_k_tiles) {
            int next_stage = (k_tile + STAGES) % STAGES;
            int k = (k_tile + STAGES) * BLOCK_K;

            for (int i = tid; i < SMEM_A_SIZE / 8; i += THREADS) {
                int m = i / (BLOCK_K / 8);
                int kk = (i % (BLOCK_K / 8)) * 8;
                int global_m = block_m + m;
                int global_k = k + kk;

                float4 val = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
                if (global_m < M && global_k + 7 < intermediate_dim) {
                    val = *reinterpret_cast<const float4*>(&intermediate[(expert_start + global_m) * intermediate_dim + global_k]);
                }
                *reinterpret_cast<float4*>(&s_inter[next_stage * SMEM_A_SIZE + m * BLOCK_K + kk]) = val;
            }

            for (int i = tid; i < SMEM_B_SIZE / 8; i += THREADS) {
                int kk = i / (BLOCK_N / 8);
                int n = (i % (BLOCK_N / 8)) * 8;
                int global_k = k + kk;
                int global_n = block_n + n;

                float4 val = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
                if (global_k < intermediate_dim && global_n + 7 < hidden_dim) {
                    val = *reinterpret_cast<const float4*>(&up_w[global_n * intermediate_dim + global_k]);
                }
                *reinterpret_cast<float4*>(&s_up[next_stage * SMEM_B_SIZE + kk * BLOCK_N + n]) = val;
            }
            cp_async_commit();
        }

        warpgroup_arrive();

        #pragma unroll
        for (int kk = 0; kk < BLOCK_K; kk += 16) {
            uint64_t desc_a = make_smem_desc(&s_inter[stage * SMEM_A_SIZE + kk]);
            uint64_t desc_b = make_smem_desc(&s_up[stage * SMEM_B_SIZE + kk * BLOCK_N]);
            wgmma_m64n64k16_f16(acc, desc_a, desc_b);
        }

        wgmma_commit();
        wgmma_wait<0>();

        __syncthreads();
    }

    // Store results
    float* s_out = reinterpret_cast<float*>(smem);
    store_accumulator_to_smem<64, 64>(s_out, acc, BLOCK_N);
    __syncthreads();

    for (int i = tid; i < BLOCK_M * BLOCK_N; i += THREADS) {
        int m = i / BLOCK_N;
        int n = i % BLOCK_N;
        int global_m = block_m + m;
        int global_n = block_n + n;

        if (global_m < M && global_n < hidden_dim) {
            int token_id = s_token_ids[m];
            float weight = s_routing[m];
            float val = s_out[m * BLOCK_N + n] * weight;

            if (top_k == 1) {
                output[token_id * hidden_dim + global_n] = __float2half(val);
            } else {
                atomic_add_half(&output[token_id * hidden_dim + global_n], __float2half(val));
            }
        }
    }
}
#endif // SM90_OR_HIGHER

extern "C" void nomic_moe_forward(
    const half* input,
    const half* gate_weights,
    const half* up_weights,
    const int* sorted_token_ids,
    const float* sorted_weights,
    const int* expert_offsets,
    half* intermediate,
    half* output,
    int num_tokens,
    int hidden_dim,
    int intermediate_dim,
    int num_experts,
    int max_tokens_per_expert,
    int top_k,
    int activation_type,
    cudaStream_t stream
) {
    if (max_tokens_per_expert <= thresholds::SMALL_GEMM_MAX_TOKENS) {
        using namespace gemm_small;
        int m_tiles = (max_tokens_per_expert + BLOCK_M - 1) / BLOCK_M;
        int n_tiles_inter = (intermediate_dim + BLOCK_N - 1) / BLOCK_N;
        int n_tiles_out = (hidden_dim + BLOCK_N - 1) / BLOCK_N;

        size_t gate_smem = max((size_t)(SMEM_A + SMEM_B) * sizeof(half),
                               (size_t)SMEM_C * sizeof(float));

        // up kernel: metadata (token_ids, routing) placed AFTER max(load_data, output_data)
        size_t up_smem = max((size_t)(SMEM_A + SMEM_B) * sizeof(half), (size_t)SMEM_C * sizeof(float));
        up_smem += BLOCK_M * (sizeof(int) + sizeof(float));

        dim3 grid_gate(n_tiles_inter, m_tiles, num_experts);
        nomic_gate_gemm_small_kernel<<<grid_gate, THREADS, gate_smem, stream>>>(
            input, gate_weights, sorted_token_ids, expert_offsets,
            intermediate, hidden_dim, intermediate_dim, activation_type
        );

        dim3 grid_up(n_tiles_out, m_tiles, num_experts);
        nomic_up_gemm_small_kernel<<<grid_up, THREADS, up_smem, stream>>>(
            intermediate, up_weights, sorted_token_ids, sorted_weights, expert_offsets,
            output, hidden_dim, intermediate_dim, top_k
        );
    } else {
#ifdef SM90_OR_HIGHER
        // Use wgmma kernels for SM90+ (Hopper)
        using namespace sm90_gemm;
        int m_tiles = (max_tokens_per_expert + BLOCK_M - 1) / BLOCK_M;
        int n_tiles_inter = (intermediate_dim + BLOCK_N - 1) / BLOCK_N;
        int n_tiles_out = (hidden_dim + BLOCK_N - 1) / BLOCK_N;

        constexpr int SMEM_A_SIZE = BLOCK_M * BLOCK_K;
        constexpr int SMEM_B_SIZE = BLOCK_K * BLOCK_N;

        size_t gate_smem = STAGES * (SMEM_A_SIZE + SMEM_B_SIZE) * sizeof(half);
        gate_smem = max(gate_smem, BLOCK_M * BLOCK_N * sizeof(float));

        // up kernel: metadata (token_ids, routing) placed AFTER max(staging, output)
        size_t up_smem = max((size_t)STAGES * (SMEM_A_SIZE + SMEM_B_SIZE) * sizeof(half),
                             (size_t)BLOCK_M * BLOCK_N * sizeof(float));
        up_smem += BLOCK_M * (sizeof(int) + sizeof(float));

        dim3 grid_gate(n_tiles_inter, m_tiles, num_experts);
        nomic_gate_gemm_sm90_kernel<<<grid_gate, THREADS, gate_smem, stream>>>(
            input, gate_weights, sorted_token_ids, expert_offsets,
            intermediate, hidden_dim, intermediate_dim, activation_type
        );

        dim3 grid_up(n_tiles_out, m_tiles, num_experts);
        nomic_up_gemm_sm90_kernel<<<grid_up, THREADS, up_smem, stream>>>(
            intermediate, up_weights, sorted_token_ids, sorted_weights, expert_offsets,
            output, hidden_dim, intermediate_dim, top_k
        );
#elif defined(SM80_OR_HIGHER)
        // Use async copy kernels for SM80+ (Ampere)
        using namespace gemm_async;
        int m_tiles = (max_tokens_per_expert + BLOCK_M - 1) / BLOCK_M;
        int n_tiles_inter = (intermediate_dim + BLOCK_N - 1) / BLOCK_N;
        int n_tiles_out = (hidden_dim + BLOCK_N - 1) / BLOCK_N;

        constexpr int SMEM_A = BLOCK_M * BLOCK_K;
        constexpr int SMEM_B = BLOCK_K * BLOCK_N;
        constexpr int SMEM_C = BLOCK_M * BLOCK_N;

        size_t gate_smem = max((size_t)STAGES * (SMEM_A + SMEM_B) * sizeof(half),
                               (size_t)SMEM_C * sizeof(float));

        // up kernel: metadata (token_ids, routing) placed AFTER max(staging, output)
        size_t up_smem = max((size_t)STAGES * (SMEM_A + SMEM_B) * sizeof(half), (size_t)SMEM_C * sizeof(float));
        up_smem += BLOCK_M * (sizeof(int) + sizeof(float));

        dim3 grid_gate(n_tiles_inter, m_tiles, num_experts);
        nomic_gate_gemm_async_kernel<<<grid_gate, THREADS, gate_smem, stream>>>(
            input, gate_weights, sorted_token_ids, expert_offsets,
            intermediate, hidden_dim, intermediate_dim, activation_type
        );

        dim3 grid_up(n_tiles_out, m_tiles, num_experts);
        nomic_up_gemm_async_kernel<<<grid_up, THREADS, up_smem, stream>>>(
            intermediate, up_weights, sorted_token_ids, sorted_weights, expert_offsets,
            output, hidden_dim, intermediate_dim, top_k
        );
#else
        using namespace gemm_large;
        int m_tiles = (max_tokens_per_expert + BLOCK_M - 1) / BLOCK_M;
        int n_tiles_inter = (intermediate_dim + BLOCK_N - 1) / BLOCK_N;
        int n_tiles_out = (hidden_dim + BLOCK_N - 1) / BLOCK_N;

        size_t gate_smem = max((size_t)(SMEM_A + SMEM_B) * sizeof(half),
                               (size_t)SMEM_C * sizeof(float));

        // up kernel: metadata (token_ids, routing) placed AFTER max(load_data, output_data)
        size_t up_smem = max((size_t)(SMEM_A + SMEM_B) * sizeof(half), (size_t)SMEM_C * sizeof(float));
        up_smem += BLOCK_M * (sizeof(int) + sizeof(float));

        dim3 grid_gate(n_tiles_inter, m_tiles, num_experts);
        nomic_gate_gemm_large_kernel<<<grid_gate, THREADS, gate_smem, stream>>>(
            input, gate_weights, sorted_token_ids, expert_offsets,
            intermediate, hidden_dim, intermediate_dim, activation_type
        );

        dim3 grid_up(n_tiles_out, m_tiles, num_experts);
        nomic_up_gemm_large_kernel<<<grid_up, THREADS, up_smem, stream>>>(
            intermediate, up_weights, sorted_token_ids, sorted_weights, expert_offsets,
            output, hidden_dim, intermediate_dim, top_k
        );
#endif
    }
}

// Direct fused kernel launcher for tiny batches (skips preprocessing)
extern "C" void nomic_direct_fused_forward(
    const half* input,
    const half* gate_weights,
    const half* up_weights,
    const float* routing_weights,
    const uint32_t* expert_indices,
    half* output,
    int num_tokens,
    int hidden_dim,
    int intermediate_dim,
    int top_k,
    int activation_type,
    cudaStream_t stream
) {
    using namespace direct_config;

    // Each block handles one (token, expert_slot) pair
    int num_blocks = num_tokens * top_k;
    size_t smem_size = intermediate_dim * sizeof(float);

    nomic_direct_fused_vec_kernel<<<num_blocks, BLOCK_SIZE, smem_size, stream>>>(
        input, gate_weights, up_weights,
        routing_weights, expert_indices, output,
        num_tokens, hidden_dim, intermediate_dim, top_k, activation_type
    );
}

#ifndef NO_BF16_KERNEL
extern "C" void nomic_direct_fused_forward_bf16(
    const __nv_bfloat16* input,
    const __nv_bfloat16* gate_weights,
    const __nv_bfloat16* up_weights,
    const float* routing_weights,
    const uint32_t* expert_indices,
    __nv_bfloat16* output,
    int num_tokens,
    int hidden_dim,
    int intermediate_dim,
    int top_k,
    int activation_type,
    cudaStream_t stream
) {
    using namespace direct_config;

    int num_blocks = num_tokens * top_k;
    size_t smem_size = intermediate_dim * sizeof(float);

    nomic_direct_fused_vec_bf16_kernel<<<num_blocks, BLOCK_SIZE, smem_size, stream>>>(
        input, gate_weights, up_weights,
        routing_weights, expert_indices, output,
        num_tokens, hidden_dim, intermediate_dim, top_k, activation_type
    );
}
#endif
