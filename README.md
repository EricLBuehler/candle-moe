# candle-moe

Fast CUDA `fused MoE` for [Candle](https://github.com/huggingface/candle) backend.

## Requirements

* SM 75+ (Volta+) GPU
* `Candle 0.9+`

## Benchmark

vs `candle 0.9.1` native kernels / `topk-softmax kernel`

| seq_len | num_experts | top k | candle 0.9 | candle-moe | speed-up |
|  :---:  | :---:       | :---: |  :---:     | :---:      | :---:    |
| 32      | 8           | 2     | 26.013 µs  | 7.968 µs   | 3.26x    |
| 512     | 8           | 2     | 25.829 µs  | 7.888 µs   | 3.27x    |
| 8192    | 8           | 2     | 46.106 µs  | 8.262 µs   | 5.58x    |
| 32768   | 8           | 2     | 100.683 µs | 9.743 µs   | 10.33x   |

Benchmarks run on A40 GPU

vs `candle 0.9.1` native kernels / `fused MoE kernel`

| moe type |   fp  | seq_len | hidden_dim | num_experts | top k | candle 0.9 | candle-moe | speed-up |
|   :---:  | :---: |  :---:  | :---:      | :---:       | :---: | :---:      | :---:      | :---:    |
| nomic    | f32   | 16      | 768        | 8           | 2     | 1.319 ms   | 770.3 µs   | 1.71x    |
| nomic    | f32   | 64      | 768        | 8           | 2     | 1.459 ms   | 987.9 µs   | 1.48x    |
| nomic    | f32   | 512     | 768        | 8           | 2     | 2.251 ms   | 1.568 ms   | 1.44x    |
| nomic    | f32   | 4096    | 768        | 8           | 2     | 11.237 ms  | 11.03 ms   | 1.02x    |
| nomic    | f32   | 32768   | 768        | 8           | 2     | 44.174 ms  | 14.18 ms   | 3.11x    |
| nomic    | f16   | 16      | 768        | 8           | 2     | 1.626 ms   | 3.065 ms   | 0.53x    |
| nomic    | f16   | 64      | 768        | 8           | 2     | 1.471 ms   | 19.34 µs   | 76.03x   |
| nomic    | f16   | 4096    | 768        | 8           | 2     | 7.038 ms   | 1.793 ms   | 3.93x    |
| nomic    | f16   | 32768   | 768        | 8           | 2     | 51.325 ms  | 10.86 ms   | 4.73x    |
| nomic    | bf16  | 16      | 768        | 8           | 2     | 1.667 ms   | 2.793 ms   | 0.60x    |
| nomic    | bf16  | 64      | 768        | 8           | 2     | 1.425 ms   | 24.82 µs   | 57.41x   |
| nomic    | bf16  | 4096    | 768        | 8           | 2     | 10.049 ms  | 1.840 ms   | 5.46x    |
| nomic    | bf16  | 32768   | 768        | 8           | 2     | 51.315 ms  | 10.81 ms   | 4.75x    |
| qwen3    | f16   | 8       | 4096       | 8           | 2     | 5.062 ms   | 11.04 µs   | 458.62x  |
| qwen3    | f16   | 16      | 4096       | 8           | 2     | 6.601 ms   | 18.25 µs   | 361.65x  |
| qwen3    | f16   | 32      | 4096       | 8           | 2     | 6.799 ms   | 11.05 µs   | 615.23x  |
| qwen3    | f16   | 64      | 4096       | 8           | 2     | 6.992 ms   | 27.78 µs   | 251.74x  |
| qwen3    | f16   | 32768   | 4096       | 8           | 2     | 340.631 ms | 58.37 ms   | 5.84x    |
| qwen3    | bf16  | 64      | 4096       | 8           | 2     | 7.068 ms   | 30.53 µs   | 231.51x  |
| qwen3    | bf16  | 32768   | 4096       | 8           | 2     | 344.266 ms | 58.28 ms   | 5.91x    |

Benchmarks run on A40 GPU

% some numeric issues with bf16 precision

## Usage

Add to your `Cargo.toml`.

```toml
[dependencies]
candle-moe = { git = "https://github.com/kozistr/candle-moe", rev = "990ac1f42248dd441c51c9b5bcb73c5b77c03f99" }
candle-core = { version = "0.9", features = ["cuda"] }
```

```rust
let topk_weight = Tensor::zeros((seq_len, self.top_k), DType::F32, device)?;
let topk_indices = Tensor::zeros((seq_len, self.top_k), DType::U32, device)?;
let token_expert_indices = Tensor::zeros((seq_len, self.top_k), DType::U32, device)?;

candle_moe::apply_topk_softmax_inplace(
    &weights,
    &topk_weight,
    &topk_indices,
    &token_expert_indices,
)?;

...

let num_experts = 32;
let top_k = 2;
let moe_act = match activation {
    HiddenAct::Silu => candle_moe::Activation::Silu,
    HiddenAct::Gelu => candle_moe::Activation::Gelu,
    HiddenAct::Relu => candle_moe::Activation::Relu,
    _ => candle::bail!("not supported activation type"),
};

let fused_moe = candle_moe::FusedMoE::new(num_experts, top_k, moe_act);

...

let mut out = self.fused_moe.forward(
    &hidden_states,
    &self.gate_weight,
    &self.up_weight,
    None,
    &top_weights,
    &top_experts,
    1_u32, // Nomic MoE
)?;
```

## Run

### Profile

```bash
$ cargo build --release --bin profile_fused_moe && nsys profile -t cuda,osrt --stats=true --force-overwrite true -o nsys_moe ./target/release/profile_fused_moe
```

### Bench

```bash
cargo bench --bench bench_fused_moe
```

### Test

```bash
cargo test
```
