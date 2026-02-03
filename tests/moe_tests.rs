use anyhow::Result;
use candle::{D, DType, Device, IndexOp, Tensor};
use candle_transformers::models::deepseek2::{BincountOp, NonZeroOp};

mod tolerance {
    pub const FP32: (f32, f32) = (1e-6, 1e-4);
    pub const FP32_QWEN3: (f32, f32) = (1e-6, 5e-4);
    pub const FP16: (f32, f32) = (5e-2, 1e-1);
    pub const BF16: (f32, f32) = (5e-2, 2e-1);
}

#[derive(Clone, Copy)]
enum MoeType {
    Nomic,
    Qwen3,
}

impl MoeType {
    fn as_u32(self) -> u32 {
        match self {
            MoeType::Nomic => 1,
            MoeType::Qwen3 => 0,
        }
    }
}

impl std::fmt::Debug for MoeType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            MoeType::Nomic => write!(f, "Nomic"),
            MoeType::Qwen3 => write!(f, "Qwen3"),
        }
    }
}

struct TestConfig {
    dtype: DType,
    n_embed: usize,
    seq_len: usize,
    num_experts: usize,
    top_k: usize,
    moe_type: MoeType,
}

impl TestConfig {
    fn n_inner(&self) -> usize {
        self.n_embed * 4
    }

    fn tolerance(&self) -> (f32, f32) {
        match (self.dtype, self.moe_type) {
            (DType::F32, MoeType::Nomic) => tolerance::FP32,
            (DType::F32, MoeType::Qwen3) => tolerance::FP32_QWEN3,
            (DType::F16, _) => tolerance::FP16,
            (DType::BF16, _) => tolerance::BF16,
            _ => panic!("Unsupported dtype: {:?}", self.dtype),
        }
    }
}

fn tensors_all_close(a: &Tensor, b: &Tensor, atol: f32, rtol: f32) -> Result<bool> {
    let a = a.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?;
    let b = b.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?;

    let mut max_err = None;

    for (i, (&va, &vb)) in a.iter().zip(b.iter()).enumerate() {
        let diff = (va - vb).abs();
        let tol = atol + rtol * va.abs().max(vb.abs());

        if diff > tol {
            let is_worse = max_err
                .as_ref()
                .map_or(true, |&(_, _, _, d, _): &(usize, f32, f32, f32, f32)| {
                    diff > d
                });
            if is_worse {
                max_err = Some((i, va, vb, diff, tol));
            }
        }
    }

    if let Some((idx, naive, fused, diff, tol)) = max_err {
        println!(
            "Max error at idx {}: naive={}, fused={}, diff={}, tol={}",
            idx, naive, fused, diff, tol
        );
        return Ok(false);
    }
    Ok(true)
}

fn assert_close(naive: &Tensor, fused: &Tensor, atol: f32, rtol: f32, msg: &str) -> Result<()> {
    assert!(
        tensors_all_close(naive, fused, atol, rtol)?,
        "{} (atol={}, rtol={})",
        msg,
        atol,
        rtol
    );
    Ok(())
}

fn forward_router(
    weights: &Tensor,
    seq_len: usize,
    top_k: usize,
    device: &Device,
) -> Result<(Tensor, Tensor)> {
    let topk_weight = Tensor::zeros((seq_len, top_k), DType::F32, device)?;
    let topk_indices = Tensor::zeros((seq_len, top_k), DType::U32, device)?;
    let token_expert_indices = Tensor::zeros((seq_len, top_k), DType::U32, device)?;

    candle_moe::apply_topk_softmax_inplace(
        weights,
        &topk_weight,
        &topk_indices,
        &token_expert_indices,
    )?;

    Ok((topk_weight, topk_indices))
}

fn naive_nomic_mlp(x: &Tensor, gate: &Tensor, up: &Tensor, expert_idx: usize) -> Result<Tensor> {
    let expert_gate = gate.narrow(0, expert_idx, 1)?.squeeze(0)?.t()?;
    let expert_up = up.narrow(0, expert_idx, 1)?.squeeze(0)?;

    let x = x.broadcast_matmul(&expert_gate)?.silu()?;
    Ok(x.broadcast_matmul(&expert_up)?)
}

fn naive_qwen3_mlp(
    x: &Tensor,
    gate: &Tensor,
    up: &Tensor,
    down: &Tensor,
    expert_idx: usize,
) -> Result<Tensor> {
    let expert_gate = gate.narrow(0, expert_idx, 1)?.squeeze(0)?.t()?;
    let expert_up = up.narrow(0, expert_idx, 1)?.squeeze(0)?.t()?;
    let expert_down = down.narrow(0, expert_idx, 1)?.squeeze(0)?.t()?;

    let gate_out = x.broadcast_matmul(&expert_gate)?.silu()?;
    let up_out = x.broadcast_matmul(&expert_up)?;
    let intermediate = gate_out.mul(&up_out)?;
    Ok(intermediate.broadcast_matmul(&expert_down)?)
}

fn naive_forward(
    hidden_states: &Tensor,
    gate: &Tensor,
    up: &Tensor,
    down: Option<&Tensor>,
    scores: &Tensor,
    indices: &Tensor,
    hidden_size: usize,
    num_experts: usize,
) -> Result<Tensor> {
    let hidden_states = hidden_states.reshape(((), hidden_size))?;
    let mut out = Tensor::zeros_like(&hidden_states)?;
    let counts = indices.flatten_all()?.bincount(num_experts as u32)?;

    for (expert_idx, &count) in counts.iter().enumerate().take(num_experts) {
        if count == 0u32 {
            continue;
        }

        let idx_top = indices.eq(expert_idx as f64)?.nonzero()?.t()?;
        let idx = &idx_top.i(0)?.contiguous()?;
        let top = &idx_top.i(1)?.contiguous()?;

        let selected_input = hidden_states.index_select(idx, 0)?;
        let expert_out = match down {
            Some(d) => naive_qwen3_mlp(&selected_input, gate, up, d, expert_idx)?,
            None => naive_nomic_mlp(&selected_input, gate, up, expert_idx)?,
        };

        let routing_weight = scores
            .index_select(idx, 0)?
            .gather(&top.unsqueeze(1)?, 1)?
            .squeeze(1)?
            .unsqueeze(D::Minus1)?
            .to_dtype(hidden_states.dtype())?;

        out = out.index_add(idx, &expert_out.broadcast_mul(&routing_weight)?, 0)?;
    }

    Ok(out)
}

struct TestTensors {
    hidden_states: Tensor,
    gate_weights: Tensor,
    up_weights: Tensor,
    down_weights: Option<Tensor>,
    scores: Tensor,
    indices: Tensor,
}

fn create_test_tensors(cfg: &TestConfig, device: &Device) -> Result<TestTensors> {
    let n_inner = cfg.n_inner();

    let hidden_states =
        Tensor::randn(0.0, 1.0, (cfg.seq_len, cfg.n_embed), device)?.to_dtype(cfg.dtype)?;
    let router_weights =
        Tensor::randn(0.0, 1.0, (cfg.seq_len, cfg.num_experts), device)?.to_dtype(DType::F32)?;

    let (scores, indices) = forward_router(&router_weights, cfg.seq_len, cfg.top_k, device)?;

    let gate_weights = Tensor::randn(0.0, 1.0, (cfg.num_experts, cfg.n_embed, n_inner), device)?
        .to_dtype(cfg.dtype)?;
    let up_weights = Tensor::randn(0.0, 1.0, (cfg.num_experts, cfg.n_embed, n_inner), device)?
        .to_dtype(cfg.dtype)?;

    let down_weights = match cfg.moe_type {
        MoeType::Qwen3 => Some(
            Tensor::randn(0.0, 1.0, (cfg.num_experts, n_inner, cfg.n_embed), device)?
                .to_dtype(cfg.dtype)?,
        ),
        MoeType::Nomic => None,
    };

    Ok(TestTensors {
        hidden_states,
        gate_weights,
        up_weights,
        down_weights,
        scores,
        indices,
    })
}

fn run_moe_test(cfg: TestConfig) -> Result<()> {
    let device = Device::new_cuda(0)?;
    let tensors = create_test_tensors(&cfg, &device)?;
    let (atol, rtol) = cfg.tolerance();

    let naive_output = naive_forward(
        &tensors.hidden_states,
        &tensors.gate_weights.permute((0, 2, 1))?,
        &tensors.up_weights.permute((0, 2, 1))?,
        tensors
            .down_weights
            .as_ref()
            .map(|d| d.permute((0, 2, 1)))
            .transpose()?
            .as_ref(),
        &tensors.scores,
        &tensors.indices,
        cfg.n_embed,
        cfg.num_experts,
    )?;

    let fused_moe = candle_moe::FusedMoE {
        num_experts: cfg.num_experts,
        num_selected_experts: cfg.top_k,
        activation: candle_moe::Activation::Silu,
    };

    let fused_output = fused_moe.forward(
        &tensors.hidden_states,
        &tensors.gate_weights,
        &tensors.up_weights,
        tensors.down_weights.as_ref(),
        &tensors.scores,
        &tensors.indices,
        cfg.moe_type.as_u32(),
    )?;

    let test_name = format!(
        "{:?} {:?} seq={} top_k={}",
        cfg.moe_type, cfg.dtype, cfg.seq_len, cfg.top_k
    );
    assert_close(&naive_output, &fused_output, atol, rtol, &test_name)?;

    Ok(())
}

#[test]
fn fused_moe_nomic_fp32() -> Result<()> {
    run_moe_test(TestConfig {
        dtype: DType::F32,
        n_embed: 16,
        seq_len: 16,
        num_experts: 8,
        top_k: 2,
        moe_type: MoeType::Nomic,
    })
}

#[test]
fn fused_moe_nomic_fp16_gemm() -> Result<()> {
    run_moe_test(TestConfig {
        dtype: DType::F16,
        n_embed: 64,
        seq_len: 32,
        num_experts: 8,
        top_k: 2,
        moe_type: MoeType::Nomic,
    })
}

#[test]
fn fused_moe_nomic_fp16_gemm_topk1() -> Result<()> {
    run_moe_test(TestConfig {
        dtype: DType::F16,
        n_embed: 64,
        seq_len: 32,
        num_experts: 8,
        top_k: 1,
        moe_type: MoeType::Nomic,
    })
}

#[test]
fn fused_moe_qwen3_fp32() -> Result<()> {
    run_moe_test(TestConfig {
        dtype: DType::F32,
        n_embed: 64,
        seq_len: 32,
        num_experts: 8,
        top_k: 2,
        moe_type: MoeType::Qwen3,
    })
}

#[test]
fn fused_moe_qwen3_fp16_direct() -> Result<()> {
    run_moe_test(TestConfig {
        dtype: DType::F16,
        n_embed: 64,
        seq_len: 4,
        num_experts: 8,
        top_k: 2,
        moe_type: MoeType::Qwen3,
    })
}

#[test]
fn fused_moe_qwen3_fp16_gemm() -> Result<()> {
    run_moe_test(TestConfig {
        dtype: DType::F16,
        n_embed: 64,
        seq_len: 32,
        num_experts: 8,
        top_k: 2,
        moe_type: MoeType::Qwen3,
    })
}

#[test]
fn fused_moe_qwen3_fp16_gemm_topk1() -> Result<()> {
    run_moe_test(TestConfig {
        dtype: DType::F16,
        n_embed: 64,
        seq_len: 32,
        num_experts: 8,
        top_k: 1,
        moe_type: MoeType::Qwen3,
    })
}

#[test]
fn fused_moe_qwen3_bf16_direct() -> Result<()> {
    run_moe_test(TestConfig {
        dtype: DType::BF16,
        n_embed: 64,
        seq_len: 4,
        num_experts: 8,
        top_k: 2,
        moe_type: MoeType::Qwen3,
    })
}

#[test]
fn fused_moe_qwen3_bf16_gemm() -> Result<()> {
    run_moe_test(TestConfig {
        dtype: DType::BF16,
        n_embed: 64,
        seq_len: 32,
        num_experts: 8,
        top_k: 2,
        moe_type: MoeType::Qwen3,
    })
}

#[test]
fn fused_moe_qwen3_bf16_gemm_topk1() -> Result<()> {
    run_moe_test(TestConfig {
        dtype: DType::BF16,
        n_embed: 64,
        seq_len: 32,
        num_experts: 8,
        top_k: 1,
        moe_type: MoeType::Qwen3,
    })
}
