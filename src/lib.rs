pub mod ffi;
pub mod moe_ops;

use candle::cuda_backend::cudarc::driver::DevicePtr;
use candle::{CudaDevice, DType, Device, Result, Storage, Tensor};
use half::{bf16, f16};

pub fn apply_topk_softmax_<
    T: candle::cuda_backend::CudaDType + candle::cuda_backend::cudarc::driver::DeviceRepr,
>(
    gating_output: &Tensor,
    topk_weight: &Tensor,
    topk_indices: &Tensor,
    token_expert_indices: &Tensor,
) -> Result<()> {
    let (g, g_l) = gating_output.storage_and_layout();
    let g: &candle::CudaStorage = match &*g {
        Storage::Cuda(g) => g,
        _ => candle::bail!("gating_output must be a cuda tensor"),
    };

    let (w, w_l) = topk_weight.storage_and_layout();
    let w = match &*w {
        Storage::Cuda(w) => w,
        _ => candle::bail!("topk_weight must be a cuda tensor"),
    };

    let (i, i_l) = topk_indices.storage_and_layout();
    let i = match &*i {
        Storage::Cuda(i) => i,
        _ => candle::bail!("topk_indices must be a cuda tensor"),
    };

    let (ei, ei_l) = token_expert_indices.storage_and_layout();
    let ei: &candle::CudaStorage = match &*ei {
        Storage::Cuda(ei) => ei,
        _ => candle::bail!("token_expert_indices must be a cuda tensor"),
    };

    let g_rank = g_l.stride().len();
    let w_rank = w_l.stride().len();
    let i_rank = i_l.stride().len();
    let ei_rank = ei_l.stride().len();

    if g_rank != 2 || w_rank != 2 || i_rank != 2 || ei_rank != 2 {
        candle::bail!(
            "apply_topk_softmax_inplace expects input tensors of rank 2 (w: {w_l:?}, i: {i_l:?}, ei: {ei_l:?}, g: {g_l:?})"
        )
    }

    // Get cuda slices for all tensors
    let g = g.as_cuda_slice::<T>()?;
    let w = w.as_cuda_slice::<T>()?;
    let i = i.as_cuda_slice::<u32>()?;
    let ei = ei.as_cuda_slice::<u32>()?;

    // Get cuda views for all tensors
    let g = g.slice(g_l.start_offset()..);
    let w = w.slice(w_l.start_offset()..);
    let i = i.slice(i_l.start_offset()..);
    let ei = ei.slice(ei_l.start_offset()..);

    let (num_tokens, top_k) = w_l.shape().dims2()?;
    let (_, num_experts) = g_l.shape().dims2()?;

    let is_pow2 = (num_experts != 0) && ((num_experts & (num_experts - 1)) == 0);
    if !is_pow2 || num_experts > 256 {
        candle::bail!(
            "num_experts should be power of 2 and smaller than 256 (num_experts: {num_experts:?})"
        )
    }

    if (num_tokens, top_k) != i_l.shape().dims2()? {
        candle::bail!(
            "shape mismatch topk_indices {:?}, expected {:?}",
            i_l.shape(),
            (num_tokens, top_k)
        )
    }

    if (num_tokens, top_k) != ei_l.shape().dims2()? {
        candle::bail!(
            "shape mismatch token_expert_indices {:?}, expected {:?}",
            ei_l.shape(),
            (num_tokens, top_k)
        )
    }

    let gate_ptr = *g.device_ptr() as *const core::ffi::c_void;
    let weight_ptr = *w.device_ptr() as *const core::ffi::c_void;
    let indices_ptr = *i.device_ptr() as *const core::ffi::c_void;
    let expert_indices_ptr = *ei.device_ptr() as *const core::ffi::c_void;

    unsafe {
        ffi::topk_softmax(
            gate_ptr,
            weight_ptr,
            indices_ptr,
            expert_indices_ptr,
            num_experts as i32,
            num_tokens as i32,
            top_k as i32,
        )
    }

    Ok(())
}

pub fn apply_topk_softmax_inplace(
    gating_output: &Tensor,
    topk_weight: &Tensor,
    topk_indices: &Tensor,
    token_expert_indices: &Tensor,
) -> Result<()> {
    match topk_weight.dtype() {
        DType::F16 => apply_topk_softmax_::<f16>(
            gating_output,
            topk_weight,
            topk_indices,
            token_expert_indices,
        ),
        DType::BF16 => apply_topk_softmax_::<bf16>(
            gating_output,
            topk_weight,
            topk_indices,
            token_expert_indices,
        ),
        DType::F32 => apply_topk_softmax_::<f32>(
            gating_output,
            topk_weight,
            topk_indices,
            token_expert_indices,
        ),
        dt => {
            candle::bail!(
                "apply_topk_softmax_inplace is only supported for f32, f16 and bf16 ({dt:?})"
            )
        }
    }
}

/// Performs fused MoE forward pass with gate, up, and down projections
pub struct FusedMoeForward {
    num_experts: usize,
    num_selected_experts: usize,
    activation: Activation,
}

#[derive(Clone, Copy, Debug)]
pub enum Activation {
    Silu,
    Gelu,
    Relu,
}

impl Activation {
    fn to_int(self) -> i32 {
        match self {
            Activation::Silu => 0,
            Activation::Gelu => 1,
            Activation::Relu => 2,
        }
    }
}

impl FusedMoeForward {
    pub fn new(num_experts: usize, num_selected_experts: usize, activation: Activation) -> Self {
        Self {
            num_experts,
            num_selected_experts,
            activation,
        }
    }

    /// Performs fused MoE forward pass
    /// Args:
    /// - input: [num_tokens, hidden_dim]
    /// - gate_weights: [num_experts, hidden_dim, intermediate_dim]
    /// - up_weights: [num_experts, hidden_dim, intermediate_dim]
    /// - down_weights: [num_experts, intermediate_dim, hidden_dim]
    /// - routing_weights: [num_tokens, num_selected_experts]
    /// - expert_indices: [num_tokens, num_selected_experts]
    ///
    /// Returns:
    /// - output: [num_tokens, hidden_dim]
    pub fn forward(
        &self,
        input: &Tensor,
        gate_weights: &Tensor,
        up_weights: &Tensor,
        down_weights: &Tensor,
        routing_weights: &Tensor,
        expert_indices: &Tensor,
    ) -> Result<Tensor> {
        let device = input.device();

        // Validate inputs
        let (num_tokens, hidden_dim) = input.dims2()?;
        let (ne_g, hd_g, id_g) = gate_weights.dims3()?;
        let (ne_u, hd_u, id_u) = up_weights.dims3()?;
        let (ne_d, id_d, hd_d) = down_weights.dims3()?;
        let (nt, nse) = routing_weights.dims2()?;
        let (nt2, nse2) = expert_indices.dims2()?;

        if ne_g != self.num_experts || ne_u != self.num_experts || ne_d != self.num_experts {
            candle::bail!("Number of experts mismatch");
        }
        if hd_g != hidden_dim || hd_u != hidden_dim {
            candle::bail!("Hidden dimension mismatch for gate/up weights");
        }
        if hd_d != hidden_dim {
            candle::bail!(
                "Hidden dimension mismatch for down weights (expected {}, got {})",
                hidden_dim,
                hd_d
            );
        }
        if id_g != id_u || id_u != id_d {
            candle::bail!(
                "Intermediate dimension mismatch (gate: {}, up: {}, down: {})",
                id_g,
                id_u,
                id_d
            );
        }

        let _intermediate_dim = id_g;
        if nt != num_tokens || nt2 != num_tokens {
            candle::bail!("Number of tokens mismatch");
        }
        if nse != self.num_selected_experts || nse2 != self.num_selected_experts {
            candle::bail!("Number of selected experts mismatch");
        }

        // Create output tensor
        let output = Tensor::zeros((num_tokens, hidden_dim), input.dtype(), device)?;

        // Call CUDA kernel
        match device {
            Device::Cuda(cuda_device) => {
                self.cuda_fwd(
                    input,
                    gate_weights,
                    up_weights,
                    down_weights,
                    routing_weights,
                    expert_indices,
                    &output,
                    cuda_device,
                )?;
            }
            _ => {
                candle::bail!("FusedMoeForward only supports CUDA device");
            }
        }

        Ok(output)
    }

    /// Performs optimized fused MoE forward pass with tiling and better memory patterns
    pub fn forward_optimized(
        &self,
        input: &Tensor,
        gate_weights: &Tensor,
        up_weights: &Tensor,
        down_weights: &Tensor,
        routing_weights: &Tensor,
        expert_indices: &Tensor,
    ) -> Result<Tensor> {
        let device = input.device();

        // Validate inputs (same as before)
        let (num_tokens, hidden_dim) = input.dims2()?;
        let (ne_g, hd_g, id_g) = gate_weights.dims3()?;
        let (ne_u, hd_u, id_u) = up_weights.dims3()?;
        let (ne_d, id_d, hd_d) = down_weights.dims3()?;
        let (nt, nse) = routing_weights.dims2()?;
        let (nt2, nse2) = expert_indices.dims2()?;

        // Validate dimensions
        if ne_g != self.num_experts || ne_u != self.num_experts || ne_d != self.num_experts {
            candle::bail!("Number of experts mismatch");
        }
        if hd_g != hidden_dim || hd_u != hidden_dim {
            candle::bail!("Hidden dimension mismatch for gate/up weights");
        }
        if hd_d != hidden_dim {
            candle::bail!("Hidden dimension mismatch for down weights");
        }
        if id_g != id_u || id_u != id_d {
            candle::bail!("Intermediate dimension mismatch");
        }
        if nt != num_tokens || nt2 != num_tokens {
            candle::bail!("Number of tokens mismatch");
        }
        if nse != self.num_selected_experts || nse2 != self.num_selected_experts {
            candle::bail!("Number of selected experts mismatch");
        }

        // Create output tensor
        let output = Tensor::zeros((num_tokens, hidden_dim), input.dtype(), device)?;

        // Call optimized CUDA kernel
        match device {
            Device::Cuda(cuda_device) => {
                self.cuda_fwd_optimized(
                    input,
                    gate_weights,
                    up_weights,
                    down_weights,
                    routing_weights,
                    expert_indices,
                    &output,
                    cuda_device,
                )?;
            }
            _ => {
                candle::bail!("FusedMoeForward only supports CUDA device");
            }
        }

        Ok(output)
    }

    #[allow(clippy::too_many_arguments)]
    fn cuda_fwd(
        &self,
        input: &Tensor,
        gate_weights: &Tensor,
        up_weights: &Tensor,
        down_weights: &Tensor,
        routing_weights: &Tensor,
        expert_indices: &Tensor,
        output: &Tensor,
        _device: &CudaDevice,
    ) -> Result<()> {
        let (num_tokens, hidden_dim) = input.dims2()?;
        let (_, hd_gate, intermediate_dim) = gate_weights.dims3()?;

        // Validate that gate weights have correct dimensions
        if hd_gate != hidden_dim {
            candle::bail!(
                "gate_weights hidden_dim {} doesn't match input {}",
                hd_gate,
                hidden_dim
            );
        }

        // Get storage and layouts
        let (input_storage, input_layout) = input.storage_and_layout();
        let (gate_storage, gate_layout) = gate_weights.storage_and_layout();
        let (up_storage, up_layout) = up_weights.storage_and_layout();
        let (down_storage, down_layout) = down_weights.storage_and_layout();
        let (routing_storage, routing_layout) = routing_weights.storage_and_layout();
        let (indices_storage, indices_layout) = expert_indices.storage_and_layout();
        let (output_storage, output_layout) = output.storage_and_layout();

        // Extract CUDA storage
        let input_cuda = match &*input_storage {
            Storage::Cuda(cuda_storage) => cuda_storage,
            _ => candle::bail!("input must be a cuda tensor"),
        };
        let gate_cuda = match &*gate_storage {
            Storage::Cuda(cuda_storage) => cuda_storage,
            _ => candle::bail!("gate_weights must be a cuda tensor"),
        };
        let up_cuda = match &*up_storage {
            Storage::Cuda(cuda_storage) => cuda_storage,
            _ => candle::bail!("up_weights must be a cuda tensor"),
        };
        let down_cuda = match &*down_storage {
            Storage::Cuda(cuda_storage) => cuda_storage,
            _ => candle::bail!("down_weights must be a cuda tensor"),
        };
        let routing_cuda = match &*routing_storage {
            Storage::Cuda(cuda_storage) => cuda_storage,
            _ => candle::bail!("routing_weights must be a cuda tensor"),
        };
        let indices_cuda = match &*indices_storage {
            Storage::Cuda(cuda_storage) => cuda_storage,
            _ => candle::bail!("expert_indices must be a cuda tensor"),
        };
        let output_cuda = match &*output_storage {
            Storage::Cuda(cuda_storage) => cuda_storage,
            _ => candle::bail!("output must be a cuda tensor"),
        };

        match (input.dtype(), expert_indices.dtype()) {
            (DType::F32, DType::U32) => {
                let input_slice = input_cuda
                    .as_cuda_slice::<f32>()?
                    .slice(input_layout.start_offset()..);
                let gate_slice = gate_cuda
                    .as_cuda_slice::<f32>()?
                    .slice(gate_layout.start_offset()..);
                let up_slice = up_cuda
                    .as_cuda_slice::<f32>()?
                    .slice(up_layout.start_offset()..);
                let down_slice = down_cuda
                    .as_cuda_slice::<f32>()?
                    .slice(down_layout.start_offset()..);
                let routing_slice = routing_cuda
                    .as_cuda_slice::<f32>()?
                    .slice(routing_layout.start_offset()..);
                let indices_slice = indices_cuda
                    .as_cuda_slice::<u32>()?
                    .slice(indices_layout.start_offset()..);
                let output_slice = output_cuda
                    .as_cuda_slice::<f32>()?
                    .slice(output_layout.start_offset()..);

                unsafe {
                    let input_ptr = *input_slice.device_ptr() as *const core::ffi::c_void;
                    let gate_ptr = *gate_slice.device_ptr() as *const core::ffi::c_void;
                    let up_ptr = *up_slice.device_ptr() as *const core::ffi::c_void;
                    let down_ptr = *down_slice.device_ptr() as *const core::ffi::c_void;
                    let routing_ptr = *routing_slice.device_ptr() as *const core::ffi::c_void;
                    let indices_ptr = *indices_slice.device_ptr() as *const core::ffi::c_void;
                    let output_ptr = *output_slice.device_ptr() as *const core::ffi::c_void;

                    ffi::fused_moe_forward(
                        input_ptr,
                        gate_ptr,
                        up_ptr,
                        down_ptr,
                        routing_ptr,
                        indices_ptr,
                        output_ptr,
                        num_tokens as i32,
                        hidden_dim as i32,
                        intermediate_dim as i32,
                        self.num_selected_experts as i32,
                        self.activation.to_int(),
                        2_u32,
                    );
                }
            }
            (DType::F16, DType::U32) => {
                let input_slice = input_cuda
                    .as_cuda_slice::<half::f16>()?
                    .slice(input_layout.start_offset()..);
                let gate_slice = gate_cuda
                    .as_cuda_slice::<half::f16>()?
                    .slice(gate_layout.start_offset()..);
                let up_slice = up_cuda
                    .as_cuda_slice::<half::f16>()?
                    .slice(up_layout.start_offset()..);
                let down_slice = down_cuda
                    .as_cuda_slice::<half::f16>()?
                    .slice(down_layout.start_offset()..);
                let routing_slice = routing_cuda
                    .as_cuda_slice::<f32>()?
                    .slice(routing_layout.start_offset()..);
                let indices_slice = indices_cuda
                    .as_cuda_slice::<u32>()?
                    .slice(indices_layout.start_offset()..);
                let output_slice = output_cuda
                    .as_cuda_slice::<half::f16>()?
                    .slice(output_layout.start_offset()..);

                unsafe {
                    let input_ptr = *input_slice.device_ptr() as *const core::ffi::c_void;
                    let gate_ptr = *gate_slice.device_ptr() as *const core::ffi::c_void;
                    let up_ptr = *up_slice.device_ptr() as *const core::ffi::c_void;
                    let down_ptr = *down_slice.device_ptr() as *const core::ffi::c_void;
                    let routing_ptr = *routing_slice.device_ptr() as *const core::ffi::c_void;
                    let indices_ptr = *indices_slice.device_ptr() as *const core::ffi::c_void;
                    let output_ptr = *output_slice.device_ptr() as *const core::ffi::c_void;

                    ffi::fused_moe_forward(
                        input_ptr,
                        gate_ptr,
                        up_ptr,
                        down_ptr,
                        routing_ptr,
                        indices_ptr,
                        output_ptr,
                        num_tokens as i32,
                        hidden_dim as i32,
                        intermediate_dim as i32,
                        self.num_selected_experts as i32,
                        self.activation.to_int(),
                        0_u32,
                    );
                }
            }
            (DType::BF16, DType::U32) => {
                let input_slice = input_cuda
                    .as_cuda_slice::<half::bf16>()?
                    .slice(input_layout.start_offset()..);
                let gate_slice = gate_cuda
                    .as_cuda_slice::<half::bf16>()?
                    .slice(gate_layout.start_offset()..);
                let up_slice = up_cuda
                    .as_cuda_slice::<half::bf16>()?
                    .slice(up_layout.start_offset()..);
                let down_slice = down_cuda
                    .as_cuda_slice::<half::bf16>()?
                    .slice(down_layout.start_offset()..);
                let routing_slice = routing_cuda
                    .as_cuda_slice::<f32>()?
                    .slice(routing_layout.start_offset()..);
                let indices_slice = indices_cuda
                    .as_cuda_slice::<u32>()?
                    .slice(indices_layout.start_offset()..);
                let output_slice = output_cuda
                    .as_cuda_slice::<half::bf16>()?
                    .slice(output_layout.start_offset()..);

                unsafe {
                    let input_ptr = *input_slice.device_ptr() as *const core::ffi::c_void;
                    let gate_ptr = *gate_slice.device_ptr() as *const core::ffi::c_void;
                    let up_ptr = *up_slice.device_ptr() as *const core::ffi::c_void;
                    let down_ptr = *down_slice.device_ptr() as *const core::ffi::c_void;
                    let routing_ptr = *routing_slice.device_ptr() as *const core::ffi::c_void;
                    let indices_ptr = *indices_slice.device_ptr() as *const core::ffi::c_void;
                    let output_ptr = *output_slice.device_ptr() as *const core::ffi::c_void;

                    ffi::fused_moe_forward(
                        input_ptr,
                        gate_ptr,
                        up_ptr,
                        down_ptr,
                        routing_ptr,
                        indices_ptr,
                        output_ptr,
                        num_tokens as i32,
                        hidden_dim as i32,
                        intermediate_dim as i32,
                        self.num_selected_experts as i32,
                        self.activation.to_int(),
                        1_u32,
                    );
                }
            }
            _ => candle::bail!("Unsupported dtype combination for FusedMoeForward"),
        }

        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn cuda_fwd_optimized(
        &self,
        input: &Tensor,
        gate_weights: &Tensor,
        up_weights: &Tensor,
        down_weights: &Tensor,
        routing_weights: &Tensor,
        expert_indices: &Tensor,
        output: &Tensor,
        _device: &CudaDevice,
    ) -> Result<()> {
        let (num_tokens, hidden_dim) = input.dims2()?;
        let (_, _, intermediate_dim) = gate_weights.dims3()?;

        // Get storage and layouts
        let (input_storage, input_layout) = input.storage_and_layout();
        let (gate_storage, gate_layout) = gate_weights.storage_and_layout();
        let (up_storage, up_layout) = up_weights.storage_and_layout();
        let (down_storage, down_layout) = down_weights.storage_and_layout();
        let (routing_storage, routing_layout) = routing_weights.storage_and_layout();
        let (indices_storage, indices_layout) = expert_indices.storage_and_layout();
        let (output_storage, output_layout) = output.storage_and_layout();

        // Extract CUDA storage
        let input_cuda = match &*input_storage {
            Storage::Cuda(cuda_storage) => cuda_storage,
            _ => candle::bail!("input must be a cuda tensor"),
        };
        let gate_cuda = match &*gate_storage {
            Storage::Cuda(cuda_storage) => cuda_storage,
            _ => candle::bail!("gate_weights must be a cuda tensor"),
        };
        let up_cuda = match &*up_storage {
            Storage::Cuda(cuda_storage) => cuda_storage,
            _ => candle::bail!("up_weights must be a cuda tensor"),
        };
        let down_cuda = match &*down_storage {
            Storage::Cuda(cuda_storage) => cuda_storage,
            _ => candle::bail!("down_weights must be a cuda tensor"),
        };
        let routing_cuda = match &*routing_storage {
            Storage::Cuda(cuda_storage) => cuda_storage,
            _ => candle::bail!("routing_weights must be a cuda tensor"),
        };
        let indices_cuda = match &*indices_storage {
            Storage::Cuda(cuda_storage) => cuda_storage,
            _ => candle::bail!("expert_indices must be a cuda tensor"),
        };
        let output_cuda = match &*output_storage {
            Storage::Cuda(cuda_storage) => cuda_storage,
            _ => candle::bail!("output must be a cuda tensor"),
        };

        match (input.dtype(), expert_indices.dtype()) {
            (DType::F32, DType::U32) => {
                let input_slice = input_cuda
                    .as_cuda_slice::<f32>()?
                    .slice(input_layout.start_offset()..);
                let gate_slice = gate_cuda
                    .as_cuda_slice::<f32>()?
                    .slice(gate_layout.start_offset()..);
                let up_slice = up_cuda
                    .as_cuda_slice::<f32>()?
                    .slice(up_layout.start_offset()..);
                let down_slice = down_cuda
                    .as_cuda_slice::<f32>()?
                    .slice(down_layout.start_offset()..);
                let routing_slice = routing_cuda
                    .as_cuda_slice::<f32>()?
                    .slice(routing_layout.start_offset()..);
                let indices_slice = indices_cuda
                    .as_cuda_slice::<u32>()?
                    .slice(indices_layout.start_offset()..);
                let output_slice = output_cuda
                    .as_cuda_slice::<f32>()?
                    .slice(output_layout.start_offset()..);

                unsafe {
                    let input_ptr = *input_slice.device_ptr() as *const core::ffi::c_void;
                    let gate_ptr = *gate_slice.device_ptr() as *const core::ffi::c_void;
                    let up_ptr = *up_slice.device_ptr() as *const core::ffi::c_void;
                    let down_ptr = *down_slice.device_ptr() as *const core::ffi::c_void;
                    let routing_ptr = *routing_slice.device_ptr() as *const core::ffi::c_void;
                    let indices_ptr = *indices_slice.device_ptr() as *const core::ffi::c_void;
                    let output_ptr = *output_slice.device_ptr() as *const core::ffi::c_void;

                    ffi::fused_moe_forward(
                        input_ptr,
                        gate_ptr,
                        up_ptr,
                        down_ptr,
                        routing_ptr,
                        indices_ptr,
                        output_ptr,
                        num_tokens as i32,
                        hidden_dim as i32,
                        intermediate_dim as i32,
                        self.num_selected_experts as i32,
                        self.activation.to_int(),
                        2_u32,
                    );
                }
            }
            (DType::F16, DType::U32) => {
                let input_slice = input_cuda
                    .as_cuda_slice::<half::f16>()?
                    .slice(input_layout.start_offset()..);
                let gate_slice = gate_cuda
                    .as_cuda_slice::<half::f16>()?
                    .slice(gate_layout.start_offset()..);
                let up_slice = up_cuda
                    .as_cuda_slice::<half::f16>()?
                    .slice(up_layout.start_offset()..);
                let down_slice = down_cuda
                    .as_cuda_slice::<half::f16>()?
                    .slice(down_layout.start_offset()..);
                let routing_slice = routing_cuda
                    .as_cuda_slice::<f32>()?
                    .slice(routing_layout.start_offset()..);
                let indices_slice = indices_cuda
                    .as_cuda_slice::<u32>()?
                    .slice(indices_layout.start_offset()..);
                let output_slice = output_cuda
                    .as_cuda_slice::<half::f16>()?
                    .slice(output_layout.start_offset()..);

                unsafe {
                    let input_ptr = *input_slice.device_ptr() as *const core::ffi::c_void;
                    let gate_ptr = *gate_slice.device_ptr() as *const core::ffi::c_void;
                    let up_ptr = *up_slice.device_ptr() as *const core::ffi::c_void;
                    let down_ptr = *down_slice.device_ptr() as *const core::ffi::c_void;
                    let routing_ptr = *routing_slice.device_ptr() as *const core::ffi::c_void;
                    let indices_ptr = *indices_slice.device_ptr() as *const core::ffi::c_void;
                    let output_ptr = *output_slice.device_ptr() as *const core::ffi::c_void;

                    ffi::fused_moe_forward(
                        input_ptr,
                        gate_ptr,
                        up_ptr,
                        down_ptr,
                        routing_ptr,
                        indices_ptr,
                        output_ptr,
                        num_tokens as i32,
                        hidden_dim as i32,
                        intermediate_dim as i32,
                        self.num_selected_experts as i32,
                        self.activation.to_int(),
                        0_u32,
                    );
                }
            }
            (DType::BF16, DType::U32) => {
                let input_slice = input_cuda
                    .as_cuda_slice::<half::bf16>()?
                    .slice(input_layout.start_offset()..);
                let gate_slice = gate_cuda
                    .as_cuda_slice::<half::bf16>()?
                    .slice(gate_layout.start_offset()..);
                let up_slice = up_cuda
                    .as_cuda_slice::<half::bf16>()?
                    .slice(up_layout.start_offset()..);
                let down_slice = down_cuda
                    .as_cuda_slice::<half::bf16>()?
                    .slice(down_layout.start_offset()..);
                let routing_slice = routing_cuda
                    .as_cuda_slice::<f32>()?
                    .slice(routing_layout.start_offset()..);
                let indices_slice = indices_cuda
                    .as_cuda_slice::<u32>()?
                    .slice(indices_layout.start_offset()..);
                let output_slice = output_cuda
                    .as_cuda_slice::<half::bf16>()?
                    .slice(output_layout.start_offset()..);

                unsafe {
                    let input_ptr = *input_slice.device_ptr() as *const core::ffi::c_void;
                    let gate_ptr = *gate_slice.device_ptr() as *const core::ffi::c_void;
                    let up_ptr = *up_slice.device_ptr() as *const core::ffi::c_void;
                    let down_ptr = *down_slice.device_ptr() as *const core::ffi::c_void;
                    let routing_ptr = *routing_slice.device_ptr() as *const core::ffi::c_void;
                    let indices_ptr = *indices_slice.device_ptr() as *const core::ffi::c_void;
                    let output_ptr = *output_slice.device_ptr() as *const core::ffi::c_void;

                    ffi::fused_moe_forward(
                        input_ptr,
                        gate_ptr,
                        up_ptr,
                        down_ptr,
                        routing_ptr,
                        indices_ptr,
                        output_ptr,
                        num_tokens as i32,
                        hidden_dim as i32,
                        intermediate_dim as i32,
                        self.num_selected_experts as i32,
                        self.activation.to_int(),
                        1_u32,
                    );
                }
            }
            _ => candle::bail!(
                "Unsupported dtype combination for FusedMoeForward: {:?} with {:?} indices",
                input.dtype(),
                expert_indices.dtype()
            ),
        }

        Ok(())
    }
}
