//! CubeCL GPU kernels for BitNet.
//!
//! This module provides GPU-accelerated implementations of BitNet operations.
//! Currently a placeholder for future implementation.

#![cfg(feature = "cuda")]

use candle_core::Tensor;

use crate::error::{BitNetError, Result};
use crate::quantization::TernaryWeight;

/// GPU-accelerated ternary matrix multiplication.
///
/// Computes `output = input @ ternary_weight.T` using GPU kernels.
///
/// # Arguments
///
/// * `input` - Input tensor [batch, in_features] (INT8 quantized)
/// * `weight` - Ternary weight
///
/// # Errors
///
/// Returns error if CUDA operation fails.
///
/// # Note
///
/// This is currently a placeholder that falls back to CPU implementation.
/// Future versions will implement optimized CubeCL kernels.
pub fn ternary_matmul_gpu(input: &Tensor, weight: &TernaryWeight) -> Result<Tensor> {
    // TODO: Implement CubeCL kernel for ternary matmul
    // For now, fall back to CPU implementation

    let device = input.device();
    let dequant_weight = crate::quantization::dequantize_weights(weight, device)?;

    let output = input
        .matmul(&dequant_weight.t()?)
        .map_err(BitNetError::from)?;

    Ok(output)
}

/// Check if the GPU kernel is available and beneficial.
///
/// Returns true if:
/// - CUDA device is available
/// - Input size is large enough to benefit from GPU acceleration
#[must_use]
pub fn should_use_gpu(input: &Tensor, weight: &TernaryWeight) -> bool {
    // Check if on CUDA device
    if !input.device().is_cuda() {
        return false;
    }

    // Heuristic: use GPU for matrices larger than threshold
    let input_size = input.elem_count();
    let weight_size = weight.out_features() * weight.in_features();

    // Threshold: 64K elements
    input_size * weight_size > 65536
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::BitNetConfig;
    use crate::quantization::quantize_weights;
    use candle_core::Device;

    #[test]
    fn test_ternary_matmul_cpu_fallback() {
        let device = Device::Cpu;
        let config = BitNetConfig::default().with_group_size(64);

        let weight_tensor = candle_core::Tensor::randn(0.0f32, 1.0, (64, 128), &device).unwrap();
        let weight = quantize_weights(&weight_tensor, &config).unwrap();

        let input = candle_core::Tensor::randn(0.0f32, 1.0, (4, 128), &device).unwrap();

        let output = ternary_matmul_gpu(&input, &weight).unwrap();
        assert_eq!(output.shape().dims(), &[4, 64]);
    }
}
