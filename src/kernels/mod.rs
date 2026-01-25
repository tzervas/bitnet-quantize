//! GPU kernels for BitNet quantization operations.
//!
//! This module provides CubeCL-based GPU kernels for efficient
//! ternary weight x activation matrix multiplication.
//!
//! ## Kernels
//!
//! - `absmean_quantize` - Quantize weights to ternary {-1, 0, +1}
//! - `ternary_dequantize` - Convert ternary back to float
//! - `ternary_matmul_gpu` - Optimized ternary matmul (no multiply ops!)
//! - `packed_ternary_matmul` - 2-bit packed weights for reduced bandwidth
//! - `bitlinear_forward` - Fused LayerNorm + ternary matmul
//!
//! ## Feature Gate
//!
//! Requires the `cuda` feature to be enabled:
//!
//! ```toml
//! [dependencies]
//! bitnet-quantize = { version = "0.1", features = ["cuda"] }
//! ```

#[cfg(feature = "cuda")]
mod cubecl;

#[cfg(feature = "cuda")]
pub use cubecl::{
    // Core operations
    absmean_quantize,
    ternary_dequantize,
    ternary_matmul_gpu,
    ternary_matmul_raw,
    // Packed operations
    pack_ternary_weights,
    packed_ternary_matmul,
    unpack_ternary_weights,
    // Fused operations
    bitlinear_forward,
    // Utilities
    has_cuda_support,
    should_use_gpu,
};

/// Check if CUDA kernels are available.
#[must_use]
pub fn cuda_available() -> bool {
    #[cfg(feature = "cuda")]
    {
        cubecl::has_cuda_support()
    }

    #[cfg(not(feature = "cuda"))]
    {
        false
    }
}
