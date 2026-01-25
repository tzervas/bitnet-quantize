//! GPU kernels for BitNet operations.
//!
//! This module provides CubeCL-based GPU kernels for efficient
//! ternary weight x INT8 activation matrix multiplication.
//!
//! Requires the `cuda` feature to be enabled.

#[cfg(feature = "cuda")]
mod cubecl;

#[cfg(feature = "cuda")]
pub use cubecl::*;

/// Check if CUDA kernels are available.
#[must_use]
pub fn cuda_available() -> bool {
    #[cfg(feature = "cuda")]
    {
        // Check for CUDA device
        candle_core::Device::cuda_if_available(0).is_ok()
    }

    #[cfg(not(feature = "cuda"))]
    {
        false
    }
}
