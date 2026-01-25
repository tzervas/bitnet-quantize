# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.0] - 2025-01-24

### Added

- `BitNetConfig`: Configuration for BitNet quantization
  - Configurable group size for weight quantization
  - Per-token or per-tensor activation scaling
  - Training mode with STE support
- `TernaryWeight`: Packed ternary weight storage
  - AbsMean quantization: `W_q = round(W / mean(|W|))`
  - Per-group scale factors
  - Compression tracking and sparsity metrics
- `QuantizedActivations`: INT8 activation quantization
  - AbsMax quantization: `X_q = round(X * 127 / max(|X|))`
  - Per-token scaling for sequence models
  - Efficient dequantization
- `BitLinear`: Drop-in replacement for `nn::Linear`
  - Compatible with candle-nn Module trait
  - Supports 2D and 3D input tensors
  - Optional bias term
  - Forward pass with automatic dequantization
  - `forward_quantized` for explicit quantization control
- Straight-Through Estimator (STE) functions
  - `ternary_ste`: Forward quantization with gradient passthrough
  - `int8_ste`: INT8 quantization with gradient passthrough
- peft-rs adapter integration (optional, `peft` feature)
  - `BitNetAdapter` implementing `Adapter` trait
  - Configuration via `BitNetAdapterConfig`
- GGUF export support (optional, `gguf-export` feature)
- CubeCL GPU kernel stubs (optional, `cuda` feature)
- Comprehensive test suite (35 unit tests)
- Criterion benchmarks for quantization and forward pass

### Technical Details

- Built on candle 0.9.x tensor library
- Minimum Rust version: 1.92
- Optional dependencies gated behind feature flags
- Integration with rust-ai workspace

### References

- BitNet b1.58: "The Era of 1-bit LLMs" (Ma et al., 2024)
- Original BitNet: "Scaling 1-bit Transformers" (Wang et al., 2023)

[Unreleased]: https://github.com/your-org/bitnet-quantize/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/your-org/bitnet-quantize/releases/tag/v0.1.0
