# bitnet-rs

Microsoft BitNet b1.58 implementation in Rust with ternary weight quantization.

[![Crates.io](https://img.shields.io/crates/v/bitnet-rs.svg)](https://crates.io/crates/bitnet-rs)
[![Documentation](https://docs.rs/bitnet-rs/badge.svg)](https://docs.rs/bitnet-rs)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## Overview

`bitnet-rs` implements the BitNet b1.58 architecture for efficient neural network inference:

- **Ternary Weights**: Quantized to {-1, 0, +1} using AbsMean
- **INT8 Activations**: Per-token AbsMax quantization
- **BitLinear Layer**: Drop-in replacement for `nn::Linear`
- **Straight-Through Estimator**: For training with quantization
- **peft-rs Integration**: Use as a PEFT adapter
- **GGUF Export**: Compatible with llama.cpp

## Installation

Add to your `Cargo.toml`:

```toml
[dependencies]
bitnet-rs = "0.1"
```

### Optional Features

```toml
[dependencies]
bitnet-rs = { version = "0.1", features = ["cuda", "peft", "gguf-export"] }
```

| Feature | Description |
|---------|-------------|
| `cuda` | GPU acceleration via CubeCL |
| `peft` | peft-rs adapter integration |
| `gguf-export` | Export to GGUF format |

## Quick Start

```rust
use bitnet_rs::{BitLinear, BitNetConfig};
use candle_core::{Device, Tensor};
use candle_nn::Module;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let device = Device::Cpu;
    let config = BitNetConfig::default();

    // Create layer from existing weights
    let weight = Tensor::randn(0.0f32, 1.0, (512, 256), &device)?;
    let layer = BitLinear::from_weight(&weight, None, &config)?;

    // Forward pass
    let input = Tensor::randn(0.0f32, 1.0, (4, 256), &device)?;
    let output = layer.forward(&input)?;

    println!("Input shape: {:?}", input.shape());
    println!("Output shape: {:?}", output.shape());
    println!("Compression ratio: {:.2}x", layer.compression_ratio());
    println!("Weight sparsity: {:.1}%", layer.sparsity() * 100.0);

    Ok(())
}
```

## BitNet b1.58 Algorithm

### Weight Quantization (AbsMean)

Weights are quantized to ternary values:

```
W_q = round(W / mean(|W|))  clamped to {-1, 0, +1}
```

- Values near zero become 0 (sparse)
- Large positive values become +1
- Large negative values become -1

### Activation Quantization (AbsMax)

Activations are quantized to INT8 per-token:

```
X_q = round(X * 127 / max(|X|))  clamped to [-127, +127]
```

### Compression Benefits

| Original | Quantized | Compression |
|----------|-----------|-------------|
| FP32 (32 bits) | 2 bits/weight | 16x |
| FP16 (16 bits) | 2 bits/weight | 8x |

## Configuration

```rust
use bitnet_rs::BitNetConfig;

let config = BitNetConfig::builder()
    .group_size(128)           // Weights per scale group
    .activation_bits(8)        // INT8 activations
    .per_token(true)           // Per-token scaling
    .use_ste(true)             // Straight-Through Estimator
    .build()?;
```

## Training with STE

The Straight-Through Estimator enables training through quantization:

```rust
use bitnet_rs::layer::{ternary_ste, int8_ste};

// Forward: quantize to ternary
let quantized = ternary_ste(&weights)?;

// Backward: gradients pass through unchanged
// (handled automatically by Candle's autograd)
```

## peft-rs Integration

Use BitNet as a PEFT adapter:

```rust
use bitnet_rs::BitNetAdapter;
use peft_rs::Adapter;

let adapter = BitNetAdapter::new(config)?;
let adapted_weight = adapter.forward(&base_weight)?;
```

## Performance

Benchmarks on CPU (Intel i7):

| Layer Size | Forward Pass | Quantization |
|------------|--------------|--------------|
| 256x512 | 0.8ms | 0.2ms |
| 512x1024 | 2.1ms | 0.5ms |
| 1024x4096 | 12ms | 2.1ms |

Run benchmarks:

```bash
cargo bench -p bitnet-rs
```

## Documentation

Full API documentation: [docs.rs/bitnet-rs](https://docs.rs/bitnet-rs)

## References

- Ma, S., et al. (2024). "The Era of 1-bit LLMs: All Large Language Models are in 1.58 Bits"
- Wang, H., et al. (2023). "BitNet: Scaling 1-bit Transformers for Large Language Models"

## License

MIT License - see [LICENSE](LICENSE) for details.
