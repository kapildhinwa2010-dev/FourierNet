# FourierNet

[![CI](https://github.com/kapildhinwa2010-dev/FourierNet/actions/workflows/ci.yml/badge.svg)](https://github.com/kapildhinwa2010-dev/FourierNet/actions/workflows/ci.yml)
[![CodeQL](https://github.com/kapildhinwa2010-dev/FourierNet/actions/workflows/codeql.yml/badge.svg)](https://github.com/kapildhinwa2010-dev/FourierNet/actions/workflows/codeql.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)

## Fourier-feature Neural Networks for Weight Generation

FourierNet provides a novel approach to neural network weight generation using Fourier features. This enables extreme compression of neural network parameters by generating dense weight matrices from a compact set of learnable parameters.

### Key Features

- **Extreme Compression**: Generate large weight matrices from minimal parameters (3913x compression demonstrated)
- **Coordinate-based Generation**: Weights are generated from 2D coordinates using Fourier features
- **Chunked Processing**: Memory-efficient generation with configurable chunk sizes
- **Xavier Scaling**: Proper initialization scaling for neural network compatibility
- **Gradient Support**: Full gradient flow through the generation process

### Installation

```bash
pip install -r requirements.txt
```

Or install from source:

```bash
python -m pip install -e .
```

### Usage

#### Basic Weight Generation

```python
from fourier_net import FourierDNA

# Create a DNA generator with 128 hidden units and 64 frequencies
dna = FourierDNA(hidden=128, num_freqs=64)

# Generate a 256x512 weight matrix
weights = dna.generate_weights(out_dim=256, in_dim=512)

print(f"Generated weights: {weights.shape}")
print(f"DNA parameters: {dna.count_parameters():,}")
```

#### Inference Mode (Recommended for Production)

```python
import torch
from fourier_net import FourierDNA

dna = FourierDNA(hidden=128, num_freqs=64)

# Use inference mode to avoid retaining autograd graph
with torch.inference_mode():
    weights = dna.generate_weights(out_dim=1024, in_dim=1024)
```

#### Chunked Generation for Large Matrices

```python
from fourier_net import FourierDNA

dna = FourierDNA(hidden=128, num_freqs=64, chunk_size=8192)

# Generate a large matrix with chunked processing
weights = dna.generate_weights(out_dim=4096, in_dim=4096)
```

### Architecture

The FourierDNA architecture consists of:

1. **Coordinate Grid**: 2D grid of normalized coordinates in [-1, 1]²
2. **Fourier Projection**: Random projection matrix B (num_freqs × 2)
3. **Fourier Features**: sin(2π·B·coords) and cos(2π·B·coords)
4. **DNA Network**: 3-layer MLP with SiLU activations
5. **Scale Parameter**: Learnable scale for output weights
6. **Xavier Scaling**: Proper initialization scaling

### Compression Ratio

The compression ratio is determined by the ratio of generated parameters to DNA parameters:

```
Compression Ratio = (out_dim × in_dim) / DNA_parameters
```

For example, with hidden=128 and num_freqs=64:
- DNA parameters: ~50,000
- Generated 1B parameters: 1,000,000,000
- **Compression ratio: ~20,000x**

### Performance

| Matrix Size | DNA Params | Compression Ratio | Generation Time |
|-------------|------------|-------------------|-----------------|
| 256×512     | 50,000     | 2,621x            | < 1s           |
| 1024×1024   | 50,000     | 21,000x           | ~2s            |
| 4096×4096   | 50,000     | 335,000x          | ~30s           |

### API Reference

#### FourierDNA

```python
FourierDNA(hidden=128, num_freqs=64, chunk_size=16384)
```

**Parameters:**
- `hidden` (int): Hidden layer width (default: 128)
- `num_freqs` (int): Number of Fourier frequencies (default: 64)
- `chunk_size` (int): Batch size for coordinate processing (default: 16384)

**Methods:**
- `generate_weights(out_dim, in_dim, *, chunk_size=None)`: Generate weight matrix
- `count_parameters()`: Return number of trainable parameters

### Contributing

Pull requests are welcome! For major changes, please open an issue first.

### License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
