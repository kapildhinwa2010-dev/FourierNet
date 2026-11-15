# FourierNet

[![CI](https://github.com/kapildhinwa2010-dev/FourierNet/actions/workflows/ci.yml/badge.svg)](https://github.com/kapildhinwa2010-dev/FourierNet/actions/workflows/ci.yml)
[![CI](https://github.com/kapildhinwa2010-dev/FourierNet/actions/workflows/codeql.yml/badge.svg)](https://github.com/kapildhinwa2010-dev/FourierNet/actions/workflows/codeql.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)

Generate dense weight matrices from a small Fourier-feature MLP.

## What it does

FourierDNA maps a 2D coordinate grid through random Fourier features and
a small MLP, producing an out_dim x in_dim weight matrix from roughly
50K generator parameters. The param-count ratio can be large for big
matrices - but see Honest results below before trusting it as
compression.

## Installation

    pip install -r requirements.txt

## Usage

    from fourier_net import FourierDNA

    dna = FourierDNA(hidden=128, num_freqs=64)
    weights = dna.generate_weights(out_dim=256, in_dim=512)

    # Inference mode avoids retaining the autograd graph
    import torch
    with torch.inference_mode():
        big = dna.generate_weights(out_dim=1024, in_dim=1024)

## Architecture

1. 2D coordinate grid normalized to [-1, 1]
2. Random Fourier projection B (num_freqs x 2), features sin/cos(2*pi*B*coords)
3. 3-layer MLP with SiLU activations and a learnable scale
4. Xavier-style output scaling; chunked generation for large matrices

## API

- FourierDNA(hidden=128, num_freqs=64, chunk_size=16384)
- generate_weights(out_dim, in_dim, chunk_size=None)
- count_parameters()

## Honest results and limitations

- The "compression ratio" is parameter arithmetic: (out_dim x in_dim) /
  generator params. It says nothing about reconstruction quality.
- An untrained generator does NOT reproduce trained weights. The output
  is a smooth, low-frequency approximation; real trained weight matrices
  are not smooth, so reconstruction error is large.
- No task-accuracy results are reported anywhere in this repo because
  none were measured in a controlled experiment. Do not cite ratios
  here as compression results.
- Generation is slower than reading stored weights: a 4096x4096 matrix
  takes on the order of tens of seconds on CPU.

## Negative results (from the source experiments)

- In the reserv.dean-part-2 experiment log, generated-weight approaches
  did not produce usable reconstructions of trained weights without
  per-model training and fine-tuning, which erases most of the claimed
  savings.
- Noise-robustness of the carrier-mode recall variant failed about half
  the trials; the safe design kept in the toolkit is the Reed-Solomon
  slot lattice (see ECC-Memory), not Fourier weight generation.

## License

MIT - see [LICENSE](LICENSE).
