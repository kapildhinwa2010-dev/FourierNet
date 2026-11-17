# FourierNet

FourierNet is a small Python/PyTorch implementation of a neural generator that maps a two-dimensional coordinate grid to a dense tensor of weights. It provides the `FourierDNA` module in `fourier_net.py`; it does not include a pretrained model or a workflow for reconstructing or replacing the weights of an existing network.

## Project status

The project metadata labels this version `0.1.0` and classifies it as Beta. The repository contains the generator implementation, dependency declarations, tests, and GitHub Actions workflows. It does not contain a training script, example trained checkpoint, dataset, model-integration layer, command-line interface, or measured task-accuracy or benchmark results. Treat generated weights as outputs of an initialized generator unless you train the module yourself; no pretrained-weight reconstruction quality is established here.

## Implemented

`FourierDNA` builds a coordinate-to-scalar MLP using sine and cosine Fourier features. A fixed random projection buffer maps the 2D coordinates into `2 * num_freqs` features, and a trainable MLP with SiLU activations produces each matrix value. A trainable scalar and a Xavier-style dimension-dependent factor scale the result. `generate_weights(out_dim, in_dim)` returns a tensor of shape `(out_dim, in_dim)`, processes coordinates in chunks, and supports gradient flow by default. The chunk size can be set when constructing the module or overridden per call. `count_parameters()` reports trainable parameters.

Defaults are `hidden=128`, `num_freqs=64`, and `chunk_size=16384`. The projection buffer and model parameters use PyTorch's normal module device and dtype behavior; move the module with `.to(...)` when needed.

## Requirements and local use

The project declares Python 3.10 or newer and dependencies on PyTorch 2.0 or newer and NumPy 1.24 or newer. From a checkout, install the listed requirements:

```bash
python -m pip install -r requirements.txt
```

There is no command-line interface. Import the module from the repository checkout and generate a matrix as follows:

```python
import torch
from fourier_net import FourierDNA

generator = FourierDNA(hidden=128, num_freqs=64)
with torch.inference_mode():
    weights = generator.generate_weights(out_dim=256, in_dim=512)
print(weights.shape)  # torch.Size([256, 512])
```

For training or gradient-based optimization, call `generate_weights` with gradients enabled and define your own objective and training loop; the repository does not supply one. Chunking limits the size of intermediate coordinate/feature batches, but the complete output matrix is still materialized in memory.

## Tests and CI

At commit `8a950e0b69982d581bf920191952cf24e464a6be`, the repository's pytest suite was run locally with `python -m pytest -q`: **12 tests passed**. This is a local run, not a claim about a GitHub-hosted CI run. The CI workflow is configured to install requirements, run pytest, compile the module and tests, and perform a small import/generation smoke test on Python 3.10, 3.11, and 3.12. A separate CodeQL workflow is configured for Python analysis; no result from that workflow is asserted here.

To run the tests locally:

```bash
python -m pytest -q
```

## Limitations

- No training procedure, pretrained checkpoint, dataset, or comparison against stored or trained weights is included.
- No reconstruction quality, downstream accuracy, speed, memory, or compression benchmark is reported by the repository. Parameter counts alone do not establish equivalent model quality or practical compression.
- The generator returns a full dense tensor; chunking does not eliminate the memory required for that final tensor.
- There is no CLI or application-specific integration. The module is a building block that callers must train and integrate for their own use.

## License

The repository includes an MIT License. See [LICENSE](LICENSE) for its terms and the copyright notice.