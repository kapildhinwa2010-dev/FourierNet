"""
FourierNet - Fourier-feature neural networks for weight generation

This module provides Fourier-feature based neural network weight generation,
allowing extreme compression of neural network parameters.

Key Features:
- Generate dense weight matrices from minimal parameters
- Fourier-feature based coordinate encoding
- Chunked processing for memory efficiency
- Xavier initialization scaling

Example:
    >>> from fourier_net import FourierDNA
    >>> dna = FourierDNA(hidden=128, num_freqs=64)
    >>> weights = dna.generate_weights(out_dim=256, in_dim=512)
    >>> print(weights.shape)
    torch.Size([256, 512])
"""

import math
import torch
import torch.nn as nn
from typing import Optional


class FourierDNA(nn.Module):
    """Generate weight matrices from coordinates using Fourier features.
    
    This implementation uses Fourier features to map 2D coordinates to
    weight values, enabling extreme compression of neural network parameters.
    
    Args:
        hidden: Hidden layer width for the DNA network (default: 128)
        num_freqs: Number of Fourier frequencies (default: 64)
        chunk_size: Maximum number of coordinates processed at once (default: 16384)
    
    Attributes:
        B: Random Fourier projection matrix (num_freqs, 2)
        net: 3-layer MLP for coordinate-to-weight mapping
        scale: Learnable scale parameter
    """
    
    def __init__(self, hidden: int = 128, num_freqs: int = 64, chunk_size: int = 16384):
        super().__init__()
        
        if hidden < 1 or num_freqs < 1 or chunk_size < 1:
            raise ValueError("hidden, num_freqs, and chunk_size must be positive integers")
        
        self.hidden = int(hidden)
        self.num_freqs = int(num_freqs)
        self.chunk_size = int(chunk_size)
        
        # Fixed random Fourier projection matrix
        self.register_buffer("B", torch.randn(self.num_freqs, 2) * 20.0)
        
        # Input dimension for the network
        input_dim = self.num_freqs * 2
        
        # DNA network: 3-layer MLP with SiLU activations
        self.net = nn.Sequential(
            nn.Linear(input_dim, self.hidden),
            nn.SiLU(),
            nn.Linear(self.hidden, self.hidden),
            nn.SiLU(),
            nn.Linear(self.hidden, self.hidden),
            nn.SiLU(),
            nn.Linear(self.hidden, 1),
        )
        
        # Learnable scale parameter
        self.scale = nn.Parameter(torch.tensor(0.01))
    
    def generate_weights(self, out_dim: int, in_dim: int, *, chunk_size: Optional[int] = None) -> torch.Tensor:
        """Generate a weight matrix of shape (out_dim, in_dim).
        
        The generation is chunked to bound intermediate activation memory.
        Gradients remain enabled by default; wrap inference-only calls in
        torch.no_grad() or torch.inference_mode() to avoid retaining the autograd graph.
        
        Args:
            out_dim: Number of output coordinates (rows)
            in_dim: Number of input coordinates (columns)
            chunk_size: Optional override for the coordinate batch size
            
        Returns:
            Weight matrix of shape (out_dim, in_dim)
            
        Raises:
            TypeError: If out_dim or in_dim are not integers
            ValueError: If out_dim or in_dim are not positive
        """
        if not isinstance(out_dim, int) or not isinstance(in_dim, int):
            raise TypeError("out_dim and in_dim must be integers")
        if out_dim < 1 or in_dim < 1:
            raise ValueError("out_dim and in_dim must be positive")
        
        batch_size = self.chunk_size if chunk_size is None else chunk_size
        if not isinstance(batch_size, int) or batch_size < 1:
            raise ValueError("chunk_size must be a positive integer")
        
        device, dtype = self.B.device, self.B.dtype
        
        # Generate coordinate grid
        row_values = torch.linspace(-1, 1, out_dim, device=device, dtype=dtype)
        col_values = torch.linspace(-1, 1, in_dim, device=device, dtype=dtype)
        
        total = out_dim * in_dim
        flat_indices = torch.arange(total, device=device)
        
        outputs = []
        
        # Process in chunks to bound memory
        for start in range(0, total, batch_size):
            end = min(start + batch_size, total)
            flat = flat_indices[start:end]
            
            # Convert flat indices to 2D coordinates
            rows = torch.div(flat, in_dim, rounding_mode="floor")
            cols = flat.remainder(in_dim)
            
            # Get coordinate values
            coords = torch.stack((row_values[rows], col_values[cols]), dim=-1)
            
            # Fourier features: sin(2π * B * coords), cos(2π * B * coords)
            projection = (2.0 * math.pi) * (coords @ self.B.T)
            features = torch.cat((torch.sin(projection), torch.cos(projection)), dim=-1)
            
            # Generate weights for this chunk
            outputs.append(self.net(features).squeeze(-1))
        
        # Reshape to final matrix
        raw_weights = torch.cat(outputs).reshape(out_dim, in_dim)
        
        # Apply Xavier initialization scaling
        xavier_scale = math.sqrt(2.0 / (in_dim + out_dim))
        return raw_weights * self.scale * xavier_scale
    
    def count_parameters(self) -> int:
        """Return the number of trainable parameters.
        
        Returns:
            Number of trainable parameters in the DNA network
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def extra_repr(self) -> str:
        return f"hidden={self.hidden}, num_freqs={self.num_freqs}, " \
               f"chunk_size={self.chunk_size}, parameters={self.count_parameters():,}"


def test_fourier_dna():
    """Test FourierDNA functionality."""
    print("Testing FourierDNA...")
    
    # Create DNA generator
    dna = FourierDNA(hidden=128, num_freqs=64)
    print(f"Created FourierDNA with {dna.count_parameters():,} parameters")
    
    # Generate small matrix
    weights = dna.generate_weights(out_dim=256, in_dim=512)
    print(f"Generated weights: {weights.shape}")
    assert weights.shape == (256, 512), "Shape mismatch"
    
    # Test chunking
    weights_chunked = dna.generate_weights(out_dim=256, in_dim=512, chunk_size=1024)
    assert torch.allclose(weights, weights_chunked), "Chunking produces different results"
    
    # Test gradients
    dna.train()
    weights_grad = dna.generate_weights(out_dim=256, in_dim=512)
    loss = weights_grad.sum()
    loss.backward()
    assert any(p.grad is not None for p in dna.parameters()), "Gradients not computed"
    
    print("✓ All tests passed!")


if __name__ == "__main__":
    test_fourier_dna()
