"""Tests for FourierDNA class."""

import torch
import pytest
from fourier_net import FourierDNA


class TestFourierDNA:
    """Test suite for FourierDNA."""
    
    def test_init_default(self):
        """Test default initialization."""
        dna = FourierDNA()
        assert dna.hidden == 128
        assert dna.num_freqs == 64
        assert dna.chunk_size == 16384
    
    def test_init_custom(self):
        """Test custom initialization."""
        dna = FourierDNA(hidden=64, num_freqs=32, chunk_size=8192)
        assert dna.hidden == 64
        assert dna.num_freqs == 32
        assert dna.chunk_size == 8192
    
    def test_init_invalid(self):
        """Test invalid initialization."""
        with pytest.raises(ValueError):
            FourierDNA(hidden=0, num_freqs=64)
        
        with pytest.raises(ValueError):
            FourierDNA(hidden=128, num_freqs=0)
        
        with pytest.raises(ValueError):
            FourierDNA(hidden=128, num_freqs=64, chunk_size=0)
    
    def test_generate_weights_shape(self):
        """Test weight generation produces correct shape."""
        dna = FourierDNA()
        weights = dna.generate_weights(out_dim=256, in_dim=512)
        assert weights.shape == (256, 512)
    
    def test_generate_weights_small(self):
        """Test weight generation with small dimensions."""
        dna = FourierDNA()
        weights = dna.generate_weights(out_dim=10, in_dim=10)
        assert weights.shape == (10, 10)
    
    def test_generate_weights_chunked(self):
        """Test chunked weight generation."""
        dna = FourierDNA(chunk_size=1024)
        weights1 = dna.generate_weights(out_dim=256, in_dim=512)
        weights2 = dna.generate_weights(out_dim=256, in_dim=512, chunk_size=512)
        assert torch.allclose(weights1, weights2)
    
    def test_generate_weights_invalid(self):
        """Test invalid weight generation."""
        dna = FourierDNA()
        
        with pytest.raises(TypeError):
            dna.generate_weights(out_dim="256", in_dim=512)
        
        with pytest.raises(TypeError):
            dna.generate_weights(out_dim=256, in_dim="512")
        
        with pytest.raises(ValueError):
            dna.generate_weights(out_dim=0, in_dim=512)
        
        with pytest.raises(ValueError):
            dna.generate_weights(out_dim=256, in_dim=0)
    
    def test_count_parameters(self):
        """Test parameter counting."""
        dna = FourierDNA(hidden=128, num_freqs=64)
        params = dna.count_parameters()
        
        # Calculate expected parameters
        # B: num_freqs * 2 (not trainable)
        # net: (num_freqs*2 * hidden) + hidden + (hidden * hidden) + hidden + (hidden * hidden) + hidden + (hidden * 1) + 1
        # scale: 1
        expected = (64 * 2 * 128) + 128 + (128 * 128) + 128 + (128 * 128) + 128 + (128 * 1) + 1
        assert params == expected
    
    def test_gradients(self):
        """Test gradient flow."""
        dna = FourierDNA()
        weights = dna.generate_weights(out_dim=256, in_dim=512)
        loss = weights.sum()
        loss.backward()
        
        # Check that gradients exist for trainable parameters
        for name, param in dna.named_parameters():
            if param.requires_grad:
                assert param.grad is not None
    
    def test_inference_mode(self):
        """Test inference mode."""
        dna = FourierDNA()
        
        with torch.inference_mode():
            weights = dna.generate_weights(out_dim=256, in_dim=512)
        
        assert weights.shape == (256, 512)
    
    def test_device_consistency(self):
        """Test device consistency."""
        dna = FourierDNA()
        weights = dna.generate_weights(out_dim=256, in_dim=512)
        assert weights.device == dna.B.device
    
    def test_dtype_consistency(self):
        """Test dtype consistency."""
        dna = FourierDNA()
        weights = dna.generate_weights(out_dim=256, in_dim=512)
        assert weights.dtype == dna.B.dtype


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
