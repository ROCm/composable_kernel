"""
Unit tests for PyTorch integration
"""

import pytest

# Check if PyTorch is available
try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

if HAS_TORCH:
    from ck_tile_dispatcher import (
        ck_gemm,
        CKLinear,
        CKMLP,
        convert_linear_to_ck,
        benchmark_vs_pytorch,
    )
    import torch.nn as nn


@pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not available")
class TestTorchGEMM:
    """Test PyTorch GEMM operations"""
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_ck_gemm_cuda(self):
        """Test CK GEMM on CUDA"""
        A = torch.randn(128, 128, device='cuda', dtype=torch.float16)
        B = torch.randn(128, 128, device='cuda', dtype=torch.float16)
        
        C = ck_gemm(A, B)
        
        assert C.shape == (128, 128)
        assert C.device.type == 'cuda'
        assert C.dtype == torch.float16
    
    def test_ck_gemm_cpu(self):
        """Test CK GEMM on CPU (fallback)"""
        A = torch.randn(64, 64, dtype=torch.float16)
        B = torch.randn(64, 64, dtype=torch.float16)
        
        C = ck_gemm(A, B)
        
        assert C.shape == (64, 64)
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_ck_gemm_correctness(self):
        """Test CK GEMM correctness"""
        A = torch.randn(64, 64, device='cuda', dtype=torch.float16)
        B = torch.randn(64, 64, device='cuda', dtype=torch.float16)
        
        C_ck = ck_gemm(A, B)
        C_pt = torch.matmul(A, B)
        
        max_diff = torch.max(torch.abs(C_ck - C_pt)).item()
        assert max_diff < 0.1


@pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not available")
class TestCKLinear:
    """Test CKLinear layer"""
    
    def test_create_layer(self):
        """Test layer creation"""
        layer = CKLinear(128, 256)
        
        assert layer.in_features == 128
        assert layer.out_features == 256
        assert layer.weight.shape == (256, 128)
    
    def test_forward_cpu(self):
        """Test forward pass on CPU"""
        layer = CKLinear(128, 256).half()
        input = torch.randn(32, 128, dtype=torch.float16)
        
        output = layer(input)
        
        assert output.shape == (32, 256)
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_forward_cuda(self):
        """Test forward pass on CUDA"""
        layer = CKLinear(128, 256).cuda().half()
        input = torch.randn(32, 128, device='cuda', dtype=torch.float16)
        
        output = layer(input)
        
        assert output.shape == (32, 256)
        assert output.device.type == 'cuda'
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_backward(self):
        """Test backward pass"""
        layer = CKLinear(64, 128).cuda().half()
        input = torch.randn(16, 64, device='cuda', dtype=torch.float16, requires_grad=True)
        
        output = layer(input)
        loss = output.sum()
        loss.backward()
        
        assert input.grad is not None
        assert layer.weight.grad is not None


@pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not available")
class TestCKMLP:
    """Test CKMLP"""
    
    def test_create_mlp(self):
        """Test MLP creation"""
        mlp = CKMLP([128, 256, 512, 256])
        
        assert len(mlp.layers) == 3
    
    def test_forward(self):
        """Test forward pass"""
        mlp = CKMLP([128, 256, 128]).half()
        input = torch.randn(16, 128, dtype=torch.float16)
        
        output = mlp(input)
        
        assert output.shape == (16, 128)
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_forward_cuda(self):
        """Test forward pass on CUDA"""
        mlp = CKMLP([128, 256, 128]).cuda().half()
        input = torch.randn(16, 128, device='cuda', dtype=torch.float16)
        
        output = mlp(input)
        
        assert output.shape == (16, 128)
        assert output.device.type == 'cuda'
    
    def test_different_activations(self):
        """Test different activation functions"""
        activations = ['relu', 'gelu', 'silu']
        
        for act in activations:
            mlp = CKMLP([64, 128, 64], activation=act).half()
            input = torch.randn(8, 64, dtype=torch.float16)
            
            output = mlp(input)
            assert output.shape == (8, 64)


@pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not available")
class TestAutograd:
    """Test autograd support"""
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_autograd_gemm(self):
        """Test autograd with GEMM"""
        A = torch.randn(64, 64, device='cuda', dtype=torch.float16, requires_grad=True)
        B = torch.randn(64, 64, device='cuda', dtype=torch.float16, requires_grad=True)
        
        C = ck_gemm(A, B)
        loss = C.sum()
        loss.backward()
        
        assert A.grad is not None
        assert B.grad is not None
        assert A.grad.shape == A.shape
        assert B.grad.shape == B.shape
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_training_loop(self):
        """Test training loop"""
        model = CKLinear(64, 32).cuda().half()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        
        for _ in range(5):
            input = torch.randn(16, 64, device='cuda', dtype=torch.float16)
            target = torch.randn(16, 32, device='cuda', dtype=torch.float16)
            
            output = model(input)
            loss = nn.functional.mse_loss(output, target)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        # Should complete without errors


@pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not available")
class TestModelConversion:
    """Test model conversion"""
    
    def test_convert_simple_model(self):
        """Test converting simple model"""
        model = nn.Sequential(
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 128)
        )
        
        model_ck = convert_linear_to_ck(model, inplace=False)
        
        # Count CKLinear layers
        ck_count = sum(1 for m in model_ck.modules() if isinstance(m, CKLinear))
        assert ck_count == 2
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_convert_preserves_weights(self):
        """Test that conversion preserves weights"""
        model = nn.Linear(64, 128).cuda().half()
        
        # Save original weights
        orig_weight = model.weight.data.clone()
        orig_bias = model.bias.data.clone() if model.bias is not None else None
        
        # Convert
        model_ck = convert_linear_to_ck(model, inplace=False)
        
        # Check weights are preserved
        ck_linear = list(model_ck.modules())[0]
        assert torch.allclose(ck_linear.weight.data, orig_weight, rtol=1e-3)
        if orig_bias is not None:
            assert torch.allclose(ck_linear.bias.data, orig_bias, rtol=1e-3)


@pytest.mark.skipif(not HAS_TORCH or not torch.cuda.is_available(), 
                    reason="PyTorch or CUDA not available")
class TestBenchmark:
    """Test benchmarking"""
    
    def test_benchmark_vs_pytorch(self):
        """Test benchmark vs PyTorch"""
        results = benchmark_vs_pytorch(
            M=256, N=256, K=256,
            num_warmup=2,
            num_iterations=5,
            dtype=torch.float16
        )
        
        assert 'ck_tile_gflops' in results
        assert 'pytorch_gflops' in results
        assert 'speedup' in results
        assert results['ck_tile_gflops'] > 0
        assert results['pytorch_gflops'] > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

