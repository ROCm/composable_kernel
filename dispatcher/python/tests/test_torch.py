"""
Unit tests for PyTorch integration
"""

import unittest

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


def has_cuda():
    """Check if CUDA is available"""
    return HAS_TORCH and torch.cuda.is_available()


@unittest.skipUnless(HAS_TORCH, "PyTorch not available")
class TestTorchGEMM(unittest.TestCase):
    """Test PyTorch GEMM operations"""

    @unittest.skipUnless(has_cuda(), "CUDA not available")
    def test_ck_gemm_cuda(self):
        """Test CK GEMM on CUDA"""
        A = torch.randn(128, 128, device="cuda", dtype=torch.float16)
        B = torch.randn(128, 128, device="cuda", dtype=torch.float16)

        C = ck_gemm(A, B)

        self.assertEqual(C.shape, (128, 128))
        self.assertEqual(C.device.type, "cuda")
        self.assertEqual(C.dtype, torch.float16)

    def test_ck_gemm_cpu(self):
        """Test CK GEMM on CPU (fallback)"""
        A = torch.randn(64, 64, dtype=torch.float16)
        B = torch.randn(64, 64, dtype=torch.float16)

        C = ck_gemm(A, B)

        self.assertEqual(C.shape, (64, 64))

    @unittest.skipUnless(has_cuda(), "CUDA not available")
    def test_ck_gemm_correctness(self):
        """Test CK GEMM correctness"""
        A = torch.randn(64, 64, device="cuda", dtype=torch.float16)
        B = torch.randn(64, 64, device="cuda", dtype=torch.float16)

        C_ck = ck_gemm(A, B)
        C_pt = torch.matmul(A, B)

        max_diff = torch.max(torch.abs(C_ck - C_pt)).item()
        self.assertLess(max_diff, 0.1)


@unittest.skipUnless(HAS_TORCH, "PyTorch not available")
class TestCKLinear(unittest.TestCase):
    """Test CKLinear layer"""

    def test_create_layer(self):
        """Test layer creation"""
        layer = CKLinear(128, 256)

        self.assertEqual(layer.in_features, 128)
        self.assertEqual(layer.out_features, 256)
        self.assertEqual(layer.weight.shape, (256, 128))

    def test_forward_cpu(self):
        """Test forward pass on CPU"""
        layer = CKLinear(128, 256).half()
        input_tensor = torch.randn(32, 128, dtype=torch.float16)

        output = layer(input_tensor)

        self.assertEqual(output.shape, (32, 256))

    @unittest.skipUnless(has_cuda(), "CUDA not available")
    def test_forward_cuda(self):
        """Test forward pass on CUDA"""
        layer = CKLinear(128, 256).cuda().half()
        input_tensor = torch.randn(32, 128, device="cuda", dtype=torch.float16)

        output = layer(input_tensor)

        self.assertEqual(output.shape, (32, 256))
        self.assertEqual(output.device.type, "cuda")

    @unittest.skipUnless(has_cuda(), "CUDA not available")
    def test_backward(self):
        """Test backward pass"""
        layer = CKLinear(64, 128).cuda().half()
        input_tensor = torch.randn(
            16, 64, device="cuda", dtype=torch.float16, requires_grad=True
        )

        output = layer(input_tensor)
        loss = output.sum()
        loss.backward()

        self.assertIsNotNone(input_tensor.grad)
        self.assertIsNotNone(layer.weight.grad)


@unittest.skipUnless(HAS_TORCH, "PyTorch not available")
class TestCKMLP(unittest.TestCase):
    """Test CKMLP"""

    def test_create_mlp(self):
        """Test MLP creation"""
        mlp = CKMLP([128, 256, 512, 256])

        self.assertEqual(len(mlp.layers), 3)

    def test_forward(self):
        """Test forward pass"""
        mlp = CKMLP([128, 256, 128]).half()
        input_tensor = torch.randn(16, 128, dtype=torch.float16)

        output = mlp(input_tensor)

        self.assertEqual(output.shape, (16, 128))

    @unittest.skipUnless(has_cuda(), "CUDA not available")
    def test_forward_cuda(self):
        """Test forward pass on CUDA"""
        mlp = CKMLP([128, 256, 128]).cuda().half()
        input_tensor = torch.randn(16, 128, device="cuda", dtype=torch.float16)

        output = mlp(input_tensor)

        self.assertEqual(output.shape, (16, 128))
        self.assertEqual(output.device.type, "cuda")

    def test_different_activations(self):
        """Test different activation functions"""
        activations = ["relu", "gelu", "silu"]

        for act in activations:
            mlp = CKMLP([64, 128, 64], activation=act).half()
            input_tensor = torch.randn(8, 64, dtype=torch.float16)

            output = mlp(input_tensor)
            self.assertEqual(output.shape, (8, 64))


@unittest.skipUnless(HAS_TORCH, "PyTorch not available")
class TestAutograd(unittest.TestCase):
    """Test autograd support"""

    @unittest.skipUnless(has_cuda(), "CUDA not available")
    def test_autograd_gemm(self):
        """Test autograd with GEMM"""
        A = torch.randn(64, 64, device="cuda", dtype=torch.float16, requires_grad=True)
        B = torch.randn(64, 64, device="cuda", dtype=torch.float16, requires_grad=True)

        C = ck_gemm(A, B)
        loss = C.sum()
        loss.backward()

        self.assertIsNotNone(A.grad)
        self.assertIsNotNone(B.grad)
        self.assertEqual(A.grad.shape, A.shape)
        self.assertEqual(B.grad.shape, B.shape)

    @unittest.skipUnless(has_cuda(), "CUDA not available")
    def test_training_loop(self):
        """Test training loop"""
        model = CKLinear(64, 32).cuda().half()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

        for _ in range(5):
            input_tensor = torch.randn(16, 64, device="cuda", dtype=torch.float16)
            target = torch.randn(16, 32, device="cuda", dtype=torch.float16)

            output = model(input_tensor)
            loss = nn.functional.mse_loss(output, target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Should complete without errors


@unittest.skipUnless(HAS_TORCH, "PyTorch not available")
class TestModelConversion(unittest.TestCase):
    """Test model conversion"""

    def test_convert_simple_model(self):
        """Test converting simple model"""
        model = nn.Sequential(nn.Linear(128, 256), nn.ReLU(), nn.Linear(256, 128))

        model_ck = convert_linear_to_ck(model, inplace=False)

        # Count CKLinear layers
        ck_count = sum(1 for m in model_ck.modules() if isinstance(m, CKLinear))
        self.assertEqual(ck_count, 2)

    @unittest.skipUnless(has_cuda(), "CUDA not available")
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
        self.assertTrue(torch.allclose(ck_linear.weight.data, orig_weight, rtol=1e-3))
        if orig_bias is not None:
            self.assertTrue(torch.allclose(ck_linear.bias.data, orig_bias, rtol=1e-3))


@unittest.skipUnless(has_cuda(), "PyTorch or CUDA not available")
class TestBenchmark(unittest.TestCase):
    """Test benchmarking"""

    def test_benchmark_vs_pytorch(self):
        """Test benchmark vs PyTorch"""
        results = benchmark_vs_pytorch(
            M=256, N=256, K=256, num_warmup=2, num_iterations=5, dtype=torch.float16
        )

        self.assertIn("ck_tile_gflops", results)
        self.assertIn("pytorch_gflops", results)
        self.assertIn("speedup", results)
        self.assertGreater(results["ck_tile_gflops"], 0)
        self.assertGreater(results["pytorch_gflops"], 0)


if __name__ == "__main__":
    unittest.main()
