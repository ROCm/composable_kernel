"""
Unit tests for core dispatcher functionality
"""

import unittest
import numpy as np

try:
    from ck_tile_dispatcher import (
        Dispatcher,
        Problem,
        DataType,
        gemm,
        batched_gemm,
    )

    HAS_DISPATCHER = True
except ImportError:
    HAS_DISPATCHER = False


@unittest.skipUnless(HAS_DISPATCHER, "ck_tile_dispatcher not available")
class TestDispatcher(unittest.TestCase):
    """Test Dispatcher class"""

    def test_create_dispatcher(self):
        """Test dispatcher creation"""
        dispatcher = Dispatcher()
        self.assertIsNotNone(dispatcher)
        self.assertEqual(dispatcher.gpu_arch, "gfx942")

    def test_register_kernels(self):
        """Test kernel registration"""
        dispatcher = Dispatcher()
        dispatcher.register_kernels("fp16_rcr_essential")

        kernels = dispatcher.get_registered_kernels()
        self.assertIn("fp16_rcr_essential", kernels)

    def test_clear_cache(self):
        """Test cache clearing"""
        dispatcher = Dispatcher()
        dispatcher.register_kernels("fp16_rcr_essential")
        dispatcher.clear_cache()
        # Should not raise


@unittest.skipUnless(HAS_DISPATCHER, "ck_tile_dispatcher not available")
class TestProblem(unittest.TestCase):
    """Test Problem class"""

    def test_create_problem(self):
        """Test problem creation"""
        problem = Problem(M=1024, N=1024, K=1024)
        self.assertEqual(problem.M, 1024)
        self.assertEqual(problem.N, 1024)
        self.assertEqual(problem.K, 1024)

    def test_validate_valid_problem(self):
        """Test validation of valid problem"""
        problem = Problem(M=1024, N=1024, K=1024)
        valid, msg = problem.validate()
        self.assertTrue(valid)
        self.assertEqual(msg, "Valid")

    def test_validate_invalid_problem(self):
        """Test validation of invalid problem"""
        problem = Problem(M=0, N=1024, K=1024)
        valid, msg = problem.validate()
        self.assertFalse(valid)
        self.assertIn("positive", msg.lower())

    def test_problem_with_arrays(self):
        """Test problem with numpy arrays"""
        A = np.random.randn(128, 256).astype(np.float16)
        B = np.random.randn(256, 512).astype(np.float16)
        C = np.zeros((128, 512), dtype=np.float16)

        problem = Problem(
            M=128,
            N=512,
            K=256,
            A=A,
            B=B,
            C=C,
            dtype_a=DataType.FP16,
            dtype_b=DataType.FP16,
            dtype_c=DataType.FP16,
        )

        valid, _ = problem.validate()
        self.assertTrue(valid)


@unittest.skipUnless(HAS_DISPATCHER, "ck_tile_dispatcher not available")
class TestGEMM(unittest.TestCase):
    """Test GEMM operations"""

    def test_simple_gemm(self):
        """Test simple GEMM"""
        M, N, K = 128, 128, 128
        A = np.random.randn(M, K).astype(np.float16)
        B = np.random.randn(K, N).astype(np.float16)

        C = gemm(A, B)

        self.assertEqual(C.shape, (M, N))
        self.assertEqual(C.dtype, np.float16)

    def test_gemm_correctness(self):
        """Test GEMM correctness against NumPy"""
        M, N, K = 64, 64, 64
        A = np.random.randn(M, K).astype(np.float16)
        B = np.random.randn(K, N).astype(np.float16)

        C_ck = gemm(A, B)
        C_ref = A @ B

        # Check relative error
        max_diff = np.max(np.abs(C_ck - C_ref))
        self.assertLess(max_diff, 0.1)  # FP16 tolerance

    def test_gemm_with_scaling(self):
        """Test GEMM with alpha/beta scaling"""
        M, N, K = 64, 64, 64
        A = np.random.randn(M, K).astype(np.float16)
        B = np.random.randn(K, N).astype(np.float16)
        C = np.random.randn(M, N).astype(np.float16)

        alpha, beta = 2.0, 0.5
        C_initial = C.copy()

        C_result = gemm(A, B, C, alpha=alpha, beta=beta)
        C_ref = alpha * (A @ B) + beta * C_initial

        max_diff = np.max(np.abs(C_result - C_ref))
        self.assertLess(max_diff, 0.1)

    def test_gemm_different_sizes(self):
        """Test GEMM with different problem sizes"""
        sizes = [(32, 32, 32), (64, 128, 256), (256, 256, 128)]

        for M, N, K in sizes:
            A = np.random.randn(M, K).astype(np.float16)
            B = np.random.randn(K, N).astype(np.float16)

            C = gemm(A, B)

            self.assertEqual(C.shape, (M, N))

    def test_gemm_dimension_mismatch(self):
        """Test GEMM with dimension mismatch"""
        A = np.random.randn(64, 128).astype(np.float16)
        B = np.random.randn(256, 64).astype(np.float16)  # Wrong K dimension

        with self.assertRaises(ValueError):
            gemm(A, B)


@unittest.skipUnless(HAS_DISPATCHER, "ck_tile_dispatcher not available")
class TestBatchedGEMM(unittest.TestCase):
    """Test batched GEMM operations"""

    def test_batched_gemm(self):
        """Test batched GEMM"""
        batch_size = 4
        M, N, K = 64, 64, 64

        A = np.random.randn(batch_size, M, K).astype(np.float16)
        B = np.random.randn(batch_size, K, N).astype(np.float16)

        C = batched_gemm(A, B)

        self.assertEqual(C.shape, (batch_size, M, N))

    def test_batched_gemm_correctness(self):
        """Test batched GEMM correctness"""
        batch_size = 2
        M, N, K = 32, 32, 32

        A = np.random.randn(batch_size, M, K).astype(np.float16)
        B = np.random.randn(batch_size, K, N).astype(np.float16)

        C = batched_gemm(A, B)

        # Check each batch
        for i in range(batch_size):
            C_ref = A[i] @ B[i]
            max_diff = np.max(np.abs(C[i] - C_ref))
            self.assertLess(max_diff, 0.1)

    def test_batched_gemm_invalid_dims(self):
        """Test batched GEMM with invalid dimensions"""
        A = np.random.randn(64, 64).astype(np.float16)  # 2D instead of 3D
        B = np.random.randn(64, 64).astype(np.float16)

        with self.assertRaises(ValueError):
            batched_gemm(A, B)


@unittest.skipUnless(HAS_DISPATCHER, "ck_tile_dispatcher not available")
class TestDataTypes(unittest.TestCase):
    """Test different data types"""

    def test_fp16(self):
        """Test FP16 data type"""
        A = np.random.randn(64, 64).astype(np.float16)
        B = np.random.randn(64, 64).astype(np.float16)

        C = gemm(A, B)
        self.assertEqual(C.dtype, np.float16)

    def test_fp32(self):
        """Test FP32 data type"""
        A = np.random.randn(64, 64).astype(np.float32)
        B = np.random.randn(64, 64).astype(np.float32)

        C = gemm(A, B)
        self.assertEqual(C.dtype, np.float32)


@unittest.skipUnless(HAS_DISPATCHER, "ck_tile_dispatcher not available")
class TestDispatcherAPI(unittest.TestCase):
    """Test Dispatcher API"""

    def test_dispatcher_gemm(self):
        """Test dispatcher GEMM method"""
        dispatcher = Dispatcher()
        dispatcher.register_kernels("fp16_rcr_essential")

        A = np.random.randn(128, 128).astype(np.float16)
        B = np.random.randn(128, 128).astype(np.float16)

        C = dispatcher.gemm(A, B)

        self.assertEqual(C.shape, (128, 128))

    def test_dispatcher_dispatch(self):
        """Test dispatcher dispatch method"""
        dispatcher = Dispatcher()
        dispatcher.register_kernels("fp16_rcr_essential")

        A = np.random.randn(128, 128).astype(np.float16)
        B = np.random.randn(128, 128).astype(np.float16)
        C = np.zeros((128, 128), dtype=np.float16)

        problem = Problem(
            M=128,
            N=128,
            K=128,
            A=A,
            B=B,
            C=C,
            dtype_a=DataType.FP16,
            dtype_b=DataType.FP16,
            dtype_c=DataType.FP16,
        )

        result = dispatcher.dispatch(problem)

        self.assertTrue(result.success or result.kernel_name == "numpy_reference")


if __name__ == "__main__":
    unittest.main()
