"""
Unit tests for C++ bindings

Tests the low-level C++ Python bindings directly to ensure proper integration.
"""

import pytest

# Try to import C++ extension
try:
    import _ck_dispatcher_cpp as cpp

    HAS_CPP = True
except ImportError:
    HAS_CPP = False
    pytest.skip("C++ extension not available", allow_module_level=True)


class TestEnums:
    """Test enum bindings"""

    def test_datatype_enum(self):
        """Test DataType enum"""
        assert hasattr(cpp, "DataType")
        assert hasattr(cpp.DataType, "FP16")
        assert hasattr(cpp.DataType, "FP32")
        assert hasattr(cpp.DataType, "BF16")
        assert hasattr(cpp.DataType, "INT8")

    def test_layout_enum(self):
        """Test LayoutTag enum"""
        assert hasattr(cpp, "LayoutTag")
        assert hasattr(cpp.LayoutTag, "RowMajor")
        assert hasattr(cpp.LayoutTag, "ColMajor")

    def test_pipeline_enum(self):
        """Test Pipeline enum"""
        assert hasattr(cpp, "Pipeline")
        assert hasattr(cpp.Pipeline, "Mem")
        assert hasattr(cpp.Pipeline, "CompV4")

    def test_scheduler_enum(self):
        """Test Scheduler enum"""
        assert hasattr(cpp, "Scheduler")
        assert hasattr(cpp.Scheduler, "Intrawave")
        assert hasattr(cpp.Scheduler, "Interwave")

    def test_epilogue_enum(self):
        """Test Epilogue enum"""
        assert hasattr(cpp, "Epilogue")
        assert hasattr(cpp.Epilogue, "CShuffle")


class TestProblem:
    """Test Problem class bindings"""

    def test_problem_construction(self):
        """Test Problem construction"""
        problem = cpp.Problem()
        assert problem.M == 0
        assert problem.N == 0
        assert problem.K == 0

        problem2 = cpp.Problem(1024, 2048, 512)
        assert problem2.M == 1024
        assert problem2.N == 2048
        assert problem2.K == 512

    def test_problem_attributes(self):
        """Test Problem attributes"""
        problem = cpp.Problem(100, 200, 300)
        assert problem.k_batch == 1
        assert problem.smem_budget == 0
        assert not problem.prefer_persistent
        assert not problem.enable_validation

    def test_problem_is_valid(self):
        """Test Problem validation"""
        problem1 = cpp.Problem(100, 200, 300)
        assert problem1.is_valid()

        problem2 = cpp.Problem(0, 200, 300)
        assert not problem2.is_valid()

    def test_problem_num_ops(self):
        """Test Problem num_ops calculation"""
        problem = cpp.Problem(100, 200, 50)
        expected_ops = 2 * 100 * 200 * 50  # 2 * M * N * K
        assert problem.num_ops() == expected_ops

    def test_problem_repr(self):
        """Test Problem string representation"""
        problem = cpp.Problem(128, 256, 64)
        repr_str = repr(problem)
        assert "Problem" in repr_str
        assert "128" in repr_str
        assert "256" in repr_str
        assert "64" in repr_str


class TestKernelKey:
    """Test KernelKey class bindings"""

    def test_signature_construction(self):
        """Test Signature construction"""
        sig = cpp.Signature()
        assert sig.dtype_a == cpp.DataType.FP16  # or UNKNOWN, depending on defaults
        assert sig.split_k == 1 or sig.split_k == 0

    def test_signature_attributes(self):
        """Test Signature attributes"""
        sig = cpp.Signature()
        sig.dtype_a = cpp.DataType.FP16
        sig.dtype_b = cpp.DataType.FP16
        sig.dtype_c = cpp.DataType.FP16
        sig.dtype_acc = cpp.DataType.FP32
        sig.layout_a = cpp.LayoutTag.RowMajor
        sig.layout_b = cpp.LayoutTag.ColMajor
        sig.layout_c = cpp.LayoutTag.RowMajor
        sig.elementwise_op = "PassThrough"
        sig.num_d_tensors = 0
        sig.structured_sparsity = False

        assert sig.dtype_a == cpp.DataType.FP16
        assert sig.elementwise_op == "PassThrough"

    def test_tile_shape_construction(self):
        """Test TileShape construction"""
        ts = cpp.TileShape()
        ts.m = 256
        ts.n = 256
        ts.k = 32

        assert ts.m == 256
        assert ts.n == 256
        assert ts.k == 32

    def test_wave_shape_construction(self):
        """Test WaveShape construction"""
        ws = cpp.WaveShape()
        ws.m = 2
        ws.n = 2
        ws.k = 1

        assert ws.m == 2
        assert ws.n == 2
        assert ws.k == 1

    def test_algorithm_construction(self):
        """Test Algorithm construction"""
        algo = cpp.Algorithm()

        algo.tile_shape.m = 256
        algo.tile_shape.n = 256
        algo.tile_shape.k = 32

        algo.wave_shape.m = 2
        algo.wave_shape.n = 2
        algo.wave_shape.k = 1

        algo.warp_tile_shape.m = 32
        algo.warp_tile_shape.n = 32
        algo.warp_tile_shape.k = 16

        algo.pipeline = cpp.Pipeline.CompV4
        algo.scheduler = cpp.Scheduler.Intrawave
        algo.epilogue = cpp.Epilogue.CShuffle
        algo.block_size = 256
        algo.persistent = False

        assert algo.tile_shape.m == 256
        assert algo.pipeline == cpp.Pipeline.CompV4

    def test_kernel_key_construction(self):
        """Test KernelKey construction"""
        key = cpp.KernelKey()

        # Set signature
        key.signature.dtype_a = cpp.DataType.FP16
        key.signature.dtype_b = cpp.DataType.FP16
        key.signature.dtype_c = cpp.DataType.FP16
        key.signature.dtype_acc = cpp.DataType.FP32
        key.signature.elementwise_op = "PassThrough"
        key.signature.num_d_tensors = 0

        # Set algorithm
        key.algorithm.tile_shape.m = 256
        key.algorithm.tile_shape.n = 256
        key.algorithm.tile_shape.k = 32
        key.algorithm.persistent = True

        # Set arch
        key.gfx_arch = "gfx942"

        assert key.gfx_arch == "gfx942"
        assert key.signature.dtype_a == cpp.DataType.FP16

    def test_kernel_key_encode_identifier(self):
        """Test KernelKey identifier encoding"""
        key = cpp.KernelKey()

        key.signature.split_k = 1
        key.signature.elementwise_op = "PassThrough"
        key.signature.num_d_tensors = 0
        key.signature.structured_sparsity = False

        key.algorithm.tile_shape.m = 256
        key.algorithm.tile_shape.n = 256
        key.algorithm.tile_shape.k = 32
        key.algorithm.wave_shape.m = 2
        key.algorithm.wave_shape.n = 2
        key.algorithm.wave_shape.k = 1
        key.algorithm.warp_tile_shape.m = 32
        key.algorithm.warp_tile_shape.n = 32
        key.algorithm.warp_tile_shape.k = 16
        key.algorithm.persistent = True

        identifier = key.encode_identifier()

        assert "256x256x32" in identifier
        assert "2x2x1" in identifier
        assert "32x32x16" in identifier
        assert "persist" in identifier

    def test_kernel_key_equality(self):
        """Test KernelKey equality"""
        key1 = cpp.KernelKey()
        key1.algorithm.tile_shape.m = 256
        key1.algorithm.tile_shape.n = 256
        key1.algorithm.tile_shape.k = 32
        key1.gfx_arch = "gfx942"

        key2 = cpp.KernelKey()
        key2.algorithm.tile_shape.m = 256
        key2.algorithm.tile_shape.n = 256
        key2.algorithm.tile_shape.k = 32
        key2.gfx_arch = "gfx942"

        # Note: Full equality requires all fields to match
        # This is a basic check
        assert key1.gfx_arch == key2.gfx_arch


class TestRegistry:
    """Test Registry class bindings"""

    def test_registry_singleton(self):
        """Test Registry singleton access"""
        registry = cpp.Registry.instance()
        assert registry is not None

        # Should get same instance
        registry2 = cpp.Registry.instance()
        assert registry is registry2

    def test_registry_size(self):
        """Test Registry size"""
        registry = cpp.Registry.instance()
        registry.clear()

        assert registry.size() == 0
        assert len(registry) == 0

    def test_registry_clear(self):
        """Test Registry clear"""
        registry = cpp.Registry.instance()
        registry.clear()
        assert registry.size() == 0

    def test_priority_enum(self):
        """Test Priority enum"""
        assert hasattr(cpp, "Priority")
        assert hasattr(cpp.Priority, "Low")
        assert hasattr(cpp.Priority, "Normal")
        assert hasattr(cpp.Priority, "High")

    def test_registry_repr(self):
        """Test Registry string representation"""
        registry = cpp.Registry.instance()
        registry.clear()

        repr_str = repr(registry)
        assert "Registry" in repr_str
        assert "size=0" in repr_str


class TestDispatcher:
    """Test Dispatcher class bindings"""

    def test_dispatcher_construction(self):
        """Test Dispatcher construction"""
        dispatcher = cpp.Dispatcher()
        assert dispatcher is not None

    def test_dispatcher_with_registry(self):
        """Test Dispatcher with custom registry"""
        registry = cpp.Registry.instance()
        dispatcher = cpp.Dispatcher(registry)
        assert dispatcher is not None

    def test_selection_strategy_enum(self):
        """Test SelectionStrategy enum"""
        assert hasattr(cpp, "SelectionStrategy")
        assert hasattr(cpp.SelectionStrategy, "FirstFit")
        assert hasattr(cpp.SelectionStrategy, "Heuristic")

    def test_dispatcher_set_strategy(self):
        """Test Dispatcher set_strategy"""
        dispatcher = cpp.Dispatcher()
        dispatcher.set_strategy(cpp.SelectionStrategy.FirstFit)
        # Should not raise

    def test_dispatcher_select_kernel(self):
        """Test Dispatcher select_kernel"""
        cpp.Registry.instance().clear()

        dispatcher = cpp.Dispatcher()
        problem = cpp.Problem(512, 512, 512)

        # No kernels registered, should return None
        kernel = dispatcher.select_kernel(problem)
        assert kernel is None

    def test_dispatcher_repr(self):
        """Test Dispatcher string representation"""
        dispatcher = cpp.Dispatcher()
        repr_str = repr(dispatcher)
        assert "Dispatcher" in repr_str


class TestIntegration:
    """Integration tests for complete workflows"""

    def test_kernel_key_creation_and_encoding(self):
        """Test creating a complete kernel key and encoding it"""
        key = cpp.KernelKey()

        # Full signature setup
        key.signature.dtype_a = cpp.DataType.FP16
        key.signature.dtype_b = cpp.DataType.FP16
        key.signature.dtype_c = cpp.DataType.FP16
        key.signature.dtype_acc = cpp.DataType.FP32
        key.signature.layout_a = cpp.LayoutTag.RowMajor
        key.signature.layout_b = cpp.LayoutTag.ColMajor
        key.signature.layout_c = cpp.LayoutTag.RowMajor
        key.signature.transpose_a = False
        key.signature.transpose_b = False
        key.signature.grouped = False
        key.signature.split_k = 1
        key.signature.elementwise_op = "PassThrough"
        key.signature.num_d_tensors = 0
        key.signature.structured_sparsity = False

        # Full algorithm setup
        key.algorithm.tile_shape.m = 256
        key.algorithm.tile_shape.n = 256
        key.algorithm.tile_shape.k = 32
        key.algorithm.wave_shape.m = 2
        key.algorithm.wave_shape.n = 2
        key.algorithm.wave_shape.k = 1
        key.algorithm.warp_tile_shape.m = 32
        key.algorithm.warp_tile_shape.n = 32
        key.algorithm.warp_tile_shape.k = 16
        key.algorithm.pipeline = cpp.Pipeline.CompV4
        key.algorithm.scheduler = cpp.Scheduler.Intrawave
        key.algorithm.epilogue = cpp.Epilogue.CShuffle
        key.algorithm.block_size = 256
        key.algorithm.double_buffer = True
        key.algorithm.persistent = False
        key.algorithm.preshuffle = False
        key.algorithm.transpose_c = False
        key.algorithm.num_wave_groups = 1

        key.gfx_arch = "gfx942"

        # Encode identifier
        identifier = key.encode_identifier()

        # Verify components
        assert "256x256x32" in identifier
        assert "2x2x1" in identifier
        assert "32x32x16" in identifier
        assert "nopers" in identifier  # not persistent

    def test_problem_creation_workflow(self):
        """Test creating and validating problems"""
        # Valid problem
        problem1 = cpp.Problem(1024, 2048, 512)
        assert problem1.is_valid()
        assert problem1.num_ops() == 2 * 1024 * 2048 * 512

        # Invalid problem
        problem2 = cpp.Problem(0, 100, 100)
        assert not problem2.is_valid()

        # Problem with settings
        problem3 = cpp.Problem(512, 512, 512)
        problem3.k_batch = 2
        problem3.prefer_persistent = True
        problem3.enable_validation = True

        assert problem3.k_batch == 2
        assert problem3.prefer_persistent
        assert problem3.enable_validation


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
