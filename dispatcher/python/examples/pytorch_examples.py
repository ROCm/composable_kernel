"""
PyTorch integration examples for CK Tile Dispatcher
"""

import torch
import torch.nn as nn
from ck_tile_dispatcher import (
    ck_gemm, 
    CKLinear, 
    CKMLP,
    convert_linear_to_ck,
    benchmark_vs_pytorch
)


def example_1_basic_torch_gemm():
    """Example 1: Basic PyTorch GEMM"""
    print("=" * 80)
    print("Example 1: Basic PyTorch GEMM")
    print("=" * 80)
    
    if not torch.cuda.is_available():
        print("CUDA not available, skipping example")
        return
    
    # Create tensors
    A = torch.randn(1024, 1024, device='cuda', dtype=torch.float16)
    B = torch.randn(1024, 1024, device='cuda', dtype=torch.float16)
    
    # CK Tile GEMM
    C = ck_gemm(A, B)
    
    print(f"✓ Computed C = A @ B using CK Tile")
    print(f"  A shape: {A.shape}")
    print(f"  B shape: {B.shape}")
    print(f"  C shape: {C.shape}")
    print()


def example_2_ck_linear_layer():
    """Example 2: CK Linear Layer"""
    print("=" * 80)
    print("Example 2: CK Linear Layer")
    print("=" * 80)
    
    if not torch.cuda.is_available():
        print("CUDA not available, skipping example")
        return
    
    # Create layer
    layer = CKLinear(1024, 2048).cuda().half()
    
    # Forward pass
    input = torch.randn(32, 1024, device='cuda', dtype=torch.float16)
    output = layer(input)
    
    print(f"✓ CKLinear layer")
    print(f"  Input shape: {input.shape}")
    print(f"  Output shape: {output.shape}")
    print(f"  Parameters: {sum(p.numel() for p in layer.parameters()):,}")
    print()


def example_3_ck_mlp():
    """Example 3: CK MLP"""
    print("=" * 80)
    print("Example 3: CK MLP")
    print("=" * 80)
    
    if not torch.cuda.is_available():
        print("CUDA not available, skipping example")
        return
    
    # Create MLP
    mlp = CKMLP([1024, 2048, 4096, 2048], activation='gelu').cuda().half()
    
    # Forward pass
    input = torch.randn(32, 1024, device='cuda', dtype=torch.float16)
    output = mlp(input)
    
    print(f"✓ CKMLP")
    print(f"  Input shape: {input.shape}")
    print(f"  Output shape: {output.shape}")
    print(f"  Layers: {len(mlp.layers)}")
    print(f"  Parameters: {sum(p.numel() for p in mlp.parameters()):,}")
    print()


def example_4_autograd():
    """Example 4: Autograd Support"""
    print("=" * 80)
    print("Example 4: Autograd Support")
    print("=" * 80)
    
    if not torch.cuda.is_available():
        print("CUDA not available, skipping example")
        return
    
    # Create tensors with gradients
    A = torch.randn(512, 512, device='cuda', dtype=torch.float16, requires_grad=True)
    B = torch.randn(512, 512, device='cuda', dtype=torch.float16, requires_grad=True)
    
    # Forward pass
    C = ck_gemm(A, B)
    loss = C.sum()
    
    # Backward pass
    loss.backward()
    
    print(f"✓ Autograd support")
    print(f"  Forward: C = A @ B")
    print(f"  Loss: {loss.item():.4f}")
    print(f"  A.grad shape: {A.grad.shape}")
    print(f"  B.grad shape: {B.grad.shape}")
    print()


def example_5_training_loop():
    """Example 5: Training Loop"""
    print("=" * 80)
    print("Example 5: Training Loop")
    print("=" * 80)
    
    if not torch.cuda.is_available():
        print("CUDA not available, skipping example")
        return
    
    # Create model
    model = CKLinear(128, 64).cuda().half()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # Training loop
    num_epochs = 5
    for epoch in range(num_epochs):
        # Dummy data
        input = torch.randn(32, 128, device='cuda', dtype=torch.float16)
        target = torch.randn(32, 64, device='cuda', dtype=torch.float16)
        
        # Forward
        output = model(input)
        loss = nn.functional.mse_loss(output, target)
        
        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        print(f"  Epoch {epoch+1}/{num_epochs}, Loss: {loss.item():.4f}")
    
    print("✓ Training complete")
    print()


def example_6_model_conversion():
    """Example 6: Model Conversion"""
    print("=" * 80)
    print("Example 6: Model Conversion")
    print("=" * 80)
    
    if not torch.cuda.is_available():
        print("CUDA not available, skipping example")
        return
    
    # Create standard PyTorch model
    model = nn.Sequential(
        nn.Linear(1024, 2048),
        nn.ReLU(),
        nn.Linear(2048, 1024),
        nn.ReLU(),
        nn.Linear(1024, 512)
    ).cuda().half()
    
    print(f"Original model:")
    print(f"  Linear layers: {sum(1 for m in model.modules() if isinstance(m, nn.Linear))}")
    
    # Convert to CK Tile
    model_ck = convert_linear_to_ck(model, inplace=False)
    
    print(f"Converted model:")
    print(f"  CKLinear layers: {sum(1 for m in model_ck.modules() if isinstance(m, CKLinear))}")
    
    # Test forward pass
    input = torch.randn(16, 1024, device='cuda', dtype=torch.float16)
    output_orig = model(input)
    output_ck = model_ck(input)
    
    # Check difference
    max_diff = torch.max(torch.abs(output_orig - output_ck)).item()
    print(f"✓ Conversion complete")
    print(f"  Max difference: {max_diff:.2e}")
    print()


def example_7_benchmark():
    """Example 7: Benchmark vs PyTorch"""
    print("=" * 80)
    print("Example 7: Benchmark vs PyTorch")
    print("=" * 80)
    
    if not torch.cuda.is_available():
        print("CUDA not available, skipping example")
        return
    
    # Run benchmark
    results = benchmark_vs_pytorch(
        M=2048, N=2048, K=2048,
        num_warmup=10,
        num_iterations=100,
        dtype=torch.float16
    )
    
    if results:
        print(f"✓ Benchmark results:")
        print(f"  Problem size: {results['problem_size']}")
        print(f"  CK Tile: {results['ck_tile_gflops']:.2f} GFLOPS ({results['ck_tile_time_ms']:.3f} ms)")
        print(f"  PyTorch: {results['pytorch_gflops']:.2f} GFLOPS ({results['pytorch_time_ms']:.3f} ms)")
        print(f"  Speedup: {results['speedup']:.2f}x")
        print(f"  Max diff: {results['max_diff']:.2e}")
    print()


def example_8_mixed_precision():
    """Example 8: Mixed Precision Training"""
    print("=" * 80)
    print("Example 8: Mixed Precision Training")
    print("=" * 80)
    
    if not torch.cuda.is_available():
        print("CUDA not available, skipping example")
        return
    
    # Create model
    model = CKMLP([512, 1024, 512]).cuda()
    
    # Use automatic mixed precision
    scaler = torch.cuda.amp.GradScaler()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # Training step
    for step in range(5):
        input = torch.randn(32, 512, device='cuda')
        target = torch.randn(32, 512, device='cuda')
        
        optimizer.zero_grad()
        
        # Forward with autocast
        with torch.cuda.amp.autocast():
            output = model(input)
            loss = nn.functional.mse_loss(output, target)
        
        # Backward with gradient scaling
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        print(f"  Step {step+1}, Loss: {loss.item():.4f}")
    
    print("✓ Mixed precision training complete")
    print()


def main():
    """Run all examples"""
    examples = [
        example_1_basic_torch_gemm,
        example_2_ck_linear_layer,
        example_3_ck_mlp,
        example_4_autograd,
        example_5_training_loop,
        example_6_model_conversion,
        example_7_benchmark,
        example_8_mixed_precision,
    ]
    
    for example in examples:
        try:
            example()
        except Exception as e:
            print(f"✗ Example failed: {e}")
            import traceback
            traceback.print_exc()
            print()


if __name__ == "__main__":
    main()

