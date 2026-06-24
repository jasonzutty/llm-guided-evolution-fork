"""
SIMPLE ExquisiteNetV2 test - just checks if the model can forward pass on GPU.
No training, no DataLoader workers, no multiprocessing hell.
"""
import os
import sys
import torch
import pytest

# Add the ExquisiteNetV2 directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../sota/ExquisiteNetV2'))

def test_model_forward_pass_on_gpu():
    """
    Dead simple test: Can the model run one forward pass on GPU?
    That's it. No training, no datasets, no workers.
    """
    # Force GPU
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    # Check GPU available
    if not torch.cuda.is_available():
        pytest.skip("No GPU available")

    device = torch.device('cuda')
    print(f"\n✓ Using device: {device}")

    # Import model
    from network import ExquisiteNetV2

    # Create model
    num_classes = 10  # CIFAR-10
    input_channels = 3  # RGB
    model = ExquisiteNetV2(num_classes, input_channels).to(device)

    print(f"✓ Model created with {sum(p.numel() for p in model.parameters())/1e6:.2f}M parameters")

    # Create dummy input (batch_size=4, channels=3, height=32, width=32)
    dummy_input = torch.randn(4, 3, 32, 32).to(device)

    # Forward pass
    model.eval()
    with torch.no_grad():
        output = model(dummy_input)

    # Check output shape
    assert output.shape == (4, num_classes), f"Expected shape (4, 10), got {output.shape}"

    print(f"✓ Forward pass successful! Output shape: {output.shape}")
    print("✓ Test PASSED - Model works on GPU")


def test_model_can_train_one_batch():
    """
    Slightly less simple: Can the model train on one batch?
    Still no DataLoader, no workers, just pure PyTorch.
    """
    # Force GPU
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    if not torch.cuda.is_available():
        pytest.skip("No GPU available")

    device = torch.device('cuda')

    # Import model
    from network import ExquisiteNetV2

    # Create model
    model = ExquisiteNetV2(10, 3).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    criterion = torch.nn.CrossEntropyLoss()

    # Create one batch of fake data
    batch_size = 8
    images = torch.randn(batch_size, 3, 32, 32).to(device)
    labels = torch.randint(0, 10, (batch_size,)).to(device)

    # Training step
    model.train()
    optimizer.zero_grad()
    outputs = model(images)
    loss = criterion(outputs, labels)
    loss.backward()
    optimizer.step()

    print(f"\n✓ Training step successful! Loss: {loss.item():.4f}")
    print("✓ Test PASSED - Model can train on GPU")


if __name__ == '__main__':
    print("="*80)
    print("Running SIMPLE ExquisiteNetV2 tests...")
    print("="*80)

    test_model_forward_pass_on_gpu()
    test_model_can_train_one_batch()

    print("\n" + "="*80)
    print("ALL TESTS PASSED!")
    print("="*80)
