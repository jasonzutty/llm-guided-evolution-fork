import os
import shutil
import subprocess
from pathlib import Path
import time

def test_train(tmp_path):
    print("\n" + "="*80)
    print("Starting ExquisiteNetV2 training test...")
    print(f"This test trains a real neural network and takes 10+ minutes")
    print(f"Batch size: 216, Epochs: 1, Image size: 32x32")
    print(f"Training on ~40,000 CIFAR-10 images (185 batches)")
    print("="*80 + "\n")

    start_time = time.time()

    # Use Popen to stream output in real-time (shows progress)
    process = subprocess.Popen(
        [
            'uv', 'run', 'sota/ExquisiteNetV2/train.py',
            '-bs', '216',
            '-network', 'network',
            '-data', 'sota/ExquisiteNetV2/cifar10',
            '-end_lr', '0.001',
            '-seed', '21',
            '-val_r', '0.2',
            '-save_dir', str(tmp_path),
            '-worker', '0',
            '-epoch', '1',
            '-imgsz', '32',
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,  # Line buffered
    )

    # Stream output line by line (shows progress bar from training script)
    output_lines = []
    for line in process.stdout:
        print(line, end='', flush=True)
        output_lines.append(line)

    process.wait()

    if process.returncode != 0:
        raise subprocess.CalledProcessError(process.returncode, process.args)

    # Check that training completed successfully
    full_output = ''.join(output_lines)
    assert 'job done' in full_output.lower(), "Training did not complete - 'job done' not found in output"

    elapsed = time.time() - start_time
    print(f"\n{'='*80}")
    print(f"Training completed in {elapsed:.1f} seconds ({elapsed/60:.1f} minutes)")
    print(f"{'='*80}\n")

if __name__ == '__main__':
    test_train()
