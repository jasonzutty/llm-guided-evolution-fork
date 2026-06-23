import os
import shutil
import subprocess
from pathlib import Path
import time

def test_train_quick(tmp_path):
    """
    Quick smoke test - trains for only a few batches to verify it works.
    Takes ~10-20 seconds instead of hours.
    Uses large batch size and high validation ratio to minimize training batches.
    """
    print("\n" + "="*80)
    print("Starting ExquisiteNetV2 QUICK smoke test...")
    print(f"Batch size: 1000 (very large to reduce number of batches)")
    print(f"Epochs: 1, Validation ratio: 0.95 (only 5% for training)")
    print("="*80 + "\n")

    start_time = time.time()

    # Use Popen to stream output in real-time (shows progress)
    process = subprocess.Popen(
        [
            'uv', 'run', 'sota/ExquisiteNetV2/train.py',
            '-bs', '30',          # HUGE batch size = fewer batches
            '-network', 'network',
            '-data', 'sota/ExquisiteNetV2/cifar10',
            '-end_lr', '0.1',
            '-seed', '21',
            '-val_r', '0.9995',       # Use 95% for validation, only 5% for training = 2,500 images = 3 batches
            '-save_dir', str(tmp_path),
            '-worker', '1',
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
    print(f"Quick test completed in {elapsed:.1f} seconds ({elapsed/60:.1f} minutes)")
    print(f"{'='*80}\n")


if __name__ == '__main__':
    from pathlib import Path
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        test_train_quick(Path(tmp))
