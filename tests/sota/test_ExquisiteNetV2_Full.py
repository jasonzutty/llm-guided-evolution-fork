import os
import shutil
import subprocess
from pathlib import Path
import time
import signal
import sys

def test_train_quick(tmp_path):
    """
    Quick smoke test - trains for only a few batches to verify it works.
    Takes ~10-20 seconds instead of hours.
    Uses large batch size and high validation ratio to minimize training batches.

    HANG PREVENTION MEASURES:
    1. num_workers=0 (no multiprocessing deadlocks)
    2. CUDA_VISIBLE_DEVICES set (force GPU)
    3. Timeout with kill (prevent infinite hangs)
    4. OMP_NUM_THREADS=1 (prevent OpenMP conflicts)
    """
    print("\n" + "="*80)
    print("Starting ExquisiteNetV2 QUICK smoke test...")
    print(f"Batch size: 30 (large to reduce number of batches)")
    print(f"Epochs: 1, Validation ratio: 0.9995 (minimal training)")
    print("="*80 + "\n")

    start_time = time.time()

    # ========== CRITICAL HANG FIXES ==========
    # Force GPU usage and prevent common hang causes
    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = '0'  # Use GPU 0
    env['OMP_NUM_THREADS'] = '1'       # Prevent OpenMP threading conflicts
    env['MKL_NUM_THREADS'] = '1'       # Prevent MKL threading conflicts
    env['NUMEXPR_NUM_THREADS'] = '1'   # Prevent NumExpr threading conflicts

    # Set a timeout (5 minutes max)
    TIMEOUT_SECONDS = 300

    # Use Popen to stream output in real-time (shows progress)
    process = subprocess.Popen(
        [
            'uv', 'run', 'sota/ExquisiteNetV2/train.py',
            '-bs', '30',              # Large batch size = fewer batches
            '-network', 'network',
            '-data', 'sota/ExquisiteNetV2/cifar10',
            '-end_lr', '0.1',
            '-seed', '21',
            '-val_r', '0.9995',       # 99.95% for validation, 0.05% for training
            '-save_dir', str(tmp_path),
            '-worker', '0',           # CRITICAL: 0 workers prevents DataLoader hangs
            '-epoch', '1',
            '-imgsz', '32',           # Smaller image size = faster
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,  # Line buffered
        env=env,     # Pass the modified environment
        preexec_fn=os.setsid if hasattr(os, 'setsid') else None  # Create process group for clean kill
    )

    # Stream output line by line with timeout
    output_lines = []
    try:
        start = time.time()
        for line in process.stdout:
            print(line, end='', flush=True)
            output_lines.append(line)

            # Check for timeout
            if time.time() - start > TIMEOUT_SECONDS:
                print(f"\n{'='*80}")
                print(f"ERROR: Test timed out after {TIMEOUT_SECONDS} seconds!")
                print(f"{'='*80}\n")

                # Kill the entire process group
                try:
                    if hasattr(os, 'killpg'):
                        os.killpg(os.getpgid(process.pid), signal.SIGKILL)
                    else:
                        process.kill()
                except:
                    pass

                raise TimeoutError(f"Training hung - exceeded {TIMEOUT_SECONDS} second timeout")

        process.wait()

    except KeyboardInterrupt:
        print("\nTest interrupted by user. Cleaning up...")
        try:
            if hasattr(os, 'killpg'):
                os.killpg(os.getpgid(process.pid), signal.SIGKILL)
            else:
                process.kill()
        except:
            pass
        raise

    if process.returncode != 0:
        print(f"\n{'='*80}")
        print(f"ERROR: Training failed with return code {process.returncode}")
        print(f"{'='*80}\n")
        raise subprocess.CalledProcessError(process.returncode, process.args)

    # Check that training completed successfully
    full_output = ''.join(output_lines)
    assert 'job done' in full_output.lower(), "Training did not complete - 'job done' not found in output"

    elapsed = time.time() - start_time
    print(f"\n{'='*80}")
    print(f"✓ Quick test completed in {elapsed:.1f} seconds ({elapsed/60:.1f} minutes)")
    print(f"{'='*80}\n")


if __name__ == '__main__':
    from pathlib import Path
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        test_train_quick(Path(tmp))
