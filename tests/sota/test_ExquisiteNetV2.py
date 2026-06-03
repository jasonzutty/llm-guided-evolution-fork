import os
import shutil
import subprocess
from pathlib import Path

def test_train(tmp_path):
    output = subprocess.check_output([
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
    ])
    print(output)

if __name__ == '__main__':
    test_train()
