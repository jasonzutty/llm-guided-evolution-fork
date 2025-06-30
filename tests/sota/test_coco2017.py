import subprocess

def test_train():
    """
    Runs the YOLO object detection model's training/evaluation script (`test.py`)
    using 'uv run'.

    This function sets up the command to execute `test.py` with a specific
    model configuration. It then runs this command as a subprocess,
    capturing and printing its output.

    Returns:
        str: The standard output from `test.py`, decoded as a string.
             This typically includes training progress and evaluation metrics.

    Raises:
        subprocess.CalledProcessError: If `test.py` exits with an error.
    """
    # Construct the command arguments as a list
    command_args = [
        'uv',
        'run',
        './sota/ultralytics/test.py', # Path to your main training script
        '-network', 'network_v3.yaml', # Model configuration (matching argparse default/type)
    ]

    # Execute the command
    print(f"Running command: {' '.join(command_args)}") # Good for debugging
    output = subprocess.check_output(command_args)
    print("Command output:")
    print(output.decode('utf-8')) # Decode to print as string
    return output

if __name__ == "__main__":
    test_train()