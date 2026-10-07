import pickle
import os

def load_global_data(global_path="checkpoints", global_file=None):
    print(global_path)
    if not os.path.exists(global_path):
        print("Path does not exist, returning none for checkpoints")
        return None
    global_data = {}

    if global_file is None:
        global_files = sorted(os.listdir(global_path), reverse=True)
        global_file = global_files[0] if global_files else None
    if global_file:
        filepath = os.path.join(global_path, global_file)
        with open(filepath, 'rb') as file:
            global_data = pickle.load(file)
        print(f"Loaded global data from {filepath}")

    return global_data

global_path = "/home/hice1/<username>/scratch/llm-island-migration/first_test/global_data/"

global_data = load_global_data(global_path=global_path)
print(global_data)