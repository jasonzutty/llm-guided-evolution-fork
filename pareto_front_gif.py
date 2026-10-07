import os
from PIL import Image
import glob
import re

# Define the directory containing the PNG files
input_dir = '<test_dir>/pareto_fronts'
output_file = '<test_dir>/pareto_fronts/output.gif'
pattern = os.path.join(input_dir, 'pareto_gen_*.png')

# Print all files in the directory for debugging
try:
    all_files = os.listdir(input_dir)
    print(f"All files in the directory '{input_dir}':")
    for filename in all_files:
        print(filename)
except FileNotFoundError:
    print(f"The directory {input_dir} does not exist.")
    raise

# Gather the list of image files
image_files = glob.glob(pattern)

# Print the pattern and the list of files found for debugging
print(f"Looking for files matching pattern: {pattern}")
print(f"Files found: {image_files}")

# Custom sort function to sort files numerically based on the number in the filename
def numerical_sort(value):
    parts = re.findall(r'\d+', value)
    return int(parts[-1]) if parts else float('inf')

# Sort the files numerically
image_files = sorted(image_files, key=numerical_sort)

# Ensure there are images to be processed
if not image_files:
    raise ValueError(f"No images found in {input_dir} matching pattern {pattern}")

# Load all images into a list
images = [Image.open(image_file) for image_file in image_files]

# Save the images as a GIF
# with image frames lasting 200 milliseconds
images[0].save(output_file, save_all=True, append_images=images[1:], duration=500, loop=0)

print(f"GIF created successfully: {output_file}")