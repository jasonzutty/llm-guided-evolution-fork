#!/bin/bash
# Make sure to run this file yourself, it is far too large for github actions to handle, submit using bash or sbatch(if you want a record)
set -euo pipefail

echo "Creating directories..."
mkdir -p datasets/coco2017/images/

echo "Downloading annotation zip..."
curl -o datasets/coco2017/coco2017labels.zip https://github.com/ultralytics/assets/releases/download/v0.0.0/coco2017labels.zip

echo "Downloading training images..."
curl -o datasets/coco2017/images/train2017.zip http://images.cocodataset.org/zips/train2017.zip

echo "Downloading validation images..."
curl -o datasets/coco2017/images/val2017.zip http://images.cocodataset.org/zips/val2017.zip

echo "Downloading test images..."
curl -o datasets/coco2017/images/test2017.zip http://images.cocodataset.org/zips/test2017.zip

echo "Unzipping annotation zip..."
unzip datasets/coco2017/coco2017labels.zip -d datasets/coco2017/

echo "Unzipping training images..."
unzip datasets/coco2017/images/train2017.zip -d datasets/coco2017/images/

echo "Unzipping validation images..."
unzip datasets/coco2017/images/val2017.zip -d datasets/coco2017/images/

echo "Unzipping test images..."
unzip datasets/coco2017/images/test2017.zip -d datasets/coco2017/images/

echo "All downloads and extractions complete."