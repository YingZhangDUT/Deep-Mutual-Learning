#!/bin/bash
#
# This script performs the following operations:
# 1. Format the Market-1501 training images with consecutive labels.
# 2. Convert the Market-1501 images into TFRecords.
#
# Usage:
# cd Deep-Mutual-Learning
# ./scripts/format_and_convert_market.sh


# Where the Market-1501 images are saved to.
IMAGE_DIR=/path/to/Market-1501/images

# Where the TFRecord data will be saved to.
TF_DIR=/path/to/market-1501/tfrecords


echo "Building the TFRecords of market1501..."

for split in bounding_box_train bounding_box_test gt_bbox query; do
    echo "Processing ${split} ..."
    python format_and_convert_data.py \
      --image_dir="$IMAGE_DIR/$split" \
      --output_dir=${TF_DIR}  \
      --dataset_name="market1501" \
      --split_name="$split"

done

echo "Finished converting all the splits!"
