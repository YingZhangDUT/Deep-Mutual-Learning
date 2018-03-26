#!/bin/bash


IMAGE_DIR=/home/zhangying/Documents/Dataset/Images/Market-1501
TF_DIR=/home/zhangying/Documents/Dataset/TFRecords/market1501


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
