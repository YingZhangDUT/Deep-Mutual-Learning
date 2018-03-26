#!/bin/bash
#
# This script performs the following operations:
# Evaluate the MobileNets trained with DML on Market-1501
#
# Usage:
# cd Deep-Mutual-Learning
# ./scripts/evaluate_dml_mobilenet_on_market.sh


# Where the TFRecords are saved to.
DATASET_DIR=/path/to/market-1501/tfrecords

# Where the checkpoints are saved to.
DATASET_NAME=market1501
SAVE_NAME=market1501_dml_mobilenet
CKPT_DIR=${SAVE_NAME}/checkpoint

# Where the results will be saved to.
RESULT_DIR=${SAVE_NAME}/results

# Model setting
MODEL_NAME=mobilenet_v1,mobilenet_v1

# Run evaluation.
for split in query bounding_box_test gt_bbox; do
    python eval_image_classifier.py \
        --dataset_name=${DATASET_NAME}\
        --split_name="$split" \
        --dataset_dir=${DATASET_DIR} \
        --checkpoint_dir=${CKPT_DIR} \
        --eval_dir=${RESULT_DIR} \
        --model_name=${MODEL_NAME} \
        --preprocessing_name=reid \
        --num_classes=751 \
        --batch_size=1 \
        --num_networks=2
done
