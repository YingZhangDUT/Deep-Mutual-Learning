#!/bin/bash
#
# Where the dataset is saved to.
DATASET_DIR=/home/zhangying/Documents/Dataset/TFRecords/market1501

# Where the checkpoint and logs saved to.
DATASET_NAME=market1501
SAVE_NAME=market1501_ind_mobilenet
CKPT_DIR=${SAVE_NAME}/checkpoint
LOG_DIR=${SAVE_NAME}/logs
RESULT_DIR=${SAVE_NAME}/results

MODEL_NAME=mobilenet_v1

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
        --num_networks=1
done
