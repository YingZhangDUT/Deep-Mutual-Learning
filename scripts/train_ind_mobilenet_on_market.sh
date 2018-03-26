#!/bin/bash
#
# This script performs the following operations:
# Training 1 MobileNet independently on Market-1501
#
# Usage:
# cd Deep-Mutual-Learning
# ./scripts/train_ind_mobilenet_on_market.sh


# Where the TFRecords are saved to.
DATASET_DIR=/path/to/market-1501/tfrecords

# Where the checkpoint and logs will be saved to.
DATASET_NAME=market1501
SAVE_NAME=market1501_ind_mobilenet
CKPT_DIR=${SAVE_NAME}/checkpoint
LOG_DIR=${SAVE_NAME}/logs

# Model setting
MODEL_NAME=mobilenet_v1
SPLIT_NAME=bounding_box_train

# Run training.
python train_image_classifier.py \
    --dataset_name=${DATASET_NAME}\
    --split_name=${SPLIT_NAME} \
    --dataset_dir=${DATASET_DIR} \
    --checkpoint_dir=${CKPT_DIR} \
    --log_dir=${LOG_DIR} \
    --model_name=${MODEL_NAME} \
    --preprocessing_name=reid \
    --max_number_of_steps=200000 \
    --ckpt_steps=5000 \
    --batch_size=16 \
    --num_classes=751 \
    --optimizer=adam \
    --learning_rate=0.0002 \
    --adam_beta1=0.5 \
    --opt_epsilon=1e-8 \
    --label_smoothing=0.1 \
    --num_networks=1

