#!/bin/sh
DEVICE_IDX="3"
PYTHON_FILE="YOUR/FILE/DIR"
RUN_FILE="SAVE/TENSORBOARD/LOG/DIR"
LEARNING_RATE=""
EPOCHS=""
SEEDS=""
trap "exit" INT

for EPOCH in $EPOCHS
do
  for SEED in $SEEDS
  do
    for LR in $LEARNING_RATE
    do
      CUDA_VISIBLE_DEVICES=$DEVICE_IDX python $PYTHON_FILE \
      --run_file $RUN_FILE \
      --n_epochs $EPOCH \
      --seed $SEED --lr $LR --weight_decay 1e-3 --d_prob 0.1
    done
  done
done