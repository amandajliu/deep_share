#!/bin/bash

iters=$1
stepsize=$2
base_lr=$3
model=$4
last_low_rank=$5
max_rounds=$6
max_stall=$7
split_thresh=$8

set -x
set -e

export PYTHONUNBUFFERED="True"

LOG="../logs/train_branch_mnist_${model}_${last_low_rank}_${max_rounds}_${max_stall}_${split_thresh}.txt.`date +'%Y-%m-%d_%H-%M-%S'`"
exec &> >(tee -a "$LOG")
echo Logging output to "$LOG"

cd ../..

time ./tools/train_cls.py \
    --traindb mnist_train \
    --valdb mnist_test \
    --iters ${iters} \
    --base_lr ${base_lr} \
    --clip_gradients 20 \
    --loss Sigmoid \
    --model ${model} \
    --last_low_rank ${last_low_rank} \
    --use_svd \
    --exp mnist_${model}_${last_low_rank}_${max_rounds}_${max_stall}_${split_thresh} \
    --max_rounds ${max_rounds} \
    --stepsize ${stepsize} \
    --share_basis \
    --use_bn \
    --max_stall ${max_stall} \
    --split_thresh ${split_thresh}
