#!/bin/bash

iters=$1
stepsize=$2
base_lr=$3
model=$4
last_low_rank=$5

set -x
set -e

export PYTHONUNBUFFERED="True"

LOG="../logs/train_baseline_scratch_mnist_${model}_${last_low_rank}.txt.`date +'%Y-%m-%d_%H-%M-%S'`"
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
    --exp mnist_baseline_scratch_${model}_${last_low_rank} \
    --max_rounds 1 \
    --stepsize ${stepsize} \
    --use_bn
