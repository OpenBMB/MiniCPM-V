#!/bin/bash
export CUDA_DEVICE_MAX_CONNECTIONS=1
DIR=`pwd`

GPUS_PER_NODE=8
NNODES=1
NODE_RANK=0
MASTER_ADDR=localhost
MASTER_PORT=6001

SAVE_PATH=/path/to/experiments/MiniCPM-V-FFT
BASE_MODEL=/path/to/pretrained_model/MiniCPM-V
TRAIN_DATASET=/path/to/train.json
VAL_DATASET=/path/to/test.json


DISTRIBUTED_ARGS="
    --nproc_per_node $GPUS_PER_NODE \
    --nnodes $NNODES \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT
"

torchrun $DISTRIBUTED_ARGS finetune_minicpmv.py \
    --model_name_or_path $BASE_MODEL \
    --data_path $TRAIN_DATASET \
    --bf16 True \
    --fix_vit True \
    --output_dir $SAVE_PATH \
    --num_train_epochs 2 \
    --per_device_train_batch_size 12 \
    --per_device_eval_batch_size 8 \
    --gradient_accumulation_steps 4 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 100 \
    --save_total_limit 2 \
    --learning_rate 1e-5 \
    --weight_decay 0.1 \
    --adam_beta2 0.95 \
    --warmup_ratio 0.01 \
    --lr_scheduler_type "cosine" \
    --logging_steps 2 \
    --report_to "tensorboard" \
    --model_max_length 512 \
    --gradient_checkpointing True \
    --deepspeed ds_config_zero2.json
