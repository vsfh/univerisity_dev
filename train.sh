#!/bin/bash
# Train unified_siglip_test.py with accelerate on 3 GPUs

cd /data/feihong/univerisity_dev

export CUDA_VISIBLE_DEVICES=0,1,2

accelerate launch \
    --multi_gpu \
    --num_processes 3 \
    --mixed_precision fp16 \
    --gradient_accumulation_steps 3 \
    unified_siglip_test.py
