#!/usr/bin/env bash
set -euo pipefail

CUDA_VISIBLE_DEVICES=1 python train_trans.py \
  --data-protocol unified_siglip_supp \
  --eval-split val \
  --checkpoint /media/data1/feihong/ckpt/trans_geo_unified_siglip_fair \
  --savename trans_geo_unified_siglip_fair \
  --max-epoch 20 \
  --batch-size 16 \
  --grad-accumulation-steps 4 \
  --lr 5e-5 \
  --backbone-lr 5e-5 \
  --weight-decay 0.01 \
  --warmup-epochs 0 \
  --min-lr 1e-10 \
  --label-smoothing 0.1 \
  --retrieval-only-epochs 8 \
  --grounding-ramp-epochs 3 \
  --memory-queue-size 512 \
  --memory-queue-start-epoch 10 \
  --heights 150 200 250 300 \
  --angles 0 45 90 135 180 225 270 315 \
  --no-identity-balanced-batches \
  --asam-rho 0 \
  --amp \
  --amp-dtype bf16 \
  --num-workers 8

CUDA_VISIBLE_DEVICES=1 python test_unify.py \
  --model-types trans_geo \
  --checkpoint /media/data1/feihong/ckpt/trans_geo_unified_siglip_fair/last.pth \
  --unify-score-mode global \
  --candidate-size 100 \
  --test-crop-ratio 1 \
  --batch-size 32 \
  --num-workers 8 \
  --output-dir eval_results/test_unify_compare \
  --output-suffix trans_geo_unified_siglip_fair_last

CUDA_VISIBLE_DEVICES=1 python test_unify.py \
  --model-types unify_geo \
  --checkpoint /media/data1/feihong/ckpt/unify_compare/unify_geo/last.pth \
  --unify-score-mode global \
  --candidate-size 100 \
  --test-crop-ratio 1 \
  --batch-size 32 \
  --num-workers 8 \
  --output-dir eval_results/test_unify_compare \
  --output-suffix unify_geo_last
