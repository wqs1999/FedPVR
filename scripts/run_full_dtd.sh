#!/usr/bin/env bash
set -euo pipefail

python federated_main.py \
  --root DATA \
  --trainer GL_SVDMSE_ADAPTIVE_PC \
  --dataset dtd \
  --backbone ViT-B/16 \
  --device_id 0 \
  --num_shots 16 \
  --num_users 10 \
  --seed 1 \
  --beta 0.5 \
  --output_dir output/dtd/protocalfed_full_seed1 \
  TRAINER.GL_SVDMSE.INFER_MODE adaptive \
  TRAINER.GL_SVDMSE.PROTOTYPE_BETA 0.3 \
  OPTIM.ROUND 50
