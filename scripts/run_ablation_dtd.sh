#!/usr/bin/env bash
set -euo pipefail

python federated_main.py --root DATA --trainer GL_SVDMSE --dataset dtd --backbone ViT-B/16 --device_id 0 --num_shots 16 --num_users 10 --seed 1 --beta 0.5 --output_dir output/dtd/fedpha_baseline_seed1 OPTIM.ROUND 50

python federated_main.py --root DATA --trainer GL_SVDMSE_ADAPTIVE --dataset dtd --backbone ViT-B/16 --device_id 0 --num_shots 16 --num_users 10 --seed 1 --beta 0.5 --output_dir output/dtd/wo_vpgc_af_only_seed1 TRAINER.GL_SVDMSE.INFER_MODE adaptive OPTIM.ROUND 50

python federated_main.py --root DATA --trainer GL_SVDMSE_PC --dataset dtd --backbone ViT-B/16 --device_id 0 --num_shots 16 --num_users 10 --seed 1 --beta 0.5 --output_dir output/dtd/wo_af_vpgc_only_seed1 TRAINER.GL_SVDMSE.PROTOTYPE_BETA 0.3 OPTIM.ROUND 50

python federated_main.py --root DATA --trainer GL_SVDMSE_ADAPTIVE_PC --dataset dtd --backbone ViT-B/16 --device_id 0 --num_shots 16 --num_users 10 --seed 1 --beta 0.5 --output_dir output/dtd/full_seed1 TRAINER.GL_SVDMSE.INFER_MODE adaptive TRAINER.GL_SVDMSE.PROTOTYPE_BETA 0.3 OPTIM.ROUND 50
