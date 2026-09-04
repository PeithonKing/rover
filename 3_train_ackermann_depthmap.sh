#!/bin/bash
# Script 3: DEPTHMAP CAMERA + ACKERMANN (2-value control) + PPO

.venv/bin/python train_ppo.py \
    --workers 10 \
    --frames-per-batch 10240 \
    --vision-mode depthmap \
    --control-mode ackermann \
    --total-timesteps 3000000
