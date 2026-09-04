#!/bin/bash
# Script 1: BLIND + DIRECT (10-value motor control) + PPO

.venv/bin/python train_ppo.py \
    --workers 10 \
    --frames-per-batch 10240 \
    --vision-mode blind \
    --control-mode direct \
    --total-timesteps 10000000
