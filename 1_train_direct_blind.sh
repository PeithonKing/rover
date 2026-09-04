#!/bin/bash
# Script 1: BLIND + DIRECT (10-value motor control) + PPO

nohup .venv/bin/python train_ppo.py \
    --workers 10 \
    --frames-per-batch 16380 \
    --vision-mode blind \
    --control-mode direct \
    --total-timesteps 10000000 \
    > log_1_direct_blind.txt 2>&1 &

echo "Started Script 1: Blind Direct PPO. Logs are tailing into log_1_direct_blind.txt"
