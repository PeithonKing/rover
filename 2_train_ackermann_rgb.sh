#!/bin/bash
# Script 2: RGB CAMERA + ACKERMANN (2-value control) + PPO

nohup .venv/bin/python train_ppo.py \
    --workers 10 \
    --frames-per-batch 16380 \
    --vision-mode rgb \
    --control-mode ackermann \
    --total-timesteps 3000000 \
    > log_2_ackermann_rgb.txt 2>&1 &

echo "Started Script 2: RGB Ackermann PPO. Logs are tailing into log_2_ackermann_rgb.txt"
