#!/bin/bash
# Experiment 1: Direct 10-Motor Control, Blind, Flat Terrain

python train_ppo.py \
    --workers 6 \
    --mini-batch-size 512 \
    --epochs 4 \
    --control-mode direct \
    --vision-mode blind \
    --terrain flat \
    --checkpoint-dir checkpoints/exp1_direct_blind

