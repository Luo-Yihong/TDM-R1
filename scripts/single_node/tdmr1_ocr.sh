#!/bin/bash
# Reinforce SD3.5-Medium with the TDM-R1 objective on the general OCR reward.
# Single-node, 8-GPU recipe (config: general_ocr_sd3_8gpu_G24_4step).
set -euo pipefail

CMD_TDMR1="accelerate launch --config_file scripts/accelerate_configs/multi_gpu.yaml --num_processes=8 scripts/train_tdmr1_pub.py"

$CMD_TDMR1 --config config/tdmr1_clean.py:general_ocr_sd3_8gpu_G24_4step \
    --config.num_train_timesteps=2 \
    --config.t_min_dgpo=300 \
    --config.t_min_dgpo_reward=300 \
    --config.rl_cfg=4.5 \
    --config.ema_update_interval=1 \
    --config.train.tdm_weight=0.3 \
    --config.use_ema_ref=False \
    --config.train.beta=0.001 \
    --config.use_tweight=True \
    --config.train.beta_dpo=10 \
    --config.ema_old_decay_min=0.001 \
    --config.ema_old_decay_max=0.3 \
    --config.train.rl_adam_beta1=0.9 \
    --config.clip_range=1e-3
