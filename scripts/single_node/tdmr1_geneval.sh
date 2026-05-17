#!/bin/bash
# Reinforce SD3.5-Medium with the TDM-R1 objective on the GenEval reward.
# Single-node, 8-GPU recipe (config: geneval_sd3_8gpu_G24_24_4step).
#
# Note: the GenEval reward additionally requires Mask2Former + OpenAI CLIP-ViT-L-14
# weights and mmdet v2.28.2 configs, all of which are installed by
# `bash download_models.sh` (see README, step 2).
set -euo pipefail

CMD_TDMR1="accelerate launch --config_file scripts/accelerate_configs/multi_gpu.yaml --num_processes=8 scripts/train_tdmr1_pub.py"

$CMD_TDMR1 --config config/tdmr1_clean.py:geneval_sd3_8gpu_G24_24_4step \
    --config.num_train_timesteps=2 \
    --config.t_min_dgpo=250 \
    --config.rl_cfg=2.5 \
    --config.ema_update_interval=1 \
    --config.train.tdm_weight=0.3 \
    --config.use_ema_ref=False \
    --config.train.beta=0.001 \
    --config.use_tweight=True \
    --config.train.beta_dpo=1 \
    --config.ema_old_decay_min=0.001 \
    --config.ema_old_decay_max=0.3 \
    --config.train.rl_adam_beta1=0.9 \
    --config.clip_range=1e-3
