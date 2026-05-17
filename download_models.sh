#!/bin/bash
# Download all models / weights required by the rewards used in the per-task
# launch scripts under `scripts/single_node/` (ocr / pickscore / imagereward /
# hpsv2 / geneval). Downloads land under /root/models and a few standard cache
# dirs; adjust the env vars below if your filesystem layout differs.
#
# Requires: huggingface-cli (pip install huggingface_hub), wget.
# Optional:  HF_TOKEN if any of the repos require auth (none currently do).
set -euo pipefail

echo "=== TDM-R1 model download (HuggingFace + public mirrors) ==="

export HF_TOKEN="${HF_TOKEN:-}"
MODELS_DIR="${MODELS_DIR:-/root/models}"
mkdir -p "${MODELS_DIR}"

# 1. SD3.5-Medium (base diffusion model)
if [ ! -d "${MODELS_DIR}/stable-diffusion-3.5-medium" ]; then
    echo "[1/8] Downloading stable-diffusion-3.5-medium ..."
    huggingface-cli download stabilityai/stable-diffusion-3.5-medium \
        --local-dir "${MODELS_DIR}/stable-diffusion-3.5-medium"
else
    echo "[1/8] stable-diffusion-3.5-medium already exists, skipping."
fi

# 2. PickScore v1 (for `pickscore` reward)
if [ ! -d "${MODELS_DIR}/PickScore_v1" ]; then
    echo "[2/8] Downloading PickScore_v1 ..."
    huggingface-cli download yuvalkirstain/PickScore_v1 \
        --local-dir "${MODELS_DIR}/PickScore_v1"
else
    echo "[2/8] PickScore_v1 already exists, skipping."
fi

# 3. CLIP-ViT-H-14 processor (PickScore dependency)
if [ ! -d "${MODELS_DIR}/CLIP-ViT-H-14-laion2B-s32B-b79K" ]; then
    echo "[3/8] Downloading CLIP-ViT-H-14-laion2B-s32B-b79K ..."
    huggingface-cli download laion/CLIP-ViT-H-14-laion2B-s32B-b79K \
        --local-dir "${MODELS_DIR}/CLIP-ViT-H-14-laion2B-s32B-b79K"
else
    echo "[3/8] CLIP-ViT-H-14-laion2B-s32B-b79K already exists, skipping."
fi

# 4. HPSv2 checkpoint (for `hpsv2` reward)
if [ ! -d "${MODELS_DIR}/HPSv2" ]; then
    echo "[4/8] Downloading HPSv2 ..."
    huggingface-cli download xswu/HPSv2 \
        --local-dir "${MODELS_DIR}/HPSv2"
else
    echo "[4/8] HPSv2 already exists, skipping."
fi

# 5. ImageReward (for `imagereward` reward).
# `RM.load("ImageReward-v1.0")` will lazily download to ~/.cache/ImageReward/
# on first use. No manual seeding needed; just make sure the network is up
# the first time you run training.
echo "[5/8] ImageReward weights are auto-downloaded by RM.load() on first use."

# 6. PaddleOCR det/rec/cls models (for `ocr` reward)
PADDLEOCR_DIR="${PADDLEOCR_DIR:-/root/.paddleocr/whl}"
if [ ! -d "${PADDLEOCR_DIR}/det/en/en_PP-OCRv3_det_infer" ]; then
    echo "[6/8] Downloading PaddleOCR det model ..."
    mkdir -p "${PADDLEOCR_DIR}/det/en"
    wget -q --show-progress -O /tmp/en_PP-OCRv3_det_infer.tar \
        "https://paddleocr.bj.bcebos.com/PP-OCRv3/english/en_PP-OCRv3_det_infer.tar"
    tar -xf /tmp/en_PP-OCRv3_det_infer.tar -C "${PADDLEOCR_DIR}/det/en/"
    rm -f /tmp/en_PP-OCRv3_det_infer.tar
else
    echo "[6/8] PaddleOCR det model already exists, skipping."
fi

if [ ! -d "${PADDLEOCR_DIR}/rec/en/en_PP-OCRv4_rec_infer" ]; then
    echo "      Downloading PaddleOCR rec model ..."
    mkdir -p "${PADDLEOCR_DIR}/rec/en"
    wget -q --show-progress -O /tmp/en_PP-OCRv4_rec_infer.tar \
        "https://paddleocr.bj.bcebos.com/PP-OCRv4/english/en_PP-OCRv4_rec_infer.tar"
    tar -xf /tmp/en_PP-OCRv4_rec_infer.tar -C "${PADDLEOCR_DIR}/rec/en/"
    rm -f /tmp/en_PP-OCRv4_rec_infer.tar
else
    echo "      PaddleOCR rec model already exists, skipping."
fi

# PaddleOCR `__init__` validates the cls path even when use_angle_cls=False,
# so we still need this lightweight cls model present.
if [ ! -d "${PADDLEOCR_DIR}/cls/ch_ppocr_mobile_v2.0_cls_infer" ]; then
    echo "      Downloading PaddleOCR cls model ..."
    mkdir -p "${PADDLEOCR_DIR}/cls"
    wget -q --show-progress -O /tmp/ch_ppocr_mobile_v2.0_cls_infer.tar \
        "https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_mobile_v2.0_cls_infer.tar"
    tar -xf /tmp/ch_ppocr_mobile_v2.0_cls_infer.tar -C "${PADDLEOCR_DIR}/cls/"
    rm -f /tmp/ch_ppocr_mobile_v2.0_cls_infer.tar
else
    echo "      PaddleOCR cls model already exists, skipping."
fi

# 7. Mask2Former + OpenAI CLIP-ViT-L-14 + mmdetection v2.28.2 configs
# (for the `geneval` reward via flow_grpo/gen_eval.py)
MASK2FORMER="mask2former_swin-s-p4-w7-224_lsj_8x2_50e_coco_20220504_001756-743b7d99.pth"
M2F_PT="${MODELS_DIR}/${MASK2FORMER}"
M2F_URL="https://download.openmmlab.com/mmdetection/v2.0/mask2former/mask2former_swin-s-p4-w7-224_lsj_8x2_50e_coco/${MASK2FORMER}"
if [ ! -s "${M2F_PT}" ] || [ "$(stat -c %s "${M2F_PT}" 2>/dev/null || echo 0)" -lt 360000000 ]; then
    echo "[7/8] Downloading ${MASK2FORMER} ..."
    wget -q --show-progress -O "${M2F_PT}" "${M2F_URL}"
else
    echo "[7/8] mask2former weights already present, skipping."
fi

# OpenAI CLIP ViT-L-14: open_clip resolves it from ~/.cache/clip/ViT-L-14.pt
# and only touches the network on cache miss, so we pre-warm it once here.
CLIP_CACHE="${HOME:-/root}/.cache/clip"
CLIP_PT="${CLIP_CACHE}/ViT-L-14.pt"
CLIP_URL="https://openaipublic.azureedge.net/clip/models/b8cca3fd41ae0c99ba7e8951adf17d267cdb84cd88be6f7c2e0eca1737a03836/ViT-L-14.pt"
if [ ! -s "${CLIP_PT}" ] || [ "$(stat -c %s "${CLIP_PT}" 2>/dev/null || echo 0)" -lt 900000000 ]; then
    echo "      Downloading OpenAI ViT-L-14 to ${CLIP_PT} ..."
    mkdir -p "${CLIP_CACHE}"
    wget -q --show-progress -O "${CLIP_PT}" "${CLIP_URL}"
else
    echo "      OpenAI ViT-L-14 cache already present, skipping."
fi

# mmdet v2.28.2 ships configs/ separately from the wheel; gen_eval.py expects
# them under the active venv's site-packages/configs/mask2former/.
# Resolve the active venv's site-packages dynamically and install configs.
VENV_SITE="$(python -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])' 2>/dev/null || true)"
if [ -n "${VENV_SITE}" ] && [ -d "${VENV_SITE}" ] && [ ! -d "${VENV_SITE}/configs/mask2former" ]; then
    echo "[8/8] Installing mmdet v2.28.2 configs into ${VENV_SITE}/configs/ ..."
    MMDET_TMP="$(mktemp -d)"
    git clone --depth 1 -b v2.28.2 https://github.com/open-mmlab/mmdetection.git "${MMDET_TMP}"
    mkdir -p "${VENV_SITE}/configs"
    cp -a "${MMDET_TMP}/configs/." "${VENV_SITE}/configs/"
    rm -rf "${MMDET_TMP}"
else
    echo "[8/8] mmdet configs already installed (or no active site-packages found), skipping."
fi

echo ""
echo "=== Done. Models ready under ${MODELS_DIR}/ ==="
ls -1 "${MODELS_DIR}/"
