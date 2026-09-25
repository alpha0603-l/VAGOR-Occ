#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

TRAIN_PKL=${TRAIN_PKL:-/home/xsyu/GSF/data/nuscenes_infos_train.pkl}
VAL_PKL=${VAL_PKL:-/home/xsyu/GSF/data/nuscenes_infos_val.pkl}
DATA_ROOT=${DATA_ROOT:-/home/xsyu/GSF/data/nuscenes}
OUT_DIR=${OUT_DIR:-/home/xsyu/GSF/data/nuscenes_yolo10}

python tools/nuscenes_3dbox_to_yolo2d.py \
  --data-root "$DATA_ROOT" \
  --train-pkl "$TRAIN_PKL" \
  --val-pkl "$VAL_PKL" \
  --out-dir "$OUT_DIR" \
  --link-mode symlink \
  --box-dim-order wlh \
  --z-origin center \
  --min-visible-corners 1 \
  --min-small-box-w 1 \
  --min-small-box-h 1 \
  --min-box-w 3 \
  --min-box-h 3 \
  --print-interval 200
