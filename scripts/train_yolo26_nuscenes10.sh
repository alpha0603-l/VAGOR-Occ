#!/usr/bin/env bash
set -euo pipefail

cd /home/xsyu/GSF
export PYTHONPATH=/home/xsyu/GSF/ultralytics:${PYTHONPATH:-}

python tools/train_yolo26_nuscenes10.py \
  --data data/nuscenes_yolo10/nuscenes10_yolo.yaml \
  --weights ckpts/yolo26m.pt \
  --ultralytics-root /home/xsyu/GSF/ultralytics \
  --use-p2 \
  --size m \
  --imgsz 704 \
  --epochs 50 \
  --batch 8 \
  --device 0 \
  --workers 8 \
  --project work_dirs/yolo26_nuscenes10 \
  --name yolo26m_p2_nuscenes10
