#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Train/fine-tune YOLO26 on generated nuScenes 10-class 2D labels."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Optional

from ultralytics import YOLO


def find_p2_yaml(ultralytics_root: str, size: str = "m") -> Optional[str]:
    root = Path(ultralytics_root)
    if not root.exists():
        return None
    patterns = [
        f"**/yolo26{size}*p2*.yaml",
        f"**/yolo26*{size}*p2*.yaml",
        f"**/*26*{size}*p2*.yaml",
        f"**/*yolo*{size}*p2*.yaml",
    ]
    for pat in patterns:
        matches = sorted(root.glob(pat))
        if matches:
            return str(matches[0])
    return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train YOLO26 nuScenes10")
    parser.add_argument("--data", default="data/nuscenes_yolo10/nuscenes10_yolo.yaml")
    parser.add_argument("--weights", default="ckpts/yolo26m.pt")
    parser.add_argument("--ultralytics-root", default="/home/xsyu/GSF/ultralytics")
    parser.add_argument("--use-p2", action="store_true", help="try to use local YOLO26-P2 yaml, then load weights")
    parser.add_argument("--size", default="m", choices=["n", "s", "m", "l", "x"])
    parser.add_argument("--imgsz", type=int, default=704)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--device", default="0")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--project", default="work_dirs/yolo26_nuscenes10")
    parser.add_argument("--name", default="yolo26m_p2_nuscenes10")
    parser.add_argument("--close-mosaic", type=int, default=10)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--cache", default=False)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.environ.setdefault("YOLO_VERBOSE", "True")
    if args.use_p2:
        p2_yaml = find_p2_yaml(args.ultralytics_root, size=args.size)
        if p2_yaml is None:
            print("[WARN] Could not find YOLO26 P2 yaml. Falling back to weights model directly.")
            model = YOLO(args.weights)
        else:
            print(f"[INFO] Using P2 yaml: {p2_yaml}")
            model = YOLO(p2_yaml)
            print(f"[INFO] Loading weights: {args.weights}")
            model.load(args.weights)
    else:
        model = YOLO(args.weights)

    model.train(
        data=args.data,
        imgsz=args.imgsz,
        epochs=args.epochs,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        pretrained=True,
        close_mosaic=args.close_mosaic,
        patience=args.patience,
        project=args.project,
        name=args.name,
        resume=args.resume,
        cache=args.cache,
        amp=False,
        val=False,
        plots=False,
        save_period=1,
    )

    best = Path(args.project) / args.name / "weights" / "best.pt"
    print("\nTraining finished.")
    print("Best weight should be at:", best)
    print("Copy to GSF ckpt, for example:")
    print(f"  cp {best} ckpts/yolo26m-p2-nuscenes10.pt")


if __name__ == "__main__":
    main()
