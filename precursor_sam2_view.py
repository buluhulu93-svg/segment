#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
from ultralytics import SAM


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def load_bgr(path: str) -> np.ndarray:
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"이미지를 읽을 수 없습니다: {path}")
    return img


def save_image(path: Path, img: np.ndarray) -> None:
    cv2.imwrite(str(path), img)


def parse_points(points_str: str | None) -> list[list[float]] | None:
    if not points_str:
        return None
    parsed = json.loads(points_str)
    if isinstance(parsed, list) and parsed and isinstance(parsed[0], (int, float)):
        if len(parsed) != 2:
            raise ValueError("단일 point는 [x, y] 형식이어야 합니다.")
        return [parsed]
    return parsed


def parse_labels(labels_str: str | None) -> list[int] | list[list[int]] | None:
    if not labels_str:
        return None
    return json.loads(labels_str)


def parse_bboxes(bboxes_str: str | None) -> list[list[float]] | None:
    if not bboxes_str:
        return None
    parsed = json.loads(bboxes_str)
    if isinstance(parsed, list) and parsed and isinstance(parsed[0], (int, float)):
        if len(parsed) != 4:
            raise ValueError("단일 bbox는 [x1, y1, x2, y2] 형식이어야 합니다.")
        return [parsed]
    return parsed


def color_for_index(index: int) -> tuple[int, int, int]:
    rng = np.random.default_rng(seed=index + 1234)
    bgr = rng.integers(64, 256, size=3, dtype=np.uint8)
    return int(bgr[0]), int(bgr[1]), int(bgr[2])


def masks_to_numpy(result) -> np.ndarray:
    if result.masks is None or result.masks.data is None:
        return np.empty((0, 0, 0), dtype=np.uint8)
    masks = result.masks.data.detach().cpu().numpy()
    return (masks > 0).astype(np.uint8)


def boxes_to_numpy(result) -> np.ndarray | None:
    if result.boxes is None or result.boxes.xyxy is None:
        return None
    return result.boxes.xyxy.detach().cpu().numpy()


def scores_to_numpy(result) -> np.ndarray | None:
    if result.boxes is None or result.boxes.conf is None:
        return None
    return result.boxes.conf.detach().cpu().numpy()


def overlay_masks(image_bgr: np.ndarray, masks: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    out = image_bgr.copy()
    if masks.size == 0:
        return out
    for idx, mask in enumerate(masks):
        color = np.array(color_for_index(idx), dtype=np.uint8)
        colored = np.zeros_like(out, dtype=np.uint8)
        colored[mask.astype(bool)] = color
        out = cv2.addWeighted(out, 1.0, colored, alpha, 0)
    return out


def draw_boxes_and_indices(
    image_bgr: np.ndarray,
    masks: np.ndarray,
    boxes: np.ndarray | None,
    scores: np.ndarray | None,
) -> np.ndarray:
    out = image_bgr.copy()
    if masks.size == 0:
        return out

    for idx, mask in enumerate(masks):
        color = color_for_index(idx)
        ys, xs = np.where(mask > 0)
        if len(xs) == 0:
            continue

        if boxes is not None and idx < len(boxes):
            x1, y1, x2, y2 = boxes[idx].astype(int)
        else:
            x1, y1, x2, y2 = int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())

        cv2.rectangle(out, (x1, y1), (x2, y2), color, 1)
        label = f"#{idx}"
        if scores is not None and idx < len(scores):
            label = f"#{idx} {scores[idx]:.2f}"
        cv2.putText(out, label, (x1, max(12, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)

    return out


def save_individual_masks(output_dir: Path, masks: np.ndarray) -> None:
    if masks.size == 0:
        return
    mask_dir = output_dir / "masks"
    ensure_dir(mask_dir)
    for idx, mask in enumerate(masks):
        save_image(mask_dir / f"mask_{idx:04d}.png", mask.astype(np.uint8) * 255)


def build_summary(masks: np.ndarray, boxes: np.ndarray | None, scores: np.ndarray | None) -> dict:
    summary = {"num_masks": int(masks.shape[0])}
    items = []
    for idx, mask in enumerate(masks):
        area_px = int(mask.sum())
        ys, xs = np.where(mask > 0)
        if len(xs) == 0:
            continue
        if boxes is not None and idx < len(boxes):
            x1, y1, x2, y2 = [float(v) for v in boxes[idx]]
        else:
            x1, y1, x2, y2 = float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())
        item = {
            "index": idx,
            "area_px": area_px,
            "bbox_xyxy": [x1, y1, x2, y2],
        }
        if scores is not None and idx < len(scores):
            item["score"] = float(scores[idx])
        items.append(item)
    summary["items"] = items
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="View Ultralytics SAM2 segmentation results on a single image")
    parser.add_argument("--input", type=str, required=True, help="입력 이미지 경로")
    parser.add_argument("--output_dir", type=str, required=True, help="출력 폴더")
    parser.add_argument("--model", type=str, default="sam2.1_b.pt", help="예: sam2_t.pt, sam2_b.pt, sam2.1_b.pt")
    parser.add_argument("--device", type=str, default=None, help="예: cpu, cuda:0")
    parser.add_argument("--imgsz", type=int, default=1024, help="Ultralytics 추론 해상도")
    parser.add_argument("--retina_masks", action="store_true", help="고해상도 마스크 사용")
    parser.add_argument("--points", type=str, default=None, help='JSON 문자열. 예: "[[500,400]]" 또는 "[[[500,400],[520,410]]]"')
    parser.add_argument("--labels", type=str, default=None, help='JSON 문자열. 예: "[1]" 또는 "[[1,0]]"')
    parser.add_argument("--bboxes", type=str, default=None, help='JSON 문자열. 예: "[[100,100,800,800]]"')
    parser.add_argument("--save_individual_masks", action="store_true", help="개별 마스크 PNG 저장")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    ensure_dir(output_dir)

    image_bgr = load_bgr(str(input_path))

    model = SAM(args.model)

    predict_kwargs = {
        "source": str(input_path),
        "imgsz": args.imgsz,
        "retina_masks": args.retina_masks,
        "verbose": False,
    }
    if args.device:
        predict_kwargs["device"] = args.device

    points = parse_points(args.points)
    labels = parse_labels(args.labels)
    bboxes = parse_bboxes(args.bboxes)

    if points is not None:
        predict_kwargs["points"] = points
    if labels is not None:
        predict_kwargs["labels"] = labels
    if bboxes is not None:
        predict_kwargs["bboxes"] = bboxes

    results = model(**predict_kwargs)
    if not results:
        raise RuntimeError("SAM2 결과가 비어 있습니다.")

    result = results[0]
    masks = masks_to_numpy(result)
    boxes = boxes_to_numpy(result)
    scores = scores_to_numpy(result)

    overlay = overlay_masks(image_bgr, masks, alpha=0.45)
    overlay_with_boxes = draw_boxes_and_indices(overlay, masks, boxes, scores)

    save_image(output_dir / "01_input.png", image_bgr)
    save_image(output_dir / "02_overlay.png", overlay)
    save_image(output_dir / "03_overlay_boxes.png", overlay_with_boxes)

    if result.plot is not None:
        plotted = result.plot()
        save_image(output_dir / "04_ultralytics_plot.png", plotted)

    if save_individual_masks:
        save_individual_masks(output_dir, masks)

    summary = build_summary(masks, boxes, scores)
    with (output_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
