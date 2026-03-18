#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
from ultralytics import SAM


@dataclass
class MaskItem:
    index: int
    area_px: int
    bbox_xyxy: list[float]
    aspect_ratio: float
    solidity: float
    score: float | None
    tile_xyxy: list[int]
    prompt_xy_global: list[int]


# =========================================================
# Basic utils
# =========================================================
def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def load_bgr(path: str) -> np.ndarray:
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"이미지를 읽을 수 없습니다: {path}")
    return img


def load_gray(path: str) -> np.ndarray:
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"이미지를 읽을 수 없습니다: {path}")
    return img


def save_image(path: Path, img: np.ndarray) -> None:
    cv2.imwrite(str(path), img)


def color_for_index(index: int) -> tuple[int, int, int]:
    rng = np.random.default_rng(seed=index + 1234)
    bgr = rng.integers(64, 256, size=3, dtype=np.uint8)
    return int(bgr[0]), int(bgr[1]), int(bgr[2])


def normalize_uint8(img: np.ndarray) -> np.ndarray:
    img = img.astype(np.float32)
    mn = float(img.min())
    mx = float(img.max())
    if mx - mn < 1e-8:
        return np.zeros_like(img, dtype=np.uint8)
    out = (img - mn) / (mx - mn)
    return (out * 255.0).clip(0, 255).astype(np.uint8)


# =========================================================
# ROI detection for secondary particle
# =========================================================
def remove_bottom_annotation(gray: np.ndarray, crop_ratio: float = 0.12) -> np.ndarray:
    h, _ = gray.shape
    out = gray.copy()
    cut_y = int(h * (1.0 - crop_ratio))
    out[cut_y:, :] = 0
    return out


def detect_secondary_particle_roi(gray: np.ndarray) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    work = remove_bottom_annotation(gray, crop_ratio=0.12)
    blur = cv2.GaussianBlur(work, (9, 9), 0)
    _, th = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, kernel, iterations=2)
    th = cv2.morphologyEx(th, cv2.MORPH_OPEN, kernel, iterations=1)

    contours, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mask = np.zeros_like(gray, dtype=np.uint8)
    if not contours:
        mask[:, :] = 255
        h, w = gray.shape
        return mask, (0, 0, w, h)

    largest = max(contours, key=cv2.contourArea)
    cv2.drawContours(mask, [largest], -1, 255, thickness=-1)
    erode_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    mask = cv2.erode(mask, erode_kernel, iterations=1)
    x, y, w, h = cv2.boundingRect(largest)
    return mask, (x, y, x + w, y + h)


# =========================================================
# Tile and point generation
# =========================================================
def generate_tiles(x1: int, y1: int, x2: int, y2: int, tile_size: int, stride: int) -> list[tuple[int, int, int, int]]:
    tiles: list[tuple[int, int, int, int]] = []
    if x2 - x1 <= tile_size and y2 - y1 <= tile_size:
        return [(x1, y1, x2, y2)]

    xs = list(range(x1, max(x1 + 1, x2 - tile_size + 1), stride))
    ys = list(range(y1, max(y1 + 1, y2 - tile_size + 1), stride))

    if xs and xs[-1] != x2 - tile_size:
        xs.append(max(x1, x2 - tile_size))
    if ys and ys[-1] != y2 - tile_size:
        ys.append(max(y1, y2 - tile_size))

    if not xs:
        xs = [x1]
    if not ys:
        ys = [y1]

    for yy in ys:
        for xx in xs:
            tx1 = xx
            ty1 = yy
            tx2 = min(xx + tile_size, x2)
            ty2 = min(yy + tile_size, y2)
            tiles.append((tx1, ty1, tx2, ty2))
    return tiles


def enhance_tile_for_points(tile_gray: np.ndarray) -> np.ndarray:
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    img = clahe.apply(tile_gray)
    blur = cv2.GaussianBlur(img, (3, 3), 0)
    grad = cv2.morphologyEx(blur, cv2.MORPH_GRADIENT, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
    blackhat = cv2.morphologyEx(blur, cv2.MORPH_BLACKHAT, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)))
    lap = cv2.Laplacian(blur, cv2.CV_32F, ksize=3)
    lap = normalize_uint8(np.abs(lap))
    combined = cv2.addWeighted(grad, 0.45, blackhat, 0.35, 0)
    combined = cv2.addWeighted(combined, 0.8, lap, 0.2, 0)
    return normalize_uint8(combined)


def sample_candidate_points(
    tile_gray: np.ndarray,
    max_points: int,
    min_distance: int,
    quality_level: float,
    mask: np.ndarray | None = None,
) -> list[tuple[int, int]]:
    enhanced = enhance_tile_for_points(tile_gray)
    corners = cv2.goodFeaturesToTrack(
        enhanced,
        maxCorners=max_points * 4,
        qualityLevel=quality_level,
        minDistance=min_distance,
        blockSize=5,
        mask=mask,
        useHarrisDetector=False,
    )
    points: list[tuple[int, int]] = []
    if corners is not None:
        for c in corners[:, 0, :]:
            x, y = int(round(c[0])), int(round(c[1]))
            points.append((x, y))

    # fallback: contour centroids from thresholded enhanced image
    if len(points) < max(8, max_points // 2):
        _, th = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        th = cv2.morphologyEx(th, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2)))
        if mask is not None:
            th[mask == 0] = 0
        contours, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in sorted(contours, key=cv2.contourArea, reverse=True):
            area = cv2.contourArea(cnt)
            if area < 6 or area > 300:
                continue
            m = cv2.moments(cnt)
            if abs(m["m00"]) < 1e-6:
                continue
            x = int(round(m["m10"] / m["m00"]))
            y = int(round(m["m01"] / m["m00"]))
            points.append((x, y))
            if len(points) >= max_points * 2:
                break

    # deduplicate with greedy distance suppression
    kept: list[tuple[int, int]] = []
    for px, py in points:
        too_close = False
        for qx, qy in kept:
            if (px - qx) ** 2 + (py - qy) ** 2 < min_distance ** 2:
                too_close = True
                break
        if not too_close:
            kept.append((px, py))
        if len(kept) >= max_points:
            break
    return kept


# =========================================================
# Geometry and filtering
# =========================================================
def mask_to_contour(mask: np.ndarray) -> np.ndarray | None:
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    return max(contours, key=cv2.contourArea)


def mask_stats(mask: np.ndarray) -> tuple[int, list[float], float, float] | None:
    cnt = mask_to_contour(mask)
    if cnt is None:
        return None
    area = int(mask.sum())
    x, y, w, h = cv2.boundingRect(cnt)
    hull = cv2.convexHull(cnt)
    hull_area = float(cv2.contourArea(hull))
    cnt_area = float(cv2.contourArea(cnt))
    solidity = cnt_area / hull_area if hull_area > 1e-8 else 0.0

    if len(cnt) >= 5:
        (_, _), (a1, a2), _ = cv2.fitEllipse(cnt)
        major = float(max(a1, a2))
        minor = float(max(1e-6, min(a1, a2)))
    else:
        major = float(max(w, h))
        minor = float(max(1.0, min(w, h)))
    aspect_ratio = major / minor
    return area, [float(x), float(y), float(x + w), float(y + h)], aspect_ratio, solidity


def iou_binary(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    inter = np.logical_and(mask_a > 0, mask_b > 0).sum()
    union = np.logical_or(mask_a > 0, mask_b > 0).sum()
    if union == 0:
        return 0.0
    return float(inter / union)


def should_keep_mask(
    mask: np.ndarray,
    prompt_xy: tuple[int, int],
    min_area: int,
    max_area: int,
    min_aspect_ratio: float,
    max_aspect_ratio: float,
    min_solidity: float,
    max_solidity: float,
    border_margin: int,
) -> tuple[bool, dict]:
    stats = mask_stats(mask)
    if stats is None:
        return False, {}
    area, bbox, aspect_ratio, solidity = stats
    x1, y1, x2, y2 = [int(v) for v in bbox]
    px, py = prompt_xy

    info = {
        "area": area,
        "bbox": bbox,
        "aspect_ratio": aspect_ratio,
        "solidity": solidity,
    }

    if area < min_area or area > max_area:
        return False, info
    if aspect_ratio < min_aspect_ratio or aspect_ratio > max_aspect_ratio:
        return False, info
    if solidity < min_solidity or solidity > max_solidity:
        return False, info
    if px < x1 or px > x2 or py < y1 or py > y2:
        return False, info
    h, w = mask.shape
    if x1 <= border_margin or y1 <= border_margin or x2 >= w - border_margin or y2 >= h - border_margin:
        return False, info
    return True, info


# =========================================================
# SAM2 inference per tile/point
# =========================================================
def masks_from_result(result) -> np.ndarray:
    if result.masks is None or result.masks.data is None:
        return np.empty((0, 0, 0), dtype=np.uint8)
    arr = result.masks.data.detach().cpu().numpy()
    return (arr > 0).astype(np.uint8)


def scores_from_result(result) -> np.ndarray | None:
    if result.boxes is None or result.boxes.conf is None:
        return None
    return result.boxes.conf.detach().cpu().numpy()


def run_small_object_sam2(
    model: SAM,
    image_bgr: np.ndarray,
    image_gray: np.ndarray,
    roi_mask: np.ndarray,
    roi_bbox: tuple[int, int, int, int],
    tile_size: int,
    stride: int,
    points_per_tile: int,
    quality_level: float,
    point_min_distance: int,
    imgsz: int,
    retina_masks: bool,
    device: str | None,
    multimask_output: bool,
    min_area: int,
    max_area: int,
    min_aspect_ratio: float,
    max_aspect_ratio: float,
    min_solidity: float,
    max_solidity: float,
    border_margin: int,
    dedup_iou: float,
) -> tuple[list[np.ndarray], list[MaskItem], dict]:
    x1, y1, x2, y2 = roi_bbox
    tiles = generate_tiles(x1, y1, x2, y2, tile_size=tile_size, stride=stride)

    kept_masks_global: list[np.ndarray] = []
    kept_items: list[MaskItem] = []
    debug_points: list[dict] = []

    predict_common = {
        "imgsz": imgsz,
        "retina_masks": retina_masks,
        "verbose": False,
    }
    if device:
        predict_common["device"] = device

    candidate_count = 0
    accepted_count = 0

    for tile_idx, (tx1, ty1, tx2, ty2) in enumerate(tiles):
        tile_bgr = image_bgr[ty1:ty2, tx1:tx2].copy()
        tile_gray = image_gray[ty1:ty2, tx1:tx2].copy()
        tile_roi = roi_mask[ty1:ty2, tx1:tx2].copy()

        # Ignore tiles with too little ROI area.
        if (tile_roi > 0).sum() < 0.15 * tile_roi.size:
            continue

        points = sample_candidate_points(
            tile_gray=tile_gray,
            max_points=points_per_tile,
            min_distance=point_min_distance,
            quality_level=quality_level,
            mask=tile_roi,
        )

        for px, py in points:
            candidate_count += 1
            debug_points.append({
                "tile_index": tile_idx,
                "tile_xyxy": [tx1, ty1, tx2, ty2],
                "point_xy_tile": [px, py],
                "point_xy_global": [tx1 + px, ty1 + py],
            })

            results = model(
                source=tile_bgr,
                points=[[px, py]],
                labels=[1],
                # multimask_output=multimask_output,
                **predict_common,
            )
            if not results:
                continue

            result = results[0]
            masks = masks_from_result(result)
            scores = scores_from_result(result)
            if masks.size == 0:
                continue

            for mask_idx, tile_mask in enumerate(masks):
                keep, info = should_keep_mask(
                    mask=tile_mask.astype(np.uint8),
                    prompt_xy=(px, py),
                    min_area=min_area,
                    max_area=max_area,
                    min_aspect_ratio=min_aspect_ratio,
                    max_aspect_ratio=max_aspect_ratio,
                    min_solidity=min_solidity,
                    max_solidity=max_solidity,
                    border_margin=border_margin,
                )
                if not keep:
                    continue

                global_mask = np.zeros(image_gray.shape, dtype=np.uint8)
                global_mask[ty1:ty2, tx1:tx2] = tile_mask.astype(np.uint8)
                global_mask[roi_mask == 0] = 0

                # Deduplicate against previously accepted masks.
                is_dup = False
                for prev in kept_masks_global:
                    if iou_binary(prev, global_mask) >= dedup_iou:
                        is_dup = True
                        break
                if is_dup:
                    continue

                accepted_count += 1
                score_val = None
                if scores is not None and mask_idx < len(scores):
                    score_val = float(scores[mask_idx])

                area, bbox, aspect_ratio, solidity = mask_stats(global_mask)  # type: ignore[misc]
                item = MaskItem(
                    index=len(kept_items),
                    area_px=int(area),
                    bbox_xyxy=bbox,
                    aspect_ratio=float(aspect_ratio),
                    solidity=float(solidity),
                    score=score_val,
                    tile_xyxy=[tx1, ty1, tx2, ty2],
                    prompt_xy_global=[tx1 + px, ty1 + py],
                )
                kept_masks_global.append(global_mask)
                kept_items.append(item)

    debug = {
        "roi_bbox_xyxy": [x1, y1, x2, y2],
        "num_tiles": len(tiles),
        "num_candidate_points": candidate_count,
        "num_accepted_masks": accepted_count,
        "candidate_points": debug_points,
    }
    return kept_masks_global, kept_items, debug


# =========================================================
# Visualization and export
# =========================================================
def overlay_masks(image_bgr: np.ndarray, masks: list[np.ndarray], alpha: float = 0.45) -> np.ndarray:
    out = image_bgr.copy()
    for idx, mask in enumerate(masks):
        color = np.array(color_for_index(idx), dtype=np.uint8)
        colored = np.zeros_like(out, dtype=np.uint8)
        colored[mask.astype(bool)] = color
        out = cv2.addWeighted(out, 1.0, colored, alpha, 0)
    return out


def draw_boxes_labels(image_bgr: np.ndarray, items: list[MaskItem]) -> np.ndarray:
    out = image_bgr.copy()
    for item in items:
        color = color_for_index(item.index)
        x1, y1, x2, y2 = [int(v) for v in item.bbox_xyxy]
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 1)
        label = f"#{item.index} ar={item.aspect_ratio:.2f}"
        if item.score is not None:
            label += f" s={item.score:.2f}"
        cv2.putText(out, label, (x1, max(12, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.38, color, 1, cv2.LINE_AA)
        px, py = item.prompt_xy_global
        cv2.circle(out, (px, py), 2, (0, 0, 255), -1)
    return out


def draw_roi_and_points(image_bgr: np.ndarray, roi_bbox: tuple[int, int, int, int], debug: dict) -> np.ndarray:
    out = image_bgr.copy()
    x1, y1, x2, y2 = roi_bbox
    cv2.rectangle(out, (x1, y1), (x2, y2), (255, 255, 0), 2)
    for item in debug.get("candidate_points", []):
        px, py = item["point_xy_global"]
        cv2.circle(out, (int(px), int(py)), 1, (0, 255, 255), -1)
    return out


def save_individual_masks(output_dir: Path, masks: list[np.ndarray]) -> None:
    mask_dir = output_dir / "masks"
    ensure_dir(mask_dir)
    for idx, mask in enumerate(masks):
        save_image(mask_dir / f"mask_{idx:04d}.png", mask.astype(np.uint8) * 255)


def build_summary(items: list[MaskItem], debug: dict) -> dict:
    return {
        "num_masks": len(items),
        "debug": debug,
        "items": [asdict(item) for item in items],
    }


# =========================================================
# Main
# =========================================================
def main() -> None:
    parser = argparse.ArgumentParser(description="Ultralytics SAM2 small-object segmentation viewer for precursor SEM images")
    parser.add_argument("--input", type=str, required=True, help="입력 이미지 경로")
    parser.add_argument("--output_dir", type=str, required=True, help="출력 폴더")
    parser.add_argument("--model", type=str, default="sam2.1_b.pt", help="예: sam2_t.pt, sam2_b.pt, sam2.1_b.pt")
    parser.add_argument("--device", type=str, default=None, help="예: cpu, cuda:0")
    parser.add_argument("--imgsz", type=int, default=1024, help="SAM2 추론 해상도")
    parser.add_argument("--retina_masks", action="store_true", help="고해상도 마스크 사용")
    parser.add_argument("--multimask_output", action="store_true", help="점 하나당 여러 개의 candidate mask 반환")
    parser.add_argument("--tile_size", type=int, default=512, help="ROI 내부 타일 크기")
    parser.add_argument("--stride", type=int, default=384, help="타일 stride")
    parser.add_argument("--points_per_tile", type=int, default=24, help="타일당 후보 foreground point 수")
    parser.add_argument("--quality_level", type=float, default=0.03, help="goodFeaturesToTrack qualityLevel")
    parser.add_argument("--point_min_distance", type=int, default=14, help="point 간 최소 거리")
    parser.add_argument("--min_area", type=int, default=18, help="유지할 최소 마스크 면적")
    parser.add_argument("--max_area", type=int, default=900, help="유지할 최대 마스크 면적")
    parser.add_argument("--min_aspect_ratio", type=float, default=1.8, help="최소 종횡비")
    parser.add_argument("--max_aspect_ratio", type=float, default=12.0, help="최대 종횡비")
    parser.add_argument("--min_solidity", type=float, default=0.25, help="최소 solidity")
    parser.add_argument("--max_solidity", type=float, default=0.98, help="최대 solidity")
    parser.add_argument("--border_margin", type=int, default=5, help="타일 경계 근처 마스크 제거 폭")
    parser.add_argument("--dedup_iou", type=float, default=0.60, help="중복 제거용 IoU threshold")
    parser.add_argument("--save_individual_masks", action="store_true", help="개별 마스크 PNG 저장")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    ensure_dir(output_dir)

    image_bgr = load_bgr(str(input_path))
    image_gray = load_gray(str(input_path))

    roi_mask, roi_bbox = detect_secondary_particle_roi(image_gray)
    model = SAM(args.model)

    masks_global, items, debug = run_small_object_sam2(
        model=model,
        image_bgr=image_bgr,
        image_gray=image_gray,
        roi_mask=roi_mask,
        roi_bbox=roi_bbox,
        tile_size=args.tile_size,
        stride=args.stride,
        points_per_tile=args.points_per_tile,
        quality_level=args.quality_level,
        point_min_distance=args.point_min_distance,
        imgsz=args.imgsz,
        retina_masks=args.retina_masks,
        device=args.device,
        multimask_output=args.multimask_output,
        min_area=args.min_area,
        max_area=args.max_area,
        min_aspect_ratio=args.min_aspect_ratio,
        max_aspect_ratio=args.max_aspect_ratio,
        min_solidity=args.min_solidity,
        max_solidity=args.max_solidity,
        border_margin=args.border_margin,
        dedup_iou=args.dedup_iou,
    )

    overlay = overlay_masks(image_bgr, masks_global, alpha=0.45)
    overlay_boxes = draw_boxes_labels(overlay, items)
    roi_points_vis = draw_roi_and_points(image_bgr, roi_bbox, debug)

    save_image(output_dir / "01_input.png", image_bgr)
    save_image(output_dir / "02_roi_mask.png", roi_mask)
    save_image(output_dir / "03_roi_points.png", roi_points_vis)
    save_image(output_dir / "04_overlay.png", overlay)
    save_image(output_dir / "05_overlay_boxes.png", overlay_boxes)

    if args.save_individual_masks:
        save_individual_masks(output_dir, masks_global)

    summary = build_summary(items, debug)
    with (output_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
