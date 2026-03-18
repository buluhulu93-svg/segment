#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
precursor_aspect_ratio.py

목적:
- SEM 이미지 1장을 입력받아 rod-like small particle 후보를 추출하고
- 각 후보의 aspect ratio(장축/단축)를 계산
- 평균/중앙값/표준편차/유효 입자 수를 저장

지원 방법:
1) classical : 고전 영상처리 기반
2) unet      : U-Net semantic segmentation 추론 + shape filtering

주의:
- classical 방법은 바로 실행 가능
- unet 방법은 학습된 checkpoint(.pth)가 필요
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np

# =========================
# 데이터 구조
# =========================
@dataclass
class ParticleMeasurement:
    contour_index: int
    area: float
    perimeter: float
    major_axis: float
    minor_axis: float
    aspect_ratio: float
    angle_deg: float
    centroid_x: float
    centroid_y: float
    solidity: float
    extent: float


# =========================
# 유틸
# =========================
def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def load_image_grayscale(image_path: str) -> np.ndarray:
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"이미지를 읽을 수 없습니다: {image_path}")
    return image


def load_image_bgr(image_path: str) -> np.ndarray:
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"이미지를 읽을 수 없습니다: {image_path}")
    return image


def normalize_to_uint8(image: np.ndarray) -> np.ndarray:
    image = image.astype(np.float32)
    min_val = float(image.min())
    max_val = float(image.max())
    if max_val - min_val < 1e-8:
        return np.zeros_like(image, dtype=np.uint8)
    norm = (image - min_val) / (max_val - min_val)
    return (norm * 255.0).clip(0, 255).astype(np.uint8)


def save_image(path: Path, image: np.ndarray) -> None:
    cv2.imwrite(str(path), image)


def overlay_contours(
    image_bgr: np.ndarray,
    contours: List[np.ndarray],
    measurements: Optional[List[ParticleMeasurement]] = None
) -> np.ndarray:
    vis = image_bgr.copy()
    cv2.drawContours(vis, contours, -1, (0, 255, 0), 1)

    if measurements is not None:
        for m, contour in zip(measurements, contours):
            cx = int(round(m.centroid_x))
            cy = int(round(m.centroid_y))
            cv2.putText(
                vis,
                f"{m.aspect_ratio:.2f}",
                (cx + 3, cy - 3),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.35,
                (0, 0, 255),
                1,
                cv2.LINE_AA,
            )
    return vis


def compute_summary(measurements: List[ParticleMeasurement]) -> dict:
    if not measurements:
        return {
            "count": 0,
            "mean_aspect_ratio": None,
            "median_aspect_ratio": None,
            "std_aspect_ratio": None,
            "min_aspect_ratio": None,
            "max_aspect_ratio": None,
        }

    aspect_ratios = np.array([m.aspect_ratio for m in measurements], dtype=np.float32)

    return {
        "count": int(len(measurements)),
        "mean_aspect_ratio": float(np.mean(aspect_ratios)),
        "median_aspect_ratio": float(np.median(aspect_ratios)),
        "std_aspect_ratio": float(np.std(aspect_ratios)),
        "min_aspect_ratio": float(np.min(aspect_ratios)),
        "max_aspect_ratio": float(np.max(aspect_ratios)),
    }


def contour_measurement(contour: np.ndarray, contour_index: int) -> Optional[ParticleMeasurement]:
    area = cv2.contourArea(contour)
    if area <= 1.0:
        return None

    perimeter = cv2.arcLength(contour, True)

    if len(contour) < 5:
        x, y, w, h = cv2.boundingRect(contour)
        major_axis = float(max(w, h))
        minor_axis = float(max(1, min(w, h)))
        angle_deg = 0.0
        cx = x + w / 2.0
        cy = y + h / 2.0
    else:
        ellipse = cv2.fitEllipse(contour)
        (cx, cy), (axis1, axis2), angle_deg = ellipse
        major_axis = float(max(axis1, axis2))
        minor_axis = float(max(1e-6, min(axis1, axis2)))

    aspect_ratio = float(major_axis / minor_axis)

    hull = cv2.convexHull(contour)
    hull_area = float(cv2.contourArea(hull))
    solidity = float(area / hull_area) if hull_area > 1e-8 else 0.0

    x, y, w, h = cv2.boundingRect(contour)
    rect_area = float(w * h)
    extent = float(area / rect_area) if rect_area > 1e-8 else 0.0

    return ParticleMeasurement(
        contour_index=contour_index,
        area=float(area),
        perimeter=float(perimeter),
        major_axis=float(major_axis),
        minor_axis=float(minor_axis),
        aspect_ratio=aspect_ratio,
        angle_deg=float(angle_deg),
        centroid_x=float(cx),
        centroid_y=float(cy),
        solidity=solidity,
        extent=extent,
    )


def filter_measurements(
    contours: List[np.ndarray],
    min_area: float,
    max_area: float,
    min_aspect_ratio: float,
    max_aspect_ratio: float,
    min_solidity: float,
    max_solidity: float,
    min_extent: float,
    max_extent: float,
) -> Tuple[List[np.ndarray], List[ParticleMeasurement]]:
    filtered_contours: List[np.ndarray] = []
    filtered_measurements: List[ParticleMeasurement] = []

    for idx, contour in enumerate(contours):
        measurement = contour_measurement(contour, contour_index=idx)
        if measurement is None:
            continue

        if not (min_area <= measurement.area <= max_area):
            continue
        if not (min_aspect_ratio <= measurement.aspect_ratio <= max_aspect_ratio):
            continue
        if not (min_solidity <= measurement.solidity <= max_solidity):
            continue
        if not (min_extent <= measurement.extent <= max_extent):
            continue

        filtered_contours.append(contour)
        filtered_measurements.append(measurement)

    return filtered_contours, filtered_measurements

def preprocess_classical(image_gray: np.ndarray) -> np.ndarray:
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(image_gray)

    denoised = cv2.GaussianBlur(enhanced, (3, 3), 0)

    # rod-like texture를 강조하기 위한 Laplacian 성분 추가
    lap = cv2.Laplacian(denoised, cv2.CV_32F, ksize=3)
    lap = normalize_to_uint8(np.abs(lap))

    combined = cv2.addWeighted(denoised, 0.8, lap, 0.2, 0)
    return combined


def segment_classical(image_gray: np.ndarray) -> Tuple[np.ndarray, dict]:
    pre = preprocess_classical(image_gray)

    # adaptive threshold
    binary = cv2.adaptiveThreshold(
        pre,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        21,
        -3,
    )

    # morphology 정리
    kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

    cleaned = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel_open, iterations=1)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel_close, iterations=1)

    debug_images = {
        "preprocessed": pre,
        "binary": binary,
        "cleaned": cleaned,
    }
    return cleaned, debug_images

# =========================
# contour 추출
# =========================
def extract_external_contours(mask: np.ndarray) -> List[np.ndarray]:
    contours_info = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours_info) == 2:
        contours, _ = contours_info
    else:
        _, contours, _ = contours_info
    return contours


# =========================
# 결과 저장
# =========================
def save_measurements_csv(output_path: Path, measurements: List[ParticleMeasurement]) -> None:
    import csv

    fieldnames = [
        "contour_index",
        "area",
        "perimeter",
        "major_axis",
        "minor_axis",
        "aspect_ratio",
        "angle_deg",
        "centroid_x",
        "centroid_y",
        "solidity",
        "extent",
    ]

    with output_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for m in measurements:
            writer.writerow(asdict(m))


def save_summary_json(output_path: Path, summary: dict) -> None:
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)


# =========================
# 메인 파이프라인
# =========================
def run_pipeline(args: argparse.Namespace) -> None:
    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    ensure_dir(output_dir)

    image_gray = load_image_grayscale(str(input_path))
    image_bgr = load_image_bgr(str(input_path))

    
    mask, debug_images = segment_classical(image_gray)
    
    contours = extract_external_contours(mask)

    filtered_contours, measurements = filter_measurements(
        contours=contours,
        min_area=args.min_area,
        max_area=args.max_area,
        min_aspect_ratio=args.min_aspect_ratio,
        max_aspect_ratio=args.max_aspect_ratio,
        min_solidity=args.min_solidity,
        max_solidity=args.max_solidity,
        min_extent=args.min_extent,
        max_extent=args.max_extent,
    )

    summary = compute_summary(measurements)

    # 저장
    save_image(output_dir / "01_input_gray.png", image_gray)

    for name, img in debug_images.items():
        save_image(output_dir / f"02_{name}.png", img)

    save_image(output_dir / "03_mask.png", mask)

    overlay = overlay_contours(image_bgr, filtered_contours, measurements)
    save_image(output_dir / "04_overlay.png", overlay)

    save_measurements_csv(output_dir / "measurements.csv", measurements)
    save_summary_json(output_dir / "summary.json", summary)

    print("===== 완료 =====")
    print(f"입력 이미지: {input_path}")
    print(f"출력 폴더: {output_dir}")
    print("----- summary -----")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


# =========================
# argparse
# =========================
def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="SEM precursor particle aspect ratio measurement")

    parser.add_argument("--input", type=str, required=True, help="입력 이미지 경로")
    parser.add_argument("--output_dir", type=str, required=True, help="출력 폴더")
    # shape filtering
    parser.add_argument("--min_area", type=float, default=20.0, help="최소 contour area")
    parser.add_argument("--max_area", type=float, default=5000.0, help="최대 contour area")
    parser.add_argument("--min_aspect_ratio", type=float, default=1.5, help="최소 aspect ratio")
    parser.add_argument("--max_aspect_ratio", type=float, default=20.0, help="최대 aspect ratio")
    parser.add_argument("--min_solidity", type=float, default=0.2, help="최소 solidity")
    parser.add_argument("--max_solidity", type=float, default=1.0, help="최대 solidity")
    parser.add_argument("--min_extent", type=float, default=0.1, help="최소 extent")
    parser.add_argument("--max_extent", type=float, default=1.0, help="최대 extent")

    return parser


def main() -> None:
    parser = build_argparser()
    args = parser.parse_args()
    run_pipeline(args)


if __name__ == "__main__":
    main()