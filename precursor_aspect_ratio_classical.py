#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np


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


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def load_gray(path: str) -> np.ndarray:
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"이미지를 읽을 수 없습니다: {path}")
    return img


def load_bgr(path: str) -> np.ndarray:
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"이미지를 읽을 수 없습니다: {path}")
    return img


def save_image(path: Path, img: np.ndarray) -> None:
    cv2.imwrite(str(path), img)


def normalize_uint8(img: np.ndarray) -> np.ndarray:
    img = img.astype(np.float32)
    mn = float(img.min())
    mx = float(img.max())
    if mx - mn < 1e-8:
        return np.zeros_like(img, dtype=np.uint8)
    out = (img - mn) / (mx - mn)
    return (out * 255.0).clip(0, 255).astype(np.uint8)


def masked_image(gray: np.ndarray, mask: np.ndarray) -> np.ndarray:
    out = gray.copy()
    out[mask == 0] = 0
    return out


def remove_bottom_annotation(gray: np.ndarray, crop_ratio: float = 0.12) -> np.ndarray:
    h, _ = gray.shape
    out = gray.copy()
    cut_y = int(h * (1.0 - crop_ratio))
    out[cut_y:, :] = 0
    return out


def detect_secondary_particle_roi(gray: np.ndarray) -> np.ndarray:
    work = remove_bottom_annotation(gray, crop_ratio=0.12)
    blur = cv2.GaussianBlur(work, (9, 9), 0)
    _, th = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, kernel, iterations=2)
    th = cv2.morphologyEx(th, cv2.MORPH_OPEN, kernel, iterations=1)

    contours, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    roi_mask = np.zeros_like(gray, dtype=np.uint8)

    if not contours:
        roi_mask[:, :] = 255
        return roi_mask

    largest = max(contours, key=cv2.contourArea)
    cv2.drawContours(roi_mask, [largest], -1, 255, thickness=-1)

    erode_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    roi_mask = cv2.erode(roi_mask, erode_kernel, iterations=1)
    return roi_mask


def enhance_rod_texture(gray_roi: np.ndarray) -> dict[str, np.ndarray]:
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    clahe_img = clahe.apply(gray_roi)
    blur = cv2.GaussianBlur(clahe_img, (3, 3), 0)

    kernel_bh = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    blackhat = cv2.morphologyEx(blur, cv2.MORPH_BLACKHAT, kernel_bh)

    kernel_grad = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    gradient = cv2.morphologyEx(blur, cv2.MORPH_GRADIENT, kernel_grad)

    lap = cv2.Laplacian(blur, cv2.CV_32F, ksize=3)
    lap_abs = normalize_uint8(np.abs(lap))

    combined = cv2.addWeighted(gradient, 0.45, blackhat, 0.35, 0)
    combined = cv2.addWeighted(combined, 0.80, lap_abs, 0.20, 0)
    combined = normalize_uint8(combined)

    return {
        "clahe": clahe_img,
        "blur": blur,
        "blackhat": blackhat,
        "gradient": gradient,
        "laplacian_abs": lap_abs,
        "combined": combined,
    }


def segment_classical(gray: np.ndarray) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    roi_mask = detect_secondary_particle_roi(gray)
    gray_roi = masked_image(gray, roi_mask)
    enhanced = enhance_rod_texture(gray_roi)
    work = enhanced["combined"]

    binary = cv2.adaptiveThreshold(
        work,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        19,
        -2,
    )
    binary[roi_mask == 0] = 0

    kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
    opened = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel_open, iterations=1)

    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
    cleaned = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel_close, iterations=1)

    debug = {
        "roi_mask": roi_mask,
        "gray_roi": gray_roi,
        **enhanced,
        "binary": binary,
        "opened": opened,
        "cleaned": cleaned,
    }
    return cleaned, debug


def contour_measurement(contour: np.ndarray, contour_index: int) -> Optional[ParticleMeasurement]:
    area = float(cv2.contourArea(contour))
    if area <= 1.0:
        return None

    perimeter = float(cv2.arcLength(contour, True))
    if len(contour) < 5:
        x, y, w, h = cv2.boundingRect(contour)
        major_axis = float(max(w, h))
        minor_axis = float(max(1.0, min(w, h)))
        angle_deg = 0.0
        cx = x + w / 2.0
        cy = y + h / 2.0
    else:
        (cx, cy), (a1, a2), angle_deg = cv2.fitEllipse(contour)
        major_axis = float(max(a1, a2))
        minor_axis = float(max(1e-6, min(a1, a2)))

    aspect_ratio = float(major_axis / minor_axis)

    hull = cv2.convexHull(contour)
    hull_area = float(cv2.contourArea(hull))
    solidity = float(area / hull_area) if hull_area > 1e-8 else 0.0

    x, y, w, h = cv2.boundingRect(contour)
    rect_area = float(w * h)
    extent = float(area / rect_area) if rect_area > 1e-8 else 0.0

    return ParticleMeasurement(
        contour_index=contour_index,
        area=area,
        perimeter=perimeter,
        major_axis=major_axis,
        minor_axis=minor_axis,
        aspect_ratio=aspect_ratio,
        angle_deg=float(angle_deg),
        centroid_x=float(cx),
        centroid_y=float(cy),
        solidity=solidity,
        extent=extent,
    )


def extract_contours(mask: np.ndarray) -> list[np.ndarray]:
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours


def filter_measurable_rods(
    contours: list[np.ndarray],
    min_area: float,
    max_area: float,
    min_aspect_ratio: float,
    max_aspect_ratio: float,
    min_solidity: float,
    max_solidity: float,
    min_extent: float,
    max_extent: float,
    border_margin: int,
    image_shape: tuple[int, int],
) -> tuple[list[np.ndarray], list[ParticleMeasurement]]:
    h, w = image_shape
    kept_contours: list[np.ndarray] = []
    kept_measurements: list[ParticleMeasurement] = []

    for idx, contour in enumerate(contours):
        m = contour_measurement(contour, idx)
        if m is None:
            continue
        if not (min_area <= m.area <= max_area):
            continue
        if not (min_aspect_ratio <= m.aspect_ratio <= max_aspect_ratio):
            continue
        if not (min_solidity <= m.solidity <= max_solidity):
            continue
        if not (min_extent <= m.extent <= max_extent):
            continue
        if (
            m.centroid_x < border_margin
            or m.centroid_x > (w - border_margin)
            or m.centroid_y < border_margin
            or m.centroid_y > (h - border_margin)
        ):
            continue
        kept_contours.append(contour)
        kept_measurements.append(m)

    return kept_contours, kept_measurements


def draw_measurements_overlay(
    image_bgr: np.ndarray,
    contours: list[np.ndarray],
    measurements: list[ParticleMeasurement],
) -> np.ndarray:
    out = image_bgr.copy()
    cv2.drawContours(out, contours, -1, (0, 255, 0), 1)

    for contour, m in zip(contours, measurements):
        cx = int(round(m.centroid_x))
        cy = int(round(m.centroid_y))
        if len(contour) >= 5:
            ellipse = cv2.fitEllipse(contour)
            cv2.ellipse(out, ellipse, (255, 0, 0), 1)
        cv2.putText(
            out,
            f"{m.aspect_ratio:.2f}",
            (cx + 2, cy - 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.30,
            (0, 0, 255),
            1,
            cv2.LINE_AA,
        )
    return out


def compute_summary(measurements: list[ParticleMeasurement]) -> dict:
    if not measurements:
        return {
            "count": 0,
            "mean_aspect_ratio": None,
            "median_aspect_ratio": None,
            "std_aspect_ratio": None,
            "min_aspect_ratio": None,
            "max_aspect_ratio": None,
        }

    arr = np.array([m.aspect_ratio for m in measurements], dtype=np.float32)
    return {
        "count": int(len(measurements)),
        "mean_aspect_ratio": float(np.mean(arr)),
        "median_aspect_ratio": float(np.median(arr)),
        "std_aspect_ratio": float(np.std(arr)),
        "min_aspect_ratio": float(np.min(arr)),
        "max_aspect_ratio": float(np.max(arr)),
    }


def save_measurements_csv(path: Path, measurements: list[ParticleMeasurement]) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
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
            ],
        )
        writer.writeheader()
        for m in measurements:
            writer.writerow(asdict(m))


def save_json(path: Path, data: dict) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def run(args: argparse.Namespace) -> None:
    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    ensure_dir(output_dir)

    gray = load_gray(str(input_path))
    bgr = load_bgr(str(input_path))

    mask, debug = segment_classical(gray)
    contours = extract_contours(mask)
    filtered_contours, measurements = filter_measurable_rods(
        contours=contours,
        min_area=args.min_area,
        max_area=args.max_area,
        min_aspect_ratio=args.min_aspect_ratio,
        max_aspect_ratio=args.max_aspect_ratio,
        min_solidity=args.min_solidity,
        max_solidity=args.max_solidity,
        min_extent=args.min_extent,
        max_extent=args.max_extent,
        border_margin=args.border_margin,
        image_shape=gray.shape,
    )

    overlay = draw_measurements_overlay(bgr, filtered_contours, measurements)
    summary = compute_summary(measurements)

    save_image(output_dir / "01_input_gray.png", gray)
    for name, img in debug.items():
        save_image(output_dir / f"02_{name}.png", img)
    save_image(output_dir / "03_final_mask.png", mask)
    save_image(output_dir / "04_overlay.png", overlay)
    save_measurements_csv(output_dir / "measurements.csv", measurements)
    save_json(output_dir / "summary.json", summary)

    print(json.dumps(summary, ensure_ascii=False, indent=2))


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Classical segmentation for precursor SEM primary-particle aspect ratio")
    p.add_argument("--input", type=str, required=True, help="입력 이미지 경로")
    p.add_argument("--output_dir", type=str, required=True, help="출력 폴더")
    p.add_argument("--min_area", type=float, default=18.0)
    p.add_argument("--max_area", type=float, default=900.0)
    p.add_argument("--min_aspect_ratio", type=float, default=2.0)
    p.add_argument("--max_aspect_ratio", type=float, default=12.0)
    p.add_argument("--min_solidity", type=float, default=0.35)
    p.add_argument("--max_solidity", type=float, default=0.95)
    p.add_argument("--min_extent", type=float, default=0.18)
    p.add_argument("--max_extent", type=float, default=0.85)
    p.add_argument("--border_margin", type=int, default=8)
    return p


def main() -> None:
    args = build_parser().parse_args()
    run(args)


if __name__ == "__main__":
    main()
