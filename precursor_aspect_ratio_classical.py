#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Classical Aspect Ratio - SEM primary-particle aspect ratio measurement using classical CV
========================================================================================
이 스크립트는 고전적인 영상 처리 기법을 사용하여 SEM 이미지 내의 rod-like 입자를 검출하고
종횡비(Aspect Ratio)를 측정합니다.

전체 처리 흐름:
1. 입력 이미지 로드 (grayscale + BGR)
2. secondary particle ROI 검출
   - 하단 스케일바 제거
   - threshold + morphology로 가장 큰 입자 영역 추출
3. ROI 내부에서 rod-like texture 강조
   - CLAHE, gradient, blackhat, Laplacian 결합
4. adaptive threshold로 binary segmentation 수행
5. morphology(open/close)로 노이즈 제거
6. contour 추출 및 형상 정보 계산
7. 조건 기반 필터링 (측정 가능한 입자만 선택)
8. 결과 저장 및 시각화
========================================================================================
"""

from __future__ import annotations
import typing as tp

import csv
import json
from dataclasses import dataclass, asdict
from pathlib import Path

import cv2
import numpy as np


@dataclass
class ParticleMeasurement:
    """
    개별 입자의 측정 정보를 담는 데이터 클래스
    """
    int_contourIndex: int
    float_area: float
    float_perimeter: float
    float_majorAxis: float
    float_minorAxis: float
    float_aspectRatio: float
    float_angleDeg: float
    float_centroidX: float
    float_centroidY: float
    float_solidity: float
    float_extent: float


@dataclass
class AspectRatioConfig:
    """
    Aspect Ratio 측정을 위한 설정 데이터 클래스
    """
    path_input: Path
    path_outputDir: Path
    float_minArea: float = 18.0
    float_maxArea: float = 900.0
    float_minAspectRatio: float = 2.0
    float_maxAspectRatio: float = 12.0
    float_minSolidity: float = 0.35
    float_maxSolidity: float = 0.95
    float_minExtent: float = 0.18
    float_maxExtent: float = 0.85
    int_borderMargin: int = 8


class AspectRatioService:
    """
    Classical CV 기반 종횡비 측정 핵심 로직 서비스
    """

    def __init__(self, obj_config: AspectRatioConfig) -> None:
        self.obj_config = obj_config

    def load_gray(self) -> np.ndarray:
        """그레이스케일 이미지 로드"""
        str_path = str(self.obj_config.path_input)
        arr_img = cv2.imread(str_path, cv2.IMREAD_GRAYSCALE)
        if arr_img is None:
            raise FileNotFoundError(f"이미지를 읽을 수 없습니다: {str_path}")
        return arr_img

    def load_bgr(self) -> np.ndarray:
        """BGR 컬러 이미지 로드"""
        str_path = str(self.obj_config.path_input)
        arr_img = cv2.imread(str_path, cv2.IMREAD_COLOR)
        if arr_img is None:
            raise FileNotFoundError(f"이미지를 읽을 수 없습니다: {str_path}")
        return arr_img

    def normalize_uint8(self, arr_img: np.ndarray) -> np.ndarray:
        """이미지 정규화 (0~255 uint8)"""
        arr_f32 = arr_img.astype(np.float32)
        float_mn = float(arr_f32.min())
        float_mx = float(arr_f32.max())
        if float_mx - float_mn < 1e-8:
            return np.zeros_like(arr_img, dtype=np.uint8)
        arr_out = (arr_f32 - float_mn) / (float_mx - float_mn)
        return (arr_out * 255.0).clip(0, 255).astype(np.uint8)

    def remove_bottom_annotation(self, arr_gray: np.ndarray, float_cropRatio: float = 0.12) -> np.ndarray:
        """하단 스케일바 영역 제거"""
        int_h, _ = arr_gray.shape
        arr_out = arr_gray.copy()
        int_cutY = int(int_h * (1.0 - float_cropRatio))
        arr_out[int_cutY:, :] = 0
        return arr_out

    def detect_roi(self, arr_gray: np.ndarray) -> np.ndarray:
        """Secondary particle ROI 검출"""
        arr_work = self.remove_bottom_annotation(arr_gray, float_cropRatio=0.12)
        arr_blur = cv2.GaussianBlur(arr_work, (9, 9), 0)
        _, arr_th = cv2.threshold(arr_blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        arr_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
        arr_th = cv2.morphologyEx(arr_th, cv2.MORPH_CLOSE, arr_kernel, iterations=2)
        arr_th = cv2.morphologyEx(arr_th, cv2.MORPH_OPEN, arr_kernel, iterations=1)

        list_contours, _ = cv2.findContours(arr_th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        arr_roiMask = np.zeros_like(arr_gray, dtype=np.uint8)

        if not list_contours:
            arr_roiMask[:, :] = 255
            return arr_roiMask

        arr_largest = max(list_contours, key=cv2.contourArea)
        cv2.drawContours(arr_roiMask, [arr_largest], -1, 255, thickness=-1)

        arr_erodeKernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        arr_roiMask = cv2.erode(arr_roiMask, arr_erodeKernel, iterations=1)
        return arr_roiMask

    def enhance_texture(self, arr_grayRoi: np.ndarray) -> tp.Dict[str, np.ndarray]:
        """rod-like texture 강조"""
        obj_clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
        arr_claheImg = obj_clahe.apply(arr_grayRoi)
        arr_blur = cv2.GaussianBlur(arr_claheImg, (3, 3), 0)

        arr_kernelBh = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
        arr_blackhat = cv2.morphologyEx(arr_blur, cv2.MORPH_BLACKHAT, arr_kernelBh)

        arr_kernelGrad = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        arr_gradient = cv2.morphologyEx(arr_blur, cv2.MORPH_GRADIENT, arr_kernelGrad)

        arr_lap = cv2.Laplacian(arr_blur, cv2.CV_32F, ksize=3)
        arr_lapAbs = self.normalize_uint8(np.abs(arr_lap))

        arr_combined = cv2.addWeighted(arr_gradient, 0.45, arr_blackhat, 0.35, 0)
        arr_combined = cv2.addWeighted(arr_combined, 0.80, arr_lapAbs, 0.20, 0)
        arr_combined = self.normalize_uint8(arr_combined)

        return {
            "clahe": arr_claheImg,
            "blur": arr_blur,
            "blackhat": arr_blackhat,
            "gradient": arr_gradient,
            "laplacian_abs": arr_lapAbs,
            "combined": arr_combined,
        }

    def segment(self, arr_gray: np.ndarray) -> tp.Tuple[np.ndarray, tp.Dict[str, np.ndarray]]:
        """Classical segmentation 파이프라인"""
        arr_roiMask = self.detect_roi(arr_gray)
        arr_grayRoi = arr_gray.copy()
        arr_grayRoi[arr_roiMask == 0] = 0
        
        dict_enhanced = self.enhance_texture(arr_grayRoi)
        arr_work = dict_enhanced["combined"]

        arr_binary = cv2.adaptiveThreshold(
            arr_work,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            19,
            -2,
        )
        arr_binary[arr_roiMask == 0] = 0

        arr_kernelOpen = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
        arr_opened = cv2.morphologyEx(arr_binary, cv2.MORPH_OPEN, arr_kernelOpen, iterations=1)

        arr_kernelClose = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
        arr_cleaned = cv2.morphologyEx(arr_opened, cv2.MORPH_CLOSE, arr_kernelClose, iterations=1)

        dict_debug = dict()
        dict_debug["roi_mask"] = arr_roiMask
        dict_debug["gray_roi"] = arr_grayRoi
        for str_k, arr_v in dict_enhanced.items():
            dict_debug[str_k] = arr_v
        dict_debug["binary"] = arr_binary
        dict_debug["opened"] = arr_opened
        dict_debug["cleaned"] = arr_cleaned
        
        return arr_cleaned, dict_debug

    def measure_contour(self, arr_contour: np.ndarray, int_index: int) -> tp.Optional[ParticleMeasurement]:
        """개별 contour 측정"""
        float_area = float(cv2.contourArea(arr_contour))
        if float_area <= 1.0:
            return None

        float_perimeter = float(cv2.arcLength(arr_contour, True))
        if len(arr_contour) < 5:
            int_x, int_y, int_w, int_h = cv2.boundingRect(arr_contour)
            float_major = float(max(int_w, int_h))
            float_minor = float(max(1.0, min(int_w, int_h)))
            float_angle = 0.0
            float_cx = float(int_x + int_w / 2.0)
            float_cy = float(int_y + int_h / 2.0)
        else:
            (float_cx, float_cy), (float_a1, float_a2), float_angle = cv2.fitEllipse(arr_contour)
            float_major = float(max(float_a1, float_a2))
            float_minor = float(max(1e-6, min(float_a1, float_a2)))

        float_ar = float_major / float_minor

        arr_hull = cv2.convexHull(arr_contour)
        float_hullArea = float(cv2.contourArea(arr_hull))
        float_solidity = float_area / float_hullArea if float_hullArea > 1e-8 else 0.0

        int_x, int_y, int_w, int_h = cv2.boundingRect(arr_contour)
        float_rectArea = float(int_w * int_h)
        float_extent = float_area / float_rectArea if float_rectArea > 1e-8 else 0.0

        return ParticleMeasurement(
            int_contourIndex=int_index,
            float_area=float_area,
            float_perimeter=float_perimeter,
            float_majorAxis=float_major,
            float_minorAxis=float_minor,
            float_aspectRatio=float_ar,
            float_angleDeg=float(float_angle),
            float_centroidX=float(float_cx),
            float_centroidY=float(float_cy),
            float_solidity=float_solidity,
            float_extent=float_extent,
        )

    def process(self) -> tp.Dict[str, tp.Any]:
        """전체 파이프라인 실행"""
        self.obj_config.path_outputDir.mkdir(parents=True, exist_ok=True)
        
        arr_gray = self.load_gray()
        arr_bgr = self.load_bgr()
        int_h, int_w = arr_gray.shape

        arr_mask, dict_debug = self.segment(arr_gray)
        list_allContours, _ = cv2.findContours(arr_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        list_filteredContours = list()
        list_measurements = list()

        for int_idx, arr_cnt in enumerate(list_allContours):
            obj_m = self.measure_contour(arr_cnt, int_idx)
            if obj_m is None:
                continue
            
            # Filtering criteria
            if not (self.obj_config.float_minArea <= obj_m.float_area <= self.obj_config.float_maxArea):
                continue
            if not (self.obj_config.float_minAspectRatio <= obj_m.float_aspectRatio <= self.obj_config.float_maxAspectRatio):
                continue
            if not (self.obj_config.float_minSolidity <= obj_m.float_solidity <= self.obj_config.float_maxSolidity):
                continue
            if not (self.obj_config.float_minExtent <= obj_m.float_extent <= self.obj_config.float_maxExtent):
                continue
            
            int_m = self.obj_config.int_borderMargin
            if (obj_m.float_centroidX < int_m or obj_m.float_centroidX > (int_w - int_m) or
                obj_m.float_centroidY < int_m or obj_m.float_centroidY > (int_h - int_m)):
                continue

            list_filteredContours.append(arr_cnt)
            list_measurements.append(obj_m)

        # Visualization
        arr_overlay = arr_bgr.copy()
        cv2.drawContours(arr_overlay, list_filteredContours, -1, (0, 255, 0), 1)
        for obj_m in list_measurements:
            int_cx = int(round(obj_m.float_centroidX))
            int_cy = int(round(obj_m.float_centroidY))
            cv2.putText(
                arr_overlay,
                f"{obj_m.float_aspectRatio:.2f}",
                (int_cx + 2, int_cy - 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.30,
                (0, 0, 255),
                1,
                cv2.LINE_AA,
            )

        # Summary statistics
        dict_summary = dict()
        if not list_measurements:
            dict_summary = {
                "count": 0,
                "mean_aspect_ratio": None,
                "median_aspect_ratio": None,
                "std_aspect_ratio": None,
            }
        else:
            arr_ars = np.array([obj_m.float_aspectRatio for obj_m in list_measurements], dtype=np.float32)
            dict_summary = {
                "count": len(list_measurements),
                "mean_aspect_ratio": float(np.mean(arr_ars)),
                "median_aspect_ratio": float(np.median(arr_ars)),
                "std_aspect_ratio": float(np.std(arr_ars)),
                "min_aspect_ratio": float(np.min(arr_ars)),
                "max_aspect_ratio": float(np.max(arr_ars)),
            }

        # Save results
        cv2.imwrite(str(self.obj_config.path_outputDir / "01_input_gray.png"), arr_gray)
        for str_name, arr_img in dict_debug.items():
            cv2.imwrite(str(self.obj_config.path_outputDir / f"02_{str_name}.png"), arr_img)
        cv2.imwrite(str(self.obj_config.path_outputDir / "03_final_mask.png"), arr_mask)
        cv2.imwrite(str(self.obj_config.path_outputDir / "04_overlay.png"), arr_overlay)

        # CSV Save
        path_csv = self.obj_config.path_outputDir / "measurements.csv"
        with path_csv.open("w", newline="", encoding="utf-8-sig") as obj_f:
            if list_measurements:
                obj_writer = csv.DictWriter(obj_f, fieldnames=list(asdict(list_measurements[0]).keys()))
                obj_writer.writeheader()
                for obj_m in list_measurements:
                    obj_writer.writerow(asdict(obj_m))

        # JSON Save
        with (self.obj_config.path_outputDir / "summary.json").open("w", encoding="utf-8") as obj_f:
            json.dump(dict_summary, obj_f, ensure_ascii=False, indent=2)

        return dict_summary


def run_classical_interactive(
    str_input: str,
    str_outputDir: str,
    float_minArea: float = 18.0,
    float_maxArea: float = 900.0,
    float_minAspectRatio: float = 2.0,
    float_maxAspectRatio: float = 12.0,
    float_minSolidity: float = 0.35,
    float_maxSolidity: float = 0.95,
    float_minExtent: float = 0.18,
    float_maxExtent: float = 0.85,
    int_borderMargin: int = 8,
) -> tp.Dict[str, tp.Any]:
    """Interactive Window에서 호출 가능한 래퍼 함수"""
    obj_config = AspectRatioConfig(
        path_input=Path(str_input),
        path_outputDir=Path(str_outputDir),
        float_minArea=float_minArea,
        float_maxArea=float_maxArea,
        float_minAspectRatio=float_minAspectRatio,
        float_maxAspectRatio=float_maxAspectRatio,
        float_minSolidity=float_minSolidity,
        float_maxSolidity=float_maxSolidity,
        float_minExtent=float_minExtent,
        float_maxExtent=float_maxExtent,
        int_borderMargin=int_borderMargin,
    )
    obj_service = AspectRatioService(obj_config)
    return obj_service.process()


def main() -> None:
    """
    진입점: 아래 변수들을 직접 수정하여 실행하세요.
    """
    # === 사용자 설정 영역 ===
    str_inputPath = "test.png"              # 입력 이미지 경로
    str_outputDir = "out_classical"         # 결과 저장 폴더
    
    float_minArea = 18.0
    float_maxArea = 900.0
    float_minAspectRatio = 2.0
    float_maxAspectRatio = 12.0
    float_minSolidity = 0.35
    float_maxSolidity = 0.95
    float_minExtent = 0.18
    float_maxExtent = 0.85
    int_borderMargin = 8
    # ========================

    dict_summary = run_classical_interactive(
        str_input=str_inputPath,
        str_outputDir=str_outputDir,
        float_minArea=float_minArea,
        float_maxArea=float_maxArea,
        float_minAspectRatio=float_minAspectRatio,
        float_maxAspectRatio=float_maxAspectRatio,
        float_minSolidity=float_minSolidity,
        float_maxSolidity=float_maxSolidity,
        float_minExtent=float_minExtent,
        float_maxExtent=float_maxExtent,
        int_borderMargin=int_borderMargin,
    )
    
    print("===== Classical CV 결과 요약 =====")
    print(json.dumps(dict_summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
