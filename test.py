#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Precursor Aspect Ratio Test - Integrated Pipeline for Particle Measurement
==========================================================================
이 스크립트는 SEM 이미지에서 입자(primary particles)를 검출하고 형상 특징을 측정하는
통합 파이프라인의 테스트 및 검증을 위해 설계되었습니다.

주요 기능:
1. 고전 영상 처리(Classical CV) 기반 세그멘테이션
2. 입자별 형태학적 특징(면적, 둘레, 종횡비 등) 계산
3. 필터링 기준에 따른 유효 입자 선별
4. 결과 요약 통계 생성 및 시각화 저장
==========================================================================
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
    개별 입자의 상세 측정 정보를 저장하는 데이터 클래스
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
class TestPipelineConfig:
    """
    테스트 파이프라인 실행을 위한 상세 설정
    """
    path_input: Path
    path_outputDir: Path
    float_minArea: float = 20.0
    float_maxArea: float = 5000.0
    float_minAspectRatio: float = 1.5
    float_maxAspectRatio: float = 20.0
    float_minSolidity: float = 0.2
    float_maxSolidity: float = 1.0
    float_minExtent: float = 0.1
    float_maxExtent: float = 1.0


class TestPipelineService:
    """
    입자 측정 및 검증을 수행하는 서비스 클래스
    """

    def __init__(self, obj_config: TestPipelineConfig) -> None:
        self.obj_config = obj_config

    def load_image_gray(self) -> np.ndarray:
        """그레이스케일 이미지 로드"""
        str_path = str(self.obj_config.path_input)
        arr_img = cv2.imread(str_path, cv2.IMREAD_GRAYSCALE)
        if arr_img is None:
            raise FileNotFoundError(f"이미지를 읽을 수 없습니다: {str_path}")
        return arr_img

    def load_image_bgr(self) -> np.ndarray:
        """BGR 컬러 이미지 로드"""
        str_path = str(self.obj_config.path_input)
        arr_img = cv2.imread(str_path, cv2.IMREAD_COLOR)
        if arr_img is None:
            raise FileNotFoundError(f"이미지를 읽을 수 없습니다: {str_path}")
        return arr_img

    def normalize_to_uint8(self, arr_img: np.ndarray) -> np.ndarray:
        """이미지 정규화 (0~255 uint8)"""
        arr_f32 = arr_img.astype(np.float32)
        float_mn = float(arr_f32.min())
        float_mx = float(arr_f32.max())
        if float_mx - float_mn < 1e-8:
            return np.zeros_like(arr_img, dtype=np.uint8)
        arr_out = (arr_f32 - float_mn) / (float_mx - float_mn)
        return (arr_out * 255.0).clip(0, 255).astype(np.uint8)

    def preprocess_image(self, arr_gray: np.ndarray) -> np.ndarray:
        """전처리: 대비 강화 및 엣지 성분 결합"""
        obj_clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        arr_enhanced = obj_clahe.apply(arr_gray)
        arr_denoised = cv2.GaussianBlur(arr_enhanced, (3, 3), 0)

        arr_lap = cv2.Laplacian(arr_denoised, cv2.CV_32F, ksize=3)
        arr_lapAbs = self.normalize_to_uint8(np.abs(arr_lap))

        arr_combined = cv2.addWeighted(arr_denoised, 0.8, arr_lapAbs, 0.2, 0)
        return arr_combined

    def segment_image(self, arr_gray: np.ndarray) -> tp.Tuple[np.ndarray, tp.Dict[str, np.ndarray]]:
        """Adaptive thresholding 기반 세그멘테이션"""
        arr_pre = self.preprocess_image(arr_gray)
        arr_binary = cv2.adaptiveThreshold(
            arr_pre,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            21,
            -3,
        )

        arr_kernelOpen = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
        arr_kernelClose = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

        arr_cleaned = cv2.morphologyEx(arr_binary, cv2.MORPH_OPEN, arr_kernelOpen, iterations=1)
        arr_cleaned = cv2.morphologyEx(arr_cleaned, cv2.MORPH_CLOSE, arr_kernelClose, iterations=1)

        dict_debug = {
            "preprocessed": arr_pre,
            "binary": arr_binary,
            "cleaned": arr_cleaned,
        }
        return arr_cleaned, dict_debug

    def calculate_measurement(self, arr_cnt: np.ndarray, int_idx: int) -> tp.Optional[ParticleMeasurement]:
        """Contour 특징 측정"""
        float_area = float(cv2.contourArea(arr_cnt))
        if float_area <= 1.0:
            return None

        float_perimeter = float(cv2.arcLength(arr_cnt, True))
        if len(arr_cnt) < 5:
            int_x, int_y, int_w, int_h = cv2.boundingRect(arr_cnt)
            float_major = float(max(int_w, int_h))
            float_minor = float(max(1.0, min(int_w, int_h)))
            float_angle = 0.0
            float_cx = float(int_x + int_w / 2.0)
            float_cy = float(int_y + int_h / 2.0)
        else:
            (float_cx, float_cy), (float_a1, float_a2), float_angle = cv2.fitEllipse(arr_cnt)
            float_major = float(max(float_a1, float_a2))
            float_minor = float(max(1e-6, min(float_a1, float_a2)))

        float_ar = float_major / float_minor

        arr_hull = cv2.convexHull(arr_cnt)
        float_hullArea = float(cv2.contourArea(arr_hull))
        float_solidity = float_area / float_hullArea if float_hullArea > 1e-8 else 0.0

        int_x, int_y, int_w, int_h = cv2.boundingRect(arr_cnt)
        float_rectArea = float(int_w * int_h)
        float_extent = float_area / float_rectArea if float_rectArea > 1e-8 else 0.0

        return ParticleMeasurement(
            int_contourIndex=int_idx,
            float_area=float_area,
            float_perimeter=float_perimeter,
            float_majorAxis=float_major,
            float_minorAxis=float_minor,
            float_aspectRatio=float_ar,
            float_angleDeg=float(float_angle),
            float_centroidX=float_cx,
            float_centroidY=float_cy,
            float_solidity=float_solidity,
            float_extent=float_extent,
        )

    def process(self) -> tp.Dict[str, tp.Any]:
        """전체 파이프라인 프로세스 실행"""
        self.obj_config.path_outputDir.mkdir(parents=True, exist_ok=True)
        
        arr_gray = self.load_image_gray()
        arr_bgr = self.load_image_bgr()

        arr_mask, dict_debug = self.segment_image(arr_gray)
        list_contours, _ = cv2.findContours(arr_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        list_keptContours = list()
        list_measurements = list()

        for int_idx, arr_cnt in enumerate(list_contours):
            obj_m = self.calculate_measurement(arr_cnt, int_idx)
            if obj_m is None:
                continue

            if not (self.obj_config.float_minArea <= obj_m.float_area <= self.obj_config.float_maxArea):
                continue
            if not (self.obj_config.float_minAspectRatio <= obj_m.float_aspectRatio <= self.obj_config.float_maxAspectRatio):
                continue
            if not (self.obj_config.float_minSolidity <= obj_m.float_solidity <= self.obj_config.float_maxSolidity):
                continue
            if not (self.obj_config.float_minExtent <= obj_m.float_extent <= self.obj_config.float_maxExtent):
                continue

            list_keptContours.append(arr_cnt)
            list_measurements.append(obj_m)

        # 요약 통계
        dict_summary = dict()
        if not list_measurements:
            dict_summary = {"count": 0, "mean_aspect_ratio": None}
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

        # 시각화
        arr_overlay = arr_bgr.copy()
        cv2.drawContours(arr_overlay, list_keptContours, -1, (0, 255, 0), 1)
        for obj_m in list_measurements:
            int_cx = int(round(obj_m.float_centroidX))
            int_cy = int(round(obj_m.float_centroidY))
            cv2.putText(
                arr_overlay, f"{obj_m.float_aspectRatio:.2f}",
                (int_cx + 3, int_cy - 3),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1, cv2.LINE_AA
            )

        # 파일 저장
        cv2.imwrite(str(self.obj_config.path_outputDir / "01_input_gray.png"), arr_gray)
        for str_name, arr_img in dict_debug.items():
            cv2.imwrite(str(self.obj_config.path_outputDir / f"02_{str_name}.png"), arr_img)
        cv2.imwrite(str(self.obj_config.path_outputDir / "03_mask.png"), arr_mask)
        cv2.imwrite(str(self.obj_config.path_outputDir / "04_overlay.png"), arr_overlay)

        # CSV/JSON 저장
        path_csv = self.obj_config.path_outputDir / "measurements.csv"
        with path_csv.open("w", newline="", encoding="utf-8-sig") as obj_f:
            if list_measurements:
                obj_writer = csv.DictWriter(obj_f, fieldnames=list(asdict(list_measurements[0]).keys()))
                obj_writer.writeheader()
                for obj_m in list_measurements:
                    obj_writer.writerow(asdict(obj_m))

        with (self.obj_config.path_outputDir / "summary.json").open("w", encoding="utf-8") as obj_f:
            json.dump(dict_summary, obj_f, ensure_ascii=False, indent=2)

        return dict_summary


def run_test_interactive(
    str_input: str,
    str_outputDir: str,
    float_minArea: float = 20.0,
    float_maxArea: float = 5000.0,
    float_minAspectRatio: float = 1.5,
    float_maxAspectRatio: float = 20.0,
    float_minSolidity: float = 0.2,
    float_maxSolidity: float = 1.0,
    float_minExtent: float = 0.1,
    float_maxExtent: float = 1.0,
) -> tp.Dict[str, tp.Any]:
    """Interactive Window에서 호출 가능한 래퍼 함수"""
    obj_config = TestPipelineConfig(
        path_input=Path(str_input),
        path_outputDir=Path(str_outputDir),
        float_minArea=float_minArea,
        float_maxArea=float_maxArea,
        float_minAspectRatio=float_minAspectRatio,
        float_maxAspectRatio=float_maxAspectRatio,
        float_minSolidity=float_minSolidity,
        float_maxSolidity=float_maxSolidity,
        float_minExtent=float_minExtent,
        float_maxExtent=float_maxExtent
    )
    obj_service = TestPipelineService(obj_config)
    return obj_service.process()


def main() -> None:
    """
    진입점: 아래 변수들을 직접 수정하여 실행하세요.
    """
    # === 사용자 설정 영역 ===
    str_inputPath = "test.png"          # 입력 이미지 경로
    str_outputDir = "output_test"       # 결과 저장 폴더
    
    float_minArea = 20.0
    float_maxArea = 5000.0
    float_minAspectRatio = 1.5
    float_maxAspectRatio = 20.0
    float_minSolidity = 0.2
    float_maxSolidity = 1.0
    float_minExtent = 0.1
    float_maxExtent = 1.0
    # ========================

    dict_summary = run_test_interactive(
        str_input=str_inputPath,
        str_outputDir=str_outputDir,
        float_minArea=float_minArea,
        float_maxArea=float_maxArea,
        float_minAspectRatio=float_minAspectRatio,
        float_maxAspectRatio=float_maxAspectRatio,
        float_minSolidity=float_minSolidity,
        float_maxSolidity=float_maxSolidity,
        float_minExtent=float_minExtent,
        float_maxExtent=float_maxExtent
    )
    
    print("===== 결과 요약 =====")
    print(json.dumps(dict_summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
