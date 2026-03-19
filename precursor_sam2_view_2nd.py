#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
SAM2 Tiled View - Small object segmentation with tiling and interest point sampling
==================================================================================
이 스크립트는 큰 이미지 내의 작은 입자(primary particles)를 정밀하게 검출하기 위해
타일링(tiling)과 관심점 샘플링(interest point sampling)을 결합한 SAM2 추론을 수행합니다.

주요 처리 단계:
1. 입자 ROI 검출: 하단 스케일바 제거 및 큰 구형 입자(secondary particle) 영역 추출
2. 타일 생성: ROI 내부를 일정 크기의 타일로 분할하여 추론 해상도 확보
3. 관심점 샘플링: 각 타일 내에서 texture 강화를 통해 입자 후보점(interest points) 추출
4. SAM2 추론: 각 관심점을 프롬프트로 사용하여 개별 입자 마스크 생성
5. 마스크 필터링 및 중복 제거: 면적, 종횡비, Solidity 기준 필터링 및 IoU 기반 중복 제거
6. 시각화 및 결과 저장: 오버레이 이미지, 개별 마스크, 통계 데이터 저장
==================================================================================
"""

from __future__ import annotations
import typing as tp

import json
from dataclasses import dataclass, asdict
from pathlib import Path

import cv2
import numpy as np
from ultralytics import SAM


@dataclass
class Sam2MaskItem:
    """
    개별 검출 마스크의 상세 정보 데이터 클래스
    """
    int_index: int
    float_areaPx: float
    list_bboxXyxy: tp.List[float]
    float_aspectRatio: float
    float_solidity: float
    float_score: tp.Optional[float]
    list_tileXyxy: tp.List[int]
    list_promptXyGlobal: tp.List[int]


@dataclass
class Sam2TiledConfig:
    """
    Tiled SAM2 실행을 위한 상세 설정 데이터 클래스
    """
    path_inputPath: Path
    path_outputDir: Path
    str_modelName: str = "sam2.1_b.pt"
    str_device: tp.Optional[str] = None
    int_imgSize: int = 1024
    bool_retinaMasks: bool = False
    int_tileSize: int = 512
    int_stride: int = 384
    int_pointsPerTile: int = 24
    float_qualityLevel: float = 0.03
    int_pointMinDistance: int = 14
    float_minArea: float = 18.0
    float_maxArea: float = 900.0
    float_minAspectRatio: float = 1.8
    float_maxAspectRatio: float = 12.0
    float_minSolidity: float = 0.25
    float_maxSolidity: float = 0.98
    int_borderMargin: int = 5
    float_dedupIou: float = 0.60
    bool_saveIndividualMasks: bool = False


# =========================================================
# Utility Functions
# =========================================================

def load_image_bgr(str_path: str) -> np.ndarray:
    """이미지를 BGR 컬러 형식으로 로드"""
    arr_img = cv2.imread(str_path, cv2.IMREAD_COLOR)
    if arr_img is None:
        raise FileNotFoundError(f"이미지를 읽을 수 없습니다: {str_path}")
    return arr_img


def load_image_gray(str_path: str) -> np.ndarray:
    """이미지를 그레이스케일 형식으로 로드"""
    arr_img = cv2.imread(str_path, cv2.IMREAD_GRAYSCALE)
    if arr_img is None:
        raise FileNotFoundError(f"이미지를 읽을 수 없습니다: {str_path}")
    return arr_img


def save_image_to_path(path_target: Path, arr_img: np.ndarray) -> None:
    """이미지 파일 저장"""
    cv2.imwrite(str(path_target), arr_img)


def generate_color_by_index(int_index: int) -> tp.Tuple[int, int, int]:
    """인덱스 기반 고유 색상 생성"""
    obj_rng = np.random.default_rng(seed=int_index + 1234)
    arr_bgr = obj_rng.integers(64, 256, size=3, dtype=np.uint8)
    return (int(arr_bgr[0]), int(arr_bgr[1]), int(arr_bgr[2]))


def normalize_image_to_uint8(arr_img: np.ndarray) -> np.ndarray:
    """이미지를 0~255 uint8 범위로 정규화"""
    arr_f32 = arr_img.astype(np.float32)
    float_mn = float(arr_f32.min())
    float_mx = float(arr_f32.max())
    if float_mx - float_mn < 1e-8:
        return np.zeros_like(arr_img, dtype=np.uint8)
    arr_out = (arr_f32 - float_mn) / (float_mx - float_mn)
    return (arr_out * 255.0).clip(0, 255).astype(np.uint8)


# =========================================================
# ROI and Tiling Logic
# =========================================================

def mask_bottom_annotation(arr_gray: np.ndarray, float_cropRatio: float = 0.12) -> np.ndarray:
    """이미지 하단 정보 영역(Scale bar 등)을 검은색으로 마스킹"""
    int_h, _ = arr_gray.shape
    arr_out = arr_gray.copy()
    int_cutY = int(int_h * (1.0 - float_cropRatio))
    arr_out[int_cutY:, :] = 0
    return arr_out


def detect_particle_roi(arr_gray: np.ndarray) -> tp.Tuple[np.ndarray, tp.Tuple[int, int, int, int]]:
    """Secondary particle (전체 입자 덩어리) ROI 검출"""
    arr_work = mask_bottom_annotation(arr_gray, float_cropRatio=0.12)
    arr_blur = cv2.GaussianBlur(arr_work, (9, 9), 0)
    _, arr_th = cv2.threshold(arr_blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    arr_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    arr_th = cv2.morphologyEx(arr_th, cv2.MORPH_CLOSE, arr_kernel, iterations=2)
    arr_th = cv2.morphologyEx(arr_th, cv2.MORPH_OPEN, arr_kernel, iterations=1)

    list_contours, _ = cv2.findContours(arr_th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    arr_mask = np.zeros_like(arr_gray, dtype=np.uint8)
    if not list_contours:
        arr_mask[:, :] = 255
        int_h, int_w = arr_gray.shape
        return arr_mask, (0, 0, int_w, int_h)

    arr_largest = max(list_contours, key=cv2.contourArea)
    cv2.drawContours(arr_mask, [arr_largest], -1, 255, thickness=-1)
    arr_erodeKernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    arr_mask = cv2.erode(arr_mask, arr_erodeKernel, iterations=1)
    int_x, int_y, int_w, int_h = cv2.boundingRect(arr_largest)
    return arr_mask, (int_x, int_y, int_x + int_w, int_y + int_h)


def create_processing_tiles(
    int_x1: int, int_y1: int, int_x2: int, int_y2: int, int_tileSize: int, int_stride: int
) -> tp.List[tp.Tuple[int, int, int, int]]:
    """ROI 내부를 타일로 분할"""
    list_tiles = list()
    if int_x2 - int_x1 <= int_tileSize and int_y2 - int_y1 <= int_tileSize:
        list_tiles.append((int_x1, int_y1, int_x2, int_y2))
        return list_tiles

    list_xs = list(range(int_x1, max(int_x1 + 1, int_x2 - int_tileSize + 1), int_stride))
    list_ys = list(range(int_y1, max(int_y1 + 1, int_y2 - int_tileSize + 1), int_stride))

    if list_xs and list_xs[-1] != int_x2 - int_tileSize:
        list_xs.append(max(int_x1, int_x2 - int_tileSize))
    if list_ys and list_ys[-1] != int_y2 - int_tileSize:
        list_ys.append(max(int_y1, int_y2 - int_tileSize))

    if not list_xs:
        list_xs = [int_x1]
    if not list_ys:
        list_ys = [int_y1]

    for int_yy in list_ys:
        for int_xx in list_xs:
            int_tx1 = int_xx
            int_ty1 = int_yy
            int_tx2 = min(int_xx + int_tileSize, int_x2)
            int_ty2 = min(int_yy + int_tileSize, int_y2)
            list_tiles.append((int_tx1, int_ty1, int_tx2, int_ty2))
    return list_tiles


def enhance_image_texture(arr_tileGray: np.ndarray) -> np.ndarray:
    """입자 특징(edge, contrast)을 강화하여 관심점 추출 준비"""
    obj_clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    arr_img = obj_clahe.apply(arr_tileGray)
    arr_blur = cv2.GaussianBlur(arr_img, (3, 3), 0)
    
    arr_kernelGrad = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    arr_grad = cv2.morphologyEx(arr_blur, cv2.MORPH_GRADIENT, arr_kernelGrad)
    
    arr_kernelBh = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    arr_blackhat = cv2.morphologyEx(arr_blur, cv2.MORPH_BLACKHAT, arr_kernelBh)
    
    arr_lap = cv2.Laplacian(arr_blur, cv2.CV_32F, ksize=3)
    arr_lapAbs = normalize_image_to_uint8(np.abs(arr_lap))
    
    arr_combined = cv2.addWeighted(arr_grad, 0.45, arr_blackhat, 0.35, 0)
    arr_combined = cv2.addWeighted(arr_combined, 0.8, arr_lapAbs, 0.2, 0)
    return normalize_image_to_uint8(arr_combined)


def sample_interest_points(
    arr_tileGray: np.ndarray,
    int_maxPoints: int,
    int_minDistance: int,
    float_qualityLevel: float,
    arr_mask: tp.Optional[np.ndarray] = None,
) -> tp.List[tp.Tuple[int, int]]:
    """타일 내에서 입자 후보 위치(interest points) 샘플링"""
    arr_enhanced = enhance_image_texture(arr_tileGray)
    arr_corners = cv2.goodFeaturesToTrack(
        arr_enhanced,
        maxCorners=int_maxPoints * 4,
        qualityLevel=float_qualityLevel,
        minDistance=int_minDistance,
        blockSize=5,
        mask=arr_mask,
        useHarrisDetector=False,
    )
    list_points = list()
    if arr_corners is not None:
        for arr_c in arr_corners[:, 0, :]:
            int_x, int_y = int(round(arr_c[0])), int(round(arr_c[1]))
            list_points.append((int_x, int_y))

    # Fallback: Contour centroids
    if len(list_points) < max(8, int_maxPoints // 2):
        _, arr_th = cv2.threshold(arr_enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        arr_kernelOpen = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
        arr_th = cv2.morphologyEx(arr_th, cv2.MORPH_OPEN, arr_kernelOpen)
        if arr_mask is not None:
            arr_th[arr_mask == 0] = 0
        list_cnts, _ = cv2.findContours(arr_th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for arr_cnt in sorted(list_cnts, key=cv2.contourArea, reverse=True):
            float_area = float(cv2.contourArea(arr_cnt))
            if float_area < 6.0 or float_area > 300.0:
                continue
            dict_m = cv2.moments(arr_cnt)
            if abs(dict_m["m00"]) < 1e-6:
                continue
            int_x = int(round(dict_m["m10"] / dict_m["m00"]))
            int_y = int(round(dict_m["m01"] / dict_m["m00"]))
            list_points.append((int_x, int_y))
            if len(list_points) >= int_maxPoints * 2:
                break

    # Deduplicate points
    list_kept = list()
    for int_px, int_py in list_points:
        bool_tooClose = False
        for int_qx, int_qy in list_kept:
            if (int_px - int_qx) ** 2 + (int_py - int_qy) ** 2 < int_minDistance ** 2:
                bool_tooClose = True
                break
        if not bool_tooClose:
            list_kept.append((int_px, int_py))
        if len(list_kept) >= int_maxPoints:
            break
    return list_kept


# =========================================================
# Mask Analysis and Filtering
# =========================================================

def calculate_mask_statistics(arr_mask: np.ndarray) -> tp.Optional[tp.Tuple[float, tp.List[float], float, float]]:
    """마스크의 면적, 바운딩 박스, 종횡비, Solidity 계산"""
    list_cnts, _ = cv2.findContours(arr_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not list_cnts:
        return None
    arr_cnt = max(list_cnts, key=cv2.contourArea)
    float_area = float(arr_mask.sum())
    int_x, int_y, int_w, int_h = cv2.boundingRect(arr_cnt)
    
    arr_hull = cv2.convexHull(arr_cnt)
    float_hullArea = float(cv2.contourArea(arr_hull))
    float_cntArea = float(cv2.contourArea(arr_cnt))
    float_solidity = float_cntArea / float_hullArea if float_hullArea > 1e-8 else 0.0

    if len(arr_cnt) >= 5:
        (_, _), (float_a1, float_a2), _ = cv2.fitEllipse(arr_cnt)
        float_major = float(max(float_a1, float_a2))
        float_minor = float(max(1e-6, min(float_a1, float_a2)))
    else:
        float_major = float(max(int_w, int_h))
        float_minor = float(max(1.0, min(int_w, int_h)))
    
    float_aspectRatio = float_major / float_minor
    list_bbox = [float(int_x), float(int_y), float(int_x + int_w), float(int_y + int_h)]
    return float_area, list_bbox, float_aspectRatio, float_solidity


def calculate_binary_iou(arr_maskA: np.ndarray, arr_maskB: np.ndarray) -> float:
    """두 이진 마스크 간의 IoU(Intersection over Union) 계산"""
    int_inter = np.logical_and(arr_maskA > 0, arr_maskB > 0).sum()
    int_union = np.logical_or(arr_maskA > 0, arr_maskB > 0).sum()
    if int_union == 0:
        return 0.0
    return float(int_inter / int_union)


def validate_mask_criteria(
    arr_mask: np.ndarray,
    tuple_promptXy: tp.Tuple[int, int],
    float_minArea: float,
    float_maxArea: float,
    float_minAspectRatio: float,
    float_maxAspectRatio: float,
    float_minSolidity: float,
    float_maxSolidity: float,
    int_borderMargin: int,
) -> tp.Tuple[bool, tp.Dict[str, tp.Any]]:
    """검출된 마스크가 주어진 형태학적 기준을 만족하는지 검증"""
    tuple_stats = calculate_mask_statistics(arr_mask)
    if tuple_stats is None:
        return False, dict()
    
    float_area, list_bbox, float_aspectRatio, float_solidity = tuple_stats
    int_x1, int_y1, int_x2, int_y2 = [int(float_v) for float_v in list_bbox]
    int_px, int_py = tuple_promptXy

    dict_info = {
        "area": float_area,
        "bbox": list_bbox,
        "aspect_ratio": float_aspectRatio,
        "solidity": float_solidity,
    }

    if float_area < float_minArea or float_area > float_maxArea:
        return False, dict_info
    if float_aspectRatio < float_minAspectRatio or float_aspectRatio > float_maxAspectRatio:
        return False, dict_info
    if float_solidity < float_minSolidity or float_solidity > float_maxSolidity:
        return False, dict_info
    if int_px < int_x1 or int_px > int_x2 or int_py < int_y1 or int_py > int_y2:
        return False, dict_info
        
    int_h, int_w = arr_mask.shape
    if (int_x1 <= int_borderMargin or int_y1 <= int_borderMargin or 
        int_x2 >= int_w - int_borderMargin or int_y2 >= int_h - int_borderMargin):
        return False, dict_info
        
    return True, dict_info


# =========================================================
# SAM2 Service Class
# =========================================================

class Sam2TiledService:
    """Tiled SAM2 추론을 관리하는 메인 서비스 클래스"""

    def __init__(self, obj_config: Sam2TiledConfig) -> None:
        self.obj_config = obj_config
        self.obj_model: tp.Optional[SAM] = None

    def initialize_model(self) -> None:
        """모델 초기화"""
        self.obj_model = SAM(self.obj_config.str_modelName)

    def execute_inference(self) -> tp.Tuple[tp.List[np.ndarray], tp.List[Sam2MaskItem], tp.Dict[str, tp.Any]]:
        """타일링 기반 전체 추론 프로세스 실행"""
        if self.obj_model is None:
            self.initialize_model()

        arr_imageBgr = load_image_bgr(str(self.obj_config.path_inputPath))
        arr_imageGray = load_image_gray(str(self.obj_config.path_inputPath))
        arr_roiMask, tuple_roiBbox = detect_particle_roi(arr_imageGray)
        
        int_rx1, int_ry1, int_rx2, int_ry2 = tuple_roiBbox
        list_tiles = create_processing_tiles(
            int_rx1, int_ry1, int_rx2, int_ry2, 
            int_tileSize=self.obj_config.int_tileSize, 
            int_stride=self.obj_config.int_stride
        )

        list_keptMasksGlobal = list()
        list_keptItems = list()
        list_debugPoints = list()

        dict_predictCommon = {
            "imgsz": self.obj_config.int_imgSize,
            "retina_masks": self.obj_config.bool_retinaMasks,
            "verbose": False,
        }
        if self.obj_config.str_device:
            dict_predictCommon["device"] = self.obj_config.str_device

        int_candidateCount = 0
        int_acceptedCount = 0

        for int_tileIdx, (int_tx1, int_ty1, int_tx2, int_ty2) in enumerate(list_tiles):
            arr_tileBgr = arr_imageBgr[int_ty1:int_ty2, int_tx1:int_tx2].copy()
            arr_tileGray = arr_imageGray[int_ty1:int_ty2, int_tx1:int_tx2].copy()
            arr_tileRoi = arr_roiMask[int_ty1:int_ty2, int_tx1:int_tx2].copy()

            if (arr_tileRoi > 0).sum() < 0.15 * arr_tileRoi.size:
                continue

            list_points = sample_interest_points(
                arr_tileGray=arr_tileGray,
                int_maxPoints=self.obj_config.int_pointsPerTile,
                int_minDistance=self.obj_config.int_pointMinDistance,
                float_qualityLevel=self.obj_config.float_qualityLevel,
                arr_mask=arr_tileRoi,
            )

            for int_px, int_py in list_points:
                int_candidateCount += 1
                list_debugPoints.append({
                    "tile_index": int_tileIdx,
                    "tile_xyxy": [int_tx1, int_ty1, int_tx2, int_ty2],
                    "point_xy_tile": [int_px, int_py],
                    "point_xy_global": [int_tx1 + int_px, int_ty1 + int_py],
                })

                list_results = self.obj_model( # type: ignore
                    source=arr_tileBgr,
                    points=[[int_px, int_py]],
                    labels=[1],
                    **dict_predictCommon,
                )
                if not list_results:
                    continue

                obj_result = list_results[0]
                if obj_result.masks is None or obj_result.masks.data is None:
                    continue
                
                arr_tileMasks = obj_result.masks.data.detach().cpu().numpy()
                arr_tileScores = None
                if obj_result.boxes is not None and obj_result.boxes.conf is not None:
                    arr_tileScores = obj_result.boxes.conf.detach().cpu().numpy()

                for int_maskIdx, arr_tm in enumerate(arr_tileMasks):
                    bool_keep, dict_info = validate_mask_criteria(
                        arr_mask=(arr_tm > 0).astype(np.uint8),
                        tuple_promptXy=(int_px, int_py),
                        float_minArea=self.obj_config.float_minArea,
                        float_maxArea=self.obj_config.float_maxArea,
                        float_minAspectRatio=self.obj_config.float_minAspectRatio,
                        float_maxAspectRatio=self.obj_config.float_maxAspectRatio,
                        float_minSolidity=self.obj_config.float_minSolidity,
                        float_maxSolidity=self.obj_config.float_maxSolidity,
                        int_borderMargin=self.obj_config.int_borderMargin,
                    )
                    if not bool_keep:
                        continue

                    arr_globalMask = np.zeros(arr_imageGray.shape, dtype=np.uint8)
                    arr_globalMask[int_ty1:int_ty2, int_tx1:int_tx2] = (arr_tm > 0).astype(np.uint8)
                    arr_globalMask[arr_roiMask == 0] = 0

                    # IoU Deduplication
                    bool_isDup = False
                    for arr_prev in list_keptMasksGlobal:
                        if calculate_binary_iou(arr_prev, arr_globalMask) >= self.obj_config.float_dedupIou:
                            bool_isDup = True
                            break
                    if bool_isDup:
                        continue

                    int_acceptedCount += 1
                    float_scoreVal = None
                    if arr_tileScores is not None and int_maskIdx < len(arr_tileScores):
                        float_scoreVal = float(arr_tileScores[int_maskIdx])

                    tuple_resStats = calculate_mask_statistics(arr_globalMask)
                    if tuple_resStats is None:
                        continue
                    float_area, list_bbox, float_ar, float_sol = tuple_resStats

                    obj_item = Sam2MaskItem(
                        int_index=len(list_keptItems),
                        float_areaPx=float_area,
                        list_bboxXyxy=list_bbox,
                        float_aspectRatio=float_ar,
                        float_solidity=float_sol,
                        float_score=float_scoreVal,
                        list_tileXyxy=[int_tx1, int_ty1, int_tx2, int_ty2],
                        list_promptXyGlobal=[int_tx1 + int_px, int_ty1 + int_py],
                    )
                    list_keptMasksGlobal.append(arr_globalMask)
                    list_keptItems.append(obj_item)

        dict_debug = {
            "roi_bbox_xyxy": [int_rx1, int_ry1, int_rx2, int_ry2],
            "num_tiles": len(list_tiles),
            "num_candidate_points": int_candidateCount,
            "num_accepted_masks": int_acceptedCount,
            "candidate_points": list_debugPoints,
        }
        return list_keptMasksGlobal, list_keptItems, dict_debug


# =========================================================
# Visualization and Output
# =========================================================

def create_mask_overlay(arr_imageBgr: np.ndarray, list_masks: tp.List[np.ndarray], float_alpha: float = 0.45) -> np.ndarray:
    """모든 마스크를 다른 색상으로 오버레이 시각화"""
    arr_out = arr_imageBgr.copy()
    for int_idx, arr_mask in enumerate(list_masks):
        tuple_color = generate_color_by_index(int_idx)
        arr_color = np.array(tuple_color, dtype=np.uint8)
        arr_colored = np.zeros_like(arr_out, dtype=np.uint8)
        arr_colored[arr_mask.astype(bool)] = arr_color
        arr_out = cv2.addWeighted(arr_out, 1.0, arr_colored, float_alpha, 0)
    return arr_out


def draw_labels_on_image(arr_imageBgr: np.ndarray, list_items: tp.List[Sam2MaskItem]) -> np.ndarray:
    """바운딩 박스, 인덱스, 종횡비 라벨 표시"""
    arr_out = arr_imageBgr.copy()
    for obj_item in list_items:
        tuple_color = generate_color_by_index(obj_item.int_index)
        int_x1, int_y1, int_x2, int_y2 = [int(float_v) for float_v in obj_item.list_bboxXyxy]
        cv2.rectangle(arr_out, (int_x1, int_y1), (int_x2, int_y2), tuple_color, 1)
        str_label = f"#{obj_item.int_index} ar={obj_item.float_aspectRatio:.2f}"
        cv2.putText(arr_out, str_label, (int_x1, max(12, int_y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.38, tuple_color, 1, cv2.LINE_AA)
        int_px, int_py = obj_item.list_promptXyGlobal
        cv2.circle(arr_out, (int_px, int_py), 2, (0, 0, 255), -1)
    return arr_out


def draw_roi_debug_info(arr_imageBgr: np.ndarray, tuple_roiBbox: tp.Tuple[int, int, int, int], dict_debug: tp.Dict[str, tp.Any]) -> np.ndarray:
    """ROI 영역 및 샘플링된 관심점 시각화 (디버그용)"""
    arr_out = arr_imageBgr.copy()
    int_rx1, int_ry1, int_rx2, int_ry2 = tuple_roiBbox
    cv2.rectangle(arr_out, (int_rx1, int_ry1), (int_rx2, int_ry2), (255, 255, 0), 2)
    for dict_p in dict_debug.get("candidate_points", list()):
        list_global = dict_p["point_xy_global"]
        cv2.circle(arr_out, (int(list_global[0]), int(list_global[1])), 1, (0, 255, 255), -1)
    return arr_out


# =========================================================
# Entry Points
# =========================================================

def run_tiled_interactive(obj_config: Sam2TiledConfig) -> tp.Dict[str, tp.Any]:
    """Interactive Window에서 호출 가능한 진입점"""
    obj_service = Sam2TiledService(obj_config)
    list_masks, list_items, dict_debug = obj_service.execute_inference()
    
    obj_config.path_outputDir.mkdir(parents=True, exist_ok=True)
    arr_imageBgr = load_image_bgr(str(obj_config.path_inputPath))
    arr_imageGray = load_image_gray(str(obj_config.path_inputPath))
    arr_roiMask, tuple_roiBbox = detect_particle_roi(arr_imageGray)
    
    arr_overlay = create_mask_overlay(arr_imageBgr, list_masks)
    arr_labeled = draw_labels_on_image(arr_overlay, list_items)
    arr_roiVis = draw_roi_debug_info(arr_imageBgr, tuple_roiBbox, dict_debug)
    
    save_image_to_path(obj_config.path_outputDir / "01_input.png", arr_imageBgr)
    save_image_to_path(obj_config.path_outputDir / "02_roi_mask.png", arr_roiMask)
    save_image_to_path(obj_config.path_outputDir / "03_roi_points.png", arr_roiVis)
    save_image_to_path(obj_config.path_outputDir / "04_overlay.png", arr_overlay)
    save_image_to_path(obj_config.path_outputDir / "05_overlay_boxes.png", arr_labeled)
    
    if obj_config.bool_saveIndividualMasks:
        path_maskDir = obj_config.path_outputDir / "masks"
        path_maskDir.mkdir(parents=True, exist_ok=True)
        for int_idx, arr_m in enumerate(list_masks):
            save_image_to_path(path_maskDir / f"mask_{int_idx:04d}.png", arr_m.astype(np.uint8) * 255)
            
    dict_summary = {
        "num_masks": len(list_items),
        "debug": dict_debug,
        "items": [asdict(obj_item) for obj_item in list_items],
    }
    
    with (obj_config.path_outputDir / "summary.json").open("w", encoding="utf-8") as obj_f:
        json.dump(dict_summary, obj_f, ensure_ascii=False, indent=2)
        
    return dict_summary


def main() -> None:
    """
    진입점: 아래 변수들을 직접 수정하여 실행하세요.
    """
    # === 사용자 설정 영역 ===
    str_inputPath = "test.png"              # 입력 이미지 경로
    str_outputDir = "out_sam2_2nd"          # 결과 저장 폴더
    str_modelName = "sam2.1_b.pt"           # 모델 파일
    str_device = None                       # 'cpu' 또는 'cuda:0'
    
    int_imgSize = 1024
    bool_retinaMasks = False
    
    int_tileSize = 512
    int_stride = 384
    int_pointsPerTile = 24
    
    float_minArea = 18.0
    float_maxArea = 900.0
    float_minAspectRatio = 1.8
    float_maxAspectRatio = 12.0
    # ========================

    obj_config = Sam2TiledConfig(
        path_inputPath=Path(str_inputPath),
        path_outputDir=Path(str_outputDir),
        str_modelName=str_modelName,
        str_device=str_device,
        int_imgSize=int_imgSize,
        bool_retinaMasks=bool_retinaMasks,
        int_tileSize=int_tileSize,
        int_stride=int_stride,
        int_pointsPerTile=int_pointsPerTile,
        float_minArea=float_minArea,
        float_maxArea=float_maxArea,
        float_minAspectRatio=float_minAspectRatio,
        float_maxAspectRatio=float_maxAspectRatio
    )
    
    dict_summary = run_tiled_interactive(obj_config)
    
    print("===== Tiled SAM2 결과 요약 =====")
    print(json.dumps(dict_summary["debug"], ensure_ascii=False, indent=2))
    print(f"검출된 마스크 수: {dict_summary['num_masks']}")


if __name__ == "__main__":
    main()
