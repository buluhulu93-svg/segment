#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
SAM2 View - Ultralytics SAM2 segmentation results viewer
========================================================
이 스크립트는 Ultralytics SAM2 모델을 사용하여 이미지에서 세그멘테이션을 수행하고,
그 결과를 시각화하여 저장하는 기능을 제공합니다.

전체 처리 흐름:
1. 입력 이미지 및 설정 로드
2. SAM2 모델 초기화 (디바이스 설정 포함)
3. 입력 프롬프트(Points, Labels, BBoxes) 파싱
4. SAM2 추론 수행
5. 결과 마스크, 박스, 점수 추출 및 정규화
6. 오버레이 시각화 생성 (Masks, Boxes, Indices)
7. 결과 저장 (Images, JSON Summary)
========================================================
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
class Sam2ViewConfig:
    """
    SAM2 View 실행을 위한 설정 데이터 클래스
    """
    path_inputPath: Path
    path_outputDir: Path
    str_modelName: str = "sam2.1_b.pt"
    str_device: tp.Optional[str] = None
    int_imgSize: int = 1024
    bool_retinaMasks: bool = False
    list_points: tp.Optional[tp.List[tp.List[float]]] = None
    list_labels: tp.Optional[tp.List[int]] = None
    list_bboxes: tp.Optional[tp.List[tp.List[float]]] = None
    bool_saveIndividualMasks: bool = False


@dataclass
class Sam2ViewResult:
    """
    SAM2 View 실행 결과를 담는 데이터 클래스
    """
    arr_masks: np.ndarray
    arr_boxes: tp.Optional[np.ndarray]
    arr_scores: tp.Optional[np.ndarray]
    dict_summary: tp.Dict[str, tp.Any]


class Sam2ViewService:
    """
    SAM2 View 핵심 로직을 담당하는 서비스 클래스
    """

    def __init__(self, obj_config: Sam2ViewConfig) -> None:
        self.obj_config = obj_config
        self.obj_model: tp.Optional[SAM] = None

    def initialize_model(self) -> None:
        """
        SAM2 모델 초기화
        """
        self.obj_model = SAM(self.obj_config.str_modelName)

    def load_image_bgr(self) -> np.ndarray:
        """
        입력 이미지를 BGR 형식으로 로드
        """
        str_path = str(self.obj_config.path_inputPath)
        arr_img = cv2.imread(str_path, cv2.IMREAD_COLOR)
        if arr_img is None:
            raise FileNotFoundError(f"이미지를 읽을 수 없습니다: {str_path}")
        return arr_img

    def save_image(self, path_target: Path, arr_img: np.ndarray) -> None:
        """
        이미지 파일 저장
        """
        cv2.imwrite(str(path_target), arr_img)

    def get_color_for_index(self, int_index: int) -> tp.Tuple[int, int, int]:
        """
        인덱스별 고유 색상 생성 (시각화용)
        """
        obj_rng = np.random.default_rng(seed=int_index + 1234)
        arr_bgr = obj_rng.integers(64, 256, size=3, dtype=np.uint8)
        return (int(arr_bgr[0]), int(arr_bgr[1]), int(arr_bgr[2]))

    def overlay_masks(self, arr_imageBgr: np.ndarray, arr_masks: np.ndarray, float_alpha: float = 0.45) -> np.ndarray:
        """
        원본 이미지에 세그멘테이션 마스크 오버레이
        """
        arr_out = arr_imageBgr.copy()
        if arr_masks.size == 0:
            return arr_out

        for int_idx, arr_mask in enumerate(arr_masks):
            tuple_color = self.get_color_for_index(int_idx)
            arr_color = np.array(tuple_color, dtype=np.uint8)
            arr_colored = np.zeros_like(arr_out, dtype=np.uint8)
            arr_colored[arr_mask.astype(bool)] = arr_color
            arr_out = cv2.addWeighted(arr_out, 1.0, arr_colored, float_alpha, 0)
        return arr_out

    def draw_boxes_and_indices(
        self,
        arr_imageBgr: np.ndarray,
        arr_masks: np.ndarray,
        arr_boxes: tp.Optional[np.ndarray],
        arr_scores: tp.Optional[np.ndarray],
    ) -> np.ndarray:
        """
        결과 이미지에 바운딩 박스와 인덱스(및 점수) 표시
        """
        arr_out = arr_imageBgr.copy()
        if arr_masks.size == 0:
            return arr_out

        for int_idx, arr_mask in enumerate(arr_masks):
            tuple_color = self.get_color_for_index(int_idx)
            arr_ys, arr_xs = np.where(arr_mask > 0)
            if len(arr_xs) == 0:
                continue

            if arr_boxes is not None and int_idx < len(arr_boxes):
                int_x1, int_y1, int_x2, int_y2 = arr_boxes[int_idx].astype(int)
            else:
                int_x1, int_y1, int_x2, int_y2 = int(arr_xs.min()), int(arr_ys.min()), int(arr_xs.max()), int(arr_ys.max())

            cv2.rectangle(arr_out, (int_x1, int_y1), (int_x2, int_y2), tuple_color, 1)
            str_label = f"#{int_idx}"
            if arr_scores is not None and int_idx < len(arr_scores):
                str_label = f"#{int_idx} {arr_scores[int_idx]:.2f}"
            cv2.putText(
                arr_out,
                str_label,
                (int_x1, max(12, int_y1 - 4)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                tuple_color,
                1,
                cv2.LINE_AA,
            )

        return arr_out

    def build_summary(
        self,
        arr_masks: np.ndarray,
        arr_boxes: tp.Optional[np.ndarray],
        arr_scores: tp.Optional[np.ndarray]
    ) -> tp.Dict[str, tp.Any]:
        """
        추론 결과 요약 정보 생성
        """
        dict_summary = dict()
        dict_summary["num_masks"] = int(arr_masks.shape[0])
        list_items = list()

        for int_idx, arr_mask in enumerate(arr_masks):
            int_areaPx = int(arr_mask.sum())
            arr_ys, arr_xs = np.where(arr_mask > 0)
            if len(arr_xs) == 0:
                continue

            if arr_boxes is not None and int_idx < len(arr_boxes):
                list_bboxXyxy = [float(float_v) for float_v in arr_boxes[int_idx]]
            else:
                list_bboxXyxy = [float(arr_xs.min()), float(arr_ys.min()), float(arr_xs.max()), float(arr_ys.max())]

            dict_item = dict()
            dict_item["index"] = int_idx
            dict_item["area_px"] = int_areaPx
            dict_item["bbox_xyxy"] = list_bboxXyxy

            if arr_scores is not None and int_idx < len(arr_scores):
                dict_item["score"] = float(arr_scores[int_idx])

            list_items.append(dict_item)

        dict_summary["items"] = list_items
        return dict_summary

    def process(self) -> Sam2ViewResult:
        """
        전체 실행 프로세스 수행
        """
        if self.obj_model is None:
            self.initialize_model()

        dict_predictKwargs = dict()
        dict_predictKwargs["source"] = str(self.obj_config.path_inputPath)
        dict_predictKwargs["imgsz"] = self.obj_config.int_imgSize
        dict_predictKwargs["retina_masks"] = self.obj_config.bool_retinaMasks
        dict_predictKwargs["verbose"] = False

        if self.obj_config.str_device:
            dict_predictKwargs["device"] = self.obj_config.str_device

        if self.obj_config.list_points is not None:
            dict_predictKwargs["points"] = self.obj_config.list_points
        if self.obj_config.list_labels is not None:
            dict_predictKwargs["labels"] = self.obj_config.list_labels
        if self.obj_config.list_bboxes is not None:
            dict_predictKwargs["bboxes"] = self.obj_config.list_bboxes

        # Inference
        list_results = self.obj_model(**dict_predictKwargs) # type: ignore
        if not list_results:
            raise RuntimeError("SAM2 결과가 비어 있습니다.")

        obj_result = list_results[0]

        # Extract masks
        arr_masks = np.empty(tuple([0, 0, 0]), dtype=np.uint8)
        if obj_result.masks is not None and obj_result.masks.data is not None:
            arr_masks_data = obj_result.masks.data.detach().cpu().numpy()
            arr_masks = (arr_masks_data > 0).astype(np.uint8)

        # Extract boxes
        arr_boxes = None
        if obj_result.boxes is not None and obj_result.boxes.xyxy is not None:
            arr_boxes = obj_result.boxes.xyxy.detach().cpu().numpy()

        # Extract scores
        arr_scores = None
        if obj_result.boxes is not None and obj_result.boxes.conf is not None:
            arr_scores = obj_result.boxes.conf.detach().cpu().numpy()

        dict_summary = self.build_summary(arr_masks, arr_boxes, arr_scores)

        # Visualization and saving
        self.obj_config.path_outputDir.mkdir(parents=True, exist_ok=True)
        arr_imageBgr = self.load_image_bgr()

        arr_overlay = self.overlay_masks(arr_imageBgr, arr_masks)
        arr_overlayWithBoxes = self.draw_boxes_and_indices(arr_overlay, arr_masks, arr_boxes, arr_scores)

        self.save_image(self.obj_config.path_outputDir / "01_input.png", arr_imageBgr)
        self.save_image(self.obj_config.path_outputDir / "02_overlay.png", arr_overlay)
        self.save_image(self.obj_config.path_outputDir / "03_overlay_boxes.png", arr_overlayWithBoxes)

        if obj_result.plot is not None:
            arr_plotted = obj_result.plot()
            self.save_image(self.obj_config.path_outputDir / "04_ultralytics_plot.png", arr_plotted)

        if self.obj_config.bool_saveIndividualMasks:
            path_maskDir = self.obj_config.path_outputDir / "masks"
            path_maskDir.mkdir(parents=True, exist_ok=True)
            for int_idx, arr_m in enumerate(arr_masks):
                self.save_image(path_maskDir / f"mask_{int_idx:04d}.png", arr_m.astype(np.uint8) * 255)

        with (self.obj_config.path_outputDir / "summary.json").open("w", encoding="utf-8") as obj_f:
            json.dump(dict_summary, obj_f, ensure_ascii=False, indent=2)

        return Sam2ViewResult(
            arr_masks=arr_masks,
            arr_boxes=arr_boxes,
            arr_scores=arr_scores,
            dict_summary=dict_summary
        )


def run_interactive(
    str_input: str,
    str_outputDir: str,
    str_model: str = "sam2.1_b.pt",
    str_device: tp.Optional[str] = None,
    int_imgsz: int = 1024,
    bool_retinaMasks: bool = False,
    list_points: tp.Optional[tp.List[tp.List[float]]] = None,
    list_labels: tp.Optional[tp.List[int]] = None,
    list_bboxes: tp.Optional[tp.List[tp.List[float]]] = None,
    bool_saveIndividualMasks: bool = False,
) -> Sam2ViewResult:
    """
    Interactive Window (Jupyter 등)에서 쉽게 호출할 수 있는 래퍼 함수
    """
    obj_config = Sam2ViewConfig(
        path_inputPath=Path(str_input),
        path_outputDir=Path(str_outputDir),
        str_modelName=str_model,
        str_device=str_device,
        int_imgSize=int_imgsz,
        bool_retinaMasks=bool_retinaMasks,
        list_points=list_points,
        list_labels=list_labels,
        list_bboxes=list_bboxes,
        bool_saveIndividualMasks=bool_saveIndividualMasks
    )

    obj_service = Sam2ViewService(obj_config)
    return obj_service.process()


def main() -> None:
    """
    진입점: 아래 변수들을 직접 수정하여 실행하세요.
    """
    # === 사용자 설정 영역 ===
    str_inputPath = "test.png"          # 입력 이미지 경로
    str_outputDir = "out_sam2"          # 결과 저장 폴더
    str_modelName = "sam2.1_b.pt"       # 모델 파일
    str_device = None                   # 'cpu' 또는 'cuda:0'
    
    int_imgsz = 1024
    bool_retinaMasks = False
    
    # 예: [[500, 400]]
    list_points = None
    # 예: [1]
    list_labels = None
    # 예: [[100, 100, 800, 800]]
    list_bboxes = None
    
    bool_saveIndividualMasks = False
    # ========================

    obj_result = run_interactive(
        str_input=str_inputPath,
        str_outputDir=str_outputDir,
        str_model=str_modelName,
        str_device=str_device,
        int_imgsz=int_imgsz,
        bool_retinaMasks=bool_retinaMasks,
        list_points=list_points,
        list_labels=list_labels,
        list_bboxes=list_bboxes,
        bool_saveIndividualMasks=bool_saveIndividualMasks
    )

    print("===== SAM2 결과 요약 =====")
    print(json.dumps(obj_result.dict_summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
