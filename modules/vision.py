"""Computer vision module for Bauflex AI Brain."""

from __future__ import annotations

import io
from typing import Any, Dict, List, Optional

import numpy as np
from PIL import Image

try:
    import cv2
except Exception:  # pragma: no cover
    cv2 = None

try:
    from ultralytics import YOLO
except Exception:  # pragma: no cover
    YOLO = None


class Eyes:
    """Object detector + real size estimator."""

    def __init__(self, model_path: str = "yolov8n.pt") -> None:
        self.model = None
        if YOLO is not None:
            try:
                self.model = YOLO(model_path)
            except Exception:
                self.model = None

    def analyze_scene(
        self,
        image_bytes: bytes,
        distance_to_target: float,
        focal_length: float = 800,
    ) -> List[Dict[str, Any]]:
        if distance_to_target <= 0:
            raise ValueError("distance_to_target must be > 0")

        frame = _decode_image(image_bytes)
        h, w = frame.shape[:2]

        detections: List[Dict[str, Any]] = []
        if self.model is not None:
            try:
                results = self.model.predict(source=frame, verbose=False)
                detections = self._yolo_to_objects(results, distance_to_target, focal_length)
            except Exception:
                detections = []

        if not detections:
            # Fallback для оффлайн/без модели: один общий bounding box.
            detections = [
                {
                    "type": "unknown",
                    "confidence": 0.5,
                    "real_width_m": round((w * distance_to_target) / focal_length, 3),
                    "real_height_m": round((h * distance_to_target) / focal_length, 3),
                    "bbox": [0, 0, w, h],
                    "center": [w // 2, h // 2],
                }
            ]
        return detections

    def _yolo_to_objects(self, results: Any, distance: float, focal_length: float) -> List[Dict[str, Any]]:
        objects: List[Dict[str, Any]] = []
        for result in results:
            names = getattr(result, "names", {})
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                conf = float(box.conf[0])
                cls = int(box.cls[0])
                px_w = max(1, x2 - x1)
                px_h = max(1, y2 - y1)
                objects.append(
                    {
                        "type": names.get(cls, f"class_{cls}"),
                        "confidence": round(conf, 3),
                        "real_width_m": round((px_w * distance) / focal_length, 3),
                        "real_height_m": round((px_h * distance) / focal_length, 3),
                        "bbox": [x1, y1, x2, y2],
                        "center": [x1 + px_w // 2, y1 + px_h // 2],
                    }
                )
        return objects


class SceneDiagnostician:
    """Image/data quality checks for design readiness."""

    def check_data_quality(
        self,
        image_bytes: bytes,
        detected_objects: Optional[List[Dict[str, Any]]],
        ar_points: Optional[List[Dict[str, float]]],
        distance: float,
    ) -> Dict[str, Any]:
        frame = _decode_image(image_bytes)
        gray = _to_gray(frame)

        warnings: List[str] = []
        instructions: List[str] = []
        score = 100.0

        brightness = float(np.mean(gray))
        contrast = float(np.std(gray))
        blur = _laplacian_variance(gray)

        if brightness < 50:
            score -= 15
            instructions.append("💡 Слишком темно! Добавьте освещение.")
        elif brightness > 240:
            score -= 10
            warnings.append("⚠️ Пересвет кадра.")

        if contrast < 30:
            score -= 10
            instructions.append("⚠️ Низкий контраст. Смените ракурс или освещение.")

        if blur < 100:
            score -= 20
            instructions.append("🔍 Изображение размыто. Стабилизируйте камеру.")

        if distance < 2 or distance > 5:
            score -= 10
            warnings.append("📏 Рекомендуемая дистанция: 2-5 м.")

        points = ar_points or []
        if len(points) < 4:
            score -= 15
            instructions.append("📍 Нужно минимум 4 AR-точки для надежной геометрии.")

        if not detected_objects:
            score -= 20
            instructions.append("👁️ Объект не распознан. Повторите съемку под другим углом.")

        score = max(0.0, round(score, 1))
        return {
            "is_ready": score >= 70,
            "quality_score": score,
            "warnings": warnings,
            "instructions": instructions,
            "metrics": {
                "brightness": round(brightness, 2),
                "contrast": round(contrast, 2),
                "blur_laplacian_var": round(blur, 2),
            },
        }


class VisionSystem:
    """Facade that combines detector and quality diagnostics."""

    def __init__(self) -> None:
        self.eyes = Eyes()
        self.diagnostician = SceneDiagnostician()


def _decode_image(image_bytes: bytes) -> np.ndarray:
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    return np.array(img)


def _to_gray(frame: np.ndarray) -> np.ndarray:
    if cv2 is not None:
        return cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    return np.dot(frame[..., :3], [0.299, 0.587, 0.114]).astype(np.uint8)


def _laplacian_variance(gray: np.ndarray) -> float:
    if cv2 is not None:
        return float(cv2.Laplacian(gray, cv2.CV_64F).var())
    # numpy fallback
    gy, gx = np.gradient(gray.astype(float))
    return float(np.var(gx) + np.var(gy))
