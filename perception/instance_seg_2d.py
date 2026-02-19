from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import numpy as np


@dataclass(frozen=True)
class Instance2D:
    label: str
    score: float
    mask_u8: np.ndarray  # HxW uint8 {0,255}


class InstanceSeg2D:
    def __init__(
        self,
        *,
        enabled: bool = True,
        model: str = "yolov8n-seg.pt",
        conf: float = 0.25,
        iou: float = 0.5,
        max_det: int = 25,
        device: str = "cpu",
    ) -> None:
        self.enabled = bool(enabled)
        self.model_name = str(model or "yolov8n-seg.pt")
        self.conf = float(conf)
        self.iou = float(iou)
        self.max_det = int(max_det)
        self.device = str(device or "cpu")
        self._yolo = None

    @property
    def available(self) -> bool:
        if not self.enabled:
            return False
        try:
            self._ensure_model()
            return self._yolo is not None
        except Exception:
            return False

    def _ensure_model(self) -> None:
        if self._yolo is not None or not self.enabled:
            return
        try:
            from ultralytics import YOLO  # type: ignore
        except Exception as exc:
            raise RuntimeError(f"ultralytics is not installed: {exc}") from exc

        self._yolo = YOLO(self.model_name)

    def predict(self, rgb_bgr: np.ndarray) -> tuple[list[Instance2D], dict[str, Any]]:
        if not self.enabled:
            return [], {"available": False, "reason": "disabled"}

        try:
            self._ensure_model()
        except Exception as exc:
            return [], {"available": False, "reason": str(exc)}

        if rgb_bgr is None or not isinstance(rgb_bgr, np.ndarray) or rgb_bgr.ndim != 3:
            return [], {"available": False, "reason": "invalid_input"}

        rgb = rgb_bgr[..., ::-1]
        try:
            results = self._yolo.predict(
                source=rgb,
                conf=self.conf,
                iou=self.iou,
                max_det=self.max_det,
                verbose=False,
                device=self.device,
            )
        except Exception as exc:
            return [], {"available": True, "reason": f"predict_failed: {exc}"}

        if not results:
            return [], {"available": True, "count": 0}

        r0 = results[0]
        masks = getattr(r0, "masks", None)
        boxes = getattr(r0, "boxes", None)
        if masks is None or getattr(masks, "data", None) is None:
            return [], {"available": True, "count": 0, "reason": "no_masks"}

        mask_data = masks.data
        try:
            mask_np = mask_data.cpu().numpy()
        except Exception:
            mask_np = np.asarray(mask_data)

        scores: Optional[np.ndarray] = None
        labels: Optional[np.ndarray] = None
        try:
            if boxes is not None:
                scores = boxes.conf.cpu().numpy()
                labels = boxes.cls.cpu().numpy().astype(np.int32)
        except Exception:
            scores = None
            labels = None

        names = getattr(r0, "names", None) or {}

        inst: list[Instance2D] = []
        for i in range(int(mask_np.shape[0])):
            m = mask_np[i]
            if m.ndim != 2:
                continue
            m_u8 = (m > 0.5).astype(np.uint8) * 255
            if int(m_u8.sum()) < 255 * 120:
                continue

            sc = float(scores[i]) if scores is not None and i < scores.shape[0] else 1.0
            lab = int(labels[i]) if labels is not None and i < labels.shape[0] else -1
            label = str(names.get(lab, f"cls_{lab}"))
            inst.append(Instance2D(label=label, score=sc, mask_u8=m_u8))

        meta = {
            "available": True,
            "backend": "ultralytics-yolov8-seg",
            "model": self.model_name,
            "conf": self.conf,
            "iou": self.iou,
            "max_det": self.max_det,
            "count": len(inst),
        }
        return inst, meta
