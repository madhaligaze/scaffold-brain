# modules/detector_2d.py
from __future__ import annotations

import io
import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np

try:
    from PIL import Image
    _PIL_OK = True
except Exception:
    Image = None
    _PIL_OK = False

try:
    from ultralytics import YOLO
    _YOLO_OK = True
except Exception:
    YOLO = None
    _YOLO_OK = False

logger = logging.getLogger(__name__)


@dataclass
class DetectorConfig:
    model_path: str = "yolov8n.pt"
    conf_thres: float = 0.25
    iou_thres: float = 0.45
    max_det: int = 50


class Detector2D:
    def __init__(self, cfg: Optional[DetectorConfig] = None) -> None:
        self.cfg = cfg or DetectorConfig()
        self.model = None
        self._status = {"ok": True, "code": "READY", "message": "detector ready"}
        if not _PIL_OK:
            self._status = {"ok": False, "code": "PIL_MISSING", "message": "Pillow is not installed"}
            return
        if not _YOLO_OK:
            self._status = {"ok": False, "code": "YOLO_MISSING", "message": "ultralytics is not installed"}
            return
        try:
            self.model = YOLO(self.cfg.model_path)
        except Exception as exc:
            self._status = {"ok": False, "code": "MODEL_LOAD_FAILED", "message": str(exc)}

    @property
    def available(self) -> bool:
        return self.model is not None

    def infer_with_status(self, image_bytes: bytes) -> Dict[str, Any]:
        if self.model is None:
            return {"status": self._status, "detections": []}
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        img_np = np.array(img)

        try:
            results = self.model.predict(
                source=img_np,
                conf=self.cfg.conf_thres,
                iou=self.cfg.iou_thres,
                max_det=self.cfg.max_det,
                verbose=False,
            )
        except Exception as exc:
            return {
                "status": {"ok": False, "code": "INFER_FAILED", "message": str(exc)},
                "detections": [],
            }

        dets: List[Dict[str, Any]] = []
        if results:
            r0 = results[0]
            names = getattr(r0, "names", {}) or {}
            boxes = getattr(r0, "boxes", None)
            xyxy = getattr(boxes, "xyxy", None) if boxes is not None else None
            cls = getattr(boxes, "cls", None) if boxes is not None else None
            conf = getattr(boxes, "conf", None) if boxes is not None else None
            if xyxy is not None and cls is not None and conf is not None:
                for (x1, y1, x2, y2), c, s in zip(xyxy.cpu().numpy(), cls.cpu().numpy(), conf.cpu().numpy()):
                    c_int = int(c)
                    label = names.get(c_int, f"class_{c_int}")
                    dets.append(
                        {
                            "class_label": str(label),
                            "bbox_xyxy": [float(x1), float(y1), float(x2), float(y2)],
                            "score": float(s),
                            "mask_rle": None,
                        }
                    )

        return {"status": {"ok": True, "code": "READY", "message": "ok"}, "detections": dets}

    def infer(self, image_bytes: bytes) -> List[Dict[str, Any]]:
        return self.infer_with_status(image_bytes)["detections"]
