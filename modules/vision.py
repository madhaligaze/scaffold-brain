# modules/vision.py
"""
Computer Vision модуль для Bauflex AI Brain.
Детекция объектов (YOLO), оценка размеров, диагностика качества данных.

ИСПРАВЛЕНИЯ v2.2 (по результатам code review):
- BUGFIX: Устранено двойное декодирование изображения (Eyes.analyze_scene теперь
  принимает готовый frame; VisionSystem.process_scene декодирует ОДИН раз)
- Жёсткий маппинг классов YOLO (BAUFLEX_CLASS_MAP): beam / pipe_obstacle /
  safety_equipment и т.д. — физический движок различает несущие и ненесущие элементы
- _check_occlusion: добавлена проверка «объект занимает >80% кадра» (нет контекста)
- Новый метод _check_depth_occlusion: IoU-анализ перекрытий между объектами
- Новый метод _check_ar_drift: сравнивает размеры из YOLO с расстоянием между AR-точками,
  сигнализирует о дрифте ARCore или ошибке детектора

Предыдущие исправления v2.1:
- RGB → BGR конвертация для YOLO
- Динамическое focal_length (fx, fy)
- Детекция occlusion (обрезки объектов по краям кадра)
- Resize для ускорения blur-анализа
- Проверка зависимостей при инициализации
"""
from __future__ import annotations

import io
import logging
import base64
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# === ИМПОРТЫ С ОБРАБОТКОЙ ОШИБОК ===
try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    cv2 = None
    CV2_AVAILABLE = False

try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    Image = None
    PIL_AVAILABLE = False

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO = None
    YOLO_AVAILABLE = False


# === НАСТРОЙКА ЛОГИРОВАНИЯ ===
logger = logging.getLogger(__name__)


class Eyes:
    """
    Детектор объектов + оценка реальных размеров.
    
    Улучшения v2.1:
    - Корректная работа с BGR (для YOLO)
    - Динамическое focal_length (fx, fy из ARCore)
    - Fallback режим при отсутствии YOLO
    """

    # Жёсткий маппинг классов для Bauflex.
    # КРИТИЧНО: различать несущие балки и трубы вентиляции,
    # иначе физический движок будет пытаться крепить леса к пластику!
    BAUFLEX_CLASS_MAP: Dict[int, str] = {
        0: "beam",              # Несущая балка — точка крепления лесов
        1: "pipe_obstacle",     # Труба-препятствие — НЕЛЬЗЯ использовать как опору
        2: "safety_equipment",  # Защитное оборудование (каска, ограждение)
        3: "column",            # Колонна / стойка
        4: "floor_slab",        # Перекрытие
        5: "cable_tray",        # Кабельный лоток — препятствие
    }

    def __init__(self, model_path: str = "yolov8n.pt") -> None:
        """
        Инициализация детектора.

        Args:
            model_path: путь к весам YOLO модели
        """
        self._ensure_dependencies()
        
        self.model = None
        if YOLO_AVAILABLE:
            try:
                self.model = YOLO(model_path)
                logger.info(f"✓ YOLO model loaded: {model_path}")
            except Exception as e:
                logger.error(f"✗ Failed to load YOLO model: {e}")
                self.model = None
        else:
            logger.warning("⚠️ YOLO not available. Running in fallback mode.")

    def _ensure_dependencies(self) -> None:
        """Проверка критических зависимостей."""
        issues = []
        
        if not CV2_AVAILABLE:
            issues.append("opencv-python (cv2) is required but not installed")
        
        if not PIL_AVAILABLE:
            issues.append("Pillow (PIL) is required but not installed")
        
        if not YOLO_AVAILABLE:
            logger.warning("ultralytics (YOLO) not installed - will use fallback detection")
        
        if issues:
            error_msg = "Missing critical dependencies:\n" + "\n".join(f"  - {i}" for i in issues)
            logger.error(error_msg)
            raise ImportError(error_msg)

    def analyze_scene(
        self,
        image_bytes: Optional[bytes] = None,
        distance_to_target: float = 0.0,
        focal_length: Optional[float] = None,
        focal_length_x: Optional[float] = None,
        focal_length_y: Optional[float] = None,
        frame: Optional[np.ndarray] = None,
    ) -> List[Dict[str, Any]]:
        """
        Анализирует сцену: детектирует объекты и оценивает их реальные размеры.

        ОПТИМИЗАЦИЯ v2.2: принимает либо image_bytes, либо уже готовый frame.
        Если frame передан — декодирование пропускается (экономия на мобильном CPU).

        Args:
            image_bytes: байты изображения (используется, если frame не передан)
            distance_to_target: расстояние до объекта (метры), от ARCore
            focal_length: фокусное расстояние (устаревший параметр для совместимости)
            focal_length_x: фокусное расстояние по оси X (предпочтительно)
            focal_length_y: фокусное расстояние по оси Y (предпочтительно)
            frame: готовый BGR-кадр np.ndarray (приоритет над image_bytes)

        Returns:
            Список обнаруженных объектов с размерами и координатами

        Raises:
            ValueError: если distance <= 0 или не передан ни image_bytes, ни frame
        """
        # Валидация входных данных
        if distance_to_target <= 0:
            raise ValueError(f"distance_to_target must be > 0, got {distance_to_target}")

        # Единственное декодирование: используем готовый frame или декодируем из байт
        if frame is None:
            if image_bytes is not None:
                frame = self._decode_image_bgr(image_bytes)
            else:
                raise ValueError("Either image_bytes or frame must be provided")
        # Если frame уже передан — пропускаем декодирование

        # Фокусное расстояние (приоритет: fx/fy > focal_length > default)
        fx = focal_length_x or focal_length or 800.0
        fy = focal_length_y or focal_length or 800.0

        if focal_length_x is None and focal_length_y is None and focal_length is None:
            logger.warning(
                "⚠️ Using default focal_length=800px. "
                "For better accuracy, provide focal_length_x and focal_length_y from ARCore."
            )

        h, w = frame.shape[:2]

        detections: List[Dict[str, Any]] = []
        
        # YOLO детекция
        if self.model is not None:
            try:
                # YOLO ожидает BGR (или RGB, но мы передаем BGR для консистентности с cv2)
                results = self.model.predict(source=frame, verbose=False)
                detections = self._yolo_to_objects(results, distance_to_target, fx, fy)
            except Exception as e:
                logger.error(f"YOLO prediction failed: {e}")
                detections = []

        # Fallback: если ничего не найдено
        if not detections:
            logger.warning("No objects detected. Using full frame as fallback.")
            detections = [
                {
                    "type": "unknown",
                    "confidence": 0.5,
                    "real_width_m": round((w * distance_to_target) / fx, 3),
                    "real_height_m": round((h * distance_to_target) / fy, 3),
                    "bbox": [0, 0, w, h],
                    "center": [w // 2, h // 2],
                }
            ]
        
        return detections

    def _decode_image_bgr(self, image_bytes: bytes | str) -> np.ndarray:
        """
        Декодирует изображение в BGR формат (для YOLO и cv2).
        
        КРИТИЧНО: PIL возвращает RGB, YOLO работает с BGR.
        Без конвертации точность распознавания падает!
        
        Args:
            image_bytes: байты изображения
        
        Returns:
            numpy array в BGR формате (H, W, 3)
        """
        if isinstance(image_bytes, str):
            # Поддержка base64-потока от Android (с возможным data URI префиксом)
            payload = image_bytes.split(",", 1)[-1] if image_bytes.startswith("data:") else image_bytes
            image_bytes = base64.b64decode(payload)

        if CV2_AVAILABLE:
            # Оптимальный путь: cv2 напрямую декодирует в BGR
            nparr = np.frombuffer(image_bytes, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if frame is None:
                raise ValueError("Failed to decode image with cv2")
            
            return frame
        
        elif PIL_AVAILABLE:
            # Fallback через PIL: RGB → BGR
            img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            frame_rgb = np.array(img)
            
            # КРИТИЧНО: Конвертируем RGB → BGR
            frame_bgr = frame_rgb[:, :, ::-1]
            
            return frame_bgr
        
        else:
            raise ImportError("Neither cv2 nor PIL is available for image decoding")

    def _yolo_to_objects(
        self, 
        results: Any, 
        distance: float, 
        fx: float, 
        fy: float
    ) -> List[Dict[str, Any]]:
        """
        Конвертирует результаты YOLO в список объектов с реальными размерами.
        
        Args:
            results: выход YOLO model.predict()
            distance: расстояние до объекта (м)
            fx, fy: фокусное расстояние по осям X и Y (пиксели)
        
        Returns:
            Список объектов
        """
        objects: List[Dict[str, Any]] = []
        
        for result in results:
            names = getattr(result, "names", {})
            
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                conf = float(box.conf[0])
                cls = int(box.cls[0])
                
                # Размеры в пикселях
                px_w = max(1, x2 - x1)
                px_h = max(1, y2 - y1)
                
                # Реальные размеры: (пиксели * расстояние) / фокусное_расстояние
                real_w = (px_w * distance) / fx
                real_h = (px_h * distance) / fy
                
                # Маппинг классов: сначала Bauflex-специфичный словарь,
                # затем имена из модели, затем generic fallback.
                # Важно: pipe_obstacle ≠ beam — физический движок должен их различать!
                obj_type = (
                    self.BAUFLEX_CLASS_MAP.get(cls)
                    or names.get(cls, f"class_{cls}")
                )

                objects.append({
                    "type": obj_type,
                    "confidence": round(conf, 3),
                    "real_width_m": round(real_w, 3),
                    "real_height_m": round(real_h, 3),
                    "bbox": [x1, y1, x2, y2],
                    "center": [x1 + px_w // 2, y1 + px_h // 2],
                })
        
        return objects


class SceneDiagnostician:
    """
    Диагностика качества данных для проектирования.
    
    Улучшения v2.1:
    - Оптимизация: принимает готовый frame (без повторного декодирования)
    - Детекция occlusion (обрезки объектов)
    - Resize для ускорения blur-анализа
    - Более детальные инструкции
    """
    
    # Пороговые значения
    MIN_BRIGHTNESS = 50
    MAX_BRIGHTNESS = 240
    MIN_CONTRAST = 30
    BLUR_THRESHOLD = 100
    MIN_AR_POINTS = 4
    OPTIMAL_DISTANCE_MIN = 2.0
    OPTIMAL_DISTANCE_MAX = 5.0
    EDGE_THRESHOLD_PX = 20  # Порог для детекции обрезки
    BLUR_ANALYSIS_SIZE = 640  # Сжимаем до этого размера для ускорения

    def check_data_quality(
        self,
        frame: np.ndarray,
        detected_objects: Optional[List[Dict[str, Any]]],
        ar_points: Optional[List[Dict[str, float]]],
        distance: float,
    ) -> Dict[str, Any]:
        """
        Комплексная проверка качества данных.
        
        ВАЖНО: Принимает готовый frame (не image_bytes) для оптимизации!
        
        Args:
            frame: декодированное изображение в BGR (H, W, 3)
            detected_objects: результат Eyes.analyze_scene()
            ar_points: список AR-точек [{x, y, z}, ...]
            distance: расстояние до объекта (м)
        
        Returns:
            {
                "is_ready": bool,
                "quality_score": float (0-100),
                "warnings": List[str],
                "instructions": List[str],
                "metrics": {...}
            }
        """
        h, w = frame.shape[:2]
        
        # Конвертируем в grayscale для анализа
        gray = self._to_gray(frame)
        
        # Для blur-анализа сжимаем (оптимизация)
        gray_small = self._resize_for_blur_analysis(gray)
        
        warnings: List[str] = []
        instructions: List[str] = []
        score = 100.0
        
        # === 1. ЯРКОСТЬ ===
        brightness = float(np.mean(gray))
        
        if brightness < self.MIN_BRIGHTNESS:
            score -= 15
            instructions.append("💡 Слишком темно! Включите фонарик или улучшите освещение.")
        elif brightness > self.MAX_BRIGHTNESS:
            score -= 10
            instructions.append("☀️ Переэкспозиция! Уменьшите яркость или измените угол.")
        elif brightness < 80:
            score -= 5
            warnings.append("Освещение недостаточное, но приемлемое.")
        
        # === 2. КОНТРАСТ ===
        contrast = float(np.std(gray))
        
        if contrast < self.MIN_CONTRAST:
            score -= 10
            instructions.append("📷 Низкий контраст. Смените ракурс или освещение.")
        
        # === 3. РАЗМЫТОСТЬ ===
        blur = self._laplacian_variance(gray_small)
        
        if blur < self.BLUR_THRESHOLD:
            score -= 20
            instructions.append("🔍 Изображение размыто! Стабилизируйте камеру или дождитесь автофокуса.")
        
        # === 4. РАССТОЯНИЕ ===
        if distance < 1.0:
            score -= 15
            instructions.append("📏 Слишком близко! Отойдите минимум на 2 метра.")
        elif distance > 10.0:
            score -= 10
            instructions.append("📏 Слишком далеко! Подойдите ближе (оптимально 2-5 метров).")
        elif distance < self.OPTIMAL_DISTANCE_MIN or distance > self.OPTIMAL_DISTANCE_MAX:
            score -= 5
            warnings.append(f"📏 Расстояние {distance:.1f}м. Рекомендуется 2-5м.")
        
        # === 5. AR-ТОЧКИ ===
        points = ar_points or []
        
        if len(points) < self.MIN_AR_POINTS:
            score -= 15
            instructions.append(
                f"📍 Мало опорных точек ({len(points)}/{self.MIN_AR_POINTS}). "
                f"Отметьте углы балок или препятствий."
            )
        
        # === 6. ОБНАРУЖЕННЫЕ ОБЪЕКТЫ ===
        if not detected_objects:
            score -= 20
            instructions.append("👁️ Объекты не распознаны. Повторите съемку под другим углом.")
        else:
            # === 7. OCCLUSION (ОБРЕЗКА ОБЪЕКТОВ) ===
            occlusion_check = self._check_occlusion(detected_objects, w, h)

            if occlusion_check["has_occlusion"]:
                score -= 10
                instructions.append(occlusion_check["message"])

            # === 8. DEPTH OCCLUSION (перекрытие одного объекта другим) ===
            depth_occlusion = self._check_depth_occlusion(detected_objects)

            if depth_occlusion["has_depth_occlusion"]:
                score -= 10
                warnings.append(depth_occlusion["message"])

            # === 9. AR DRIFT CHECK (сравниваем размеры YOLO с дистанциями AR-точек) ===
            ar_drift = self._check_ar_drift(detected_objects, points)

            if ar_drift["has_drift"]:
                score -= 10
                warnings.append(ar_drift["message"])
        
        # === ИТОГОВАЯ ОЦЕНКА ===
        score = max(0.0, min(100.0, round(score, 1)))

        return {
            "is_ready": score >= 70 and not instructions,
            "quality_score": score,
            "warnings": warnings,
            "instructions": instructions,
            "metrics": {
                "brightness": round(brightness, 2),
                "contrast": round(contrast, 2),
                "blur_laplacian_var": round(blur, 2),
                "distance_m": round(distance, 2),
                "ar_points_count": len(points),
                "detected_objects_count": len(detected_objects or []),
                "depth_occlusion_pairs": (
                    depth_occlusion.get("occluded_pairs", [])
                    if detected_objects else []
                ),
                "ar_drift_detected": (
                    ar_drift.get("has_drift", False)
                    if detected_objects else False
                ),
            },
        }
    
    def _check_occlusion(
        self,
        detected_objects: List[Dict[str, Any]],
        frame_width: int,
        frame_height: int
    ) -> Dict[str, Any]:
        """
        Проверяет, не обрезаны ли объекты краем кадра,
        и не занимают ли они слишком большую часть кадра (нет контекста).

        Args:
            detected_objects: список объектов с bbox
            frame_width, frame_height: размеры кадра

        Returns:
            {"has_occlusion": bool, "message": str, "occluded_objects": [...]}
        """
        occluded = []
        frame_area = max(1, frame_width * frame_height)

        for obj in detected_objects:
            bbox = obj.get("bbox", [])
            if len(bbox) != 4:
                continue

            x1, y1, x2, y2 = bbox

            # --- Проверка 1: обрезка краем кадра ---
            near_left   = x1 < self.EDGE_THRESHOLD_PX
            near_right  = x2 > frame_width  - self.EDGE_THRESHOLD_PX
            near_top    = y1 < self.EDGE_THRESHOLD_PX
            near_bottom = y2 > frame_height - self.EDGE_THRESHOLD_PX

            if near_left or near_right or near_top or near_bottom:
                direction = []
                if near_left:   direction.append("слева")
                if near_right:  direction.append("справа")
                if near_top:    direction.append("сверху")
                if near_bottom: direction.append("снизу")

                occluded.append({
                    "type": obj.get("type", "unknown"),
                    "direction": ", ".join(direction),
                    "reason": "edge_crop",
                })
                continue  # Нет смысла проверять «слишком близко» для уже обрезанного

            # --- Проверка 2: объект занимает >80% кадра (нет контекста) ---
            obj_area = (x2 - x1) * (y2 - y1)
            coverage = obj_area / frame_area

            if coverage > 0.80:
                occluded.append({
                    "type": obj.get("type", "unknown"),
                    "direction": "весь кадр",
                    "reason": "too_close",
                    "coverage_pct": round(coverage * 100, 1),
                })

        if occluded:
            edge_crop = [o for o in occluded if o.get("reason") == "edge_crop"]
            too_close = [o for o in occluded if o.get("reason") == "too_close"]

            messages = []
            if edge_crop:
                types = ", ".join(set(o["type"] for o in edge_crop))
                messages.append(
                    f"📐 Объект '{types}' обрезан краем кадра. "
                    f"Отойдите на 3 метра и возьмите ракурс шире."
                )
            if too_close:
                types = ", ".join(set(o["type"] for o in too_close))
                messages.append(
                    f"📷 Объект '{types}' занимает >80% кадра — нет контекста для понимания "
                    f"точек крепления. Отойдите дальше, чтобы захватить окружение."
                )

            return {
                "has_occlusion": True,
                "message": " | ".join(messages),
                "occluded_objects": occluded,
            }

        return {"has_occlusion": False, "message": "", "occluded_objects": []}
    
    def _check_ar_drift(
        self,
        detected_objects: List[Dict[str, Any]],
        ar_points: List[Dict[str, float]],
    ) -> Dict[str, Any]:
        """
        Сравнивает размеры объектов из YOLO с расстояниями между AR-точками.

        Если YOLO говорит «балка 2 м», а AR-точки показывают «3 м» —
        произошёл AR-дрифт или YOLO ошибся. В обоих случаях нужно предупредить пользователя.

        Алгоритм:
          1. Берём самый широкий beam из YOLO (реальная ширина в метрах).
          2. Вычисляем максимальное расстояние между AR-точками по осям X и Z.
          3. Если разрыв > DRIFT_TOLERANCE — сигнализируем о несоответствии.

        Args:
            detected_objects: список объектов (должны содержать real_width_m)
            ar_points: список AR-точек [{x, y, z}, ...]

        Returns:
            {"has_drift": bool, "message": str, "details": dict}
        """
        DRIFT_TOLERANCE = 0.40  # 40 см — допустимое расхождение

        if len(ar_points) < 2 or not detected_objects:
            return {"has_drift": False, "message": "", "details": {}}

        # Ширина самого крупного beam'а по YOLO
        beam_widths = [
            obj["real_width_m"]
            for obj in detected_objects
            if obj.get("type") == "beam" and "real_width_m" in obj
        ]

        if not beam_widths:
            # Если нет beam'ов — берём любой крупный объект
            beam_widths = [
                obj["real_width_m"]
                for obj in detected_objects
                if "real_width_m" in obj
            ]

        if not beam_widths:
            return {"has_drift": False, "message": "", "details": {}}

        yolo_size = max(beam_widths)

        # Максимальное расстояние между AR-точками в плоскости XZ (горизонталь)
        xs = [p.get("x", 0.0) for p in ar_points]
        zs = [p.get("z", 0.0) for p in ar_points]
        ar_span_x = max(xs) - min(xs)
        ar_span_z = max(zs) - min(zs)
        ar_span = max(ar_span_x, ar_span_z)

        delta = abs(yolo_size - ar_span)

        if delta > DRIFT_TOLERANCE:
            return {
                "has_drift": True,
                "message": (
                    f"📡 AR-дрифт или ошибка ИИ: YOLO определяет ширину объекта {yolo_size:.2f} м, "
                    f"а AR-точки показывают {ar_span:.2f} м (расхождение {delta:.2f} м). "
                    f"Пересканируйте пространство или переставьте AR-маркеры."
                ),
                "details": {
                    "yolo_size_m": round(yolo_size, 3),
                    "ar_span_m": round(ar_span, 3),
                    "delta_m": round(delta, 3),
                },
            }

        return {"has_drift": False, "message": "", "details": {}}

    def _check_depth_occlusion(
        self,
        detected_objects: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Проверяет перекрытие одних объектов другими (Depth Occlusion).

        Если балка частично закрыта другим объектом, генератор лесов не сможет
        найти точки крепления на перекрытой части.

        Алгоритм: для каждой пары объектов вычисляем IoU (Intersection over Union).
        Если IoU > порога — один объект перекрывает другой.

        Args:
            detected_objects: список объектов с bbox

        Returns:
            {"has_depth_occlusion": bool, "message": str, "occluded_pairs": [...]}
        """
        OVERLAP_THRESHOLD = 0.15  # 15% пересечения считается значимым перекрытием

        occluded_pairs = []

        for i, obj_a in enumerate(detected_objects):
            for j, obj_b in enumerate(detected_objects):
                if j <= i:
                    continue  # Проверяем каждую пару один раз

                bbox_a = obj_a.get("bbox", [])
                bbox_b = obj_b.get("bbox", [])

                if len(bbox_a) != 4 or len(bbox_b) != 4:
                    continue

                ax1, ay1, ax2, ay2 = bbox_a
                bx1, by1, bx2, by2 = bbox_b

                # Площадь пересечения
                inter_x1 = max(ax1, bx1)
                inter_y1 = max(ay1, by1)
                inter_x2 = min(ax2, bx2)
                inter_y2 = min(ay2, by2)

                if inter_x2 <= inter_x1 or inter_y2 <= inter_y1:
                    continue  # Нет пересечения

                inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
                area_a = max(1, (ax2 - ax1) * (ay2 - ay1))
                area_b = max(1, (bx2 - bx1) * (by2 - by1))
                union_area = area_a + area_b - inter_area

                iou = inter_area / union_area
                overlap_of_smaller = inter_area / min(area_a, area_b)

                if overlap_of_smaller > OVERLAP_THRESHOLD:
                    occluded_pairs.append({
                        "object_a": obj_a.get("type", "unknown"),
                        "object_b": obj_b.get("type", "unknown"),
                        "overlap_pct": round(overlap_of_smaller * 100, 1),
                    })

        if occluded_pairs:
            pair_desc = "; ".join(
                f"'{p['object_a']}' ↔ '{p['object_b']}' ({p['overlap_pct']}%)"
                for p in occluded_pairs
            )
            return {
                "has_depth_occlusion": True,
                "message": (
                    f"⚠️ Обнаружено перекрытие объектов: {pair_desc}. "
                    f"ИИ может не найти точки крепления на перекрытых участках. "
                    f"Сфотографируйте элементы отдельно."
                ),
                "occluded_pairs": occluded_pairs,
            }

        return {"has_depth_occlusion": False, "message": "", "occluded_pairs": []}

    def _to_gray(self, frame: np.ndarray) -> np.ndarray:
        """Конвертирует BGR в grayscale."""
        if CV2_AVAILABLE:
            return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Fallback через numpy (BGR → Gray)
        # Веса для BGR: B=0.114, G=0.587, R=0.299
        return np.dot(frame[..., :3], [0.114, 0.587, 0.299]).astype(np.uint8)
    
    def _resize_for_blur_analysis(self, gray: np.ndarray) -> np.ndarray:
        """
        Сжимает изображение до BLUR_ANALYSIS_SIZE для ускорения.
        Laplacian на больших изображениях медленный!
        
        Args:
            gray: grayscale изображение
        
        Returns:
            Уменьшенное изображение
        """
        h, w = gray.shape[:2]
        
        # Если уже достаточно маленькое, не трогаем
        if max(h, w) <= self.BLUR_ANALYSIS_SIZE:
            return gray
        
        # Вычисляем масштаб
        scale = self.BLUR_ANALYSIS_SIZE / max(h, w)
        new_w = int(w * scale)
        new_h = int(h * scale)
        
        if CV2_AVAILABLE:
            return cv2.resize(gray, (new_w, new_h), interpolation=cv2.INTER_AREA)
        else:
            # Fallback через numpy (простое сжатие)
            step_h = max(1, h // new_h)
            step_w = max(1, w // new_w)
            return gray[::step_h, ::step_w]
    
    def _laplacian_variance(self, gray: np.ndarray) -> float:
        """
        Вычисляет вариацию Laplacian (мера резкости).
        Чем больше значение, тем резче изображение.
        
        Args:
            gray: grayscale изображение
        
        Returns:
            Вариация Laplacian (float)
        """
        if CV2_AVAILABLE:
            laplacian = cv2.Laplacian(gray, cv2.CV_64F)
            return float(laplacian.var())
        
        # Fallback через numpy gradient
        gy, gx = np.gradient(gray.astype(float))
        return float(np.var(gx) + np.var(gy))


class VisionSystem:
    """
    Фасад, объединяющий детектор и диагностику.
    
    Улучшения v2.1:
    - Единое декодирование изображения
    - Передача готового frame в диагностику
    """

    def __init__(self, model_path: str = "yolov8n.pt") -> None:
        """
        Инициализация системы компьютерного зрения.
        
        Args:
            model_path: путь к YOLO модели
        """
        self.eyes = Eyes(model_path=model_path)
        self.diagnostician = SceneDiagnostician()
        logger.info("✓ VisionSystem initialized")

    def process_scene(
        self,
        image_bytes: bytes,
        distance: float,
        ar_points: Optional[List[Dict[str, float]]] = None,
        focal_length: Optional[float] = None,
        focal_length_x: Optional[float] = None,
        focal_length_y: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Полный цикл обработки сцены.
        
        ОПТИМИЗАЦИЯ: Декодирует изображение один раз!
        
        Args:
            image_bytes: байты изображения
            distance: расстояние до объекта (м)
            ar_points: опционально, AR-точки
            focal_length: фокусное расстояние (устаревший)
            focal_length_x, focal_length_y: фокусное расстояние по осям
        
        Returns:
            {
                "objects": [...],
                "quality": {...},
                "ready_for_design": bool
            }
        """
        try:
            # ОПТИМИЗАЦИЯ: Декодируем ОДИН раз, передаем готовый frame в оба модуля
            frame = self.eyes._decode_image_bgr(image_bytes)

            # 1. Детекция объектов — передаем готовый frame, декодирование внутри НЕ повторяется
            detected_objects = self.eyes.analyze_scene(
                frame=frame,
                distance_to_target=distance,
                focal_length=focal_length,
                focal_length_x=focal_length_x,
                focal_length_y=focal_length_y,
            )
            
            # 2. Проверка качества (передаем готовый frame!)
            quality = self.diagnostician.check_data_quality(
                frame,  # Готовый frame, не image_bytes!
                detected_objects,
                ar_points or [],
                distance
            )
            
            return {
                "objects": detected_objects,
                "quality": quality,
                "ready_for_design": quality["is_ready"]
            }
        
        except Exception as e:
            logger.error(f"Error in process_scene: {e}")
            return {
                "objects": [],
                "quality": {
                    "is_ready": False,
                    "quality_score": 0,
                    "instructions": [f"Ошибка обработки: {str(e)}"],
                    "warnings": [],
                    "metrics": {}
                },
                "ready_for_design": False
            }


# === УТИЛИТЫ ДЛЯ ОБРАТНОЙ СОВМЕСТИМОСТИ ===

def check_dependencies() -> Dict[str, bool]:
    """
    Проверяет доступность зависимостей.
    
    Returns:
        {"cv2": bool, "PIL": bool, "YOLO": bool}
    """
    return {
        "cv2": CV2_AVAILABLE,
        "PIL": PIL_AVAILABLE,
        "YOLO": YOLO_AVAILABLE
    }


def get_recommended_focal_length(camera_info: Optional[Dict[str, Any]] = None) -> Tuple[float, float]:
    """
    Возвращает рекомендуемое фокусное расстояние.
    
    Args:
        camera_info: опционально, данные из ARCore CameraIntrinsics
    
    Returns:
        (fx, fy) tuple
    """
    if camera_info:
        fx = camera_info.get("focal_length_x")
        fy = camera_info.get("focal_length_y")
        
        if fx and fy:
            return (fx, fy)
    
    # Default для типичных смартфонов
    return (800.0, 800.0)


# === ЭКСПОРТ ===
__all__ = [
    "Eyes",
    "SceneDiagnostician",
    "VisionSystem",
    "check_dependencies",
    "get_recommended_focal_length",
]
