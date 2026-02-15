# modules/vision_enhanced.py
"""
УЛУЧШЕННЫЙ VISION МОДУЛЬ С КОСМИЧЕСКИМ AI

Интегрирует:
• Базовую YOLO детекцию (Eyes)
• Advanced Intelligence Engine
• 3D реконструкцию
• Предсказание скрытых структур
• ML-based анализ качества

АРХИТЕКТУРА:
┌──────────────────────────────────────────────────────────┐
│                     INPUT FRAME                           │
└─────────────────┬────────────────────────────────────────┘
                  │
       ┌──────────┴──────────┐
       │                      │
   ┌───▼────┐          ┌─────▼──────┐
   │  YOLO  │          │  FEATURES  │
   │Detection│          │ Extraction │
   └───┬────┘          └─────┬──────┘
       │                      │
       └──────────┬───────────┘
                  │
       ┌──────────▼──────────┐
       │  Advanced AI Engine  │
       │  • 3D Reconstruction │
       │  • Prediction        │
       │  • Interpolation     │
       └──────────┬───────────┘
                  │
       ┌──────────▼──────────┐
       │    OUTPUT MODEL      │
       │  • Complete 3D       │
       │  • Confidence map    │
       │  • AI feedback       │
       └──────────────────────┘
"""

import logging
from typing import Dict, List, Optional, Any, Tuple
import numpy as np
import cv2
from dataclasses import dataclass

try:
    from modules.advanced_intelligence import (
        AdvancedIntelligenceEngine,
        StructuralElement,
        Point3D,
        StructureType,
        ConfidenceLevel
    )
    ADVANCED_AI_AVAILABLE = True
except ImportError:
    ADVANCED_AI_AVAILABLE = False

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO = None
    YOLO_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class EnhancedFeedback:
    """
    Расширенная обратная связь для пользователя
    """
    # Основные метрики
    quality_score: float  # 0-100
    coverage_percentage: float  # Процент покрытия пространства
    reconstruction_quality: str  # POOR / FAIR / GOOD / EXCELLENT
    
    # Подсказки для пользователя
    instructions: List[str]
    warnings: List[str]
    suggestions: List[str]
    
    # Статус готовности
    is_ready: bool  # Готовность к моделированию
    min_points_needed: int
    current_points: int
    
    # Детали AI анализа
    detected_elements_count: int
    predicted_elements_count: int
    confidence_avg: float
    
    # Визуальная обратная связь
    heatmap_data: Optional[Dict[str, Any]] = None
    recommended_scan_areas: List[Dict[str, float]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Конвертация в словарь для API"""
        return {
            'quality_score': round(self.quality_score, 1),
            'coverage': round(self.coverage_percentage, 1),
            'reconstruction_quality': self.reconstruction_quality,
            'instructions': self.instructions,
            'warnings': self.warnings,
            'suggestions': self.suggestions,
            'is_ready': self.is_ready,
            'min_points_needed': self.min_points_needed,
            'current_points': self.current_points,
            'detected_count': self.detected_elements_count,
            'predicted_count': self.predicted_elements_count,
            'confidence': round(self.confidence_avg, 2)
        }


class EnhancedVisionSystem:
    """
    Продвинутая система компьютерного зрения с AI предсказанием.
    
    Возможности:
    • YOLO детекция конструктивных элементов
    • 3D реконструкция сцены
    • Предсказание скрытых частей
    • Качественная обратная связь
    • Адаптивные рекомендации
    """
    
    def __init__(self, model_path: str = "yolov8n.pt"):
        # Инициализация базового детектора
        self.yolo_model = None
        if YOLO_AVAILABLE:
            try:
                self.yolo_model = YOLO(model_path)
                logger.info(f"✓ YOLO model loaded: {model_path}")
            except Exception as e:
                logger.error(f"Failed to load YOLO: {e}")
        
        # Инициализация Advanced AI Engine
        self.ai_engine = None
        if ADVANCED_AI_AVAILABLE:
            try:
                self.ai_engine = AdvancedIntelligenceEngine()
                logger.info("✓ Advanced AI Engine initialized")
            except Exception as e:
                logger.error(f"Failed to init AI Engine: {e}")
        
        # Параметры качества
        self.min_quality_score = 60.0
        self.min_coverage = 50.0
        self.min_points = 2
        
        # Кеш для оптимизации
        self.frame_cache: List[Dict] = []
        self.last_feedback: Optional[EnhancedFeedback] = None
        
        logger.info("✓ Enhanced Vision System initialized")
    
    def process_frame(
        self,
        image_bytes: bytes,
        camera_pose: np.ndarray,
        ar_points: List[Dict[str, float]]
    ) -> EnhancedFeedback:
        """
        Обработка кадра с полным AI анализом.
        
        Args:
            image_bytes: Байты изображения
            camera_pose: 4x4 матрица камеры или 7 значений (x,y,z,qx,qy,qz,qw)
            ar_points: Точки опоры от пользователя
            
        Returns:
            EnhancedFeedback с подробным анализом
        """
        # Декодирование изображения
        frame = self._decode_image(image_bytes)
        
        if frame is None:
            return self._create_error_feedback("Ошибка декодирования кадра")
        
        # Детекция объектов через YOLO
        detected_objects = self._detect_objects(frame)
        
        # Конвертация camera_pose
        pose_matrix = self._convert_pose(camera_pose)
        
        # Обработка через Advanced AI Engine
        if self.ai_engine:
            ai_result = self.ai_engine.process_frame(
                image=frame,
                camera_pose=pose_matrix,
                detected_objects=detected_objects,
                ar_points=ar_points
            )
        else:
            ai_result = {
                'point_cloud_size': 0,
                'detected_elements': len(detected_objects),
                'predicted_elements': 0,
                'coverage_percentage': 0,
                'reconstruction_quality': 'UNAVAILABLE'
            }
        
        # Генерация обратной связи
        feedback = self._generate_feedback(
            frame=frame,
            detected_objects=detected_objects,
            ar_points=ar_points,
            ai_result=ai_result
        )
        
        self.last_feedback = feedback
        return feedback
    
    def _detect_objects(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """
        YOLO детекция объектов на кадре.
        """
        if self.yolo_model is None:
            return []
        
        try:
            results = self.yolo_model.predict(source=frame, verbose=False)
            
            objects = []
            for result in results:
                boxes = result.boxes
                
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    conf = float(box.conf[0])
                    cls = int(box.cls[0])
                    
                    # Маппинг класса
                    class_name = self._map_class_id(cls)
                    
                    objects.append({
                        'type': class_name,
                        'bbox': [float(x1), float(y1), float(x2), float(y2)],
                        'confidence': conf,
                        'center': [(x1 + x2) / 2, (y1 + y2) / 2],
                        'depth': 5.0  # Placeholder, должно идти от ARCore
                    })
            
            return objects
            
        except Exception as e:
            logger.error(f"YOLO detection failed: {e}")
            return []
    
    def _generate_feedback(
        self,
        frame: np.ndarray,
        detected_objects: List[Dict],
        ar_points: List[Dict],
        ai_result: Dict[str, Any]
    ) -> EnhancedFeedback:
        """
        Генерация умной обратной связи для пользователя.
        """
        instructions = []
        warnings = []
        suggestions = []
        
        # Количество точек
        num_points = len(ar_points)
        
        # Анализ качества кадра
        blur_score = self._assess_blur(frame)
        lighting_score = self._assess_lighting(frame)
        
        # Качество реконструкции
        quality_score = self._calculate_quality_score(
            blur_score=blur_score,
            lighting_score=lighting_score,
            num_points=num_points,
            coverage=ai_result.get('coverage_percentage', 0),
            ai_quality=ai_result.get('reconstruction_quality', 'POOR')
        )
        
        # Генерация инструкций
        if num_points < self.min_points:
            instructions.append(
                f"Установите минимум {self.min_points} точек опоры. "
                f"Сейчас: {num_points}"
            )
        else:
            instructions.append(
                f"Точек установлено: {num_points}. "
                f"Качество данных: {quality_score:.0f}%"
            )
        
        # Предупреждения по качеству
        if blur_score < 0.3:
            warnings.append("⚠️ Кадр размыт. Двигайтесь медленнее")
        
        if lighting_score < 0.4:
            warnings.append("⚠️ Плохое освещение. Используйте фонарик")
        
        coverage = ai_result.get('coverage_percentage', 0)
        if coverage < self.min_coverage:
            warnings.append(
                f"⚠️ Покрытие {coverage:.0f}%. "
                f"Рекомендуется ≥{self.min_coverage:.0f}%"
            )
        
        # Предложения по улучшению
        if len(detected_objects) < 3:
            suggestions.append(
                "💡 Мало видимых элементов. Подойдите ближе к конструкции"
            )
        
        if num_points >= self.min_points and coverage < 70:
            suggestions.append(
                "💡 Обойдите конструкцию с других сторон для лучшей модели"
            )
        
        # Проверка готовности к моделированию
        is_ready = (
            num_points >= self.min_points and
            quality_score >= self.min_quality_score and
            coverage >= self.min_coverage
        )
        
        # Средняя уверенность AI
        confidence_avg = 0.7  # Placeholder
        if self.ai_engine:
            conf_map = self.ai_engine._generate_confidence_map()
            confidence_avg = conf_map.get('average_confidence', 0.7)
        
        return EnhancedFeedback(
            quality_score=quality_score,
            coverage_percentage=coverage,
            reconstruction_quality=ai_result.get('reconstruction_quality', 'FAIR'),
            instructions=instructions,
            warnings=warnings,
            suggestions=suggestions,
            is_ready=is_ready,
            min_points_needed=self.min_points,
            current_points=num_points,
            detected_elements_count=ai_result.get('detected_elements', 0),
            predicted_elements_count=ai_result.get('predicted_elements', 0),
            confidence_avg=confidence_avg
        )
    
    def get_complete_model(self) -> Dict[str, Any]:
        """
        Получить полную 3D модель с предсказанными элементами.
        """
        if self.ai_engine:
            return self.ai_engine.get_complete_model()
        else:
            return {
                'point_cloud': [],
                'structural_elements': [],
                'bounds': None,
                'quality_metrics': {
                    'coverage': 0,
                    'reconstruction_quality': 'UNAVAILABLE',
                    'total_frames': 0
                }
            }
    
    def clear_session(self):
        """Очистка данных (новая сессия)"""
        if self.ai_engine:
            self.ai_engine.clear()
        
        self.frame_cache.clear()
        self.last_feedback = None
        
        logger.info("Vision system cleared")
    
    # ═══════════════════════════════════════════════════════════════════════
    # ВСПОМОГАТЕЛЬНЫЕ МЕТОДЫ
    # ═══════════════════════════════════════════════════════════════════════
    
    def _decode_image(self, image_bytes: bytes) -> Optional[np.ndarray]:
        """Декодирование изображения из байтов"""
        try:
            nparr = np.frombuffer(image_bytes, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            return frame
        except Exception as e:
            logger.error(f"Image decode error: {e}")
            return None
    
    def _convert_pose(self, camera_pose: Any) -> np.ndarray:
        """
        Конвертация camera_pose в 4x4 матрицу.
        
        Принимает:
        - 4x4 numpy array
        - 7 значений (x, y, z, qx, qy, qz, qw)
        - List из 7 элементов
        """
        if isinstance(camera_pose, np.ndarray):
            if camera_pose.shape == (4, 4):
                return camera_pose
            elif camera_pose.shape == (7,):
                return self._pose_from_quaternion(camera_pose)
        
        if isinstance(camera_pose, (list, tuple)):
            if len(camera_pose) == 7:
                return self._pose_from_quaternion(np.array(camera_pose))
        
        # Fallback: identity matrix
        return np.eye(4)
    
    def _pose_from_quaternion(self, pose_7: np.ndarray) -> np.ndarray:
        """
        Создание 4x4 матрицы из [x,y,z,qx,qy,qz,qw]
        """
        x, y, z, qx, qy, qz, qw = pose_7
        
        # Конвертация quaternion в rotation matrix
        # (упрощенная версия, в продакшене использовать scipy.spatial.transform)
        
        matrix = np.eye(4)
        matrix[0, 3] = x
        matrix[1, 3] = y
        matrix[2, 3] = z
        
        # Rotation part (simplified)
        xx, yy, zz = qx*qx, qy*qy, qz*qz
        xy, xz, yz = qx*qy, qx*qz, qy*qz
        wx, wy, wz = qw*qx, qw*qy, qw*qz
        
        matrix[0, 0] = 1 - 2*(yy + zz)
        matrix[0, 1] = 2*(xy - wz)
        matrix[0, 2] = 2*(xz + wy)
        
        matrix[1, 0] = 2*(xy + wz)
        matrix[1, 1] = 1 - 2*(xx + zz)
        matrix[1, 2] = 2*(yz - wx)
        
        matrix[2, 0] = 2*(xz - wy)
        matrix[2, 1] = 2*(yz + wx)
        matrix[2, 2] = 1 - 2*(xx + yy)
        
        return matrix
    
    def _map_class_id(self, class_id: int) -> str:
        """Маппинг YOLO class ID на строковое название"""
        class_map = {
            0: "beam",
            1: "pipe_obstacle",
            2: "safety_equipment",
            3: "column",
            4: "floor_slab",
            5: "cable_tray"
        }
        return class_map.get(class_id, "unknown")
    
    def _assess_blur(self, frame: np.ndarray) -> float:
        """
        Оценка размытости кадра (0-1, выше = лучше).
        
        Использует variance of Laplacian.
        """
        if frame is None or frame.size == 0:
            return 0.0
        
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Resize для скорости
            small = cv2.resize(gray, (320, 240))
            
            # Laplacian variance
            laplacian_var = cv2.Laplacian(small, cv2.CV_64F).var()
            
            # Нормализация к 0-1
            # Обычно хорошее изображение имеет variance > 100
            score = min(laplacian_var / 200.0, 1.0)
            
            return score
            
        except Exception as e:
            logger.error(f"Blur assessment failed: {e}")
            return 0.5
    
    def _assess_lighting(self, frame: np.ndarray) -> float:
        """
        Оценка освещённости кадра (0-1).
        """
        if frame is None or frame.size == 0:
            return 0.0
        
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Средняя яркость
            mean_brightness = np.mean(gray)
            
            # Нормализация: оптимально 80-180
            if mean_brightness < 80:
                score = mean_brightness / 80.0
            elif mean_brightness > 180:
                score = (255 - mean_brightness) / 75.0
            else:
                score = 1.0
            
            return max(0.0, min(1.0, score))
            
        except Exception as e:
            logger.error(f"Lighting assessment failed: {e}")
            return 0.5
    
    def _calculate_quality_score(
        self,
        blur_score: float,
        lighting_score: float,
        num_points: int,
        coverage: float,
        ai_quality: str
    ) -> float:
        """
        Комбинированный скор качества (0-100).
        """
        # Веса компонентов
        weights = {
            'blur': 0.15,
            'lighting': 0.15,
            'points': 0.25,
            'coverage': 0.30,
            'ai_quality': 0.15
        }
        
        # Нормализация точек (2-10)
        points_norm = min(num_points / 10.0, 1.0)
        
        # Нормализация coverage (0-100%)
        coverage_norm = coverage / 100.0
        
        # AI quality: POOR=0.3, FAIR=0.6, GOOD=0.8, EXCELLENT=1.0
        ai_quality_map = {
            'POOR': 0.3,
            'FAIR': 0.6,
            'GOOD': 0.8,
            'EXCELLENT': 1.0,
            'UNAVAILABLE': 0.5
        }
        ai_quality_norm = ai_quality_map.get(ai_quality, 0.5)
        
        # Комбинированный скор
        score = (
            weights['blur'] * blur_score +
            weights['lighting'] * lighting_score +
            weights['points'] * points_norm +
            weights['coverage'] * coverage_norm +
            weights['ai_quality'] * ai_quality_norm
        ) * 100
        
        return round(score, 1)
    
    def _create_error_feedback(self, error_message: str) -> EnhancedFeedback:
        """Создание feedback при ошибке"""
        return EnhancedFeedback(
            quality_score=0.0,
            coverage_percentage=0.0,
            reconstruction_quality='ERROR',
            instructions=[],
            warnings=[f"❌ {error_message}"],
            suggestions=[],
            is_ready=False,
            min_points_needed=self.min_points,
            current_points=0,
            detected_elements_count=0,
            predicted_elements_count=0,
            confidence_avg=0.0
        )


# ═══════════════════════════════════════════════════════════════════════════
# ПУБЛИЧНЫЙ API
# ═══════════════════════════════════════════════════════════════════════════

def create_vision_system(model_path: str = "yolov8n.pt") -> EnhancedVisionSystem:
    """Фабрика для создания vision system"""
    return EnhancedVisionSystem(model_path=model_path)


# Обратная совместимость со старым API
class VisionSystem:
    """
    Обёртка для обратной совместимости.
    """
    def __init__(self):
        self.enhanced = create_vision_system()
    
    def process_scene(
        self,
        image_bytes: bytes,
        pose_matrix: Any,
        markers: List[Dict]
    ) -> Dict[str, Any]:
        """Старый API endpoint"""
        feedback = self.enhanced.process_frame(
            image_bytes=image_bytes,
            camera_pose=pose_matrix,
            ar_points=markers
        )
        
        # Конвертация в старый формат
        return feedback.to_dict()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Тест системы
    vision = create_vision_system()
    logger.info("✓ Enhanced Vision System initialized successfully")
    
    # Симуляция обработки
    dummy_image = np.zeros((720, 1280, 3), dtype=np.uint8)
    _, image_bytes = cv2.imencode('.jpg', dummy_image)
    
    feedback = vision.process_frame(
        image_bytes=image_bytes.tobytes(),
        camera_pose=np.eye(4),
        ar_points=[]
    )
    
    logger.info(f"Test feedback: Quality={feedback.quality_score}%")
    logger.info(f"Instructions: {feedback.instructions}")