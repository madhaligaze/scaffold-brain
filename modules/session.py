# modules/session.py
import math
import time
import logging
from typing import List, Dict, Any, Optional

import numpy as np

logger = logging.getLogger(__name__)


class DesignSession:
    """
    Сессия замера: хранит историю кадров, маркеры пользователя и AI-точки опор.

    ИСПРАВЛЕНИЯ:
    - Добавлен self.session_id (раньше был self.id — несоответствие с main.py)
    - Добавлен self.status (раньше AttributeError в /session/stream и /session/model)
    - detected_supports теперь заполняется из Vision (раньше всегда пустой)
    - distance вычисляется из pose_matrix (раньше захардкожен 3.0)
    - Дедупликация маркеров по расстоянию (раньше дублей не было защиты)
    """

    DEDUP_THRESHOLD_M = 0.15  # Маркеры ближе 15 см считаются дублями

    def __init__(self, session_id: str, vision_system):
        self.session_id = session_id          # ИСПРАВЛЕНО: было self.id
        self.vision = vision_system
        self.status = "MEASURING"             # ИСПРАВЛЕНО: атрибут добавлен
        self.start_time = time.time()
        self.keyframes: List[Dict] = []
        self.user_anchors: List[Dict] = []
        self.detected_supports: List[Dict] = []

    def update_world_model(
        self,
        image_bytes: bytes,
        pose_matrix: List[float],
        markers: List[Dict],
    ) -> Dict[str, Any]:
        """
        Обработка кадра и маркеров от Android.

        Args:
            image_bytes: JPEG/PNG кадр с камеры (base64-декодированный)
            pose_matrix: [tx, ty, tz, qx, qy, qz, qw] — позиция и ориентация камеры
            markers: список точек от пользователя [{x, y, z}, ...]

        Returns:
            quality dict для отображения в tvAiHint
        """
        # 1. Вычисляем расстояние из pose_matrix вместо хардкода 3.0
        #    ИСПРАВЛЕНО: раньше всегда было distance=3.0
        distance = self._extract_distance(pose_matrix)

        # 2. Анализ через VisionSystem
        try:
            analysis = self.vision.process_scene(
                image_bytes,
                distance=distance,
                ar_points=markers or [],
            )
        except Exception:
            logger.error("VisionSystem.process_scene failed", exc_info=True)
            analysis = {
                "objects": [],
                "quality": {"instructions": [], "warnings": ["Ошибка анализа кадра"]},
                "ready_for_design": False,
            }

        # 3. Сохраняем маркеры пользователя с дедупликацией
        #    ИСПРАВЛЕНО: раньше дубли добавлялись без ограничений
        if markers:
            for m in markers:
                if self._is_new_marker(m):
                    self.user_anchors.append(m)

        # 4. Сохраняем AI-детектированные опоры из объектов на кадре
        #    ИСПРАВЛЕНО: раньше detected_supports никогда не заполнялся
        self._update_detected_supports(
            analysis.get("objects", []),
            pose_matrix,
        )

        # 5. Сохраняем ключевой кадр если качество OK
        if analysis.get("ready_for_design", False):
            self.keyframes.append({
                "pose": pose_matrix,
                "objects": analysis.get("objects", []),
                "timestamp": time.time(),
            })
            logger.info(
                f"Session {self.session_id}: keyframe saved "
                f"(total={len(self.keyframes)}, distance={distance:.2f}m)"
            )

        return analysis.get("quality", {"instructions": [], "warnings": []})

    def get_bounds(self) -> Dict[str, float]:
        """
        Вычисляет габариты зоны работ по точкам пользователя.

        Returns:
            {"w": float, "h": float, "d": float} — ширина, высота, глубина в метрах
        """
        if not self.user_anchors:
            logger.warning(f"Session {self.session_id}: no user anchors, using defaults")
            return {"w": 4.0, "h": 3.0, "d": 2.0}

        try:
            points = np.array([
                [float(p.get("x", 0)), float(p.get("y", 0)), float(p.get("z", 0))]
                for p in self.user_anchors
            ])
            min_b = points.min(axis=0)
            max_b = points.max(axis=0)

            return {
                "w": float(max_b[0] - min_b[0]) + 1.0,  # + запас 1м
                "h": float(max_b[2] - min_b[2]) + 1.0,
                "d": float(max_b[1] - min_b[1]) + 1.0,
            }
        except Exception:
            logger.error("get_bounds failed, using defaults", exc_info=True)
            return {"w": 4.0, "h": 3.0, "d": 2.0}

    # === ВСПОМОГАТЕЛЬНЫЕ МЕТОДЫ ===

    def _extract_distance(self, pose_matrix: List[float]) -> float:
        """
        Вычисляет расстояние от камеры до начала координат из pose_matrix.

        ARCore pose_matrix = [tx, ty, tz, qx, qy, qz, qw]
        Расстояние = евклидова длина вектора трансляции (tx, ty, tz).

        Args:
            pose_matrix: список из 7 элементов (или пустой)

        Returns:
            Расстояние в метрах (зажато в диапазоне [1.0, 10.0])
        """
        if not pose_matrix or len(pose_matrix) < 3:
            return 3.0  # разумный дефолт

        tx, ty, tz = pose_matrix[0], pose_matrix[1], pose_matrix[2]
        distance = math.sqrt(tx**2 + ty**2 + tz**2)

        # Зажимаем: < 1м — слишком близко для детекции, > 10м — слишком далеко
        return max(1.0, min(10.0, distance)) if distance > 0 else 3.0

    def _is_new_marker(self, marker: Dict) -> bool:
        """
        Проверяет, что маркер не является дублём существующего.
        Считает дублём точки ближе DEDUP_THRESHOLD_M метров.
        """
        try:
            mx = float(marker.get("x", 0))
            my = float(marker.get("y", 0))
            mz = float(marker.get("z", 0))
        except (TypeError, ValueError):
            return False

        for existing in self.user_anchors:
            try:
                dx = mx - float(existing.get("x", 0))
                dy = my - float(existing.get("y", 0))
                dz = mz - float(existing.get("z", 0))
                if math.sqrt(dx**2 + dy**2 + dz**2) < self.DEDUP_THRESHOLD_M:
                    return False
            except (TypeError, ValueError):
                continue

        return True

    def _update_detected_supports(
        self,
        objects: List[Dict],
        pose_matrix: List[float],
    ) -> None:
        """
        Сохраняет несущие балки из Vision в detected_supports.

        Конвертация из пиксельных координат в метровые — грубая,
        требует camera intrinsics для точности. Сейчас используем
        нормализованные координаты (центр bbox / размер кадра).
        """
        for obj in objects:
            obj_type = obj.get("type", "")

            # Сохраняем только несущие элементы (не препятствия)
            if obj_type not in ("beam", "column", "floor_slab"):
                continue

            center = obj.get("center", [])
            if len(center) < 2:
                continue

            # Грубое приближение: используем реальные размеры из YOLO
            # и позицию камеры. Точная версия требует CameraIntrinsics от ARCore.
            support_point = {
                "x": float(obj.get("real_width_m", 1.0)) / 2.0,
                "y": float(self._extract_distance(pose_matrix)),
                "z": float(obj.get("real_height_m", 1.0)) / 2.0,
                "type": obj_type,
                "confidence": float(obj.get("confidence", 0.5)),
            }

            self.detected_supports.append(support_point)

        # Ограничиваем размер, чтобы не копить мусор
        if len(self.detected_supports) > 200:
            self.detected_supports = self.detected_supports[-200:]


class SessionStorage:
    """
    Хранилище сессий в памяти процесса.

    ИСПРАВЛЕНИЯ:
    - Добавлены методы save() и load() (раньше их не было → AttributeError в main.py)
    - Унификация интерфейса: save/load/delete используют session_id строку

    NOTE для продакшена: хранилище in-memory — все сессии теряются при рестарте.
    Для production нужен Redis или файловый бекенд (см. рекомендацию в REPORT.md).
    """

    def __init__(self):
        self._sessions: Dict[str, DesignSession] = {}

    def save(self, session: DesignSession) -> None:
        """
        Сохраняет сессию по session.session_id.

        ДОБАВЛЕНО: метод отсутствовал → main.py падал с AttributeError
        """
        self._sessions[session.session_id] = session
        logger.debug(f"SessionStorage: saved session {session.session_id}")

    def load(self, session_id: str) -> Optional[DesignSession]:
        """
        Загружает сессию по ID.

        ДОБАВЛЕНО: метод отсутствовал → main.py падал с AttributeError

        Returns:
            DesignSession или None если не найдена
        """
        session = self._sessions.get(session_id)
        if session:
            logger.debug(f"SessionStorage: loaded session {session_id}")
        else:
            logger.warning(f"SessionStorage: session {session_id} not found")
        return session

    def delete(self, session_id: str) -> bool:
        """
        Удаляет сессию.

        Returns:
            True если удалена, False если не существовала
        """
        if session_id in self._sessions:
            del self._sessions[session_id]
            logger.info(f"SessionStorage: deleted session {session_id}")
            return True
        return False

    # === Обратная совместимость со старым API ===

    def create_session(self, session_id: str, vision_system) -> DesignSession:
        """Создаёт и сохраняет новую сессию."""
        session = DesignSession(session_id, vision_system)
        self.save(session)
        return session

    def get_session(self, session_id: str) -> Optional[DesignSession]:
        """Псевдоним для load() — обратная совместимость."""
        return self.load(session_id)

    def delete_session(self, session_id: str) -> None:
        """Псевдоним для delete() — обратная совместимость."""
        self.delete(session_id)

    @property
    def active_count(self) -> int:
        """Количество активных сессий."""
        return len(self._sessions)


# Совместимость с обновленным импортом в main.py
