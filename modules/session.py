# modules/session.py
import numpy as np
import time
from typing import List, Dict, Any, Optional

class DesignSession:
    def __init__(self, session_id: str, vision_system):
        self.id = session_id
        self.vision = vision_system
        self.start_time = time.time()
        self.keyframes = []
        self.user_anchors = []
        self.detected_supports = []

    def update_world_model(self, image_bytes: bytes, pose_matrix: List[float], markers: List[Dict]) -> Dict[str, Any]:
        """
        Обработка кадра и маркеров от Android
        """
        # 1. Анализ через Vision (с заглушкой distance=3.0)
        # В идеале нужно вытаскивать дистанцию из pose_matrix
        analysis = self.vision.process_scene(image_bytes, distance=3.0)
        
        # 2. Сохраняем маркеры, если они пришли
        if markers:
            for m in markers:
                # Простейшая защита от дублей (можно улучшить)
                self.user_anchors.append(m)

        # 3. Сохраняем ключевой кадр, если качество ОК
        if analysis.get("ready_for_design", False):
            self.keyframes.append({
                "pose": pose_matrix,
                "objects": analysis.get("objects", [])
            })

        return analysis.get("quality", {})

    def get_bounds(self) -> Dict[str, float]:
        """
        Вычисляет габариты зоны работ по точкам юзера.
        """
        if not self.user_anchors:
            # Дефолтные размеры, если точек нет
            return {"w": 4.0, "h": 3.0, "d": 2.0}
            
        points = np.array([[p['x'], p['y'], p['z']] for p in self.user_anchors])
        min_b = points.min(axis=0)
        max_b = points.max(axis=0)
        
        return {
            "w": float(max_b[0] - min_b[0]) + 1.0, # + запас
            "h": float(max_b[2] - min_b[2]) + 1.0,
            "d": float(max_b[1] - min_b[1]) + 1.0
        }

# --- ВОТ ЭТОГО КЛАССА НЕ ХВАТАЛО ---
class SessionStorage:
    def __init__(self):
        self._sessions: Dict[str, DesignSession] = {}

    def create_session(self, session_id: str, vision_system) -> DesignSession:
        session = DesignSession(session_id, vision_system)
        self._sessions[session_id] = session
        return session

    def get_session(self, session_id: str) -> Optional[DesignSession]:
        return self._sessions.get(session_id)

    def delete_session(self, session_id: str):
        if session_id in self._sessions:
            del self._sessions[session_id]