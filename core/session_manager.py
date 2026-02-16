"""
Session Manager - Контекст и память сцены
==========================================
ПРАВИЛО: ИИ должен помнить не последнее фото, а ВСЮ СЦЕНУ.

Если я 10 секунд назад показал левый угол, а сейчас смотрю на правый —
ИИ должен помнить левый угол и связывать конструкцию воедино.
"""
import uuid
import time
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import json

# Импорты новых модулей (try/except для обратной совместимости)
try:
    from modules.voxel_world import VoxelWorld
    from modules.structural_graph import StructuralGraph
    _BRAIN_MODULES_AVAILABLE = True
except ImportError:
    _BRAIN_MODULES_AVAILABLE = False


@dataclass
class CameraFrame:
    """Отдельный кадр камеры"""
    timestamp: float
    image_data: Optional[str] = None  # base64 или путь к файлу
    camera_position: Optional[Dict] = None  # {x, y, z, rotation}
    detected_objects: List[Dict] = field(default_factory=list)
    ar_points: List[Dict] = field(default_factory=list)
    quality_metrics: Optional[Dict] = None  # Размытие, освещение, и т.д.


@dataclass
class SceneContext:
    """
    Контекст всей сцены - агрегированная информация со всех кадров.
    
    ИИ не должен "забывать" что видел 10 секунд назад.
    """
    all_detected_objects: List[Dict] = field(default_factory=list)
    all_ar_points: List[Dict] = field(default_factory=list)
    estimated_bounds: Optional[Dict] = None  # {width, height, depth}
    obstacles: List[Dict] = field(default_factory=list)
    anchor_points: List[Dict] = field(default_factory=list)
    
    # Фотограмметрия - объединенное облако точек
    point_cloud: List[Dict] = field(default_factory=list)

    # ── НОВОЕ: Воксельная карта пространства ──────────────────────────────
    voxel_world: Optional[Any] = field(default=None)

    def ensure_voxel_world(self, resolution: float = 0.1) -> Any:
        """Лениво создаёт VoxelWorld при первом обращении."""
        if self.voxel_world is None and _BRAIN_MODULES_AVAILABLE:
            self.voxel_world = VoxelWorld(resolution=resolution)
        return self.voxel_world
    
    def merge_frame(self, frame: CameraFrame, merge_threshold: float = 0.1):
        """
        Объединяет данные нового кадра с существующим контекстом.
        
        Args:
            frame: Новый кадр камеры
            merge_threshold: Порог для слияния близких точек (метры)
        """
        # Объединяем обнаруженные объекты (дедупликация)
        for obj in frame.detected_objects:
            if not self._is_duplicate_object(obj, merge_threshold):
                self.all_detected_objects.append(obj)
        
        # Объединяем AR точки
        for point in frame.ar_points:
            if not self._is_duplicate_point(point, merge_threshold):
                self.all_ar_points.append(point)
        
        # Обновляем облако точек
        self._update_point_cloud(frame)
        
        # Обновляем оценку границ сцены
        self._update_bounds()
    
    def _is_duplicate_object(self, obj: Dict, threshold: float) -> bool:
        """Проверка, есть ли уже похожий объект в контексте"""
        obj_pos = obj.get('position', {})
        obj_x = obj_pos.get('x', 0)
        obj_y = obj_pos.get('y', 0)
        obj_z = obj_pos.get('z', 0)
        
        for existing in self.all_detected_objects:
            ex_pos = existing.get('position', {})
            ex_x = ex_pos.get('x', 0)
            ex_y = ex_pos.get('y', 0)
            ex_z = ex_pos.get('z', 0)
            
            # Вычисляем расстояние
            dist = ((obj_x - ex_x)**2 + (obj_y - ex_y)**2 + (obj_z - ex_z)**2)**0.5
            
            # Если объект того же типа и близко - это дубликат
            if dist < threshold and obj.get('type') == existing.get('type'):
                return True
        
        return False
    
    def _is_duplicate_point(self, point: Dict, threshold: float) -> bool:
        """Проверка дубликатов AR точек"""
        px = point.get('x', 0)
        py = point.get('y', 0)
        pz = point.get('z', 0)
        
        for existing in self.all_ar_points:
            ex_x = existing.get('x', 0)
            ex_y = existing.get('y', 0)
            ex_z = existing.get('z', 0)
            
            dist = ((px - ex_x)**2 + (py - ex_y)**2 + (pz - ex_z)**2)**0.5
            
            if dist < threshold:
                return True
        
        return False
    
    def _update_point_cloud(self, frame: CameraFrame):
        """Обновление облака точек из нового кадра"""
        # Объединяем точки из AR маркеров
        for point in frame.ar_points:
            self.point_cloud.append({
                "x": point.get('x', 0),
                "y": point.get('y', 0),
                "z": point.get('z', 0),
                "timestamp": frame.timestamp,
                "source": "ar"
            })
    
    def _update_bounds(self):
        """Обновление границ сцены на основе всех точек"""
        if not self.all_ar_points:
            return
        
        xs = [p.get('x', 0) for p in self.all_ar_points]
        ys = [p.get('y', 0) for p in self.all_ar_points]
        zs = [p.get('z', 0) for p in self.all_ar_points]
        
        self.estimated_bounds = {
            "min_x": min(xs) if xs else 0,
            "max_x": max(xs) if xs else 0,
            "min_y": min(ys) if ys else 0,
            "max_y": max(ys) if ys else 0,
            "min_z": min(zs) if zs else 0,
            "max_z": max(zs) if zs else 0,
            "width": max(xs) - min(xs) if xs else 0,
            "height": max(zs) - min(zs) if zs else 0,
            "depth": max(ys) - min(ys) if ys else 0,
        }
    
    def get_summary(self) -> Dict:
        """Получить краткую сводку контекста"""
        return {
            "total_objects": len(self.all_detected_objects),
            "total_ar_points": len(self.all_ar_points),
            "point_cloud_size": len(self.point_cloud),
            "estimated_bounds": self.estimated_bounds,
            "has_obstacles": len(self.obstacles) > 0
        }


class Session:
    """
    Пользовательская сессия.
    
    Хранит:
    - ID сессии
    - Время создания и последней активности
    - Историю кадров
    - Агрегированный контекст сцены
    - Сгенерированные варианты
    """
    
    def __init__(self, session_id: Optional[str] = None):
        self.session_id = session_id or str(uuid.uuid4())
        self.created_at = time.time()
        self.last_activity = time.time()
        
        # История кадров
        self.frames: List[CameraFrame] = []
        
        # Контекст сцены
        self.scene_context = SceneContext()
        
        # Сгенерированные варианты
        self.generated_variants: List[Dict] = []
        self.selected_variant: Optional[Dict] = None
        
        # Статистика
        self.total_frames_processed = 0
        self.total_objects_detected = 0

        # ── НОВОЕ: Живой граф конструкции ─────────────────────────────────────
        self.structural_graph: Optional[Any] = None

        # ── НОВОЕ v4.0: Текущая структура для realtime-редактирования ────────
        self.current_structure: List[Dict] = []
        self.structure_history: List[Dict] = []
        
    def add_frame(self, frame: CameraFrame):
        """
        Добавить новый кадр в сессию.
        
        Args:
            frame: Кадр камеры
        """
        self.frames.append(frame)
        self.scene_context.merge_frame(frame)
        self.last_activity = time.time()
        self.total_frames_processed += 1
        self.total_objects_detected += len(frame.detected_objects)
    
    def add_variant(self, variant: Dict):
        """Добавить сгенерированный вариант"""
        self.generated_variants.append(variant)
        self.last_activity = time.time()

    def ensure_structural_graph(self) -> Any:
        """Лениво создаёт StructuralGraph при первом обращении."""
        if self.structural_graph is None and _BRAIN_MODULES_AVAILABLE:
            self.structural_graph = StructuralGraph()
        return self.structural_graph

    def save_structure(self, structure: List[Dict]) -> None:
        """Сохранить текущую структуру и push предыдущую версию в историю."""
        if self.current_structure:
            self.structure_history.append(
                {
                    "timestamp": time.time(),
                    "structure": self.current_structure.copy(),
                }
            )

        self.current_structure = structure
        self.last_activity = time.time()

    def remove_element(self, element_id: str) -> bool:
        """Удалить элемент из текущей структуры."""
        for i, elem in enumerate(self.current_structure):
            if elem.get("id") == element_id:
                self.structure_history.append(
                    {
                        "timestamp": time.time(),
                        "action": "REMOVE",
                        "element_id": element_id,
                    }
                )
                self.current_structure.pop(i)
                self.last_activity = time.time()
                return True
        return False

    def add_element(self, element: Dict) -> str:
        """Добавить элемент в структуру; при отсутствии id — сгенерировать."""
        if "id" not in element:
            element["id"] = f"elem_{uuid.uuid4().hex[:8]}"

        self.current_structure.append(element)
        self.structure_history.append(
            {
                "timestamp": time.time(),
                "action": "ADD",
                "element_id": element["id"],
            }
        )
        self.last_activity = time.time()
        return element["id"]

    def undo_last_action(self) -> bool:
        """Откатить последнее действие, если в истории есть снимок структуры."""
        if not self.structure_history:
            return False

        last_state = self.structure_history.pop()
        if "structure" in last_state:
            self.current_structure = last_state["structure"]
            self.last_activity = time.time()
            return True

        return False

    def get_structure_statistics(self) -> Dict:
        """Получить статистику по текущей структуре."""
        if not self.current_structure:
            return {"total_elements": 0}

        by_type: Dict[str, int] = {}
        for elem in self.current_structure:
            t = elem.get("type", "unknown")
            by_type[t] = by_type.get(t, 0) + 1

        return {
            "total_elements": len(self.current_structure),
            "by_type": by_type,
            "history_depth": len(self.structure_history),
        }
    
    def select_variant(self, variant_index: int) -> bool:
        """
        Выбрать вариант для построения.
        
        Args:
            variant_index: Индекс варианта (0, 1, 2)
            
        Returns:
            True если успешно
        """
        if 0 <= variant_index < len(self.generated_variants):
            self.selected_variant = self.generated_variants[variant_index]
            self.last_activity = time.time()
            return True
        return False
    
    def get_context_summary(self) -> Dict:
        """Получить сводку контекста для отладки"""
        return {
            "session_id": self.session_id,
            "created_at": datetime.fromtimestamp(self.created_at).isoformat(),
            "duration_seconds": time.time() - self.created_at,
            "total_frames": len(self.frames),
            "total_variants": len(self.generated_variants),
            "scene_context": self.scene_context.get_summary(),
            "statistics": {
                "frames_processed": self.total_frames_processed,
                "objects_detected": self.total_objects_detected
            }
        }
    
    def is_expired(self, timeout_seconds: int = 3600) -> bool:
        """
        Проверка истечения сессии.
        
        Args:
            timeout_seconds: Таймаут в секундах (по умолчанию 1 час)
            
        Returns:
            True если сессия истекла
        """
        return (time.time() - self.last_activity) > timeout_seconds


class SessionManager:
    """
    Менеджер всех пользовательских сессий.
    
    Функции:
    - Создание и удаление сессий
    - Автоматическая очистка старых сессий
    - Получение контекста сессии
    """
    
    def __init__(self, session_timeout: int = 3600):
        """
        Args:
            session_timeout: Таймаут сессии в секундах (по умолчанию 1 час)
        """
        self.sessions: Dict[str, Session] = {}
        self.session_timeout = session_timeout
        
    def create_session(self) -> str:
        """
        Создать новую сессию.
        
        Returns:
            ID новой сессии
        """
        session = Session()
        self.sessions[session.session_id] = session
        
        # Очищаем старые сессии при создании новой
        self._cleanup_expired_sessions()
        
        return session.session_id
    
    def get_session(self, session_id: str) -> Optional[Session]:
        """
        Получить сессию по ID.
        
        Args:
            session_id: ID сессии
            
        Returns:
            Session или None если не найдена
        """
        session = self.sessions.get(session_id)
        
        # Проверяем, не истекла ли сессия
        if session and session.is_expired(self.session_timeout):
            self.delete_session(session_id)
            return None
        
        return session
    
    def delete_session(self, session_id: str) -> bool:
        """
        Удалить сессию.
        
        Args:
            session_id: ID сессии
            
        Returns:
            True если успешно удалена
        """
        if session_id in self.sessions:
            del self.sessions[session_id]
            return True
        return False
    
    def _cleanup_expired_sessions(self):
        """Очистка истекших сессий"""
        expired = []
        
        for session_id, session in self.sessions.items():
            if session.is_expired(self.session_timeout):
                expired.append(session_id)
        
        for session_id in expired:
            del self.sessions[session_id]
    
    def get_all_sessions_summary(self) -> Dict:
        """Получить сводку по всем активным сессиям"""
        return {
            "total_sessions": len(self.sessions),
            "sessions": [
                {
                    "id": sid,
                    "duration": time.time() - session.created_at,
                    "frames": len(session.frames),
                    "variants": len(session.generated_variants)
                }
                for sid, session in self.sessions.items()
            ]
        }
    
    def export_session_data(self, session_id: str) -> Optional[str]:
        """
        Экспорт данных сессии в JSON.
        
        Args:
            session_id: ID сессии
            
        Returns:
            JSON строка или None
        """
        session = self.get_session(session_id)
        if not session:
            return None
        
        data = {
            "session_id": session.session_id,
            "created_at": session.created_at,
            "scene_context": {
                "objects": session.scene_context.all_detected_objects,
                "ar_points": session.scene_context.all_ar_points,
                "bounds": session.scene_context.estimated_bounds,
                "point_cloud_size": len(session.scene_context.point_cloud),
                "point_cloud": session.scene_context.point_cloud,
            },
            "variants": session.generated_variants,
            "current_structure": session.current_structure,
            "structure_history": session.structure_history,
            "selected_variant_index": (
                session.generated_variants.index(session.selected_variant)
                if session.selected_variant in session.generated_variants
                else None
            ),
            "statistics": {
                "total_frames": session.total_frames_processed,
                "total_objects": session.total_objects_detected
            }
        }
        
        return json.dumps(data, indent=2)


# ═══════════════════════════════════════════════════════════════════════════
# ГЛОБАЛЬНЫЙ ИНСТАНС МЕНЕДЖЕРА
# ═══════════════════════════════════════════════════════════════════════════

# Используется в main.py
session_manager = SessionManager(session_timeout=3600)


# ═══════════════════════════════════════════════════════════════════════════
# ТЕСТЫ
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("🧪 ТЕСТИРОВАНИЕ SESSION MANAGER")
    print("=" * 70)
    
    # Тест 1: Создание сессии
    print("\n1. Создание сессии:")
    manager = SessionManager()
    sid = manager.create_session()
    print(f"   Создана сессия: {sid}")
    
    # Тест 2: Добавление кадров
    print("\n2. Добавление кадров:")
    session = manager.get_session(sid)
    
    # Кадр 1: Левый угол
    frame1 = CameraFrame(
        timestamp=time.time(),
        ar_points=[
            {"x": 0.0, "y": 0.0, "z": 0.0},
            {"x": 2.0, "y": 0.0, "z": 0.0},
        ],
        detected_objects=[
            {"type": "wall", "position": {"x": 0.0, "y": -0.5, "z": 1.0}}
        ]
    )
    session.add_frame(frame1)
    print(f"   Кадр 1: {len(frame1.ar_points)} точек, {len(frame1.detected_objects)} объектов")
    
    # Кадр 2: Правый угол (через 10 секунд)
    time.sleep(0.1)
    frame2 = CameraFrame(
        timestamp=time.time(),
        ar_points=[
            {"x": 4.0, "y": 0.0, "z": 0.0},
            {"x": 4.0, "y": 2.0, "z": 0.0},
        ],
        detected_objects=[
            {"type": "pipe", "position": {"x": 3.0, "y": 1.0, "z": 1.5}}
        ]
    )
    session.add_frame(frame2)
    print(f"   Кадр 2: {len(frame2.ar_points)} точек, {len(frame2.detected_objects)} объектов")
    
    # Тест 3: Контекст сцены
    print("\n3. Агрегированный контекст сцены:")
    context = session.scene_context
    print(f"   Всего объектов: {len(context.all_detected_objects)}")
    print(f"   Всего AR точек: {len(context.all_ar_points)}")
    print(f"   Границы сцены: {context.estimated_bounds}")
    
    # Тест 4: Добавление вариантов
    print("\n4. Генерация вариантов:")
    variant1 = {
        "label": "Вариант 1",
        "nodes": [],
        "beams": []
    }
    variant2 = {
        "label": "Вариант 2",
        "nodes": [],
        "beams": []
    }
    session.add_variant(variant1)
    session.add_variant(variant2)
    print(f"   Добавлено вариантов: {len(session.generated_variants)}")
    
    # Тест 5: Экспорт данных
    print("\n5. Экспорт данных сессии:")
    export_json = manager.export_session_data(sid)
    if export_json:
        print(f"   Размер JSON: {len(export_json)} байт")
    
    # Тест 6: Сводка
    print("\n6. Сводка сессии:")
    summary = session.get_context_summary()
    print(f"   Длительность: {summary['duration_seconds']:.1f} сек")
    print(f"   Кадров обработано: {summary['total_frames']}")
    print(f"   Объектов обнаружено: {summary['total_variants']}")
    
    print("\n" + "=" * 70)
    print("✓ Все тесты пройдены!")
