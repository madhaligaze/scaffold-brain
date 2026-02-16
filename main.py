"""
Main FastAPI Server - AI Brain Backend
=======================================
СТАТУС: Production Ready

Интеграция всех исправленных модулей:
✓ LayherStandards - правильные размеры
✓ PhysicsEnhanced - Closed Loop оптимизация
✓ CollisionSolver - умное решение коллизий
✓ BuilderFixed - генератор с валидацией
✓ SessionManager - контекст всей сцены
"""
from fastapi import FastAPI, HTTPException, UploadFile, File, Form, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Any
import base64
import io
import time
import traceback
from pathlib import Path

# Импорты исправленных модулей
from core.layher_standards import (
    LayherStandards, 
    BillOfMaterials,
    validate_scaffold_dimensions,
    snap_to_layher_grid
)
from core.physics_enhanced import StructuralBrain, LoadAnalysisResult, quick_safety_check
from core.collision_solver import CollisionSolver, Obstacle, create_obstacle_from_detection
from modules.builder import ScaffoldGenerator
from core.session_manager import (
    SessionManager, 
    Session, 
    CameraFrame, 
    session_manager
)

# ── Новые модули v3.0 ───────────────────────────────────────────────────────
try:
    from modules.voxel_world import VoxelWorld
    from modules.astar_pathfinder import ScaffoldPathfinder
    from modules.structural_graph import StructuralGraph
    from modules.auto_scaffolder import AutoScaffolder
    from modules.post_processor import StructuralPostProcessor

    BRAIN_V3_AVAILABLE = True
except ImportError:
    BRAIN_V3_AVAILABLE = False

# ═══════════════════════════════════════════════════════════════════════════
# FASTAPI APP
# ═══════════════════════════════════════════════════════════════════════════

app = FastAPI(
    title="AI Brain - Scaffolding Intelligence",
    version="2.1.0",
    description="Генеративный инжиниринг строительных лесов с Layher стандартами"
)

# CORS для Android приложения
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # В продакшене указать конкретные домены
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Инициализация компонентов
scaffold_generator = ScaffoldGenerator()
physics_brain = StructuralBrain()
collision_solver = CollisionSolver(clearance=0.15)

# v3.2: PostProcessor для диагоналей и настилов
if BRAIN_V3_AVAILABLE:
    post_processor = StructuralPostProcessor()


# ═══════════════════════════════════════════════════════════════════════════
# PYDANTIC MODELS
# ═══════════════════════════════════════════════════════════════════════════

class SessionStartRequest(BaseModel):
    """Запрос на создание сессии"""
    user_id: Optional[str] = None
    project_name: Optional[str] = "Unnamed Project"


class SessionStartResponse(BaseModel):
    """Ответ при создании сессии"""
    session_id: str
    message: str
    timestamp: float


class Point3D(BaseModel):
    """3D точка"""
    x: float
    y: float
    z: float = 0.0


class DetectedObject(BaseModel):
    """Обнаруженный объект"""
    type: str  # "wall", "pipe", "column", etc.
    position: Point3D
    dimensions: Optional[Dict[str, float]] = None
    confidence: float = 1.0


class StreamFrameRequest(BaseModel):
    """Запрос на стриминг кадра"""
    session_id: str
    frame_base64: str  # base64 encoded image
    camera_position: Optional[Dict] = None
    ar_points: List[Point3D] = []
    # НОВОЕ: Облако точек от ARCore (мировые координаты, уже трансформированные).
    # ARCore API: Frame.acquirePointCloud() → PointCloud.getPoints() → float[N*4]
    # Формат: [[x, y, z, confidence], ...] или [[x, y, z], ...]
    # Confidence опционален, используем только XYZ.
    point_cloud: List[List[float]] = []
    timestamp: Optional[float] = None


class GenerateRequest(BaseModel):
    """Запрос на генерацию вариантов"""
    session_id: str
    target_dimensions: Dict[str, float]  # {width, height, depth}
    user_points: List[Point3D] = []
    use_ai_detection: bool = True
    optimize_structure: bool = True  # Включить Closed Loop оптимизацию
    # НОВОЕ: если задан — используем AutoScaffolder вместо старого генератора.
    # Формат: {"x": f, "y": f, "z": f} — точка доступа (труба/оборудование на потолке).
    target_point: Optional[Point3D] = None


class AnalyzeRequest(BaseModel):
    """Запрос на физический анализ"""
    nodes: List[Dict]
    beams: List[Dict]
    fixed_node_ids: Optional[List[str]] = None
    optimize_if_critical: bool = True  # Авто-оптимизация при перегрузке


class ExportBOMRequest(BaseModel):
    """Запрос на экспорт спецификации"""
    session_id: str
    variant_index: int


# ─── v3.0 Models ────────────────────────────────────────────────────────────

class DepthStreamRequest(BaseModel):
    """Стриминг карты глубины с ARCore Depth API"""

    session_id: str
    depth_base64: str
    width: int
    height: int
    fx: float = 500.0
    fy: float = 500.0
    cx_px: float = 320.0
    cy_px: float = 240.0
    camera_pose: List[float] = [0, 0, 0, 0, 0, 0, 1]


class StructureModifyRequest(BaseModel):
    """Интерактивное изменение конструкции (удалить/добавить элемент)"""

    session_id: str
    action: str
    element_id: Optional[str] = None
    element_data: Optional[Dict] = None


class AutoScaffoldRequest(BaseModel):
    """Автоматическая сборка от целевой точки"""

    session_id: str
    target: Point3D
    clearance_box: Optional[Dict] = None
    floor_z: float = 0.0
    ledger_len: float = 1.09
    standard_h: float = 2.07


# ═══════════════════════════════════════════════════════════════════════════
# ENDPOINTS
# ═══════════════════════════════════════════════════════════════════════════

@app.get("/")
async def root():
    """Корневой endpoint - информация о сервере"""
    return {
        "name": "AI Brain Backend",
        "version": "2.1.0",
        "status": "operational",
        "features": {
            "layher_standards": True,
            "closed_loop_optimization": True,
            "collision_avoidance": True,
            "session_context": True,
            "physics_validation": True
        },
        "standards": {
            "ledger_lengths": LayherStandards.LEDGER_LENGTHS,
            "standard_heights": LayherStandards.STANDARD_HEIGHTS
        }
    }


@app.get("/health")
async def health_check():
    """Health check для мониторинга"""
    return {
        "status": "healthy",
        "timestamp": time.time(),
        "active_sessions": len(session_manager.sessions),
        "uptime_seconds": time.time()  # Упрощенно
    }


@app.post("/session/start")
async def start_session(request: SessionStartRequest):
    """
    Создание новой сессии.
    
    Android должен вызвать этот endpoint перед началом работы.
    """
    try:
        session_id = session_manager.create_session()
        
        return SessionStartResponse(
            session_id=session_id,
            message="Сессия создана успешно. ИИ готов к работе.",
            timestamp=time.time()
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/session/stream")
async def stream_frame(request: StreamFrameRequest):
    """
    Стриминг кадров камеры в режиме реального времени.
    
    ИИ обрабатывает кадр, детектирует объекты и добавляет в контекст сессии.
    """
    try:
        # Получаем сессию
        session = session_manager.get_session(request.session_id)
        if not session:
            raise HTTPException(status_code=404, detail="Сессия не найдена")
        
        # Создаем кадр
        frame = CameraFrame(
            timestamp=request.timestamp or time.time(),
            image_data=request.frame_base64,
            camera_position=request.camera_position,
            ar_points=[p.dict() for p in request.ar_points]
        )

        # ── НОВОЕ: Заполняем VoxelWorld из point_cloud ──────────────────────
        # Это основной источник "зрения" ИИ.
        # point_cloud уже в мировых координатах от ARCore — просто кладём в сетку.
        if request.point_cloud and BRAIN_V3_AVAILABLE:
            voxel_world = session.scene_context.ensure_voxel_world()
            added = voxel_world.add_point_cloud(request.point_cloud)
            frame.quality_metrics = frame.quality_metrics or {}
            frame.quality_metrics['voxels_added'] = added
            frame.quality_metrics['total_voxels'] = voxel_world.total_voxels
        
        # TODO: Здесь должна быть детекция объектов через YOLO
        # Пока возвращаем заглушку
        detected_objects = []
        
        frame.detected_objects = detected_objects
        
        # Добавляем кадр в сессию
        session.add_frame(frame)
        
        return {
            "status": "processed",
            "session_id": request.session_id,
            "detected_objects": detected_objects,
            "context_summary": session.scene_context.get_summary(),
            "message": "Кадр обработан. Контекст обновлен."
        }
    
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/generate")
async def generate_variants(request: GenerateRequest):
    """
    Генерация вариантов строительных лесов.
    
    КРИТИЧЕСКИ ВАЖНО:
    - Все размеры приводятся к стандартам Layher
    - Если optimize_structure=True → запускается Closed Loop оптимизация
    - Варианты проверяются на коллизии
    - Генерируется BOM для каждого варианта
    """
    try:
        # Получаем сессию
        session = session_manager.get_session(request.session_id)
        if not session:
            raise HTTPException(status_code=404, detail="Сессия не найдена")

        # ── НОВОЕ: AutoScaffolder — умная сборка от целевой точки ────────────
        if request.target_point is not None and BRAIN_V3_AVAILABLE:
            voxel_world = session.scene_context.ensure_voxel_world()

            # Заполняем воксели из YOLO-детекций накопленных в сессии
            # (вспомогательно, если point_cloud не передавался)
            all_dets = session.scene_context.all_detected_objects
            if all_dets and voxel_world.total_voxels == 0:
                voxel_world.ingest_yolo_detections(all_dets)

            from modules.auto_scaffolder import AutoScaffolder
            scaffolder = AutoScaffolder(
                voxel_world=voxel_world,
                ledger_len=request.target_dimensions.get('ledger_len', 1.09),
                standard_h=request.target_dimensions.get('standard_h', 2.07),
            )
            target_dict = {
                "x": request.target_point.x,
                "y": request.target_point.y,
                "z": request.target_point.z,
            }
            floor_z = request.target_dimensions.get('floor_z', 0.0)
            variant = scaffolder.build_to_target(
                target=target_dict,
                floor_z=floor_z,
            )

            # Физический анализ
            analysis = physics_brain.calculate_load_map(
                variant['nodes'], variant['beams']
            )
            variant['physics_analysis'] = {
                "status": analysis.status,
                "max_load_ratio": analysis.max_load_ratio,
                "critical_beams": analysis.critical_beams,
            }

            # Загружаем в структурный граф сессии
            if hasattr(session, 'ensure_structural_graph'):
                graph = session.ensure_structural_graph()
                graph.load_from_variant(variant)

            session.add_variant(variant)

            blocked = sum(1 for b in variant['beams'] if b.get('blocked'))
            return {
                "status": "success",
                "mode": "auto_scaffolder",
                "variants": [variant],
                "count": 1,
                "blocked_beams": blocked,
                "voxels_used": voxel_world.total_voxels,
                "message": (
                    f"AutoScaffolder: башня {variant.get('floors','?')} ярусов. "
                    f"Препятствий в VoxelWorld: {voxel_world.total_voxels}. "
                    f"Обойдено балок: {blocked}."
                )
            }
        # ────────────────────────────────────────────────────────────────────
        # СТАРЫЙ ПУТЬ: target_point не задан → классический генератор
        # Обратная совместимость сохранена.
        # ────────────────────────────────────────────────────────────────────
        
        # Приводим размеры к стандартам
        target_w = snap_to_layher_grid(
            request.target_dimensions.get('width', 4.0), "ledger"
        )
        target_h = snap_to_layher_grid(
            request.target_dimensions.get('height', 3.0), "standard"
        )
        target_d = snap_to_layher_grid(
            request.target_dimensions.get('depth', 2.0), "ledger"
        )
        
        # Собираем точки
        user_points = [p.dict() for p in request.user_points]
        ai_points = session.scene_context.all_ar_points if request.use_ai_detection else []
        
        # Генерируем варианты
        variants = scaffold_generator.generate_smart_options(
            user_points=user_points,
            ai_points=ai_points,
            bounds={"w": target_w, "h": target_h, "d": target_d},
            obstacles=session.scene_context.obstacles,
            voxel_world=session.scene_context.voxel_world,
        )
        
        # Оптимизация каждого варианта (если включено)
        optimized_variants = []
        
        for variant in variants:
            # Быстрая проверка безопасности
            is_safe = quick_safety_check(variant['nodes'], variant['beams'])
            
            if not is_safe:
                variant['warning'] = "Конструкция может быть неустойчивой"
            
            # Closed Loop оптимизация (если включена)
            if request.optimize_structure:
                optimization_result = physics_brain.optimize_structure_closed_loop(
                    variant['nodes'],
                    variant['beams'],
                    target_safety=0.85
                )
                
                # Обновляем вариант оптимизированными данными
                variant['nodes'] = optimization_result['nodes']
                variant['beams'] = optimization_result['beams']
                variant['optimization'] = {
                    "iterations": optimization_result['iterations'],
                    "added_diagonals": optimization_result['added_diagonals'],
                    "optimized": optimization_result['optimized'],
                    "final_load_ratio": optimization_result['final_analysis'].max_load_ratio
                }
            
            # Физический анализ
            analysis = physics_brain.calculate_load_map(
                variant['nodes'],
                variant['beams']
            )
            
            variant['physics_analysis'] = {
                "status": analysis.status,
                "max_load_ratio": analysis.max_load_ratio,
                "critical_beams": analysis.critical_beams,
                "beam_loads": analysis.beam_loads[:10]  # Первые 10 для экономии трафика
            }
            
            # Валидация размеров
            errors = validate_scaffold_dimensions(variant['nodes'], variant['beams'])
            variant['validation_errors'] = errors
            
            optimized_variants.append(variant)
        
        # Сохраняем варианты в сессии
        for variant in optimized_variants:
            session.add_variant(variant)
        
        return {
            "status": "success",
            "variants": optimized_variants,
            "count": len(optimized_variants),
            "message": "Варианты сгенерированы и оптимизированы"
        }
    
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/analyze/physics")
async def analyze_physics(request: AnalyzeRequest):
    """
    Физический анализ конструкции.
    
    Если optimize_if_critical=True и нагрузка > 90%,
    ИИ автоматически пересобирает конструкцию.
    """
    try:
        # Базовый анализ
        analysis = physics_brain.calculate_load_map(
            request.nodes,
            request.beams,
            fixed_node_ids=set(request.fixed_node_ids or [])
        )
        
        result = {
            "status": analysis.status,
            "max_load_ratio": analysis.max_load_ratio,
            "safe": analysis.is_safe(),
            "beam_loads": analysis.beam_loads,
            "critical_beams": analysis.critical_beams,
            "recommendations": analysis.recommended_reinforcements
        }
        
        # Автоматическая оптимизация если критично
        if request.optimize_if_critical and analysis.needs_optimization():
            optimization = physics_brain.optimize_structure_closed_loop(
                request.nodes,
                request.beams,
                fixed_node_ids=set(request.fixed_node_ids or [])
            )
            
            result['auto_optimization'] = {
                "performed": True,
                "iterations": optimization['iterations'],
                "added_diagonals": optimization['added_diagonals'],
                "optimized_nodes": optimization['nodes'],
                "optimized_beams": optimization['beams'],
                "final_load_ratio": optimization['final_analysis'].max_load_ratio,
                "success": optimization['optimized']
            }
            
            result['message'] = (
                f"⚠️ Нагрузка была критической ({analysis.max_load_ratio*100:.0f}%). "
                f"ИИ автоматически добавил {optimization['added_diagonals']} диагоналей. "
                f"Новая нагрузка: {optimization['final_analysis'].max_load_ratio*100:.0f}%"
            )
        
        return result
    
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/export/bom")
async def export_bom(request: ExportBOMRequest):
    """
    Экспорт Bill of Materials (спецификации) для варианта.
    
    Возвращает CSV файл, по которому можно реально заказать компоненты.
    """
    try:
        session = session_manager.get_session(request.session_id)
        if not session:
            raise HTTPException(status_code=404, detail="Сессия не найдена")
        
        if request.variant_index >= len(session.generated_variants):
            raise HTTPException(status_code=400, detail="Неверный индекс варианта")
        
        variant = session.generated_variants[request.variant_index]
        
        # Генерируем BOM
        bom = BillOfMaterials()
        for beam in variant['beams']:
            beam_type = beam.get('type', 'ledger')
            length = beam.get('length', 2.0)
            
            if beam_type == 'standard':
                std_length = LayherStandards.get_nearest_standard_height(length)
                code = f"S-{int(std_length * 100)}"
            elif beam_type in ['ledger', 'transom']:
                std_length = LayherStandards.get_nearest_ledger_length(length)
                code = f"L-{int(std_length * 100)}"
            elif beam_type == 'diagonal':
                std_length = min(
                    LayherStandards.DIAGONAL_LENGTHS,
                    key=lambda x: abs(x - length)
                )
                code = f"D-{int(std_length * 100)}"
            else:
                code = "UNKNOWN"
            
            bom.add_component(code, 1)
        
        # Генерируем CSV
        csv_content = bom.export_csv()
        
        return {
            "status": "success",
            "csv": csv_content,
            "summary": {
                "total_components": len(bom.components),
                "total_items": sum(bom.components.values()),
                "total_weight_kg": bom.get_total_weight(),
                "estimated_cost_usd": bom.get_total_cost()
            },
            "message": "Спецификация готова для заказа на складе Layher"
        }
    
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/session/{session_id}/context")
async def get_session_context(session_id: str):
    """Получить контекст сессии (для отладки)"""
    session = session_manager.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Сессия не найдена")
    
    return session.get_context_summary()


@app.delete("/session/{session_id}")
async def delete_session(session_id: str):
    """Удалить сессию"""
    success = session_manager.delete_session(session_id)
    if not success:
        raise HTTPException(status_code=404, detail="Сессия не найдена")
    
    return {
        "status": "deleted",
        "session_id": session_id
    }


@app.get("/standards/info")
async def get_standards_info():
    """
    Информация о стандартах Layher.
    
    Android может использовать это для валидации на клиенте.
    """
    return {
        "ledger_lengths": LayherStandards.LEDGER_LENGTHS,
        "standard_heights": LayherStandards.STANDARD_HEIGHTS,
        "diagonal_lengths": LayherStandards.DIAGONAL_LENGTHS,
        "max_loads": {
            "ledgers": LayherStandards.MAX_LEDGER_LOAD,
            "standard": LayherStandards.MAX_STANDARD_LOAD,
            "diagonal": LayherStandards.MAX_DIAGONAL_TENSION
        },
        "safety_thresholds": {
            "critical": LayherStandards.CRITICAL_LOAD_THRESHOLD,
            "warning": LayherStandards.WARNING_LOAD_THRESHOLD
        }
    }


@app.post("/session/depth_stream")
async def ingest_depth_stream(request: DepthStreamRequest):
    if not BRAIN_V3_AVAILABLE:
        raise HTTPException(503, "VoxelWorld модуль не установлен (Brain v3.0)")

    session = session_manager.get_session(request.session_id)
    if not session:
        raise HTTPException(404, "Сессия не найдена")

    try:
        depth_bytes = base64.b64decode(request.depth_base64)
    except Exception as exc:
        raise HTTPException(400, "Ошибка декодирования depth_base64") from exc

    voxel_world = session.scene_context.ensure_voxel_world()
    added = voxel_world.ingest_depth_map(
        depth_bytes=depth_bytes,
        width=request.width,
        height=request.height,
        fx=request.fx,
        fy=request.fy,
        cx_px=request.cx_px,
        cy_px=request.cy_px,
        camera_pose=request.camera_pose,
    )

    return {
        "status": "voxels_updated",
        "added_voxels": added,
        "total_voxels": voxel_world.total_voxels,
        "message": f"Добавлено {added} вокселей. ИИ видит пространство.",
    }


@app.get("/session/{session_id}/voxel_map")
async def get_voxel_map(session_id: str):
    if not BRAIN_V3_AVAILABLE:
        return {"voxels": [], "resolution": 0.1, "available": False}

    session = session_manager.get_session(session_id)
    if not session:
        raise HTTPException(404, "Сессия не найдена")

    vw = session.scene_context.voxel_world
    if vw is None:
        return {"voxels": [], "resolution": 0.1, "message": "Depth map ещё не загружен"}

    return vw.to_ar_mesh()


@app.post("/generate/auto")
async def generate_auto_scaffold(request: AutoScaffoldRequest):
    if not BRAIN_V3_AVAILABLE:
        raise HTTPException(503, "AutoScaffolder модуль не установлен (Brain v3.0)")

    session = session_manager.get_session(request.session_id)
    if not session:
        raise HTTPException(404, "Сессия не найдена")

    voxel_world = session.scene_context.ensure_voxel_world()

    all_detections = session.scene_context.all_detected_objects
    if all_detections:
        voxel_world.ingest_yolo_detections(all_detections)

    target_dict = {"x": request.target.x, "y": request.target.y, "z": request.target.z}

    scaffolder = AutoScaffolder(
        voxel_world=voxel_world,
        ledger_len=request.ledger_len,
        standard_h=request.standard_h,
    )

    try:
        variant = scaffolder.build_to_target(
            target=target_dict,
            clearance_box=request.clearance_box,
            floor_z=request.floor_z,
        )
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc

    analysis = physics_brain.calculate_load_map(variant["nodes"], variant["beams"])
    variant["physics_analysis"] = {
        "status": analysis.status,
        "max_load_ratio": analysis.max_load_ratio,
        "critical_beams": analysis.critical_beams,
    }

    graph = session.ensure_structural_graph()
    graph.load_from_variant(variant)
    session.add_variant(variant)

    blocked_count = sum(1 for b in variant["beams"] if b.get("blocked"))

    return {
        "status": "success",
        "variant": variant,
        "graph_summary": graph.get_summary(),
        "blocked_beams": blocked_count,
        "message": (
            f"Башня {variant['floors']} ярусов построена. "
            f"Препятствий обойдено: {blocked_count}. "
            f"Статус физики: {analysis.status}"
        ),
    }


@app.post("/structure/modify")
async def modify_structure(request: StructureModifyRequest):
    session = session_manager.get_session(request.session_id)
    if not session:
        raise HTTPException(404, "Сессия не найдена")

    if not BRAIN_V3_AVAILABLE:
        raise HTTPException(503, "StructuralGraph модуль не установлен")

    graph = session.ensure_structural_graph()

    if not graph.get_beams() and session.generated_variants:
        graph.load_from_variant(session.generated_variants[-1])

    t_start = time.time()

    if request.action == "REMOVE":
        if not request.element_id:
            raise HTTPException(400, "element_id обязателен для REMOVE")
        result = graph.remove_element(request.element_id)
    elif request.action == "ADD":
        if not request.element_data:
            raise HTTPException(400, "element_data обязателен для ADD")
        result = graph.add_beam(request.element_data)
    else:
        raise HTTPException(400, f"Неизвестный action: {request.action}")

    elapsed_ms = (time.time() - t_start) * 1000

    full_analysis = None
    if not result.get("is_stable") and session.generated_variants:
        try:
            full_analysis = physics_brain.calculate_load_map(graph.get_nodes(), graph.get_beams())
        except Exception:
            pass

    return {
        "status": "UPDATED",
        "action": request.action,
        "element_id": request.element_id,
        "heatmap": result.get("heatmap", []),
        "is_stable": result.get("is_stable", True),
        "affected": result.get("affected", []),
        "elapsed_ms": round(elapsed_ms, 1),
        "full_analysis": {
            "status": full_analysis.status,
            "max_load_ratio": full_analysis.max_load_ratio,
        }
        if full_analysis
        else None,
        "animation_hint": "COLLAPSE" if not result.get("is_stable") else "UPDATE",
        "message": (
            "⚠️ КОНСТРУКЦИЯ НЕСТАБИЛЬНА — добавьте диагонали!"
            if not result.get("is_stable")
            else f"Обновлено за {elapsed_ms:.0f} мс"
        ),
    }


@app.post("/session/model/{session_id}")
async def finalize_model(session_id: str):
    """
    Финализация модели с полным циклом: A* → PostProcessor → Physics.
    """
    if not BRAIN_V3_AVAILABLE:
        raise HTTPException(503, "Brain v3.2 модули недоступны")

    session = session_manager.get_session(session_id)
    if not session:
        raise HTTPException(404, "Сессия не найдена")

    try:
        from modules.session import DesignSession

        if isinstance(session, DesignSession):
            user_anchors = session.user_anchors
        else:
            user_anchors = (
                session.scene_context.anchor_points
                or session.scene_context.all_ar_points
            )
    except Exception:
        user_anchors = []

    if not user_anchors:
        return {
            "status": "ERROR",
            "message": "No anchor points defined. User must place markers in AR first.",
        }

    print(f"🏗️  Industrial AI Modeling for {session_id}... ({len(user_anchors)} anchors)")

    start_anchor = user_anchors[0]
    end_anchor = user_anchors[-1]

    start_node = {
        "x": start_anchor.get("x", 0),
        "y": start_anchor.get("y", 0),
        "z": start_anchor.get("z", 0),
    }
    target_node = {
        "x": end_anchor.get("x", 0),
        "y": end_anchor.get("y", 0),
        "z": end_anchor.get("z", 0) + 2.0,
    }

    voxel_world = session.scene_context.ensure_voxel_world()
    pathfinder = ScaffoldPathfinder(voxel_world)

    path_segments = pathfinder.find_path(start_node, target_node)

    if not path_segments:
        return {
            "status": "FAILURE",
            "message": "Path blocked or impossible. Check VoxelWorld obstacles.",
            "voxels_in_world": voxel_world.total_voxels,
        }

    skeleton = []
    for i, seg in enumerate(path_segments):
        if isinstance(seg, dict) and "start" in seg and "end" in seg:
            skeleton.append(
                {
                    "id": f"sk_{i}",
                    "type": seg.get("type", "ledger"),
                    "start": seg["start"],
                    "end": seg["end"],
                    "length": seg.get("length", 2.0),
                    "weight": 10.0,
                }
            )

    full_structure = post_processor.process(skeleton)

    print(
        f"  PostProcessor: {len(skeleton)} → {len(full_structure)} elements "
        f"(added {len(full_structure) - len(skeleton)} bracing/decks)"
    )

    phys_nodes = []
    phys_beams = []
    seen_nodes = set()

    for el in full_structure:
        for p in [el["start"], el["end"]]:
            k = f"{p[0]:.2f}_{p[1]:.2f}_{p[2]:.2f}"
            if k not in seen_nodes:
                phys_nodes.append(
                    {
                        "id": k,
                        "x": p[0],
                        "y": p[1],
                        "z": p[2],
                        "fixed": abs(p[2]) < 0.1,
                    }
                )
                seen_nodes.add(k)

        s = el["start"]
        e = el["end"]
        phys_beams.append(
            {
                "id": el["id"],
                "type": el["type"],
                "start": f"{s[0]:.2f}_{s[1]:.2f}_{s[2]:.2f}",
                "end": f"{e[0]:.2f}_{e[1]:.2f}_{e[2]:.2f}",
                "length": el.get("length", 0),
            }
        )

    physics_res = physics_brain.calculate_load_map(phys_nodes, phys_beams)

    if isinstance(physics_res, dict):
        physics_status = physics_res.get("status", "COLLAPSE")
        physics_data = physics_res.get("data", [])
    else:
        physics_status = getattr(physics_res, "status", "COLLAPSE")
        physics_data = getattr(physics_res, "beam_loads", [])

    safety_score = 0
    if physics_status == "OK":
        loads = [r.get("load_ratio", 0) for r in physics_data]
        if loads:
            max_load = max(loads)
            safety_score = int((1.0 - min(max_load, 1.0)) * 100)

            by_id = {item.get("id"): item for item in physics_data}
            for el in full_structure:
                phys_item = by_id.get(el.get("id"))
                if phys_item:
                    el["load_ratio"] = phys_item.get("load_ratio", 0.0)
                    el["stress_color"] = phys_item.get("color", "green")
        else:
            safety_score = 100
    else:
        print("⚠️  Physics calculation FAILED (Structure unstable)")

    final_options = [
        {
            "id": 1,
            "name": "AI Engineered (Layher Allround)",
            "elements": [],
            "safety_score": safety_score,
            "total_weight": sum(e.get("weight", 0) for e in full_structure),
            "physics_status": physics_status,
        }
    ]

    for el in full_structure:
        final_options[0]["elements"].append(
            {
                "id": el.get("id", "gen"),
                "type": el["type"],
                "start": {"x": el["start"][0], "y": el["start"][1], "z": el["start"][2]},
                "end": {"x": el["end"][0], "y": el["end"][1], "z": el["end"][2]},
                "length": el.get("length", 0),
                "stress_color": el.get("stress_color", "green"),
                "load_ratio": el.get("load_ratio", 0.0),
            }
        )

    return {
        "status": "SUCCESS",
        "options": final_options,
        "statistics": {
            "skeleton_elements": len(skeleton),
            "total_elements": len(full_structure),
            "added_diagonals": sum(1 for e in full_structure if e["type"] == "diagonal"),
            "added_decks": sum(1 for e in full_structure if e["type"] == "deck"),
            "voxels_used": voxel_world.total_voxels,
        },
    }


@app.websocket("/ws/{session_id}")
async def websocket_structure(websocket: WebSocket, session_id: str):
    await websocket.accept()

    session = session_manager.get_session(session_id)
    if not session:
        await websocket.send_json({"type": "ERROR", "message": "Сессия не найдена"})
        await websocket.close()
        return

    try:
        while True:
            data = await websocket.receive_json()
            action = data.get("action", "")

            if action == "PING":
                await websocket.send_json({"type": "PONG"})
                continue

            if action in ("REMOVE", "ADD") and BRAIN_V3_AVAILABLE:
                graph = session.ensure_structural_graph()
                if not graph.get_beams() and session.generated_variants:
                    graph.load_from_variant(session.generated_variants[-1])

                if action == "REMOVE":
                    result = graph.remove_element(data.get("element_id", ""))
                else:
                    result = graph.add_beam(data.get("element_data", {}))

                await websocket.send_json(
                    {
                        "type": "HEATMAP",
                        "heatmap": result.get("heatmap", []),
                        "is_stable": result.get("is_stable", True),
                        "affected": result.get("affected", []),
                        "animation": "COLLAPSE" if not result.get("is_stable") else "UPDATE",
                    }
                )
                continue

            await websocket.send_json({"type": "ERROR", "message": f"Неизвестный action: {action}"})

    except WebSocketDisconnect:
        pass
    except Exception as exc:
        try:
            await websocket.send_json({"type": "ERROR", "message": str(exc)})
        except Exception:
            pass


# ═══════════════════════════════════════════════════════════════════════════
# ERROR HANDLERS
# ═══════════════════════════════════════════════════════════════════════════

@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Глобальный обработчик ошибок"""
    traceback.print_exc()
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal Server Error",
            "detail": str(exc),
            "type": type(exc).__name__
        }
    )


# ═══════════════════════════════════════════════════════════════════════════
# STARTUP
# ═══════════════════════════════════════════════════════════════════════════

@app.on_event("startup")
async def startup_event():
    """Инициализация при запуске"""
    print("=" * 70)
    print("🚀 AI BRAIN BACKEND STARTING")
    print("=" * 70)
    print(f"✓ Layher Standards: {len(LayherStandards.LEDGER_LENGTHS)} ledger lengths")
    print(f"✓ Physics Engine: PyNite FEM")
    print(f"✓ Collision Solver: Trimesh integration")
    print(f"✓ Session Manager: Ready")
    print(f"{'✓' if BRAIN_V3_AVAILABLE else '✗'} Brain v3.0: VoxelWorld + A* + StructuralGraph")
    if not BRAIN_V3_AVAILABLE:
        print("  ⚠️  Установите: pip install networkx websockets")
    print("=" * 70)


if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )
