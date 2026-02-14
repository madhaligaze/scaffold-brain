# main.py
"""
Bauflex AI Brain - Серверный "мозг" для Engineering Intelligence системы.

Интегрирует:
- Computer Vision (YOLO)
- Computational Physics (FEA)
- Dynamic Loads (Wind, Vibration)
- Multi-view Photogrammetry
- Expert System (Safety Rules)
"""
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Body
from pydantic import BaseModel
from typing import Dict, List, Optional, Any
import uuid
from modules.vision import Eyes, SceneDiagnostician, VisionSystem
from modules.physics import StructuralBrain
from modules.builder import ScaffoldExpert, ScaffoldGenerator
from modules.dynamics import DynamicLoadAnalyzer, ProgressiveCollapseAnalyzer
from modules.photogrammetry import PhotogrammetrySystem
from modules.session import DesignSession
from modules.geometry import WorldGeometry
from modules.storage import SessionStorage

app = FastAPI(
    title="Bauflex AI Brain",
    description="Engineering Intelligence для строительных лесов",
    version="2.0.0"
)

# === ИНИЦИАЛИЗАЦИЯ МОДУЛЕЙ ===

eyes = Eyes()
diagnostician = SceneDiagnostician()
vision_system = VisionSystem()
engineer = StructuralBrain()
expert = ScaffoldExpert()
wind_analyzer = DynamicLoadAnalyzer()
photogrammetry = PhotogrammetrySystem()

# Ленивая инициализация для collapse analyzer (требует physics engine)
collapse_analyzer = None
active_sessions: Dict[str, DesignSession] = {}
generator = ScaffoldGenerator()
geometry = WorldGeometry(padding=3.0)
session_storage = SessionStorage()

def get_collapse_analyzer():
    global collapse_analyzer
    if collapse_analyzer is None:
        collapse_analyzer = ProgressiveCollapseAnalyzer(engineer)
    return collapse_analyzer


def get_or_restore_session(session_id: str) -> DesignSession | None:
    session = active_sessions.get(session_id)
    if session is not None:
        return session

    restored = session_storage.load(session_id, vision_system=vision_system)
    if restored is not None:
        active_sessions[session_id] = restored
    return restored


# === МОДЕЛИ ДАННЫХ ===

class Node(BaseModel):
    id: str
    x: float
    y: float
    z: float

class Beam(BaseModel):
    id: str
    start: str
    end: str

class StructureData(BaseModel):
    nodes: List[Node]
    beams: List[Beam]
    fixed_node_ids: Optional[List[str]] = None  # Навесные опоры

class VibrationSource(BaseModel):
    x: float
    y: float
    z: float
    frequency_hz: float = 25.0
    amplitude_m: float = 0.002
    type: str = "conveyor"



class VibrationAnalysisRequest(StructureData):
    vibration_source: VibrationSource


class Annotation(BaseModel):
    """Подписи размеров на модели."""

    text: str
    position: Dict[str, float]


class VisualElement(BaseModel):
    """Описание одной детали для отрисовки в AR."""

    id: str
    type: str
    mesh_type: str
    start: Dict[str, float]
    end: Dict[str, float]
    stress_color: str
    load_value: float


class ProposalVariant(BaseModel):
    """Один вариант предложения лесов."""

    variant_id: int
    name: str
    description: str
    elements: List[VisualElement]
    annotations: List[Annotation]
    total_weight: float
    safety_rating: float

# === SESSION PROTOCOL (Bauflex Session Protocol) ===

@app.post("/session/start")
async def start_session():
    """Создает новую сессию замера и включает сбор keyframes."""
    sid = str(uuid.uuid4())
    active_sessions[sid] = DesignSession(session_id=sid, vision_system=vision_system)
    session_storage.save(active_sessions[sid])
    return {"session_id": sid, "status": "MEASURING"}


@app.post("/session/stream/{session_id}")
async def stream_session_data(session_id: str, data: Dict[str, Any] = Body(...)):
    """Принимает потоковые данные Android: image/pose/markers."""
    session = get_or_restore_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    if session.status != "MEASURING":
        raise HTTPException(status_code=409, detail="Session is not in measuring state")

    try:
        image_payload = data.get("image", b"")
        if isinstance(image_payload, str):
            import base64
            image_bytes = base64.b64decode(image_payload)
        else:
            image_bytes = bytes(image_payload)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid image payload: {exc}")

    feedback = session.update_world_model(
        image_bytes=image_bytes,
        pose_matrix=data.get("pose", []),
        markers=data.get("markers", []),
    )
    session_storage.save(session)
    return {"status": "RECEIVING", "ai_hints": feedback}


@app.post("/session/model/{session_id}")
async def session_model(session_id: str):
    """Финализирует сессию, выполняет геометрию+физику и возвращает AR-ready формат."""
    session = get_or_restore_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    session.status = "MODELING"
    session_storage.save(session)

    # 1) ROI фильтр для пользовательских/AI-точек
    all_points = list(session.user_anchors) + list(session.detected_supports)
    roi_min, roi_max = geometry.create_roi_filter(all_points)
    scoped_user = [
        p for p in session.user_anchors
        if roi_min[0] <= float(p.get("x", 0.0)) <= roi_max[0]
        and roi_min[1] <= float(p.get("y", 0.0)) <= roi_max[1]
        and roi_min[2] <= float(p.get("z", 0.0)) <= roi_max[2]
    ]
    scoped_ai = [
        p for p in session.detected_supports
        if roi_min[0] <= float(p.get("x", 0.0)) <= roi_max[0]
        and roi_min[1] <= float(p.get("y", 0.0)) <= roi_max[1]
        and roi_min[2] <= float(p.get("z", 0.0)) <= roi_max[2]
    ]

    # 2) Обновляем препятствия из AI (pipe-like объекты в supports)
    geometry.obstacles.clear()
    geometry.scene = geometry.scene.__class__()
    for support in scoped_ai:
        if str(support.get("type", "")).upper() not in {"PIPE", "AI_PIPE"}:
            continue
        start = support.get("start")
        end = support.get("end")
        radius = float(support.get("radius", 0.08))
        if isinstance(start, dict) and isinstance(end, dict):
            geometry.add_obstacle(
                "pipe",
                [float(start.get("x", 0.0)), float(start.get("y", 0.0)), float(start.get("z", 0.0))],
                [float(end.get("x", 0.0)), float(end.get("y", 0.0)), float(end.get("z", 0.0))],
                radius,
            )

    raw_options = generator.generate_smart_options(
        user_points=scoped_user,
        ai_points=scoped_ai,
        bounds=session.get_bounds(),
    )

    final_options: List[ProposalVariant] = []

    for idx, option in enumerate(raw_options, start=1):
        node_map = {n["id"]: n for n in option.get("nodes", [])}

        def beams_with_coords() -> List[Dict[str, Any]]:
            resolved = []
            for b in option.get("beams", []):
                start_node = node_map.get(b.get("start"))
                end_node = node_map.get(b.get("end"))
                if not start_node or not end_node:
                    continue
                resolved.append({
                    "id": b.get("id", f"beam_{len(resolved)}"),
                    "type": b.get("type", "ledger"),
                    "start": {"x": start_node["x"], "y": start_node["y"], "z": start_node["z"]},
                    "end": {"x": end_node["x"], "y": end_node["y"], "z": end_node["z"]},
                })
            return resolved

        # 3) Цикл самокритики: до 3 итераций коррекции коллизий
        attempts = 0
        elements = beams_with_coords()
        collisions = geometry.check_collisions(elements)
        while collisions and attempts < 3:
            for c in collisions:
                beam_id = c.get("beam_id")
                beam = next((e for e in elements if e["id"] == beam_id), None)
                if not beam:
                    continue
                # Простая автокоррекция: смещаем проблемную балку по Y на 0.15м
                beam["start"]["y"] += 0.15
                beam["end"]["y"] += 0.15
            collisions = geometry.check_collisions(elements)
            attempts += 1

        physics_result = engineer.calculate_load_map(option.get("nodes", []), option.get("beams", []))
        physics_rows = physics_result.get("data", [])
        beam_colors = {item.get("id"): item for item in physics_rows}

        visual_elements: List[VisualElement] = []
        annotations: List[Annotation] = []

        for el in elements:
            beam_physics = beam_colors.get(el["id"], {})
            load_ratio = float(beam_physics.get("utilization", beam_physics.get("load_ratio", 0.0)))
            if load_ratio < 0.6:
                color = "green"
            elif load_ratio < 0.85:
                color = "yellow"
            else:
                color = "red"

            visual_elements.append(
                VisualElement(
                    id=el["id"],
                    type=str(el.get("type", "ledger")),
                    mesh_type="layher_allround_beam",
                    start=el["start"],
                    end=el["end"],
                    stress_color=color,
                    load_value=round(load_ratio * 100.0, 2),
                )
            )

            length = geometry.get_distance(el["start"], el["end"])
            layher_size = geometry.align_to_layher(length)
            annotations.append(
                Annotation(
                    text=f"Ригель {layher_size}м",
                    position=geometry.get_midpoint(el["start"], el["end"]),
                )
            )

        final_options.append(
            ProposalVariant(
                variant_id=idx,
                name=option.get("variant_name", f"Вариант {idx}"),
                description=f"Стратегия: {option.get('strategy', 'DEFAULT')}. Коллизий после аудита: {len(collisions)}",
                elements=visual_elements,
                annotations=annotations,
                total_weight=float(option.get("stats", {}).get("total_weight_kg", 0.0)),
                safety_rating=float(max(0.0, 100.0 - max([float(r.get("load_ratio", 0.0)) for r in physics_rows], default=0.0) * 100.0)),
            )
        )

    session_storage.save(session)
    return {
        "status": "SUCCESS",
        "roi": {"min": roi_min.tolist(), "max": roi_max.tolist()},
        "options": [o.model_dump() for o in final_options],
    }


# === БАЗОВЫЕ ЭНДПОИНТЫ (Computer Vision) ===

@app.post("/analyze/photo")
async def analyze_photo(
    file: UploadFile = File(...),
    distance: float = Form(...),
    focal_length: float = Form(800)
):
    """
    Анализ фото: детекция объектов + оценка реальных размеров.
    
    Args:
        file: изображение
        distance: расстояние до объекта (м), от ARCore
        focal_length: фокусное расстояние камеры (пиксели)
    """
    contents = await file.read()
    
    try:
        detected = eyes.analyze_scene(contents, distance, focal_length)
        return {
            "status": "OK",
            "objects": detected,
            "count": len(detected)
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/analyze/quality-check")
async def quality_check(
    file: UploadFile = File(...),
    distance: float = Form(...),
    ar_points: str = Form("[]")  # JSON string
):
    """
    Проверка качества данных для проектирования.
    Возвращает инструкции, если фото недостаточно хорошее.
    
    Args:
        file: изображение
        distance: расстояние
        ar_points: JSON-массив AR-точек
    """
    import json
    contents = await file.read()
    
    try:
        ar_points_list = json.loads(ar_points)
    except:
        ar_points_list = []
    
    try:
        frame = eyes._decode_image_bgr(contents)
        detected = eyes.analyze_scene(distance_to_target=distance, frame=frame)
        quality = diagnostician.check_data_quality(
            frame, detected, ar_points_list, distance
        )
        
        return {
            "status": "OK",
            "quality": quality,
            "can_proceed": quality["is_ready"]
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


# === ФИЗИКА И РАСЧЕТ НАГРУЗОК ===

@app.post("/engineer/calculate")
async def calculate_structure(data: StructureData):
    """
    Полный статический расчет нагрузок.
    Цветовая карта напряжений: green/yellow/red.
    """
    nodes_dict = [n.dict() for n in data.nodes]
    beams_dict = [b.dict() for b in data.beams]
    
    result = engineer.calculate_load_map(
        nodes_dict, 
        beams_dict, 
        fixed_node_ids=data.fixed_node_ids
    )
    return result


@app.post("/engineer/simulate-removal")
async def simulate_removal(data: StructureData, remove_id: str):
    """
    AI-ПРЕДСКАЗАТЕЛЬ: что будет, если удалить эту балку?
    """
    nodes_dict = [n.dict() for n in data.nodes]
    beams_dict = [b.dict() for b in data.beams]
    
    prediction = engineer.simulate_removal(
        nodes_dict, 
        beams_dict, 
        remove_id,
        fixed_node_ids=data.fixed_node_ids
    )
    return prediction


# === ДИНАМИЧЕСКИЕ НАГРУЗКИ (НОВИНКА!) ===

@app.post("/dynamics/wind-analysis")
async def wind_analysis(
    data: StructureData,
    wind_speed: float = 20.0,
    wind_direction: str = "X"
):
    """
    Анализ ветровой нагрузки.
    КРИТИЧНО для высотных лесов (>10м).
    
    Args:
        wind_speed: скорость ветра (м/с), 20 м/с ≈ 72 км/ч
        wind_direction: "X", "Y", или "XY"
    """
    nodes_dict = [n.dict() for n in data.nodes]
    beams_dict = [b.dict() for b in data.beams]
    
    result = wind_analyzer.calculate_wind_load(
        nodes_dict, beams_dict, wind_speed, wind_direction
    )
    return result


@app.post("/dynamics/vibration-analysis")
async def vibration_analysis(data: VibrationAnalysisRequest):
    """
    Анализ вибрации от оборудования (конвейер, станок).
    Проверяет риск резонанса и предлагает решения.
    
    ВАЖНО: Вибрация конвейера может привести к резонансу!
    """
    nodes_dict = [n.dict() for n in data.nodes]
    beams_dict = [b.dict() for b in data.beams]
    
    vib_source = data.vibration_source.dict()
    
    result = wind_analyzer.calculate_vibration_impact(
        nodes_dict, beams_dict, vib_source
    )
    
    # Добавляем решения
    if result['status'] != "OK":
        solutions = wind_analyzer.suggest_vibration_dampening(result)
        result['solutions'] = solutions
    
    return result


@app.post("/dynamics/progressive-collapse")
async def progressive_collapse_analysis(data: StructureData):
    """
    Анализ прогрессирующего обрушения.
    Проверяет: если одна балка сломается, упадет ли вся конструкция?
    
    Высший пилотаж инженерной безопасности!
    """
    nodes_dict = [n.dict() for n in data.nodes]
    beams_dict = [b.dict() for b in data.beams]
    
    analyzer = get_collapse_analyzer()
    result = analyzer.analyze_progressive_collapse(nodes_dict, beams_dict)
    
    return result


# === ГЕНЕРАЦИЯ ВАРИАНТОВ ===

@app.post("/engineer/generate-variants")
async def generate_variants(
    width: float, 
    height: float, 
    depth: float,
    obstacles: Optional[str] = Form(None)
):
    """
    Генерирует 3 варианта лесов и сразу считает их надежность.
    
    Args:
        width, height, depth: размеры
        obstacles: JSON-массив препятствий (опционально)
    """
    import json
    
    obstacles_list = None
    if obstacles:
        try:
            obstacles_list = json.loads(obstacles)
        except:
            pass
    
    generator = ScaffoldGenerator()
    raw_variants = generator.generate_options(
        width, height, depth, obstacles=obstacles_list
    )
    
    final_proposals = []
    
    for var in raw_variants:
        physics_res = engineer.calculate_load_map(var["nodes"], var["beams"])
        
        if physics_res["status"] == "OK":
            max_stress = max([r["load_ratio"] for r in physics_res["data"]])
            reliability_score = round((1.0 - max_stress) * 100, 1)
        else:
            reliability_score = 0
            
        final_proposals.append({
            "name": var["variant_name"],
            "description": var["material_info"],
            "nodes": var["nodes"],
            "beams": var["beams"],
            "stats": var["stats"],
            "reliability": reliability_score,
            "efficiency": 100 - (len(var["beams"]) / 10),
            "status": physics_res["status"]
        })
        
    return {"options": final_proposals}


# === ПОЛНЫЙ ЦИКЛ: ФОТО → ПРОЕКТ ===

@app.post("/ai/auto-design")
async def auto_design_scaffold(
    file: UploadFile = File(...), 
    distance: float = Form(...),
    wind_speed: float = Form(0),  # Если 0, не проверяем
    vibration_source: Optional[str] = Form(None)  # JSON
):
    """
    ПОЛНЫЙ ЦИКЛ: 
    Фото → Размеры → 3 Варианта → Физика → Ветер → Вибрация
    
    Самый мощный эндпоинт!
    """
    import json
    
    photo_content = await file.read()
    
    # 1. Детекция объектов
    found_stuff = eyes.analyze_scene(photo_content, distance_to_target=distance)
    
    if not found_stuff:
        return {"status": "ERROR", "message": "Объекты не найдены"}

    target = found_stuff[0] 
    W = target["real_width_m"]
    H = target["real_height_m"]
    D = 1.0  # стандартная глубина площадки

    # 2. Генерация вариантов
    generator = ScaffoldGenerator()
    options = generator.generate_options(W, H, D)
    
    final_proposals = []
    
    for opt in options:
        # Статический расчет
        physics = engineer.calculate_load_map(opt["nodes"], opt["beams"])
        
        reliability = 0
        if physics["status"] == "OK":
            max_load = max([r["load_ratio"] for r in physics["data"]])
            reliability = int((1.0 - max_load) * 100)
        
        # Ветровой анализ (если указан)
        wind_result = None
        if wind_speed > 0:
            wind_result = wind_analyzer.calculate_wind_load(
                opt["nodes"], opt["beams"], wind_speed
            )
        
        # Вибрационный анализ (если указан)
        vibration_result = None
        if vibration_source:
            try:
                vib_data = json.loads(vibration_source)
                vibration_result = wind_analyzer.calculate_vibration_impact(
                    opt["nodes"], opt["beams"], vib_data
                )
            except:
                pass
        
        final_proposals.append({
            "variant": opt["variant_name"],
            "dims": f"{W}x{H}м",
            "material": opt["material_info"],
            "reliability": reliability,
            "nodes": opt["nodes"],
            "beams": opt["beams"],
            "wind_analysis": wind_result,
            "vibration_analysis": vibration_result
        })

    return {
        "status": "SUCCESS",
        "detected_object": target["type"],
        "detected_dims": {"w": W, "h": H},
        "proposals": final_proposals
    }


# === ЭКСПЕРТНАЯ СИСТЕМА (ДЕМОНТАЖ) ===

@app.post("/expert/dismantle-check")
async def dismantle_check(data: StructureData, element_id: str):
    """
    Проверяет правила демонтажа:
    - Логическая безопасность (не упадет ли что-то на голову)
    - Физическая безопасность (выдержат ли оставшиеся балки)
    """
    nodes_dict = [n.dict() for n in data.nodes]
    beams_dict = [b.dict() for b in data.beams]
    
    physics_res = engineer.simulate_removal(
        nodes_dict, beams_dict, element_id, 
        fixed_node_ids=data.fixed_node_ids
    )
    logic_res = expert.validate_dismantle(element_id, nodes_dict, beams_dict)
    
    return {
        "physics_safe": physics_res["safe"],
        "logic_safe": logic_res["can_remove"],
        "message": physics_res["message"] if not physics_res["safe"] else logic_res["reason"],
        "overall_safe": physics_res["safe"] and logic_res["can_remove"]
    }


@app.post("/expert/dismantle-plan")
async def dismantle_plan(data: StructureData):
    """
    Возвращает правильную последовательность разбора лесов.
    Стратегия: сверху вниз, от периферии к центру.
    """
    nodes_dict = [n.dict() for n in data.nodes]
    beams_dict = [b.dict() for b in data.beams]
    
    order = expert.suggest_order(nodes_dict, beams_dict)
    
    return {
        "order": order,
        "total_steps": len(order),
        "strategy": "Top-down, periphery-first"
    }


# === МНОГОРАКУРСНАЯ ФОТОГРАММЕТРИЯ ===

@app.post("/photogrammetry/add-view")
async def add_photogrammetry_view(
    file: UploadFile = File(...),
    ar_points: str = Form(...),  # JSON
    camera_pose: str = Form(...)  # JSON
):
    """
    Добавляет новый ракурс в систему фотограмметрии.
    Объединяет данные из нескольких фото для повышения точности.
    
    Args:
        file: фото
        ar_points: JSON-массив AR-точек
        camera_pose: позиция камеры {position: [x,y,z], rotation: [qx,qy,qz,qw]}
    """
    import json
    
    contents = await file.read()
    
    try:
        ar_points_list = json.loads(ar_points)
        camera_pose_dict = json.loads(camera_pose)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid JSON: {e}")
    
    result = photogrammetry.add_photo_view(
        contents, ar_points_list, camera_pose_dict
    )
    
    return result


@app.get("/photogrammetry/final-model")
async def get_photogrammetry_model():
    """
    Возвращает финальную объединенную 3D-модель из всех ракурсов.
    """
    final_model = photogrammetry.get_final_model()
    return final_model


# === HEALTH CHECK ===

@app.get("/health")
async def health_check():
    """Проверка работоспособности сервера"""
    return {
        "status": "ONLINE",
        "modules": {
            "vision": eyes.model is not None,
            "physics": True,
            "expert": True,
            "dynamics": True,
            "photogrammetry": True
        },
        "version": "2.0.0"
    }


# === СТАТИСТИКА ===

@app.get("/stats")
async def get_stats():
    """Статистика использования"""
    return {
        "photogrammetry_views": len(photogrammetry.views),
        "server_uptime": "N/A"  # Можно добавить через time.time()
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
