# main.py
"""
Build AI Brain — серверный мозг Engineering Intelligence.
ВЕРСИЯ 2.1 — все критические баги устранены.
"""
import json
import logging
import uuid
import base64
from binascii import Error as B64Error
from typing import List, Optional, Dict, Any

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Body
from pydantic import BaseModel

from modules.vision import Eyes, SceneDiagnostician, VisionSystem
from modules.physics import StructuralBrain
from modules.builder import ScaffoldExpert, ScaffoldGenerator
from modules.dynamics import DynamicLoadAnalyzer, ProgressiveCollapseAnalyzer
from modules.photogrammetry import PhotogrammetrySystem
from modules.session import DesignSession, SessionStorage
from modules.geometry import WorldGeometry   # ИСПРАВЛЕНО: класс теперь существует

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
)

app = FastAPI(
    title="Bauflex AI Brain",
    description="Engineering Intelligence для строительных лесов",
    version="2.1.0",
)

# ── Инициализация модулей ──────────────────────────────────────────────────────
eyes           = Eyes()
diagnostician  = SceneDiagnostician()
vision_system  = VisionSystem()
engineer       = StructuralBrain()
expert         = ScaffoldExpert()
generator      = ScaffoldGenerator()
wind_analyzer  = DynamicLoadAnalyzer()
photogrammetry = PhotogrammetrySystem()
geometry       = WorldGeometry()

collapse_analyzer: Optional[ProgressiveCollapseAnalyzer] = None

# ── Хранилище сессий ───────────────────────────────────────────────────────────
active_sessions: Dict[str, DesignSession] = {}
session_storage = SessionStorage()


def get_or_restore_session(session_id: str) -> Optional[DesignSession]:
    if session_id in active_sessions:
        return active_sessions[session_id]
    restored = session_storage.load(session_id)          # ИСПРАВЛЕНО: load() теперь существует
    if restored:
        active_sessions[session_id] = restored
    return restored


def get_collapse_analyzer() -> ProgressiveCollapseAnalyzer:
    global collapse_analyzer
    if collapse_analyzer is None:
        collapse_analyzer = ProgressiveCollapseAnalyzer(engineer)
    return collapse_analyzer


# ── Pydantic модели ────────────────────────────────────────────────────────────

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
    fixed_node_ids: Optional[List[str]] = None


class VibrationSource(BaseModel):
    x: float
    y: float
    z: float
    frequency_hz: float = 25.0
    amplitude_m: float = 0.002
    type: str = "conveyor"


class VibrationAnalysisRequest(StructureData):
    vibration_source: VibrationSource


# ════════════════════════════════════════════════════════════
#  СЕССИИ
# ════════════════════════════════════════════════════════════

@app.post("/session/start")
async def start_session():
    """Создаёт новую AR-сессию замера."""
    sid = str(uuid.uuid4())
    session = DesignSession(session_id=sid, vision_system=vision_system)
    active_sessions[sid] = session
    session_storage.save(session)                        # ИСПРАВЛЕНО: save() теперь существует
    logger.info(f"Session started: {sid}")
    return {"session_id": sid, "status": "MEASURING"}


@app.post("/session/stream/{session_id}")
async def stream_session_data(session_id: str, data: Dict[str, Any] = Body(...)):
    """Принимает потоковые данные от Android: image / pose / markers."""
    session = get_or_restore_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    if session.status != "MEASURING":
        raise HTTPException(status_code=409, detail=f"Session status is '{session.status}', expected 'MEASURING'")

    # Шаг 1: Декодируем изображение (ошибка клиента → 200 с warning)
    try:
        image_payload = data.get("image", "")
        if isinstance(image_payload, str):
            if not image_payload:
                raise ValueError("empty image payload")
            image_bytes = base64.b64decode(image_payload)
        else:
            image_bytes = bytes(image_payload)
    except (B64Error, ValueError):
        logger.warning("Bad base64 image from client", exc_info=True)
        return {"status": "RECEIVING", "ai_hints": {"instructions": [], "warnings": ["Ошибка кадра"]}}

    # Шаг 2: Обрабатываем кадр (ошибки движка → 200 с warning)
    try:
        feedback = session.update_world_model(
            image_bytes=image_bytes,
            pose_matrix=data.get("pose", []),
            markers=data.get("markers", []),
        )
    except Exception:
        logger.error("Frame processing error", exc_info=True)
        return {"status": "RECEIVING", "ai_hints": {"instructions": [], "warnings": ["Ошибка обработки кадра"]}}

    session_storage.save(session)
    return {"status": "RECEIVING", "ai_hints": feedback}


@app.post("/session/model/{session_id}")
async def session_model(session_id: str):
    """
    Финализирует сессию:
    Точки + AI детекция → умная генерация → коллизии → физика → оценка → ответ.
    """
    session = get_or_restore_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    session.status = "MODELING"
    session_storage.save(session)

    # 1. Умная генерация вариантов
    try:
        proposals = generator.generate_smart_options(   # ИСПРАВЛЕНО: метод теперь существует
            user_points=session.user_anchors,
            ai_points=session.detected_supports,
            bounds=session.get_bounds(),
        )
    except Exception:
        logger.error("generate_smart_options failed", exc_info=True)
        raise HTTPException(status_code=500, detail="Ошибка генерации вариантов")

    # 2. Для каждого варианта: коллизии → физика → оценка
    final_options = []
    for prop in proposals:
        nodes_list = prop.get("nodes", [])
        beams_list = prop.get("beams", [])

        # Проверка и исправление коллизий
        try:
            collisions = geometry.check_collisions(beams_list, nodes_list)
            if collisions:
                prop = generator.fix_collisions(prop, collisions)  # ИСПРАВЛЕНО: метод теперь существует
                beams_list = prop.get("beams", [])
        except Exception:
            logger.warning("Collision check failed, skipping fix", exc_info=True)
            collisions = []

        # Физический расчёт нагрузок
        try:
            physics_res = engineer.calculate_load_map(nodes_list, beams_list)
        except Exception:
            logger.warning("Physics calc failed for variant", exc_info=True)
            physics_res = {"status": "ERROR", "data": []}

        # Safety score 0–100
        safety_score = 0
        critique = []
        if physics_res.get("status") == "OK":
            loads = [r["load_ratio"] for r in physics_res.get("data", [])]
            if loads:
                max_load = max(loads)
                safety_score = int((1.0 - min(max_load, 1.0)) * 100)
                # Самокритика — что именно слабо
                overloaded = [r for r in physics_res["data"] if r["load_ratio"] > 0.7]
                if overloaded:
                    critique.append(f"⚠️ {len(overloaded)} балок нагружены более чем на 70%")
                if max_load > 0.9:
                    critique.append(f"🔴 Критичная нагрузка: {int(max_load * 100)}% — нужно усиление")
                if not critique:
                    critique.append("✅ Конструкция в норме по нагрузкам")
        elif physics_res.get("status") == "COLLAPSE":
            critique.append("🔴 Конструкция нестабильна — обрушение при расчёте")
        else:
            critique.append("⚠️ Не удалось рассчитать нагрузки")

        if collisions:
            critique.append(f"🔧 Исправлено {len(collisions)} коллизий")

        prop["safety_score"] = safety_score
        prop["physics"] = physics_res
        prop["ai_critique"] = critique
        final_options.append(prop)

    session.status = "DONE"
    session_storage.save(session)

    # Сортируем: лучший вариант первый
    final_options.sort(key=lambda x: x.get("safety_score", 0), reverse=True)

    return {"status": "SUCCESS", "options": final_options}


# ════════════════════════════════════════════════════════════
#  COMPUTER VISION
# ════════════════════════════════════════════════════════════

@app.post("/analyze/photo")
async def analyze_photo(
    file: UploadFile = File(...),
    distance: float = Form(...),
    focal_length: float = Form(800),
):
    """Детекция объектов + оценка реальных размеров."""
    contents = await file.read()
    try:
        detected = eyes.analyze_scene(contents, distance, focal_length)
        return {"status": "OK", "objects": detected, "count": len(detected)}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/analyze/quality-check")
async def quality_check(
    file: UploadFile = File(...),
    distance: float = Form(...),
    ar_points: str = Form("[]"),
):
    """
    Проверка качества данных. Возвращает инструкции, если фото недостаточно.

    ИСПРАВЛЕНО: передавали bytes вместо np.ndarray в diagnostician.check_data_quality()
    Теперь: один раз декодируем → передаём готовый frame везде.
    """
    contents = await file.read()

    try:
        ar_points_list = json.loads(ar_points)
    except Exception:
        ar_points_list = []

    try:
        # Декодируем один раз
        frame = eyes._decode_image_bgr(contents)

        # Детектируем объекты (передаём готовый frame)
        detected = eyes.analyze_scene(frame=frame, distance_to_target=distance)

        # Проверяем качество (передаём готовый frame — ИСПРАВЛЕНО)
        quality = diagnostician.check_data_quality(frame, detected, ar_points_list, distance)

        return {"status": "OK", "quality": quality, "can_proceed": quality["is_ready"]}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


# ════════════════════════════════════════════════════════════
#  ФИЗИКА
# ════════════════════════════════════════════════════════════

@app.post("/engineer/calculate")
async def calculate_structure(data: StructureData):
    """Статический расчёт нагрузок. Цветовая карта: green/yellow/red."""
    result = engineer.calculate_load_map(
        [n.dict() for n in data.nodes],
        [b.dict() for b in data.beams],
        fixed_node_ids=data.fixed_node_ids,
    )
    return result


@app.post("/engineer/simulate-removal")
async def simulate_removal(data: StructureData, remove_id: str):
    """Что будет, если удалить эту балку?"""
    return engineer.simulate_removal(
        [n.dict() for n in data.nodes],
        [b.dict() for b in data.beams],
        remove_id,
        fixed_node_ids=data.fixed_node_ids,
    )


# ════════════════════════════════════════════════════════════
#  ДИНАМИКА
# ════════════════════════════════════════════════════════════

@app.post("/dynamics/wind-analysis")
async def wind_analysis(data: StructureData, wind_speed: float = 20.0, wind_direction: str = "X"):
    return wind_analyzer.calculate_wind_load(
        [n.dict() for n in data.nodes], [b.dict() for b in data.beams], wind_speed, wind_direction
    )


@app.post("/dynamics/vibration-analysis")
async def vibration_analysis(data: VibrationAnalysisRequest):
    result = wind_analyzer.calculate_vibration_impact(
        [n.dict() for n in data.nodes], [b.dict() for b in data.beams], data.vibration_source.dict()
    )
    if result['status'] != "OK":
        result['solutions'] = wind_analyzer.suggest_vibration_dampening(result)
    return result


@app.post("/dynamics/progressive-collapse")
async def progressive_collapse(data: StructureData):
    return get_collapse_analyzer().analyze_progressive_collapse(
        [n.dict() for n in data.nodes], [b.dict() for b in data.beams]
    )


# ════════════════════════════════════════════════════════════
#  ГЕНЕРАЦИЯ ВАРИАНТОВ
# ════════════════════════════════════════════════════════════

@app.post("/engineer/generate-variants")
async def generate_variants(
    width: float,
    height: float,
    depth: float,
    obstacles: Optional[str] = Form(None),
):
    obstacles_list = None
    if obstacles:
        try:
            obstacles_list = json.loads(obstacles)
        except Exception:
            pass

    raw_variants = generator.generate_options(width, height, depth, obstacles=obstacles_list)
    final_proposals = []
    for var in raw_variants:
        physics_res = engineer.calculate_load_map(var["nodes"], var["beams"])
        reliability_score = 0
        if physics_res["status"] == "OK" and physics_res["data"]:
            max_stress = max(r["load_ratio"] for r in physics_res["data"])
            reliability_score = round((1.0 - max_stress) * 100, 1)
        final_proposals.append({
            "name": var["variant_name"],
            "description": var["material_info"],
            "nodes": var["nodes"],
            "beams": var["beams"],
            "stats": var["stats"],
            "reliability": reliability_score,
            "status": physics_res["status"],
        })
    return {"options": final_proposals}


@app.post("/ai/auto-design")
async def auto_design(
    file: UploadFile = File(...),
    distance: float = Form(...),
    wind_speed: float = Form(0),
    vibration_source: Optional[str] = Form(None),
):
    """Полный цикл: Фото → Размеры → 3 варианта → Физика → Ветер → Вибрация."""
    photo_content = await file.read()
    found_stuff = eyes.analyze_scene(photo_content, distance_to_target=distance)
    if not found_stuff:
        return {"status": "ERROR", "message": "Объекты не найдены"}
    target = found_stuff[0]
    W, H, D = target["real_width_m"], target["real_height_m"], 1.0

    options = generator.generate_options(W, H, D)
    final_proposals = []
    for opt in options:
        physics = engineer.calculate_load_map(opt["nodes"], opt["beams"])
        reliability = 0
        if physics["status"] == "OK" and physics["data"]:
            reliability = int((1.0 - max(r["load_ratio"] for r in physics["data"])) * 100)
        wind_result = None
        if wind_speed > 0:
            wind_result = wind_analyzer.calculate_wind_load(opt["nodes"], opt["beams"], wind_speed)
        vibration_result = None
        if vibration_source:
            try:
                vib_data = json.loads(vibration_source)
                vibration_result = wind_analyzer.calculate_vibration_impact(opt["nodes"], opt["beams"], vib_data)
            except Exception:
                pass
        final_proposals.append({
            "variant": opt["variant_name"],
            "dims": f"{W}x{H}м",
            "material": opt["material_info"],
            "reliability": reliability,
            "nodes": opt["nodes"],
            "beams": opt["beams"],
            "wind_analysis": wind_result,
            "vibration_analysis": vibration_result,
        })
    return {"status": "SUCCESS", "detected_object": target["type"],
            "detected_dims": {"w": W, "h": H}, "proposals": final_proposals}


# ════════════════════════════════════════════════════════════
#  ЭКСПЕРТНАЯ СИСТЕМА
# ════════════════════════════════════════════════════════════

@app.post("/expert/dismantle-check")
async def dismantle_check(data: StructureData, element_id: str):
    nodes_dict = [n.dict() for n in data.nodes]
    beams_dict = [b.dict() for b in data.beams]
    physics_res = engineer.simulate_removal(nodes_dict, beams_dict, element_id,
                                            fixed_node_ids=data.fixed_node_ids)
    logic_res = expert.validate_dismantle(element_id, nodes_dict, beams_dict)
    return {
        "physics_safe": physics_res["safe"],
        "logic_safe": logic_res["can_remove"],
        "message": physics_res["message"] if not physics_res["safe"] else logic_res["reason"],
        "overall_safe": physics_res["safe"] and logic_res["can_remove"],
    }


@app.post("/expert/dismantle-plan")
async def dismantle_plan(data: StructureData):
    nodes_dict = [n.dict() for n in data.nodes]
    beams_dict = [b.dict() for b in data.beams]
    order = expert.suggest_order(nodes_dict, beams_dict)
    return {"order": order, "total_steps": len(order), "strategy": "Top-down, periphery-first"}


# ════════════════════════════════════════════════════════════
#  ФОТОГРАММЕТРИЯ
# ════════════════════════════════════════════════════════════

@app.post("/photogrammetry/add-view")
async def add_photogrammetry_view(
    file: UploadFile = File(...),
    ar_points: str = Form(...),
    camera_pose: str = Form(...),
):
    contents = await file.read()
    try:
        ar_points_list = json.loads(ar_points)
        camera_pose_dict = json.loads(camera_pose)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid JSON: {e}")
    return photogrammetry.add_photo_view(contents, ar_points_list, camera_pose_dict)


@app.get("/photogrammetry/final-model")
async def get_photogrammetry_model():
    return photogrammetry.get_final_model()


# ════════════════════════════════════════════════════════════
#  SERVICE
# ════════════════════════════════════════════════════════════

@app.get("/health")
async def health_check():
    return {
        "status": "ONLINE",
        "version": "2.1.0",
        "modules": {
            "vision": eyes.model is not None,
            "physics": True,
            "expert": True,
            "dynamics": True,
            "photogrammetry": True,
            "geometry": True,
        },
    }


@app.get("/stats")
async def get_stats():
    return {
        "active_sessions": session_storage.active_count,
        "photogrammetry_views": len(photogrammetry.views),
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=False)