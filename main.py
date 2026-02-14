# main.py
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from typing import List
from modules.vision import Eyes
from modules.physics import StructuralBrain

app = FastAPI(title="Bauflex AI Brain")

# Инициализация модулей
eyes = Eyes()
engineer = StructuralBrain()

# --- Модели данных ---
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

# --- Эндпоинты ---

@app.post("/analyze/photo")
async def analyze_photo(file: UploadFile = File(...)):
    """
    1. Принимает фото со стройки.
    2. AI ищет трубы и препятствия.
    3. Возвращает список найденного.
    """
    contents = await file.read()
    obstacles = eyes.detect_obstacles(contents)
    return {"found_objects": obstacles}

@app.post("/engineer/calculate")
async def calculate_structure(data: StructureData):
    """
    Полный расчет нагрузок текущей конструкции.
    """
    # Преобразуем Pydantic модели в словари для инженера
    nodes_dict = [n.dict() for n in data.nodes]
    beams_dict = [b.dict() for b in data.beams]
    
    result = engineer.calculate_load_map(nodes_dict, beams_dict)
    return result

@app.post("/engineer/simulate-removal")
async def simulate_removal(data: StructureData, remove_id: str):
    """
    ТОТ САМЫЙ "AI-ПРЕДСКАЗАТЕЛЬ".
    Что будет, если я удалю эту стойку?
    """
    nodes_dict = [n.dict() for n in data.nodes]
    beams_dict = [b.dict() for b in data.beams]
    
    prediction = engineer.simulate_removal(nodes_dict, beams_dict, remove_id)
    return prediction