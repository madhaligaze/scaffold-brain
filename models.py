# models.py
from pydantic import BaseModel
from typing import List, Optional

class Point3D(BaseModel):
    id: str
    x: float
    y: float
    z: float

class BeamElement(BaseModel):
    id: str
    start_node_id: str
    end_node_id: str
    type: str  # "standard" (стойка), "ledger" (ригель), "brace" (диагональ)

class ScaffoldStructure(BaseModel):
    nodes: List[Point3D]
    beams: List[BeamElement]
    # Дополнительные нагрузки (кроме собственного веса)
    live_load_kg: float = 200.0  # Например, 200 кг/м2 (нагрузка от людей)

class AnalysisResult(BaseModel):
    element_id: str
    load_percent: float  # 0.5 = 50% нагрузки
    color: str           # "green", "yellow", "red"
    warning: Optional[str] = None