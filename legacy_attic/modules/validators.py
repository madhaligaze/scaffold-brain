"""Валидаторы для входных данных API."""

import logging
from math import sqrt
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator, model_validator

logger = logging.getLogger(__name__)


class Point3D(BaseModel):
    """Точка в 3D пространстве."""

    x: float = Field(..., description="X координата в метрах")
    y: float = Field(..., ge=0, description="Y координата (высота), должна быть >= 0")
    z: float = Field(..., description="Z координата в метрах")

    @field_validator("x", "y", "z")
    @classmethod
    def check_reasonable_bounds(cls, value: float):
        if abs(value) > 100:
            raise ValueError(f"Координата {value} вне разумных пределов (|x| <= 100)")
        return value


class ScaffoldElementInput(BaseModel):
    """Входные данные элемента строительных лесов."""

    id: Optional[str] = Field(None, description="ID элемента")
    type: str = Field(..., description="Тип элемента")
    start: Point3D
    end: Point3D

    @field_validator("type")
    @classmethod
    def validate_element_type(cls, value: str):
        allowed_types = {"standard", "vertical", "ledger", "horizontal", "diagonal", "bracing", "deck"}
        if value not in allowed_types:
            raise ValueError(f"Недопустимый тип элемента: {value}. Разрешены: {allowed_types}")
        return value

    @model_validator(mode="after")
    def validate_element_length(self):
        length = sqrt(
            (self.end.x - self.start.x) ** 2
            + (self.end.y - self.start.y) ** 2
            + (self.end.z - self.start.z) ** 2
        )
        if length < 0.1:
            raise ValueError(f"Элемент слишком короткий: {length:.2f}м (минимум 0.1м)")
        if length > 10.0:
            raise ValueError(f"Элемент слишком длинный: {length:.2f}м (максимум 10м)")
        return self


class SessionUpdateAction(BaseModel):
    """Действие обновления сессии."""

    action: str = Field(..., description="Тип действия: REMOVE или ADD")
    element_id: Optional[str] = None
    element_data: Optional[ScaffoldElementInput] = None

    @field_validator("action")
    @classmethod
    def validate_action_type(cls, value: str):
        if value not in {"REMOVE", "ADD"}:
            raise ValueError(f"Недопустимое действие: {value}. Разрешены: REMOVE, ADD")
        return value

    @model_validator(mode="after")
    def validate_action_payload(self):
        if self.action == "REMOVE" and not self.element_id:
            raise ValueError("Для действия REMOVE необходимо указать element_id")
        if self.action == "ADD" and not self.element_data:
            raise ValueError("Для действия ADD необходимо указать element_data")
        return self


class PointCloudInput(BaseModel):
    """Входные данные облака точек."""

    points: List[Point3D] = Field(..., description="Список точек")
    confidence: Optional[List[float]] = Field(None, description="Уверенность для каждой точки (0-1)")

    @field_validator("points")
    @classmethod
    def validate_point_count(cls, value: List[Point3D]):
        if len(value) < 10:
            raise ValueError(f"Слишком мало точек: {len(value)} (минимум 10)")
        if len(value) > 1_000_000:
            raise ValueError(f"Слишком много точек: {len(value)} (максимум 1,000,000)")
        return value

    @model_validator(mode="after")
    def validate_confidence_values(self):
        if self.confidence is None:
            return self
        if len(self.confidence) != len(self.points):
            raise ValueError("Длина confidence должна совпадать с количеством точек")
        for conf in self.confidence:
            if not 0 <= conf <= 1:
                raise ValueError(f"Confidence должен быть в диапазоне [0, 1], получено: {conf}")
        return self


def validate_session_exists(session_id: str, session_manager) -> bool:
    session = session_manager.get_session(session_id)
    if not session:
        logger.warning("Попытка доступа к несуществующей сессии: %s", session_id)
        return False
    return True


def validate_structure_stability(structure: List[Dict]) -> Dict[str, Any]:
    errors: List[str] = []
    warnings: List[str] = []

    ground_elements = [
        element
        for element in structure
        if element.get("start", {}).get("y", 1) < 0.1 or element.get("end", {}).get("y", 1) < 0.1
    ]
    if not ground_elements:
        errors.append("Нет элементов на земле - структура будет левитировать")

    verticals = [element for element in structure if element.get("type") in {"standard", "vertical"}]
    if not verticals:
        warnings.append("Нет вертикальных стоек - структура может быть нестабильной")

    horizontals = [element for element in structure if element.get("type") in {"ledger", "horizontal"}]
    if len(horizontals) > len(verticals) * 3:
        warnings.append(
            f"Много горизонтальных элементов ({len(horizontals)}) на мало вертикальных ({len(verticals)})"
        )

    return {
        "is_valid": len(errors) == 0,
        "errors": errors,
        "warnings": warnings,
    }
