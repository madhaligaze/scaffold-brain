# core/config.py
"""
Конфигурация Build AI Brain
"""
import os
from typing import Optional
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    """Настройки приложения"""
    
    # === Основные настройки ===
    APP_NAME: str = "Build AI Brain"
    VERSION: str = "2.0.0"
    DEBUG: bool = False
    
    # === Сервер ===
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    RELOAD: bool = False
    
    # === Vision ===
    YOLO_MODEL_PATH: str = "yolov8n.pt"
    YOLO_CONFIDENCE_THRESHOLD: float = 0.5
    DEFAULT_FOCAL_LENGTH: float = 800.0  # пиксели
    
    # === Physics ===
    STEEL_YOUNGS_MODULUS: float = 2.1e11  # Па
    STEEL_SHEAR_MODULUS: float = 8.1e10  # Па
    STEEL_POISSON_RATIO: float = 0.3
    STEEL_DENSITY: float = 7850  # кг/м³
    STEEL_YIELD_STRENGTH: float = 235e6  # Па (С235)
    
    # Труба 48x3 мм
    PIPE_OUTER_DIAMETER: float = 0.048  # м
    PIPE_WALL_THICKNESS: float = 0.003  # м
    PIPE_CROSS_SECTION_AREA: float = 4.5e-4  # м²
    PIPE_MOMENT_OF_INERTIA: float = 1.1e-7  # м⁴
    PIPE_TORSION_CONSTANT: float = 2.2e-7  # м⁴
    PIPE_SECTION_MODULUS: float = 4.58e-6  # м³
    
    # === Нагрузки ===
    DEFAULT_DISTRIBUTED_LOAD: float = -1000.0  # Н/м (≈100 кг/м)
    SAFETY_FACTOR: float = 1.5
    
    # Пороги для цветовой индикации
    LOAD_RATIO_GREEN: float = 0.6
    LOAD_RATIO_YELLOW: float = 0.9
    
    # === Динамика ===
    # Ветер
    BASE_WIND_PRESSURE: float = 300.0  # Па
    HEIGHT_FACTOR_COEFFICIENT: float = 0.15
    AIR_DENSITY: float = 1.225  # кг/м³
    
    WIND_FORCE_OK: float = 2000.0  # Н
    WIND_FORCE_WARNING: float = 5000.0  # Н
    
    # Вибрация
    CONVEYOR_VIBRATION_FREQ: float = 25.0  # Гц
    MACHINE_VIBRATION_AMPLITUDE: float = 0.002  # м
    
    RESONANCE_RISK_WARNING: float = 0.3
    RESONANCE_RISK_DANGER: float = 0.7
    
    # === Качество данных ===
    MIN_BRIGHTNESS: float = 50.0
    MAX_BRIGHTNESS: float = 240.0
    MIN_CONTRAST: float = 30.0
    BLUR_THRESHOLD: float = 100.0
    
    MIN_AR_POINTS: int = 4
    OPTIMAL_DISTANCE_MIN: float = 2.0  # м
    OPTIMAL_DISTANCE_MAX: float = 5.0  # м
    
    # === Фотограмметрия ===
    POINT_CLOUD_MERGE_THRESHOLD: float = 0.05  # м
    AR_DRIFT_THRESHOLD: float = 0.1  # м
    MIN_PHOTOGRAMMETRY_VIEWS: int = 2
    
    # === Демонтаж ===
    MIN_GROUND_SUPPORTS: int = 4
    VERTICAL_SEARCH_RADIUS: float = 2.0  # м
    
    # === Файлы ===
    MAX_UPLOAD_SIZE_MB: int = 10
    ALLOWED_IMAGE_EXTENSIONS: list = [".jpg", ".jpeg", ".png", ".webp"]
    
    # === Безопасность ===
    SECRET_KEY: Optional[str] = None
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    
    # === Логирование ===
    LOG_LEVEL: str = "INFO"
    LOG_FORMAT: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    # === База данных (для будущего) ===
    DATABASE_URL: Optional[str] = None
    
    # === Redis (кэширование) ===
    REDIS_URL: Optional[str] = None
    CACHE_TTL: int = 3600  # секунды
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = True


# Глобальный объект настроек
settings = Settings()


# === Вспомогательные функции ===

def get_pipe_properties():
    """Возвращает свойства трубы 48x3"""
    return {
        "diameter": settings.PIPE_OUTER_DIAMETER,
        "thickness": settings.PIPE_WALL_THICKNESS,
        "area": settings.PIPE_CROSS_SECTION_AREA,
        "inertia": settings.PIPE_MOMENT_OF_INERTIA,
        "torsion": settings.PIPE_TORSION_CONSTANT,
        "section_modulus": settings.PIPE_SECTION_MODULUS
    }


def get_steel_properties():
    """Возвращает свойства стали С235"""
    return {
        "E": settings.STEEL_YOUNGS_MODULUS,
        "G": settings.STEEL_SHEAR_MODULUS,
        "nu": settings.STEEL_POISSON_RATIO,
        "rho": settings.STEEL_DENSITY,
        "yield_strength": settings.STEEL_YIELD_STRENGTH
    }


def validate_settings():
    """Проверяет корректность настроек"""
    issues = []
    
    # Проверка YOLO модели
    if not os.path.exists(settings.YOLO_MODEL_PATH):
        issues.append(f"YOLO model not found: {settings.YOLO_MODEL_PATH}")
    
    # Проверка физических констант
    if settings.STEEL_YOUNGS_MODULUS <= 0:
        issues.append("Invalid Young's modulus")
    
    if settings.SAFETY_FACTOR < 1.0:
        issues.append("Safety factor must be >= 1.0")
    
    # Проверка порогов
    if settings.LOAD_RATIO_GREEN >= settings.LOAD_RATIO_YELLOW:
        issues.append("Invalid load ratio thresholds")
    
    return issues


if __name__ == "__main__":
    # Тест конфигурации
    issues = validate_settings()
    
    if issues:
        print("⚠️ Configuration issues:")
        for issue in issues:
            print(f"  - {issue}")
    else:
        print("✓ Configuration valid")
        print(f"\nApp: {settings.APP_NAME} v{settings.VERSION}")
        print(f"Server: {settings.HOST}:{settings.PORT}")
        print(f"YOLO: {settings.YOLO_MODEL_PATH}")
