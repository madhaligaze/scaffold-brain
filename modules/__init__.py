# modules/__init__.py
"""
Build AI Brain — модули искусственного интеллекта для строительных лесов.
"""

__version__ = "2.1.0"
__author__  = "Build Engineering Team"

from .vision        import Eyes, SceneDiagnostician, VisionSystem
from .physics       import StructuralBrain
from .builder       import ScaffoldGenerator, ScaffoldExpert, PathFinder
from .dynamics      import DynamicLoadAnalyzer, ProgressiveCollapseAnalyzer
from .photogrammetry import PhotogrammetrySystem, MultiViewFusion
from .geometry      import GeometryUtils, CollisionDetector, WorldGeometry  # ИСПРАВЛЕНО
from .session       import DesignSession, SessionStorage                     # ИСПРАВЛЕНО

__all__ = [
    # Vision
    "Eyes", "SceneDiagnostician", "VisionSystem",
    # Physics
    "StructuralBrain",
    # Builder
    "ScaffoldGenerator", "ScaffoldExpert", "PathFinder",
    # Dynamics
    "DynamicLoadAnalyzer", "ProgressiveCollapseAnalyzer",
    # Photogrammetry
    "PhotogrammetrySystem", "MultiViewFusion",
    # Geometry
    "GeometryUtils", "CollisionDetector", "WorldGeometry",
    # Session
    "DesignSession", "SessionStorage",
]