# modules/__init__.py
"""
Bauflex AI Brain - Модули искусственного интеллекта для строительных лесов.
"""

__version__ = "2.0.0"
__author__ = "Bauflex Engineering Team"

from .vision import Eyes, SceneDiagnostician, VisionSystem
from .physics import StructuralBrain
from .builder import ScaffoldGenerator, ScaffoldExpert, PathFinder
from .dynamics import DynamicLoadAnalyzer, ProgressiveCollapseAnalyzer
from .photogrammetry import PhotogrammetrySystem, MultiViewFusion
from .geometry import WorldGeometry

__all__ = [
    # Vision
    "Eyes",
    "SceneDiagnostician",
    "VisionSystem",
    
    # Physics
    "StructuralBrain",
    
    # Builder
    "ScaffoldGenerator",
    "ScaffoldExpert",
    "PathFinder",
    
    # Dynamics
    "DynamicLoadAnalyzer",
    "ProgressiveCollapseAnalyzer",
    
    # Photogrammetry
    "PhotogrammetrySystem",
    "MultiViewFusion",
    
    # Geometry
    "WorldGeometry"
]
