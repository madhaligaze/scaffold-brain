# modules/__init__.py
"""
Build AI Brain — module package exports.

This package intentionally uses lazy attribute loading so lightweight submodules
(e.g. `modules.active_scanning`) can be imported in minimal environments
without pulling heavyweight optional dependencies.
"""

from importlib import import_module
from typing import Dict, Tuple

__version__ = "2.1.0"
__author__ = "Build Engineering Team"

_EXPORTS: Dict[str, Tuple[str, str]] = {
    # Vision
    "Eyes": (".vision", "Eyes"),
    "SceneDiagnostician": (".vision", "SceneDiagnostician"),
    "VisionSystem": (".vision", "VisionSystem"),
    # Physics
    "StructuralBrain": (".physics", "StructuralBrain"),
    # Builder
    "ScaffoldGenerator": (".builder", "ScaffoldGenerator"),
    # Dynamics
    "DynamicLoadAnalyzer": (".dynamics", "DynamicLoadAnalyzer"),
    "ProgressiveCollapseAnalyzer": (".dynamics", "ProgressiveCollapseAnalyzer"),
    # Photogrammetry
    "PhotogrammetrySystem": (".photogrammetry", "PhotogrammetrySystem"),
    "MultiViewFusion": (".photogrammetry", "MultiViewFusion"),
    # Geometry
    "GeometryUtils": (".geometry", "GeometryUtils"),
    "CollisionDetector": (".geometry", "CollisionDetector"),
    "WorldGeometry": (".geometry", "WorldGeometry"),
    # Session
    "DesignSession": (".session", "DesignSession"),
    "SessionStorage": (".session", "SessionStorage"),
}

__all__ = list(_EXPORTS.keys())


def __getattr__(name: str):
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    module_name, attr_name = _EXPORTS[name]
    module = import_module(module_name, __name__)
    value = getattr(module, attr_name)
    globals()[name] = value
    return value
