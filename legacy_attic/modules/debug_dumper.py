"""Debug session dumper."""
from __future__ import annotations

import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional


class DebugDumper:
    """Система отладочных дампов."""

    def __init__(self, dump_dir: str = "/tmp/ai_brain_dumps"):
        self.dump_dir = Path(dump_dir)
        self.dump_dir.mkdir(parents=True, exist_ok=True)

    def dump_session(
        self,
        session_id: str,
        session_data: Dict[str, Any],
        reason: str = "manual",
        include_voxels: bool = False,
    ) -> str:
        """Сохранить полный дамп сессии."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"dump_{session_id}_{timestamp}_{reason}.json"
        filepath = self.dump_dir / filename

        dump_data: Dict[str, Any] = {
            "metadata": {
                "session_id": session_id,
                "timestamp": timestamp,
                "reason": reason,
                "dump_version": "4.0",
            },
            "session": session_data,
            "voxels_included": include_voxels,
        }

        if include_voxels and "scene_context" in session_data:
            voxel_world = session_data["scene_context"].get("voxel_world")
            if voxel_world:
                dump_data["voxel_data"] = self._serialize_voxel_world(voxel_world)

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(dump_data, f, indent=2, default=str)

        print(f"📦 Debug dump saved: {filepath}")
        return str(filepath)

    def dump_generation_error(
        self,
        session_id: str,
        point_cloud: List[Any],
        user_anchors: List[Any],
        target_dimensions: Dict[str, Any],
        error: Exception,
        traceback_str: str,
    ) -> str:
        """Специальный дамп при ошибке генерации."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"ERROR_{session_id}_{timestamp}.json"
        filepath = self.dump_dir / filename

        dump_data = {
            "metadata": {
                "session_id": session_id,
                "timestamp": timestamp,
                "type": "GENERATION_ERROR",
                "error_type": type(error).__name__,
                "error_message": str(error),
            },
            "inputs": {
                "point_cloud": point_cloud[:1000] if len(point_cloud) > 1000 else point_cloud,
                "point_cloud_size": len(point_cloud),
                "user_anchors": user_anchors,
                "target_dimensions": target_dimensions,
            },
            "error": {
                "type": type(error).__name__,
                "message": str(error),
                "traceback": traceback_str,
            },
        }

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(dump_data, f, indent=2, default=str)

        print(f"💥 ERROR dump saved: {filepath}")
        return str(filepath)

    def dump_comparison(
        self,
        session_id: str,
        before: Dict[str, Any],
        after: Dict[str, Any],
        action: str,
    ) -> str:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filepath = self.dump_dir / f"compare_{session_id}_{timestamp}.json"
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "metadata": {
                        "session_id": session_id,
                        "timestamp": timestamp,
                        "type": "COMPARISON",
                        "action": action,
                    },
                    "before": before,
                    "after": after,
                },
                f,
                indent=2,
                default=str,
            )
        return str(filepath)

    def list_dumps(self, session_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """Список всех дампов."""
        dumps = []
        for file in self.dump_dir.glob("*.json"):
            if session_id and session_id not in file.name:
                continue

            stat = file.stat()
            dumps.append(
                {
                    "filename": file.name,
                    "path": str(file),
                    "size_bytes": stat.st_size,
                    "size_mb": round(stat.st_size / 1024 / 1024, 2),
                    "created": datetime.fromtimestamp(stat.st_ctime).isoformat(),
                }
            )

        dumps.sort(key=lambda x: x["created"], reverse=True)
        return dumps

    def load_dump(self, filepath: str) -> Optional[Dict[str, Any]]:
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return None

    def cleanup_old_dumps(self, days: int = 7) -> int:
        """Удалить старые дампы."""
        cutoff = time.time() - (days * 24 * 3600)
        removed = 0

        for file in self.dump_dir.glob("*.json"):
            if file.stat().st_mtime < cutoff:
                file.unlink()
                removed += 1

        print(f"🧹 Cleaned {removed} old dumps (>{days} days)")
        return removed

    def _serialize_voxel_world(self, voxel_world: Any) -> Dict[str, Any]:
        try:
            return {
                "resolution": voxel_world.resolution,
                "total_voxels": voxel_world.total_voxels,
                "occupied_sample": list(list(voxel_world.occupied)[:100]),
            }
        except Exception:
            return {"error": "Failed to serialize VoxelWorld"}
