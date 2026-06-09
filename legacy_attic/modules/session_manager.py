import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional
from uuid import uuid4

# Stage 9/10: reproducible world snapshots
try:
    from modules.world_snapshot import (
        DEFAULT_SNAPSHOT_DIR,
        commit_snapshot,
        list_snapshots,
        load_snapshot,
        restore_voxel_world,
    )
except Exception:  # pragma: no cover
    DEFAULT_SNAPSHOT_DIR = "/tmp/ai_brain_snapshots"
    commit_snapshot = None
    list_snapshots = None
    load_snapshot = None
    restore_voxel_world = None

from modules.session_state import LockInfo, SessionState, compute_world_revision

# Stage 13: raw inputs -> revision artifacts
try:
    from modules.raw_input_artifacts import finalize_incoming_raw_to_revision
except Exception:  # pragma: no cover
    finalize_incoming_raw_to_revision = None


@dataclass
class CameraFrame:
    timestamp: float
    image_data: str = ""
    camera_position: List[float] = field(default_factory=list)
    ar_points: List[Dict[str, Any]] = field(default_factory=list)
    quality_metrics: Dict[str, Any] = field(default_factory=dict)
    detected_objects: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CameraFrame":
        return cls(**data)


class SceneContext:
    def __init__(self) -> None:
        self.anchor_points: List[Dict[str, Any]] = []
        self.all_ar_points: List[Dict[str, Any]] = []
        self.all_detected_objects: List[Dict[str, Any]] = []
        # Stage 2: stable world objects
        self.world_objects: List[Dict[str, Any]] = []
        self._perception_backend = None
        self.point_cloud: List[Dict[str, Any]] = []
        self.obstacles: List[Dict[str, Any]] = []
        self.voxel_world = None
        self.tsdf_integrator = None
        # Stage 5/6/7: last diagnostics + calibration state
        self.last_reprojection: Optional[Dict[str, Any]] = None
        self.reprojection_history: List[Dict[str, Any]] = []
        self.last_scan_suggestions: List[Dict[str, Any]] = []
        self.last_scan_plan: Optional[Dict[str, Any]] = None
        self.readiness_profile: Dict[str, Any] = {}
        self.last_readiness: Optional[Dict[str, Any]] = None

        # Stage 13: explicit measurement protocol packets
        self.measurements: List[Dict[str, Any]] = []
        self.marker_observations: List[Dict[str, Any]] = []

    def ensure_voxel_world(self):
        if self.voxel_world is None:
            try:
                from modules.voxel_world import VoxelWorld

                self.voxel_world = VoxelWorld()
            except Exception:
                self.voxel_world = None
        return self.voxel_world

    def ensure_tsdf_integrator(self):
        if self.tsdf_integrator is None:
            try:
                from modules.tsdf_integrator import TSDFIntegrator

                self.tsdf_integrator = TSDFIntegrator()
            except Exception:
                self.tsdf_integrator = None
        return self.tsdf_integrator

    def ensure_perception_backend(self):
        """Creates PerceptionBackend (Stage 2) and seeds it from stored world_objects."""
        if self._perception_backend is None:
            try:
                from modules.perception_backend import PerceptionBackend

                self._perception_backend = PerceptionBackend()
                self._perception_backend.seed_from_scene(self)
            except Exception:
                self._perception_backend = None
        return self._perception_backend

    def ingest_frame(self, frame: CameraFrame) -> None:
        if frame.ar_points:
            self.anchor_points.extend(frame.ar_points)
            self.all_ar_points.extend(frame.ar_points)
        if frame.detected_objects:
            self.all_detected_objects.extend(frame.detected_objects)
        try:
            qm = frame.quality_metrics or {}
            gs = qm.get("geometry_stats") or {}
            repro = gs.get("reprojection") or qm.get("reprojection")
            if repro:
                self.last_reprojection = repro
                self.reprojection_history.append(repro)
                if len(self.reprojection_history) > 240:
                    self.reprojection_history = self.reprojection_history[-240:]
            ss = qm.get("scan_suggestions")
            if isinstance(ss, list):
                self.last_scan_suggestions = ss
            sp = qm.get("scan_plan")
            if isinstance(sp, dict):
                self.last_scan_plan = sp
            rd = qm.get("readiness")
            if isinstance(rd, dict):
                self.last_readiness = rd
        except Exception:
            pass
        point_cloud = frame.quality_metrics.get("point_cloud") if frame.quality_metrics else None
        if point_cloud:
            self.point_cloud.extend(point_cloud)

    def ingest_measurement_packet(self, packet: Dict[str, Any]) -> None:
        """Persist measurement packet into context (Stage 13)."""
        if not isinstance(packet, dict):
            return
        self.measurements.append(packet)
        if len(self.measurements) > 500:
            self.measurements = self.measurements[-500:]
        obs = packet.get("markers")
        if isinstance(obs, list) and obs:
            self.marker_observations.extend(obs)
            if len(self.marker_observations) > 1000:
                self.marker_observations = self.marker_observations[-1000:]

    def get_summary(self) -> Dict[str, Any]:
        return {
            "anchors": len(self.anchor_points),
            "ar_points": len(self.all_ar_points),
            "detected_objects": len(self.all_detected_objects),
            "world_objects": len(self.world_objects),
            "point_cloud_points": len(self.point_cloud),
            "scan_suggestions": len(self.last_scan_suggestions),
            "has_reprojection": bool(self.last_reprojection),
            "has_scan_plan": bool(self.last_scan_plan),
            "measurement_packets": len(self.measurements),
            "marker_observations": len(self.marker_observations),
        }

    def to_dict(self) -> Dict[str, Any]:
        return {
            "anchor_points": self.anchor_points,
            "all_ar_points": self.all_ar_points,
            "all_detected_objects": self.all_detected_objects,
            "world_objects": self.world_objects,
            "point_cloud": self.point_cloud,
            "obstacles": self.obstacles,
            "last_reprojection": self.last_reprojection,
            "reprojection_history": self.reprojection_history,
            "last_scan_suggestions": self.last_scan_suggestions,
            "last_scan_plan": self.last_scan_plan,
            "readiness_profile": self.readiness_profile,
            "last_readiness": self.last_readiness,
            "measurements": self.measurements,
            "marker_observations": self.marker_observations,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SceneContext":
        ctx = cls()
        ctx.anchor_points = data.get("anchor_points", [])
        ctx.all_ar_points = data.get("all_ar_points", [])
        ctx.all_detected_objects = data.get("all_detected_objects", [])
        ctx.world_objects = data.get("world_objects", [])
        ctx.point_cloud = data.get("point_cloud", [])
        ctx.obstacles = data.get("obstacles", [])
        ctx.last_reprojection = data.get("last_reprojection")
        ctx.reprojection_history = data.get("reprojection_history", [])
        ctx.last_scan_suggestions = data.get("last_scan_suggestions", [])
        ctx.last_scan_plan = data.get("last_scan_plan")
        ctx.readiness_profile = data.get("readiness_profile", {})
        ctx.last_readiness = data.get("last_readiness")
        ctx.measurements = data.get("measurements", [])
        ctx.marker_observations = data.get("marker_observations", [])
        return ctx


class Session:
    def __init__(self, session_id: str):
        self.session_id = session_id
        self.created_at = time.time()
        self.updated_at = self.created_at
        self.last_activity = self.created_at
        self.status = "ACTIVE"
        self.lifecycle_state = SessionState.SCANNING
        self.lock_info: Optional[LockInfo] = None
        self.world_revision: str = ""
        self.frames: List[CameraFrame] = []
        self.scene_context = SceneContext()
        self.generated_variants: List[Dict[str, Any]] = []
        self.current_structure: List[Dict[str, Any]] = []
        self.structure_history: List[Dict[str, Any]] = []
        self.user_anchors: List[Dict[str, Any]] = []
        self.total_frames_processed = 0
        self.total_objects_detected = 0
        self._structural_graph = None
        self.world_locked = False
        self.locked_mesh_version: Optional[int] = None
        self.mesh_version = 0
        self.locked_snapshot_revision: Optional[str] = None
        self.current_structure_revision: Optional[str] = None

    def _touch(self) -> None:
        now = time.time()
        self.updated_at = now
        self.last_activity = now

    def add_frame(self, frame: CameraFrame) -> None:
        self.frames.append(frame)
        self.scene_context.ingest_frame(frame)
        if frame.ar_points:
            self.user_anchors.extend(frame.ar_points)
        self.total_frames_processed += 1
        self.mesh_version += 1
        self.total_objects_detected += len(frame.detected_objects)
        self._touch()

    # ── Stage 13: measurement packets (anchors / ruler / markers) ─────────

    @staticmethod
    def _upsert_by_id(items: List[Dict[str, Any]], item: Dict[str, Any], key: str = "id") -> None:
        try:
            v = item.get(key)
            if not v:
                items.append(item)
                return
            for i, existing in enumerate(items):
                if existing.get(key) == v:
                    items[i] = item
                    return
            items.append(item)
        except Exception:
            items.append(item)

    def ingest_measurement_packet(self, packet: Dict[str, Any]) -> None:
        """Ingest a validated measurement protocol packet."""
        try:
            self.scene_context.measurements.append(packet)
            if len(self.scene_context.measurements) > 500:
                self.scene_context.measurements = self.scene_context.measurements[-500:]

            for a in packet.get("anchors", []) or []:
                anchor = {
                    "id": a.get("anchor_id") or a.get("id"),
                    "x": float((a.get("world_point") or {}).get("x", 0.0)),
                    "y": float((a.get("world_point") or {}).get("y", 0.0)),
                    "z": float((a.get("world_point") or {}).get("z", 0.0)),
                    "label": a.get("label"),
                    "confidence": float(a.get("confidence", 1.0)) if a.get("confidence") is not None else 1.0,
                    "source": a.get("source", "measurement"),
                }
                self._upsert_by_id(self.scene_context.anchor_points, anchor, key="id")
                self._upsert_by_id(self.user_anchors, anchor, key="id")

            for m in packet.get("markers", []) or []:
                marker = {
                    "id": m.get("marker_id") or m.get("id"),
                    "world_point": m.get("world_point"),
                    "normal": m.get("normal"),
                    "size_m": m.get("size_m"),
                    "confidence": m.get("confidence"),
                    "source": m.get("source", "marker"),
                }
                self.scene_context.marker_observations.append(marker)
                if len(self.scene_context.marker_observations) > 800:
                    self.scene_context.marker_observations = self.scene_context.marker_observations[-800:]
        finally:
            self._touch()

    def lock_world(
        self,
        reason: str = "user_lock",
        thresholds: Optional[Dict[str, Any]] = None,
        readiness: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Freeze current geometry snapshot for planning/export stage."""
        self.world_locked = True
        self.locked_mesh_version = self.mesh_version
        self.lifecycle_state = SessionState.LOCKED
        self.status = "LOCKED"
        self.world_revision = compute_world_revision(self.to_dict())
        self.lock_info = LockInfo(
            locked_at=time.time(),
            reason=reason,
            world_revision=self.world_revision,
            readiness=readiness or (self.scene_context.last_readiness or {}),
            thresholds=thresholds or {},
        )
        self._touch()
        return {
            "locked_mesh_version": self.locked_mesh_version,
            "lifecycle_state": self.lifecycle_state.value,
            "lock_info": self.lock_info.to_dict(),
        }

    def unlock_world(self, reason: str = "user_unlock") -> Dict[str, Any]:
        self.world_locked = False
        self.locked_mesh_version = None
        self.lifecycle_state = SessionState.SCANNING
        self.status = "ACTIVE"
        if self.lock_info:
            self.structure_history.append(
                {
                    "timestamp": time.time(),
                    "action": "UNLOCK_WORLD",
                    "reason": reason,
                    "world_revision": self.lock_info.world_revision,
                }
            )
        self.lock_info = None
        self._touch()
        return {"lifecycle_state": self.lifecycle_state.value}

    def is_locked(self) -> bool:
        return self.lifecycle_state == SessionState.LOCKED or self.status == "LOCKED"

    def get_state(self) -> Dict[str, Any]:
        return {
            "status": self.status,
            "lifecycle_state": getattr(self.lifecycle_state, "value", str(self.lifecycle_state)),
            "is_locked": self.is_locked(),
            "world_revision": self.world_revision,
            "lock_info": self.lock_info.to_dict() if self.lock_info else None,
        }

    def add_variant(self, variant: Dict[str, Any]) -> None:
        self.generated_variants.append(variant)
        self._touch()

    def save_structure(self, elements: List[Dict[str, Any]], revision: Optional[str] = None) -> None:
        self.current_structure = elements
        if revision:
            self.current_structure_revision = str(revision)
        self.structure_history.append(
            {
                "timestamp": time.time(),
                "action": "SAVE_STRUCTURE",
                "elements_count": len(elements),
                "revision": self.current_structure_revision,
            }
        )
        self._touch()

    def remove_element(self, element_id: str) -> bool:
        before = len(self.current_structure)
        self.current_structure = [el for el in self.current_structure if el.get("id") != element_id]
        changed = len(self.current_structure) != before
        if changed:
            self.structure_history.append(
                {
                    "timestamp": time.time(),
                    "action": "REMOVE",
                    "element_id": element_id,
                    "elements_count": len(self.current_structure),
                }
            )
            self._touch()
        return changed

    def add_element(self, element_data: Dict[str, Any]) -> None:
        self.current_structure.append(element_data)
        self.structure_history.append(
            {
                "timestamp": time.time(),
                "action": "ADD",
                "element_id": element_data.get("id"),
                "elements_count": len(self.current_structure),
            }
        )
        self._touch()

    def is_expired(self, timeout_seconds: int) -> bool:
        return (time.time() - self.last_activity) > timeout_seconds

    def ensure_structural_graph(self):
        if self._structural_graph is None:
            try:
                from modules.structural_graph import StructuralGraph

                self._structural_graph = StructuralGraph()
            except Exception:
                self._structural_graph = None
        return self._structural_graph

    def get_context_summary(self) -> Dict[str, Any]:
        return {
            "session_id": self.session_id,
            "status": self.status,
            "lifecycle_state": getattr(self.lifecycle_state, "value", str(self.lifecycle_state)),
            "world_revision": self.world_revision,
            "lock_info": self.lock_info.to_dict() if self.lock_info else None,
            "frames": len(self.frames),
            "variants": len(self.generated_variants),
            "current_structure_elements": len(self.current_structure),
            "scene": self.scene_context.get_summary(),
            "last_activity": self.last_activity,
            "total_frames_processed": self.total_frames_processed,
            "total_objects_detected": self.total_objects_detected,
            "world_locked": self.world_locked,
            "locked_mesh_version": self.locked_mesh_version,
            "mesh_version": self.mesh_version,
            "locked_snapshot_revision": self.locked_snapshot_revision,
            "current_structure_revision": self.current_structure_revision,
        }

    # ── Stage 9: world snapshot (reproducibility) ─────────────────────────

    def commit_world_snapshot(
        self,
        base_dir: str = DEFAULT_SNAPSHOT_DIR,
        reason: str = "auto",
        revision: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """Persist a reproducible snapshot of current world model.

        Stores: session context + sparse voxel world.
        Returns snapshot metadata (revision, counts, created_at) or None.
        """
        if commit_snapshot is None:
            return None
        try:
            session_dict = self.to_dict()
            voxel_world = self.scene_context.ensure_voxel_world()
            if voxel_world is None:
                return None
            ref = commit_snapshot(
                session_id=self.session_id,
                session_dict=session_dict,
                voxel_world=voxel_world,
                root_dir=base_dir,
                reason=reason,
            )
            self.locked_snapshot_revision = self.locked_snapshot_revision or ref.revision

            # Stage 13: attach raw inputs captured during scanning to the revision.
            if finalize_incoming_raw_to_revision is not None:
                try:
                    finalize_incoming_raw_to_revision(
                        session_id=self.session_id,
                        revision=str(ref.revision),
                        root_dir=str(base_dir),
                    )
                except Exception:
                    pass
            return {"session_id": self.session_id, "revision": ref.revision}
        except Exception:
            return None

    def list_world_snapshots(self, base_dir: str = DEFAULT_SNAPSHOT_DIR) -> List[Dict[str, Any]]:
        if list_snapshots is None:
            return []
        try:
            return list_snapshots(self.session_id, root_dir=base_dir)
        except Exception:
            return []

    def restore_world_snapshot(
        self,
        revision: str,
        base_dir: str = DEFAULT_SNAPSHOT_DIR,
    ) -> bool:
        """Restore snapshot into this in-memory session.

        WARNING: overwrites scene_context, frames (truncated), and voxel world.
        """
        if load_snapshot is None:
            return False
        try:
            snap = load_snapshot(self.session_id, revision, root_dir=base_dir)
            if not snap:
                return False
            voxel_world = self.scene_context.ensure_voxel_world()
            if voxel_world is None or restore_voxel_world is None:
                return False
            restore_voxel_world(voxel_world, snap.get("voxel_payload") or {})
            self.locked_snapshot_revision = str(revision)
            self._touch()
            return True
        except Exception:
            return False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "session_id": self.session_id,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "last_activity": self.last_activity,
            "status": self.status,
            "lifecycle_state": getattr(self.lifecycle_state, "value", str(self.lifecycle_state)),
            "world_revision": self.world_revision,
            "lock_info": self.lock_info.to_dict() if self.lock_info else None,
            "frames": [f.to_dict() for f in self.frames[-200:]],
            "scene_context": self.scene_context.to_dict(),
            "generated_variants": self.generated_variants,
            "current_structure": self.current_structure,
            "structure_history": self.structure_history,
            "user_anchors": self.user_anchors,
            "total_frames_processed": self.total_frames_processed,
            "total_objects_detected": self.total_objects_detected,
            "world_locked": self.world_locked,
            "locked_mesh_version": self.locked_mesh_version,
            "mesh_version": self.mesh_version,
            "locked_snapshot_revision": self.locked_snapshot_revision,
            "current_structure_revision": self.current_structure_revision,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Session":
        s = cls(data["session_id"])
        s.created_at = data.get("created_at", s.created_at)
        s.updated_at = data.get("updated_at", s.updated_at)
        s.last_activity = data.get("last_activity", s.updated_at)
        s.status = data.get("status", "ACTIVE")
        try:
            s.lifecycle_state = SessionState(data.get("lifecycle_state") or ("LOCKED" if s.status == "LOCKED" else "SCANNING"))
        except Exception:
            s.lifecycle_state = SessionState.SCANNING
        s.world_revision = data.get("world_revision", "")
        s.lock_info = LockInfo.from_dict(data.get("lock_info"))
        s.frames = [CameraFrame.from_dict(item) for item in data.get("frames", [])]
        s.scene_context = SceneContext.from_dict(data.get("scene_context", {}))
        s.generated_variants = data.get("generated_variants", [])
        s.current_structure = data.get("current_structure", [])
        s.structure_history = data.get("structure_history", [])
        s.user_anchors = data.get("user_anchors", [])
        s.total_frames_processed = data.get("total_frames_processed", len(s.frames))
        s.total_objects_detected = data.get("total_objects_detected", len(s.scene_context.all_detected_objects))
        s.world_locked = bool(data.get("world_locked", False))
        s.locked_mesh_version = data.get("locked_mesh_version")
        s.mesh_version = int(data.get("mesh_version", s.mesh_version))
        s.locked_snapshot_revision = data.get("locked_snapshot_revision")
        s.current_structure_revision = data.get("current_structure_revision")
        return s

    def save_to_disk(self, base_dir: str = "/tmp/ai_brain_sessions") -> bool:
        """Сохранить сессию на диск."""
        try:
            session_dir = Path(base_dir) / self.session_id
            session_dir.mkdir(parents=True, exist_ok=True)

            metadata = {
                "session_id": self.session_id,
                "created_at": self.created_at,
                "last_activity": self.last_activity,
                "updated_at": self.updated_at,
                "status": self.status,
                "total_frames_processed": self.total_frames_processed,
                "total_objects_detected": self.total_objects_detected,
                "world_locked": self.world_locked,
                "locked_mesh_version": self.locked_mesh_version,
                "mesh_version": self.mesh_version,
                "locked_snapshot_revision": self.locked_snapshot_revision,
                "current_structure_revision": self.current_structure_revision,
                "lifecycle_state": getattr(self.lifecycle_state, "value", str(self.lifecycle_state)),
                "world_revision": self.world_revision,
                "lock_info": self.lock_info.to_dict() if self.lock_info else None,
            }
            with open(session_dir / "metadata.json", "w", encoding="utf-8") as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False)

            if self.current_structure:
                with open(session_dir / "current_structure.json", "w", encoding="utf-8") as f:
                    json.dump(self.current_structure, f, indent=2, ensure_ascii=False)

            if self.scene_context.point_cloud:
                with open(session_dir / "point_cloud.json", "w", encoding="utf-8") as f:
                    json.dump(self.scene_context.point_cloud, f, indent=2, ensure_ascii=False)

            history_payload = {
                "frames": [f.to_dict() for f in self.frames[-200:]],
                "generated_variants": self.generated_variants,
                "scene_context": self.scene_context.to_dict(),
                "user_anchors": self.user_anchors,
                "structure_history": self.structure_history,
            }
            with open(session_dir / "history.json", "w", encoding="utf-8") as f:
                json.dump(history_payload, f, indent=2, ensure_ascii=False)

            print(f"✓ Session {self.session_id} saved to disk")
            return True
        except Exception as e:
            print(f"✗ Failed to save session: {e}")
            return False

    @classmethod
    def load_from_disk(cls, session_id: str, base_dir: str = "/tmp/ai_brain_sessions") -> Optional["Session"]:
        """Загрузить сессию с диска."""
        try:
            session_dir = Path(base_dir) / session_id
            if not session_dir.exists():
                return None

            with open(session_dir / "metadata.json", "r", encoding="utf-8") as f:
                metadata = json.load(f)

            session = cls(session_id=session_id)
            session.created_at = metadata.get("created_at", session.created_at)
            session.updated_at = metadata.get("updated_at", session.updated_at)
            session.last_activity = metadata.get("last_activity", session.updated_at)
            session.status = metadata.get("status", "ACTIVE")
            session.total_frames_processed = metadata.get("total_frames_processed", 0)
            session.total_objects_detected = metadata.get("total_objects_detected", 0)
            session.world_locked = metadata.get("world_locked", False)
            session.locked_mesh_version = metadata.get("locked_mesh_version")
            session.mesh_version = metadata.get("mesh_version", session.mesh_version)
            session.locked_snapshot_revision = metadata.get("locked_snapshot_revision")
            session.current_structure_revision = metadata.get("current_structure_revision")
            try:
                session.lifecycle_state = SessionState(
                    metadata.get("lifecycle_state")
                    or ("LOCKED" if metadata.get("status") == "LOCKED" else "SCANNING")
                )
            except Exception:
                session.lifecycle_state = SessionState.SCANNING
            session.world_revision = metadata.get("world_revision", "")
            session.lock_info = LockInfo.from_dict(metadata.get("lock_info"))

            structure_file = session_dir / "current_structure.json"
            if structure_file.exists():
                with open(structure_file, "r", encoding="utf-8") as f:
                    session.current_structure = json.load(f)

            pc_file = session_dir / "point_cloud.json"
            if pc_file.exists():
                with open(pc_file, "r", encoding="utf-8") as f:
                    session.scene_context.point_cloud = json.load(f)

            history_file = session_dir / "history.json"
            if history_file.exists():
                with open(history_file, "r", encoding="utf-8") as f:
                    history_data = json.load(f)
                session.frames = [CameraFrame.from_dict(item) for item in history_data.get("frames", [])]
                session.generated_variants = history_data.get("generated_variants", [])
                session.scene_context = SceneContext.from_dict(history_data.get("scene_context", {}))
                session.user_anchors = history_data.get("user_anchors", [])
                session.structure_history = history_data.get("structure_history", [])

            print(f"✓ Session {session_id} loaded from disk")
            return session
        except Exception as e:
            print(f"✗ Failed to load session: {e}")
            return None


class SessionManager:
    def __init__(
        self,
        base_dir: str = "/tmp/ai_brain_sessions",
        session_timeout: int = 60 * 60 * 12,
        snapshot_dir: str = DEFAULT_SNAPSHOT_DIR,
    ):
        self.sessions: Dict[str, Session] = {}
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.session_timeout = session_timeout
        self.snapshot_dir = Path(snapshot_dir)
        self.snapshot_dir.mkdir(parents=True, exist_ok=True)

    def create_session(self) -> str:
        session_id = f"session_{uuid4().hex[:10]}"
        self.sessions[session_id] = Session(session_id)
        self.auto_save_session(session_id)
        return session_id

    def get_session(self, session_id: str) -> Optional[Session]:
        """Получить сессию (сначала из RAM, потом с диска)."""
        session = self.sessions.get(session_id)
        if session:
            if session.is_expired(self.session_timeout):
                self.delete_session(session_id)
                return None
            return session

        session = Session.load_from_disk(session_id, base_dir=str(self.base_dir))
        if session:
            if session.is_expired(self.session_timeout):
                self.delete_session(session_id)
                return None
            self.sessions[session_id] = session
            return session
        return None

    def delete_session(self, session_id: str) -> bool:
        removed = self.sessions.pop(session_id, None) is not None
        session_dir = self.base_dir / session_id
        if session_dir.exists():
            for path in session_dir.glob("*"):
                path.unlink(missing_ok=True)
            session_dir.rmdir()
            removed = True
        return removed

    def export_session_data(self, session_id: str) -> Dict[str, Any]:
        session = self.get_session(session_id)
        if not session:
            return {}
        return session.to_dict()

    # ── Stage 9: snapshot API helpers ─────────────────────────────────────

    def commit_snapshot(self, session_id: str, base_dir: str = DEFAULT_SNAPSHOT_DIR, reason: str = "manual") -> Dict[str, Any]:
        session = self.get_session(session_id)
        if not session:
            return {}
        meta = session.commit_world_snapshot(base_dir=base_dir, reason=reason)
        return meta or {}

    def list_snapshots(self, session_id: str, base_dir: str = DEFAULT_SNAPSHOT_DIR) -> List[Dict[str, Any]]:
        session = self.get_session(session_id)
        if not session:
            # allow listing even if session not loaded in RAM
            if list_snapshots is None:
                return []
            try:
                return list_snapshots(session_id, root_dir=base_dir)
            except Exception:
                return []
        return session.list_world_snapshots(base_dir=base_dir)

    def restore_snapshot(self, session_id: str, revision: str, base_dir: str = DEFAULT_SNAPSHOT_DIR) -> bool:
        session = self.get_session(session_id)
        if not session:
            return False
        ok = session.restore_world_snapshot(revision=revision, base_dir=base_dir)
        if ok:
            self.auto_save_session(session_id)
        return ok

    # Stage 10 convenience API
    def commit_world_snapshot(self, session_id: str, reason: str = "manual") -> str:
        meta = self.commit_snapshot(session_id, base_dir=str(self.snapshot_dir), reason=reason)
        revision = meta.get("revision")
        if not revision:
            raise ValueError("Snapshot commit failed")
        self.auto_save_session(session_id)
        return str(revision)

    def list_world_snapshots(self, session_id: str) -> List[Dict[str, Any]]:
        return self.list_snapshots(session_id, base_dir=str(self.snapshot_dir))

    def restore_world_snapshot(self, session_id: str, revision: str) -> bool:
        ok = self.restore_snapshot(session_id, revision=revision, base_dir=str(self.snapshot_dir))
        if ok:
            self.auto_save_session(session_id)
        return ok

    def lock_session(self, session_id: str, reason: str = "lock") -> Dict[str, Any]:
        session = self.get_session(session_id)
        if not session:
            raise ValueError("Session not found")
        lock_state = session.lock_world(reason=reason)
        revision = self.commit_world_snapshot(session_id, reason=f"lock:{reason}")
        session.locked_snapshot_revision = revision
        self.auto_save_session(session_id)
        return {
            "locked": True,
            "mesh_version": lock_state.get("locked_mesh_version"),
            "snapshot_revision": revision,
        }

    def unlock_session(self, session_id: str) -> Dict[str, Any]:
        session = self.get_session(session_id)
        if not session:
            raise ValueError("Session not found")
        session.unlock_world(reason="unlock")
        session.locked_snapshot_revision = None
        self.auto_save_session(session_id)
        return {"locked": False}

    def save_to_disk(self, session: Session) -> bool:
        return session.save_to_disk(str(self.base_dir))

    def load_from_disk(self, session_id: str) -> Optional[Session]:
        return Session.load_from_disk(session_id, base_dir=str(self.base_dir))

    def auto_save_session(self, session_id: str) -> bool:
        """Автосохранение сессии на диск."""
        session = self.sessions.get(session_id)
        if session:
            return session.save_to_disk(str(self.base_dir))
        return False

    def restore_sessions(self) -> int:
        restored = 0
        for entry in self.base_dir.glob("session_*"):
            if not entry.is_dir():
                continue
            session = self.load_from_disk(entry.name)
            if session and not session.is_expired(self.session_timeout):
                self.sessions[entry.name] = session
                restored += 1
        return restored


session_manager = SessionManager()
