from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from api.rate_limit import RateLimiter
from config.load_config import AppConfig, find_default_config, load_app_config
from perception.scene_graph import SceneGraph
from policy.load_policy import find_policy_file, load_policy_from_yaml
from policy.policy_config import PolicyConfig
from session.session_store import SessionStore
from world.world_model import WorldModel


@dataclass
class RuntimeState:
    config: AppConfig = field(default_factory=AppConfig)
    config_source: str | None = None

    store: SessionStore = field(default_factory=SessionStore)

    policy: PolicyConfig = field(default_factory=PolicyConfig)
    policy_source: str | None = None

    worlds: dict[str, WorldModel] = field(default_factory=dict)
    scene_graphs: dict[str, SceneGraph] = field(default_factory=dict)

    anchors: dict[str, list[dict]] = field(default_factory=dict)
    traces: dict[str, list[dict]] = field(default_factory=dict)
    last_rev: dict[str, str] = field(default_factory=dict)
    last_rev_meta: dict[str, dict] = field(default_factory=dict)
    restored_revision_state: dict[str, dict] = field(default_factory=dict)

    # STAGE A: monotonic timestamp tracking per session
    last_timestamp: dict[str, float] = field(default_factory=dict)

    # Session-level counters for adaptive readiness heuristics.
    session_stats: dict[str, dict] = field(default_factory=dict)

    perception_unavailable: bool = False

    # Telemetry/report rate limiting (in-memory)
    rate_limiter: RateLimiter = field(default_factory=RateLimiter)

    @classmethod
    def build(cls) -> "RuntimeState":
        cfg_path = find_default_config()
        if cfg_path is not None:
            config = load_app_config(cfg_path)
            config_source = str(cfg_path).replace("\\", "/")
        else:
            config = AppConfig()
            config_source = None

        store = SessionStore(sessions_root=config.storage.sessions_root)

        policy_source = None
        policy_file: Path | None = None
        if config.policy.policy_yaml_path:
            candidate = Path(config.policy.policy_yaml_path)
            if candidate.exists() and candidate.is_file():
                policy_file = candidate
        if policy_file is None:
            policy_file = find_policy_file()
        if policy_file is not None:
            policy = load_policy_from_yaml(policy_file)
            policy_source = str(policy_file).replace("\\", "/")
        else:
            policy = PolicyConfig(
                unknown_mode="forbid",
                unknown_buffer_m=0.5,
                min_clearance_m=0.2,
                readiness_observed_ratio_min=0.1,
                unknown_ratio_near_support_max=0.6,
            )

        return cls(
            config=config,
            config_source=config_source,
            store=store,
            policy=policy,
            policy_source=policy_source,
        )

    def get_world(self, session_id: str) -> WorldModel:
        if session_id not in self.worlds:
            self.worlds[session_id] = WorldModel(
                voxel_size=float(self.config.world.voxel_size_m),
                tsdf_trunc=float(self.config.world.tsdf_trunc_m),
            )
        return self.worlds[session_id]

    def get_scene_graph(self, session_id: str) -> SceneGraph:
        if session_id not in self.scene_graphs:
            self.scene_graphs[session_id] = SceneGraph()
        return self.scene_graphs[session_id]

    def policy_status(self) -> dict:
        return {
            "config": {"source": self.config_source, "config": self.config.model_dump()},
            "policy": {"source": self.policy_source, "policy": self.policy.__dict__},
        }
