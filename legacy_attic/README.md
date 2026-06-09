# legacy_attic

Quarantined dead code, removed from the live import graph but preserved for
reference. **Nothing here is imported by the running server or the test-suite.**

## What's here and why

The repository carried two generations of engine side by side. The live engine
is `world/` + `scaffold/` + `perception/` + `scanning/` + `session/`, exposed
through `api/routes_*_v2.py` and the legacy-compat shim `api/routes_legacy.py`.

- `modules/` — the previous-generation engine (~5000 lines: `vision.py`,
  `session_manager.py`, `builder.py`, `dynamics.py`, `photogrammetry.py`,
  `voxel_world.py`, …). Verified unreachable: importing `main` loads only
  `modules.mesh_builder` (kept in `modules/`, used by `world/mesh_export.py`),
  and no `modules.*` import exists anywhere in the live packages or `tests/`.
- `api/routes_session.py`, `api/routes_planning.py` — near-duplicates of the
  `*_v2` routers that were **never** included in `main.py`'s app.

## Recovering something

`git mv legacy_attic/<path> <original/path>` (or restore from history). If you
revive a `modules/*` file, note its internal `from modules.X` imports were left
intact and expect to move its dependencies too.
