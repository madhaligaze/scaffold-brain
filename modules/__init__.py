# modules/__init__.py
"""Legacy `modules` package.

Only `modules.mesh_builder` (used by `world.mesh_export`) remains live. The
previous generation of engine code that lived here — vision, physics, builder,
dynamics, photogrammetry, geometry, session managers, etc. — was unreachable
from the running app and from the test-suite and has been quarantined in
`legacy_attic/modules/` (see `legacy_attic/README.md`). Recover from there or
from git history if any of it is needed again.
"""

__version__ = "3.0.0"
