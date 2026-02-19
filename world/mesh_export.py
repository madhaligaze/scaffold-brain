from __future__ import annotations

import numpy as np
import trimesh


def env_mesh_obj_bytes(world_model) -> bytes:
    fn_b = getattr(world_model, "export_env_mesh_obj_bytes", None)
    if callable(fn_b):
        out = fn_b()
        if isinstance(out, (bytes, bytearray)):
            return bytes(out)

    fn_s = getattr(world_model, "export_env_mesh_obj", None)
    if callable(fn_s):
        s = fn_s()
        if isinstance(s, str):
            return s.encode("utf-8")

    return b"# empty obj\n"


def env_mesh_glb_bytes(world_model) -> bytes:
    tsdf = getattr(world_model, "tsdf", None)
    verts = None
    tris = None
    if tsdf is not None:
        fn = getattr(tsdf, "extract_mesh", None)
        if callable(fn):
            try:
                verts, tris = fn()
            except Exception:
                verts, tris = None, None

    if verts is None or tris is None:
        try:
            obj = env_mesh_obj_bytes(world_model).decode("utf-8", errors="ignore")
            mesh = trimesh.load_mesh(trimesh.util.wrap_as_stream(obj.encode("utf-8")), file_type="obj")
            scene = trimesh.Scene()
            scene.add_geometry(mesh, node_name="environment_mesh")
            return scene.export(file_type="glb")
        except Exception:
            return trimesh.Scene().export(file_type="glb")

    v = np.asarray(verts, dtype=np.float32)
    f = np.asarray(tris, dtype=np.int64)
    mesh = trimesh.Trimesh(vertices=v, faces=f, process=False)
    scene = trimesh.Scene()
    scene.add_geometry(mesh, node_name="environment_mesh")
    return scene.export(file_type="glb")
