from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request


router = APIRouter(tags=["export"])


@router.get("/session/{session_id}/export/latest")
def export_latest(request: Request, session_id: str):
    """
    STAGE 7: Android consumes a stable bundle without server-side logic.
    Returns last exported revision (exports/<rev>/scene_bundle.json).
    """
    state = request.app.state.runtime
    # Prefer explicit last_rev, fall back to latest export directory.
    rev = state.last_rev.get(session_id)
    if not rev:
        exports = state.store.list_exports(session_id)
        rev = exports[-1] if exports else None
    if not rev:
        raise HTTPException(status_code=404, detail={"status": "NO_EXPORT", "msg": "No exported bundle yet"})

    bundle = state.store.load_export(session_id, rev)
    if bundle is None:
        raise HTTPException(status_code=404, detail={"status": "MISSING_BUNDLE", "rev_id": rev})
    return bundle


@router.get("/session/{session_id}/export/{rev_id}")
def export_by_rev(request: Request, session_id: str, rev_id: str):
    state = request.app.state.runtime
    bundle = state.store.load_export(session_id, rev_id)
    if bundle is None:
        raise HTTPException(status_code=404, detail={"status": "MISSING_BUNDLE", "rev_id": rev_id})
    return bundle
