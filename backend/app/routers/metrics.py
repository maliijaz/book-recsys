from fastapi import APIRouter, Request

router = APIRouter(prefix="/metrics", tags=["metrics"])


@router.get("")
def get_metrics(request: Request):
    """Offline evaluation results per model -- powers the frontend's model-card page."""
    return request.app.state.store.get_metrics()
