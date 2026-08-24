"""The flagship no-login flow: pick a few books you like, get freshly
re-ranked recommendations computed on the fly from precomputed embeddings.
No database write, no session, no account -- purely a numpy forward pass in
`Store.live_recommendations`."""
from fastapi import APIRouter, HTTPException, Request

from app.schemas import LiveRecommendationRequest, LiveRecommendationResponse

router = APIRouter(prefix="/recommendations", tags=["recommendations"])


@router.post("/live", response_model=LiveRecommendationResponse)
def live_recommendations(payload: LiveRecommendationRequest, request: Request):
    store = request.app.state.store
    result = store.live_recommendations(payload.liked_book_ids, k=payload.k)
    if result is None:
        raise HTTPException(
            status_code=400,
            detail="None of the provided liked_book_ids exist in the catalog",
        )
    return result
