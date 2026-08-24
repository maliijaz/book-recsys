from fastapi import APIRouter, HTTPException, Request

from app.schemas import BookSummary, PersonaDetail, PersonaSummary

router = APIRouter(prefix="/personas", tags=["personas"])


@router.get("", response_model=list[PersonaSummary])
def list_personas(request: Request):
    return request.app.state.store.list_personas()


@router.get("/{persona_id}", response_model=PersonaDetail)
def get_persona(persona_id: int, request: Request):
    persona = request.app.state.store.get_persona(persona_id)
    if persona is None:
        raise HTTPException(status_code=404, detail=f"Persona {persona_id} not found")
    return persona


@router.get("/{persona_id}/recommendations", response_model=list[BookSummary])
def get_persona_recommendations(persona_id: int, request: Request):
    recs = request.app.state.store.get_persona_recommendations(persona_id)
    if recs is None:
        raise HTTPException(status_code=404, detail=f"Persona {persona_id} not found")
    return recs
