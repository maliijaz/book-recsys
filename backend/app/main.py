"""FastAPI entrypoint. All reference data is loaded once at startup via the
lifespan handler below (see app/store.py) -- there is no database and no
per-request I/O anywhere in this service."""
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import ALLOWED_ORIGINS, ARTIFACTS_DIR
from app.routers import books, metrics, personas, recommendations
from app.store import Store


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.store = Store(ARTIFACTS_DIR)
    yield


app = FastAPI(
    title="Book Recommender API",
    description=(
        "Serves precomputed batch recommendations and a live re-ranking "
        "endpoint from artifacts produced offline by the pipeline package. "
        "No database, no accounts -- see /docs."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

app.include_router(books.router)
app.include_router(personas.router)
app.include_router(recommendations.router)
app.include_router(metrics.router)


@app.get("/health", tags=["health"])
def health():
    return {"status": "ok"}
