from __future__ import annotations

from contextlib import asynccontextmanager
import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from backend.service import SemanticSearchService


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1, description="Natural-language search query.")
    k: int = Field(default=10, ge=1, le=50, description="Number of results to return.")


class SearchResult(BaseModel):
    id: int
    text: str
    score: float


class SearchResponse(BaseModel):
    results: list[SearchResult]


class StatsResponse(BaseModel):
    total_passages: int
    embedding_model: str
    index_type: str
    index_loaded: bool
    artifacts_present: bool
    model_loaded: bool


service = SemanticSearchService()


@asynccontextmanager
async def lifespan(_: FastAPI):
    service.warmup(
        preload_model=os.environ.get("PRELOAD_MODEL_ON_STARTUP", "false").lower()
        == "true"
    )
    yield


app = FastAPI(
    title="Semantic Search Engine API",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def get_ready_service() -> SemanticSearchService:
    if not service.ready:
        service.load_artifacts()

    if not service.ready:
        raise HTTPException(
            status_code=503,
            detail="Search index is not ready. Build the FAISS artifacts first.",
        )

    return service


@app.get("/health")
def health() -> dict[str, object]:
    stats = service.stats()
    return {
        "status": "ok",
        "index_loaded": stats["index_loaded"],
        "artifacts_present": stats["artifacts_present"],
        "model_loaded": stats["model_loaded"],
    }


@app.get("/")
def root() -> dict[str, object]:
    return health()


@app.post("/search", response_model=SearchResponse)
def search(request: SearchRequest) -> SearchResponse:
    search_service = get_ready_service()
    results = [
        SearchResult(id=item["id"], text=item["text"], score=item["score"])
        for item in search_service.search(request.query, k=request.k)
    ]
    return SearchResponse(results=results)


@app.get("/stats", response_model=StatsResponse)
def stats() -> StatsResponse:
    service_stats = service.stats()
    return StatsResponse(**service_stats)
