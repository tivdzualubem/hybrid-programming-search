from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.config import STATIC_DIR, ASSETS_DIR, VALID_SYSTEMS
from app.schemas import SearchRequest, SearchResponse, HealthResponse
from app.search_service import search_service

app = FastAPI(title="Hybrid Programming Search Engine")

app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/")
def home():
    return FileResponse(str(STATIC_DIR / "index.html"))


@app.get("/api/health", response_model=HealthResponse)
def health():
    return HealthResponse(
        status="ok",
        documents_loaded=bool(search_service.documents),
        assets_available=Path(ASSETS_DIR).exists()
    )


@app.post("/api/search", response_model=SearchResponse)
def search(request: SearchRequest):
    if request.system not in VALID_SYSTEMS:
        raise HTTPException(status_code=400, detail="Invalid retrieval system.")

    try:
        results = search_service.search(
            query=request.query,
            system=request.system,
            top_k=request.top_k
        )
        return SearchResponse(
            query=request.query,
            system=request.system,
            top_k=request.top_k,
            results=results
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
