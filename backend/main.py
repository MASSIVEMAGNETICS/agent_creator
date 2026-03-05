import os
from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from backend.api.agents import router as agents_router
from backend.api.deployments import router as deployments_router


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    from backend.core.capability_registry import get_global_registry

    registry = get_global_registry()
    print(
        f"[Agent Creator] started — "
        f"{len(registry.list_all())} built-in capabilities registered."
    )
    yield
    print("[Agent Creator] shutting down.")


app = FastAPI(
    title="Agent Creator",
    description="Next-Generation AI Agent Studio",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(agents_router)
app.include_router(deployments_router)


@app.get("/health")
async def health_check() -> dict:
    return {"status": "healthy", "service": "agent-creator"}


# Serve the frontend SPA only when the frontend directory exists
# (keeps test runs clean when the directory is absent)
_frontend_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "frontend")

if os.path.isdir(_frontend_dir):
    app.mount("/static", StaticFiles(directory=_frontend_dir), name="static")

    @app.get("/")
    async def serve_frontend() -> FileResponse:
        return FileResponse(os.path.join(_frontend_dir, "index.html"))
