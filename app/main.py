from __future__ import annotations

from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pathlib import Path

from app.config import settings
from app.models import create_engine, create_session_factory, init_db
from app.routers import chat, dashboard


@asynccontextmanager
async def lifespan(app: FastAPI):
    engine = create_engine(settings.database_url)
    app.state.engine = engine
    app.state.session_factory = create_session_factory(engine)
    await init_db(engine)
    yield
    await engine.dispose()


app = FastAPI(
    title="Wakeel AI",
    description="AI lead-qualification agent for Dubai & Riyadh real estate",
    version="1.0.0",
    lifespan=lifespan,
)

app.include_router(chat.router)
app.include_router(dashboard.router)

_STATIC = Path(__file__).parent / "static"


@app.get("/", response_class=HTMLResponse)
async def index():
    return (_STATIC / "index.html").read_text(encoding="utf-8")


@app.get("/health")
async def health():
    return {"status": "ok", "version": "1.0.0"}
