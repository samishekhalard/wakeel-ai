from __future__ import annotations

import secrets
from datetime import datetime
from fastapi import APIRouter, Request
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.agent import run_agent_turn
from app.models import Lead, Message
from app.scoring import score_lead, build_summary

router = APIRouter(prefix="/api", tags=["chat"])


# ── Dependency ───────────────────────────────────────────────────────────────

async def get_db(request) -> AsyncSession:  # type: ignore[override]
    async with request.app.state.session_factory() as session:
        yield session


# ── Schemas ──────────────────────────────────────────────────────────────────

class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=2000)
    session_id: str | None = None


class ChatResponse(BaseModel):
    reply: str
    session_id: str
    score: str
    handoff: bool
    viewing_booked: bool


class HistoryMessage(BaseModel):
    role: str
    content: str
    created_at: datetime


# ── Helpers ──────────────────────────────────────────────────────────────────

async def _get_or_create_lead(session: AsyncSession, session_id: str) -> Lead:
    result = await session.execute(select(Lead).where(Lead.session_id == session_id))
    lead = result.scalar_one_or_none()
    if not lead:
        lead = Lead(session_id=session_id)
        session.add(lead)
        await session.flush()
    return lead


async def _get_history(session: AsyncSession, session_id: str) -> list[dict]:
    result = await session.execute(
        select(Message)
        .where(Message.session_id == session_id)
        .order_by(Message.created_at)
    )
    rows = result.scalars().all()
    return [{"role": r.role, "content": r.content} for r in rows]


def _apply_context(lead: Lead, context) -> None:
    q = context.qualification
    for field in ("intent", "budget_min", "budget_max", "currency", "city", "area",
                  "property_type", "bedrooms", "payment_method", "timeline",
                  "contact_name", "contact_phone"):
        val = q.get(field)
        if val is not None:
            setattr(lead, field, val)
    if q.get("language"):
        lead.language = q["language"]
    if context.handoff:
        lead.handoff_requested = True
    if context.viewing_booked:
        lead.viewing_booked = True


# ── Endpoints ────────────────────────────────────────────────────────────────

@router.post("/chat", response_model=ChatResponse)
async def chat(request_body: ChatRequest, request: Request):
    session_id = request_body.session_id or secrets.token_hex(16)

    async with request.app.state.session_factory() as session:
        lead = await _get_or_create_lead(session, session_id)
        history = await _get_history(session, session_id)

        history.append({"role": "user", "content": request_body.message})

        reply, context = await run_agent_turn(history, language=lead.language or "en")

        # Persist messages
        session.add(Message(session_id=session_id, role="user", content=request_body.message))
        session.add(Message(session_id=session_id, role="assistant", content=reply))

        # Update lead
        lead.message_count += 1
        _apply_context(lead, context)
        lead.score = score_lead(lead).value
        lead.summary = build_summary(lead)
        lead.updated_at = datetime.utcnow()

        await session.commit()

    return ChatResponse(
        reply=reply,
        session_id=session_id,
        score=lead.score,
        handoff=lead.handoff_requested,
        viewing_booked=lead.viewing_booked,
    )


@router.get("/chat/history/{session_id}", response_model=list[HistoryMessage])
async def get_history(session_id: str, request: Request):
    async with request.app.state.session_factory() as session:
        result = await session.execute(
            select(Message)
            .where(Message.session_id == session_id)
            .order_by(Message.created_at)
        )
        rows = result.scalars().all()
        return [HistoryMessage(role=r.role, content=r.content, created_at=r.created_at) for r in rows]
