from __future__ import annotations

import secrets
from datetime import datetime
from fastapi import APIRouter, Request, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.agent import run_agent_turn
from app.models import Lead, Message
from app.scoring import score_lead, build_summary

router = APIRouter(prefix="/api", tags=["chat"])


async def get_db(request) -> AsyncSession:  # type: ignore[override]
    async with request.app.state.session_factory() as session:
        yield session


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=2000)
    session_id: str | None = None


class ChatResponse(BaseModel):
    reply: str
    session_id: str
    score: str
    handoff: bool
    viewing_booked: bool
    listings: list[dict] = []


class HistoryMessage(BaseModel):
    role: str
    content: str
    created_at: datetime


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
    if context.viewing_time:
        lead.viewing_time = context.viewing_time


@router.post("/chat", response_model=ChatResponse)
async def chat(request_body: ChatRequest, request: Request):
    session_id = request_body.session_id or secrets.token_hex(16)

    async with request.app.state.session_factory() as session:
        lead = await _get_or_create_lead(session, session_id)
        history = await _get_history(session, session_id)

        history.append({"role": "user", "content": request_body.message})

        reply, context = await run_agent_turn(history, language=lead.language or "en")

        session.add(Message(session_id=session_id, role="user", content=request_body.message))
        session.add(Message(session_id=session_id, role="assistant", content=reply))

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
        listings=context.listings,
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


@router.post("/leads/{session_id}/contacted")
async def mark_contacted(session_id: str, request: Request):
    async with request.app.state.session_factory() as session:
        result = await session.execute(select(Lead).where(Lead.session_id == session_id))
        lead = result.scalar_one_or_none()
        if not lead:
            raise HTTPException(status_code=404, detail="Lead not found")
        lead.contacted = not lead.contacted  # toggle
        await session.commit()
    return {"contacted": lead.contacted}


@router.get("/leads/{session_id}/conversation", response_class=HTMLResponse)
async def conversation_view(session_id: str, request: Request):
    async with request.app.state.session_factory() as session:
        lead_res = await session.execute(select(Lead).where(Lead.session_id == session_id))
        lead = lead_res.scalar_one_or_none()
        msgs_res = await session.execute(
            select(Message).where(Message.session_id == session_id).order_by(Message.created_at)
        )
        messages = msgs_res.scalars().all()

    lang_flag = {"ar": "🇦🇪", "en": "🇬🇧", "ru": "🇷🇺"}.get(lead.language or "", "🌐") if lead else "🌐"
    name = (lead.contact_name or "Unknown") if lead else "Unknown"
    score = (lead.score or "cold") if lead else "cold"
    score_color = {"hot": "#e74c3c", "warm": "#f39c12", "cold": "#3498db"}.get(score, "#999")

    bubbles = ""
    for m in messages:
        is_user = m.role == "user"
        bg = "#DCF8C6" if is_user else "#fff"
        align = "flex-end" if is_user else "flex-start"
        ts = m.created_at.strftime("%H:%M") if m.created_at else ""
        safe = m.content.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        bubbles += f"""
        <div style="display:flex;justify-content:{align};margin-bottom:4px">
          <div style="max-width:75%;background:{bg};padding:8px 12px;border-radius:10px;
                      font-size:.9rem;white-space:pre-wrap;box-shadow:0 1px 2px rgba(0,0,0,.1)">
            {safe}
            <div style="font-size:.65rem;color:#999;text-align:right;margin-top:2px">{ts}</div>
          </div>
        </div>"""

    viewing_info = ""
    if lead and lead.viewing_booked:
        viewing_info = "✅ " + (getattr(lead, "viewing_time", None) or "booked")

    return f"""<!DOCTYPE html><html><head><meta charset="UTF-8">
<title>Conversation — {name}</title>
<style>
  body{{font-family:-apple-system,sans-serif;background:#ECE5DD;margin:0;padding:0}}
  header{{background:#0B1E3F;color:white;padding:14px 20px;display:flex;align-items:center;gap:12px}}
  header h1{{font-size:1rem;margin:0}}
  .back{{color:#C9962E;text-decoration:none;font-size:.85rem;font-weight:600}}
  .chat{{padding:16px;max-width:600px;margin:0 auto}}
  .meta{{background:white;border-radius:10px;padding:12px 16px;margin-bottom:16px;font-size:.85rem;display:flex;gap:16px;flex-wrap:wrap}}
  .meta span{{color:#555}}
</style>
</head><body>
<header>
  <a href="/dashboard" class="back">← Dashboard</a>
  <h1>{lang_flag} {name} &nbsp;<span style="color:{score_color};font-weight:700">{score.upper()}</span></h1>
</header>
<div class="chat">
  <div class="meta">
    <span>Score: <strong style="color:{score_color}">{score.upper()}</strong></span>
    <span>Messages: <strong>{lead.message_count if lead else 0}</strong></span>
    <span>Phone: <strong>{lead.contact_phone or "—"}</strong></span>
    <span>Intent: <strong>{lead.intent or "—"}</strong></span>
    <span>Budget: <strong>{f"{lead.budget_max:,.0f} {lead.currency or ''}" if lead and lead.budget_max else "—"}</strong></span>
    <span>Viewing: <strong>{viewing_info or "—"}</strong></span>
  </div>
  {bubbles if bubbles else '<p style="text-align:center;color:#888;padding:40px">No messages yet.</p>'}
</div>
</body></html>"""
