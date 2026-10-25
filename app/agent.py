"""LLM agent: language detection, qualification flow, listing recommendations."""
from __future__ import annotations

import json
import logging
from typing import Any
import httpx

from app.config import settings
from app.listings import match_listings, format_listing_for_chat

log = logging.getLogger(__name__)


# ── System prompt ────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are Wakeel, an AI real estate assistant for a premium agency operating in Dubai and Riyadh.

LANGUAGE RULES:
- Detect the user's language from their first message.
- Always reply in EXACTLY the same language: Gulf Arabic (خليجي dialect), English, or Russian.
- Gulf Arabic: use casual Gulf dialect (وش, بغيت, عندنا, شلون, إيش). Never use formal MSA.
- Russian: natural, warm, professional tone.
- English: friendly, professional, brief.

PERSONALITY:
- Short, human, warm. Max 3-4 sentences per reply unless listing details.
- No corporate jargon. Sound like a helpful friend who knows real estate.
- Use occasional emojis (1-2 max) to feel warm, not spammy.

QUALIFICATION GOAL — extract these fields through natural conversation:
1. intent: buy or rent
2. budget (min/max, currency AED or SAR)
3. city: Dubai or Riyadh
4. area/neighborhood preference
5. property_type: apartment / villa / townhouse
6. bedrooms
7. payment_method: cash or mortgage
8. timeline: when they want to move/buy

Extract info progressively — don't ask all questions at once. Ask 1-2 questions per message.

LISTING RECOMMENDATIONS:
When you have enough info (at least: intent + budget + city + bedrooms or type),
call the function `recommend_listings` to get matches.
Present listings naturally, one by one, briefly. Offer to arrange a viewing.

HANDOFF:
- If the user asks for a human agent, says "call me", provides phone/name, or the lead is very hot:
  call the function `request_handoff`.
- If the user wants to book a viewing: call `book_viewing`.

EXTRACT DATA:
After each user message, call `extract_qualification` with whatever new data you just learned.
This updates the lead record silently in the background.

IMPORTANT: Never make up listings. Only present listings when you call recommend_listings and get results.
Never mention competitor agencies. Focus only on properties in Dubai and Riyadh."""


# ── LLM tools (function definitions) ─────────────────────────────────────────

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "extract_qualification",
            "description": "Save qualification data extracted from the conversation. Call after every user message with whatever you learned.",
            "parameters": {
                "type": "object",
                "properties": {
                    "intent": {"type": "string", "enum": ["buy", "rent"], "description": "buy or rent"},
                    "budget_min": {"type": "number", "description": "Minimum budget"},
                    "budget_max": {"type": "number", "description": "Maximum budget"},
                    "currency": {"type": "string", "enum": ["AED", "SAR"], "description": "Currency"},
                    "city": {"type": "string", "description": "Dubai or Riyadh"},
                    "area": {"type": "string", "description": "Neighborhood or area"},
                    "property_type": {"type": "string", "description": "apartment, villa, townhouse, etc."},
                    "bedrooms": {"type": "integer", "description": "Number of bedrooms"},
                    "payment_method": {"type": "string", "enum": ["cash", "mortgage"], "description": "Payment method"},
                    "timeline": {"type": "string", "description": "When they want to move or buy"},
                    "contact_name": {"type": "string", "description": "Lead's name if given"},
                    "contact_phone": {"type": "string", "description": "Lead's phone if given"},
                    "language": {"type": "string", "description": "Detected language: ar, en, ru"},
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "recommend_listings",
            "description": "Search for matching property listings to recommend to the user.",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string"},
                    "intent": {"type": "string", "enum": ["buy", "rent"]},
                    "budget_max": {"type": "number"},
                    "area": {"type": "string"},
                    "property_type": {"type": "string"},
                    "bedrooms": {"type": "integer"},
                    "payment_method": {"type": "string"},
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "request_handoff",
            "description": "Mark this lead for immediate human agent follow-up.",
            "parameters": {
                "type": "object",
                "properties": {
                    "reason": {"type": "string", "description": "Why handoff is needed"},
                },
                "required": ["reason"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "book_viewing",
            "description": "Record that the user wants to book a property viewing.",
            "parameters": {
                "type": "object",
                "properties": {
                    "listing_id": {"type": "string", "description": "ID of the listing"},
                    "preferred_time": {"type": "string", "description": "When they want to view"},
                },
                "required": [],
            },
        },
    },
]


# ── LLM client ────────────────────────────────────────────────────────────────

async def call_llm(messages: list[dict]) -> dict[str, Any]:
    """Call the configured OpenAI-compatible LLM API."""
    async with httpx.AsyncClient(timeout=60) as client:
        headers = {
            "Authorization": f"Bearer {settings.llm_api_key}",
            "Content-Type": "application/json",
        }
        if "openrouter" in settings.llm_base_url:
            headers["HTTP-Referer"] = "https://wakeel.ai"
            headers["X-Title"] = "Wakeel AI"

        payload = {
            "model": settings.llm_model,
            "messages": messages,
            "tools": TOOLS,
            "tool_choice": "auto",
            "temperature": 0.7,
            "max_tokens": 600,
        }

        resp = await client.post(
            f"{settings.llm_base_url.rstrip('/')}/chat/completions",
            headers=headers,
            json=payload,
        )
        if not resp.is_success:
            body = resp.json() if resp.headers.get("content-type", "").startswith("application/json") else {}
            err_msg = body.get("error", {}).get("message", resp.text[:200])
            log.error("LLM %s — model=%s — %s", resp.status_code, settings.llm_model, err_msg)
            raise LLMError(resp.status_code, err_msg)
        return resp.json()


# ── Error type ───────────────────────────────────────────────────────────────

class LLMError(Exception):
    def __init__(self, status: int, message: str):
        self.status = status
        self.message = message
        super().__init__(f"LLM {status}: {message}")


# ── Tool execution ────────────────────────────────────────────────────────────

class AgentContext:
    """Holds per-turn extracted data and events."""

    def __init__(self):
        self.qualification: dict[str, Any] = {}
        self.listings: list[dict] = []
        self.handoff: bool = False
        self.viewing_booked: bool = False
        self.language: str = "en"


def _execute_tool(name: str, args: dict, context: AgentContext, language: str) -> str:
    if name == "extract_qualification":
        context.qualification.update({k: v for k, v in args.items() if v is not None})
        if "language" in args:
            context.language = args["language"]
        return "ok"

    if name == "recommend_listings":
        results = match_listings(
            city=args.get("city"),
            intent=args.get("intent"),
            budget_max=args.get("budget_max"),
            area=args.get("area"),
            property_type=args.get("property_type"),
            bedrooms=args.get("bedrooms"),
            payment_method=args.get("payment_method"),
        )
        context.listings = results
        if not results:
            if language == "ar":
                return "لا توجد عقارات متطابقة حالياً. أخبرني بالمزيد عن متطلباتك."
            elif language == "ru":
                return "Подходящих объектов не найдено. Расскажите больше о ваших требованиях."
            return "No matching listings found. Let me ask a few more questions to refine the search."
        return json.dumps(
            [format_listing_for_chat(l, language) for l in results],
            ensure_ascii=False
        )

    if name == "request_handoff":
        context.handoff = True
        return "handoff_noted"

    if name == "book_viewing":
        context.viewing_booked = True
        return "viewing_booked"

    return "unknown_tool"


# ── Main agent turn ───────────────────────────────────────────────────────────

async def run_agent_turn(
    conversation: list[dict],
    language: str = "en",
) -> tuple[str, AgentContext]:
    """
    Run one agent turn.
    Returns (reply_text, context_with_extracted_data).
    conversation: full message history [{"role": ..., "content": ...}]
    """
    context = AgentContext()
    context.language = language

    messages = [{"role": "system", "content": SYSTEM_PROMPT}] + conversation

    _generic_error = {
        "ar": "عذراً، حدث خطأ مؤقت. حاول مجدداً بعد لحظة.",
        "ru": "Извините, временная ошибка. Попробуйте ещё раз.",
        "en": "Sorry, something went wrong. Please try again in a moment.",
    }

    try:
        # Agentic loop: keep calling LLM until it returns a text response
        for _ in range(5):  # max 5 tool calls per turn
            response = await call_llm(messages)
            choice = response["choices"][0]
            message = choice["message"]

            tool_calls = message.get("tool_calls") or []

            if not tool_calls:
                content = message.get("content") or ""
                return content, context

            # Execute tool calls and add results to message history
            messages.append(message)
            for tc in tool_calls:
                fn_name = tc["function"]["name"]
                try:
                    fn_args = json.loads(tc["function"]["arguments"])
                except (json.JSONDecodeError, KeyError):
                    fn_args = {}

                result = _execute_tool(fn_name, fn_args, context, language)
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc["id"],
                    "content": result,
                })

    except LLMError as exc:
        debug_suffix = f"\n\n[DEBUG: LLM {exc.status} — {exc.message}]" if settings.debug else ""
        return _generic_error.get(language, _generic_error["en"]) + debug_suffix, context
    except Exception as exc:
        log.exception("Unexpected error in agent turn")
        debug_suffix = f"\n\n[DEBUG: {type(exc).__name__}: {exc}]" if settings.debug else ""
        return _generic_error.get(language, _generic_error["en"]) + debug_suffix, context

    # Fallback if tool loop exhausted without text reply
    return _generic_error.get(language, _generic_error["en"]), context
