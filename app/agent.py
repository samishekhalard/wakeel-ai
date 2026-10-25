"""LLM agent: language detection, qualification flow, listing recommendations."""
from __future__ import annotations

import asyncio
import json
import logging
import re
from typing import Any
import httpx

from app.config import settings
from app.listings import match_listings, format_listing_for_chat

log = logging.getLogger(__name__)

GROQ_BASE_URL = "https://api.groq.com/openai/v1"
# OpenRouter fallback model – known-working with tool calls
_OR_FALLBACK_MODEL = "nvidia/nemotron-3.5-lightning:free"

# ── System prompt ─────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are Wakeel, an AI real estate assistant for a premium agency operating in Dubai and Riyadh.

════════════════════════════════════════════════════════════════
LANGUAGE RULES — MANDATORY, NO EXCEPTIONS
════════════════════════════════════════════════════════════════

Detect the user's language from their FIRST message. Reply in that language for the ENTIRE conversation.

ENGLISH: friendly, professional, brief.

RUSSIAN: natural, warm. Use "ты" form. Be concise.

GULF ARABIC — STRICTEST RULES:
  • Reply 100% in Gulf Arabic (خليجي). NO English words anywhere in your reply.
  • The ONLY allowed Latin text: area names (Dubai Marina, JBR, Downtown Dubai,
    Business Bay, Palm Jumeirah, DIFC, JVC, KAFD), currencies (AED, SAR),
    listing IDs (DXB001, RYD001, etc.).
  • WRONG: "budget وش بتدور عليها"   RIGHT: "وش الميزانية اللي بالك؟"
  • WRONG: "bedrooms كم؟"             RIGHT: "كم غرفة تبيها؟"
  • WRONG: "property type إيش؟"       RIGHT: "وش نوع العقار؟"
  • WRONG: "cash أو mortgage؟"        RIGHT: "تدفع كاش ولا تقسيط؟"
  • WRONG: "timeline متى؟"            RIGHT: "متى تبي تنتقل؟"
  • Use Gulf phrases naturally: أبشر، حياك، تمام، وش رايك، ما شاء الله،
    بغيت، كيف أقدر أساعدك، إن شاء الله، هلا وغلا
  • Numbers in Arabic words when natural: واحدة، ثنتين، ثلاث، أربع، خمس

════════════════════════════════════════════════════════════════
INTENT DETECTION (BUY vs RENT) — READ CAREFULLY
════════════════════════════════════════════════════════════════

When calling extract_qualification, set intent correctly:
  BUY  = купить / покупка / куплю / purchase / invest / شراء / أشتري / تملك
  RENT = снять / аренда / арендовать / rent / lease / إيجار / أستأجر / استئجار

"I want to buy for investment" → intent = "buy"  (investment purchase, not rental)
"хочу купить под инвестиции"  → intent = "buy"  (купить = buy, always)
"أبي أشتري"                   → intent = "buy"
"أبي أستأجر"                  → intent = "rent"

════════════════════════════════════════════════════════════════
PERSONALITY
════════════════════════════════════════════════════════════════

Short, human, warm. Max 3-4 sentences unless listing details.
No corporate jargon. Sound like a helpful friend who knows real estate.
Use 1-2 emojis max to feel warm, not spammy.

════════════════════════════════════════════════════════════════
QUALIFICATION GOAL
════════════════════════════════════════════════════════════════

Extract these fields through natural conversation (1-2 questions at a time):
1. intent: buy or rent
2. budget (min/max, currency AED or SAR)
3. city: Dubai or Riyadh
4. area/neighborhood preference
5. property_type: apartment / villa / townhouse
6. bedrooms
7. payment_method: cash or mortgage
8. timeline: when they want to move/buy

LISTING RECOMMENDATIONS:
When you have: intent + budget + city + (bedrooms or type), call recommend_listings.
Present listings naturally. Offer to arrange a viewing.

HANDOFF:
If user asks for human agent, says "call me", gives phone/name, or lead is very hot:
call request_handoff. If user wants to book a viewing: call book_viewing.

EXTRACT DATA:
After EVERY user message, call extract_qualification with whatever you learned.

IMPORTANT: Never invent listings. Only show listings from recommend_listings results."""


# ── LLM tool definitions ──────────────────────────────────────────────────────

def _ns(schema: dict) -> dict:
    """Make a JSON Schema field nullable — needed for Groq's strict validator."""
    t = schema.get("type")
    if t and isinstance(t, str):
        return {**schema, "type": [t, "null"]}
    return schema


TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "extract_qualification",
            "description": (
                "Save qualification data from the conversation. Call after every user message. "
                "Only include fields you have actual data for — omit fields you don't know yet."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "intent": {
                        "type": ["string", "null"],
                        "enum": ["buy", "rent", None],
                        "description": (
                            "buy or rent. "
                            "BUY: купить/покупка/куплю/invest/purchase/شراء/أشتري/تملك. "
                            "RENT: снять/аренда/арендовать/rent/lease/إيجار/أستأجر. "
                            "'купить под инвестиции'=buy. 'أبي أشتري'=buy. 'أبي أستأجر'=rent."
                        ),
                    },
                    "budget_min": _ns({"type": "number", "description": "Minimum budget (number only)"}),
                    "budget_max": _ns({"type": "number", "description": "Maximum budget (number only)"}),
                    "currency": {
                        "type": ["string", "null"],
                        "enum": ["AED", "SAR", None],
                        "description": "AED for Dubai, SAR for Riyadh",
                    },
                    "city": _ns({"type": "string", "description": "Dubai or Riyadh"}),
                    "area": _ns({"type": "string", "description": "Neighborhood or area name"}),
                    "property_type": _ns({"type": "string", "description": "apartment, villa, or townhouse"}),
                    "bedrooms": _ns({"type": "integer", "description": "Number of bedrooms"}),
                    "payment_method": {
                        "type": ["string", "null"],
                        "enum": ["cash", "mortgage", None],
                        "description": "cash or mortgage",
                    },
                    "timeline": _ns({"type": "string", "description": "When they want to move or buy"}),
                    "contact_name": _ns({"type": "string", "description": "Lead name if given"}),
                    "contact_phone": _ns({"type": "string", "description": "Lead phone if given"}),
                    "language": {
                        "type": ["string", "null"],
                        "enum": ["ar", "en", "ru", None],
                        "description": "Detected language code",
                    },
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "recommend_listings",
            "description": "Search for matching property listings. Call when you have intent + budget + city.",
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
            "description": "Mark lead for immediate human agent follow-up.",
            "parameters": {
                "type": "object",
                "properties": {
                    "reason": {"type": "string"},
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
                    "listing_id": {"type": "string"},
                    "preferred_time": {"type": "string"},
                },
                "required": [],
            },
        },
    },
]


# ── Error type ────────────────────────────────────────────────────────────────

class LLMError(Exception):
    def __init__(self, status: int, message: str):
        self.status = status
        self.message = message
        super().__init__(f"LLM {status}: {message}")


# ── Intent safety-net ─────────────────────────────────────────────────────────

_BUY_WORDS = frozenset([
    "buy", "purchase", "buying", "invest", "investment",
    "купить", "куплю", "покупк", "покупаю", "инвестиц",
    "شراء", "أشتري", "اشتري", "تملك", "امتلاك", "ملكية",
])
_RENT_WORDS = frozenset([
    "rent", "rental", "lease", "renting",
    "снять", "аренд", "арендов", "снимаю", "снимать",
    "إيجار", "أستأجر", "استئجار", "استأجر", "أجار",
])


def correct_intent(user_msg: str, extracted: str | None) -> str | None:
    """Validate extracted intent against keywords in the raw user message.

    Returns the keyword-derived intent when unambiguous, otherwise trusts
    whatever the LLM extracted (or None if nothing extracted yet).
    """
    low = user_msg.lower()
    is_buy  = any(w in low for w in _BUY_WORDS)
    is_rent = any(w in low for w in _RENT_WORDS)
    if is_buy and not is_rent:
        return "buy"
    if is_rent and not is_buy:
        return "rent"
    return extracted   # ambiguous or no keywords → trust the LLM


# ── LLM client ────────────────────────────────────────────────────────────────

async def _call_provider(
    messages: list[dict],
    base_url: str,
    api_key: str,
    model: str,
    extra_headers: dict | None = None,
) -> dict[str, Any]:
    headers: dict[str, str] = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    if extra_headers:
        headers.update(extra_headers)

    payload = {
        "model": model,
        "messages": messages,
        "tools": TOOLS,
        "tool_choice": "auto",
        "temperature": 0.7,
        "max_tokens": 600,
    }

    async with httpx.AsyncClient(timeout=60) as client:
        resp = await client.post(
            f"{base_url.rstrip('/')}/chat/completions",
            headers=headers,
            json=payload,
        )

    if not resp.is_success:
        try:
            body = resp.json()
        except Exception:
            body = {}
        err_msg = body.get("error", {}).get("message", resp.text[:200])
        log.error("LLM %s — %s [%s] — %s", resp.status_code, base_url, model, err_msg)
        raise LLMError(resp.status_code, err_msg)

    return resp.json()


_GROQ_PLACEHOLDERS = frozenset(["", "your_groq_key_here", "gsk_placeholder"])


def _resolve_groq_key() -> str:
    """Return a usable Groq key, or empty string if none available.

    Checks GROQ_API_KEY first; if empty, falls back to LLM_API_KEY when it
    looks like a Groq key (starts with 'gsk_'), so users who paste a Groq key
    into the wrong field still get correct routing.
    """
    if settings.groq_api_key and settings.groq_api_key not in _GROQ_PLACEHOLDERS:
        return settings.groq_api_key
    if settings.llm_api_key.startswith("gsk_"):
        return settings.llm_api_key
    return ""


def _groq_available() -> bool:
    return bool(_resolve_groq_key())


def _openrouter_available() -> bool:
    """True only when LLM_API_KEY is a real OpenRouter key (not a Groq key)."""
    k = settings.llm_api_key
    return bool(k and k not in ("sk-placeholder", "") and not k.startswith("gsk_"))


def _parse_retry_after(err_msg: str) -> float:
    """Extract seconds from Groq's 429 message: 'Please try again in 6.0075s'."""
    m = re.search(r"try again in ([0-9.]+)s", err_msg)
    return min(float(m.group(1)) if m else 15.0, 30.0)


async def call_llm(messages: list[dict]) -> dict[str, Any]:
    """Try Groq first; fall back to OpenRouter on 429/5xx.
    On 429 with no OpenRouter fallback, retry up to 2× with backoff.
    """
    groq_key = _resolve_groq_key()
    if groq_key:
        for attempt in range(3):
            try:
                log.debug("Using Groq model=%s (attempt %d)", settings.groq_model, attempt + 1)
                return await _call_provider(
                    messages, GROQ_BASE_URL, groq_key, settings.groq_model
                )
            except LLMError as exc:
                if exc.status not in (429, 500, 502, 503, 529):
                    raise  # auth/model errors surface immediately
                if _openrouter_available():
                    log.warning("Groq %s — falling back to OpenRouter", exc.status)
                    break  # exit retry loop, use OpenRouter below
                if attempt < 2:
                    wait = _parse_retry_after(exc.message) if exc.status == 429 else 10.0
                    log.warning("Groq %s — no OR fallback, retrying in %.1fs", exc.status, wait)
                    await asyncio.sleep(wait)
                else:
                    raise  # retries exhausted, no fallback

    if not _openrouter_available():
        raise LLMError(503, "No LLM provider available — set GROQ_API_KEY or LLM_API_KEY")

    log.debug("Using OpenRouter model=%s", settings.llm_model)
    or_headers = {"HTTP-Referer": "https://wakeel.ai", "X-Title": "Wakeel AI"}
    return await _call_provider(
        messages,
        settings.llm_base_url,
        settings.llm_api_key,
        settings.llm_model,
        extra_headers=or_headers,
    )


# ── Tool execution ────────────────────────────────────────────────────────────

class AgentContext:
    """Per-turn extracted data and side-effect flags."""

    def __init__(self):
        self.qualification: dict[str, Any] = {}
        self.listings: list[dict] = []
        self.handoff: bool = False
        self.viewing_booked: bool = False
        self.language: str = "en"


_KNOWN_TOOLS = frozenset(["extract_qualification", "recommend_listings", "request_handoff", "book_viewing"])


def _sanitize_tool_name(raw: str) -> str:
    """Strip hallucinated suffixes that models sometimes append (e.g. '<|channel|>commentary')."""
    if raw in _KNOWN_TOOLS:
        return raw
    if "<|" in raw:
        return raw.split("<|")[0].rstrip("_- ")
    return raw


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
            msgs = {
                "ar": "ما لقيت عقارات مطابقة. خبرني أكثر عن اللي تبيه.",
                "ru": "Подходящих объектов не найдено. Расскажите подробнее.",
                "en": "No matching listings found. Let me ask a few more questions.",
            }
            return msgs.get(language, msgs["en"])
        return json.dumps(
            [format_listing_for_chat(l, language) for l in results],
            ensure_ascii=False,
        )

    if name == "request_handoff":
        context.handoff = True
        return "handoff_noted"

    if name == "book_viewing":
        context.viewing_booked = True
        return "viewing_booked"

    return "unknown_tool"


# ── Main agent turn ───────────────────────────────────────────────────────────

def _apply_intent_correction(conversation: list[dict], context: AgentContext) -> None:
    """Keyword safety-net: correct buy/rent misclassification after tool calls complete."""
    if not conversation or "intent" not in context.qualification:
        return
    last_user = next(
        (m["content"] for m in reversed(conversation) if m["role"] == "user"), ""
    )
    original  = context.qualification["intent"]
    corrected = correct_intent(last_user, original)
    if corrected != original:
        log.warning("Intent corrected %r → %r for: %r", original, corrected, last_user[:80])
        context.qualification["intent"] = corrected


async def run_agent_turn(
    conversation: list[dict],
    language: str = "en",
) -> tuple[str, AgentContext]:
    """Run one agent turn. Returns (reply_text, context).

    conversation: full message history [{"role": ..., "content": ...}]
    """
    context = AgentContext()
    context.language = language
    messages = [{"role": "system", "content": SYSTEM_PROMPT}] + conversation

    _err_msgs = {
        "ar": "عذراً، حدث خطأ مؤقت. حاول مجدداً بعد لحظة.",
        "ru": "Извините, временная ошибка. Попробуйте ещё раз.",
        "en": "Sorry, something went wrong. Please try again in a moment.",
    }

    reply: str | None = None

    try:
        for _ in range(5):   # max tool-call depth
            response   = await call_llm(messages)
            choice     = response["choices"][0]
            message    = choice["message"]
            tool_calls = message.get("tool_calls") or []

            if not tool_calls:
                reply = message.get("content") or ""
                break

            # Sanitize names before history append — prevents cascading 400s
            for tc in tool_calls:
                tc["function"]["name"] = _sanitize_tool_name(tc["function"]["name"])
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
        suffix = f"\n\n[DEBUG: LLM {exc.status} — {exc.message}]" if settings.debug else ""
        return _err_msgs.get(language, _err_msgs["en"]) + suffix, context
    except Exception as exc:
        log.exception("Unexpected error in agent turn")
        suffix = f"\n\n[DEBUG: {type(exc).__name__}: {exc}]" if settings.debug else ""
        return _err_msgs.get(language, _err_msgs["en"]) + suffix, context

    # Apply keyword safety-net after all tool calls settle
    _apply_intent_correction(conversation, context)

    if reply is not None:
        return reply, context
    return _err_msgs.get(language, _err_msgs["en"]), context
