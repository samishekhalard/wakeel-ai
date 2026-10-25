"""Tests for buy/rent intent detection and Arabic language quality."""
from __future__ import annotations

import asyncio
import re
import pytest

from app.agent import correct_intent, run_agent_turn
from app.config import settings


# ── Unit tests: correct_intent() ─────────────────────────────────────────────
# These never call the LLM — they test the keyword safety-net directly.

class TestCorrectIntent:
    # ── English ──────────────────────────────────────────────────────────────

    def test_en_buy(self):
        assert correct_intent("I want to buy an apartment", None) == "buy"

    def test_en_buy_investment(self):
        assert correct_intent("I want to buy for investment purposes", "rent") == "buy"

    def test_en_purchase(self):
        assert correct_intent("Looking to purchase a villa", None) == "buy"

    def test_en_rent(self):
        assert correct_intent("I want to rent a flat in Dubai", None) == "rent"

    def test_en_lease(self):
        assert correct_intent("Looking to lease an apartment", "buy") == "rent"

    # ── Russian ───────────────────────────────────────────────────────────────

    def test_ru_kupit(self):
        assert correct_intent("хочу купить квартиру в Дубае", None) == "buy"

    def test_ru_kupit_investitsii(self):
        """The original bug: 'купить под инвестиции' must be buy."""
        assert correct_intent(
            "хочу купить квартиру в Дубае под инвестиции, бюджет 500 тысяч долларов",
            "rent",   # simulate LLM misclassification
        ) == "buy"

    def test_ru_kupit_budget_usd(self):
        """Large USD budget shouldn't flip buy → rent."""
        assert correct_intent(
            "купить апартаменты, бюджет 2 миллиона долларов",
            "rent",
        ) == "buy"

    def test_ru_snyat(self):
        assert correct_intent("хочу снять квартиру", None) == "rent"

    def test_ru_arenda(self):
        assert correct_intent("интересует аренда в Dubai Marina", "buy") == "rent"

    def test_ru_arenda_long(self):
        assert correct_intent("ищу квартиру в аренду на год", None) == "rent"

    # ── Arabic ────────────────────────────────────────────────────────────────

    def test_ar_ishtari(self):
        assert correct_intent("أبي أشتري شقة في دبي", None) == "buy"

    def test_ar_shira(self):
        assert correct_intent("أريد الشراء في الرياض", "rent") == "buy"

    def test_ar_istajar(self):
        assert correct_intent("أبي أستأجر فيلا في دبي", None) == "rent"

    def test_ar_ijaar(self):
        assert correct_intent("أبحث عن إيجار في دبي ماريناً", "buy") == "rent"

    # ── Edge cases ────────────────────────────────────────────────────────────

    def test_ambiguous_returns_extracted(self):
        """When no keyword matches, trust the LLM."""
        assert correct_intent("I need something in Dubai", "buy") == "buy"

    def test_ambiguous_returns_none(self):
        assert correct_intent("tell me about Dubai properties", None) is None

    def test_both_keywords_returns_extracted(self):
        """Ambiguous message with both buy and rent → trust extracted."""
        assert correct_intent("buy or rent, not sure yet", "buy") == "buy"

    def test_empty_message(self):
        assert correct_intent("", "rent") == "rent"


# ── Integration tests: real LLM replies ──────────────────────────────────────
# Skipped when no real API key is configured.

from app.agent import _groq_available, _openrouter_available

_HAS_LLM = _groq_available() or _openrouter_available()
_SKIP_NO_LLM = pytest.mark.skipif(not _HAS_LLM, reason="No real LLM API key configured")


@_SKIP_NO_LLM
@pytest.mark.asyncio
async def test_ru_investment_buy_end_to_end():
    """The exact failing message: купить + инвестиции must extract intent=buy."""
    _, ctx = await run_agent_turn([{
        "role": "user",
        "content": "хочу купить квартиру в Дубае под инвестиции, бюджет 500 тысяч долларов",
    }])
    assert ctx.qualification.get("intent") == "buy", (
        f"Expected buy, got {ctx.qualification.get('intent')!r}. "
        f"Full qualification: {ctx.qualification}"
    )


@_SKIP_NO_LLM
@pytest.mark.asyncio
async def test_ar_buy_intent_end_to_end():
    _, ctx = await run_agent_turn([{
        "role": "user",
        "content": "أبي أشتري شقة في دبي ماريناً",
    }])
    assert ctx.qualification.get("intent") == "buy"


@_SKIP_NO_LLM
@pytest.mark.asyncio
async def test_ar_rent_intent_end_to_end():
    _, ctx = await run_agent_turn([{
        "role": "user",
        "content": "أبي أستأجر فيلا في الرياض",
    }])
    assert ctx.qualification.get("intent") == "rent"


@_SKIP_NO_LLM
@pytest.mark.asyncio
async def test_en_buy_intent_end_to_end():
    _, ctx = await run_agent_turn([{
        "role": "user",
        "content": "I want to buy a 2-bedroom apartment in Downtown Dubai",
    }])
    assert ctx.qualification.get("intent") == "buy"


@_SKIP_NO_LLM
@pytest.mark.asyncio
async def test_en_rent_intent_end_to_end():
    _, ctx = await run_agent_turn([{
        "role": "user",
        "content": "Looking to rent a villa in Dubai for my family",
    }])
    assert ctx.qualification.get("intent") == "rent"


# ── Arabic language quality ───────────────────────────────────────────────────

# Known-allowed Latin tokens in Arabic replies: area names, currencies, IDs
_ARABIC_ALLOWED = re.compile(
    r"\b("
    r"Dubai|Abu\s*Dhabi|Riyadh|Marina|JBR|DIFC|JVC|KAFD|"
    r"Downtown|Business\s*Bay|Palm\s*Jumeirah|Bluewaters|Creek|"
    r"Arabian\s*Ranches|Emirates\s*Hills|Jumeirah|Mirdif|"
    r"Al\s*Barsha|Al\s*Furjan|The\s*Springs|Motor\s*City|"
    r"Al\s*Olaya|Al\s*Nakheel|Al\s*Malqa|Al\s*Yasmin|Hittin|"
    r"AED|SAR|[A-Z]{3}\d{3}|ID|OK"
    r")\b",
    re.IGNORECASE,
)


@_SKIP_NO_LLM
@pytest.mark.asyncio
async def test_arabic_reply_no_english_words():
    """Arabic reply must contain no English words except area names, currencies, IDs."""
    reply, _ = await run_agent_turn(
        [{"role": "user", "content": "أبي أستأجر شقة في دبي ماريناً"}],
        language="ar",
    )
    assert reply, "Expected a non-empty reply"

    cleaned = _ARABIC_ALLOWED.sub("", reply)
    # Also strip punctuation and whitespace before checking
    latin_words = re.findall(r"[a-zA-Z]{2,}", cleaned)
    assert not latin_words, (
        f"Arabic reply contains English words: {latin_words}\n"
        f"Full reply: {reply!r}"
    )


@_SKIP_NO_LLM
@pytest.mark.asyncio
async def test_arabic_reply_contains_gulf_phrases():
    """Arabic reply should use Gulf dialect phrases."""
    reply, _ = await run_agent_turn(
        [{"role": "user", "content": "مرحبا، أبي أشتري فيلا في دبي"}],
        language="ar",
    )
    gulf_markers = ["وش", "بغيت", "تمام", "أبشر", "حياك", "هلا", "إن شاء الله", "ما شاء"]
    has_gulf = any(phrase in reply for phrase in gulf_markers)
    assert has_gulf, (
        f"Reply doesn't contain any Gulf dialect marker.\n"
        f"Expected one of: {gulf_markers}\n"
        f"Got: {reply!r}"
    )
