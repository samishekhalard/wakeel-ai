"""Unit tests for LLM provider fallback logic in call_llm."""
from __future__ import annotations

import pytest
import unittest.mock as mock

from app.agent import LLMError, call_llm, _is_daily_limit


# ── _is_daily_limit helper ────────────────────────────────────────────────────

class TestIsDailyLimit:
    def test_tpd_keyword(self):
        assert _is_daily_limit("Rate limit reached on tokens per day (TPD)")

    def test_per_day_keyword(self):
        assert _is_daily_limit("Limit 200000 per_day exceeded")

    def test_tokens_per_day(self):
        assert _is_daily_limit("Used 200000 tokens per day, limit reached")

    def test_tpm_is_not_daily(self):
        assert not _is_daily_limit("Rate limit on tokens per minute (TPM): try again in 5s")

    def test_empty_message(self):
        assert not _is_daily_limit("")


# ── call_llm fallback paths ───────────────────────────────────────────────────

_OK_RESPONSE = {"choices": [{"message": {"content": "ok", "tool_calls": None}}]}


@pytest.mark.asyncio
async def test_groq_success_no_fallback_needed():
    """Happy path — Groq succeeds, OpenRouter is never called."""
    async def provider(messages, base_url, api_key, model, extra_headers=None):
        return _OK_RESPONSE

    with mock.patch("app.agent._call_provider", side_effect=provider), \
         mock.patch("app.agent._resolve_groq_key", return_value="gsk_fake"), \
         mock.patch("app.agent._openrouter_available", return_value=False):
        result = await call_llm([{"role": "user", "content": "hi"}])

    assert result["choices"][0]["message"]["content"] == "ok"


@pytest.mark.asyncio
async def test_groq_429_immediately_falls_back_to_openrouter():
    """Groq 429 with OR configured → OR called on the very next attempt, no wait."""
    calls: list[str] = []

    async def provider(messages, base_url, api_key, model, extra_headers=None):
        calls.append(base_url)
        if "groq" in base_url:
            raise LLMError(429, "Rate limit reached. Please try again in 5s.")
        return _OK_RESPONSE

    with mock.patch("app.agent._call_provider", side_effect=provider), \
         mock.patch("app.agent._resolve_groq_key", return_value="gsk_fake"), \
         mock.patch("app.agent._openrouter_available", return_value=True), \
         mock.patch("app.agent.asyncio.sleep") as sleep_mock:
        result = await call_llm([{"role": "user", "content": "hi"}])

    assert result["choices"][0]["message"]["content"] == "ok"
    assert len(calls) == 2
    assert "groq" in calls[0]
    assert "groq" not in calls[1]
    sleep_mock.assert_not_called()  # no wait before OR


@pytest.mark.asyncio
async def test_groq_5xx_immediately_falls_back_to_openrouter():
    """Groq 503 with OR configured → OR used immediately."""
    calls: list[str] = []

    async def provider(messages, base_url, api_key, model, extra_headers=None):
        calls.append(base_url)
        if "groq" in base_url:
            raise LLMError(503, "Service unavailable")
        return _OK_RESPONSE

    with mock.patch("app.agent._call_provider", side_effect=provider), \
         mock.patch("app.agent._resolve_groq_key", return_value="gsk_fake"), \
         mock.patch("app.agent._openrouter_available", return_value=True), \
         mock.patch("app.agent.asyncio.sleep") as sleep_mock:
        result = await call_llm([{"role": "user", "content": "hi"}])

    assert result["choices"][0]["message"]["content"] == "ok"
    sleep_mock.assert_not_called()


@pytest.mark.asyncio
async def test_groq_429_no_fallback_retries_once_with_backoff():
    """Groq 429, no OR configured → retry Groq once after backoff."""
    calls: list[str] = []

    async def provider(messages, base_url, api_key, model, extra_headers=None):
        calls.append("call")
        if len(calls) == 1:
            raise LLMError(429, "Rate limit. Please try again in 0.1s.")
        return _OK_RESPONSE

    with mock.patch("app.agent._call_provider", side_effect=provider), \
         mock.patch("app.agent._resolve_groq_key", return_value="gsk_fake"), \
         mock.patch("app.agent._openrouter_available", return_value=False), \
         mock.patch("app.agent.asyncio.sleep") as sleep_mock:
        result = await call_llm([{"role": "user", "content": "hi"}])

    assert result["choices"][0]["message"]["content"] == "ok"
    assert len(calls) == 2
    sleep_mock.assert_called_once()


@pytest.mark.asyncio
async def test_groq_daily_limit_no_fallback_fails_immediately():
    """Groq daily quota exhausted, no OR → raise immediately without retry."""
    calls: list[str] = []

    async def provider(messages, base_url, api_key, model, extra_headers=None):
        calls.append("call")
        raise LLMError(429, "Limit 200000 per_day exceeded. Upgrade for more.")

    with mock.patch("app.agent._call_provider", side_effect=provider), \
         mock.patch("app.agent._resolve_groq_key", return_value="gsk_fake"), \
         mock.patch("app.agent._openrouter_available", return_value=False), \
         mock.patch("app.agent.asyncio.sleep") as sleep_mock:
        with pytest.raises(LLMError) as exc_info:
            await call_llm([{"role": "user", "content": "hi"}])

    assert exc_info.value.status == 429
    assert len(calls) == 1          # no retry
    sleep_mock.assert_not_called()  # no wait


@pytest.mark.asyncio
async def test_groq_daily_limit_with_openrouter_falls_back():
    """Groq daily quota exhausted but OR is available → OR used immediately."""
    calls: list[str] = []

    async def provider(messages, base_url, api_key, model, extra_headers=None):
        calls.append(base_url)
        if "groq" in base_url:
            raise LLMError(429, "tokens per day limit exceeded")
        return _OK_RESPONSE

    with mock.patch("app.agent._call_provider", side_effect=provider), \
         mock.patch("app.agent._resolve_groq_key", return_value="gsk_fake"), \
         mock.patch("app.agent._openrouter_available", return_value=True), \
         mock.patch("app.agent.asyncio.sleep") as sleep_mock:
        result = await call_llm([{"role": "user", "content": "hi"}])

    assert result["choices"][0]["message"]["content"] == "ok"
    assert len(calls) == 2
    sleep_mock.assert_not_called()


@pytest.mark.asyncio
async def test_both_providers_fail_raises_last_error():
    """Both Groq and OR fail → raises LLMError."""
    async def provider(messages, base_url, api_key, model, extra_headers=None):
        raise LLMError(429, "Rate limit everywhere")

    with mock.patch("app.agent._call_provider", side_effect=provider), \
         mock.patch("app.agent._resolve_groq_key", return_value="gsk_fake"), \
         mock.patch("app.agent._openrouter_available", return_value=True):
        with pytest.raises(LLMError) as exc_info:
            await call_llm([{"role": "user", "content": "hi"}])

    assert exc_info.value.status == 429


@pytest.mark.asyncio
async def test_groq_auth_error_surfaces_immediately():
    """Groq 401 must not be retried or redirected to OR — surface immediately."""
    calls: list[str] = []

    async def provider(messages, base_url, api_key, model, extra_headers=None):
        calls.append("call")
        raise LLMError(401, "Invalid API key")

    with mock.patch("app.agent._call_provider", side_effect=provider), \
         mock.patch("app.agent._resolve_groq_key", return_value="gsk_fake"), \
         mock.patch("app.agent._openrouter_available", return_value=True):
        with pytest.raises(LLMError) as exc_info:
            await call_llm([{"role": "user", "content": "hi"}])

    assert exc_info.value.status == 401
    assert len(calls) == 1  # no fallback for auth errors


@pytest.mark.asyncio
async def test_no_providers_configured_raises_503():
    """No Groq key and no OR key → clean 503 error."""
    with mock.patch("app.agent._resolve_groq_key", return_value=""), \
         mock.patch("app.agent._openrouter_available", return_value=False):
        with pytest.raises(LLMError) as exc_info:
            await call_llm([{"role": "user", "content": "hi"}])

    assert exc_info.value.status == 503
