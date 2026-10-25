# Wakeel AI — Real Estate Lead Qualification Agent

> Turn website visitors into qualified leads — automatically, in **Gulf Arabic, English, or Russian**.

Wakeel is an AI agent built for real estate agencies in **Dubai** and **Riyadh**. It chats with buyers and renters, extracts qualification data through natural conversation, recommends matching listings, scores leads as hot/warm/cold, and hands off the best prospects to human agents — all without human involvement until the lead is ready.

**Target market:** Dubai and Saudi Arabia agencies, including Russian-speaking agent teams in Dubai.

---

## What it does

- **Auto-detects language** from the first message and replies in Gulf Arabic (خليجي), English, or Russian throughout the conversation
- **Qualifies leads** by asking 1–2 questions at a time: intent, budget, city, area, property type, bedrooms, payment method, timeline
- **Recommends listings** once enough data is collected — fuzzy-matched against 45 real-looking properties (30 Dubai + 15 Riyadh)
- **Scores leads** with a transparent points model: 🔥 Hot / ⚡ Warm / ❄️ Cold
- **Triggers handoff** when a lead gives contact info, asks for a human agent, or books a viewing
- **Dashboard** at `/dashboard` shows all leads, scores, and qualification summaries

---

## Architecture

```
Browser
  WhatsApp-style chat UI (HTML/CSS/JS · mobile-first · RTL Arabic)
        │
        │  POST /api/chat
        ▼
  FastAPI Backend
  ┌────────────────────────────────────────────────────┐
  │  agent.py       LLM tool calling, lang detection  │
  │  listings.py    Fuzzy matching (budget flex ±15%) │
  │  scoring.py     Points-based hot/warm/cold model  │
  │  models.py      SQLAlchemy async (Lead, Message)  │
  └────────────────────────┬───────────────────────────┘
                           │
                     SQLite (aiosqlite)
                           │
        ┌──────────────────┴──────────────────┐
        │  Groq (primary)  OpenRouter (fallback) │
        │  openai/gpt-oss-20b  nemotron-3.5      │
        └────────────────────────────────────────┘
```

**Key design decisions**

| Decision | Rationale |
|----------|-----------|
| Groq primary → OpenRouter fallback | Groq is faster and free; auto-retries with backoff; falls back on 429/5xx |
| LLM tool calling for extraction | Structured JSON extraction beats regex parsing of free text |
| Keyword safety-net for intent | Post-call correction catches model misclassification (e.g. "купить под инвестиции" → buy) |
| SQLite + aiosqlite | Zero infra cost; trivial migration to Postgres for production |
| Vanilla JS frontend | No build step; instant load; works offline for demos |
| Points-based scoring | Transparent and tunable — no ML dependency |

---

## Qualification fields

| Field | Examples |
|-------|---------|
| Intent | buy / rent |
| City | Dubai / Riyadh |
| Area | Dubai Marina, Downtown Dubai, KAFD, Al Olaya |
| Property type | apartment / villa / townhouse |
| Bedrooms | 1–5+ |
| Budget | AED or SAR |
| Payment | cash / mortgage |
| Timeline | immediately / 3 months / 1 year |

### Lead scoring

| Score | Points | Trigger |
|-------|--------|---------|
| 🔥 Hot | ≥ 10 | Full qualification + contact info or viewing booked |
| ⚡ Warm | 5–9 | Partial qualification (intent + budget + city) |
| ❄️ Cold | < 5 | Just started |

---

## Quick start

### Prerequisites

- Python 3.12 (via Homebrew: `brew install python@3.12`)
- A free API key from [Groq](https://console.groq.com/keys) **or** [OpenRouter](https://openrouter.ai/)

### 1. Clone and configure

```bash
git clone https://github.com/samishekhalard/wakeel-ai.git
cd wakeel-ai
cp .env.example .env
```

Open `.env` and set your key:

```env
# Option A — Groq (recommended: fastest, free, best tool calling)
GROQ_API_KEY=gsk_your_key_here

# Option B — OpenRouter (fallback, also free models available)
LLM_API_KEY=sk-or-your_key_here
```

Both providers use free tiers — no credit card needed for basic usage.

### 2. Run locally

```bash
python3.12 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
uvicorn app.main:app --reload
```

- Chat UI: http://localhost:8000
- Dashboard: http://localhost:8000/dashboard

### 3. Run with Docker

```bash
docker-compose up --build
```

---

## Tests

```bash
pytest tests/ -v
```

64 tests covering:
- Listing data integrity and fuzzy matching
- Lead scoring thresholds
- Buy/rent intent detection in all 3 languages (unit + LLM integration)
- Arabic reply quality — no English words except allowed area names
- Startup smoke tests (health, UI, dashboard, empty-message rejection)

Integration tests (those hitting a real LLM) are automatically skipped when no API key is configured.

---

## Project structure

```
wakeel-ai/
├── app/
│   ├── main.py              # FastAPI app, lifespan, routing
│   ├── agent.py             # LLM agent, tool calling, language detection, intent safety-net
│   ├── config.py            # Pydantic v2 settings from .env
│   ├── models.py            # SQLAlchemy async models (Lead, Message)
│   ├── listings.py          # Listing loader + fuzzy matching (budget ±15%, bedrooms ±1)
│   ├── scoring.py           # Points-based lead scoring
│   ├── routers/
│   │   ├── chat.py          # POST /api/chat · GET /api/chat/history/{session_id}
│   │   └── dashboard.py     # GET /dashboard (HTML, score badges)
│   └── static/
│       └── index.html       # Chat UI — WhatsApp-style, RTL Arabic, mobile-first
├── data/
│   ├── dubai_listings.json      # 30 Dubai properties
│   └── riyadh_listings.json     # 15 Riyadh properties
├── tests/
│   ├── test_qualification.py    # Listing matching + scoring
│   ├── test_scoring.py          # Hot/warm/cold point thresholds
│   ├── test_intent_and_language.py  # Intent detection (all 3 langs) + Arabic quality
│   └── test_startup.py          # FastAPI smoke tests
├── .env.example
├── docker-compose.yml
├── Dockerfile
├── requirements.txt
├── DEMO_SCRIPT.md
└── CLAUDE.md
```

---

## Roadmap

| Phase | Feature | Status |
|-------|---------|--------|
| 1 | Web chat demo (this repo) | ✅ Done |
| 2 | WhatsApp Cloud API integration | Planned |
| 3 | n8n workflows + Google Sheets / CRM sync | Planned |
| 4 | Instagram DM integration | Planned |
| 5 | Multi-client SaaS (one deployment, many agencies) | Planned |

---

## Tech stack

| Layer | Technology |
|-------|-----------|
| Backend | Python 3.12, FastAPI, Pydantic v2 |
| Database | SQLite via SQLAlchemy async + aiosqlite |
| LLM | Groq (primary) → OpenRouter (fallback) — any OpenAI-compatible API |
| Frontend | Vanilla HTML/CSS/JS — no build step, RTL support |
| Container | Docker + docker-compose |

---

## License

MIT
