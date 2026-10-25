# ⚖️ Wakeel AI — Real Estate Lead Qualification Agent

> AI-powered sales agent that chats with property buyers & renters in **Gulf Arabic, English, and Russian**, qualifies leads, recommends listings, and hands off hot prospects to human agents.

Built for real estate agencies in **Dubai** and **Riyadh**.

---

## Screenshots

| Chat UI (English) | Chat UI (Arabic/RTL) | Lead Dashboard |
|:-:|:-:|:-:|
| *(screenshot placeholder)* | *(screenshot placeholder)* | *(screenshot placeholder)* |

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        Browser                              │
│  WhatsApp-style chat UI (HTML/CSS/JS, mobile-first, RTL)    │
└───────────────────────┬─────────────────────────────────────┘
                        │ POST /api/chat
┌───────────────────────▼─────────────────────────────────────┐
│                    FastAPI Backend                           │
│                                                             │
│  ┌─────────────┐  ┌──────────────┐  ┌──────────────────┐  │
│  │  Agent      │  │  Listings    │  │  Scoring         │  │
│  │  (LLM +     │  │  (30 Dubai + │  │  (hot/warm/cold  │  │
│  │   tools)    │  │   15 Riyadh) │  │   points model)  │  │
│  └──────┬──────┘  └──────────────┘  └──────────────────┘  │
│         │                                                   │
│  ┌──────▼──────────────────────────────────────────────┐   │
│  │         SQLite (leads + message history)             │   │
│  └─────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
                        │
           Any OpenAI-compatible LLM API
        (OpenRouter / Groq / Gemini free tiers)
```

### Key design decisions

| Decision | Rationale |
|----------|-----------|
| LLM provider abstraction via env vars | Zero vendor lock-in; swap OpenRouter → Groq → Gemini without code changes |
| Tool calling for extraction | Structured JSON extraction is more reliable than regex parsing of free text |
| SQLite | No infra cost; easy to migrate to Postgres later |
| Vanilla JS frontend | No build step; instant load; easy to demo offline |
| Points-based scoring | Transparent, tunable, no ML dependency |

---

## Qualification Flow

The agent asks 1–2 questions per message and extracts:

1. **Intent** — buy or rent
2. **City** — Dubai or Riyadh
3. **Area** — neighborhood preference
4. **Property type** — apartment / villa / townhouse
5. **Bedrooms**
6. **Budget** (AED or SAR)
7. **Payment** — cash or mortgage
8. **Timeline** — when they want to move

Once enough data is collected, it recommends 2–3 matching listings and offers to book a viewing.

### Lead scoring

| Score | Condition |
|-------|-----------|
| 🔥 Hot | ≥10 points (contact info + viewing + full qualification) |
| ⚡ Warm | 5–9 points (partial qualification) |
| ❄️ Cold | <5 points (just started) |

---

## Quick Start

### 1. Clone and configure

```bash
git clone https://github.com/youruser/wakeel-ai.git
cd wakeel-ai
cp .env.example .env
```

Edit `.env`:
```env
LLM_BASE_URL=https://openrouter.ai/api/v1
LLM_API_KEY=your_openrouter_key_here
LLM_MODEL=meta-llama/llama-3.1-8b-instruct:free
```

**Free LLM options:**
- [OpenRouter](https://openrouter.ai/) — many free models (Llama 3.1, Gemma, Mistral)
- [Groq](https://console.groq.com/) — blazing fast, free tier
- [Google AI Studio](https://aistudio.google.com/) — Gemini 1.5 Flash free tier

### 2. Run locally (Python)

```bash
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
uvicorn app.main:app --reload
```

Open: http://localhost:8000  
Dashboard: http://localhost:8000/dashboard

### 3. Run with Docker

```bash
docker-compose up --build
```

Open: http://localhost:8000

---

## Running Tests

```bash
pytest tests/ -v
```

Tests cover:
- Listing data integrity (all 45 listings have required fields)
- Matching filters (city, intent, budget, bedrooms, payment, area)
- Lead scoring (hot/warm/cold point thresholds)
- Summary generation

---

## Project Structure

```
wakeel-ai/
├── app/
│   ├── main.py          # FastAPI app, lifespan, routing
│   ├── agent.py         # LLM agent, tool calling, language detection
│   ├── config.py        # Pydantic settings from .env
│   ├── models.py        # SQLAlchemy async models (Lead, Message)
│   ├── listings.py      # Listing loader + fuzzy matching
│   ├── scoring.py       # Points-based lead scoring
│   ├── routers/
│   │   ├── chat.py      # POST /api/chat, GET /api/chat/history/{id}
│   │   └── dashboard.py # GET /dashboard
│   └── static/
│       └── index.html   # Chat UI (single file, no build step)
├── data/
│   ├── dubai_listings.json    # 30 Dubai properties
│   └── riyadh_listings.json   # 15 Riyadh properties
├── tests/
│   ├── test_qualification.py
│   └── test_scoring.py
├── .env.example
├── docker-compose.yml
├── Dockerfile
└── requirements.txt
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

## Tech Stack

- **Python 3.11** · **FastAPI** · **SQLAlchemy async** · **SQLite**
- **Pydantic v2** for validation and settings
- **httpx** for async LLM API calls
- **Docker** + **docker-compose**
- Vanilla HTML/CSS/JS — no build tooling

---

## License

MIT
