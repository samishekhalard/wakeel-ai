# Wakeel AI — CLAUDE.md

## Business Goal
Sell AI lead-qualification agents to real estate agencies in Dubai and Saudi Arabia (including Russian-speaking agents in Dubai). The agent chats with property buyers/renters, qualifies them, recommends listings, and hands off hot leads to human agents.

## Target Clients
- Dubai real estate agencies (Arabic and English)
- Russian-speaking agent teams operating in Dubai
- Saudi Arabia agencies (Riyadh, Jeddah)

## Languages
- **Gulf Arabic** (خليجي dialect) — primary for UAE/KSA
- **English** — universal fallback
- **Russian** — Russian-speaking Dubai market

The agent auto-detects language from the user's first message and replies in the same language throughout the conversation.

## Budget Constraint
**$0 — free tiers and self-hosted only.**
- LLM: OpenRouter free models, Groq free tier, or Gemini free tier
- All credentials via environment variables — never hardcoded
- SQLite instead of PostgreSQL
- Docker for local/VPS deployment

## Stack
| Layer | Technology |
|-------|-----------|
| Backend | Python 3.11, FastAPI |
| Database | SQLite (via SQLAlchemy + aiosqlite) |
| LLM | Any OpenAI-compatible API (OpenRouter / Groq / Gemini) |
| Frontend | Vanilla HTML/CSS/JS (no build step) |
| Container | Docker + docker-compose |
| Automation | n8n (Phase 3+) |

## Coding Standards
- Python: type hints everywhere, async FastAPI handlers, Pydantic v2 models
- No secrets in code — `.env` only, `.env.example` committed
- One responsibility per module
- Tests for all business logic (qualification, scoring)
- Mobile-first frontend, RTL support for Arabic
- Brand: navy `#0B1E3F`, gold `#C9962E`
- Clean git history — one commit per logical unit

## Project Structure
```
wakeel-ai/
├── app/
│   ├── main.py          # FastAPI app entry point
│   ├── agent.py         # LLM agent, language detection, qualification flow
│   ├── models.py        # SQLAlchemy models + DB init
│   ├── listings.py      # Listing loader + matching logic
│   ├── scoring.py       # Lead hot/warm/cold scoring
│   ├── routers/
│   │   ├── chat.py      # POST /chat, GET /chat/history/{session_id}
│   │   └── dashboard.py # GET /dashboard
│   └── static/
│       └── index.html   # Single-page WhatsApp-style chat UI
├── data/
│   ├── dubai_listings.json
│   └── riyadh_listings.json
├── tests/
│   ├── test_qualification.py
│   └── test_scoring.py
├── .env.example
├── docker-compose.yml
├── Dockerfile
├── requirements.txt
├── README.md
├── DEMO_SCRIPT.md
└── CLAUDE.md
```

## Roadmap

### Phase 1 — Web Chat Demo (current)
- FastAPI backend + WhatsApp-style web UI
- LLM provider abstraction via env vars
- Dubai (30) + Riyadh (15) fake listings
- Lead qualification: buy/rent, budget, area, type, bedrooms, payment, timeline
- SQLite lead storage with hot/warm/cold scoring
- `/dashboard` page
- Docker, tests, README

### Phase 2 — WhatsApp Cloud API (do NOT build yet)
- Webhook handler for Meta WhatsApp Cloud API
- Session state via Redis or SQLite
- Outbound message sending with template support

### Phase 3 — n8n Workflows + CRM Sync (do NOT build yet)
- n8n self-hosted for automation
- Google Sheets sync for hot leads
- CRM webhook integration (HubSpot free tier)

### Phase 4 — Instagram DM Integration (do NOT build yet)
- Meta Graph API for Instagram DMs
- Same agent pipeline, new channel adapter

### Phase 5 — Multi-Client SaaS (do NOT build yet)
- One deployment, many agencies
- Per-agency config: listings, brand, language, prompts
- Agency admin panel for lead management
