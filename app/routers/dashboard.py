from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse
from sqlalchemy import select

from app.models import Lead

router = APIRouter(tags=["dashboard"])

_SCORE_EMOJI = {"hot": "🔥", "warm": "⚡", "cold": "❄️"}
_SCORE_COLOR = {"hot": "#e74c3c", "warm": "#f39c12", "cold": "#3498db"}


def _row(lead: Lead) -> str:
    score_emoji = _SCORE_EMOJI.get(lead.score, "")
    score_color = _SCORE_COLOR.get(lead.score, "#999")
    name = lead.contact_name or "—"
    phone = lead.contact_phone or "—"
    lang_flag = {"ar": "🇦🇪", "en": "🇬🇧", "ru": "🇷🇺"}.get(lead.language or "", "🌐")
    viewing = "✅" if lead.viewing_booked else ""
    handoff = "🤝" if lead.handoff_requested else ""
    created = lead.created_at.strftime("%d %b %H:%M") if lead.created_at else "—"
    summary = lead.summary or "—"

    return f"""
    <tr>
      <td>{created}</td>
      <td>{lang_flag} {name}</td>
      <td>{phone}</td>
      <td style="color:{score_color};font-weight:700">{score_emoji} {lead.score.upper()}</td>
      <td>{lead.message_count}</td>
      <td style="max-width:320px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap">{summary}</td>
      <td>{viewing} {handoff}</td>
    </tr>"""


@router.get("/dashboard", response_class=HTMLResponse)
async def dashboard(request: Request):
    async with request.app.state.session_factory() as session:
        result = await session.execute(
            select(Lead).order_by(Lead.updated_at.desc())
        )
        leads = result.scalars().all()

        # Stats
        total = len(leads)
        hot = sum(1 for l in leads if l.score == "hot")
        warm = sum(1 for l in leads if l.score == "warm")
        cold = sum(1 for l in leads if l.score == "cold")
        viewings = sum(1 for l in leads if l.viewing_booked)

    rows = "".join(_row(l) for l in leads)
    if not rows:
        rows = '<tr><td colspan="7" style="text-align:center;padding:40px;color:#888">No leads yet — start a chat at <a href="/">/</a></td></tr>'

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Wakeel AI — Lead Dashboard</title>
  <style>
    * {{ box-sizing: border-box; margin: 0; padding: 0; }}
    body {{
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      background: #f0f2f5;
      color: #1a1a2e;
    }}
    header {{
      background: #0B1E3F;
      color: white;
      padding: 18px 32px;
      display: flex;
      align-items: center;
      gap: 16px;
    }}
    header h1 {{ font-size: 1.3rem; font-weight: 700; }}
    header span {{ color: #C9962E; font-size: 1.4rem; }}
    .stats {{
      display: flex;
      gap: 16px;
      padding: 24px 32px 0;
      flex-wrap: wrap;
    }}
    .stat-card {{
      background: white;
      border-radius: 12px;
      padding: 16px 24px;
      min-width: 140px;
      box-shadow: 0 1px 4px rgba(0,0,0,.08);
      text-align: center;
    }}
    .stat-card .val {{
      font-size: 2rem;
      font-weight: 800;
      color: #0B1E3F;
    }}
    .stat-card .lbl {{ font-size: .8rem; color: #888; margin-top: 4px; }}
    .table-wrap {{
      margin: 24px 32px;
      background: white;
      border-radius: 12px;
      box-shadow: 0 1px 4px rgba(0,0,0,.08);
      overflow: hidden;
    }}
    table {{ width: 100%; border-collapse: collapse; font-size: .9rem; }}
    th {{
      background: #0B1E3F;
      color: white;
      padding: 12px 16px;
      text-align: left;
      font-weight: 600;
    }}
    td {{ padding: 11px 16px; border-bottom: 1px solid #f0f2f5; }}
    tr:last-child td {{ border-bottom: none; }}
    tr:hover td {{ background: #fafbfc; }}
    .badge-hot {{ background: #ffe5e5; color: #c0392b; padding: 2px 10px; border-radius: 99px; font-size:.8rem; }}
    .badge-warm {{ background: #fff3cd; color: #856404; padding: 2px 10px; border-radius: 99px; font-size:.8rem; }}
    .badge-cold {{ background: #d1ecf1; color: #0c5460; padding: 2px 10px; border-radius: 99px; font-size:.8rem; }}
    .actions {{
      padding: 16px 32px;
      display: flex;
      gap: 12px;
    }}
    .btn {{
      background: #C9962E;
      color: white;
      padding: 10px 20px;
      border-radius: 8px;
      text-decoration: none;
      font-weight: 600;
      font-size: .9rem;
    }}
    .btn:hover {{ background: #b07d22; }}
    @media (max-width: 700px) {{
      .stats {{ padding: 16px; }}
      .table-wrap {{ margin: 16px; }}
      th:nth-child(6), td:nth-child(6) {{ display: none; }}
    }}
  </style>
</head>
<body>
  <header>
    <span>⚖️</span>
    <div>
      <h1>Wakeel AI — Lead Dashboard</h1>
      <div style="font-size:.8rem;opacity:.7">Real-time lead qualification overview</div>
    </div>
  </header>

  <div class="stats">
    <div class="stat-card"><div class="val">{total}</div><div class="lbl">Total Leads</div></div>
    <div class="stat-card"><div class="val" style="color:#e74c3c">🔥 {hot}</div><div class="lbl">Hot</div></div>
    <div class="stat-card"><div class="val" style="color:#f39c12">⚡ {warm}</div><div class="lbl">Warm</div></div>
    <div class="stat-card"><div class="val" style="color:#3498db">❄️ {cold}</div><div class="lbl">Cold</div></div>
    <div class="stat-card"><div class="val" style="color:#27ae60">📅 {viewings}</div><div class="lbl">Viewings</div></div>
  </div>

  <div class="actions">
    <a href="/" class="btn">💬 Open Chat Demo</a>
  </div>

  <div class="table-wrap">
    <table>
      <thead>
        <tr>
          <th>Time</th>
          <th>Name</th>
          <th>Phone</th>
          <th>Score</th>
          <th>Messages</th>
          <th>Summary</th>
          <th>Actions</th>
        </tr>
      </thead>
      <tbody>
        {rows}
      </tbody>
    </table>
  </div>
</body>
</html>"""
