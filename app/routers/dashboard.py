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
    viewing_time = f" {lead.viewing_time}" if lead.viewing_booked and getattr(lead, "viewing_time", None) else ""
    handoff = "🤝" if lead.handoff_requested else ""
    contacted = getattr(lead, "contacted", False)
    contacted_style = "background:#27ae60" if contacted else "background:#95a5a6"
    contacted_label = "Contacted ✓" if contacted else "Mark contacted"
    created = lead.created_at.strftime("%d %b %H:%M") if lead.created_at else "—"
    summary = (lead.summary or "—").replace('"', "&quot;")

    return f"""
    <tr>
      <td>{created}</td>
      <td>{lang_flag}&nbsp;{name}</td>
      <td>{phone}</td>
      <td><span class="badge-{lead.score}">{score_emoji} {lead.score.upper()}</span></td>
      <td>{lead.message_count}</td>
      <td style="max-width:280px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap" title="{summary}">{summary}</td>
      <td>{viewing}{viewing_time} {handoff}</td>
      <td>
        <a href="/api/leads/{lead.session_id}/conversation" target="_blank"
           style="background:#0B1E3F;color:white;padding:4px 10px;border-radius:6px;
                  font-size:.78rem;text-decoration:none;margin-right:4px">View</a>
        <button onclick="markContacted('{lead.session_id}', this)"
                style="{contacted_style};color:white;padding:4px 10px;border-radius:6px;
                        font-size:.78rem;border:none;cursor:pointer">{contacted_label}</button>
      </td>
    </tr>"""


@router.get("/dashboard", response_class=HTMLResponse)
async def dashboard(request: Request):
    async with request.app.state.session_factory() as session:
        result = await session.execute(
            select(Lead).order_by(Lead.updated_at.desc())
        )
        leads = result.scalars().all()

        total = len(leads)
        hot = sum(1 for l in leads if l.score == "hot")
        warm = sum(1 for l in leads if l.score == "warm")
        cold = sum(1 for l in leads if l.score == "cold")
        viewings = sum(1 for l in leads if l.viewing_booked)
        contacted_count = sum(1 for l in leads if getattr(l, "contacted", False))

    rows = "".join(_row(l) for l in leads)
    if not rows:
        rows = '<tr><td colspan="8" style="text-align:center;padding:40px;color:#888">No leads yet — start a chat at <a href="/">/</a></td></tr>'

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
      padding: 14px 32px;
      display: flex;
      align-items: center;
      gap: 14px;
    }}
    .logo-wrap {{
      width: 40px; height: 40px;
      border-radius: 50%;
      overflow: hidden;
      background: #C9962E;
      display: flex; align-items: center; justify-content: center;
      font-size: 1.4rem;
      flex-shrink: 0;
    }}
    .logo-wrap img {{ width: 100%; height: 100%; object-fit: cover; border-radius: 50%; }}
    header h1 {{ font-size: 1.2rem; font-weight: 700; }}
    header .sub {{ color: rgba(255,255,255,.6); font-size: .78rem; }}
    .stats {{
      display: flex;
      gap: 14px;
      padding: 20px 32px 0;
      flex-wrap: wrap;
    }}
    .stat-card {{
      background: white;
      border-radius: 12px;
      padding: 14px 22px;
      min-width: 130px;
      box-shadow: 0 1px 4px rgba(0,0,0,.08);
      text-align: center;
    }}
    .stat-card .val {{ font-size: 1.9rem; font-weight: 800; color: #0B1E3F; }}
    .stat-card .lbl {{ font-size: .78rem; color: #888; margin-top: 3px; }}
    .table-wrap {{
      margin: 20px 32px;
      background: white;
      border-radius: 12px;
      box-shadow: 0 1px 4px rgba(0,0,0,.08);
      overflow: hidden;
    }}
    table {{ width: 100%; border-collapse: collapse; font-size: .88rem; }}
    th {{
      background: #0B1E3F;
      color: white;
      padding: 11px 14px;
      text-align: left;
      font-weight: 600;
    }}
    td {{ padding: 10px 14px; border-bottom: 1px solid #f0f2f5; vertical-align: middle; }}
    tr:last-child td {{ border-bottom: none; }}
    tr:hover td {{ background: #fafbfc; }}
    .badge-hot  {{ background:#ffe5e5;color:#c0392b;padding:2px 9px;border-radius:99px;font-size:.78rem;white-space:nowrap; }}
    .badge-warm {{ background:#fff3cd;color:#856404;padding:2px 9px;border-radius:99px;font-size:.78rem;white-space:nowrap; }}
    .badge-cold {{ background:#d1ecf1;color:#0c5460;padding:2px 9px;border-radius:99px;font-size:.78rem;white-space:nowrap; }}
    .toolbar {{
      padding: 14px 32px;
      display: flex;
      gap: 10px;
      align-items: center;
    }}
    .btn {{
      background: #C9962E;
      color: white;
      padding: 9px 18px;
      border-radius: 8px;
      text-decoration: none;
      font-weight: 600;
      font-size: .88rem;
      border: none;
      cursor: pointer;
    }}
    .btn:hover {{ background: #b07d22; }}
    @media (max-width: 700px) {{
      .stats {{ padding: 14px; gap: 10px; }}
      .table-wrap {{ margin: 14px; }}
      th:nth-child(6), td:nth-child(6),
      th:nth-child(7), td:nth-child(7) {{ display: none; }}
    }}
  </style>
</head>
<body>
  <header>
    <div class="logo-wrap">
      <img src="/static/wakeel-logo.png" alt="Wakeel"
           onerror="this.style.display='none';this.parentElement.textContent='⚖️'">
    </div>
    <div>
      <h1>Wakeel AI — Lead Dashboard</h1>
      <div class="sub">Real-time lead qualification overview</div>
    </div>
  </header>

  <div class="stats">
    <div class="stat-card"><div class="val">{total}</div><div class="lbl">Total</div></div>
    <div class="stat-card"><div class="val" style="color:#e74c3c">🔥 {hot}</div><div class="lbl">Hot</div></div>
    <div class="stat-card"><div class="val" style="color:#f39c12">⚡ {warm}</div><div class="lbl">Warm</div></div>
    <div class="stat-card"><div class="val" style="color:#3498db">❄️ {cold}</div><div class="lbl">Cold</div></div>
    <div class="stat-card"><div class="val" style="color:#27ae60">📅 {viewings}</div><div class="lbl">Viewings</div></div>
    <div class="stat-card"><div class="val" style="color:#8e44ad">✅ {contacted_count}</div><div class="lbl">Contacted</div></div>
  </div>

  <div class="toolbar">
    <a href="/" class="btn">💬 Open Chat</a>
    <button class="btn" style="background:#0B1E3F" onclick="location.reload()">↺ Refresh</button>
  </div>

  <div class="table-wrap">
    <table>
      <thead>
        <tr>
          <th>Time</th>
          <th>Name</th>
          <th>Phone</th>
          <th>Score</th>
          <th>Msgs</th>
          <th>Summary</th>
          <th>Status</th>
          <th>Actions</th>
        </tr>
      </thead>
      <tbody>
        {rows}
      </tbody>
    </table>
  </div>

<script>
async function markContacted(sessionId, btn) {{
  try {{
    const r = await fetch(`/api/leads/${{sessionId}}/contacted`, {{method: 'POST'}});
    const d = await r.json();
    if (d.contacted) {{
      btn.textContent = 'Contacted ✓';
      btn.style.background = '#27ae60';
    }} else {{
      btn.textContent = 'Mark contacted';
      btn.style.background = '#95a5a6';
    }}
  }} catch(e) {{ alert('Error updating status'); }}
}}
</script>
</body>
</html>"""
