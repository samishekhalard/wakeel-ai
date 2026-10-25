"""Lead scoring: hot / warm / cold based on qualification completeness and signals."""

from app.models import Lead, LeadScore


def score_lead(lead: Lead) -> LeadScore:
    points = 0

    # Qualification completeness
    if lead.intent:
        points += 2
    if lead.budget_max:
        points += 2
    if lead.area or lead.city:
        points += 1
    if lead.property_type:
        points += 1
    if lead.bedrooms is not None:
        points += 1
    if lead.payment_method:
        points += 2
    if lead.timeline:
        points += 1
    if lead.contact_name or lead.contact_phone:
        points += 3

    # Engagement signals
    if lead.viewing_booked:
        points += 5
    if lead.handoff_requested:
        points += 4
    if lead.message_count >= 6:
        points += 1

    if points >= 10:
        return LeadScore.HOT
    if points >= 5:
        return LeadScore.WARM
    return LeadScore.COLD


def build_summary(lead: Lead) -> str:
    parts = []
    if lead.intent:
        parts.append(f"Intent: {lead.intent}")
    if lead.city:
        parts.append(f"City: {lead.city}")
    if lead.area:
        parts.append(f"Area: {lead.area}")
    if lead.property_type:
        parts.append(f"Type: {lead.property_type}")
    if lead.bedrooms is not None:
        parts.append(f"Beds: {lead.bedrooms}")
    if lead.budget_max:
        currency = lead.currency or ""
        parts.append(f"Budget: up to {lead.budget_max:,.0f} {currency}")
    if lead.payment_method:
        parts.append(f"Payment: {lead.payment_method}")
    if lead.timeline:
        parts.append(f"Timeline: {lead.timeline}")
    if lead.contact_name:
        parts.append(f"Name: {lead.contact_name}")
    if lead.contact_phone:
        parts.append(f"Phone: {lead.contact_phone}")
    if lead.viewing_booked:
        parts.append("Viewing: BOOKED")
    if lead.handoff_requested:
        parts.append("Handoff: requested")
    return " | ".join(parts) if parts else "No data yet"
