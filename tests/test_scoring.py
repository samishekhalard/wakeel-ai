"""Tests for lead scoring logic."""

import pytest
from app.models import Lead, LeadScore
from app.scoring import score_lead, build_summary


def make_lead(**kwargs) -> Lead:
    defaults = dict(
        session_id="test",
        score=LeadScore.COLD,
        handoff_requested=False,
        viewing_booked=False,
        message_count=0,
    )
    defaults.update(kwargs)
    return Lead(**defaults)


class TestScoreLead:
    def test_empty_lead_is_cold(self):
        lead = make_lead()
        assert score_lead(lead) == LeadScore.COLD

    def test_full_qualification_is_hot(self):
        lead = make_lead(
            intent="buy",
            budget_max=2000000,
            city="Dubai",
            area="Marina",
            property_type="apartment",
            bedrooms=2,
            payment_method="cash",
            timeline="3 months",
            contact_name="Ahmed",
            contact_phone="+971501234567",
        )
        assert score_lead(lead) == LeadScore.HOT

    def test_partial_qualification_is_warm(self):
        lead = make_lead(
            intent="rent",
            budget_max=100000,
            city="Dubai",
        )
        assert score_lead(lead) == LeadScore.WARM

    def test_viewing_booked_pushes_to_hot(self):
        lead = make_lead(
            intent="buy",
            budget_max=1500000,
            city="Dubai",
            viewing_booked=True,
        )
        assert score_lead(lead) == LeadScore.HOT

    def test_handoff_requested_raises_score(self):
        lead = make_lead(
            intent="rent",
            budget_max=80000,
            handoff_requested=True,
        )
        result = score_lead(lead)
        assert result in (LeadScore.HOT, LeadScore.WARM)

    def test_contact_name_adds_points(self):
        cold_lead = make_lead()
        named_lead = make_lead(contact_name="Sara", contact_phone="+971501111111")
        assert score_lead(named_lead).value >= score_lead(cold_lead).value

    def test_high_message_count_adds_point(self):
        lead = make_lead(
            intent="buy",
            budget_max=1000000,
            city="Dubai",
            message_count=10,
        )
        result = score_lead(lead)
        assert result in (LeadScore.WARM, LeadScore.HOT)

    def test_cash_payment_scores_higher_than_no_payment(self):
        no_pay = make_lead(intent="buy", budget_max=1500000, city="Dubai", area="Marina")
        cash   = make_lead(intent="buy", budget_max=1500000, city="Dubai", area="Marina",
                           payment_method="cash")
        # cash adds points — should have same or higher score
        s1 = score_lead(no_pay)
        s2 = score_lead(cash)
        order = [LeadScore.COLD, LeadScore.WARM, LeadScore.HOT]
        assert order.index(s2) >= order.index(s1)


class TestBuildSummary:
    def test_empty_lead_summary(self):
        lead = make_lead()
        assert build_summary(lead) == "No data yet"

    def test_summary_contains_intent(self):
        lead = make_lead(intent="buy", city="Dubai")
        summary = build_summary(lead)
        assert "buy" in summary
        assert "Dubai" in summary

    def test_summary_contains_budget(self):
        lead = make_lead(budget_max=2000000, currency="AED")
        summary = build_summary(lead)
        assert "2,000,000" in summary
        assert "AED" in summary

    def test_summary_contains_viewing(self):
        lead = make_lead(viewing_booked=True)
        summary = build_summary(lead)
        assert "BOOKED" in summary

    def test_summary_contains_handoff(self):
        lead = make_lead(handoff_requested=True)
        summary = build_summary(lead)
        assert "handoff" in summary.lower() or "Handoff" in summary

    def test_full_lead_summary(self):
        lead = make_lead(
            intent="rent",
            city="Riyadh",
            area="KAFD",
            property_type="apartment",
            bedrooms=2,
            budget_max=150000,
            currency="SAR",
            payment_method="cash",
            timeline="1 month",
            contact_name="Olga",
            contact_phone="+79161234567",
        )
        summary = build_summary(lead)
        assert "rent" in summary
        assert "Riyadh" in summary
        assert "KAFD" in summary
        assert "Olga" in summary
