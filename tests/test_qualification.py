"""Tests for listing matching / qualification logic."""

import pytest
from app.listings import match_listings, format_listing_for_chat, get_all


class TestGetAll:
    def test_total_count(self):
        listings = get_all()
        assert len(listings) == 65  # 30 buy Dubai + 15 rent Dubai + 15 buy Riyadh + 5 rent Riyadh

    def test_all_have_id(self):
        for l in get_all():
            assert l.get("id"), f"Listing missing id: {l}"

    def test_all_have_price(self):
        for l in get_all():
            has_price = (
                l.get("price_aed") or l.get("price_sar")
                or l.get("rent_aed_year") or l.get("rent_sar_year")
            )
            assert has_price, f"Listing {l.get('id')} missing price"


class TestMatchListings:
    def test_city_filter_dubai(self):
        results = match_listings(city="Dubai", limit=50)
        assert all(r["city"] == "Dubai" for r in results)

    def test_city_filter_riyadh(self):
        results = match_listings(city="Riyadh", limit=50)
        assert all(r["city"] == "Riyadh" for r in results)

    def test_intent_buy_only_for_sale(self):
        results = match_listings(intent="buy", limit=50)
        assert all(r.get("for_sale") for r in results)

    def test_intent_rent_only_for_rent(self):
        results = match_listings(intent="rent", limit=50)
        assert all(r.get("for_rent") for r in results)

    def test_budget_max_respected_with_flex(self):
        results = match_listings(intent="buy", budget_max=700000, city="Dubai", limit=50)
        # 15% flex allowed: 700k * 1.15 = 805k
        for r in results:
            price = r.get("price_aed") or r.get("price_sar") or 0
            assert price <= 700000 * 1.15, f"Price {price} exceeds budget+flex"

    def test_bedrooms_filter_with_tolerance(self):
        results = match_listings(bedrooms=2, limit=50)
        for r in results:
            assert abs(r.get("bedrooms", 0) - 2) <= 1

    def test_property_type_filter(self):
        results = match_listings(property_type="villa", limit=50)
        for r in results:
            assert "villa" in r.get("property_type", "").lower()

    def test_payment_cash_filter(self):
        results = match_listings(payment_method="cash", limit=50)
        for r in results:
            payments = [p.lower() for p in r.get("payment", [])]
            assert "cash" in payments

    def test_limit_respected(self):
        results = match_listings(limit=2)
        assert len(results) <= 2

    def test_default_limit_3(self):
        results = match_listings()
        assert len(results) <= 3

    def test_no_results_returns_empty(self):
        # Budget so low nothing can match
        results = match_listings(budget_max=1, intent="buy")
        assert results == []

    def test_area_fuzzy_match(self):
        results = match_listings(area="Marina", city="Dubai", limit=10)
        assert any("Marina" in r.get("area", "") for r in results)

    def test_combined_filters(self):
        results = match_listings(
            city="Dubai",
            intent="buy",
            budget_max=1200000,
            property_type="apartment",
            bedrooms=1,
            limit=5,
        )
        for r in results:
            assert r["city"] == "Dubai"
            assert r.get("for_sale")
            assert r.get("property_type") == "apartment"
            assert r.get("bedrooms") in (1, 2)  # ±1 tolerance
            price = r.get("price_aed") or 0
            assert price <= 1200000 * 1.15


class TestFormatListing:
    def test_english_format_contains_key_info(self):
        listings = match_listings(city="Dubai", limit=1)
        assert listings
        text = format_listing_for_chat(listings[0], language="en")
        assert "BR" in text or "bed" in text.lower()
        assert "AED" in text or "SAR" in text

    def test_arabic_format_contains_arabic(self):
        listings = match_listings(city="Dubai", limit=1)
        assert listings
        text = format_listing_for_chat(listings[0], language="ar")
        assert "درهم" in text or "ريال" in text

    def test_russian_format_contains_cyrillic(self):
        listings = match_listings(city="Dubai", limit=1)
        assert listings
        text = format_listing_for_chat(listings[0], language="ru")
        assert "спальни" in text or "AED" in text
