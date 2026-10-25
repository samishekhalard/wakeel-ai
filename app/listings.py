from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

_DATA_DIR = Path(__file__).parent.parent / "data"

_dubai: list[dict[str, Any]] = []
_riyadh: list[dict[str, Any]] = []


def _load() -> None:
    global _dubai, _riyadh
    if not _dubai:
        _dubai = json.loads((_DATA_DIR / "dubai_listings.json").read_text())
    if not _riyadh:
        _riyadh = json.loads((_DATA_DIR / "riyadh_listings.json").read_text())


def get_all() -> list[dict[str, Any]]:
    _load()
    return _dubai + _riyadh


def match_listings(
    city: str | None = None,
    intent: str | None = None,          # "buy" | "rent"
    budget_max: float | None = None,
    area: str | None = None,
    property_type: str | None = None,
    bedrooms: int | None = None,
    payment_method: str | None = None,
    limit: int = 3,
) -> list[dict[str, Any]]:
    _load()

    pool = _dubai + _riyadh
    if city:
        city_lower = city.lower()
        if "riyadh" in city_lower or "riad" in city_lower or "رياض" in city_lower:
            pool = _riyadh
        elif "dubai" in city_lower or "دبي" in city_lower:
            pool = _dubai

    results = []
    for listing in pool:
        # intent filter
        if intent == "buy" and not listing.get("for_sale"):
            continue
        if intent == "rent" and not listing.get("for_rent"):
            continue

        # budget filter
        if budget_max:
            price = listing.get("price_aed") or listing.get("price_sar") or 0
            rent = listing.get("rent_aed_year") or listing.get("rent_sar_year") or 0
            relevant_price = rent if intent == "rent" else price
            if relevant_price > budget_max * 1.15:  # 15% flex
                continue

        # area filter (fuzzy)
        if area:
            area_lower = area.lower()
            listing_area = listing.get("area", "").lower()
            # partial match check
            area_words = area_lower.split()
            if not any(w in listing_area for w in area_words if len(w) > 2):
                continue

        # property type filter
        if property_type:
            pt = property_type.lower()
            lt = listing.get("property_type", "").lower()
            if pt not in lt and lt not in pt:
                # map common aliases
                aliases = {"flat": "apartment", "house": "villa", "flat": "apartment"}
                pt = aliases.get(pt, pt)
                if pt not in lt:
                    continue

        # bedrooms filter (allow ±1)
        if bedrooms is not None:
            lb = listing.get("bedrooms", 0)
            if abs(lb - bedrooms) > 1:
                continue

        # payment filter
        if payment_method:
            pm = payment_method.lower()
            listing_payments = [p.lower() for p in listing.get("payment", [])]
            if pm not in listing_payments:
                continue

        results.append(listing)

    # Score results by how well they match (more fields matched = higher score)
    def relevance(l: dict) -> int:
        score = 0
        if area and area.lower() in l.get("area", "").lower():
            score += 3
        if property_type and property_type.lower() in l.get("property_type", "").lower():
            score += 2
        if bedrooms and l.get("bedrooms") == bedrooms:
            score += 2
        if payment_method and payment_method.lower() in [p.lower() for p in l.get("payment", [])]:
            score += 1
        return score

    results.sort(key=relevance, reverse=True)
    return results[:limit]


def format_listing_for_chat(listing: dict[str, Any], language: str = "en") -> str:
    area = listing.get("area", "")
    city = listing.get("city", "")
    beds = listing.get("bedrooms", "")
    ptype = listing.get("property_type", "")
    size = listing.get("size_sqft", "")
    desc = listing.get("description", "")
    amenities = ", ".join(listing.get("amenities", [])[:4])
    lid = listing.get("id", "")

    price_aed = listing.get("price_aed")
    rent_aed = listing.get("rent_aed_year")
    price_sar = listing.get("price_sar")
    rent_sar = listing.get("rent_sar_year")

    if language == "ar":
        price_str = ""
        if price_aed:
            price_str = f"السعر: {price_aed:,} درهم"
        elif price_sar:
            price_str = f"السعر: {price_sar:,} ريال"
        if rent_aed:
            price_str += f" | الإيجار: {rent_aed:,} درهم/سنة"
        elif rent_sar:
            price_str += f" | الإيجار: {rent_sar:,} ريال/سنة"
        return (
            f"🏠 *{beds} غرف — {ptype} في {area}، {city}*\n"
            f"📐 {size:,} قدم² | {price_str}\n"
            f"✨ {desc}\n"
            f"🔑 {amenities}\n"
            f"ID: {lid}"
        )
    elif language == "ru":
        price_str = ""
        if price_aed:
            price_str = f"Цена: {price_aed:,} AED"
        elif price_sar:
            price_str = f"Цена: {price_sar:,} SAR"
        if rent_aed:
            price_str += f" | Аренда: {rent_aed:,} AED/год"
        elif rent_sar:
            price_str += f" | Аренда: {rent_sar:,} SAR/год"
        return (
            f"🏠 *{beds} спальни — {ptype} в {area}, {city}*\n"
            f"📐 {size:,} кв.фут | {price_str}\n"
            f"✨ {desc}\n"
            f"🔑 {amenities}\n"
            f"ID: {lid}"
        )
    else:
        price_str = ""
        if price_aed:
            price_str = f"Price: AED {price_aed:,}"
        elif price_sar:
            price_str = f"Price: SAR {price_sar:,}"
        if rent_aed:
            price_str += f" | Rent: AED {rent_aed:,}/yr"
        elif rent_sar:
            price_str += f" | Rent: SAR {rent_sar:,}/yr"
        return (
            f"🏠 *{beds}BR {ptype} — {area}, {city}*\n"
            f"📐 {size:,} sqft | {price_str}\n"
            f"✨ {desc}\n"
            f"🔑 {amenities}\n"
            f"ID: {lid}"
        )
