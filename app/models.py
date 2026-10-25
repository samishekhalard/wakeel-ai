from datetime import datetime
from enum import Enum
from typing import Optional
from sqlalchemy import String, Text, Float, DateTime, Integer, Boolean
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column
from sqlalchemy.ext.asyncio import AsyncAttrs, create_async_engine, async_sessionmaker


class Base(AsyncAttrs, DeclarativeBase):
    pass


class LeadScore(str, Enum):
    HOT = "hot"
    WARM = "warm"
    COLD = "cold"


class Lead(Base):
    __tablename__ = "leads"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    session_id: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    language: Mapped[Optional[str]] = mapped_column(String(8), nullable=True)
    intent: Mapped[Optional[str]] = mapped_column(String(16), nullable=True)   # buy | rent
    budget_min: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    budget_max: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    currency: Mapped[Optional[str]] = mapped_column(String(8), nullable=True)  # AED | SAR
    area: Mapped[Optional[str]] = mapped_column(String(128), nullable=True)
    city: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    property_type: Mapped[Optional[str]] = mapped_column(String(32), nullable=True)
    bedrooms: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    payment_method: Mapped[Optional[str]] = mapped_column(String(16), nullable=True)  # cash | mortgage
    timeline: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)

    score: Mapped[str] = mapped_column(String(8), default=LeadScore.COLD)
    handoff_requested: Mapped[bool] = mapped_column(Boolean, default=False)
    viewing_booked: Mapped[bool] = mapped_column(Boolean, default=False)
    summary: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    contact_name: Mapped[Optional[str]] = mapped_column(String(128), nullable=True)
    contact_phone: Mapped[Optional[str]] = mapped_column(String(32), nullable=True)
    message_count: Mapped[int] = mapped_column(Integer, default=0)


class Message(Base):
    __tablename__ = "messages"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    session_id: Mapped[str] = mapped_column(String(64), index=True)
    role: Mapped[str] = mapped_column(String(16))   # user | assistant
    content: Mapped[str] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


def create_engine(database_url: str):
    return create_async_engine(database_url, echo=False)


def create_session_factory(engine):
    return async_sessionmaker(engine, expire_on_commit=False)


async def init_db(engine) -> None:
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
