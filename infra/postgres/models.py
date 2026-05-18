from sqlalchemy import (
    Column, String, Float, Integer,
    Text, Boolean, DateTime, JSON
)
from sqlalchemy.sql import func
from infra.postgres.database import Base
import uuid


def generate_uuid() -> str:
    """generates a unique id for each row"""
    return str(uuid.uuid4())


class Job(Base):
    """
    tracks every filing analysis request.
    when user types 'analyze AAPL 10-K', one Job row is created.
    status moves: pending → running → done (or failed or review)
    thread_id links this job to its LangGraph graph state in Redis
    """
    __tablename__ = "jobs"

    id = Column(String, primary_key=True, default=generate_uuid)
    ticker = Column(String(20), nullable=False, index=True)
    filing_type = Column(String(10), nullable=False)   # 10-K, 10-Q, 8-K
    status = Column(String(20), default="pending")
    thread_id = Column(String, unique=True, nullable=False)
    retry_count = Column(Integer, default=0)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())


class FilingSignal(Base):
    """
    the core output of the whole pipeline.
    stores extracted signals from a filing.
    this is what the user sees on the dashboard.
    this is what the voice agent reads from Redis cache.
    """
    __tablename__ = "filing_signals"

    id = Column(String, primary_key=True, default=generate_uuid)
    job_id = Column(String, nullable=False, index=True)
    ticker = Column(String(20), nullable=False, index=True)
    filing_type = Column(String(10), nullable=False)
    filing_date = Column(String(20))

    # the actual signals extracted by the LLM
    revenue_growth_yoy = Column(Float, nullable=True)
    gross_margin = Column(Float, nullable=True)
    guidance_sentiment = Column(String(20))    # positive/neutral/negative
    key_risks = Column(JSON, default=list)     # ["risk1", "risk2", ...]
    red_flags = Column(JSON, default=list)     # ["flag1", "flag2", ...]
    summary = Column(Text)                     # one paragraph summary

    # confidence is how sure the LLM was about its extraction
    # below 0.85 triggers human review
    confidence = Column(Float, nullable=False)
    human_reviewed = Column(Boolean, default=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())


class OutboxEvent(Base):
    """
    outbox pattern implementation.
    when gateway receives a request, it writes Job row AND
    OutboxEvent row in ONE transaction.
    a separate CDC relay process reads unsent events and
    publishes them to Kafka.
    this guarantees we never lose a message even if app crashes.
    """
    __tablename__ = "outbox_events"

    id = Column(String, primary_key=True, default=generate_uuid)
    topic = Column(String(100), nullable=False)    # which kafka topic
    payload = Column(JSON, nullable=False)          # message content
    sent = Column(Boolean, default=False, index=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    sent_at = Column(DateTime(timezone=True), nullable=True)


class WatchlistItem(Base):
    """
    tickers the watchlist monitor checks every 15 minutes.
    when a new filing is detected, pipeline starts automatically.
    """
    __tablename__ = "watchlist"

    id = Column(String, primary_key=True, default=generate_uuid)
    ticker = Column(String(20), nullable=False, unique=True)
    filing_types = Column(JSON, default=["10-K", "10-Q", "8-K"])
    last_checked_at = Column(DateTime(timezone=True), nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())