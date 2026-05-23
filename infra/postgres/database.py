from sqlalchemy.ext.asyncio import (
    create_async_engine,
    AsyncSession,
    async_sessionmaker
)
from typing import AsyncGenerator

from sqlalchemy.orm import DeclarativeBase
from infra.settings import get_settings
import logging

logger = logging.getLogger(__name__)

class Base(DeclarativeBase):
    pass

_engine = None
_session_factory = None

def init_db():
    """
    call this once when app starts.
    creates the cinnection pool to neon postgress
    pool_pre_ping=True check connection is alive before using it.
    this prevents errors from stake connections.
    """
    global _engine, _session_factory

    settings = get_settings()

    _engine = create_async_engine(
         settings.neon_database_url,
        echo=False,    # logs all SQL in development
        pool_pre_ping=True,      # checks connection health
        pool_size=5,             # keep 5 connections open
        max_overflow=10,         # allow 10 extra connections under load
    )

    _session_factory = async_sessionmaker(
        _engine,
        class_=AsyncSession,
        expire_on_commit=False
    )

    logger.info("Database engine initialized")

async def get_session() -> AsyncGenerator[AsyncSession, None]:
    """
    yields a database session.
    use this in fastapi routes with depends(get_session)
    automatically closes session when request is done.
    """
    if _session_factory is None:
        raise RuntimeError("DB not initialized. Call init_db() at startup.")
    
    async with _session_factory() as session:
        yield session

async def close_db():
    """call this when app shuts down to close all connections."""
    global _engine
    if _engine:
        await _engine.dispose()
        logger.info("Database engine closed")