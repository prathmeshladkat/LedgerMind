# infra/postgres/migrations/env.py
# custom async env.py for Neon DB connection
# replaces the default sync version alembic generates

import asyncio
import sys
import os
from logging.config import fileConfig

from sqlalchemy.ext.asyncio import create_async_engine
from alembic import context

# add project root to path so we can import infra.*
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../.."))

from infra.postgres.database import Base
from infra.postgres import models   # noqa - registers all models with Base
from infra.settings import get_settings

# reads logging config from alembic.ini
config = context.config
if config.config_file_name:
    fileConfig(config.config_file_name)

# Base.metadata has all our table definitions
target_metadata = Base.metadata


def run_migrations_offline():
    """Run without a live DB connection."""
    url = get_settings().neon_database_url
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True
    )
    with context.begin_transaction():
        context.run_migrations()


def do_run_migrations(connection):
    """Run migrations on a live connection."""
    context.configure(
        connection=connection,
        target_metadata=target_metadata
    )
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online():
    """Connect to Neon DB and run migrations."""
    # alembic needs sync psycopg2 URL not asyncpg
    # replace +asyncpg with +psycopg2 for migrations only
    url = get_settings().neon_database_url.replace(
        "postgresql+asyncpg://",
        "postgresql+psycopg2://"
    )
    
    from sqlalchemy import create_engine
    # use sync engine for alembic - it handles async internally
    connectable = create_engine(url)

    with connectable.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=target_metadata
        )
        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    # alembic is sync - don't use asyncio.run()
    run_migrations_online()