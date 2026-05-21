"""initial tables

Revision ID: 001
Revises: 
Create Date: 2025-01-01
"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSON

# these three variables must exist at module level
revision = "001"
down_revision = None
branch_labels = None
depends_on = None


def upgrade():
    op.create_table("jobs",
        sa.Column("id", sa.String(), primary_key=True),
        sa.Column("ticker", sa.String(20), nullable=False),
        sa.Column("filing_type", sa.String(10), nullable=False),
        sa.Column("status", sa.String(20), default="pending"),
        sa.Column("thread_id", sa.String(), unique=True, nullable=False),
        sa.Column("retry_count", sa.Integer(), default=0),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(timezone=True), onupdate=sa.func.now()),
    )
    op.create_table("filing_signals",
        sa.Column("id", sa.String(), primary_key=True),
        sa.Column("job_id", sa.String(), nullable=False),
        sa.Column("ticker", sa.String(20), nullable=False),
        sa.Column("filing_type", sa.String(10), nullable=False),
        sa.Column("filing_date", sa.String(20)),
        sa.Column("revenue_growth_yoy", sa.Float(), nullable=True),
        sa.Column("gross_margin", sa.Float(), nullable=True),
        sa.Column("guidance_sentiment", sa.String(20)),
        sa.Column("key_risks", JSON, default=list),
        sa.Column("red_flags", JSON, default=list),
        sa.Column("summary", sa.Text()),
        sa.Column("confidence", sa.Float(), nullable=False),
        sa.Column("human_reviewed", sa.Boolean(), default=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )
    op.create_table("outbox_events",
        sa.Column("id", sa.String(), primary_key=True),
        sa.Column("topic", sa.String(100), nullable=False),
        sa.Column("payload", JSON, nullable=False),
        sa.Column("sent", sa.Boolean(), default=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column("sent_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.create_table("watchlist",
        sa.Column("id", sa.String(), primary_key=True),
        sa.Column("ticker", sa.String(20), nullable=False, unique=True),
        sa.Column("filing_types", JSON, default=list),
        sa.Column("last_checked_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )


def downgrade():
    op.drop_table("watchlist")
    op.drop_table("outbox_events")
    op.drop_table("filing_signals")
    op.drop_table("jobs")