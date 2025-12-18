"""add_celery_task_id_to_model_variant

Revision ID: 92ac66e14782
Revises: c65a7d640da8
Create Date: 2025-12-18 08:16:07.095096

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '92ac66e14782'
down_revision: Union[str, Sequence[str], None] = 'c65a7d640da8'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Add celery_task_id column to model_variants table."""
    op.add_column('model_variants', sa.Column('celery_task_id', sa.String(length=255), nullable=True))


def downgrade() -> None:
    """Remove celery_task_id column from model_variants table."""
    op.drop_column('model_variants', 'celery_task_id')
