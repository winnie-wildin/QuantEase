"""add_detected_labels_to_experiments

Revision ID: c65a7d640da8
Revises: 3fb3c0a27317
Create Date: 2025-11-27 19:37:49.519471

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'c65a7d640da8'
down_revision: Union[str, Sequence[str], None] = '3fb3c0a27317'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None
def upgrade():
    op.add_column('experiments', sa.Column('detected_labels', sa.JSON(), nullable=True))

def downgrade():
    op.drop_column('experiments', 'detected_labels')