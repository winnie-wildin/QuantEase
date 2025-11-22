"""add_is_draft_to_experiments

Revision ID: b5a60ec853f2
Revises: 2858b78908aa
Create Date: 2025-11-19 15:03:44.505742

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'b5a60ec853f2'
down_revision: Union[str, Sequence[str], None] = '2858b78908aa'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

def upgrade():
    op.add_column('experiments', sa.Column('is_draft', sa.Boolean(), server_default='true', nullable=True))

def downgrade():
    op.drop_column('experiments', 'is_draft')