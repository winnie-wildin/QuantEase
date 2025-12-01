"""add task aware evaluation

Revision ID: 3fb3c0a27317
Revises: b5a60ec853f2
Create Date: 2025-11-22 08:51:01.732044

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '3fb3c0a27317'
down_revision: Union[str, Sequence[str], None] = 'b5a60ec853f2'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade():
    """Add fields for task-aware evaluation"""
    
    # 1. Add task_type to experiments
    op.add_column('experiments', 
        sa.Column('task_type', sa.String(50), nullable=True)
    )
    
    # 2. Add normalization_metadata (JSON) to experiments
    op.add_column('experiments',
        sa.Column('normalization_metadata', sa.JSON(), nullable=True)
    )
    
    # 3. Add judge configuration to experiments
    op.add_column('experiments',
        sa.Column('judge_enabled', sa.Boolean(), server_default='false', nullable=False)
    )
    
    op.add_column('experiments',
        sa.Column('judge_sample_percentage', sa.Float(), server_default='10.0', nullable=False)
    )
    
    # 4. Add context field for RAG tasks in dataset_samples
    op.add_column('dataset_samples',
        sa.Column('context', sa.Text(), nullable=True)
    )
    
    # 5. Add evaluation_results (JSON) to comparative_metrics
    op.add_column('comparative_metrics',
        sa.Column('evaluation_results', sa.JSON(), nullable=True)
    )
    
    # 6. Create index on task_type for faster queries
    op.create_index(
        'idx_experiments_task_type',
        'experiments',
        ['task_type']
    )
    
    # 7. Backfill existing experiments with default task_type
    op.execute("""
        UPDATE experiments 
        SET task_type = 'text_generation' 
        WHERE task_type IS NULL
    """)


def downgrade():
    """Remove task-aware evaluation fields"""
    
    # Remove index
    op.drop_index('idx_experiments_task_type', table_name='experiments')
    
    # Remove columns in reverse order
    op.drop_column('comparative_metrics', 'evaluation_results')
    op.drop_column('dataset_samples', 'context')
    op.drop_column('experiments', 'judge_sample_percentage')
    op.drop_column('experiments', 'judge_enabled')
    op.drop_column('experiments', 'normalization_metadata')
    op.drop_column('experiments', 'task_type')