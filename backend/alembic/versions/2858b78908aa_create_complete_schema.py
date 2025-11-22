"""create_complete_schema

Revision ID: 2858b78908aa
Revises: 
Create Date: 2025-11-17 19:27:52.075869

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '2858b78908aa'
down_revision: Union[str, Sequence[str], None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade():
    """Create complete database schema"""
    
    # ===== CREATE BASE TABLES =====
    
    # Models table
    op.create_table(
        'models',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('name', sa.String(length=255), nullable=False),
        sa.Column('description', sa.Text(), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()')),
        sa.PrimaryKeyConstraint('id')
    )
    
    # Experiments table
    op.create_table(
        'experiments',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('name', sa.String(length=255), nullable=False),
        sa.Column('baseline_model_id', sa.Integer(), nullable=False),
        sa.Column('has_ground_truth', sa.Boolean(), server_default='0'),
        sa.Column('sample_count', sa.Integer(), server_default='0'),
        sa.Column('generate_baseline', sa.Boolean(), nullable=True),
        sa.Column('status', sa.String(length=50), server_default='created'),
        sa.Column('progress', sa.Integer(), server_default='0'),
        sa.Column('error_message', sa.String(length=1024), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()')),
        sa.Column('updated_at', sa.DateTime(timezone=True), nullable=True),
        sa.ForeignKeyConstraint(['baseline_model_id'], ['models.id']),
        sa.PrimaryKeyConstraint('id')
    )
    
    # Dataset samples table
    op.create_table(
        'dataset_samples',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('experiment_id', sa.Integer(), nullable=False),
        sa.Column('input_text', sa.Text(), nullable=False),
        sa.Column('ground_truth_output', sa.Text(), nullable=True),
        sa.Column('position', sa.Integer(), nullable=False, server_default='0'),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()')),
        sa.ForeignKeyConstraint(['experiment_id'], ['experiments.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index('ix_dataset_samples_experiment_id', 'dataset_samples', ['experiment_id'])
    
    # ===== CREATE NEW TABLES =====
    
    # Model variants table
    op.create_table(
        'model_variants',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('experiment_id', sa.Integer(), nullable=False),
        sa.Column('variant_type', sa.String(length=50), nullable=False),
        sa.Column('model_name', sa.String(length=255), nullable=False),
        sa.Column('quantization_level', sa.String(length=50), nullable=True),
        sa.Column('model_path', sa.String(length=512), nullable=True),
        sa.Column('inference_provider', sa.String(length=50), nullable=False, server_default='gguf'),
        sa.Column('generation_params', sa.JSON(), nullable=True),
        sa.Column('status', sa.String(length=50), server_default='pending'),
        sa.Column('progress', sa.Float(), server_default='0.0'),
        sa.Column('error_message', sa.String(length=1024), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()')),
        sa.Column('updated_at', sa.DateTime(timezone=True), nullable=True),
        sa.ForeignKeyConstraint(['experiment_id'], ['experiments.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index('ix_model_variants_experiment_id', 'model_variants', ['experiment_id'])
    
    # Generated outputs table
    op.create_table(
        'generated_outputs',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('sample_id', sa.Integer(), nullable=False),
        sa.Column('variant_id', sa.Integer(), nullable=False),
        sa.Column('output_text', sa.Text(), nullable=False),
        sa.Column('latency_ms', sa.Float(), nullable=True),
        sa.Column('token_count', sa.Integer(), nullable=True),
        sa.Column('tokens_per_second', sa.Float(), nullable=True),
        sa.Column('repetition_score', sa.Float(), nullable=True),
        sa.Column('generation_params_used', sa.JSON(), nullable=True),
        sa.Column('prompt_template', sa.Text(), nullable=True),
        sa.Column('generation_error', sa.String(length=1024), nullable=True),
        sa.Column('is_successful', sa.Integer(), nullable=False, server_default='1'),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()')),
        sa.ForeignKeyConstraint(['sample_id'], ['dataset_samples.id'], ondelete='CASCADE'),
        sa.ForeignKeyConstraint(['variant_id'], ['model_variants.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index('ix_generated_outputs_sample_id', 'generated_outputs', ['sample_id'])
    op.create_index('ix_generated_outputs_variant_id', 'generated_outputs', ['variant_id'])
    
    # Comparative metrics table
    op.create_table(
        'comparative_metrics',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('variant_id', sa.Integer(), nullable=False),
        sa.Column('experiment_id', sa.Integer(), nullable=False),
        sa.Column('model_size_mb', sa.Float(), nullable=True),
        sa.Column('avg_latency_ms', sa.Float(), nullable=True),
        sa.Column('avg_token_count', sa.Float(), nullable=True),
        sa.Column('avg_tokens_per_second', sa.Float(), nullable=True),
        sa.Column('avg_repetition_score', sa.Float(), nullable=True),
        sa.Column('bertscore_precision_vs_gt', sa.Float(), nullable=True),
        sa.Column('bertscore_recall_vs_gt', sa.Float(), nullable=True),
        sa.Column('bertscore_f1_vs_gt', sa.Float(), nullable=True),
        sa.Column('cosine_similarity_vs_gt', sa.Float(), nullable=True),
        sa.Column('avg_perplexity_vs_gt', sa.Float(), nullable=True),
        sa.Column('bertscore_precision_vs_baseline', sa.Float(), nullable=True),
        sa.Column('bertscore_recall_vs_baseline', sa.Float(), nullable=True),
        sa.Column('bertscore_f1_vs_baseline', sa.Float(), nullable=True),
        sa.Column('cosine_similarity_vs_baseline', sa.Float(), nullable=True),
        sa.Column('output_divergence_score', sa.Float(), nullable=True),
        sa.Column('evaluation_status', sa.String(length=50), server_default='pending'),
        sa.Column('evaluation_error', sa.String(length=1024), nullable=True),
        sa.Column('samples_evaluated', sa.Integer(), server_default='0'),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()')),
        sa.Column('updated_at', sa.DateTime(timezone=True), nullable=True),
        sa.ForeignKeyConstraint(['variant_id'], ['model_variants.id'], ondelete='CASCADE'),
        sa.ForeignKeyConstraint(['experiment_id'], ['experiments.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('variant_id')
    )
    op.create_index('ix_comparative_metrics_variant_id', 'comparative_metrics', ['variant_id'])
    op.create_index('ix_comparative_metrics_experiment_id', 'comparative_metrics', ['experiment_id'])


def downgrade():
    """Drop all tables"""
    op.drop_table('comparative_metrics')
    op.drop_table('generated_outputs')
    op.drop_table('model_variants')
    op.drop_table('dataset_samples')
    op.drop_table('experiments')
    op.drop_table('models')