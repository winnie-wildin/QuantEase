"""
Complete Evaluation Test - Full QuantEase Pipeline
Tests baseline generation, quantized generation, and evaluation metrics
"""
from app.database import SessionLocal
from app.models import (
    Experiment, DatasetSample, ModelVariant, 
    GeneratedOutput, ComparativeMetrics
)
from app.tasks.baseline_generation import generate_baseline_outputs
from app.tasks.quantized_generation import generate_quantized_outputs
from app.tasks.evaluation_task import evaluate_variant
from sqlalchemy import text

print("="*70)
print("🧪 COMPLETE QUANTEASE EVALUATION TEST")
print("="*70)
print("\nThis will test:")
print("  ✅ Phase 2: Baseline generation (Groq)")
print("  ✅ Phase 3: Quantized generation (GGUF)")
print("  ✅ Phase 4: Evaluation metrics (BERTScore, similarity)")
print("\n" + "="*70 + "\n")

db = SessionLocal()

try:
    # ===== SETUP =====
    print("📋 SETUP: Creating experiment with ground truth\n")
    
    # Create test model
    print("1️⃣ Creating test model...")
    db.execute(text("INSERT INTO models (name, description) VALUES ('comparison-test', 'Full comparison test')"))
    db.commit()
    model_id = db.execute(text("SELECT id FROM models WHERE name='comparison-test'")).fetchone()[0]
    print(f"   ✅ Model ID: {model_id}\n")
    
    # Create experiment with ground truth
    print("2️⃣ Creating experiment with ground truth...")
    exp = Experiment(
        name="complete_evaluation_test",
        baseline_model_id=model_id,
        has_ground_truth=True,  # ✅ Has ground truth for evaluation
        sample_count=3,
        status="created"
    )
    db.add(exp)
    db.commit()
    print(f"   ✅ Experiment ID: {exp.id}")
    print(f"   ✅ Has ground truth: {exp.has_ground_truth}\n")
    
    # Create samples with ground truth
    print("3️⃣ Creating samples with ground truth...")
    samples = [
        DatasetSample(
            experiment_id=exp.id,
            input_text="What is artificial intelligence?",
            ground_truth_output="Artificial intelligence is the simulation of human intelligence by machines, particularly computer systems. It includes learning, reasoning, and self-correction.",
            position=0
        ),
        DatasetSample(
            experiment_id=exp.id,
            input_text="What is machine learning?",
            ground_truth_output="Machine learning is a subset of AI that enables systems to learn and improve from experience without being explicitly programmed. It uses algorithms to parse data and make predictions.",
            position=1
        ),
        DatasetSample(
            experiment_id=exp.id,
            input_text="What is deep learning?",
            ground_truth_output="Deep learning is a subset of machine learning based on artificial neural networks with multiple layers. It excels at processing unstructured data like images, audio, and text.",
            position=2
        ),
    ]
    db.add_all(samples)
    db.commit()
    print(f"   ✅ Created {len(samples)} samples with ground truth\n")
    
    # ===== PHASE 2: BASELINE GENERATION =====
    print("="*70)
    print("🚀 PHASE 2: BASELINE GENERATION (GROQ)")
    print("="*70 + "\n")
    
    print("4️⃣ Creating baseline variant...")
    baseline_variant = ModelVariant(
        experiment_id=exp.id,
        variant_type="baseline",
        model_name="llama-3.3-70b-versatile",
        inference_provider="groq",
        status="pending"
    )
    db.add(baseline_variant)
    db.commit()
    print(f"   ✅ Baseline variant ID: {baseline_variant.id}\n")
    
    print("5️⃣ Generating baseline outputs...")
    baseline_result = generate_baseline_outputs(exp.id, baseline_variant.id)
    print(f"   ✅ {baseline_result['message']}\n")
    
    # ===== PHASE 3: QUANTIZED GENERATION =====
    print("="*70)
    print("🚀 PHASE 3: QUANTIZED GENERATION (GGUF)")
    print("="*70 + "\n")
    
    print("6️⃣ Creating quantized variant...")
    quantized_variant = ModelVariant(
        experiment_id=exp.id,
        variant_type="quantized",
        model_name="tinyllama-1.1b-chat",
        quantization_level="INT4",
        model_path="data/models/quantized/tinyllama-1.1b.gguf",
        inference_provider="gguf",
        status="pending"
    )
    db.add(quantized_variant)
    db.commit()
    print(f"   ✅ Quantized variant ID: {quantized_variant.id}\n")
    
    print("7️⃣ Generating quantized outputs...")
    quantized_result = generate_quantized_outputs(exp.id, quantized_variant.id)
    print(f"   ✅ {quantized_result['message']}\n")
    
    # ===== PHASE 4: EVALUATION =====
    print("="*70)
    print("🚀 PHASE 4: EVALUATION METRICS")
    print("="*70 + "\n")
    
    print("8️⃣ Evaluating baseline variant...")
    print("   (Calculating BERTScore, cosine similarity vs ground truth...)")
    baseline_eval = evaluate_variant(baseline_variant.id)
    print(f"   ✅ Baseline evaluation complete!\n")
    
    print("9️⃣ Evaluating quantized variant...")
    print("   (Calculating metrics vs ground truth AND baseline...)")
    quantized_eval = evaluate_variant(quantized_variant.id)
    print(f"   ✅ Quantized evaluation complete!\n")
    
    # ===== RESULTS =====
    print("="*70)
    print("📊 EVALUATION RESULTS")
    print("="*70 + "\n")
    
    # Refresh from DB
    db.refresh(baseline_variant)
    db.refresh(quantized_variant)
    
    baseline_metrics = baseline_variant.metrics
    quantized_metrics = quantized_variant.metrics
    
    print("BASELINE (Groq - llama-3.3-70b):")
    print("-" * 70)
    print(f"  ⏱️  Avg Latency:        {baseline_metrics.avg_latency_ms:.2f}ms")
    print(f"  🚀 Avg Speed:          {baseline_metrics.avg_tokens_per_second:.2f} tok/s")
    print(f"  📏 Avg Tokens:         {baseline_metrics.avg_token_count:.1f}")
    print(f"  🎯 BERTScore F1 (GT):  {baseline_metrics.bertscore_f1_vs_gt:.4f}")
    print(f"  🔗 Similarity (GT):    {baseline_metrics.cosine_similarity_vs_gt:.4f}")
    print()
    
    print("QUANTIZED (GGUF - tinyllama-1.1b INT4):")
    print("-" * 70)
    print(f"  ⏱️  Avg Latency:        {quantized_metrics.avg_latency_ms:.2f}ms")
    print(f"  🚀 Avg Speed:          {quantized_metrics.avg_tokens_per_second:.2f} tok/s")
    print(f"  📏 Avg Tokens:         {quantized_metrics.avg_token_count:.1f}")
    print(f"  💾 Model Size:         {quantized_metrics.model_size_mb:.2f} MB")
    print(f"  🎯 BERTScore F1 (GT):  {quantized_metrics.bertscore_f1_vs_gt:.4f}")
    print(f"  🔗 Similarity (GT):    {quantized_metrics.cosine_similarity_vs_gt:.4f}")
    print()
    
    print("COMPARISON (Quantized vs Baseline):")
    print("-" * 70)
    print(f"  🎯 BERTScore F1:       {quantized_metrics.bertscore_f1_vs_baseline:.4f}")
    print(f"  🔗 Similarity:         {quantized_metrics.cosine_similarity_vs_baseline:.4f}")
    print(f"  📊 Divergence:         {quantized_metrics.output_divergence_score:.4f}")
    print()
    
    # Speed comparison
    speedup = baseline_metrics.avg_tokens_per_second / quantized_metrics.avg_tokens_per_second
    latency_diff = quantized_metrics.avg_latency_ms / baseline_metrics.avg_latency_ms
    
    print("TRADE-OFFS:")
    print("-" * 70)
    print(f"  ⚡ Groq is {speedup:.1f}x faster than local GGUF")
    print(f"  ⏱️  Quantized has {latency_diff:.1f}x higher latency")
    print(f"  💾 Quantized uses only 638MB (vs 70B params in cloud)")
    print(f"  🎯 Quality retention: {quantized_metrics.bertscore_f1_vs_baseline:.1%}")
    print()
    
    # Cleanup
    print("="*70)
    print("🧹 CLEANUP")
    print("="*70 + "\n")
    
    print("Cleaning up test data...")
    db.delete(quantized_metrics)
    db.delete(baseline_metrics)
    
    for variant in [baseline_variant, quantized_variant]:
        outputs = db.query(GeneratedOutput).filter(GeneratedOutput.variant_id == variant.id).all()
        for output in outputs:
            db.delete(output)
        db.delete(variant)
    
    for sample in samples:
        db.delete(sample)
    
    db.delete(exp)
    db.commit()
    db.execute(text("DELETE FROM models WHERE name='comparison-test'"))
    db.commit()
    print("✅ Cleanup complete\n")
    
    # Final summary
    print("="*70)
    print("🎉 COMPLETE EVALUATION TEST PASSED!")
    print("="*70)
    print("\n✅ All phases working:")
    print("   • Phase 1: Database schema ✅")
    print("   • Phase 2: Baseline generation (Groq) ✅")
    print("   • Phase 3: Quantized generation (GGUF) ✅")
    print("   • Phase 4: Evaluation metrics ✅")
    print("\n🚀 QuantEase backend is fully operational!")
    print("\n📋 Next steps:")
    print("   • Phase 5: API endpoints")
    print("   • Phase 6-7: Frontend dashboard")
    print()
    
except Exception as e:
    print(f"\n❌ Error: {e}")
    import traceback
    traceback.print_exc()
    db.rollback()
finally:
    db.close()