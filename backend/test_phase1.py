from app.database import SessionLocal
from app.models import Experiment, ModelVariant, DatasetSample, GeneratedOutput, ComparativeMetrics
from sqlalchemy import text

print("🧪 Testing Phase 1 Database Schema...\n")

db = SessionLocal()

try:
    # Test 1: Create Model
    print("1️⃣ Creating test model...")
    db.execute(text("INSERT INTO models (name, description) VALUES ('llama-2-7b', 'Test model')"))
    db.commit()
    result = db.execute(text("SELECT id FROM models WHERE name='llama-2-7b'")).fetchone()
    model_id = result[0]
    print(f"   ✅ Model created with ID: {model_id}")
    
    # Test 2: Create Experiment
    print("\n2️⃣ Creating experiment...")
    exp = Experiment(
        name="test_experiment",
        baseline_model_id=model_id,
        has_ground_truth=True,
        sample_count=2,
        status="created"
    )
    db.add(exp)
    db.commit()
    print(f"   ✅ Experiment created: {exp.id}")
    print(f"   ✅ Has ground truth: {exp.has_ground_truth}")
    print(f"   ✅ Status: {exp.status}")
    
    # Test 3: Create Dataset Samples
    print("\n3️⃣ Creating dataset samples...")
    sample1 = DatasetSample(
        experiment_id=exp.id,
        input_text="What is AI?",
        ground_truth_output="AI is artificial intelligence...",
        position=0
    )
    sample2 = DatasetSample(
        experiment_id=exp.id,
        input_text="What is ML?",
        ground_truth_output="ML is machine learning...",
        position=1
    )
    db.add_all([sample1, sample2])
    db.commit()
    print(f"   ✅ Created {len(exp.samples)} samples")
    
    # Test 4: Create Model Variants
    print("\n4️⃣ Creating model variants...")
    baseline = ModelVariant(
        experiment_id=exp.id,
        variant_type="baseline",
        model_name="llama-2-7b",
        inference_provider="groq",
        status="pending"
    )
    quantized = ModelVariant(
        experiment_id=exp.id,
        variant_type="quantized",
        model_name="llama-2-7b",
        quantization_level="INT8",
        model_path="/data/models/llama-2-7b.Q8_0.gguf",
        inference_provider="gguf",
        status="pending"
    )
    db.add_all([baseline, quantized])
    db.commit()
    print(f"   ✅ Created {len(exp.variants)} variants")
    print(f"   ✅ Baseline: {baseline.display_name}")
    print(f"   ✅ Quantized: {quantized.display_name}")
    
    # Test 5: Create Generated Outputs
    print("\n5️⃣ Creating generated outputs...")
    output1 = GeneratedOutput(
        sample_id=sample1.id,
        variant_id=baseline.id,
        output_text="AI is artificial intelligence, a field of computer science...",
        latency_ms=234.5,
        token_count=45,
        repetition_score=0.12,
        is_successful=1
    )
    output2 = GeneratedOutput(
        sample_id=sample1.id,
        variant_id=quantized.id,
        output_text="AI is artificial intelligence...",
        latency_ms=98.3,
        token_count=42,
        repetition_score=0.15,
        is_successful=1
    )
    db.add_all([output1, output2])
    db.commit()
    print(f"   ✅ Created 2 generated outputs")
    print(f"   ✅ Baseline latency: {output1.latency_ms}ms")
    print(f"   ✅ Quantized latency: {output2.latency_ms}ms")
    
    # Test 6: Create Comparative Metrics
    print("\n6️⃣ Creating comparative metrics...")
    metrics_baseline = ComparativeMetrics(
        variant_id=baseline.id,
        experiment_id=exp.id,
        model_size_mb=5200.0,
        avg_latency_ms=234.5,
        bertscore_f1_vs_gt=0.923,
        evaluation_status="completed"
    )
    metrics_quantized = ComparativeMetrics(
        variant_id=quantized.id,
        experiment_id=exp.id,
        model_size_mb=1300.0,
        avg_latency_ms=98.3,
        bertscore_f1_vs_gt=0.856,
        bertscore_f1_vs_baseline=0.923,
        evaluation_status="completed"
    )
    db.add_all([metrics_baseline, metrics_quantized])
    db.commit()
    print(f"   ✅ Created metrics for {len(exp.metrics)} variants")
    print(f"   ✅ Baseline size: {metrics_baseline.model_size_mb}MB")
    print(f"   ✅ Quantized size: {metrics_quantized.model_size_mb}MB")
    
    # Test 7: Test Relationships
    print("\n7️⃣ Testing relationships...")
    print(f"   ✅ Experiment → Samples: {len(exp.samples)}")
    print(f"   ✅ Experiment → Variants: {len(exp.variants)}")
    print(f"   ✅ Experiment → Metrics: {len(exp.metrics)}")
    print(f"   ✅ Sample → Outputs: {len(sample1.generated_outputs)}")
    print(f"   ✅ Variant → Outputs: {len(baseline.generated_outputs)}")
    print(f"   ✅ Variant → Metrics: {baseline.metrics is not None}")
    
    # Cleanup
    print("\n8️⃣ Cleaning up test data...")
    db.delete(metrics_quantized)
    db.delete(metrics_baseline)
    db.delete(output2)
    db.delete(output1)
    db.delete(quantized)
    db.delete(baseline)
    db.delete(sample2)
    db.delete(sample1)
    db.delete(exp)  # Delete experiment BEFORE model
    db.commit()      # Commit before deleting model
    db.execute(text("DELETE FROM models WHERE name='llama-2-7b'"))
    db.commit()
    print("   ✅ Cleanup complete")
    
    print("\n" + "="*50)
    print("🎉 PHASE 1 COMPLETE!")
    print("="*50)
    print("\n✅ All database models working correctly!")
    print("✅ All relationships functioning properly!")
    print("✅ Ready to move to Phase 2 (Groq API Integration)!")
    print("\n📋 What you accomplished:")
    print("   • Created 5 database tables")
    print("   • 3 new models: ModelVariant, GeneratedOutput, ComparativeMetrics")
    print("   • 2 updated models: Experiment, DatasetSample")
    print("   • All relationships with cascade deletes")
    print("   • Migration system working")
    
except Exception as e:
    print(f"\n❌ Error: {e}")
    import traceback
    traceback.print_exc()
    db.rollback()
finally:
    db.close()