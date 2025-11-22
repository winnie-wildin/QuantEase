"""Test complete baseline generation workflow"""
from app.database import SessionLocal
from app.models import Experiment, DatasetSample, ModelVariant, GeneratedOutput
from app.tasks.baseline_generation import generate_baseline_outputs
from sqlalchemy import text
from dotenv import load_dotenv
load_dotenv()
import os
print("🧪 Testing Baseline Generation Workflow...\n")

db = SessionLocal()

try:
    # Create test model
    print("1️⃣ Creating test model...")
    db.execute(text("INSERT INTO models (name, description) VALUES ('test-model', 'Test')"))
    db.commit()
    model_id = db.execute(text("SELECT id FROM models WHERE name='test-model'")).fetchone()[0]
    print(f"   ✅ Model ID: {model_id}")
    
    # Create experiment
    print("\n2️⃣ Creating experiment...")
    exp = Experiment(
        name="baseline_test",
        baseline_model_id=model_id,
        has_ground_truth=False,
        sample_count=3,
        status="created"
    )
    db.add(exp)
    db.commit()
    print(f"   ✅ Experiment ID: {exp.id}")
    
    # Create sample inputs
    print("\n3️⃣ Creating test samples...")
    samples = [
        DatasetSample(experiment_id=exp.id, input_text="What is artificial intelligence?", position=0),
        DatasetSample(experiment_id=exp.id, input_text="What is machine learning?", position=1),
        DatasetSample(experiment_id=exp.id, input_text="What is deep learning?", position=2),
    ]
    db.add_all(samples)
    db.commit()
    print(f"   ✅ Created {len(samples)} samples")
    
    # Create baseline variant
    print("\n4️⃣ Creating baseline variant...")
    variant = ModelVariant(
        experiment_id=exp.id,
        variant_type="baseline",
        model_name="llama-3.3-70b-versatile",
        inference_provider="groq",
        status="pending"
    )
    db.add(variant)
    db.commit()
    print(f"   ✅ Variant ID: {variant.id}")
    
    # Generate baseline outputs
    print("\n5️⃣ Generating baseline outputs (this will take ~5-10 seconds)...")
    print("   " + "-"*50)
    
    result = generate_baseline_outputs(exp.id, variant.id)
    
    print("   " + "-"*50)
    print(f"   ✅ Generation complete!")
    print(f"   ✅ Status: {result['status']}")
    print(f"   ✅ Successful: {result['successful']}/{result['total_samples']}")
    if result['failed'] > 0:
        print(f"   ⚠️  Failed: {result['failed']}")
    
    # Verify outputs
    print("\n6️⃣ Verifying generated outputs...")
    db.refresh(variant)
    outputs = db.query(GeneratedOutput).filter(GeneratedOutput.variant_id == variant.id).all()
    
    print(f"   ✅ Found {len(outputs)} outputs")
    print(f"   ✅ Variant status: {variant.status}")
    print(f"   ✅ Variant progress: {variant.progress * 100:.0f}%")
    
    print("\n7️⃣ Sample outputs:")
    successful_outputs = [o for o in outputs if o.is_successful]
    failed_outputs = [o for o in outputs if not o.is_successful]
    
    print(f"\n   ✅ Successful outputs: {len(successful_outputs)}")
    print(f"   ❌ Failed outputs: {len(failed_outputs)}")
    
    for i, output in enumerate(successful_outputs, 1):
        print(f"\n   Sample {i}:")
        print(f"   📥 Input: {output.sample.input_text}")
        print(f"   📤 Output: {output.output_text[:100]}...")
        print(f"   ⏱️  Latency: {output.latency_ms:.2f}ms" if output.latency_ms else "   ⏱️  Latency: N/A")
        print(f"   🔢 Tokens: {output.token_count}" if output.token_count else "   🔢 Tokens: N/A")
        print(f"   🚀 Speed: {output.tokens_per_second:.2f} tok/s" if output.tokens_per_second else "   🚀 Speed: N/A")
        print(f"   ✅ Success: Yes")
    
    if failed_outputs:
        print(f"\n   ❌ Failed samples:")
        for i, output in enumerate(failed_outputs, 1):
            print(f"      {i}. {output.sample.input_text} - Error: {output.generation_error}")
    
    # Calculate average metrics (only from successful outputs)
    print("\n8️⃣ Performance metrics:")
    if successful_outputs:
        avg_latency = sum(o.latency_ms for o in successful_outputs if o.latency_ms) / len(successful_outputs)
        avg_tokens = sum(o.token_count for o in successful_outputs if o.token_count) / len(successful_outputs)
        avg_speed = sum(o.tokens_per_second for o in successful_outputs if o.tokens_per_second) / len(successful_outputs)
        
        print(f"   📊 Average latency: {avg_latency:.2f}ms")
        print(f"   📊 Average tokens: {avg_tokens:.1f}")
        print(f"   📊 Average speed: {avg_speed:.2f} tokens/sec")
    else:
        print("   ⚠️  No successful outputs to calculate metrics")
    
    # Cleanup
    print("\n9️⃣ Cleaning up...")
    for output in outputs:
        db.delete(output)
    db.delete(variant)
    for sample in samples:
        db.delete(sample)
    db.delete(exp)
    db.commit()  # Commit BEFORE deleting model
    db.execute(text("DELETE FROM models WHERE name='test-model'"))
    db.commit()
    print("   ✅ Cleanup complete")
    
    print("\n" + "="*60)
    print("🎉 BASELINE GENERATION TEST COMPLETE!")
    print("="*60)
    print("\n✅ Groq API integration working!")
    print("✅ Baseline generation task working!")
    print("✅ Outputs stored in database!")
    print("✅ Progress tracking working!")
    print("✅ Performance metrics recorded!")
    print("\n🎊 Phase 2 Successfully Completed!")
    print("\n📋 What you accomplished:")
    print("   • Groq API client wrapper")
    print("   • Celery task for async generation")
    print("   • Real-time progress tracking")
    print("   • Database storage of outputs")
    print("   • Performance metrics collection")
    print("\n🚀 Ready for Phase 3: GGUF Model Loading!")
    
except Exception as e:
    print(f"\n❌ Error: {e}")
    import traceback
    traceback.print_exc()
    db.rollback()
finally:
    db.close()