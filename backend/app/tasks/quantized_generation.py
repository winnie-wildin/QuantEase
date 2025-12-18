"""
Celery task for quantized model generation using GGUF models.
Generates outputs for all samples using locally loaded quantized models.

Production-ready implementation: Loads model once and processes samples sequentially.
Parallelism is achieved via Celery worker-level distribution (multiple variants run on different workers).
"""
from app.tasks.celery_app import celery_app
from app.database import SessionLocal
from app.models import Experiment, ModelVariant, DatasetSample, GeneratedOutput
from app.utils.gguf_loader import GGUFLoader
from app.utils.task_prompt_builder import TaskPromptBuilder
from sqlalchemy.orm import Session
import time
import sys


@celery_app.task(bind=True, name="generate_quantized_outputs")
def generate_quantized_outputs(self, experiment_id: int, variant_id: int):
    """
    Generate quantized outputs using GGUF model.
    
    Production approach: Loads model once and processes samples sequentially.
    Parallelism is achieved at the Celery worker level - multiple variants 
    run concurrently on different workers.
    
    Args:
        experiment_id: ID of the experiment
        variant_id: ID of the quantized model variant
    
    Returns:
        Dict with status and results
    """
    db = SessionLocal()
    loader = None
    
    try:
        # Get experiment and variant
        experiment = db.query(Experiment).filter(Experiment.id == experiment_id).first()
        variant = db.query(ModelVariant).filter(ModelVariant.id == variant_id).first()
        
        if not experiment or not variant:
            raise ValueError(f"Experiment {experiment_id} or Variant {variant_id} not found")
        
        if not variant.model_path:
            raise ValueError(f"No model path specified for variant {variant_id}")
        
        # Get all samples for this experiment
        samples = db.query(DatasetSample).filter(
            DatasetSample.experiment_id == experiment_id
        ).order_by(DatasetSample.position).all()
        
        if not samples:
            raise ValueError(f"No samples found for experiment {experiment_id}")
        
        # Start overall timing
        overall_start_time = time.time()
        model_load_start = time.time()
        
        print(f"\n{'='*60}", flush=True)
        print(f"🚀 Starting quantized generation", flush=True)
        print(f"{'='*60}", flush=True)
        print(f"📝 Samples: {len(samples)}", flush=True)
        print(f"🔧 Model: {variant.model_path}", flush=True)
        print(f"⏰ Started at: {time.strftime('%H:%M:%S')}", flush=True)
        print(f"{'='*60}\n", flush=True)
        
        # Load model once (production approach - memory efficient)
        print(f"📦 Loading model...", flush=True)
        loader = GGUFLoader(
            model_path=variant.model_path,
            n_ctx=2048,
            n_threads=4
        )
        loader.load()
        model_load_time = (time.time() - model_load_start) * 1000
        print(f"✅ Model loaded in {model_load_time:.0f}ms\n", flush=True)
        
        # Update variant status
        variant.status = "generating"
        variant.progress = 0.0
        db.commit()
        
        total_samples = len(samples)
        successful_count = 0
        failed_count = 0
        # Process samples sequentially (production approach)
        # Parallelism achieved via Celery workers handling different variants
        for i, sample in enumerate(samples):
            # Check if task has been cancelled (check variant status in DB)
            db.refresh(variant)
            if variant.status == "cancelled":
                print(f"\n🛑 Generation cancelled by user. Stopping at sample {i+1}/{total_samples}", flush=True)
                loader.unload()
                variant.progress = i / total_samples
                db.commit()
                return {
                    "status": "cancelled",
                    "experiment_id": experiment_id,
                    "variant_id": variant_id,
                    "samples_processed": i,
                    "total_samples": total_samples,
                    "message": "Generation cancelled by user"
                }
            
            try:
                print(f"\n📝 Sample {i+1}/{total_samples} (ID: {sample.id})", flush=True)
                print(f"📥 Input: {sample.input_text[:100]}...", flush=True)
                
                # Build task-specific prompt
                task_prompt = sample.input_text  # Default to raw input
                
                if experiment.task_type:
                    try:
                        if experiment.task_type == "classification":
                            labels = experiment.detected_labels or []
                            if labels:
                                task_prompt = TaskPromptBuilder.build(
                                    task_type="classification",
                                    input_text=sample.input_text,
                                    labels=labels
                                )
                                print(f"🏷️  Using classification prompt with labels: {labels}", flush=True)
                        
                        elif experiment.task_type == "rag":
                            task_prompt = TaskPromptBuilder.build(
                                task_type="rag",
                                input_text=sample.input_text,
                                context=sample.context or ""
                            )
                            print(f"🔍 Using RAG prompt with context", flush=True)
                        
                        elif experiment.task_type == "text_generation":
                            task_prompt = TaskPromptBuilder.build(
                                task_type="text_generation",
                                input_text=sample.input_text
                            )
                    except Exception as e:
                        print(f"⚠️  Task prompt building failed, using raw input: {e}", flush=True)
                        task_prompt = sample.input_text
                
                # Task-aware max_tokens
                if experiment.task_type == 'classification':
                    max_tokens = 10  # Classification needs only a label
                elif experiment.task_type == 'rag':
                    max_tokens = 256  # RAG needs full answers
                else:
                    max_tokens = 256  # Text generation needs full responses
                
                # Generate using loaded model
                t0 = time.time()
                result = loader.generate(
                    prompt=task_prompt,
                    max_tokens=max_tokens,
                    temperature=0.7
                )
                t1 = time.time()
                
                # Extract output
                output_text = result["output_text"].strip()
                
                # Validate output (task-aware minimum length)
                min_length = 3 if experiment.task_type == 'classification' else 10
                
                if not output_text:
                    raise ValueError(f"Empty output after generation")
                
                if len(output_text) < min_length:
                    raise ValueError(f"Output too short ({len(output_text)} chars, min: {min_length})")
                
                if output_text.startswith("[Generation failed"):
                    raise ValueError("Generation failed after retries")
                
                print(f"📤 Output: {output_text[:150]}...", flush=True)
                
                # Calculate metrics
                latency_ms = result.get('latency_ms', (t1 - t0) * 1000.0)
                token_count = result.get('token_count', len(output_text.split()))
                tokens_per_second = result.get('tokens_per_second', token_count / max((t1 - t0), 1e-6))
                
                print(f"⏱️  Latency: {latency_ms:.0f}ms", flush=True)
                print(f"🚀 Speed: {tokens_per_second:.1f} tok/s", flush=True)
                
                # Create GeneratedOutput record
                output = GeneratedOutput(
                    sample_id=sample.id,
                    variant_id=variant.id,
                    output_text=output_text,
                    latency_ms=latency_ms,
                    token_count=token_count,
                    tokens_per_second=tokens_per_second,
                    generation_params_used={
                        "model_path": variant.model_path,
                        "max_tokens": max_tokens,
                        "temperature": 0.7,
                        "quantization": variant.quantization_level,
                        "model_load_time_ms": model_load_time
                    },
                    is_successful=1,
                    generation_error=None
                )
                db.add(output)
                successful_count += 1
                
                # Update progress
                progress = (i + 1) / total_samples
                variant.progress = progress
                db.commit()
                db.refresh(variant)  # Ensure state is current
                
                # Update task state for real-time tracking
                self.update_state(
                    state='PROGRESS',
                    meta={
                        'current': i + 1,
                        'total': total_samples,
                        'successful': successful_count,
                        'failed': failed_count,
                        'status': f'Generated {i + 1}/{total_samples} outputs'
                    }
                )
                
                progress_pct = progress * 100
                print(f"✅ Generated output {i+1}/{total_samples} for sample {sample.id} | Progress: {progress_pct:.1f}%", flush=True)
                
            except Exception as e:
                print(f"❌ Error generating output for sample {sample.id}: {e}", flush=True)
                import traceback
                traceback.print_exc()
                
                failed_count += 1
                
                # Store failed output
                output = GeneratedOutput(
                    sample_id=sample.id,
                    variant_id=variant.id,
                    output_text="[Generation Error]",
                    latency_ms=0,
                    token_count=0,
                    tokens_per_second=0,
                    is_successful=0,
                    generation_error=str(e)
                )
                db.add(output)
                db.commit()
        
        # Unload model
        if loader:
            print(f"\n📦 Unloading model...", flush=True)
            loader.unload()
        
        # Calculate final timing
        overall_elapsed = time.time() - overall_start_time
        
        # Update final status
        if failed_count == 0:
            variant.status = "completed"
            print(f"\n🎉 All {successful_count} samples generated successfully!", flush=True)
        elif successful_count > 0:
            variant.status = "completed"
            variant.error_message = f"{failed_count} samples failed"
            print(f"\n⚠️  Completed with {failed_count} failures", flush=True)
        else:
            variant.status = "failed"
            variant.error_message = "All samples failed to generate"
            print(f"\n❌ All samples failed!", flush=True)
        
        variant.progress = 1.0
        db.commit()
        
        # Timing summary
        generation_time = overall_elapsed - (model_load_time / 1000)
        print(f"\n{'='*60}", flush=True)
        print(f"⏱️  TIMING SUMMARY", flush=True)
        print(f"{'='*60}", flush=True)
        print(f"Model load time: {model_load_time:.0f}ms", flush=True)
        print(f"Generation time: {generation_time:.1f}s", flush=True)
        print(f"Total time: {overall_elapsed:.1f}s ({overall_elapsed/60:.2f} minutes)", flush=True)
        print(f"Samples processed: {successful_count}/{total_samples}", flush=True)
        print(f"Average per sample: {generation_time/total_samples:.1f}s (excluding model load)", flush=True)
        print(f"Throughput: {total_samples/(overall_elapsed/60):.1f} samples/min", flush=True)
        print(f"{'='*60}\n", flush=True)
        
        result = {
            "status": "completed" if failed_count == 0 else "completed_with_errors",
            "experiment_id": experiment_id,
            "variant_id": variant_id,
            "total_samples": total_samples,
            "successful": successful_count,
            "failed": failed_count,
            "total_time_seconds": overall_elapsed,
            "model_load_time_ms": model_load_time,
            "samples_per_minute": total_samples/(overall_elapsed/60),
            "message": f"Generated {successful_count}/{total_samples} outputs successfully"
        }
        
        print(f"🎉 Quantized generation complete: {result['message']}", flush=True)
        print(f"⚡ Processed {result['samples_per_minute']:.1f} samples/min\n", flush=True)
        return result
        
    except Exception as e:
        # Update variant status on failure
        variant = None
        try:
            variant = db.query(ModelVariant).filter(ModelVariant.id == variant_id).first()
            if variant:
                variant.status = "failed"
                variant.error_message = str(e)
                db.commit()
        except:
            pass
        
        print(f"❌ Fatal error in quantized generation: {e}", flush=True)
        import traceback
        traceback.print_exc()
        sys.stdout.flush()
        raise
        
    finally:
        # Clean up model if still loaded
        if loader:
            try:
                loader.unload()
            except:
                pass
        db.close()