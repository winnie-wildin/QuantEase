"""
Celery task for quantized model generation using GGUF models.
Generates outputs for all samples using locally loaded quantized models.
"""
from app.tasks.celery_app import celery_app
from app.database import SessionLocal
from app.models import Experiment, ModelVariant, DatasetSample, GeneratedOutput
from app.utils.gguf_loader import GGUFLoader
from sqlalchemy.orm import Session
import time


@celery_app.task(bind=True, name="generate_quantized_outputs")
def generate_quantized_outputs(self, experiment_id: int, variant_id: int):
    """
    Generate quantized outputs using GGUF model.
    
    Args:
        experiment_id: ID of the experiment
        variant_id: ID of the quantized model variant
    
    Returns:
        Dict with status and results
    """
    db = SessionLocal()
    
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
        
        # Load GGUF model
        print(f"Loading GGUF model: {variant.model_path}")
        loader = GGUFLoader(
            model_path=variant.model_path,
            n_ctx=2048,
            n_threads=4
        )
        loader.load()
        
        # Update variant status
        variant.status = "generating"
        variant.progress = 0.0
        db.commit()
        
        total_samples = len(samples)
        successful_count = 0
        failed_count = 0
        
        # Generate output for each sample
        for i, sample in enumerate(samples):
            try:
                print(f"\n{'='*60}")
                print(f"📝 Sample {i+1}/{total_samples} (ID: {sample.id})")
                print(f"{'='*60}")
                
                # Use raw input text — formatter handles prompt formatting now
                print(f"📥 Input: {sample.input_text[:100]}...")
                
                # Generate (formatter handles prompt formatting automatically)
                result = loader.generate(
                    prompt=sample.input_text,  # Raw prompt - loader will format it
                    max_tokens=256,
                    temperature=0.7
                )
                
                # Validate output
                output_text = result["output_text"].strip()
                is_successful = True
                error_message = None
                
                if not output_text:
                    print(f"⚠️  Empty output generated!")
                    is_successful = False
                    error_message = "Empty output after generation"
                    output_text = "[No output generated]"
                    
                elif len(output_text) < 5:
                    print(f"⚠️  Output too short: '{output_text}'")
                    is_successful = False
                    error_message = f"Output too short ({len(output_text)} chars)"
                    
                elif output_text.startswith("[Generation failed"):
                    print(f"⚠️  Generation failed marker detected")
                    is_successful = False
                    error_message = "Generation failed after retries"
                    
                else:
                    print(f"✅ Valid output generated!")
                
                print(f"📤 Output: {output_text[:150]}...")
                print(f"📊 Stats: {len(output_text)} chars, {result['token_count']} tokens, {result['latency_ms']:.0f}ms")
                print(f"🚀 Speed: {result['tokens_per_second']:.2f} tok/s")
                print(f"{'='*60}\n")
                
                # Create GeneratedOutput record with validation
                output = GeneratedOutput(
                    sample_id=sample.id,
                    variant_id=variant.id,
                    output_text=output_text,
                    latency_ms=result["latency_ms"],
                    token_count=result["token_count"],
                    tokens_per_second=result["tokens_per_second"],
                    generation_params_used={
                        "model_path": result["model_path"],
                        "max_tokens": 256,
                        "temperature": 0.7,
                        "quantization": variant.quantization_level,
                        "prompt_format": "tinyllama_instruct"
                    },
                    is_successful=1 if is_successful else 0,
                    generation_error=error_message or result.get("error")
                )
                db.add(output)
                
                if is_successful:
                    successful_count += 1
                else:
                    failed_count += 1
                
                # Update progress
                progress = (i + 1) / total_samples
                variant.progress = progress
                db.commit()
                
                # Update task state
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
                
                print(f"✅ Generated output {i+1}/{total_samples} for sample {sample.id}")
                
            except Exception as e:
                print(f"❌ Error generating output for sample {sample.id}: {e}")
                import traceback
                traceback.print_exc()
                
                failed_count += 1
                
                # Store failed output with error details
                output = GeneratedOutput(
                    sample_id=sample.id,
                    variant_id=variant.id,
                    output_text="[Generation Error]",
                    latency_ms=0,
                    token_count=0,
                    tokens_per_second=0,
                    is_successful=0,
                    generation_error=f"Exception: {str(e)}"
                )
                db.add(output)
                db.commit()
        
        # Unload model to free memory
        print("\n🧹 Unloading model from memory...")
        loader.unload()
        
        # Update final status
        if failed_count == 0:
            variant.status = "completed"
            print(f"🎉 All {successful_count} samples generated successfully!")
        elif successful_count > 0:
            variant.status = "completed"
            variant.error_message = f"{failed_count} samples failed"
            print(f"⚠️  Completed with {failed_count} failures")
        else:
            variant.status = "failed"
            variant.error_message = "All samples failed to generate"
            print(f"❌ All samples failed!")
        
        variant.progress = 1.0
        db.commit()
        
        result = {
            "status": "completed" if failed_count == 0 else "completed_with_errors",
            "experiment_id": experiment_id,
            "variant_id": variant_id,
            "total_samples": total_samples,
            "successful": successful_count,
            "failed": failed_count,
            "message": f"Generated {successful_count}/{total_samples} outputs successfully"
        }
        
        print(f"\n🎉 Quantized generation complete: {result['message']}")
        return result
        
    except Exception as e:
        # Update variant status on failure
        if variant:
            variant.status = "failed"
            variant.error_message = str(e)
            db.commit()
        
        print(f"❌ Fatal error in quantized generation: {e}")
        import traceback
        traceback.print_exc()
        raise
        
    finally:
        db.close()