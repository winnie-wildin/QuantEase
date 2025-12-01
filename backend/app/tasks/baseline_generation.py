#app/tasks/baseline_generation.py
"""
Celery task for baseline generation using Groq API.
Generates outputs for all samples in an experiment using the baseline model.
"""
from app.tasks.celery_app import celery_app
from app.database import SessionLocal
from app.models import Experiment, ModelVariant, DatasetSample, GeneratedOutput
from groq import Groq
from sqlalchemy.orm import Session
from app.utils.prompt_formatter import PromptFormatter
import time
import os


@celery_app.task(bind=True, name="generate_baseline_outputs")
def generate_baseline_outputs(self, experiment_id: int, variant_id: int):
    """
    Generate baseline outputs using Groq API.
    
    Args:
        experiment_id: ID of the experiment
        variant_id: ID of the baseline model variant
    
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
        
        # Get all samples for this experiment
        samples = db.query(DatasetSample).filter(
            DatasetSample.experiment_id == experiment_id
        ).order_by(DatasetSample.position).all()
        
        if not samples:
            raise ValueError(f"No samples found for experiment {experiment_id}")
        
        # Initialize Groq client with explicit API key
        client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        
        # Get model-specific configuration using PromptFormatter
        system_message = PromptFormatter.get_system_message(variant.model_name)
        stop_sequences_raw = PromptFormatter.get_stop_sequences_for_api(variant.model_name)
        
        # Filter stop sequences - only send thinking tags to API, not format tokens
        stop_sequences = [s for s in stop_sequences_raw if '<' not in s or 'think' in s.lower()] if stop_sequences_raw else None
        
        print(f"\n{'='*60}")
        print(f"🚀 Starting baseline generation")
        print(f"📋 Model: {variant.model_name}")
        if system_message:
            print(f"💬 System message: {system_message[:80]}...")
        if stop_sequences:
            print(f"🛑 Stop sequences: {stop_sequences}")
        print(f"{'='*60}\n")
        
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
                print(f"\n📝 Sample {i+1}/{total_samples} (ID: {sample.id})")
                print(f"📥 Input: {sample.input_text[:100]}...")
                
                # Build messages with optional system message
                # Build task-specific prompt
                from app.utils.task_prompt_builder import TaskPromptBuilder
                
                task_prompt = sample.input_text  # Default to raw input
                
                # Build task-aware prompt if we have task metadata
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
                                print(f"🏷️  Using classification prompt with labels: {labels}")
                        
                        elif experiment.task_type == "rag":
                            task_prompt = TaskPromptBuilder.build(
                                task_type="rag",
                                input_text=sample.input_text,
                                context=sample.context or ""
                            )
                            print(f"🔍 Using RAG prompt with context")
                        
                        elif experiment.task_type == "text_generation":
                            task_prompt = TaskPromptBuilder.build(
                                task_type="text_generation",
                                input_text=sample.input_text
                            )
                    except Exception as e:
                        print(f"⚠️  Task prompt building failed, using raw input: {e}")
                        task_prompt = sample.input_text
                
                # Build messages with optional system message
                messages = []
                if system_message:
                    messages.append({
                        "role": "system",
                        "content": system_message
                    })
                messages.append({
                    "role": "user",
                    "content": task_prompt  # ✅ Use task-aware prompt
                })
                
                # Generate using Groq API
                t0 = time.time()
                completion = client.chat.completions.create(
                    model=variant.model_name,
                    messages=messages,
                    temperature=0.7,
                    max_tokens=256,
                    top_p=0.95,
                    stop=stop_sequences if stop_sequences else None
                )
                t1 = time.time()

                # Extract text safely from completion object
                try:
                    output_text = completion.choices[0].message.content
                except Exception as extract_error:
                    print(f"⚠️  Error extracting content: {extract_error}")
                    try:
                        output_text = completion.choices[0].message["content"]
                    except Exception:
                        output_text = str(completion)
                
                # Clean output: Remove thinking tags if present (for Qwen)
                original_length = len(output_text)
                if '<think>' in output_text:
                    print(f"⚠️  Detected <think> tags in output, cleaning...")
                    parts = output_text.split('</think>')
                    output_text = parts[-1].strip() if len(parts) > 1 else output_text.split('<think>')[0].strip()
                
                # Remove any stop sequences that leaked through
                for stop in (stop_sequences or []):
                    if stop in output_text:
                        output_text = output_text.split(stop)[0].strip()
                
                # Validate output
                if not output_text or len(output_text) < 5:
                    raise ValueError(f"Empty or too short output after cleaning (original: {original_length} chars)")
                
                print(f"📤 Output: {output_text[:150]}...")
                
                # Calculate metrics
                latency_ms = (t1 - t0) * 1000.0
                token_count = len(output_text.split())
                tokens_per_second = token_count / max((t1 - t0), 1e-6)
                
                print(f"⏱️  Latency: {latency_ms:.0f}ms")
                print(f"🚀 Speed: {tokens_per_second:.1f} tok/s")

                # Create GeneratedOutput record
                output = GeneratedOutput(
                    sample_id=sample.id,
                    variant_id=variant.id,
                    output_text=output_text,
                    latency_ms=latency_ms,
                    token_count=token_count,
                    tokens_per_second=tokens_per_second,
                    generation_params_used={
                        "model": variant.model_name,
                        "max_tokens": 256,
                        "temperature": 0.7,
                        "system_message": system_message,
                        "stop_sequences": stop_sequences
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
                
                print(f"✅ Generated output {i+1}/{total_samples} for sample {sample.id}")
                
            except Exception as e:
                print(f"❌ Error generating output for sample {sample.id}: {e}")
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
        
        # Update final status
        if failed_count == 0:
            variant.status = "completed"
            print(f"\n🎉 All {successful_count} samples generated successfully!")
        elif successful_count > 0:
            variant.status = "completed"
            variant.error_message = f"{failed_count} samples failed"
            print(f"\n⚠️  Completed with {failed_count} failures")
        else:
            variant.status = "failed"
            variant.error_message = "All samples failed to generate"
            print(f"\n❌ All samples failed!")
        
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
        
        print(f"\n🎉 Baseline generation complete: {result['message']}\n")
        return result
        
    except Exception as e:
        # Update variant status on failure
        if variant:
            variant.status = "failed"
            variant.error_message = str(e)
            db.commit()
        
        print(f"❌ Fatal error in baseline generation: {e}")
        import traceback
        traceback.print_exc()
        raise
        
    finally:
        db.close()