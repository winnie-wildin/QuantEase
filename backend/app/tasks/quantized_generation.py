"""
Celery task for quantized model generation using GGUF models.
Generates outputs for all samples using locally loaded quantized models.
NOW WITH PARALLEL PROCESSING for faster inference!
"""
from app.tasks.celery_app import celery_app
from app.database import SessionLocal
from app.models import Experiment, ModelVariant, DatasetSample, GeneratedOutput
from app.utils.gguf_loader import GGUFLoader
from app.utils.task_prompt_builder import TaskPromptBuilder
from sqlalchemy.orm import Session
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import cpu_count
import time


def process_single_sample(args):
    """
    Worker function for parallel processing.
    Each worker loads its own model instance and processes one sample.
    
    Args:
        args: Tuple of (sample_dict, variant_id, model_path, experiment_dict)
    
    Returns:
        Dict with generation results
    """
    sample_dict, variant_id, model_path, experiment_dict = args
    
    # Import here to avoid pickling issues
    from app.utils.gguf_loader import GGUFLoader
    from app.utils.task_prompt_builder import TaskPromptBuilder
    import time
    
    # Timing instrumentation
    worker_start = time.time()
    timings = {}
    
    try:
        # Load model for this worker
        load_start = time.time()
        worker_loader = GGUFLoader(
            model_path=model_path,
            n_ctx=2048,
            n_threads=4  # ✅ Increased from 2 to 4 threads
        )
        worker_loader.load()
        timings['model_load_ms'] = (time.time() - load_start) * 1000
        
        # Build task-specific prompt
        prompt_start = time.time()
        task_prompt = sample_dict['input_text']  # Default to raw input
        task_type = experiment_dict.get('task_type')
        
        if task_type:
            try:
                if task_type == "classification":
                    labels = experiment_dict.get('detected_labels', [])
                    if labels:
                        task_prompt = TaskPromptBuilder.build(
                            task_type="classification",
                            input_text=sample_dict['input_text'],
                            labels=labels
                        )
                
                elif task_type == "rag":
                    task_prompt = TaskPromptBuilder.build(
                        task_type="rag",
                        input_text=sample_dict['input_text'],
                        context=sample_dict.get('context', '')
                    )
                
                elif task_type == "text_generation":
                    task_prompt = TaskPromptBuilder.build(
                        task_type="text_generation",
                        input_text=sample_dict['input_text']
                    )
            except Exception as e:
                # Fallback to raw input on error
                task_prompt = sample_dict['input_text']
        
        timings['prompt_build_ms'] = (time.time() - prompt_start) * 1000
        
        # ✅ Task-aware max_tokens
        if task_type == 'classification':
            max_tokens = 10  # Classification needs only a label
        elif task_type == 'rag':
            max_tokens = 256  # RAG needs full answers
        else:
            max_tokens = 256  # Text generation needs full responses
        
        # Generate
        gen_start = time.time()
        result = worker_loader.generate(
            prompt=task_prompt,
            max_tokens=max_tokens,
            temperature=0.7
        )
        timings['generation_ms'] = (time.time() - gen_start) * 1000
        
        # Validate output
        validate_start = time.time()
        output_text = result["output_text"].strip()
        is_successful = True
        error_message = None
        
        # ✅ Task-aware validation
        min_length = 3 if task_type == 'classification' else 5
        
        if not output_text:
            is_successful = False
            error_message = "Empty output after generation"
            output_text = "[No output generated]"
        elif len(output_text) < min_length:
            is_successful = False
            error_message = f"Output too short ({len(output_text)} chars, min: {min_length})"
        elif output_text.startswith("[Generation failed"):
            is_successful = False
            error_message = "Generation failed after retries"
        
        timings['validation_ms'] = (time.time() - validate_start) * 1000
        
        # Unload model
        unload_start = time.time()
        worker_loader.unload()
        timings['model_unload_ms'] = (time.time() - unload_start) * 1000
        timings['total_worker_ms'] = (time.time() - worker_start) * 1000
        
        return {
            'sample_id': sample_dict['id'],
            'success': is_successful,
            'output_text': output_text,
            'latency_ms': result['latency_ms'],
            'token_count': result['token_count'],
            'tokens_per_second': result['tokens_per_second'],
            'error_message': error_message or result.get('error'),
            'model_path': model_path,
            'timings': timings
        }
        
    except Exception as e:
        return {
            'sample_id': sample_dict['id'],
            'success': False,
            'output_text': "[Generation Error]",
            'latency_ms': 0,
            'token_count': 0,
            'tokens_per_second': 0,
            'error_message': f"Exception: {str(e)}",
            'model_path': model_path,
            'timings': {}
        }


@celery_app.task(bind=True, name="generate_quantized_outputs")
def generate_quantized_outputs(self, experiment_id: int, variant_id: int):
    """
    Generate quantized outputs using GGUF model with parallel processing.
    
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
        
        # Calculate optimal parallelization
        total_cpus = cpu_count()
        num_workers = min(4, max(1, total_cpus // 2))
        
        # Start overall timing
        overall_start_time = time.time()
        
        print(f"\n{'='*60}")
        print(f"🚀 PARALLEL QUANTIZED GENERATION")
        print(f"{'='*60}")
        print(f"📊 CPU cores available: {total_cpus}")
        print(f"👷 Workers: {num_workers}")
        print(f"📝 Samples: {len(samples)}")
        print(f"🔧 Model: {variant.model_path}")
        print(f"⚡ Expected speedup: ~{num_workers}x")
        print(f"⏰ Started at: {time.strftime('%H:%M:%S')}")
        print(f"{'='*60}\n")
        
        # Update variant status
        variant.status = "generating"
        variant.progress = 0.0
        db.commit()
        
        total_samples = len(samples)
        successful_count = 0
        failed_count = 0
        
        # Prepare sample data for workers
        sample_dicts = []
        for sample in samples:
            sample_dicts.append({
                'id': sample.id,
                'input_text': sample.input_text,
                'context': sample.context,
                'position': sample.position
            })
        
        # Prepare experiment data
        experiment_dict = {
            'task_type': experiment.task_type,
            'detected_labels': experiment.detected_labels
        }
        
        # Process samples in parallel
        print(f"🔄 Starting parallel processing with {num_workers} workers...\n")
        
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            # Submit all samples
            future_to_sample = {}
            for sample_dict in sample_dicts:
                args = (sample_dict, variant.id, variant.model_path, experiment_dict)
                future = executor.submit(process_single_sample, args)
                future_to_sample[future] = sample_dict
            
            # Process results as they complete
            completed = 0
            batch_start = time.time()
            db_time_total = 0
            
            for future in as_completed(future_to_sample):
                sample_dict = future_to_sample[future]
                completed += 1
                
                try:
                    worker_result = future.result()
                    
                    # Extract timings
                    timings = worker_result.get('timings', {})
                    
                    print(f"📝 Sample {completed}/{total_samples} (ID: {worker_result['sample_id']})")
                    print(f"   {'✅' if worker_result['success'] else '❌'} Output: {worker_result['output_text'][:80]}...")
                    print(f"   📊 {worker_result['token_count']} tokens, {worker_result['latency_ms']:.0f}ms, {worker_result['tokens_per_second']:.2f} tok/s")
                    
                    # Show error details if failed
                    if not worker_result['success']:
                        error_msg = worker_result.get('error_message', 'Unknown error')
                        print(f"   ❌ ERROR DETAILS: {error_msg}")
                        print()
                    
                    # Show timing breakdown
                    if timings:
                        print(f"   ⏱️  Timings:")
                        print(f"      Model load: {timings.get('model_load_ms', 0):.0f}ms")
                        print(f"      Prompt build: {timings.get('prompt_build_ms', 0):.1f}ms")
                        print(f"      Generation: {timings.get('generation_ms', 0):.0f}ms")
                        print(f"      Validation: {timings.get('validation_ms', 0):.1f}ms")
                        print(f"      Model unload: {timings.get('model_unload_ms', 0):.0f}ms")
                        print(f"      Total worker: {timings.get('total_worker_ms', 0):.0f}ms")
                    
                    # Time database operations
                    db_start = time.time()
                    
                    # Create GeneratedOutput record
                    output = GeneratedOutput(
                        sample_id=worker_result['sample_id'],
                        variant_id=variant.id,
                        output_text=worker_result['output_text'],
                        latency_ms=worker_result['latency_ms'],
                        token_count=worker_result['token_count'],
                        tokens_per_second=worker_result['tokens_per_second'],
                        generation_params_used={
                            "model_path": worker_result['model_path'],
                            "max_tokens": 256,
                            "temperature": 0.7,
                            "quantization": variant.quantization_level,
                            "parallel_workers": num_workers,
                            "worker_timings": timings
                        },
                        is_successful=1 if worker_result['success'] else 0,
                        generation_error=worker_result.get('error_message')
                    )
                    db.add(output)
                    
                    if worker_result['success']:
                        successful_count += 1
                    else:
                        failed_count += 1
                    
                    # Update progress
                    variant.progress = completed / total_samples
                    db.commit()
                    
                    db_time = (time.time() - db_start) * 1000
                    db_time_total += db_time
                    print(f"   💾 DB commit: {db_time:.1f}ms")
                    
                    # Calculate throughput
                    elapsed = time.time() - batch_start
                    samples_per_sec = completed / elapsed if elapsed > 0 else 0
                    samples_per_min = samples_per_sec * 60
                    eta_seconds = (total_samples - completed) / samples_per_sec if samples_per_sec > 0 else 0
                    
                    print(f"   📈 Throughput: {samples_per_min:.1f} samples/min | ETA: {eta_seconds/60:.1f} min")
                    print()
                    
                    # Update task state
                    self.update_state(
                        state='PROGRESS',
                        meta={
                            'current': completed,
                            'total': total_samples,
                            'successful': successful_count,
                            'failed': failed_count,
                            'status': f'Generated {completed}/{total_samples} outputs (parallel)'
                        }
                    )
                    
                except Exception as e:
                    print(f"❌ Error processing sample {sample_dict['id']}: {e}")
                    import traceback
                    traceback.print_exc()
                    
                    failed_count += 1
                    
                    # Store failed output
                    output = GeneratedOutput(
                        sample_id=sample_dict['id'],
                        variant_id=variant.id,
                        output_text="[Generation Error]",
                        latency_ms=0,
                        token_count=0,
                        tokens_per_second=0,
                        is_successful=0,
                        generation_error=f"Worker exception: {str(e)}"
                    )
                    db.add(output)
                    db.commit()
        
        # Calculate final timing
        overall_elapsed = time.time() - overall_start_time
        
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
        
        # Timing summary
        print(f"\n{'='*60}")
        print(f"⏱️  TIMING SUMMARY")
        print(f"{'='*60}")
        print(f"Total time: {overall_elapsed/60:.2f} minutes ({overall_elapsed:.1f}s)")
        print(f"Samples processed: {successful_count}/{total_samples}")
        print(f"Average per sample: {overall_elapsed/total_samples:.1f}s")
        print(f"Throughput: {total_samples/(overall_elapsed/60):.1f} samples/min")
        print(f"Database time total: {db_time_total/1000:.1f}s ({db_time_total/overall_elapsed/10:.1f}%)")
        print(f"Workers used: {num_workers}")
        print(f"Speedup vs sequential: ~{num_workers:.1f}x")
        print(f"{'='*60}\n")
        
        result = {
            "status": "completed" if failed_count == 0 else "completed_with_errors",
            "experiment_id": experiment_id,
            "variant_id": variant_id,
            "total_samples": total_samples,
            "successful": successful_count,
            "failed": failed_count,
            "workers_used": num_workers,
            "total_time_seconds": overall_elapsed,
            "samples_per_minute": total_samples/(overall_elapsed/60),
            "message": f"Generated {successful_count}/{total_samples} outputs successfully (parallel)"
        }
        
        print(f"🎉 Parallel quantized generation complete: {result['message']}")
        print(f"⚡ Processed {result['samples_per_minute']:.1f} samples/min with {num_workers} workers\n")
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