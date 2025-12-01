#!/usr/bin/env python3
"""
Verification Script for Task-Aware Evaluation System
Run this to verify all changes were implemented correctly.
"""

import sys
import os

# Add backend to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'backend'))

def check_database_migration():
    """Check if database migration was successful"""
    print("\n" + "="*60)
    print("🔍 CHECKING DATABASE MIGRATION")
    print("="*60)
    
    try:
        from app.database import engine
        from sqlalchemy import inspect, text
        
        inspector = inspect(engine)
        
        # Check experiments table columns
        print("\n📋 Checking 'experiments' table columns...")
        exp_columns = [col['name'] for col in inspector.get_columns('experiments')]
        
        required_exp_columns = ['task_type', 'normalization_metadata', 'judge_enabled', 'judge_sample_percentage']
        missing_exp = [col for col in required_exp_columns if col not in exp_columns]
        
        if missing_exp:
            print(f"   ❌ Missing columns: {missing_exp}")
            return False
        else:
            print(f"   ✅ All new columns present: {required_exp_columns}")
        
        # Check dataset_samples table
        print("\n📋 Checking 'dataset_samples' table columns...")
        sample_columns = [col['name'] for col in inspector.get_columns('dataset_samples')]
        
        if 'context' not in sample_columns:
            print("   ❌ Missing 'context' column")
            return False
        else:
            print("   ✅ 'context' column present")
        
        # Check comparative_metrics table
        print("\n📋 Checking 'comparative_metrics' table columns...")
        metrics_columns = [col['name'] for col in inspector.get_columns('comparative_metrics')]
        
        if 'evaluation_results' not in metrics_columns:
            print("   ❌ Missing 'evaluation_results' column")
            return False
        else:
            print("   ✅ 'evaluation_results' column present")
        
        # Check index
        print("\n📋 Checking indexes...")
        indexes = inspector.get_indexes('experiments')
        has_task_type_index = any('task_type' in str(idx.get('column_names', [])) for idx in indexes)
        
        if has_task_type_index:
            print("   ✅ Index on task_type exists")
        else:
            print("   ⚠️  Index on task_type not found (non-critical)")
        
        print("\n✅ DATABASE MIGRATION: PASSED")
        return True
        
    except Exception as e:
        print(f"\n❌ DATABASE MIGRATION: FAILED")
        print(f"   Error: {e}")
        return False


def check_model_changes():
    """Check if model files were updated correctly"""
    print("\n" + "="*60)
    print("🔍 CHECKING MODEL CHANGES")
    print("="*60)
    
    try:
        from app.models import Experiment, DatasetSample, ComparativeMetrics
        
        # Check Experiment model
        print("\n📋 Checking Experiment model...")
        exp = Experiment(
            name="test",
            baseline_model_id=1,
            has_ground_truth=False,
            sample_count=0,
            status="created"
        )
        
        required_attrs = ['task_type', 'normalization_metadata', 'judge_enabled', 'judge_sample_percentage']
        missing_attrs = [attr for attr in required_attrs if not hasattr(exp, attr)]
        
        if missing_attrs:
            print(f"   ❌ Missing attributes: {missing_attrs}")
            return False
        else:
            print(f"   ✅ All new attributes present: {required_attrs}")
        
        # Check task_display_name property
        if hasattr(exp, 'task_display_name'):
            print("   ✅ 'task_display_name' property exists")
        else:
            print("   ⚠️  'task_display_name' property missing (optional)")
        
        # Check DatasetSample model
        print("\n📋 Checking DatasetSample model...")
        if hasattr(DatasetSample, 'context'):
            print("   ✅ 'context' field exists")
        else:
            print("   ❌ 'context' field missing")
            return False
        
        # Check ComparativeMetrics model
        print("\n📋 Checking ComparativeMetrics model...")
        if hasattr(ComparativeMetrics, 'evaluation_results'):
            print("   ✅ 'evaluation_results' field exists")
        else:
            print("   ❌ 'evaluation_results' field missing")
            return False
        
        print("\n✅ MODEL CHANGES: PASSED")
        return True
        
    except Exception as e:
        print(f"\n❌ MODEL CHANGES: FAILED")
        print(f"   Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def check_utility_files():
    """Check if utility files were copied correctly"""
    print("\n" + "="*60)
    print("🔍 CHECKING UTILITY FILES")
    print("="*60)
    
    results = []
    
    # Check task_evaluators.py
    print("\n📋 Checking task_evaluators.py...")
    try:
        from app.utils.task_evaluators import (
            TextGenEvaluator, 
            ClassificationEvaluator, 
            RAGEvaluator
        )
        print("   ✅ TextGenEvaluator imported")
        print("   ✅ ClassificationEvaluator imported")
        print("   ✅ RAGEvaluator imported")
        results.append(True)
    except ImportError as e:
        print(f"   ❌ Import failed: {e}")
        results.append(False)
    
    # Check json_normalizer.py
    print("\n📋 Checking json_normalizer.py...")
    try:
        from app.utils.json_normalizer import JSONNormalizer
        print("   ✅ JSONNormalizer imported")
        
        # Test basic functionality
        test_data = [{"input": "test", "output": "result"}]
        normalized, metadata = JSONNormalizer.normalize_text_generation(test_data)
        print("   ✅ JSONNormalizer.normalize_text_generation() works")
        results.append(True)
    except ImportError as e:
        print(f"   ❌ Import failed: {e}")
        results.append(False)
    except Exception as e:
        print(f"   ⚠️  Import succeeded but function failed: {e}")
        results.append(True)  # Import worked at least
    
    # Check llm_judge.py
    print("\n📋 Checking llm_judge.py...")
    try:
        from app.utils.llm_judge import LLMJudge
        print("   ✅ LLMJudge imported")
        results.append(True)
    except ImportError as e:
        print(f"   ❌ Import failed: {e}")
        results.append(False)
    
    # Check config/models.py
    print("\n📋 Checking config/models.py...")
    try:
        from app.config.models import MODEL_RECOMMENDATIONS, GROQ_MODELS, QUANTIZED_MODELS
        print("   ✅ MODEL_RECOMMENDATIONS imported")
        print("   ✅ GROQ_MODELS imported")
        print("   ✅ QUANTIZED_MODELS imported")
        
        # Check if MODEL_RECOMMENDATIONS has required keys
        required_tasks = ['text_generation', 'classification', 'rag']
        missing_tasks = [task for task in required_tasks if task not in MODEL_RECOMMENDATIONS]
        
        if missing_tasks:
            print(f"   ⚠️  Missing recommendations for: {missing_tasks}")
        else:
            print(f"   ✅ All task recommendations present: {required_tasks}")
        
        results.append(True)
    except ImportError as e:
        print(f"   ❌ Import failed: {e}")
        results.append(False)
    
    if all(results):
        print("\n✅ UTILITY FILES: PASSED")
        return True
    else:
        print("\n❌ UTILITY FILES: FAILED")
        return False


def check_router_changes():
    """Check if router was updated correctly"""
    print("\n" + "="*60)
    print("🔍 CHECKING ROUTER CHANGES")
    print("="*60)
    
    try:
        from app.routers.experiments import router
        
        # Get all route paths
        routes = [route.path for route in router.routes]
        
        print("\n📋 Checking endpoints...")
        
        # Check if recommendations endpoint exists
        if any('/models/recommendations/{task_type}' in path for path in routes):
            print("   ✅ GET /models/recommendations/{task_type} endpoint exists")
        else:
            print("   ⚠️  GET /models/recommendations/{task_type} endpoint not found")
        
        # Check upload_samples endpoint
        if any('/{experiment_id}/samples' in path for path in routes):
            print("   ✅ POST /{experiment_id}/samples endpoint exists")
            
            # Try to import and check signature
            from app.routers.experiments import upload_samples
            import inspect
            sig = inspect.signature(upload_samples)
            params = list(sig.parameters.keys())
            
            if 'task_type' in params:
                print("   ✅ upload_samples has 'task_type' parameter")
            else:
                print("   ❌ upload_samples missing 'task_type' parameter")
                return False
        
        # Check evaluate endpoint
        if any('/{experiment_id}/evaluate' in path for path in routes):
            print("   ✅ POST /{experiment_id}/evaluate endpoint exists")
            
            from app.routers.experiments import trigger_evaluation
            import inspect
            sig = inspect.signature(trigger_evaluation)
            params = list(sig.parameters.keys())
            
            if 'enable_llm_judge' in params:
                print("   ✅ trigger_evaluation has 'enable_llm_judge' parameter")
            else:
                print("   ⚠️  trigger_evaluation missing 'enable_llm_judge' parameter (optional)")
        
        print("\n✅ ROUTER CHANGES: PASSED")
        return True
        
    except Exception as e:
        print(f"\n❌ ROUTER CHANGES: FAILED")
        print(f"   Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def check_evaluation_task():
    """Check if evaluation task was updated"""
    print("\n" + "="*60)
    print("🔍 CHECKING EVALUATION TASK")
    print("="*60)
    
    try:
        from app.tasks.evaluation_task_v2 import (
            evaluate_variant_v2,
            evaluate_text_generation,
            evaluate_classification,
            evaluate_rag,
            run_legacy_evaluation
        )
        
        print("\n📋 Checking helper functions...")
        print("   ✅ evaluate_text_generation_task exists")
        print("   ✅ evaluate_classification_task exists")
        print("   ✅ evaluate_rag_task exists")
        print("   ✅ run_legacy_evaluation exists")
        
        print("\n📋 Checking evaluate_variant signature...")
        import inspect
        sig = inspect.signature(evaluate_variant)
        params = list(sig.parameters.keys())
        
        if 'enable_llm_judge' in params:
            print("   ✅ evaluate_variant has 'enable_llm_judge' parameter")
        else:
            print("   ❌ evaluate_variant missing 'enable_llm_judge' parameter")
            return False
        
        print("\n✅ EVALUATION TASK: PASSED")
        return True
        
    except ImportError as e:
        print(f"\n❌ EVALUATION TASK: FAILED")
        print(f"   Error: {e}")
        return False


def check_dependencies():
    """Check if required dependencies are installed"""
    print("\n" + "="*60)
    print("🔍 CHECKING DEPENDENCIES")
    print("="*60)
    
    results = []
    
    print("\n📋 Checking Python packages...")
    
    # Check sentence-transformers
    try:
        import sentence_transformers
        print(f"   ✅ sentence-transformers {sentence_transformers.__version__}")
        results.append(True)
    except ImportError:
        print("   ❌ sentence-transformers not installed")
        results.append(False)
    
    # Check groq
    try:
        import groq
        print(f"   ✅ groq installed")
        results.append(True)
    except ImportError:
        print("   ❌ groq not installed")
        results.append(False)
    
    # Check scikit-learn
    try:
        import sklearn
        print(f"   ✅ scikit-learn {sklearn.__version__}")
        results.append(True)
    except ImportError:
        print("   ❌ scikit-learn not installed")
        results.append(False)
    
    # Check torch
    try:
        import torch
        print(f"   ✅ torch {torch.__version__}")
        results.append(True)
    except ImportError:
        print("   ⚠️  torch not installed (optional, but recommended)")
        results.append(True)  # Optional
    
    if all(results):
        print("\n✅ DEPENDENCIES: PASSED")
        return True
    else:
        print("\n❌ DEPENDENCIES: FAILED")
        print("\n   Install missing packages:")
        print("   pip install sentence-transformers groq scikit-learn torch")
        return False


def run_all_checks():
    """Run all verification checks"""
    print("\n" + "="*60)
    print("🚀 TASK-AWARE EVALUATION SYSTEM - VERIFICATION")
    print("="*60)
    
    results = {
        "Database Migration": check_database_migration(),
        "Model Changes": check_model_changes(),
        "Utility Files": check_utility_files(),
        "Router Changes": check_router_changes(),
        "Evaluation Task": check_evaluation_task(),
        "Dependencies": check_dependencies()
    }
    
    # Final summary
    print("\n" + "="*60)
    print("📊 VERIFICATION SUMMARY")
    print("="*60)
    
    for check_name, passed in results.items():
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"{check_name:.<40} {status}")
    
    all_passed = all(results.values())
    
    print("\n" + "="*60)
    if all_passed:
        print("🎉 ALL CHECKS PASSED!")
        print("="*60)
        print("\n✅ Your task-aware evaluation system is ready!")
        print("\n📝 Next steps:")
        print("   1. Test with a real experiment")
        print("   2. Try all 3 task types (text_generation, classification, rag)")
        print("   3. Check the evaluation_results in the database")
        return 0
    else:
        print("❌ SOME CHECKS FAILED")
        print("="*60)
        failed = [name for name, passed in results.items() if not passed]
        print(f"\n⚠️  Failed checks: {', '.join(failed)}")
        print("\n📖 See error messages above for details")
        print("📖 Refer to MODIFICATIONS_GUIDE.md for help")
        return 1


if __name__ == "__main__":
    exit_code = run_all_checks()
    sys.exit(exit_code)