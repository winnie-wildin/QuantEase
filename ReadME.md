#  QuantEase - LLM Quantization Evaluation Platform

**Compare baseline and quantized LLM models side-by-side to find the perfect speed-quality-size trade-off.**

QuantEase helps you evaluate quantized models against baseline models with comprehensive metrics including performance benchmarks, quality scores (BERTScore), and optional LLM-as-judge evaluation.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

##  Table of Contents

- [Why QuantEase?](#-why-quantease)
- [Key Features](#-key-features)
- [Quick Start](#-quick-start)
- [Architecture](#-architecture)
- [Task Types](#-task-types)
- [Installation](#-installation)
- [Usage Guide](#-usage-guide)
- [LLM Judge](#-llm-judge)
- [Customization](#-customization)
- [Troubleshooting](#-troubleshooting)
- [API Reference](#-api-reference)

---

##  Why QuantEase?

**The Problem:** You need to compare quantized models against baseline models to find the perfect speed-quality-size trade-off, but current tools compare cloud vs local models which isn't useful for speed comparisons.

**The Solution:** QuantEase automates the evaluation process:
-  Compare baseline models vs quantized models side-by-side
-  Measure quality degradation at different quantization levels
-  Benchmark speed, latency, and model size with fair comparisons
-  Get actionable insights with visual trade-off analysis

**Use Cases:**
-  Enterprise: Find optimal quantization level for cost/quality balance
-  Research: Evaluate quantization techniques systematically
-  Edge Deployment: Identify models that meet quality requirements
-  Education: Understand quantization trade-offs hands-on

---

##  Key Features

###  Task-Aware Evaluation
- **Text Generation:** BERTScore, length ratio, divergence
- **Classification:** Accuracy, F1, per-class metrics, confusion matrix
- **RAG:** Answer relevance, hallucination detection, factual correctness

###  Comprehensive Metrics
- **Performance:** Speed (tok/s), latency, model size
- **Quality:** BERTScore, cosine similarity, task-specific metrics
- **Optional LLM Judge:** Qualitative assessment by Llama 3.3 70B

###  Visual Dashboard
- Side-by-side output comparison
- Interactive trade-off charts (size vs quality, speed comparison)
- Per-sample divergence heatmaps
- Actionable insights and recommendations

###  Flexible Architecture
- Multiple quantized model formats (GGUF and more coming soon)
- Multiple quantization levels (INT4, INT8, FP16, etc.)
- Cloud-based inference for fair speed comparisons

---

##  Quick Start

### Prerequisites
- **Docker & Docker Compose** (recommended)
- **Python 3.10+** and **Node.js 18+** (if running without Docker)
- **Groq API Key** (free tier available: https://console.groq.com)

### 5-Minute Setup

```bash
# 1. Clone the repository
git clone https://github.com/yourusername/quantease.git
cd quantease

# 2. Set environment variables
cp .env.example .env
# Edit .env and add your GROQ_API_KEY

# 3. Start the stack
docker-compose up -d

# 4. Wait for services to be ready (~30 seconds)
docker-compose logs -f backend | grep "Application startup complete"

# 5. Open the dashboard
open http://localhost:3000
```

### Your First Experiment

1. **Create Experiment:** Click "New Experiment" → Select task type
2. **Upload Dataset:** Drag-drop JSON/CSV with prompts, or use preloaded datasets (coming soon)
3. **Add Models:** Select baseline + quantized variants
4. **Generate:** Click "Generate Outputs" (2-3 min)
5. **Evaluate:** Click "Run Evaluation" (1-2 min)
6. **Analyze:** View metrics, charts, and trade-offs!

---

##  Architecture

**Frontend:** React + TypeScript + TailwindCSS  
**Backend:** FastAPI + SQLAlchemy + Celery  
**Database:** PostgreSQL  
**Task Queue:** Redis + Celery  
**Infrastructure:** Docker Compose

The platform uses a microservices architecture with async task processing for model inference and evaluation.

---

##  Task Types

QuantEase supports multiple task types, each with specialized metrics and evaluation criteria. More granular subcategories coming soon.

###  Text Generation
**Use case:** Open-ended generation (summaries, rewrites, creative writing)

**Coming soon:** More specific subcategories like summarization, translation, creative writing, code generation, etc.

**Dataset format:**
```json
[
  {
    "input_text": "Summarize this article: ...",
    "ground_truth_output": "The article discusses..." // Optional
  }
]
```

**Metrics:**
- BERTScore F1 (semantic similarity)
- Length ratio (verbosity check)
- Divergence from baseline (style consistency)
- LLM Judge: Accuracy, Fluency, Coherence (1-5 scale)

---

###  Classification
**Use case:** Categorization tasks (sentiment, topic, intent)

**Coming soon:** More specific subcategories like sentiment analysis, topic classification, intent detection, entity recognition, etc.

**Dataset format:**
```json
[
  {
    "input_text": "This movie was amazing!",
    "ground_truth_output": "positive"  // Required
  }
]
```

**Metrics:**
- Accuracy
- Macro/Weighted F1
- Per-class F1 scores
- Confusion matrix
- Class imbalance detection

**Special features:**
- Auto-detects label set from dataset
- Task-specific prompt: "Output ONLY the label"
- Single-token output enforced

---

###  RAG (Retrieval-Augmented Generation)
**Use case:** Question answering with context

**Coming soon:** More specific subcategories like document Q&A, knowledge base querying, conversational AI, etc.

**Dataset format:**
```json
[
  {
    "input_text": "What is the return policy?",
    "context": "Company return policy: All items can be returned within 30 days...",
    "ground_truth_output": "Items can be returned within 30 days."  // Optional
  }
]
```

**Metrics:**
- Answer relevance (context utilization)
- BERTScore (answer quality)
- LLM Judge: Hallucination detection, Factual correctness, Completeness

**Special features:**
- Hallucination rate (% of samples with facts not in context)
- Context precision/recall
- Visual hallucination bar

---

##  Installation

### Option 1: Docker (Recommended)

```bash
# 1. Clone and setup
git clone https://github.com/yourusername/quantease.git
cd quantease
cp .env.example .env

# 2. Configure environment
nano .env  # Add GROQ_API_KEY and other settings

# 3. Start services
docker-compose up -d

# 4. Run database migrations
docker-compose exec backend alembic upgrade head

# 5. Verify
curl http://localhost:8000/health
curl http://localhost:3000
```

### Option 2: Local Development

**Backend:**
```bash
cd backend

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Setup database
export DATABASE_URL="postgresql://user:pass@localhost:5432/quantease"
alembic upgrade head

# Start backend
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

# Start Celery worker (separate terminal)
celery -A app.tasks.celery_app worker --loglevel=info
```

**Frontend:**
```bash
cd frontend

# Install dependencies
npm install

# Start dev server
npm run dev
```

**Required Services:**
- PostgreSQL 14+ running on port 5432
- Redis 7+ running on port 6379

---

##  Usage Guide

### Creating an Experiment

**Step 1: Upload Dataset**
```bash
# Example dataset format (JSON)
[
  {
    "input_text": "Translate to French: Hello world",
    "ground_truth_output": "Bonjour le monde"
  }
]

# Or CSV format
input_text,ground_truth_output
"Translate to French: Hello world","Bonjour le monde"
```

**Step 2: Select Task Type**
- **Text Generation:** For open-ended outputs
- **Classification:** For label prediction
- **RAG:** For QA with context (requires `context` field)

**Step 3: Add Models**

**Baseline (Cloud):**
- Recommended: `llama-3.3-70b-versatile` (best quality)
- Fast option: `llama-3.1-8b-instant` (lower cost)

**Quantized:**
- Currently supports GGUF models
- More quantized model formats coming soon (TensorRT, ONNX, etc.)
- Multiple quantization levels (INT4, INT8, FP16, etc.)

**Step 4: Generate & Evaluate**
- Generation: 1-3 minutes depending on dataset size
- Evaluation: 1-2 minutes (add 30s if LLM judge enabled)

---

##  LLM Judge

### What is LLM Judge?

An optional qualitative evaluation using Llama 3.3 70B to assess outputs on criteria beyond simple metrics.

### When to Use

 **Use when:**
- Quality metrics alone don't capture nuances
- You need human-like assessment at scale
- Evaluating creative or subjective tasks
- Budget allows (~$0.001 per sample)

 **Skip when:**
- Classification (accuracy is sufficient)
- Simple keyword matching tasks
- Large datasets (costs add up)
- No GROQ_API_KEY available

### Configuration

**Backend (`.env`):**
```bash
GROQ_API_KEY=gsk_your_key_here
```

**Frontend:**
```typescript
// When running evaluation
 Enable LLM Judge (slower, requires GROQ_API_KEY)
```

**Sampling:**
Default is 10% of dataset (configurable in backend):
```python
# backend/app/models.py
experiment.judge_sample_percentage = 20.0  # Sample 20%
```

### Metrics Provided

**Text Generation:**
- Accuracy (1-5): Factual correctness
- Fluency (1-5): Grammar and naturalness
- Coherence (1-5): Logical structure

**RAG:**
- Hallucination Rate (%): Made-up facts not in context
- Factual Correctness (1-5): Accuracy given context
- Completeness (1-5): Fully answers question

### Cost Estimate

Using Groq (as of Dec 2024):
- ~$0.001 per sample judged
- 100 samples @ 10% = $0.10
- 1000 samples @ 10% = $1.00

---

##  Customization

You can customize task-specific prompts, model prompt formatting, and evaluation metrics by editing the corresponding utility files in the backend:

- `backend/app/utils/task_prompt_builder.py` - Task-specific prompts
- `backend/app/utils/prompt_formatter.py` - Model prompt formatting
- `backend/app/utils/task_evaluators.py` - Custom evaluation metrics

---

##  Troubleshooting

### Common Issues

**Generation stuck or taking too long:**
- Check Celery worker logs: `docker-compose logs celery-worker`
- Verify workers are running: `docker-compose ps`

**LLM Judge not showing:**
- Ensure checkbox is enabled before running evaluation
- Verify GROQ_API_KEY is set in `.env`

**Database connection errors:**
- Check Postgres is running: `docker-compose ps postgres`
- Restart services: `docker-compose restart`

**Frontend can't connect:**
- Verify backend is running: `curl http://localhost:8000/health`
- Check API_BASE_URL in frontend configuration

---

##  API Reference

The platform exposes a REST API for experiment management, model configuration, and evaluation. Key endpoints:

- `POST /experiments/` - Create new experiment
- `POST /experiments/{id}/samples` - Upload dataset
- `POST /experiments/{id}/variants` - Add model variants
- `POST /experiments/{id}/generate` - Start generation
- `POST /experiments/{id}/evaluate` - Run evaluation
- `GET /experiments/{id}/metrics` - Get evaluation results
- `GET /experiments/models/quantized` - List available quantized models

---

##  Roadmap

### Short-term Goals

**Fair Speed Comparisons**
- [ ] Move quantized models to cloud inference for fair speed comparisons
- [ ] Enable true baseline vs quantized speed benchmarking (cloud vs cloud)

**Expanded Model Support**
- [ ] Add support for other quantized model formats beyond GGUF (TensorRT, ONNX, etc.)
- [ ] Integration with multiple quantization frameworks

**Enhanced Task Categorization**
- [ ] Add granular subcategories within existing task types
- [ ] Specialized metrics and evaluation criteria for each subcategory
  - Text Generation: Summarization, Translation, Creative Writing, Code Generation
  - Classification: Sentiment Analysis, Topic Classification, Intent Detection, Entity Recognition
  - RAG: Document Q&A, Knowledge Base Querying, Conversational AI

**Dataset Features**
- [ ] Preloaded datasets for common use cases (no need to bring your own)
- [ ] Dataset generator: Create datasets using simple prompts
- [ ] Dataset templates and examples for each task type

### Future Considerations
- [ ] Export evaluation reports (PDF/CSV)
- [ ] Cost calculator (baseline vs quantized)
- [ ] More advanced metrics and evaluation methods

---

##  License

MIT License - see [LICENSE](LICENSE) file for details.

---

##  Acknowledgments

- [BERTScore](https://github.com/Tiiiger/bert_score) - Semantic similarity metric
- [FastAPI](https://fastapi.tiangolo.com/) - Modern Python web framework

