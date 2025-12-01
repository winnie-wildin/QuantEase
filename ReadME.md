#  QuantEase - LLM Quantization Evaluation Platform

**Compare cloud LLMs with locally-quantized models to find the perfect speed-quality-size trade-off.**

Quantease helps you evaluate quantized GGUF models against baseline cloud models with comprehensive metrics including performance benchmarks, quality scores (BERTScore), and optional LLM-as-judge evaluation.

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
- [Contributing](#-contributing)

---

##  Why QuantEase?

**The Problem:** Cloud LLMs are powerful but expensive and slow. Quantized models run locally but you need to know which quantization level maintains acceptable quality.

**The Solution:** QuantEase automates the evaluation process:
-  Compare cloud baseline (Groq) vs local quantized models (GGUF)
-  Measure quality degradation at different quantization levels
-  Benchmark speed, latency, and model size
-  Get actionable insights with visual trade-off analysis

**Use Cases:**
-  Enterprise: Reduce inference costs by 90% while maintaining quality
-  Research: Evaluate quantization techniques systematically
-  Edge Deployment: Find the smallest model that meets your quality bar
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
- Any GGUF model from Hugging Face
- Multiple quantization levels (INT4, INT8, FP16, etc.)
- Groq API for baseline (blazing fast)
- Local llama.cpp for quantized inference

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
2. **Upload Dataset:** Drag-drop JSON/CSV with prompts
3. **Add Models:** Select baseline + quantized variants
4. **Generate:** Click "Generate Outputs" (2-3 min)
5. **Evaluate:** Click "Run Evaluation" (1-2 min)
6. **Analyze:** View metrics, charts, and trade-offs!

---

##  Architecture

```
┌─────────────────┐
│   Frontend      │  React + TypeScript + TailwindCSS
│   (Port 3000)   │  
└────────┬────────┘
         │ REST API
┌────────▼────────┐
│   Backend       │  FastAPI + SQLAlchemy
│   (Port 8000)   │  
└────────┬────────┘
         │
    ┌────┴─────┬──────────┬──────────┐
    │          │          │          │
┌───▼───┐  ┌──▼──┐  ┌────▼────┐  ┌─▼──┐
│ Celery│  │Redis│  │Postgres │  │Groq│
│Worker │  │     │  │         │  │API │
└───┬───┘  └─────┘  └─────────┘  └────┘
    │
┌───▼───────────┐
│ llama.cpp     │  Local GGUF inference
│ (llama-cpp-   │
│  python)      │
└───────────────┘
```

### Technology Stack

**Backend:**
- FastAPI - Modern async web framework
- Celery - Distributed task queue
- SQLAlchemy - Database ORM
- llama-cpp-python - GGUF model inference
- sentence-transformers - BERTScore calculation

**Frontend:**
- React 18 - UI framework
- TypeScript - Type safety
- TailwindCSS - Styling
- Recharts - Data visualization
- React Query - Server state management

**Infrastructure:**
- PostgreSQL - Relational database
- Redis - Task broker
- Docker - Containerization

---

##  Task Types

###  Text Generation
**Use case:** Open-ended generation (summaries, rewrites, creative writing)

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

**Quantized (Local):**
```python
# Popular models on Hugging Face
"bartowski/Qwen2.5-3B-Instruct-GGUF"  # 3B - fast, efficient
"bartowski/Phi-3-mini-128k-instruct-GGUF"  # 3B - long context
"TheBloke/Llama-2-7B-Chat-GGUF"  # 7B - classic

# Quantization levels (quality ↓, speed ↑)
Q8_0   # 8-bit (best quality, largest)
Q6_K   # 6-bit (great balance)
Q5_K_M # 5-bit medium (recommended)
Q4_K_M # 4-bit medium (good speed)
Q3_K_M # 3-bit medium (smallest)
```

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

### Task-Specific Prompts

Edit `backend/app/utils/task_prompt_builder.py`:

```python
# Example: Custom classification prompt
CLASSIFICATION_PROMPT = """
You are a sentiment classifier.

Text: {input_text}

Classify into ONE category: {labels}

RULES:
1. Output ONLY the category name
2. No explanation
3. No punctuation

Category:
"""
```

### Model Prompt Formatting

Edit `backend/app/utils/prompt_formatter.py`:

```python
# Add new model family
"custom-model": """<|system|>
You are a helpful assistant.
<|user|>
{prompt}
<|assistant|>"""
```

### Custom Metrics

Add to `backend/app/utils/task_evaluators.py`:

```python
class CustomEvaluator:
    def evaluate(self, candidates, references):
        # Your custom metric
        scores = [self.custom_metric(c, r) 
                  for c, r in zip(candidates, references)]
        return {"custom_score": np.mean(scores)}
```

---

##  Troubleshooting

### LLM Judge Not Showing

**Symptoms:**
```
 LLM Judge: DISABLED 
```

**Solution:**
1. Check checkbox is enabled in UI before clicking "Run Evaluation"
2. Verify API key: `docker-compose exec backend env | grep GROQ_API_KEY`
3. Check backend logs: `docker-compose logs backend | grep Judge`
4. Ensure model name is correct (not `gpt-oss-120b`)

---

### Model Not Found Error

**Symptoms:**
```
 Model file not found: /models/qwen2.5-3b-INT4.gguf
```

**Solution:**
1. Download GGUF model from Hugging Face
2. Place in `backend/models/` directory
3. Update `model_path` when creating variant
4. Restart backend: `docker-compose restart backend`

---

### Generation Stuck

**Symptoms:**
- Status shows "generating" for >5 minutes
- No progress in logs

**Solution:**
```bash
# Check Celery worker is running
docker-compose ps

# View worker logs
docker-compose logs celery-worker

# Restart worker
docker-compose restart celery-worker

# Check task status
docker-compose exec backend python
>>> from app.tasks.celery_app import celery_app
>>> celery_app.control.inspect().active()
```

---

### High Memory Usage

**Symptoms:**
- System becomes slow during generation
- Docker crashes with OOM

**Solution:**
```yaml
# docker-compose.yml
services:
  celery-worker:
    deploy:
      resources:
        limits:
          memory: 8G  # Adjust based on model size
```

**Model size requirements:**
- 3B INT4: 2-3 GB RAM
- 7B INT4: 4-5 GB RAM
- 13B INT4: 8-10 GB RAM

---

### Evaluation Takes Too Long

**Symptoms:**
- Evaluation runs >10 minutes
- LLM judge causes timeouts

**Solution:**
1. **Disable LLM judge** if not needed
2. **Reduce sample percentage:**
   ```python
   # Evaluate only 5% with judge
   experiment.judge_sample_percentage = 5.0
   ```
3. **Use smaller baseline model:**
   ```python
   "llama-3.1-8b-instant"  # Instead of 70B
   ```

---

### Database Connection Errors

**Symptoms:**
```
sqlalchemy.exc.OperationalError: connection refused
```

**Solution:**
```bash
# Check Postgres is running
docker-compose ps postgres

# View Postgres logs
docker-compose logs postgres

# Recreate database
docker-compose down -v
docker-compose up -d postgres
docker-compose exec backend alembic upgrade head
```

---

### Frontend Can't Connect to Backend

**Symptoms:**
```
Failed to fetch experiments
```

**Solution:**
1. Check backend is running: `curl http://localhost:8000/health`
2. Verify CORS settings in `backend/app/main.py`
3. Check network in Docker: `docker-compose exec frontend ping backend`
4. Ensure API_BASE_URL is correct in `frontend/src/services/api.ts`

---

##  API Reference

### Core Endpoints

#### Create Experiment
```http
POST /experiments/
Content-Type: application/json

{
  "name": "Sentiment Analysis Test",
  "baseline_model_id": 1,
  "has_ground_truth": true
}
```

#### Upload Dataset
```http
POST /experiments/{id}/samples
Content-Type: application/json

{
  "samples": [...],
  "task_type": "classification"
}
```

#### Add Model Variant
```http
POST /experiments/{id}/variants
Content-Type: application/json

{
  "variant_type": "quantized",
  "model_name": "qwen2.5-3b",
  "quantization_level": "INT4",
  "model_path": "/models/qwen2.5-3b-INT4.gguf",
  "inference_provider": "llama.cpp"
}
```

#### Generate Outputs
```http
POST /experiments/{id}/generate
```

#### Run Evaluation
```http
POST /experiments/{id}/evaluate?enable_llm_judge=true
```

#### Get Results
```http
GET /experiments/{id}/metrics
GET /experiments/{id}/samples/comparison?page=1&page_size=20
```

### Model Recommendations
```http
GET /experiments/models/recommendations/{task_type}

Response:
{
  "model_name": "llama-3.3-70b",
  "display_name": "Llama 3.3 70B",
  "reason": "Best for reasoning tasks",
  "provider": "groq"
}
```

---

##  Contributing

### Development Setup

```bash
# 1. Fork and clone
git clone https://github.com/yourusername/quantease.git
cd quantease

# 2. Create feature branch
git checkout -b feature/amazing-feature

# 3. Install pre-commit hooks
pip install pre-commit
pre-commit install

# 4. Make changes and test
pytest backend/tests
npm test --prefix frontend

# 5. Submit PR
git push origin feature/amazing-feature
```

### Code Style

**Backend:**
- Follow PEP 8
- Type hints required
- Docstrings for public functions

**Frontend:**
- ESLint + Prettier
- TypeScript strict mode
- Functional components with hooks

### Testing

**Backend:**
```bash
# Unit tests
pytest backend/tests/unit

# Integration tests
pytest backend/tests/integration

# Coverage
pytest --cov=app backend/tests
```

**Frontend:**
```bash
# Component tests
npm test

# E2E tests
npm run test:e2e
```

---

##  Roadmap

### v1.1 (Next Release)
- [ ] Multi-GPU support for large models
- [ ] Batch processing for >1000 samples
- [ ] Export evaluation reports (PDF/CSV)
- [ ] Model hub integration (auto-download from HF)

### v1.2
- [ ] Fine-tuning evaluation mode
- [ ] Cost calculator (cloud vs local)
- [ ] A/B testing mode (production traffic)
- [ ] More metrics (ROUGE, BLEU, Perplexity)

### Future
- [ ] Auto-quantization optimizer
- [ ] Multi-user support with auth
- [ ] Scheduled evaluations
- [ ] Webhook notifications

---

##  License

MIT License - see [LICENSE](LICENSE) file for details.

---

##  Support

- **Issues:** [GitHub Issues](https://github.com/yourusername/quantease/issues)
- **Discussions:** [GitHub Discussions](https://github.com/yourusername/quantease/discussions)
- **Email:** support@quantease.dev
- **Discord:** [Join our community](https://discord.gg/quantease)

---

##  Acknowledgments

- [llama.cpp](https://github.com/ggerganov/llama.cpp) - Fast GGUF inference
- [Groq](https://groq.com) - Lightning-fast LLM API
- [BERTScore](https://github.com/Tiiiger/bert_score) - Semantic similarity metric
- [FastAPI](https://fastapi.tiangolo.com/) - Modern Python web framework

