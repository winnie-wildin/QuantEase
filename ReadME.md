# Quantease — Model Quantization Evaluation Dashboard

Quantease is an end-to-end evaluation platform for comparing baseline LLMs (cloud / managed) with locally-run quantized models (GGUF). It helps you measure trade-offs in speed, latency, size, and quality (BERTScore / other metrics) and optionally uses an LLM-as-judge for qualitative scoring.

This README covers project goals, architecture, installation, local development, testing, and where to change prompts and judge behavior.

---

## Key Features
- Baseline generation (cloud / API, e.g., Groq)
- Quantized local generation (GGUF via llama-cpp-python)
- Automatic evaluation: hardware performance + quality metrics (BERTScore, cosine similarity, perplexity, etc.)
- Optional LLM-as-judge qualitative evaluation (Groq gpt-oss-120b)
- Per-task prompt formatting (text generation, classification, RAG)
- Frontend dashboard for experiment overview and sample outputs

---

## Repo Structure
- backend/
  - app/
    - tasks/ — Celery tasks: `baseline_generation.py`, `quantized_generation.py`, `evaluation_task.py`
    - utils/ — helpers: `prompt_formatter.py`, `gguf_loader.py`, `llm_judge.py`
    - models.py, database.py — SQLAlchemy models and DB setup
    - api/ — REST endpoints (create experiments, trigger generation/evaluation)
- frontend/
  - src/
    - pages/ — `Home.tsx`, `ExperimentDetails.tsx`
    - services/ — API client `api.ts`
    - components/ — UI building blocks
- docker-compose.yml — optional local stack (DB, Redis, backend, frontend)
- README.md (this file)

---

## Prerequisites
- Node.js v18+ and npm/yarn
- Python 3.10+
- PostgreSQL (or other DB supported by SQLAlchemy)
- Redis for Celery as broker
- For local quantized models: models in .gguf format and `llama-cpp-python` installed with a supported backend
- Groq API key if using cloud baseline or judge

---

## Local Setup (Backend)
1. Create a Python venv and install:
   - python -m venv venv
   - venv\Scripts\Activate (Windows)
   - pip install -r backend/requirements.txt

2. Environment variables (.env)
   - DATABASE_URL=postgresql://user:pass@localhost:5432/quantease
   - REDIS_URL=redis://localhost:6379/0
   - GROQ_API_KEY=<your_groq_key> (optional — required for baseline and judge)
   - FASTAPI_HOST=0.0.0.0
   - FASTAPI_PORT=8000
   - OTHER config variables (JWT, secrets) depending on your setup

3. Database migrations
   - Alembic or similar: run migrations to create tables
   - Or run the included DB setup script (see `backend/app/models`)

4. Run backend dev server
   - uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

5. Run Celery worker
   - celery -A app.tasks.celery_app.celery_app worker --loglevel=info
   - (Optional) Add a beat scheduler or Flower for monitoring

---

## Local Setup (Frontend)
1. cd frontend
2. npm install or yarn
3. Add `.env` settings for frontend (API base URL)
4. Run dev server
   - npm run dev

---

## Start an Experiment
- Use the front-end to create an experiment or call the API:
  - POST /api/experiments — create with task_type (text_generation | classification | rag)
  - Add baseline and quantized model variants (baseline is optional)
  - Upload a dataset: CSV/JSON with input prompts and optional ground-truth outputs
  - Trigger generation:
    - POST /api/experiments/{id}/generate_baseline
    - POST /api/experiments/{id}/generate_quantized
  - Trigger evaluation:
    - POST /api/experiments/{id}/trigger_evaluation (pass enable_llm_judge true/false)

---

## Where Prompts and Judging Live
- `backend/app/utils/prompt_formatter.py`
  - Centralized prompt templates by model family and optional per-task templates.
  - To change how prompts are formatted per task (classification, text generation, RAG), add task-specific templates and use `task_type` argument when formatting.

- `backend/app/utils/gguf_loader.py`
  - Local GGUF model loader (llama-cpp-python)
  - Should call `PromptFormatter.format_prompt(prompt, model_path, task_type)` when generating.

- `backend/app/tasks/*_generation.py`
  - Baseline vs quantized generation call out to Groq or GGUF loader respectively:
    - Pass `task_type` into the formatter/generator.
    - For classification, you should include explicit instruction to return labels only.

- `backend/app/utils/llm_judge.py`
  - LLM-as-judge for qualitative assessment. Ensure `GROQ_API_KEY` is set when enabling judge.
  - Returns aggregated judge measures (accuracy, fluency, coherence, or unified `avg_factual_correctness`).

- `backend/app/tasks/evaluation_task.py`
  - Orchestrates metric computation and optionally calls LLMJudge and merges results into `ComparativeMetrics.evaluation_results`.

---

## How to Enable/Use the LLM Judge
- Backend: set GROQ_API_KEY, make sure `LLMJudge` is used by evaluation tasks.
- Frontend: when triggering evaluation, pass `enableLLMJudge: true` or toggle “Enable LLM Judge”.
- The judge runs on a sample (default 10%) and returns aggregated stats. For visibility, ensure the backend includes UI-friendly keys; if not, add an alias mapping (e.g., `avg_factual_correctness`) so the front-end renders results.

---

## Ground-Truth vs Baseline Comparison
- You can upload ground truth to an experiment. If `experiment.has_ground_truth` is set, the UI shows a badge and a Compare toggle ("Baseline" vs "Ground truth").
- The visualization chooses BERTScore vs baseline or vs ground truth based on the toggle; it falls back to the available metric if neither is present.
- If you want to compare only quantized vs ground truth and skip baseline, you can create an experiment without a baseline variant and upload GT. The frontend and metrics logic handles fallback.

---

## Running Tests & Verification
- Backend unit tests:
  - From repo root: pytest backend/tests
- Frontend tests:
  - npm run test from `frontend/`
- Optional verify script:
  - `python backend/verify_implementation.py` — will check migrations, tasks and utilities are wired correctly (script exists, improve as needed).

---

## Debugging Tips
- If LLM judge results don't show in the UI:
  - Confirm `enable_llm_judge` was true during evaluation
  - Ensure `GROQ_API_KEY` is present on backend
  - Check DB `comparative_metrics.evaluation_results` for llm judge content and confirm key names match frontend expectations
  - If keys mismatch (e.g., avg_accuracy vs avg_factual_correctness), add alias mapping in `evaluation_task.py` or `llm_judge.py`

- If prompts aren't appropriate per task:
  - Edit `PromptFormatter` to implement per-task templates.
  - Pass `task_type` to `GGUFLoader.generate()` and baseline generation calls.

- If quantized generation ignores ground truth:
  - Ensure `DatasetSample` has `ground_truth` fields and `generate_*` tasks save the sample's `ground_truth` into the `GeneratedOutput` records for later side-by-side display.

---

## Deployment / Docker
- A `docker-compose.yml` can spin up:
  - Postgres, Redis, Backend API + worker, Frontend (Nginx)
- Add persistent volume for local GGUF model files.
- In production, run Celery worker(s) and a scheduler for periodic refreshes.

---

## Contributing
- Create a branch per feature/fix.
- Add tests for any new behavior, including prompt templates and judge aggregation.
- Run unit tests and linters before submitting PRs.

---

## Roadmap Ideas
- Per-sample judge results in UI for inspection (store per-sample judge JSON).
- More metrics (ROUGE, BLEU, instruction following rate).
- Per-task robust prompts with test fixtures for classification and RAG.
- Support for GPU offloading and other GGUF optimizations.

---

## License & Contact
- MIT license (or update to your license).
- Questions / issues: submit via repo issue tracker or PRs.
