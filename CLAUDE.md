# StrainlessAI v2

## Goal

Rebuild StrainlessAI from scratch. A singer records one fixed arpeggio for their voice type. The app checks the recording quality, then a small model trained on acoustic features says **Healthy / Borderline / Strained**. A voice-teacher chatbot answers technique questions using retrieved material.

v2 replaces v1's spectrogram CNN (Healthy / Strained / Weak, letter grades) with a binary Healthy-vs-Strained model on acoustic features. The model's probabilities are calibrated and shown as three bands. Retrieval is rebuilt on Qdrant with heading-matched search, the LLM is gpt-oss on Groq, and the front end is React served by a Flask JSON API.

**`docs/SPEC_v2.md` is the source of truth.** It says *what* to build; section numbers are the "Parts" below. If code and spec disagree, fix the spec first, then the code. `[DEFAULT]` = provisional choice, `[TUNE]` = starting value to adjust with real data.

## Folder layout

```
CLAUDE.md
pyproject.toml, uv.lock   uv project (one project at the top level)
docs/      SPEC_v2.md, progress.md, data_protocol.md, model_card.md, rag/, research/
api/       Flask JSON API: app.py, llm.py, coach.py, quality.py, static/audio/ (example arpeggios)
ml/        features and training: arpeggios.py (arpeggio definitions, one home), collect/, rate/, features/ (segment.py, extract.py), train/
rag/       retrieval: sections.json, paraphrases.json, ingest.py, retrieve.py, config.json, eval/
web/       React app: Vite + React + TypeScript + Tailwind + shadcn/ui (built to web/dist, gitignored)
models/    strain_v2.joblib (small, committed)
reports/   generated plots and tables (gitignored)
scripts/   one-off checks (check_llm.py, quality_report.py, compare_v1_v2.py, make_synth_examples.py)
data/      recordings: raw/, labels/, splits.json, features.parquet. NEVER committed.
```

**v1 files at the top level** (`app.py`, `trainer.html`, `static/`, `singing_health_cnn.keras`, `requirements.txt`, `Dockerfile`) are kept for reference and as the v1 baseline. Don't edit them and don't import them. Things carried over from v1 `app.py`:
- the 28-term `VOCAB_DEFINITIONS`, used in `api/coach.py`
- the v1 Qdrant collection name `singing_chatbot`, the RAG eval baseline
- `trainer.html`, ported to `/trainer`
- the Google Analytics tag

## Rules

- **uv for all Python.** Use `uv add <pkg>` / `uv add --dev <pkg>`, `uv run <cmd>`, and `uv run pytest`. No pip, no requirements.txt for v2, no other virtualenvs. Python 3.12.
- **Banned in v2:** TensorFlow (and Keras), librosa, LangChain. Don't add them, even as a transitive dependency you could avoid.
  - Instead: scipy for resampling/DSP, parselmouth/openSMILE/MoSQITo for features, the `groq` SDK for the LLM, `qdrant-client` + `fastembed` for RAG.
  - matplotlib is allowed only in `ml/` and `scripts/` for reports. Never import it in `api/`.
- **Secrets only in `.env`** (gitignored). Never commit them, hard-code them, log them, or paste them into prompts or docs. Read them with python-dotenv. Keys: `GROQ_API_KEY`, `LLM_MODEL`, `QDRANT_CLOUD_URL`, `QDRANT_CLOUD_API_KEY`, `QDRANT_COLLECTION`, `COLLECT_BUCKET`, `PAYMENT_FORM_URL`.
- **`data/` is never committed.** Recordings, consent records, invites, and anything that identifies a singer stay out of git and out of cloud-synced notes. Before every commit, check that `git status` shows nothing under `data/`.
- **Progress log:** after each finished build step, append exactly one line to `docs/progress.md`: `YYYY-MM-DD · step · result`.
- Thresholds have one home each. Quality thresholds live only in `QUALITY_CONFIG` in `api/quality.py`. RAG `THRESHOLD`/`MARGIN` live in `rag/config.json`. Band cutoffs live in `models/strain_v2.joblib`.
- Test singers in `data/splits.json` are off limits. Code refuses to read them unless run with `--final`.
- Voice analysis never depends on the LLM. Coaching for results is static text from `rag/sections.json`.
- Every user-facing screen must work at 375 / 768 / 1280 px (spec Section 1.1).

## Build order

From the build guide (kept outside the repo). Work in this order, and log each finished step in `docs/progress.md` before starting the next.

1. Part 1 setup (done)
2. Step 2a: docs/data_protocol.md
3. Step 2b: example arpeggios (synthesized, used for all of data collection; sung examples for the app come in Step 8a)
4. Step 2c: Part 5 quality checks and Step 6a note segmentation
5. Step 2d: collection site (ml/collect/, separate Cloud Run service strainless-collect)
6. Step 2e: pilot run with 3 invites
7. Step 2f: invites and payment (recording then runs in the background)
8. Step 2g: blind rating tool and agreement script
9. Part 3: Groq gpt-oss swap, then Part 4: RAG v2 (built while recordings come in)
10. Steps 6b-6c: per-note features and the training table
11. Part 7: model, calibration and bands
12. Part 8: React UI
13. Part 9: integration, Cloud Run deploy, v1 vs v2

## Commands

```
uv sync                 install from uv.lock
uv run pytest           run tests
uv add <pkg>            add a dependency (updates pyproject.toml and uv.lock)
```
