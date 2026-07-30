# PortWatch AI Execution Plan

## Phase 1: Plan & Environment Verification
- [x] Verify Python (3.11+) and Git availability.
- [x] Create this task breakdown.

## Phase 2: Implement Scaffolding & Configs
- [ ] Create repository structure (`src`, `api`, `app`, `tests`, `docs`, etc.).
- [ ] Implement `pyproject.toml`, `requirements.txt`, `.gitignore`, `.env.example`.
- [ ] Create `Makefile`.

## Phase 3: Implement Data Layer
- [ ] Implement `scripts/generate_samples.py` for synthetic deterministic data with realistic distributions.
- [ ] Implement `src/data/ingestion.py` (live scraping/API calls with fallback).
- [ ] Implement `src/data/cleaning.py`.
- [ ] Implement `src/features/build_features.py`.
- [ ] Draft `scripts/init_db.sql` and `scripts/init_duckdb.py`.

## Phase 4: Implement ML Pipeline
- [ ] Implement `src/models/train.py`, `predict.py`, `explainability.py`.
- [ ] Configure local MLflow experiment tracking.

## Phase 5: Implement API Layer
- [ ] Build FastAPI app (`api/main.py`, `config.py`, routes).
- [ ] Implement DuckDB/PostgreSQL data service.

## Phase 6: Implement Streamlit Application
- [ ] Build Streamlit pages (`app/Home.py`, etc.).
- [ ] Add "Demo Mode" warning banners.

## Phase 7: Implement Tests
- [ ] Write unit tests (`tests/unit/`).
- [ ] Write integration tests (`tests/integration/`).

## Phase 8: RUN, TEST, DEBUG, RETEST
- [ ] Execute `make setup`, `make data`, `make train`.
- [ ] Run `pytest`, `ruff`, `mypy`.
- [ ] Fix errors, ensure tests pass.

## Phase 9: DOCUMENT
- [ ] Write `README.md`.
- [ ] Write `docs/` (`architecture.md`, `model_card.md`, etc.).
- [ ] Create `CONTRIBUTING.md`, `SECURITY.md`.

## Phase 10: DEPLOY_VERIFY
- [ ] Create `Dockerfile`, `docker-compose.yml`, `render.yaml`.
- [ ] Build Docker image and test health endpoint headlessly.

## Phase 11: FINAL_AUDIT
- [ ] Audit requirements, generate `JULES_REPORT.md`.

## Phase 12: Submit
- [ ] Complete pre-commit checks and submit.
