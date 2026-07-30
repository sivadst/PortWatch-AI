.PHONY: setup data train evaluate test run api app docker lint all

setup:
	pip install -e ".[dev]"

data:
	python scripts/download_data.py
	python -m src.data.cleaning
	python -m src.features.build_features

train:
	python -m src.models.train

evaluate:
	python -m src.evaluation.metrics

test:
	pytest tests/ -v --cov=src --cov=api --cov=app --cov-report=term-missing

api:
	uvicorn api.main:app --reload --host 0.0.0.0 --port 8000

app:
	streamlit run app/Home.py

docker:
	docker build -t portwatch-ai .
	docker run -d -p 8000:8000 -p 8501:8501 --name portwatch-ai-demo portwatch-ai

lint:
	ruff check src/ api/ app/ tests/
	ruff format src/ api/ app/ tests/
	mypy src/ api/ app/

all: lint test data train evaluate
