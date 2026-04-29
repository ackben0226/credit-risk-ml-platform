# =====================================================
# ML PLATFORM MAKEFILE (PRODUCTION-GRADE)
# =====================================================

# ----------------------------
# SETUP
# ----------------------------
install:
	pip install -r requirements.txt

install-dev:
	pip install -r requirements-dev.txt

# ----------------------------
# CODE QUALITY
# ----------------------------
lint:
	flake8 src

format:
	black src

type-check:
	mypy src

# ----------------------------
# TESTING
# ----------------------------
test:
	pytest -q

test-cov:
	pytest --cov=src --cov-report=term-missing

# ----------------------------
# FEATURE PIPELINE
# ----------------------------
features:
	python src/run_pipeline.py

validate-data:
	python src/data/validation.py

# ----------------------------
# TRAINING
# ----------------------------
train:
	python src/training/train_model.py

tune:
	python src/training/tuning.py

retrain:
	python src/pipelines/retrain_pipeline.py

# ----------------------------
# ML LIFECYCLE (REGISTRY)
# ----------------------------
register:
	python -c "from src.training.registry import register_model; print('Registry ready')"

promote:
	python -c "from src.training.registry import promote_if_valid; print('Use automation layer for promotion')"

# ----------------------------
# AUTOMATION ENGINE
# ----------------------------
pipeline:
	python src/automation/orchestrator.py

prefect:
	python src/pipelines/prefect_pipeline.py

# ----------------------------
# API (LOCAL)
# ----------------------------
run:
	uvicorn api.main:app --reload --host 0.0.0.0 --port 8000

# ----------------------------
# MONITORING
# ----------------------------
dashboard:
	streamlit run dashboard.py

mlflow:
	mlflow ui --port 5000

# ----------------------------
# DOCKER
# ----------------------------
build:
	docker-compose build

up:
	docker-compose up -d

down:
	docker-compose down

logs:
	docker-compose logs -f

restart:
	docker-compose down && docker-compose up -d

# ----------------------------
# CLEAN
# ----------------------------
clean:
	rm -rf __pycache__
	rm -rf .pytest_cache
	rm -rf .mypy_cache
	find . -type d -name "__pycache__" -exec rm -r {} + || true

# ----------------------------
# FULL REBUILD (SAFE MODE)
# ----------------------------
reset:
	make clean
	make install
	make test