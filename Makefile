.PHONY: help format lint test test-fast test-api dev docs install install-dev

help:		## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}'

install:	## Install production dependencies
	pip install -r requirements.txt

install-dev:	## Install development dependencies
	pip install -r requirements.txt -r requirements-dev.txt

format:		## Format code with Black
	black src/ api/ tests/

lint:		## Run linting checks
	flake8 src/ api/ tests/ --max-line-length=100 --ignore=E203,W503
	mypy src/ api/ --ignore-missing-imports
	bandit -r src/ api/

test:		## Run full test suite
	pytest tests/ -v --cov=src --cov=api --cov-report=html

test-fast:	## Run unit tests only
	pytest tests/test_fraud_detector.py -v

test-api:	## Run API tests only
	pytest tests/test_api.py -v

dev:		## Start development server
	uvicorn api.app:app --reload --host 0.0.0.0 --port 8000

docs:		## Generate API documentation
	@echo "API docs available at: http://localhost:8000/docs"

clean:		## Clean temporary files
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	rm -rf .coverage htmlcov/ .pytest_cache/