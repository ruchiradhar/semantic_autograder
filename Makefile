PYTHON ?= python3
PKG = src

.PHONY: install format lint train precommit

install:
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -r requirements.txt
	$(PYTHON) -m pip install pre-commit
	pre-commit install

format:
	ruff format .
	black .

lint:
	ruff check .

train:
	$(PYTHON) -m src.pipeline.train_pipeline

precommit:
	pre-commit run --all-files
