.PHONY: init venv install install-ml iaa test lint clean
PY ?= python3

init:                ## создать дерево каталогов + .gitkeep
	bash setup_project.sh

venv:                ## создать виртуальное окружение .venv
	$(PY) -m venv .venv
	@echo "-> активируй: source .venv/bin/activate"

install:             ## ядро + dev (калибровка работает без ML-стека)
	$(PY) -m pip install -e ".[dev]"

install-ml:          ## тяжёлый ML-стек (стадия моделей)
	$(PY) -m pip install -e ".[ml,dev]"

iaa:                 ## прогнать IAA-калибровку (пути по умолчанию)
	$(PY) scripts/run_iaa.py

test:                ## юнит-тесты харнесса
	$(PY) -m pytest tests/ -q

lint:                ## статанализ
	ruff check src tests scripts

clean:
	rm -rf .pytest_cache .ruff_cache **/__pycache__
