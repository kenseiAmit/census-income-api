.PHONY: install train test lint serve

install:
	pip install -r requirements.txt
	pip install --no-build-isolation -e .

train:
	python -m income_classifier.train_model

test:
	pytest -v

lint:
	flake8 . --count --max-complexity=10 --max-line-length=79 --statistics

serve:
	uvicorn main:app --reload --host 127.0.0.1 --port 8000
