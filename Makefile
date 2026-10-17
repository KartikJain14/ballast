.PHONY: setup audit train replay test demo api ui

setup:
	pip install -r requirements.txt

audit:
	python scripts/audit.py

test:
	python -m pytest tests/ -q

train:
	python scripts/train.py

replay:
	python scripts/replay.py

api:
	uvicorn api.main:app --reload --port 8000

ui:
	streamlit run ui/app.py

demo: test audit replay
	streamlit run ui/app.py
