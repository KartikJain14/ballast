.PHONY: setup audit replay precompute test demo api ui

setup:
	pip install -r requirements.txt

audit:
	python scripts/audit.py

test:
	python -m pytest tests/ -q

replay:
	python scripts/replay.py

precompute:
	python scripts/precompute.py

api:
	uvicorn api.main:app --reload --port 8000

ui:
	streamlit run ui/app.py

app:
	uvicorn api.main:app --port 8000

demo: test audit replay precompute
	uvicorn api.main:app --port 8000
