setup:
	python -m venv .venv
	pip install -r requirements.txt

data:
	python src/generate_data.py
	python src/sql_features.py

train:
	python src/train.py

explain:
	python src/explain.py

test:
	pytest -q

api:
	uvicorn src.api:app --host 0.0.0.0 --port 8000

ui:
	streamlit run src/app.py
