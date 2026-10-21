PY ?= python
HF_REPO_ID ?= $(FRAUD_HF_REPO_ID)

.PHONY: install etl train monitor pipeline serve replay publish test clean

install:            ## install runtime + dev dependencies
	$(PY) -m pip install -r requirements.txt -r requirements-dev.txt

etl:                ## extract raw CSVs -> engineer features -> processed Parquet
	$(PY) -m fraud_detection.etl

train:              ## tune, compare and save LR / RF / XGBoost; writes reports/
	$(PY) -m fraud_detection.train

monitor:            ## month-by-month performance + drift report on the test period
	$(PY) -m fraud_detection.monitor --log

pipeline: etl train monitor   ## full automated run: ETL -> train -> evaluate -> monitor

serve:              ## start the FastAPI service on :8000 (docs at /docs)
	uvicorn fraud_detection.api.app:app --host 0.0.0.0 --port 8000

replay:             ## stream test transactions to the running API, e.g. make replay ARGS="--limit 5000"
	$(PY) -m fraud_detection.replay $(ARGS)

publish:            ## upload the trained model to the Hugging Face Hub (needs HF_REPO_ID)
	$(PY) -m fraud_detection.publish --repo-id $(HF_REPO_ID)

test:
	$(PY) -m pytest -q

clean:
	rm -rf artifacts .pytest_cache
