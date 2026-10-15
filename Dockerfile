# Scoring API for the step 4 model (step 7).
#
# The image holds the code and the model; the feature store (built from the transaction history with
# `python main.py build-store`, ~600 MB) is mounted at run time, so it can be rebuilt without a new image:
#
#   docker build -t aml-scoring .
#   docker run -p 8000:8000 -v "$(pwd)/data/processed:/data" aml-scoring
#
# Then open http://localhost:8000/docs.
FROM python:3.11-slim

# XGBoost needs the OpenMP runtime.
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements-serve.txt .
RUN pip install --no-cache-dir -r requirements-serve.txt

# Code runs from /app/src, so the project root (models/, reports/) is /app.
COPY src ./src
# The served model and its validation threshold. The drift monitor's reference (validation scores,
# graph_scores.parquet) is read from the mounted data folder.
COPY models/xgb_graph.json ./models/
COPY reports/metrics_step4_graph.json ./reports/

ENV PYTHONPATH=/app/src \
    AML_PROCESSED_DIR=/data \
    AML_FEATURE_STORE=/data/feature_store.pkl

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"
CMD ["uvicorn", "aml.api:app", "--host", "0.0.0.0", "--port", "8000"]
