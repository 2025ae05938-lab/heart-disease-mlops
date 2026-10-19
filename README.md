# Heart Disease Risk Prediction - MLOps Assignment 1

End-to-end ML project: data -> EDA -> models -> MLflow -> tests/CI -> FastAPI -> Docker -> Kubernetes -> monitoring.

## Setup (run all commands from the project root)

```bash
python -m venv venv
# Windows:      venv\Scripts\activate
# Mac / Linux:  source venv/bin/activate
pip install -r requirements.txt
```

## 1. Get the data and run EDA
```bash
python src/download_data.py   # saves data/heart_raw.csv
python src/eda.py             # saves data/heart_clean.csv and plots in screenshots/eda/
```

## 2. Train models (logs to MLflow, saves models/model.joblib)
```bash
python src/train.py
mlflow ui                     # open http://localhost:5000
```

## 3. Tests and lint
```bash
flake8 src tests api
pytest -v
```

## 4. Run the API locally
```bash
uvicorn api.app:app --port 8000
# open http://localhost:8000/docs  (metrics at /metrics)
```
Sample request:
```bash
curl -X POST http://localhost:8000/predict -H "Content-Type: application/json" \
  -d '{"age":63,"sex":1,"cp":1,"trestbps":145,"chol":233,"fbs":1,"restecg":2,"thalach":150,"exang":0,"oldpeak":2.3,"slope":3,"ca":0,"thal":6}'
```

## 5. Docker
```bash
docker build -t heart-api .
docker run -p 8000:8000 heart-api
```

## 6. Kubernetes (Docker Desktop with Kubernetes enabled)
```bash
kubectl apply -f k8s/
kubectl get pods,svc
# service is exposed at http://localhost/ (port 80); try http://localhost/docs
```

## Repository layout
```
src/        data download, cleaning, EDA, training
tests/      pytest unit tests
api/        FastAPI app
k8s/        Kubernetes manifests
.github/    GitHub Actions CI workflow
models/     saved model (created by train.py)
screenshots/ evidence for the report
```
