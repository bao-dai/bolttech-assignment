# API

FastAPI app that serves the ML prediction and GenAI explanation endpoints.

## Endpoints

- `POST /predict` - predict claim approval, returns probability + top SHAP factors + damage assessment
- `POST /explain` - generate a persona-specific explanation (customer/adjuster/manager)
- `GET /health` - health check
- `GET /model/info` - model metadata

## How to run

```
conda activate bolt
cd source/4_api_deployment
uvicorn app:app --host 127.0.0.1 --port 8000
```

Then go to http://127.0.0.1:8000/docs for the Swagger UI

## Example requests

```bash
curl -X POST http://localhost:8000/predict -H "Content-Type: application/json" -d '{"claim_idx": 13}'

curl -X POST http://localhost:8000/explain -H "Content-Type: application/json" -d '{"claim_idx": 13, "persona": "customer"}'
```

## Deployment notes

AWS deployment design (ECS Fargate, API Gateway, etc) and MLOps considerations are in `docs/4_api_deployment/`
